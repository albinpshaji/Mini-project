"""
Streaming ASR module.

Wraps faster-whisper to provide:
  - continuous microphone capture
  - periodic "partial hypothesis" emission (re-transcribing the growing buffer)
  - VAD-based end-of-utterance detection -> "final hypothesis"

This is intentionally simple for Phase 1 (sequential baseline). Partial hypotheses
aren't used for speculation yet -- they're just logged/available for Phase 4 later.

Event contract (used by later speculative code too, so keep this stable):
    {
        "type": "partial" | "final",
        "text": str,
        "timestamp_ms": int,
        "is_stable": bool          # heuristic: has this text stopped changing recently?
    }
"""

import time
import queue
import threading
import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from scipy.signal import butter, sosfilt


class StreamingASR:
    def __init__(self, config: dict, on_event=None):
        self.cfg = config
        self.on_event = on_event or (lambda e: None)

        self.sample_rate = 16000
        self.frame_ms = 30
        self.frame_samples = int(self.sample_rate * self.frame_ms / 1000)

        # WebRTC VAD mode 2 (balanced sensitivity: allows wider hearing range without missing soft speech)
        self.vad = webrtcvad.Vad(2)
        self.silence_ms_threshold = config.get("vad_silence_ms", 500)
        self.partial_interval_ms = config.get("partial_emit_interval_ms", 250)

        # High-pass filter at 80Hz to remove sub-audible rumbles while preserving male & female speech fundamentals (85-255Hz)
        self._hpf_sos = butter(4, 80, btype='highpass', fs=self.sample_rate, output='sos')
        self._hpf_zi = np.zeros((self._hpf_sos.shape[0], 2))

        self.model = WhisperModel(
            config.get("model_size", "small"),
            device=config.get("device", "cuda"),
            compute_type=config.get("compute_type", "int8_float16"),
        )

        final_model_size = config.get("final_model_size")
        if final_model_size and final_model_size != config.get("model_size", "small"):
            self.final_model = WhisperModel(
                final_model_size,
                device=config.get("final_model_device", config.get("device", "cpu")),
                compute_type=config.get("final_model_compute_type", config.get("compute_type", "int8")),
            )
        else:
            self.final_model = self.model

        self.input_device = config.get("input_device", None)

        from src.asr.semantic_vad import SemanticVAD
        self.semantic_vad = SemanticVAD(
            closed_silence_ms=config.get("semantic_vad_closed_ms", 150),
            open_silence_ms=self.silence_ms_threshold
        )

        from src.asr.adaptive_endpointing import DyspneaAwareEndpointing
        self.adaptive_endpointing = DyspneaAwareEndpointing(
            tau_base_ms=config.get("vad_base_ms", 450),
            alpha_slot_ms=config.get("vad_slot_alpha_ms", 270),
            beta_dyspnea_ms=config.get("vad_dyspnea_beta_ms", 450),
            sample_rate=self.sample_rate
        )
        self.use_slot_entropy = config.get("use_slot_entropy", True)

        self._audio_q = queue.Queue()
        self._running = False
        self._stream = None
        self._last_partial_text = ""
        self._stable_count = 0
        self._muted = False

    def mute(self):
        """Call this right before TTS playback starts to prevent self-hearing."""
        self._muted = True
        # Drain any pending frames from the queue immediately
        while not self._audio_q.empty():
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                break

    def unmute(self):
        """Call this after TTS playback finishes and acoustic room tail settles."""
        # Drain any residual reverb/echo frames before accepting new speech
        while not self._audio_q.empty():
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                break
        self._last_partial_text = ""
        self._stable_count = 0
        self._muted = False

    def feed_audio(self, frame: np.ndarray):
        """
        Feed clean audio frame from external source (e.g. FullDuplexAudioStream with AEC).
        Accepts float32 [-1, 1] or int16 PCM.
        Applies a 300Hz high-pass filter to remove low-frequency noise.
        """
        if self._muted:
            return
        if frame.dtype != np.int16:
            # Apply high-pass filter to remove low-frequency noise (fan/HVAC)
            filtered, self._hpf_zi = sosfilt(self._hpf_sos, frame, zi=self._hpf_zi)
            frame_int16 = np.clip(filtered * 32768.0, -32768.0, 32767.0).astype(np.int16)
        else:
            frame_int16 = frame
        self._audio_q.put(frame_int16)

    def _audio_callback(self, indata, frames, time_info, status):
        if self._muted:
            return
        self._audio_q.put(indata.copy())

    def start(self, use_internal_mic: bool = True):
        self._running = True
        if use_internal_mic:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.frame_samples,
                device=self.input_device,
                callback=self._audio_callback,
            )
            self._stream.start()
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()

    def stop(self):
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _run_loop(self):
        try:
            self._run_loop_inner()
        except Exception:
            import traceback
            print("StreamingASR worker thread crashed:")
            traceback.print_exc()

    def _run_loop_inner(self):
        buffer = []
        preroll = []
        in_speech = False
        silence_ms = 0
        last_partial_emit = 0
        consecutive_speech_frames = 0
        # Require 5 consecutive VAD-true frames (100ms at 20ms/frame) before entering speech
        speech_onset_frames = 5

        while self._running:
            try:
                frame = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._muted:
                buffer = []
                preroll = []
                in_speech = False
                silence_ms = 0
                consecutive_speech_frames = 0
                continue

            frame_flat = frame.flatten()
            frame_len = len(frame_flat)
            if frame_len == 0:
                continue

            frame_float = frame_flat.astype(np.float32) / 32768.0
            frame_rms = float(np.sqrt(np.mean(frame_float ** 2) + 1e-9))

            try:
                is_speech = self.vad.is_speech(frame_flat.tobytes(), self.sample_rate)
            except Exception:
                is_speech = False
            frame_ms = int(frame_len * 1000 / self.sample_rate)

            if not in_speech:
                preroll.append(frame_flat)
                # Keep ~250ms of pre-roll to preserve word onsets
                if len(preroll) > int(250 / max(1, frame_ms)):
                    preroll.pop(0)

                # Debounce: require consecutive speech frames before entering speech mode
                if is_speech:
                    consecutive_speech_frames += 1
                else:
                    consecutive_speech_frames = 0

                if consecutive_speech_frames >= speech_onset_frames:
                    in_speech = True
                    buffer = list(preroll)
                    silence_ms = 0
                    consecutive_speech_frames = 0
            else:
                buffer.append(frame_flat)
                if is_speech:
                    silence_ms = 0
                else:
                    silence_ms += frame_ms

                now_ms = int(time.time() * 1000)
                full_buf = np.concatenate(buffer) if buffer else np.zeros((0,), dtype=np.int16)

                # Periodic partial transcription
                if full_buf.size > 0 and (now_ms - last_partial_emit) >= self.partial_interval_ms:
                    self._emit_partial(full_buf)
                    last_partial_emit = now_ms

                # Dynamic silence threshold via Dyspnea-Aware Slot-Entropy Endpointing (Phase 6)
                min_utterance_samples = int(self.sample_rate * 0.4)
                if self.use_slot_entropy:
                    dynamic_silence_threshold = self.adaptive_endpointing.evaluate(
                        self._last_partial_text, pause_audio_chunk=frame_flat
                    ).silence_timeout_ms
                else:
                    dynamic_silence_threshold = self.semantic_vad.get_silence_timeout_ms(self._last_partial_text)

                # Finalize turn if silence threshold reached or max duration exceeded (12s)
                max_samples = self.sample_rate * 12
                if (silence_ms >= dynamic_silence_threshold and full_buf.size >= min_utterance_samples) or full_buf.size >= max_samples:
                    self._emit_final(full_buf)
                    buffer = []
                    preroll = []
                    in_speech = False
                    silence_ms = 0
                    self._last_partial_text = ""
                    self._stable_count = 0

    def _transcribe(self, audio_int16: np.ndarray) -> str:
        audio_float = audio_int16.astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(
            audio_float,
            language=self.cfg.get("language", "en"),
            vad_filter=True,
            beam_size=1,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
        )
        return " ".join(seg.text for seg in segments).strip()

    def _emit_partial(self, buffer: np.ndarray):
        text = self._transcribe(buffer)
        if not text:
            return

        is_stable = (text == self._last_partial_text)
        self._stable_count = self._stable_count + 1 if is_stable else 0
        self._last_partial_text = text

        self.on_event({
            "type": "partial",
            "text": text,
            "timestamp_ms": int(time.time() * 1000),
            "is_stable": self._stable_count >= 2,
        })

    def _emit_final(self, buffer: np.ndarray):
        audio_float = buffer.astype(np.float32) / 32768.0
        segments, _ = self.final_model.transcribe(
            audio_float,
            language=self.cfg.get("language", "en"),
            vad_filter=True,
            beam_size=2,
            condition_on_previous_text=False,
            initial_prompt="Hospital triage intake: patient describing symptoms, fever, body temperature, duration, pain.",
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
        )
        text = " ".join(seg.text for seg in segments).strip()

        # Fallback to last partial hypothesis if final transcribe was filtered
        if not text and self._last_partial_text:
            text = self._last_partial_text

        if not text:
            return

        self.on_event({
            "type": "final",
            "text": text,
            "timestamp_ms": int(time.time() * 1000),
            "is_stable": True,
        })
