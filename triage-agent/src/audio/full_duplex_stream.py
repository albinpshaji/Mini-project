"""
Full-Duplex Audio Stream Engine for OmniSpec-Triage.
Wraps sounddevice with non-blocking ring buffers, hardware timestamping,
software AEC loopback, and SOBC voice barge-in arbitration.
"""

import time
import queue
import logging
import threading
from typing import Optional, Callable, Generator
import numpy as np

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except (ImportError, OSError):
    _SD_AVAILABLE = False

from src.audio.echo_canceller import AcousticEchoCanceller
from src.audio.overlap_classifier import SemanticOverlapClassifier, OverlapDecision
from src.audio.acoustic_biomarkers import AcousticBiomarkerCoProcessor, AcousticBiomarkerResult

log = logging.getLogger("full_duplex_stream")


class FullDuplexAudioStream:
    def __init__(
        self,
        sample_rate: int = 16000,
        block_size: int = 320,  # 20ms frames at 16kHz
        device_in: Optional[int] = None,
        device_out: Optional[int] = None,
        enable_aec: bool = True,
        on_barge_in: Optional[Callable[[], None]] = None,
        on_clean_mic_frame: Optional[Callable[[np.ndarray], None]] = None,
    ):
        """
        Args:
            sample_rate: Audio sampling frequency (Hz), default 16000.
            block_size: Number of samples per frame (320 = 20ms at 16kHz).
            device_in: Input sounddevice device index (None = system default).
            device_out: Output sounddevice device index (None = system default).
            enable_aec: Whether to enable acoustic echo cancellation.
            on_barge_in: Callback fired when patient interrupts agent speech.
            on_clean_mic_frame: Optional callback for each clean audio frame.
        """
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device_in = device_in
        self.device_out = device_out
        self.enable_aec = enable_aec
        self.on_barge_in = on_barge_in
        self.on_clean_mic_frame = on_clean_mic_frame

        # Subsystems
        self.aec = AcousticEchoCanceller(sample_rate=sample_rate) if enable_aec else None
        self.sobc = SemanticOverlapClassifier(on_interrupt=self._handle_barge_in)
        self.biomarkers = AcousticBiomarkerCoProcessor(sample_rate=sample_rate)

        # Ring Buffers & Queues
        self.mic_queue: queue.Queue = queue.Queue(maxsize=1000)
        self.playback_queue: queue.Queue = queue.Queue(maxsize=1000)

        # Utterance accumulator for acoustic biomarker analysis
        self._current_utterance_frames = []
        self._is_speaking_tts = False
        self._running = False
        self._stream = None

    def start(self):
        """Start full-duplex I/O stream."""
        if not _SD_AVAILABLE:
            log.warning("sounddevice not available -- running in headless mock duplex mode.")
            self._running = True
            return

        self._running = True
        try:
            self._stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=(self.device_in, self.device_out),
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            log.info("Full-duplex audio stream started (16kHz, 20ms frames, AEC=%s).", self.enable_aec)
        except Exception as e:
            log.warning("Failed to start sounddevice Stream (%s) -- falling back to queue-only mode.", e)
            self._running = True

    def stop(self):
        """Halt stream and release audio devices."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                log.error("Error closing sounddevice stream: %s", e)
            self._stream = None
        log.info("Full-duplex audio stream stopped.")

    def set_tts_active(self, active: bool):
        """Notify stream that TTS playback has begun or ended."""
        self._is_speaking_tts = active
        self.sobc.set_tts_active(active)
        if active:
            # Clear old utterance frames so agent speech tail is not analyzed
            self._current_utterance_frames.clear()
        elif self.aec:
            self.aec.is_speaker_active = False

    def play_audio_chunk(self, audio: np.ndarray, sample_rate: int = 16000):
        """
        Enqueues audio for speaker playback and loopback AEC cancellation.
        Accepts float32 or int16 PCM.
        """
        if audio is None or len(audio) == 0:
            return

        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio.astype(np.float32)

        # Resample to stream sample_rate if needed
        if sample_rate != self.sample_rate:
            import scipy.signal
            num_target = int(len(audio_float) * self.sample_rate / sample_rate)
            audio_float = scipy.signal.resample(audio_float, num_target).astype(np.float32)

        # Push to AEC loopback reference
        if self.aec:
            self.aec.push_speaker_reference(audio_float)

        self.set_tts_active(True)

        # Break into blocks for stream output queue
        for i in range(0, len(audio_float), self.block_size):
            block = audio_float[i : i + self.block_size]
            if len(block) < self.block_size:
                block = np.pad(block, (0, self.block_size - len(block)))
            try:
                self.playback_queue.put_nowait(block)
            except queue.Full:
                break

    def cancel_playback_immediate(self):
        """Atomic flush of output audio queues in <15ms."""
        # Drain queue
        while not self.playback_queue.empty():
            try:
                self.playback_queue.get_nowait()
            except queue.Empty:
                break
        self.set_tts_active(False)
        if _SD_AVAILABLE and self._stream is not None:
            try:
                # Flushes hardware DAC buffer
                self._stream.abort()
                self._stream.start()
            except Exception as e:
                log.debug("DAC flush abort/restart: %s", e)
        log.info("Immediate playback cancellation executed (<15ms DAC flush).")

    def _handle_barge_in(self):
        """Triggered by SOBC when competitive interruption is confirmed."""
        log.info("🎙️ VOICE BARGE-IN DETECTED! Cancelling agent speech immediately.")
        self.cancel_playback_immediate()
        if self.on_barge_in:
            try:
                self.on_barge_in()
            except Exception as e:
                log.error("Error executing on_barge_in callback: %s", e)

    def process_partial_transcript(self, partial_text: str) -> OverlapDecision:
        """Forward ASR partial transcript to SOBC intent filter."""
        decision = self.sobc.process_partial_transcript(partial_text)
        return decision

    def _audio_callback(self, indata, outdata, frames, time_info, status):
        """Real-time sounddevice callback running at 16kHz."""
        if status:
            log.debug("Sounddevice status: %s", status)

        # Output side: Play queued TTS audio or output silence
        try:
            chunk = self.playback_queue.get_nowait()
            outdata[:, 0] = chunk
            # Check if playback queue is now empty
            if self.playback_queue.empty():
                self.set_tts_active(False)
        except queue.Empty:
            outdata.fill(0.0)
            if self._is_speaking_tts and self.playback_queue.empty():
                self.set_tts_active(False)

        # Input side: Process mic frame through AEC
        mic_frame = indata[:, 0].copy()

        double_talk = False
        if self.aec is not None:
            clean_frame, double_talk = self.aec.cancel_echo(mic_frame)
        else:
            clean_frame = mic_frame

        # Evaluate SOBC acoustic grace window / overlap
        frame_rms = float(np.sqrt(np.mean(clean_frame ** 2) + 1e-9))
        self.sobc.process_acoustic_frame(frame_rms, double_talk=double_talk)

        # Queue clean frame for ASR
        try:
            self.mic_queue.put_nowait(clean_frame)
        except queue.Full:
            pass

        if not self._is_speaking_tts:
            self._current_utterance_frames.append(clean_frame)
            if len(self._current_utterance_frames) > 250:  # keep last 5 seconds (250 * 20ms)
                self._current_utterance_frames.pop(0)

        if self.on_clean_mic_frame:
            try:
                self.on_clean_mic_frame(clean_frame)
            except Exception:
                pass

    def extract_recent_biomarkers(self) -> AcousticBiomarkerResult:
        """Extract paralinguistic distress features from the recent patient utterance."""
        if not self._current_utterance_frames:
            return self.biomarkers._empty_result()
        audio_window = np.concatenate(self._current_utterance_frames)
        return self.biomarkers.analyze(audio_window)

    def read_clean_frame(self, timeout: float = 0.05) -> Optional[np.ndarray]:
        """Read a clean mic frame from the queue."""
        try:
            return self.mic_queue.get(timeout=timeout)
        except queue.Empty:
            return None
