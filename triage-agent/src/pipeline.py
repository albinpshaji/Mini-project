"""
Main pipeline entry point -- Phase 1 (sequential baseline).

Flow: mic -> ASR (final transcript only, partials logged but unused here) ->
      SLM (full generation) -> TTS (spoken response).

This is deliberately the "slow but correct" version. Its measured latency
becomes the baseline that the Phase 4 speculative pipeline is benchmarked
against in eval/benchmark_latency.py.

Run with: python -m src.pipeline
"""

import time
import yaml
import logging
import numpy as np

from src.asr.streaming_asr import StreamingASR
from src.slm.slm_engine import SLMEngine
from src.tts.piper_engine import TTSEngine
from src.safety.handoff import evaluate_handoff, contains_fabricated_placeholder, FABRICATION_FALLBACK_MESSAGE
from src.safety.slot_extraction import extract_symptom_slots
from src.scheduler.async_scheduler import ResourceScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
log = logging.getLogger("pipeline")


class SequentialTriagePipeline:
    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.slm = SLMEngine(self.cfg["slm"])
        self.tts = TTSEngine(self.cfg["tts"])
        self.conversation_history = []
        self.turn_count = 0
        self.latest_slots = None

        self.scheduler = ResourceScheduler(self.cfg.get("pipeline", {}))

        self.engagement = None
        cv_cfg = self.cfg.get("cv", {})
        if cv_cfg.get("enabled", False):
            from src.cv.engagement_detector import EngagementDetector
            self.engagement = EngagementDetector(
                camera_index=cv_cfg.get("camera_index", 0),
                check_interval_s=cv_cfg.get("check_interval_s", 1.0),
                absence_grace_s=cv_cfg.get("absence_grace_s", 5.0),
                cascade_path=cv_cfg.get("cascade_path"),
            )

        self.audio_cfg = self.cfg.get("audio", {})
        self.full_duplex_enabled = self.audio_cfg.get("full_duplex", True)
        self.audio_stream = None

        if self.full_duplex_enabled:
            from src.audio.full_duplex_stream import FullDuplexAudioStream
            self.audio_stream = FullDuplexAudioStream(
                sample_rate=16000,
                enable_aec=self.audio_cfg.get("enable_aec", True),
                on_barge_in=self._on_barge_in,
                on_clean_mic_frame=self._on_clean_mic_frame,
            )
            self.tts.on_audio_chunk = self.audio_stream.play_audio_chunk

        self.asr = StreamingASR(self.cfg["asr"], on_event=self._on_asr_event)

        self.interrupt_listener = None
        if not self.full_duplex_enabled and self.cfg.get("pipeline", {}).get("keyboard_interrupt_enabled", True):
            from src.tts.keyboard_interrupt import KeyboardInterruptListener
            try:
                self.interrupt_listener = KeyboardInterruptListener(self.tts, self.asr)
            except RuntimeError as e:
                log.warning("Keyboard interrupt listener unavailable: %s", e)

    def _on_barge_in(self):
        log.info("⚡ Voice barge-in confirmed! Halting agent audio output immediately.")
        self.tts.cancel_immediate()
        if self.audio_stream is not None:
            self.audio_stream.set_tts_active(False)
        self.asr.unmute()

    def _on_clean_mic_frame(self, frame: np.ndarray):
        if self.asr is not None:
            self.asr.feed_audio(frame)

    def _on_asr_event(self, event: dict):
        if event["type"] == "partial":
            text = event.get("text", "").strip()
            if text:
                print(f"\r🎙️ Hearing: \"{text}\"...", end="", flush=True)
            if self.audio_stream is not None:
                self.audio_stream.process_partial_transcript(text)
            return

        # event["type"] == "final" -> run the sequential turn
        print()
        t0 = time.time()
        transcript = event["text"]
        if not transcript:
            return

        if self.engagement is not None and not self.engagement.is_present():
            log.info("No patient detected by camera -- ignoring transcript: %r", transcript)
            return

        vram_status = self.scheduler.check_budget()
        if vram_status["used_mb"] is not None:
            log.info(
                "VRAM: %d/%d MB (budget: %d MB)%s",
                vram_status["used_mb"], vram_status["total_mb"], vram_status["budget_mb"],
                " -- OVER BUDGET" if vram_status["over_budget"] else "",
            )

        # Extract paralinguistic acoustic biomarkers (Google HeAR / Innovation 9)
        biomarkers = None
        if self.audio_stream is not None:
            biomarkers = self.audio_stream.extract_recent_biomarkers()
            if biomarkers.acuity_boost > 0:
                log.warning(
                    "🚨 Paralinguistic Distress Alert: %s (KTAS Acuity Boost: +%d)",
                    biomarkers.clinical_rationale,
                    biomarkers.acuity_boost,
                )

        log.info(f"Patient said: {transcript}")
        self.conversation_history.append({"role": "user", "text": transcript})
        self.turn_count += 1

        handoff_decision = evaluate_handoff(
            transcript,
            self.turn_count,
            acoustic_biomarkers=biomarkers,
            hospital_router=getattr(self.slm, "hospital_router", None),
        )

        t_start = time.time()
        ttfa_ms = None

        def _on_first_audio():
            nonlocal ttfa_ms
            ttfa_ms = (time.time() - t_start) * 1000
            log.info("🔊 Speech started! (Time to First Audio: %.0fms)", ttfa_ms)

        self.asr.mute()
        if self.audio_stream is not None:
            self.audio_stream.set_tts_active(True)

        if handoff_decision.escalate:
            response = handoff_decision.override_message
            log.warning(
                "HANDOFF TRIGGERED | reason=%s | red_flag_category=%s",
                handoff_decision.reason,
                handoff_decision.red_flag_category,
            )
            _on_first_audio()
            self.tts.speak(response, blocking=True)
        else:
            token_stream = self.slm.generate_stream(transcript, self.conversation_history)
            response = self.tts.speak_stream(token_stream, on_first_audio_callback=_on_first_audio)
            if contains_fabricated_placeholder(response):
                log.warning("FABRICATION DETECTED in SLM response -- replacing with safe fallback. Original: %r", response)
                response = FABRICATION_FALLBACK_MESSAGE

        if self.audio_stream is not None:
            self.audio_stream.set_tts_active(False)
        time.sleep(0.2)
        self.asr.unmute()

        t_total_done = time.time()

        self.conversation_history.append({"role": "assistant", "text": response})
        log.info(f"Agent response: {response}")

        total_wait = ttfa_ms if ttfa_ms is not None else ((t_total_done - t_start) * 1000)
        playback_duration = max(0.0, ((t_total_done - t_start) * 1000) - total_wait)
        log.info(
            "Latency | Wait until speech started (TTFA): %.0fms | Audio speech playback: %.0fms",
            total_wait,
            playback_duration,
        )

        def _extract_async(history, model):
            t_extract_start = time.time()
            slots = extract_symptom_slots(history, model_name=model)
            t_extract_end = time.time()
            self.latest_slots = slots
            log.info(
                "Structured slots (%.0fms): %s",
                (t_extract_end - t_extract_start) * 1000,
                slots.to_dict(),
            )

        import threading
        threading.Thread(
            target=_extract_async,
            args=(list(self.conversation_history), self.cfg["slm"].get("model_name", "qwen3.5:4b")),
            daemon=True,
        ).start()

    def start(self):
        log.info("Starting sequential triage pipeline (non-blocking). Speak into the mic.")
        if self.engagement is not None:
            self.engagement.start()
        if self.interrupt_listener is not None:
            self.interrupt_listener.start()
        if self.audio_stream is not None:
            self.audio_stream.start()
            self.asr.start(use_internal_mic=False)
        else:
            self.asr.start(use_internal_mic=True)

    def stop(self):
        log.info("Stopping.")
        if self.audio_stream is not None:
            self.audio_stream.stop()
        self.asr.stop()
        if self.engagement is not None:
            self.engagement.stop()
        if self.interrupt_listener is not None:
            self.interrupt_listener.stop()

    def run(self):
        self.start()
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()


if __name__ == "__main__":
    pipeline = SequentialTriagePipeline()
    pipeline.run()
