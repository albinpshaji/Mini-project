"""
Speculative pipeline -- Phase 4. Subclasses SequentialTriagePipeline.
"""

import time
import logging

from src.pipeline import SequentialTriagePipeline, log
from src.speculative.speculator import Speculator
from src.speculative.rollback_manager import resolve_speculation
from src.safety.handoff import evaluate_handoff, contains_fabricated_placeholder, FABRICATION_FALLBACK_MESSAGE
from src.safety.slot_extraction import extract_symptom_slots

MIN_WORDS_TO_SPECULATE = 3


class SpeculativeTriagePipeline(SequentialTriagePipeline):
    def __init__(self, config_path="config.yaml"):
        super().__init__(config_path=config_path)

        self.speculator = Speculator(self.cfg["slm"])
        self.divergence_threshold = self.cfg.get("speculative", {}).get("divergence_threshold", 0.7)
        self._last_speculated_partial = None

        self.asr.on_event = self._on_asr_event

    def _on_asr_event(self, event: dict):
        if event["type"] == "partial":
            self._handle_partial(event)
            return
        self._handle_final(event)

    def _handle_partial(self, event: dict):
        text = event.get("text", "").strip()
        if text:
            print(f"\r🎙️ Hearing: \"{text}\"...", end="", flush=True)

        if self.audio_stream is not None:
            self.audio_stream.process_partial_transcript(text)

        if len(text.split()) < MIN_WORDS_TO_SPECULATE:
            return
        if self._last_speculated_partial is not None:
            return

        self._last_speculated_partial = text
        self.speculator.start_speculation(text, self.conversation_history)

    def _handle_final(self, event: dict):
        print()
        t0 = time.time()
        transcript = event["text"]
        if not transcript:
            return

        if self.engagement is not None and not self.engagement.is_present():
            log.info("No patient detected by camera -- ignoring transcript: %r", transcript)
            self.speculator.cancel()
            self._last_speculated_partial = None
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

        handoff_decision = evaluate_handoff(transcript, self.turn_count, acoustic_biomarkers=biomarkers)

        t_slm_start = time.time()
        speculative_used = False

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
            self.speculator.cancel()
            response = handoff_decision.override_message
            log.warning(
                "HANDOFF TRIGGERED | reason=%s | red_flag_category=%s",
                handoff_decision.reason,
                handoff_decision.red_flag_category,
            )
            _on_first_audio()
            self.tts.speak(response, blocking=True)
        else:
            rollback_decision = resolve_speculation(
                self.speculator, self.tts, transcript,
                divergence_threshold=self.divergence_threshold,
            )
            if rollback_decision.committed:
                response = rollback_decision.speculative_response
                speculative_used = True
                log.info("⚡ Using SPECULATIVE response (Instant Playback).")
                _on_first_audio()
                self.tts.speak(response, blocking=True)
            else:
                log.info("Generated FRESH response with streaming audio playback.")
                token_stream = self.slm.generate_stream(transcript, self.conversation_history)
                response = self.tts.speak_stream(token_stream, on_first_audio_callback=_on_first_audio)

            if contains_fabricated_placeholder(response):
                log.warning("FABRICATION DETECTED -- replacing with safe fallback. Original: %r", response)
                response = FABRICATION_FALLBACK_MESSAGE

        if self.audio_stream is not None:
            self.audio_stream.set_tts_active(False)
        time.sleep(0.2)
        self.asr.unmute()

        t_total_done = time.time()
        self._last_speculated_partial = None

        self.conversation_history.append({"role": "assistant", "text": response})
        log.info(f"Agent response: {response}")

        total_wait = ttfa_ms if ttfa_ms is not None else ((t_total_done - t_start) * 1000)
        playback_duration = max(0.0, ((t_total_done - t_start) * 1000) - total_wait)
        log.info(
            "Latency | speculative_used=%s | Wait until speech started (TTFA): %.0fms | Audio speech playback: %.0fms",
            speculative_used,
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


if __name__ == "__main__":
    pipeline = SpeculativeTriagePipeline()
    pipeline.run()
