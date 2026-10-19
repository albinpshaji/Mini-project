"""
Speculator -- triggers SLM generation on stable partial ASR hypotheses,
buffers the response on a background thread, and resolves against the
final transcript once available.
"""

import time
import threading
import logging

from src.slm.slm_engine import SLMEngine

log = logging.getLogger("speculator")


class Speculator:
    def __init__(self, slm_config: dict):
        self.engine = SLMEngine(slm_config)

        self._lock = threading.Lock()
        self._thread = None
        self._buffer = []
        self._triggering_partial = None
        self._generation_done = threading.Event()
        self._cancelled = False
        self._start_ts = None

        # Phase 6: Adaptive Context-Shift Self-Speculative Verification (CSSV)
        self.use_cssv = slm_config.get("use_cssv", True)
        from src.speculative.cssv_engine import AdaptiveCSSVEngine
        self.cssv_engine = AdaptiveCSSVEngine(
            model_path=slm_config.get("model_path", None),
            system_prompt="",
            ollama_model_name=slm_config.get("model_name", "qwen3.5:4b"),
        )

    def is_active(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def has_result(self) -> bool:
        return self._generation_done.is_set() and not self._cancelled

    @property
    def system_prompt_tokens(self) -> int:
        """Expose pinned system prompt token count from underlying SLMEngine."""
        return self.engine.system_prompt_tokens

    def get_lookahead_telemetry(self) -> dict:
        """Expose Phase 4 Lookahead Jacobi Speculation telemetry."""
        return self.engine.get_lookahead_telemetry()

    def get_snapkv_telemetry(self) -> dict:
        """Expose Phase 5 SnapKV Attention Anchor Pruning telemetry."""
        return self.engine.get_snapkv_telemetry()

    def get_cssv_telemetry(self) -> dict:
        """Expose Phase 6 Adaptive CSSV telemetry."""
        return self.cssv_engine.get_telemetry()

    def start_speculation(self, partial_text: str, conversation_history: list):
        self.cancel()

        with self._lock:
            self._buffer = []
            self._triggering_partial = partial_text
            self._generation_done.clear()
            self._cancelled = False
            self._start_ts = time.time()

        self._thread = threading.Thread(
            target=self._run_generation,
            args=(partial_text, conversation_history),
            daemon=True,
        )
        self._thread.start()
        log.info("Speculation started on partial: %r", partial_text)

    def _run_generation(self, partial_text: str, conversation_history: list):
        try:
            for chunk in self.engine.generate_stream(partial_text, conversation_history):
                if chunk.get("cancelled"):
                    return
                with self._lock:
                    self._buffer.append(chunk["token"])
                if chunk["done"]:
                    break
        except Exception:
            import traceback
            log.error("Speculative generation thread crashed:")
            traceback.print_exc()
        finally:
            self._generation_done.set()

    def cancel(self):
        self.engine.cancel()
        with self._lock:
            self._cancelled = True

    def resolve(self, final_text: str, wait_timeout_s: float = 3.0):
        if self._thread is None:
            return None

        finished_in_time = self._generation_done.wait(timeout=wait_timeout_s)
        if not finished_in_time:
            log.warning(
                "Speculative generation did not finish within %.1fs of final transcript.",
                wait_timeout_s,
            )
            self.cancel()
            return None

        with self._lock:
            if self._cancelled:
                return None
            return "".join(self._buffer).strip()

    def get_triggering_partial(self):
        with self._lock:
            return self._triggering_partial

    def resolve_cssv(self, final_transcript: str, on_prefix_ready=None):
        """
        Executes Adaptive CSSV verification and prefix salvage on divergent turns.
        """
        if self._thread is None:
            return None

        # Give background thread a short grace window to accumulate draft tokens
        self._generation_done.wait(timeout=0.6)

        with self._lock:
            triggering_partial = self._triggering_partial
            draft_str = "".join(self._buffer).strip()

        if not draft_str or not triggering_partial:
            return None

        full_response, telemetry = self.cssv_engine.verify_and_splice(
            partial_text=triggering_partial,
            final_text=final_transcript,
            draft_response=draft_str,
            on_prefix_ready=on_prefix_ready,
            slm_engine_fallback=self.engine,
        )

        accepted_len = telemetry.get("accepted_prefix_len", 0)
        if accepted_len > 0 and not telemetry.get("negation_flip_detected", False):
            prefix_words = draft_str.split()[:accepted_len]
            return {
                "salvaged_prefix": True,
                "prefix_text": " ".join(prefix_words),
                "full_response": full_response,
                "telemetry": telemetry,
            }
        return None
