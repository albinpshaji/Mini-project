"""
High-Resolution Microsecond Stage Latency Profiler for OmniSpec Triage Pipeline.
Instruments and breaks down the complete Glass-to-Glass turnaround time into 6 distinct stages.
"""

import time
from typing import Dict, Any, Optional


class PipelineLatencyTracker:
    """
    Microsecond-accurate stage tracker using time.perf_counter_ns().
    Records every milestone from patient voice cessation to speaker audio dispatch.
    """

    def __init__(self):
        self.timestamps_ns: Dict[str, int] = {}
        self.metadata: Dict[str, Any] = {}

    def mark(self, stage_name: str):
        """Records the current high-resolution timestamp for a milestone."""
        self.timestamps_ns[stage_name] = time.perf_counter_ns()

    def set_meta(self, key: str, value: Any):
        self.metadata[key] = value

    def _diff_ms(self, start_key: str, end_key: str, default: float = 0.0) -> float:
        if start_key in self.timestamps_ns and end_key in self.timestamps_ns:
            diff_ns = self.timestamps_ns[end_key] - self.timestamps_ns[start_key]
            return max(0.0, diff_ns / 1_000_000.0)
        return default

    def compute_stages(self, mode: str = "sequential", committed: bool = False) -> Dict[str, float]:
        """
        Computes the 6 pipeline stages and total Glass-to-Glass latency in milliseconds.
        
        Milestones tracked:
          1. 'patient_speech_end': User stops vocalizing
          2. 'vad_turn_closed': VAD silence timeout expires (or syntactic closure fires)
          3. 'asr_final_ready': Whisper final model finishes transcription
          4. 'slm_prefill_done': System prompt KV-cache prefilled (or restored)
          5. 'slm_first_token': First generated token emitted (or draft verified)
          6. 'tools_guard_done': Guardrail + asynchronous tools complete
          7. 'first_audio_packet_out': Piper TTS dispatches first PCM chunk to speaker DAC
        """
        # 1. Stage 1: VAD Silence Wait (ms)
        vad_silence_ms = self._diff_ms("patient_speech_end", "vad_turn_closed", default=500.0)

        # 2. Stage 2: Final ASR Pass (ms)
        asr_final_ms = self._diff_ms("vad_turn_closed", "asr_final_ready", default=120.0)

        # 3. Stage 3: SLM Prefill (ms)
        # If system prompt is pinned in KV cache, prefill is effectively 0ms
        slm_prefill_ms = self._diff_ms("asr_final_ready", "slm_prefill_done", default=0.0)

        # 4. Stage 4: SLM Generation (ms)
        # In speculative committed mode, generation is 0ms perceived delay
        if mode == "speculative" and committed:
            slm_gen_ms = 0.0
        else:
            slm_gen_ms = self._diff_ms("slm_prefill_done", "slm_first_token", default=0.0)
            if slm_gen_ms == 0.0:
                # Fallback if first_token was marked directly from asr_final_ready
                slm_gen_ms = self._diff_ms("asr_final_ready", "slm_first_token", default=0.0)

        # 5. Stage 5: Clinical Tools & Guardrail (ms)
        tools_guard_ms = self._diff_ms("slm_first_token", "tools_guard_done", default=1.5)

        # 6. Stage 6: TTS Audio Synthesis (TTFA Chunk) (ms)
        tts_synthesis_ms = self._diff_ms("tools_guard_done", "first_audio_packet_out", default=0.0)
        if tts_synthesis_ms == 0.0:
            tts_synthesis_ms = self._diff_ms("slm_first_token", "first_audio_packet_out", default=75.0)

        # Total Glass-to-Glass (ms): from patient_speech_end to first_audio_packet_out
        total_g2g_ms = self._diff_ms("patient_speech_end", "first_audio_packet_out")
        if total_g2g_ms == 0.0:
            # Reconstruct from stage components if end-to-end span was interrupted
            total_g2g_ms = vad_silence_ms + asr_final_ms + slm_prefill_ms + slm_gen_ms + tools_guard_ms + tts_synthesis_ms

        return {
            "1_vad_silence_ms": round(vad_silence_ms, 2),
            "2_asr_final_ms": round(asr_final_ms, 2),
            "3_slm_prefill_ms": round(slm_prefill_ms, 2),
            "4_slm_gen_ms": round(slm_gen_ms, 2),
            "5_tools_guard_ms": round(tools_guard_ms, 2),
            "6_tts_synthesis_ms": round(tts_synthesis_ms, 2),
            "total_glass_to_glass_ms": round(total_g2g_ms, 2),
            "mode": mode,
            "committed": committed,
        }
