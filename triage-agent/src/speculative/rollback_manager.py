"""
src/speculative/rollback_manager.py
=============================================================================
Rollback Manager -- coordinates cancelling in-flight speculation and TTS
when divergence is detected, or invokes Adaptive CSSV to salvage verified
prefixes on benign context-shift turns (Phase 6).
=============================================================================
"""

import logging
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any

from src.speculative.divergence_detector import compute_divergence, DivergenceResult

log = logging.getLogger("rollback_manager")


@dataclass
class RollbackDecision:
    committed: bool
    divergence: Optional[DivergenceResult]
    speculative_response: Optional[str]
    salvaged_prefix: bool = False
    prefix_text: Optional[str] = None
    cssv_telemetry: Optional[Dict[str, Any]] = None


def resolve_speculation(
    speculator,
    tts_engine,
    final_transcript: str,
    divergence_threshold: float = 0.7,
    use_cssv: bool = True,
    on_prefix_ready: Optional[Callable[[str], None]] = None,
) -> RollbackDecision:
    """
    Resolves speculation against final transcript.
    - If similarity >= threshold: full speculative commit.
    - If divergence detected:
        * If negation polarity flipped: immediate 100% safe rollback.
        * If polarity intact and CSSV enabled: executes Context-Shift Self-Speculative
          Verification (CSSV) to salvage verified prefix Y_commit and dispatch to TTS.
        * Otherwise: falls back to full sequential regeneration.
    """
    triggering_partial = speculator.get_triggering_partial()
    if triggering_partial is None:
        return RollbackDecision(committed=False, divergence=None, speculative_response=None)

    divergence = compute_divergence(triggering_partial, final_transcript, threshold=divergence_threshold)

    # 1. Exact or High Lexical Overlap Commit
    if not divergence.diverged:
        speculative_response = speculator.resolve(final_transcript)
        if speculative_response is not None:
            log.info(
                "COMMIT (similarity=%.2f >= threshold=%.2f) -- using full speculative response.",
                divergence.similarity, divergence_threshold,
            )
            return RollbackDecision(
                committed=True,
                divergence=divergence,
                speculative_response=speculative_response,
                salvaged_prefix=False,
                prefix_text=speculative_response,
                cssv_telemetry={"mode": "full_commit", "similarity": divergence.similarity},
            )

    # 2. Polarity / Negation Inversion Fast-Fail Shield
    if divergence.polarity_shifted:
        log.warning(
            "POLARITY INVERSION SHIELD TRIGGERED (similarity=%.2f) -- 100%% safe rollback. "
            "Triggering: %r | Final: %r",
            divergence.similarity, triggering_partial, final_transcript,
        )
        speculator.cancel()
        tts_engine.cancel()
        return RollbackDecision(
            committed=False,
            divergence=divergence,
            speculative_response=None,
            salvaged_prefix=False,
            prefix_text=None,
            cssv_telemetry={"mode": "polarity_abort", "similarity": divergence.similarity},
        )

    # 3. Phase 6 Adaptive CSSV Prefix Salvage
    if use_cssv and hasattr(speculator, "resolve_cssv"):
        log.info(
            "DIVERGENCE DETECTED (similarity=%.2f < threshold=%.2f) -- Invoking Adaptive CSSV prefix salvage.",
            divergence.similarity, divergence_threshold,
        )
        salvage_res = speculator.resolve_cssv(
            final_transcript=final_transcript,
            on_prefix_ready=on_prefix_ready,
        )
        if salvage_res and salvage_res.get("salvaged_prefix"):
            prefix_text = salvage_res.get("prefix_text", "")
            full_response = salvage_res.get("full_response", prefix_text)
            telemetry = salvage_res.get("telemetry", {})
            log.info(
                "⚡ CSSV PREFIX SALVAGED (%d tokens / %.1f%% of draft) -- Dispatched to TTS in %sms.",
                telemetry.get("accepted_prefix_len", 0),
                telemetry.get("prefix_acceptance_rate", 0.0) * 100.0,
                telemetry.get("cssv_verify_ms", 0.0),
            )
            return RollbackDecision(
                committed=True,
                divergence=divergence,
                speculative_response=full_response,
                salvaged_prefix=True,
                prefix_text=prefix_text,
                cssv_telemetry=telemetry,
            )

    # 4. Fallback Full Rollback
    log.warning(
        "ROLLBACK EXECUTED (similarity=%.2f < threshold=%.2f) -- Regenerating response.",
        divergence.similarity, divergence_threshold,
    )
    speculator.cancel()
    tts_engine.cancel()
    return RollbackDecision(
        committed=False,
        divergence=divergence,
        speculative_response=None,
        salvaged_prefix=False,
        prefix_text=None,
        cssv_telemetry={"mode": "fallback_abort", "similarity": divergence.similarity},
    )
