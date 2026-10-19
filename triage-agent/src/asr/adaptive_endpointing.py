"""
src/asr/adaptive_endpointing.py
=============================================================================
Clinical Slot-Entropy & Dyspnea-Aware Adaptive Endpointing (Phase 6).
Dynamically modulates VAD silence cutoff based on clinical slot satisfaction
and acoustic dyspneic gasp energy.
=============================================================================
"""

import re
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class EndpointingResult:
    silence_timeout_ms: int
    slot_resolved: bool
    dyspnea_energy: float
    rule_matched: str


class ClinicalSlotEntropyClassifier:
    """
    Evaluates whether an incoming partial ASR hypothesis achieves semantic closure
    for queried clinical slots (Duration, Severity, Location, Red-Flag Emergency).
    """

    # Slot 1: Clinical Duration / Onset resolution
    DURATION_PATTERNS = re.compile(
        r"\b(?:since|for)\s+(?:yesterday|this\s+morning|last\s+night|(?:one|two|three|four|five|six|seven|eight|several|\d+)\s+(?:days?|hours?|weeks?|months?))\b"
        r"|\b(?:all\s+day|at\s+night|this\s+morning|yesterday|earlier\s+today|an\s+hour\s+ago|last\s+week)\b",
        re.IGNORECASE
    )

    # Slot 2: Pain Severity / Scale resolution (0-10 or qualitative)
    SEVERITY_PATTERNS = re.compile(
        r"\b(?:[0-9]|10)\s*(?:out\s+of\s+10|/10|\s*scale)\b"
        r"|\b(?:mild|moderate|severe|excruciating|unbearable|intense|sharp|dull|crushing)\s+(?:pain|ache|pressure)?\b",
        re.IGNORECASE
    )

    # Slot 3: Anatomical Location resolution
    ANATOMY_PATTERNS = re.compile(
        r"\b(?:in|on|around|radiating\s+to)\s+(?:my\s+)?"
        r"(?:chest|head|left\s+arm|right\s+arm|jaw|back|neck|throat|stomach|abdomen|leg|ankle|knee)\b",
        re.IGNORECASE
    )

    # Slot 4: Critical High-Acuity Red-Flag Triggers (Instant Slot Saturation)
    RED_FLAG_PATTERNS = re.compile(
        r"\b(?:can'?t\s+breathe|trouble\s+breathing|shortness\s+of\s+breath|chest\s+pressure|passed\s+out|coughing\s+blood|face\s+droop|sudden\s+weakness)\b",
        re.IGNORECASE
    )

    # Conversational Terminal Statements
    CONVERSATIONAL_TERMINALS = re.compile(
        r"\b(?:that'?s\s+(?:all|everything|it)|that\s+is\s+(?:all|everything|it)|nothing\s+else|no\s+other\s+symptoms|none\s+at\s+all)\s*$"
        r"|^(?:yes|no|nope|yeah)\s*$",
        re.IGNORECASE
    )

    def is_slot_resolved(self, partial_text: str) -> bool:
        """Returns True if the partial utterance resolves a clinical triage slot."""
        text = (partial_text or "").strip()
        if not text:
            return False

        if (
            self.RED_FLAG_PATTERNS.search(text)
            or self.DURATION_PATTERNS.search(text)
            or self.SEVERITY_PATTERNS.search(text)
            or self.ANATOMY_PATTERNS.search(text)
            or self.CONVERSATIONAL_TERMINALS.search(text)
        ):
            return True

        return False


class AcousticDyspneaDetector:
    """
    Sub-3ms feature extractor computing energy-gated inspiratory gasp and stridor/wheeze
    energy in the 500-2500 Hz frequency band during conversational pauses.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        tau_noise: float = 0.003,
        tau_speech: float = 0.040,
        gamma_1: float = 1.2,
        gamma_2: float = 2.0,
    ):
        self.sample_rate = sample_rate
        self.tau_noise = tau_noise
        self.tau_speech = tau_speech
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2

    def compute_dyspnea_index(self, audio_chunk: Optional[np.ndarray]) -> float:
        """
        Computes E_resp in [0, 1].
        Energy-gated: Returns 0.0 in true ambient silence (RMS <= tau_noise)
        to prevent false 900ms timeouts on quiet rooms.
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return 0.0

        # Convert to float32 normalized [-1.0, 1.0]
        if audio_chunk.dtype == np.int16:
            samples = audio_chunk.astype(np.float32) / 32768.0
        else:
            samples = audio_chunk.astype(np.float32)

        rms = np.sqrt(np.mean(samples ** 2) + 1e-12)

        # Signal Processing Safeguard: Energy noise-floor bandpass indicator
        if rms <= self.tau_noise or rms >= self.tau_speech:
            return 0.0

        # 1. Zero Crossing Rate (ZCR)
        zero_crossings = np.sum(np.abs(np.diff(samples > 0)))
        zcr = zero_crossings / max(1, len(samples) - 1)

        # 2. Spectral Mid-Band Energy Ratio (500 Hz - 2500 Hz)
        fft_vals = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(len(samples), d=1.0 / self.sample_rate)

        total_energy = np.sum(fft_vals ** 2) + 1e-12
        mid_band_mask = (freqs >= 500.0) & (freqs <= 2500.0)
        mid_band_energy = np.sum(fft_vals[mid_band_mask] ** 2)
        mid_band_ratio = mid_band_energy / total_energy

        # 3. Sigmoid activation
        logit = self.gamma_1 * zcr + self.gamma_2 * mid_band_ratio - 1.5
        e_resp = 1.0 / (1.0 + np.exp(-logit))
        return float(np.clip(e_resp, 0.0, 1.0))


class DyspneaAwareEndpointing:
    """
    Unified Adaptive Endpointing engine combining Slot-Entropy closure
    with Acoustic Dyspnea pause expansion.
    Formula: tau_vad = tau_base - alpha * I_slot + beta * E_resp
    """

    def __init__(
        self,
        tau_base_ms: int = 450,
        alpha_slot_ms: int = 270,
        beta_dyspnea_ms: int = 450,
        sample_rate: int = 16000,
    ):
        self.tau_base_ms = tau_base_ms
        self.alpha_slot_ms = alpha_slot_ms
        self.beta_dyspnea_ms = beta_dyspnea_ms

        self.slot_classifier = ClinicalSlotEntropyClassifier()
        self.dyspnea_detector = AcousticDyspneaDetector(sample_rate=sample_rate)

    def evaluate(
        self,
        partial_text: str,
        pause_audio_chunk: Optional[np.ndarray] = None,
    ) -> EndpointingResult:
        """
        Computes dynamic silence threshold in milliseconds.
        - Concise slot completion: collapses to 180 ms (saving 270 ms dead air).
        - Dyspneic pause/gasp: expands up to 900 ms (preventing premature cutoffs).
        """
        is_slot_res = self.slot_classifier.is_slot_resolved(partial_text)
        e_resp = self.dyspnea_detector.compute_dyspnea_index(pause_audio_chunk)

        # Baseline: 450ms
        timeout = float(self.tau_base_ms)

        if is_slot_res:
            timeout -= self.alpha_slot_ms
            rule = "slot_resolved_acceleration"
        else:
            rule = "open_turn"

        if e_resp > 0.40:
            timeout += self.beta_dyspnea_ms * e_resp
            rule = "dyspneic_gasp_expansion"

        final_timeout_ms = int(np.clip(round(timeout), 180, 900))

        return EndpointingResult(
            silence_timeout_ms=final_timeout_ms,
            slot_resolved=is_slot_res,
            dyspnea_energy=round(e_resp, 3),
            rule_matched=rule,
        )
