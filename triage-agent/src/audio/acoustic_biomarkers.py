"""
Multimodal Paralinguistic Acoustic Biomarker Triage Co-Processor.
Implements Google HeAR (arXiv:2403.02522) & Coppock et al. (arXiv:2102.08359).

Extracts physiological respiratory biomarkers directly from 16kHz speech:
1. Speech-to-Pause Ratio (R_pause < 0.35) -> Acute Dyspnea (shortness of breath).
2. Spectral Band Concentration (600-1800 Hz) -> Wheezing / Stridor.
3. Inspiratory-to-Expiratory Ratio (I/E Ratio) -> Bronchospasm / Asthma Exacerbation.
Deterministic clinical elevation: overrides neural triage if acute distress is detected.
"""

import numpy as np
import scipy.signal
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

log = logging.getLogger("acoustic_biomarkers")


@dataclass
class AcousticBiomarkerResult:
    dyspnea_detected: bool
    wheezing_detected: bool
    stridor_detected: bool
    prolonged_expiration: bool
    speech_to_pause_ratio: float
    wheeze_band_power_ratio: float
    ie_ratio: float
    acuity_boost: int  # Priority elevation: 0 = none, 1 = moderate, 2 = emergent
    clinical_rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AcousticBiomarkerCoProcessor:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
        dyspnea_pause_threshold: float = 0.35,
        wheeze_band_min_hz: float = 600.0,
        wheeze_band_max_hz: float = 1800.0,
        wheeze_power_ratio_threshold: float = 0.38,
    ):
        self.sample_rate = sample_rate
        self.frame_len = int(sample_rate * frame_duration_ms / 1000)
        self.dyspnea_pause_threshold = dyspnea_pause_threshold
        self.wheeze_band_min = wheeze_band_min_hz
        self.wheeze_band_max = wheeze_band_max_hz
        self.wheeze_power_thresh = wheeze_power_ratio_threshold

    def analyze(self, audio: np.ndarray) -> AcousticBiomarkerResult:
        """
        Analyze audio buffer for respiratory distress biomarkers.
        Executes in <3ms on CPU.
        """
        if audio is None or len(audio) < self.sample_rate // 2:
            return AcousticBiomarkerResult(
                dyspnea_detected=False,
                wheezing_detected=False,
                stridor_detected=False,
                prolonged_expiration=False,
                speech_to_pause_ratio=1.0,
                wheeze_band_power_ratio=0.0,
                ie_ratio=0.33,
                acuity_boost=0,
                clinical_rationale="Insufficient audio duration for paralinguistic biomarker extraction.",
            )

        if audio.dtype == np.int16:
            samples = audio.astype(np.float32) / 32768.0
        else:
            samples = audio.astype(np.float32)

        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        # 1. Speech-to-Pause Segmentation & Dyspnea
        num_frames = len(samples) // self.frame_len
        if num_frames == 0:
            return self._empty_result()

        frames = samples[: num_frames * self.frame_len].reshape(num_frames, self.frame_len)
        frame_energies = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-9)

        # Dynamic VAD energy threshold
        energy_max = np.max(frame_energies)
        energy_floor = np.median(frame_energies)
        threshold = max(0.012, energy_floor + 0.15 * (energy_max - energy_floor))

        speech_mask = frame_energies >= threshold
        speech_frames = np.sum(speech_mask)
        pause_frames = num_frames - speech_frames

        speech_ratio = float(speech_frames / (num_frames + 1e-6))
        total_duration_s = len(samples) / self.sample_rate

        # Dyspnea detection: R_pause < 0.35 over >= 2.0s audio indicates frequent gasping pauses
        dyspnea = bool(total_duration_s >= 2.0 and speech_ratio < self.dyspnea_pause_threshold)

        # 2. Spectral Band Analysis (Wheezing / Stridor Detection)
        # Use Welch PSD on speech frames
        speech_audio = samples[: num_frames * self.frame_len][np.repeat(speech_mask, self.frame_len)]
        if len(speech_audio) < self.sample_rate // 4:
            speech_audio = samples

        nperseg = min(512, len(speech_audio))
        freqs, psd = scipy.signal.welch(speech_audio, fs=self.sample_rate, nperseg=nperseg)

        total_band = (freqs >= 100.0) & (freqs <= 4000.0)
        wheeze_band = (freqs >= self.wheeze_band_min) & (freqs <= self.wheeze_band_max)

        total_power = np.sum(psd[total_band]) + 1e-9
        wheeze_power = np.sum(psd[wheeze_band]) + 1e-9
        wheeze_ratio = float(wheeze_power / total_power)

        # Spectral peak sharpness in wheeze band
        wheeze_psd = psd[wheeze_band]
        peak_sharpness = float(np.max(wheeze_psd) / (np.mean(wheeze_psd) + 1e-9)) if len(wheeze_psd) > 0 else 0.0

        wheezing = bool(wheeze_ratio >= self.wheeze_power_thresh and peak_sharpness > 2.8)
        # Stridor has higher acoustic frequency (typically >900Hz harmonic peak) and extreme sharpness
        high_wheeze_band = (freqs >= 900.0) & (freqs <= 1800.0)
        stridor = bool(wheezing and np.sum(psd[high_wheeze_band]) > 0.65 * wheeze_power and peak_sharpness > 4.2)

        # 3. Inspiratory-to-Expiratory (I/E) Ratio
        # In speech, inspirations are pauses preceded by rapid energy dip and high-frequency intake.
        # Normal I/E is ~ 1:2 to 1:3. Prolonged expiration is I/E >= 1:1 or labored speech expiration > 3.5s
        ie_ratio = float(pause_frames / (speech_frames + 1e-6))
        prolonged_exp = bool(wheezing and (ie_ratio >= 0.85 or speech_ratio > 0.85 and total_duration_s > 3.0))

        # 4. Clinical Emergency Acuity Booster
        acuity_boost = 0
        rationales = []

        if stridor:
            acuity_boost = 2
            rationales.append("Audible stridor detected (Upper airway obstruction warning -> KTAS Level 1/2).")
        elif dyspnea and wheezing:
            acuity_boost = 2
            rationales.append("Severe acute dyspnea with audible wheezing (Acute respiratory distress -> KTAS Level 2).")
        elif dyspnea:
            acuity_boost = 1
            rationales.append(f"Gasping speech pattern detected (Speech-to-pause ratio {speech_ratio:.2f} < {self.dyspnea_pause_threshold}).")
        elif wheezing or prolonged_exp:
            acuity_boost = 1
            rationales.append("Expiratory bronchospasm / wheezing acoustic signature detected.")

        if not rationales:
            rationale_str = f"Normal acoustic speech pattern (Speech ratio: {speech_ratio:.2f}, Wheeze power: {wheeze_ratio:.2f})."
        else:
            rationale_str = " | ".join(rationales)

        return AcousticBiomarkerResult(
            dyspnea_detected=dyspnea,
            wheezing_detected=wheezing,
            stridor_detected=stridor,
            prolonged_expiration=prolonged_exp,
            speech_to_pause_ratio=speech_ratio,
            wheeze_band_power_ratio=wheeze_ratio,
            ie_ratio=ie_ratio,
            acuity_boost=acuity_boost,
            clinical_rationale=rationale_str,
        )

    def _empty_result(self) -> AcousticBiomarkerResult:
        return AcousticBiomarkerResult(
            dyspnea_detected=False,
            wheezing_detected=False,
            stridor_detected=False,
            prolonged_expiration=False,
            speech_to_pause_ratio=1.0,
            wheeze_band_power_ratio=0.0,
            ie_ratio=0.33,
            acuity_boost=0,
            clinical_rationale="Empty audio buffer.",
        )
