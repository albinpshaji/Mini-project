"""
Acoustic Echo Canceller (AEC) for OmniSpec-Triage Full-Duplex Audio.

Implements real-time software echo cancellation with:
1. Hardware-timestamped delay tracking via cross-correlation.
2. Normalized Least Mean Squares (NLMS) adaptive filtering.
3. Energy-ratio Double Talk Detection (DTD) to prevent filter divergence.
4. Non-linear spectral suppression for residual acoustic echo bleed.
"""

import numpy as np
import logging
from typing import Tuple, Optional

log = logging.getLogger("echo_canceller")


class AcousticEchoCanceller:
    def __init__(
        self,
        sample_rate: int = 16000,
        filter_length: int = 512,
        step_size: float = 0.25,
        leakage: float = 0.9999,
        dtd_threshold: float = 1.8,
        residual_suppression_db: float = 18.0,
        max_delay_ms: int = 250,
    ):
        """
        Args:
            sample_rate: Audio sampling frequency (Hz), default 16000.
            filter_length: Number of FIR filter taps (512 taps = 32ms impulse response at 16kHz).
            step_size: NLMS adaptation step size (mu).
            leakage: Leaky factor to prevent weight explosion.
            dtd_threshold: Double-Talk Detection threshold ratio.
            residual_suppression_db: Non-linear attenuation of residual echo during agent speech.
            max_delay_ms: Maximum expected hardware/room loopback delay to search.
        """
        self.sample_rate = sample_rate
        self.filter_length = filter_length
        self.step_size = step_size
        self.leakage = leakage
        self.dtd_threshold = dtd_threshold
        self.residual_suppression = 10.0 ** (-residual_suppression_db / 20.0)
        self.max_delay_samples = int(max_delay_ms * sample_rate / 1000)

        # Adaptive filter weights
        self.weights = np.zeros(self.filter_length, dtype=np.float32)

        # Circular buffer for reference (speaker) audio
        self.ref_buffer_size = self.max_delay_samples + self.filter_length + 4096
        self.ref_buffer = np.zeros(self.ref_buffer_size, dtype=np.float32)
        self.ref_write_idx = 0

        # Delay tracking
        self.estimated_delay_samples = int(0.040 * sample_rate)  # initial guess 40ms
        self.delay_update_counter = 0
        self.delay_update_interval = 25  # evaluate correlation every 25 frames

        # State stats
        self.is_speaker_active = False
        self.adaptation_enabled = True
        self.echo_return_loss_enhancement_db = 0.0

    def push_speaker_reference(self, speaker_pcm: np.ndarray):
        """
        Queue speaker audio emitted by TTS DAC into the reference loopback ring buffer.
        Accepts float32 [-1, 1] or int16 PCM.
        """
        if speaker_pcm is None or len(speaker_pcm) == 0:
            return

        if speaker_pcm.dtype == np.int16:
            samples = speaker_pcm.astype(np.float32) / 32768.0
        else:
            samples = speaker_pcm.astype(np.float32)

        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        n = len(samples)
        if n >= self.ref_buffer_size:
            samples = samples[-self.ref_buffer_size:]
            n = len(samples)

        end_idx = self.ref_write_idx + n
        if end_idx <= self.ref_buffer_size:
            self.ref_buffer[self.ref_write_idx:end_idx] = samples
        else:
            first_part = self.ref_buffer_size - self.ref_write_idx
            self.ref_buffer[self.ref_write_idx:] = samples[:first_part]
            self.ref_buffer[: end_idx % self.ref_buffer_size] = samples[first_part:]

        self.ref_write_idx = end_idx % self.ref_buffer_size
        ref_rms = np.sqrt(np.mean(samples ** 2) + 1e-9)
        self.is_speaker_active = ref_rms > 0.01

    def cancel_echo(self, mic_pcm: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Process a microphone frame through the adaptive echo canceller.
        
        Args:
            mic_pcm: 1D numpy array of mic samples (float32 or int16).
            
        Returns:
            Tuple of:
                clean_pcm: Echo-cancelled output in same dtype as input.
                double_talk_detected: True if patient speech was detected during active playback.
        """
        is_int16 = (mic_pcm.dtype == np.int16)
        if is_int16:
            mic_float = mic_pcm.astype(np.float32) / 32768.0
        else:
            mic_float = mic_pcm.astype(np.float32)

        if mic_float.ndim > 1:
            mic_float = mic_float.mean(axis=1)

        frame_len = len(mic_float)
        if frame_len == 0:
            return mic_pcm, False

        # If speaker is completely silent, pass mic audio directly with zero overhead
        ref_rms = self._get_recent_ref_rms(frame_len + self.estimated_delay_samples)
        if ref_rms < 0.005 and not self.is_speaker_active:
            return mic_pcm, False

        # Periodic delay estimation update
        self.delay_update_counter += 1
        if self.delay_update_counter >= self.delay_update_interval:
            self.delay_update_counter = 0
            self._update_delay_estimate(mic_float)

        # Extract aligned reference segment
        ref_segment = self._get_delayed_reference(frame_len, self.estimated_delay_samples)

        # Run NLMS sample-by-sample or block-wise
        error_float = np.zeros(frame_len, dtype=np.float32)
        double_talk = False

        eps = 1e-6
        weights = self.weights
        w_len = self.filter_length

        # Maintain a sliding history of the reference for filtering
        hist_ref = self._get_delayed_reference(frame_len + w_len, self.estimated_delay_samples)

        # Double-talk detection: compare mic energy to reference energy
        mic_energy = np.mean(mic_float ** 2) + eps
        ref_energy = np.mean(ref_segment ** 2) + eps

        # Geigel DTD criterion
        max_ref = np.max(np.abs(hist_ref)) + eps
        max_mic = np.max(np.abs(mic_float)) + eps
        dtd_ratio = max_mic / max_ref

        # Double talk if mic is strong while ref is present
        if ref_energy > 0.001 and (dtd_ratio > self.dtd_threshold or mic_energy > 3.0 * ref_energy):
            double_talk = True

        for i in range(frame_len):
            # Input vector x(n) from reference history
            x_vec = hist_ref[i : i + w_len][::-1]
            norm_x = np.dot(x_vec, x_vec) + eps

            # Estimated echo d_hat(n) = w^T * x
            d_hat = np.dot(weights, x_vec)

            # Error e(n) = y_mic(n) - d_hat(n)
            e = mic_float[i] - d_hat
            error_float[i] = e

            # Adapt filter only when NOT in double-talk and reference has significant energy
            if not double_talk and self.adaptation_enabled and norm_x > 0.002:
                mu_eff = self.step_size / norm_x
                weights = self.leakage * weights + mu_eff * e * x_vec

        self.weights = weights

        # Post-filter non-linear suppression when speaker is loud and no double-talk
        if not double_talk and ref_energy > 0.008:
            clean_float = error_float * self.residual_suppression
        else:
            clean_float = error_float

        # Calculate ERLE (Echo Return Loss Enhancement) metric
        err_energy = np.mean(clean_float ** 2) + eps
        self.echo_return_loss_enhancement_db = float(10.0 * np.log10(mic_energy / err_energy))

        # Convert back to original format
        if is_int16:
            clean_pcm = np.clip(clean_float * 32768.0, -32768.0, 32767.0).astype(np.int16)
        else:
            clean_pcm = clean_float

        return clean_pcm, double_talk

    def _get_recent_ref_rms(self, num_samples: int) -> float:
        num = min(num_samples, self.ref_buffer_size)
        start_idx = (self.ref_write_idx - num) % self.ref_buffer_size
        if start_idx + num <= self.ref_buffer_size:
            seg = self.ref_buffer[start_idx : start_idx + num]
        else:
            first = self.ref_buffer[start_idx:]
            second = self.ref_buffer[: (start_idx + num) % self.ref_buffer_size]
            seg = np.concatenate([first, second])
        return float(np.sqrt(np.mean(seg ** 2) + 1e-9))

    def _get_delayed_reference(self, length: int, delay_samples: int) -> np.ndarray:
        total_back = delay_samples + length
        start_idx = (self.ref_write_idx - total_back) % self.ref_buffer_size
        if start_idx + length <= self.ref_buffer_size:
            return self.ref_buffer[start_idx : start_idx + length].copy()
        first = self.ref_buffer[start_idx:]
        second = self.ref_buffer[: (start_idx + length) % self.ref_buffer_size]
        return np.concatenate([first, second])

    def _update_delay_estimate(self, mic_frame: np.ndarray):
        """Cross-correlate mic with reference history to align delay."""
        search_len = min(self.max_delay_samples, self.ref_buffer_size // 2)
        if search_len < len(mic_frame):
            return

        ref_hist = self._get_delayed_reference(search_len, 0)
        if np.max(np.abs(ref_hist)) < 0.02 or np.max(np.abs(mic_frame)) < 0.02:
            return

        # Cross correlation
        corr = np.correlate(ref_hist, mic_frame, mode="valid")
        if len(corr) > 0:
            best_lag = int(np.argmax(corr))
            # Smooth delay updates to avoid jitter
            self.estimated_delay_samples = int(0.8 * self.estimated_delay_samples + 0.2 * best_lag)

    def reset(self):
        """Reset adaptive weights and internal buffers."""
        self.weights.fill(0)
        self.ref_buffer.fill(0)
        self.ref_write_idx = 0
        self.is_speaker_active = False
        self.echo_return_loss_enhancement_db = 0.0
