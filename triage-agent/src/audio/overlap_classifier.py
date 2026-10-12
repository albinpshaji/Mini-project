"""
Two-Tier Semantic Overlap & Backchannel Gate (SOBC) for Voice Barge-In.
Implements Meta dGSLM (arXiv:2203.16502) & Mini-Omni2 (arXiv:2410.11190).

Tier 1: 300ms Acoustic Grace Window (eliminates hair-trigger cutoffs on coughs/breaths).
Tier 2: Semantic Intent Filter:
        - Non-competitive backchannels ('uh-huh', 'yeah', 'okay') -> Keep TTS playing.
        - Competitive floor grabs ('wait', 'stop', 'no', 'actually') or >= 3 words -> Immediate Cancel.
"""

import time
import re
import logging
from enum import Enum
from typing import Callable, Optional, Set

log = logging.getLogger("overlap_classifier")


class OverlapDecision(Enum):
    NONE = "none"                  # No overlap detected (silence or non-speech)
    IN_GRACE = "in_grace"          # Overlap detected, inside 300ms acoustic grace window
    CONTINUE = "continue"          # Overlap classified as non-competitive backchannel (keep playing)
    INTERRUPT = "interrupt"        # Competitive floor grab detected (atomic flush / cancel TTS)


class SemanticOverlapClassifier:
    DEFAULT_BACKCHANNELS: Set[str] = {
        "uh-huh", "yeah", "okay", "ok", "right", "mm-hmm", "sure",
        "yep", "mhm", "yes", "got it", "i see", "ah", "yup", "alright",
    }

    DEFAULT_INTERRUPTS: Set[str] = {
        "wait", "stop", "no", "actually", "listen", "hold on", "hang on",
        "excuse me", "doctor", "nurse", "help", "pause", "shut up",
        "dont", "don't", "cancel", "wait a minute", "hold up",
    }

    def __init__(
        self,
        grace_window_ms: float = 300.0,
        max_unclassified_speech_ms: float = 450.0,
        speech_rms_threshold: float = 0.02,
        on_interrupt: Optional[Callable[[], None]] = None,
    ):
        """
        Args:
            grace_window_ms: Acoustic grace duration (ms) before taking action on speech energy.
            max_unclassified_speech_ms: Max sustained acoustic speech duration before triggering interrupt.
            speech_rms_threshold: Minimum RMS level for clean speech to trigger overlap detection.
            on_interrupt: Callback fired immediately when a competitive floor grab is confirmed.
        """
        self.grace_window_ms = grace_window_ms
        self.max_unclassified_speech_ms = max_unclassified_speech_ms
        self.speech_rms_threshold = speech_rms_threshold
        self.on_interrupt = on_interrupt

        self._overlap_start_time: Optional[float] = None
        self._is_tts_active: bool = False
        self._last_decision: OverlapDecision = OverlapDecision.NONE
        self._interrupted: bool = False

        # Metrics & Telemetry
        self.backchannel_rejections: int = 0
        self.competitive_interrupts: int = 0
        self.grace_expired_interrupts: int = 0

    def set_tts_active(self, active: bool):
        """Notified by pipeline/TTS engine when speech playback starts or stops."""
        self._is_tts_active = active
        self._overlap_start_time = None
        self._last_decision = OverlapDecision.NONE
        self._interrupted = False

    def process_acoustic_frame(self, frame_rms: float, double_talk: bool = False) -> OverlapDecision:
        """
        Evaluates real-time audio frame energy during active TTS playback.
        
        Args:
            frame_rms: RMS energy of the clean mic frame.
            double_talk: Boolean flag from AEC DTD indicating concurrent mic energy.
        """
        if not self._is_tts_active or self._interrupted:
            return OverlapDecision.NONE

        has_speech_energy = (frame_rms >= self.speech_rms_threshold) or double_talk

        now = time.time()
        if not has_speech_energy:
            # If speech stopped during grace window, reset cleanly (transient cough/breath)
            if self._overlap_start_time is not None:
                elapsed_ms = (now - self._overlap_start_time) * 1000.0
                if elapsed_ms < self.grace_window_ms:
                    log.debug("Acoustic transient resolved within grace window (%.1fms).", elapsed_ms)
                    self._overlap_start_time = None
                    self._last_decision = OverlapDecision.NONE
            return OverlapDecision.NONE

        # Speech energy present while TTS is active
        if self._overlap_start_time is None:
            self._overlap_start_time = now
            self._last_decision = OverlapDecision.IN_GRACE
            log.debug("TTS overlap initiated: entering %.0fms acoustic grace window.", self.grace_window_ms)
            return OverlapDecision.IN_GRACE

        elapsed_ms = (now - self._overlap_start_time) * 1000.0

        # Tier 1: Still within acoustic grace window
        if elapsed_ms < self.grace_window_ms:
            self._last_decision = OverlapDecision.IN_GRACE
            return OverlapDecision.IN_GRACE

        # If speech persists past max threshold without ASR classification, assume floor takeover
        if elapsed_ms >= self.max_unclassified_speech_ms:
            log.info("Sustained acoustic speech overlap (%.0fms) -> triggering barge-in interrupt.", elapsed_ms)
            self._trigger_interrupt()
            self.grace_expired_interrupts += 1
            return OverlapDecision.INTERRUPT

        return self._last_decision

    def process_partial_transcript(self, partial_text: str) -> OverlapDecision:
        """
        Tier 2: Semantic Intent Filter evaluating incoming ASR partial hypotheses.
        
        Args:
            partial_text: Partial transcription received from streaming ASR during active TTS.
        """
        if not self._is_tts_active or self._interrupted:
            return OverlapDecision.NONE

        cleaned = re.sub(r"[^\w\s\-]", "", partial_text.lower()).strip()
        if not cleaned:
            return self._last_decision

        tokens = cleaned.split()
        num_tokens = len(tokens)

        # Check for explicit competitive floor grabs
        for token in tokens:
            if token in self.DEFAULT_INTERRUPTS:
                log.info("SOBC: Competitive interrupt keyword detected (%r) -> triggering barge-in.", token)
                self.competitive_interrupts += 1
                self._trigger_interrupt()
                return OverlapDecision.INTERRUPT

        # Multi-word competitive utterance (>= 3 words is floor takeover)
        if num_tokens >= 3:
            log.info("SOBC: Multi-word utterance (%d words: %r) -> triggering barge-in.", num_tokens, partial_text)
            self.competitive_interrupts += 1
            self._trigger_interrupt()
            return OverlapDecision.INTERRUPT

        # Non-competitive backchannels ("uh-huh", "yeah", "okay")
        is_backchannel = any(cleaned == bc or (num_tokens <= 2 and all(t in self.DEFAULT_BACKCHANNELS for t in tokens))
                             for bc in self.DEFAULT_BACKCHANNELS)

        if is_backchannel:
            log.info("SOBC: Non-competitive backchannel detected (%r) -> maintaining TTS playback.", cleaned)
            self.backchannel_rejections += 1
            # Reset overlap timer so backchannel does not trigger max_unclassified_speech_ms
            self._overlap_start_time = None
            self._last_decision = OverlapDecision.CONTINUE
            return OverlapDecision.CONTINUE

        return self._last_decision

    def _trigger_interrupt(self):
        """Execute atomic cancellation and invoke user interrupt callback."""
        self._interrupted = True
        self._last_decision = OverlapDecision.INTERRUPT
        self._overlap_start_time = None
        if self.on_interrupt:
            try:
                self.on_interrupt()
            except Exception as e:
                log.error("Error executing on_interrupt callback: %s", e)
