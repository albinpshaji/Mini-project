"""
tests/test_cssv_speculation.py
=============================================================================
Unit and Functional Tests for Phase 6:
Continuous Speculative Shift & Verification (CSSV) and Dyspnea-Aware Endpointing.

Validates:
1. Adaptive CSSV Engine Initialization and KV Backend Probe.
2. Exact Match Zero-Latency Speculative Verification.
3. Divergent Prefix Salvage on Context-Shift Utterances.
4. Negation Flip Shield Fast-Fail (Zero False Speculative Commits).
5. Surgical RoPE Positional Shift & In-VRAM Sequence Eviction Logic.
6. RollbackManager Integration with Immediate Handoff Callback.
7. Clinical Slot Entropy Classifier (Slot-Driven VAD Compression).
8. Acoustic Dyspnea & Respiratory Gasp Detector.
9. Dyspnea-Aware Adaptive Endpointing Dynamic VAD Threshold Bounding.
=============================================================================
"""

import os
import sys
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.speculative.cssv_engine import (
    AdaptiveCSSVEngine,
    KVBackendMode,
)
from src.speculative.rollback_manager import resolve_speculation, RollbackDecision
from src.asr.adaptive_endpointing import (
    ClinicalSlotEntropyClassifier,
    AcousticDyspneaDetector,
    DyspneaAwareEndpointing,
    EndpointingResult,
)


class TestCSSVSpeculation(unittest.TestCase):
    """Test suite for Phase 6 Adaptive CSSV and Endpointing Innovations."""

    def setUp(self):
        self.cssv = AdaptiveCSSVEngine(
            model_path=None,
            system_prompt="You are a clinical triage AI.",
            ollama_model_name="qwen3.5:4b",
        )

    def test_01_backend_probe(self):
        """Verify CSSV engine detects KV backend mode without crashing."""
        self.assertIn(
            self.cssv.backend_mode,
            [KVBackendMode.SURGICAL_POSITIONAL, KVBackendMode.MULTI_SLOT_RECURRENT, KVBackendMode.OLLAMA_BRIDGE],
        )
        self.assertGreaterEqual(self.cssv.draft_k, 4)
        self.assertLessEqual(self.cssv.draft_k, 16)

    def test_02_exact_match_zero_latency(self):
        """When final transcript matches partial exactly, commit immediately with 0 ms re-verification."""
        partial = "I have chest pain"
        final = "I have chest pain"
        draft = "I understand you have chest pain. Are you experiencing shortness of breath or dizziness?"

        resp, telem = self.cssv.verify_and_splice(
            partial_text=partial,
            final_text=final,
            draft_response=draft,
        )

        self.assertEqual(resp, draft)
        self.assertEqual(telem["mode"], "exact_match")
        self.assertEqual(telem["accepted_prefix_len"], len(draft.split()))
        self.assertEqual(telem["prefix_acceptance_rate"], 1.0)
        self.assertLess(telem["cssv_verify_ms"], 5.0)

    def test_03_divergent_prefix_salvage(self):
        """When final transcript adds context, CSSV salvages common verified prefix."""
        partial = "I have severe chest pain"
        final = "I have severe chest pain since yesterday and it radiates to my left arm"
        draft = "I hear that you have severe chest pain. Can you tell me if it radiates anywhere else?"

        prefix_received = []

        def _on_prefix(p):
            prefix_received.append(p)

        resp, telem = self.cssv.verify_and_splice(
            partial_text=partial,
            final_text=final,
            draft_response=draft,
            on_prefix_ready=_on_prefix,
        )

        self.assertGreater(telem["accepted_prefix_len"], 0)
        self.assertGreater(telem["prefix_acceptance_rate"], 0.0)
        self.assertFalse(telem["negation_flip_detected"])
        self.assertIn("cssv_verify_ms", telem)
        self.assertIn("salvage_speedup", telem)

    def test_04_negation_flip_fast_fail(self):
        """Hard polarity flip (no pain -> severe pain or vice-versa) must trigger immediate abort."""
        partial = "I have no chest pain"
        final = "I have severe chest pain radiating to my neck"
        draft = "It is good that you do not have chest pain. What other symptoms are bothering you?"

        resp, telem = self.cssv.verify_and_splice(
            partial_text=partial,
            final_text=final,
            draft_response=draft,
        )

        self.assertTrue(telem["negation_flip_detected"])
        self.assertEqual(telem["accepted_prefix_len"], 0)
        self.assertEqual(telem["prefix_acceptance_rate"], 0.0)
        self.assertEqual(telem["mode"], "negation_flip_abort")

    def test_05_surgical_rope_evict_logic(self):
        """Verify surgical sequence eviction mathematics and RoPE shifting."""
        evicted = self.cssv.surgical_kv_evict_span(pos_start=10, pos_end=15)
        self.assertEqual(evicted, 5)

    def test_06_rollback_manager_cssv_integration(self):
        """RollbackManager coordinates CSSV salvage and delivers prefix to TTS callback."""
        mock_speculator = MagicMock()
        mock_speculator.get_triggering_partial.return_value = "I have a cough"
        mock_speculator.resolve.return_value = None
        mock_speculator.resolve_cssv.return_value = {
            "salvaged_prefix": True,
            "prefix_text": "I hear you have a cough.",
            "full_response": "I hear you have a cough. How long has it been going on?",
            "telemetry": {
                "accepted_prefix_len": 6,
                "prefix_acceptance_rate": 0.5,
                "cssv_verify_ms": 38.5,
            },
        }

        mock_tts = MagicMock()
        delivered_prefixes = []

        def _on_prefix(p):
            delivered_prefixes.append(p)

        decision = resolve_speculation(
            speculator=mock_speculator,
            tts_engine=mock_tts,
            final_transcript="I have a cough and high fever for 3 days",
            divergence_threshold=0.70,
            use_cssv=True,
            on_prefix_ready=_on_prefix,
        )

        self.assertTrue(decision.committed)
        self.assertTrue(decision.salvaged_prefix)
        self.assertEqual(decision.prefix_text, "I hear you have a cough.")
        self.assertIn("accepted_prefix_len", decision.cssv_telemetry)

    def test_07_slot_entropy_classifier(self):
        """ClinicalSlotEntropyClassifier distinguishes concise answers from rambling narrative."""
        classifier = ClinicalSlotEntropyClassifier()

        # Concise closed responses should resolve clinical slot (cut VAD silence)
        self.assertTrue(classifier.is_slot_resolved("yes"))
        self.assertTrue(classifier.is_slot_resolved("no"))
        self.assertTrue(classifier.is_slot_resolved("10 out of 10"))
        self.assertTrue(classifier.is_slot_resolved("in my left arm"))
        self.assertTrue(classifier.is_slot_resolved("since yesterday"))
        self.assertTrue(classifier.is_slot_resolved("that is all"))

        # Open-ended rambling responses should not prematurely resolve
        rambling = "Well you see doctor it all began when I was walking my dog in the park"
        self.assertFalse(classifier.is_slot_resolved(rambling))

    def test_08_acoustic_dyspnea_detector(self):
        """AcousticDyspneaDetector identifies respiratory gasps and wheezes."""
        detector = AcousticDyspneaDetector()

        # 1. Normal silence / background noise (low energy, RMS <= tau_noise)
        silence_pcm = np.zeros(16000, dtype=np.float32)
        e_silence = detector.compute_dyspnea_index(silence_pcm)
        self.assertEqual(e_silence, 0.0)

        # 2. Simulated breath gasp / stridor (energy in 500-2500Hz with RMS in [0.003, 0.040])
        t = np.linspace(0, 1, 16000, endpoint=False)
        gasp_signal = (
            0.015 * np.sin(2 * np.pi * 1200 * t)
            + 0.010 * np.sin(2 * np.pi * 1800 * t)
        ).astype(np.float32)

        e_gasp = detector.compute_dyspnea_index(gasp_signal)
        self.assertGreater(e_gasp, 0.4)

    def test_09_dyspnea_aware_endpointing_bounds(self):
        """Adaptive endpointing clamps VAD timeout strictly within [180ms, 900ms]."""
        endpointing = DyspneaAwareEndpointing(
            tau_base_ms=450,
            alpha_slot_ms=270,
            beta_dyspnea_ms=450,
        )

        # Slot filled concise answer (e.g. "yes"): should drop close to 180 ms
        dec_concise = endpointing.evaluate(
            partial_text="yes",
            pause_audio_chunk=np.zeros(3200, dtype=np.float32),
        )
        self.assertEqual(dec_concise.silence_timeout_ms, 180)
        self.assertTrue(dec_concise.slot_resolved)

        # Severe dyspnea gasp: should expand up to 900 ms to avoid premature patient interruption
        t = np.linspace(0, 0.5, 8000, endpoint=False)
        gasp_signal = (
            0.018 * np.sin(2 * np.pi * 1200 * t)
            + 0.012 * np.sin(2 * np.pi * 1800 * t)
        ).astype(np.float32)

        dec_dyspnea = endpointing.evaluate(
            partial_text="chest pain",
            pause_audio_chunk=gasp_signal,
        )
        self.assertGreater(dec_dyspnea.silence_timeout_ms, 450)
        self.assertLessEqual(dec_dyspnea.silence_timeout_ms, 900)
        self.assertGreater(dec_dyspnea.dyspnea_energy, 0.40)


if __name__ == "__main__":
    unittest.main()
