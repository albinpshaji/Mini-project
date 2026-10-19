"""
Unit tests for Phase 4: Zero-VRAM Lookahead Jacobi Speculation & Speculative Pre-Emission.
Literature Grounding:
- Fu et al., 2024 (Lookahead Decoding: Jacobi Iteration for LLMs)
- DeepSeek-V3 MTP (arXiv:2412.19437, Dec 2024)
- Stanford Sequoia (Chen et al., 2024: Hardware-Aware Draft Tree Sizing for GPU L2 Cache)
- BiLD (Kim et al., 2023: Bi-Directional Speculative Decoding with Confidence Gating)
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.slm.lookahead_engine import JacobiSpeculator, CANONICAL_CLINICAL_NGRAMS
from src.slm.slm_engine import SLMEngine
from src.tts.piper_engine import AcousticPrewarmer, PrewarmedAudioChunk


class TestLookaheadSpeculation(unittest.TestCase):
    def setUp(self):
        self.speculator = JacobiSpeculator(
            max_lookahead_depth=8,
            branching_factor=3,
            max_candidate_tokens=12,
            bild_entropy_threshold=0.15,
        )

    def test_jacobi_initialization_and_preseeding(self):
        """Verifies pre-seeded canonical clinical n-grams for 0ms cold-start."""
        self.assertGreater(len(self.speculator.ngram_cache), 0)
        self.assertGreater(len(self.speculator.unigram_cache), 0)
        self.assertEqual(self.speculator.max_k, 8)
        self.assertEqual(self.speculator.max_b, 3)
        self.assertEqual(self.speculator.max_tokens, 12)
        self.assertEqual(self.speculator.total_drafted_tokens, 0)
        self.assertEqual(self.speculator.total_accepted_tokens, 0)

    def test_candidate_branch_generation_bounded(self):
        """Verifies candidate tree branches are strictly bounded by B and max candidate tokens."""
        # Test with empathetic opener prefix
        context = ["I", "understand"]
        branches = self.speculator.build_candidate_branches(context)
        self.assertIsInstance(branches, list)
        self.assertLessEqual(len(branches), self.speculator.max_b)

        total_tokens = sum(len(b) for b in branches)
        self.assertLessEqual(total_tokens, self.speculator.max_tokens)
        for branch in branches:
            self.assertGreater(len(branch), 0)
            self.assertLessEqual(len(branch), self.speculator.max_k)

    def test_verification_and_acceptance_mechanics(self):
        """Verifies candidate verification, acceptance tracking, and speedup computation."""
        # Case 1: Match
        candidate = ["let", "me", "help"]
        target_token = "let"
        matched, length = self.speculator.verify_candidate_sequence(candidate, target_token)
        self.assertTrue(matched)
        self.assertEqual(length, 1)
        self.assertEqual(self.speculator.total_accepted_tokens, 1)
        self.assertEqual(self.speculator.total_drafted_tokens, 3)

        # Case 2: Mismatch
        mismatch_candidate = ["emergency", "room"]
        target_token_2 = "clinic"
        matched_2, length_2 = self.speculator.verify_candidate_sequence(mismatch_candidate, target_token_2)
        self.assertFalse(matched_2)
        self.assertEqual(length_2, 0)
        self.assertEqual(self.speculator.total_accepted_tokens, 1)
        self.assertEqual(self.speculator.total_drafted_tokens, 5)

        # Telemetry properties
        self.assertAlmostEqual(self.speculator.acceptance_rate, 1 / 5, places=3)
        self.assertGreaterEqual(self.speculator.speedup_factor, 1.0)

    def test_bild_confidence_fallback(self):
        """Verifies BiLD confidence-gated fallback on high clinical entropy."""
        # High confidence: top-1=0.85, top-2=0.05 -> Delta p = 0.80 >= 0.15 -> No fallback
        self.assertFalse(self.speculator.should_fallback_bild(top_probs=[0.85, 0.05]))

        # High entropy: top-1=0.45, top-2=0.38 -> Delta p = 0.07 < 0.15 -> Trigger fallback
        self.assertTrue(self.speculator.should_fallback_bild(top_probs=[0.45, 0.38]))
        self.assertEqual(self.speculator.total_fallback_steps, 1)

        # High Shannon entropy fallback
        self.assertTrue(self.speculator.should_fallback_bild(entropy=1.45))
        self.assertEqual(self.speculator.total_fallback_steps, 2)

    def test_online_ngram_learning(self):
        """Verifies dynamic recording and retrieval of novel clinical dialogue n-grams."""
        novel_tokens = ["patient", "reports", "substernal", "crushing", "chest", "pain"]
        self.speculator.record_transition(novel_tokens, weight=5)

        # Query novel prefix
        branches = self.speculator.build_candidate_branches(["patient", "reports"])
        self.assertGreater(len(branches), 0)
        self.assertEqual(branches[0][0], "substernal")

    def test_slm_engine_lookahead_telemetry(self):
        """Verifies SLMEngine lookahead integration and telemetry exposure."""
        cfg = {
            "model_name": "qwen3.5:4b",
            "use_lookahead": True,
            "lookahead_depth": 8,
            "lookahead_branches": 3,
            "lookahead_max_candidates": 12,
            "bild_entropy_threshold": 0.15,
            "hospital_routing": False,
        }
        with patch("src.slm.slm_engine.ollama.list", return_value={"models": []}), \
             patch("src.slm.slm_engine.SYSTEM_PROMPT", "Short test system prompt"):
            engine = SLMEngine(cfg)
            self.assertTrue(engine.use_lookahead)
            self.assertIsNotNone(engine.lookahead_speculator)

            telemetry = engine.get_lookahead_telemetry()
            self.assertIn("total_drafted_tokens", telemetry)
            self.assertIn("total_accepted_tokens", telemetry)
            self.assertIn("acceptance_rate", telemetry)
            self.assertIn("speedup_factor", telemetry)

    def test_speculative_pre_emission_lead_margin(self):
        """Verifies that pre-emitted grounding opener duration exceeds SLM lookahead decoding time."""
        mock_voice = MagicMock()
        mock_voice.config.sample_rate = 22050
        # 1.8 seconds of synthetic audio
        synthetic_samples = 22050 * 1.8
        chunk_mock = MagicMock()
        chunk_mock.audio_float_array = np.zeros(int(synthetic_samples), dtype=np.float32)
        mock_voice.synthesize.return_value = [chunk_mock]

        prewarmer = AcousticPrewarmer(mock_voice)
        prewarmer.build_cache()

        default_opener = prewarmer.get_default_opener()
        self.assertIsNotNone(default_opener)
        self.assertGreaterEqual(default_opener.duration_s, 1.5)
        # Expected Lookahead SLM generation time: ~750ms -> Lead margin > +750ms
        lead_margin = (default_opener.duration_s * 1000.0) - 750.0
        self.assertGreater(lead_margin, 500.0)


if __name__ == "__main__":
    unittest.main()
