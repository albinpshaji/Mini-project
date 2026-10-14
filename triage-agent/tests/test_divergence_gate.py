"""
Unit Test & Diagnostic Benchmark for the Multi-Stage Clinical Invariant Gate.
Tests Innovation 1 from the OmniSpec-Triage Master Plan:
  - Stage 1: Hard Polarity / Negation Invariant Gate
  - Stage 2: Critical Anatomical & Emergency Entity Invariant Gate
  - Stage 3: Lexical Similarity Thresholding
"""

import sys
import os
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.speculative.divergence_detector import compute_divergence, DivergenceResult


class TestClinicalInvariantGate(unittest.TestCase):
    """Exhaustive test suite for the Multi-Stage Clinical Invariant Gate."""

    def test_exact_and_near_matches_commit(self):
        """Standard matching partials should pass and commit without divergence."""
        cases = [
            ("I have a fever", "I have a fever", 0.70),
            ("mild cough and runny nose", "mild cough and runny nose", 0.70),
            ("my head hurts since morning", "my head hurts since morning", 0.70),
        ]
        for partial, final, threshold in cases:
            res = compute_divergence(partial, final, threshold=threshold)
            self.assertFalse(res.diverged, f"Expected commit for '{partial}' vs '{final}'")
            self.assertEqual(res.reason, "passed")
            self.assertFalse(res.polarity_shifted)
            self.assertFalse(res.entity_shifted)
            self.assertGreaterEqual(res.similarity, threshold)

    def test_stage1_hard_negation_invariance(self):
        """
        Adversarial test: Any negation token insertion or deletion must trigger
        immediate divergence (diverged=True) and reason='polarity_mismatch'.
        """
        adversarial_pairs = [
            # Partial had no negation, final added negation
            ("I have pain in my chest", "I have no pain in my chest"),
            ("I have difficulty breathing", "I have no difficulty breathing"),
            ("I feel dizzy and weak", "I do not feel dizzy and weak"),
            ("cough with blood", "cough without blood"),
            ("the pain has continued", "the pain has stopped"),
            ("symptoms are severe", "symptoms have resolved"),
            ("shortness of breath present", "denies shortness of breath"),
            # Partial had negation, final removed negation
            ("I have no fever", "I have high fever"),
            ("I do not have chest pain", "I have severe chest pain"),
            ("without any vomiting", "with severe vomiting"),
        ]

        for partial, final in adversarial_pairs:
            res = compute_divergence(partial, final, threshold=0.70)
            self.assertTrue(
                res.diverged,
                f"SAFETY VIOLATION: Failed to rollback on negation shift '{partial}' -> '{final}'!"
            )
            self.assertEqual(
                res.reason,
                "polarity_mismatch",
                f"Expected reason='polarity_mismatch', got '{res.reason}' for '{partial}' -> '{final}'"
            )
            self.assertTrue(res.polarity_shifted)

    def test_stage2_critical_entity_mismatch(self):
        """
        Adversarial test: If a critical anatomical target or emergency condition shifts,
        the gate must force immediate divergence (diverged=True) and reason='critical_entity_mismatch'.
        """
        critical_pairs = [
            ("I have severe pain in my wrist", "I have severe pain in my chest"),
            ("I am coughing and sneezing", "I am coughing up dark blood"),
            ("pain radiating to my neck", "pain with difficulty breathing"),
            ("I felt dizzy for an hour", "I had a sudden stroke and facial droop"),
            ("a mild twitch in my eyelid", "a full body seizure"),
            ("swelling in my finger", "swelling in my throat and anaphylaxis"),
        ]

        for partial, final in critical_pairs:
            res = compute_divergence(partial, final, threshold=0.70)
            self.assertTrue(
                res.diverged,
                f"SAFETY VIOLATION: Failed to rollback on critical entity shift '{partial}' -> '{final}'!"
            )
            self.assertEqual(
                res.reason,
                "critical_entity_mismatch",
                f"Expected reason='critical_entity_mismatch', got '{res.reason}'"
            )
            self.assertTrue(res.entity_shifted)

    def test_stage3_low_similarity_divergence(self):
        """Completely different statements should fail Stage 3 lexical similarity."""
        different_pairs = [
            ("I came here by bus today", "My back has been hurting all week"),
            ("Can you hear me clearly", "I need to see a cardiologist"),
        ]
        for partial, final in different_pairs:
            res = compute_divergence(partial, final, threshold=0.70)
            self.assertTrue(res.diverged)
            self.assertEqual(res.reason, "low_similarity")

    def test_sub_millisecond_latency_overhead(self):
        """The 3-stage invariant gate must execute in <0.1ms (100 microseconds) on CPU."""
        partial = "I have a sharp pain in my lower right abdomen"
        final = "I have a sharp pain in my lower right abdomen since 2 hours"

        # Warm-up
        for _ in range(50):
            compute_divergence(partial, final)

        n_iters = 1000
        start_t = time.perf_counter()
        for _ in range(n_iters):
            compute_divergence(partial, final)
        elapsed_s = time.perf_counter() - start_t
        avg_latency_ms = (elapsed_s / n_iters) * 1000.0

        print(f"\n⚡ Invariant Gate Average Execution Latency: {avg_latency_ms:.4f} ms per check ({n_iters} iterations)")
        self.assertLess(avg_latency_ms, 0.5, "Execution latency must remain sub-millisecond (<0.5ms)")


def run_divergence_diagnostic():
    print("=" * 65)
    print(" 🛡️  MULTI-STAGE CLINICAL INVARIANT GATE DIAGNOSTIC")
    print("=" * 65)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestClinicalInvariantGate)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_divergence_diagnostic()
    sys.exit(0 if success else 1)
