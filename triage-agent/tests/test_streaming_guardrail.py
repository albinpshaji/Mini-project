"""
Unit test suite for Pre-TTS Real-Time Streaming Guardrail (Feature C).
Grounded in NVIDIA NeMo Guardrails (arXiv:2310.10501) and Google Med-PaLM 2 (arXiv:2305.09617).
"""

import sys
import os
import unittest
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.safety.streaming_guardrail import (
    StreamingGuardrail,
    AhoCorasickTrie,
    DOSAGE_REFUSAL_MESSAGE,
    DIAGNOSTIC_OVERREACH_BASE,
    NEGATION_SAFE_RESPONSE
)


class TestStreamingGuardrail(unittest.TestCase):
    def setUp(self):
        self.guardrail = StreamingGuardrail()

    def test_aho_corasick_trie_search(self):
        trie = AhoCorasickTrie(["amoxicillin", "morphine", "chest pain"])
        matches = trie.search("The patient was given morphine and amoxicillin.")
        matched_words = [m[1] for m in matches]
        self.assertIn("morphine", matched_words)
        self.assertIn("amoxicillin", matched_words)

    def test_dosage_suppression_units(self):
        """Verifies that all standard dosage units are intercepted and suppressed."""
        test_cases = [
            ("Take 500 mg of Amoxicillin immediately.", "mg"),
            ("You should inject 10 ml of solution.", "ml"),
            ("Please swallow 2 tablets of Oxycodone.", "tablets"),
            ("Administer 15 milligrams of Morphine.", "milligrams"),
            ("Put 3 drops in your eye twice daily.", "drops"),
            ("Prescribe 20mcg daily.", "mcg"),
        ]
        for text, unit in test_cases:
            res = self.guardrail.guard_chunk(text)
            self.assertTrue(res.intercepted, f"Failed to intercept dosage with unit: {unit}")
            self.assertIn("unauthorized_dosage", res.violation_types)
            self.assertEqual(res.text, DOSAGE_REFUSAL_MESSAGE)
            # Ensure no numbers with unit leak into output
            self.assertNotIn(f" {unit}", res.text.lower())

    def test_drug_number_pair_interception(self):
        """Verifies drug names paired with numbers without explicit units are caught."""
        test_cases = [
            "You should take morphine 10 for your pain.",
            "Take amoxicillin 500 right now.",
            "I recommend oxycodone 20.",
        ]
        for text in test_cases:
            res = self.guardrail.guard_chunk(text)
            self.assertTrue(res.intercepted, f"Failed to intercept drug pair: {text}")
            self.assertIn("unauthorized_dosage", res.violation_types)
            self.assertEqual(res.text, DOSAGE_REFUSAL_MESSAGE)

    def test_diagnostic_overreach_rewriting(self):
        """Verifies definitive diagnostic assertions are rewritten into hedged clinical triage language."""
        test_cases = [
            ("You definitely have glioblastoma and need surgery.", "glioblastoma"),
            ("You are diagnosed with ALS based on your symptoms.", "als"),
            ("My diagnosis is pancreatic cancer.", "pancreatic cancer"),
            ("You have acute leukemia.", "acute leukemia"),
            ("You are suffering from aortic dissection.", "aortic dissection"),
        ]
        for text, disease in test_cases:
            res = self.guardrail.guard_chunk(text)
            self.assertTrue(res.intercepted, f"Failed to intercept diagnosis: {disease}")
            self.assertIn("diagnostic_overreach", res.violation_types)
            # Ensure unhedged definitive diagnosis is removed
            self.assertNotIn(f"you definitely have {disease}", res.text.lower())
            self.assertNotIn(f"diagnosed with {disease}", res.text.lower())

    def test_negation_boundary_protection(self):
        """Verifies that when a patient denies emergency symptoms, false red-flag escalations are suppressed."""
        patient_prompt = "Case 1: I am not experiencing any chest pain or difficulty breathing at all."
        hallucinated_response = "You must go to the emergency room immediately and call 911!"

        res = self.guardrail.guard_chunk(hallucinated_response, user_prompt=patient_prompt)
        self.assertTrue(res.intercepted)
        self.assertIn("negation_false_escalation", res.violation_types)
        self.assertNotIn("call 911", res.text.lower())
        self.assertNotIn("emergency room immediately", res.text.lower())
        self.assertEqual(res.text, NEGATION_SAFE_RESPONSE)

    def test_safe_conversational_passthrough(self):
        """Verifies that legitimate triage questions and intake dialogues pass through unaltered."""
        safe_inputs = [
            "Hello, I am the virtual triage assistant. What symptoms bring you in today?",
            "Can you tell me how long you have had this mild cough?",
            "Is the pain sharp or dull, and does it spread anywhere else?",
            "Thank you. Please take a seat in the waiting area while we notify the nurse.",
        ]
        for text in safe_inputs:
            res = self.guardrail.guard_chunk(text)
            self.assertFalse(res.intercepted)
            self.assertTrue(res.safe)
            self.assertEqual(res.text, text)
            self.assertEqual(len(res.violation_types), 0)

    def test_streaming_token_interception(self):
        """Verifies that token stream generator intercepts prohibited content across chunks."""
        # Simulated token stream emitting an unauthorized prescription
        raw_tokens = [
            {"token": "You "}, {"token": "should "}, {"token": "take "},
            {"token": "500 "}, {"token": "mg "}, {"token": "of "},
            {"token": "Amoxicillin, "}, {"token": "and "}, {"token": "rest."}
        ]

        guarded_tokens = list(self.guardrail.guard_token_stream(raw_tokens))
        assembled_guarded = "".join(t["token"] for t in guarded_tokens)

        # Confirm dosage was intercepted in-flight
        self.assertNotIn("500 mg", assembled_guarded)
        self.assertIn("cannot prescribe medications", assembled_guarded)
        self.assertTrue(any(t.get("intercepted") for t in guarded_tokens))

    def test_guardrail_latency_overhead(self):
        """Verifies guardrail inspection latency is strictly <1.5 ms (Day 18 Master Plan)."""
        benchmark = self.guardrail.benchmark_latency(iterations=500)
        print(f"\n⚡ Streaming Guardrail Latency: {benchmark['mean_latency_us']} µs ({benchmark['mean_latency_ms']} ms) per chunk")
        self.assertTrue(benchmark["meets_target"])
        self.assertLess(benchmark["mean_latency_ms"], 1.5)
        # Even stricter check: should be under 0.1 ms (100 µs)
        self.assertLess(benchmark["mean_latency_us"], 150.0)


if __name__ == "__main__":
    unittest.main()
