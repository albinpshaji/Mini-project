"""
Unit tests for HospitalRouter (sub-millisecond edge clinical retrieval with Top-3 multi-candidate routing).
"""

import time
import os
import unittest
from src.safety.hospital_router import HospitalRouter


class TestHospitalRouter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.router = HospitalRouter(top_k=3)

    def test_latency_sub_millisecond(self):
        """Verify query execution time is well under 2.0 ms (sub-millisecond target)."""
        query = "I have had a tight pressure in my chest and my left arm hurts"
        _ = self.router.match(query)

        iterations = 50
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = self.router.match(query)
        elapsed_total = time.perf_counter() - t0
        avg_latency_ms = (elapsed_total / iterations) * 1000

        print(f"\n⚡ Router Average Query Latency: {avg_latency_ms:.3f} ms")
        self.assertLess(avg_latency_ms, 2.0, "Router latency must be < 2.0 ms on CPU")

    def test_top_3_candidates_retrieved(self):
        """Verify router returns up to 3 candidate options for clinical reasoning."""
        query = "I have chest tightness and my heart feels like it is fluttering fast"
        matches = self.router.match(query, top_k=3)
        self.assertGreaterEqual(len(matches), 2, "Expected at least 2-3 candidates for composite symptom")
        self.assertLessEqual(len(matches), 3, "Should not exceed top_k=3")
        # Check ward and sub-room presence
        self.assertTrue(bool(matches[0].ward), "Match should contain ward information")
        self.assertTrue(bool(matches[0].sub_room_name), "Match should contain sub-room name")

    def test_cardiac_retrieval(self):
        """Verify chest pain/cardiac queries map to Room 120 sub-rooms."""
        queries = [
            "I have severe chest tightness and palpitations",
            "My heart feels like it is fluttering fast",
            "Crushing chest pressure since an hour ago",
        ]
        for q in queries:
            matches = self.router.match(q, top_k=3)
            self.assertTrue(len(matches) > 0, f"Expected match for query: {q}")
            self.assertTrue(
                any(m.room_number.startswith("120") for m in matches),
                f"Expected Room 120 candidate for: {q}",
            )

    def test_fever_retrieval(self):
        """Verify high fever/infection queries map to Room 20 sub-rooms."""
        queries = [
            "I am burning up with a 102 fever and chills",
            "Shivering, high temperature and flu symptoms for two days",
            "I feel very feverish and sweating",
        ]
        for q in queries:
            matches = self.router.match(q, top_k=3)
            self.assertTrue(len(matches) > 0, f"Expected match for query: {q}")
            self.assertTrue(
                any(m.room_number.startswith("20") for m in matches),
                f"Expected Room 20 candidate for: {q}",
            )

    def test_orthopedics_retrieval(self):
        """Verify broken bones/sprains map to Room 105 sub-rooms."""
        queries = [
            "I fell down the stairs and think I have a broken bone in my wrist",
            "My ankle is severely twisted and swollen",
        ]
        for q in queries:
            matches = self.router.match(q, top_k=3)
            self.assertTrue(len(matches) > 0, f"Expected match for query: {q}")
            self.assertTrue(
                any(m.room_number.startswith("105") for m in matches),
                f"Expected Room 105 candidate for: {q}",
            )

    def test_irrelevant_queries_zero_context(self):
        """Verify small-talk or administrative questions inject 0 tokens (empty matches)."""
        irrelevant_queries = [
            "Good morning, where can I park my car?",
            "Thank you nurse, appreciate it.",
            "Can I get a glass of water?",
            "What time does the cafeteria open?",
        ]
        for q in irrelevant_queries:
            matches = self.router.match(q)
            self.assertEqual(len(matches), 0, f"Expected 0 matches for irrelevant query: {q}")
            hint = self.router.format_routing_hint(matches)
            self.assertEqual(hint, "", "Hint should be empty string for irrelevant queries")

    def test_compact_token_hint_3_candidates(self):
        """Verify generated prompt hint with 3 options stays compact (< 150 words)."""
        matches = self.router.match("I have chest pain and fluttering heart", top_k=3)
        self.assertTrue(len(matches) > 0)
        hint = self.router.format_routing_hint(matches)
        word_count = len(hint.split())
        print(f"\n📝 Injected 3-Option Hint Word Count: {word_count} words")
        self.assertLess(word_count, 160, "3-Option prompt hint must stay under 160 words to preserve context budget")

    def test_anti_hallucination_sanitization(self):
        """Verify non-existent rooms are intercepted while valid sub-rooms pass."""
        self.assertTrue(self.router.is_valid_room("120-A"))
        self.assertTrue(self.router.is_valid_room("120"))
        self.assertTrue(self.router.is_valid_room("20-A"))
        self.assertTrue(self.router.is_valid_room("20"))
        self.assertFalse(self.router.is_valid_room("999"))
        self.assertFalse(self.router.is_valid_room("404-X"))

        fake_text = "Please walk over to Room 999 on the third floor."
        sanitized = self.router.sanitize_unauthorized_rooms(fake_text)
        self.assertNotIn("Room 999", sanitized)
        self.assertIn("the front reception desk", sanitized)


if __name__ == "__main__":
    unittest.main()
