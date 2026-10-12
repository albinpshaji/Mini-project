"""
Unit tests for HospitalRouter (sub-millisecond edge clinical retrieval).
"""

import time
import os
import unittest
from src.safety.hospital_router import HospitalRouter


class TestHospitalRouter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.router = HospitalRouter()

    def test_latency_sub_millisecond(self):
        """Verify query execution time is well under 2.0 ms (sub-millisecond target)."""
        query = "I have had a tight pressure in my chest and my left arm hurts"
        # Warmup
        _ = self.router.match(query)

        iterations = 50
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = self.router.match(query)
        elapsed_total = time.perf_counter() - t0
        avg_latency_ms = (elapsed_total / iterations) * 1000

        print(f"\n⚡ Router Average Query Latency: {avg_latency_ms:.3f} ms")
        self.assertLess(avg_latency_ms, 2.0, "Router latency must be < 2.0 ms on CPU")

    def test_cardiac_retrieval(self):
        """Verify chest pain/cardiac queries map to Room 120."""
        queries = [
            "I have severe chest tightness and palpitations",
            "My heart feels like it is fluttering fast",
            "Crushing chest pressure since an hour ago",
        ]
        for q in queries:
            matches = self.router.match(q, top_k=1)
            self.assertTrue(len(matches) > 0, f"Expected match for query: {q}")
            self.assertEqual(matches[0].room_number, "120", f"Expected Room 120 for: {q}")
            self.assertIn("Cardiology", matches[0].department)

    def test_fever_retrieval(self):
        """Verify high fever/infection queries map to Room 20."""
        queries = [
            "I am burning up with a 102 fever and chills",
            "Shivering, high temperature and flu symptoms for two days",
            "I feel very feverish and sweating",
        ]
        for q in queries:
            matches = self.router.match(q, top_k=1)
            self.assertTrue(len(matches) > 0, f"Expected match for query: {q}")
            self.assertEqual(matches[0].room_number, "20", f"Expected Room 20 for: {q}")
            self.assertIn("Fever", matches[0].department)

    def test_orthopedics_retrieval(self):
        """Verify broken bones/sprains map to Room 105."""
        queries = [
            "I fell down the stairs and think I have a broken bone in my wrist",
            "My ankle is severely twisted and swollen",
        ]
        for q in queries:
            matches = self.router.match(q, top_k=1)
            self.assertTrue(len(matches) > 0, f"Expected match for query: {q}")
            self.assertEqual(matches[0].room_number, "105", f"Expected Room 105 for: {q}")

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

    def test_compact_token_hint(self):
        """Verify generated prompt hint is ultra-compact (< 60 words / ~45 tokens)."""
        matches = self.router.match("I have chest pain")
        self.assertTrue(len(matches) > 0)
        hint = self.router.format_routing_hint(matches)
        word_count = len(hint.split())
        print(f"\n📝 Injected Hint Word Count: {word_count} words")
        self.assertLess(word_count, 65, "Prompt hint must stay under 65 words to preserve VRAM/context")

    def test_anti_hallucination_sanitization(self):
        """Verify non-existent rooms are intercepted and sanitized."""
        self.assertTrue(self.router.is_valid_room("120"))
        self.assertTrue(self.router.is_valid_room("20"))
        self.assertFalse(self.router.is_valid_room("999"))
        self.assertFalse(self.router.is_valid_room("404"))

        fake_text = "Please walk over to Room 999 on the third floor."
        sanitized = self.router.sanitize_unauthorized_rooms(fake_text)
        self.assertNotIn("Room 999", sanitized)
        self.assertIn("the front reception desk", sanitized)


if __name__ == "__main__":
    unittest.main()
