"""
Unit Tests for Phase 5: SnapKV Attention Anchor Pruning & Multi-Turn Cache Compression.

Validates:
1. Clinical Anchor Extraction (symptoms, vitals, onsets, anatomy, allergies, meds).
2. DropKV Decoupled Residual Conversational Filler Pruning.
3. Observation Window Preservation (DuoAttention Streaming Heads).
4. Turn 1 Chief Complaint Pinning.
5. Needle-in-a-Haystack Multi-Turn Allergy Retention across 12 turns.
6. Multi-Turn KV Footprint Bounding (<512 tokens, <160 MB active cache).
7. SLMEngine Integration & Telemetry Reporting.
"""

import unittest
from src.slm.snapkv_engine import ClinicalAnchorExtractor, SnapKVCompressor
from src.slm.slm_engine import SLMEngine


class TestSnapKVPruning(unittest.TestCase):

    def setUp(self):
        self.compressor = SnapKVCompressor(
            observation_window_tokens=48,
            max_compressed_history_tokens=512,
            trigger_turn_threshold=4,
        )

    def test_clinical_anchor_extraction(self):
        """Verifies extraction of symptoms, vitals, anatomical sites, and drug allergies."""
        extractor = ClinicalAnchorExtractor()

        text = (
            "I have severe crushing chest pain radiating to my left arm and jaw. "
            "It started suddenly 45 minutes ago. My blood pressure was 160/100 and pain is 9/10. "
            "I am taking warfarin and I am severely allergic to penicillin."
        )
        anchors = extractor.extract_anchors(text)

        self.assertIn("crushing", anchors)
        self.assertIn("chest", anchors)
        self.assertIn("left arm", anchors)
        self.assertIn("jaw", anchors)
        self.assertIn("suddenly", anchors)
        self.assertIn("45 minutes", anchors)
        self.assertIn("160/100", anchors)
        self.assertIn("9/10", anchors)
        self.assertIn("warfarin", anchors)
        self.assertIn("penicillin", anchors)

    def test_conversational_filler_pruning(self):
        """Verifies DropKV residual pruning of low-information conversational chaff."""
        extractor = ClinicalAnchorExtractor()

        raw_text = "Well, um, you know, like I said earlier, doctor, my chest pain feels worse now, thank you."
        sanitized = extractor.sanitize_filler(raw_text)

        self.assertNotIn("well", sanitized.lower().split())
        self.assertNotIn("you know", sanitized.lower())
        self.assertNotIn("like i said", sanitized.lower())
        self.assertNotIn("thank you", sanitized.lower())
        self.assertIn("chest pain", sanitized.lower())

    def test_observation_window_preservation(self):
        """Verifies that the most recent observation window (last 2 messages) is preserved verbatim."""
        history = [
            {"role": "user", "text": "I have chest pain since this morning."},
            {"role": "assistant", "text": "Can you describe the pain?"},
            {"role": "user", "text": "It feels heavy and tight."},
            {"role": "assistant", "text": "Does it radiate anywhere?"},
            {"role": "user", "text": "Yes, it goes to my left shoulder."},
            {"role": "assistant", "text": "Are you experiencing any shortness of breath?"},
            {"role": "user", "text": "No, just the left shoulder pain right now."},
        ]

        compressed, telemetry = self.compressor.compress_history(history)

        self.assertTrue(telemetry.get("compressed"))
        # The last 2 messages must match the original last 2 messages verbatim
        self.assertEqual(compressed[-1]["text"], history[-1]["text"])
        self.assertEqual(compressed[-2]["text"], history[-2]["text"])

    def test_turn_1_chief_complaint_pinning(self):
        """Verifies that Turn 1 is permanently pinned and never evicted under multi-turn compression."""
        turn_1_complaint = "I have severe substernal chest pressure that started 1 hour ago."
        history = [
            {"role": "user", "text": turn_1_complaint},
            {"role": "assistant", "text": "How severe is the pressure on a scale of 1 to 10?"},
            {"role": "user", "text": "It is about an 8/10."},
            {"role": "assistant", "text": "Are you taking any cardiac medications?"},
            {"role": "user", "text": "I take nitroglycerin when needed."},
            {"role": "assistant", "text": "Did you take a dose today?"},
            {"role": "user", "text": "Yes, one tablet 20 minutes ago."},
        ]

        compressed, _ = self.compressor.compress_history(history)

        # Turn 1 must be present at index 0 verbatim
        self.assertEqual(compressed[0]["text"], turn_1_complaint)
        self.assertEqual(compressed[0]["role"], "user")

    def test_needle_in_a_haystack_multi_turn_retention(self):
        """
        Needle-in-a-Haystack Clinical Stress Test:
        Turn 1: Discloses life-threatening allergy to Penicillin and Warfarin therapy.
        Turns 2 to 10: 10 intervening turns of generic symptom and hospital intake discussion.
        Turn 11-12: Recent observation window.
        Verifies that Turn 12 compressed context strictly retains the Penicillin allergy and Warfarin anchor.
        """
        needle_turn = (
            "I have severe migraine headaches, and I have an anaphylactic shock allergy to Penicillin "
            "and Cephalosporins, and I take Warfarin every night."
        )

        history = [
            {"role": "user", "text": needle_turn},
            {"role": "assistant", "text": "Thank you for that information. How long have you had the headache?"},
            {"role": "user", "text": "Well, um, like I said, it started about 2 days ago."},
            {"role": "assistant", "text": "Is the light bothering your eyes?"},
            {"role": "user", "text": "Yes, photophobia is definitely present."},
            {"role": "assistant", "text": "Have you experienced any vomiting or neck stiffness?"},
            {"role": "user", "text": "No vomiting, but my neck feels a bit stiff on the left."},
            {"role": "assistant", "text": "Have you taken any pain relievers today?"},
            {"role": "user", "text": "I took some tylenol but it did not help much."},
            {"role": "assistant", "text": "Which hospital department are you trying to reach?"},
            {"role": "user", "text": "I was looking for the Neurology clinic or emergency room."},
            {"role": "assistant", "text": "Can you tell me if you have any vision changes right now?"},
            {"role": "user", "text": "My vision is slightly blurry in the left eye."},
        ]

        compressed, telemetry = self.compressor.compress_history(history)

        # Re-aggregate all text in compressed context
        compressed_text = " ".join(m["text"] for m in compressed).lower()

        # Needle verification
        self.assertIn("penicillin", compressed_text, "Critical allergy to Penicillin was dropped!")
        self.assertIn("warfarin", compressed_text, "Critical anticoagulant Warfarin was dropped!")
        self.assertIn("cephalosporin", compressed_text, "Critical allergy to Cephalosporin was dropped!")

        # Verify anchors in telemetry
        active_anchors = telemetry.get("active_anchors", [])
        self.assertIn("penicillin", active_anchors)
        self.assertIn("warfarin", active_anchors)

    def test_memory_footprint_bounding(self):
        """Verifies that across 14 dialogue turns, compressed context remains strictly bounded."""
        history = []
        for i in range(14):
            history.append({"role": "user", "text": f"Patient turn {i+1}: I am experiencing pain level {i%10}/10 in my chest."})
            history.append({"role": "assistant", "text": f"Nurse turn {i+1}: Understood, please tell me more about symptom {i+1}."})

        self.assertEqual(len(history), 28)  # 28 raw messages (~2,000+ tokens)

        compressed, telemetry = self.compressor.compress_history(history)

        self.assertLess(len(compressed), len(history))
        self.assertLessEqual(telemetry.get("estimated_comp_tokens", 9999), 512)
        self.assertLess(telemetry.get("active_kv_footprint_mb", 999.0), 160.0)
        self.assertGreater(telemetry.get("tokens_pruned", 0), 100)

    def test_slm_engine_snapkv_integration(self):
        """Verifies live SLMEngine integration with SnapKV enabled."""
        mock_cfg = {
            "model_name": "qwen3.5:4b",
            "temperature": 0.3,
            "num_ctx": 3072,
            "max_history_messages": 24,
            "use_lookahead": False,
            "use_snapkv": True,
            "snapkv_observation_window": 48,
            "snapkv_max_tokens": 512,
            "snapkv_trigger_turn_threshold": 4,
            "hospital_routing": False,
        }

        try:
            engine = SLMEngine(mock_cfg)
            self.assertTrue(engine.use_snapkv)
            self.assertIsNotNone(engine.snapkv_compressor)

            # Test _build_messages with 8 turns
            history = [
                {"role": "user", "text": "Turn 1: Severe chest pressure and shortness of breath."},
                {"role": "assistant", "text": "Understood. How long has this lasted?"},
                {"role": "user", "text": "Turn 2: Started 30 minutes ago, pain is 8/10."},
                {"role": "assistant", "text": "Does it radiate?"},
                {"role": "user", "text": "Turn 3: Radiating to left arm."},
                {"role": "assistant", "text": "Are you taking any medications?"},
                {"role": "user", "text": "Turn 4: Nitroglycerin, and I have a penicillin allergy."},
                {"role": "assistant", "text": "Please sit down and rest."},
                {"role": "user", "text": "Turn 5: I am sitting down now, what should I do next?"},
            ]

            messages = engine._build_messages("What should I do next?", history)
            self.assertGreater(len(messages), 2)
            self.assertEqual(messages[0]["role"], "system")

            telemetry = engine.get_snapkv_telemetry()
            self.assertIn("total_compressions", telemetry)
            self.assertIn("last_turn_compression", telemetry)
        except RuntimeError:
            # If Ollama service is not reachable in CI/unit test environment, skip gracefully
            pass


if __name__ == "__main__":
    unittest.main()
