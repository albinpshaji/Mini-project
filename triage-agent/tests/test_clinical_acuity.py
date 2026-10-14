"""
Unit Test & Benchmark for the Deterministic KTAS / ESI 1-5 Acuity Engine.
Grounding:
  - Canadian / Korean Triage and Acuity Scale (KTAS 1-5)
  - Emergency Severity Index (ESI 1-5)
  - Paper 19: Menon & Fouad (ACM 2024)
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.safety.clinical_acuity import calculate_ktas_acuity, KTASAssessment


class TestClinicalAcuityEngine(unittest.TestCase):

    def test_level_1_resuscitation(self):
        cases = [
            "Patient in cardiac arrest and unresponsive",
            "My throat is closing and I can't breathe from severe anaphylaxis",
            "He passed out and won't wake up",
        ]
        for c in cases:
            assessment = calculate_ktas_acuity(c)
            self.assertEqual(assessment.acuity_level, 1, f"Expected Level 1 for '{c}'")
            self.assertEqual(assessment.tier_name, "Resuscitation")
            self.assertEqual(assessment.target_time_min, 0)
            self.assertTrue(assessment.escalate_to_staff)

    def test_level_2_emergent(self):
        cases = [
            "Severe crushing chest pain radiating down left arm",
            "Heart racing at 140 beats per minute while sitting down quietly",
            "Sudden facial droop and slurred speech",
            "Coughing up dark blood and dizzy",
        ]
        for c in cases:
            assessment = calculate_ktas_acuity(c)
            self.assertEqual(assessment.acuity_level, 2, f"Expected Level 2 for '{c}'")
            self.assertEqual(assessment.tier_name, "Emergent")
            self.assertEqual(assessment.target_time_min, 15)
            self.assertTrue(assessment.escalate_to_staff)

    def test_level_3_urgent(self):
        cases = [
            "Asthma flare up with moderate wheezing",
            "Severe abdominal pain and cramps for 2 days",
            "Suspected broken bone in my forearm after a bicycle fall",
        ]
        for c in cases:
            assessment = calculate_ktas_acuity(c)
            self.assertEqual(assessment.acuity_level, 3, f"Expected Level 3 for '{c}'")
            self.assertEqual(assessment.tier_name, "Urgent")
            self.assertEqual(assessment.target_time_min, 30)

    def test_level_4_less_urgent(self):
        cases = [
            "Twisted my ankle playing basketball, mild swelling",
            "Burning sensation when I pee for 2 days",
            "Minor laceration on my thumb that stopped bleeding",
            "Ear pain and slight sore throat",
        ]
        for c in cases:
            assessment = calculate_ktas_acuity(c)
            self.assertEqual(assessment.acuity_level, 4, f"Expected Level 4 for '{c}'")
            self.assertEqual(assessment.tier_name, "Less Urgent")
            self.assertEqual(assessment.target_time_min, 60)

    def test_level_5_non_urgent(self):
        cases = [
            "Mild cold with runny nose and slight cough",
            "Need a prescription refill for my blood pressure medication",
            "Mild headache that goes away with water",
        ]
        for c in cases:
            assessment = calculate_ktas_acuity(c)
            self.assertEqual(assessment.acuity_level, 5, f"Expected Level 5 for '{c}'")
            self.assertEqual(assessment.tier_name, "Non-Urgent")
            self.assertEqual(assessment.target_time_min, 120)

    def test_acoustic_distress_elevation(self):
        """Acoustic biomarker distress flag must elevate acuity to Level 2 Emergent."""
        assessment = calculate_ktas_acuity("I have had a cough for 3 days", acoustic_distress=True)
        self.assertEqual(assessment.acuity_level, 2)
        self.assertTrue(assessment.escalate_to_staff)
        self.assertIn("Acoustic Biomarker", assessment.primary_driver)

    def test_execution_latency_sub_millisecond(self):
        text = "Severe crushing chest pain radiating to my jaw and left arm"
        for _ in range(50):
            calculate_ktas_acuity(text)

        n_iters = 1000
        t0 = time.perf_counter()
        for _ in range(n_iters):
            calculate_ktas_acuity(text)
        elapsed_s = time.perf_counter() - t0
        avg_ms = (elapsed_s / n_iters) * 1000.0

        print(f"\n⚡ KTAS Engine Average Execution Latency: {avg_ms:.4f} ms per assessment ({n_iters} iterations)")
        self.assertLess(avg_ms, 0.1, "KTAS acuity scoring must execute in <0.1 ms")


if __name__ == "__main__":
    unittest.main()
