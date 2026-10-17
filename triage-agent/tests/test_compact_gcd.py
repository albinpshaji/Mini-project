"""
Unit Test Suite for Phase 1: Compact Schema GCD & Deterministic KTAS Offloading.
Literature Grounding:
  - XGrammar (MLSys 2025, Dong et al., arXiv:2411.15100)
  - EMNLP 2023 GCD (Saibo Geng et al., arXiv:2305.13971)
  - ACM 2024 KTAS ED Triage (Menon & Fouad)
"""

import unittest
from pydantic import ValidationError
from src.safety.clinical_schema import (
    ClinicalTriageRecord,
    CoreClinicalSlots,
    get_core_clinical_slots_json_schema,
    generate_core_clinical_slots_gbnf
)
from src.safety.gcd_engine import (
    extract_clinical_triage_gcd,
    extract_clinical_triage_fast_gcd
)


class TestCompactGCDSchema(unittest.TestCase):
    def test_core_slots_valid_instantiation(self):
        core = CoreClinicalSlots(
            chief_complaint="Severe crushing chest pain",
            symptom_duration="1 hour",
            pain_severity="severe",
            recommended_department="Emergency"
        )
        self.assertEqual(core.chief_complaint, "Severe crushing chest pain")
        self.assertEqual(core.pain_severity, "severe")
        self.assertEqual(core.recommended_department, "Emergency")

    def test_core_slots_forbid_extra(self):
        with self.assertRaises(ValidationError):
            CoreClinicalSlots(
                chief_complaint="Headache",
                recommended_department="General Medicine",
                hallucinated_extra_key="invalid"
            )

    def test_core_slots_enum_validation(self):
        with self.assertRaises(ValidationError):
            CoreClinicalSlots(
                chief_complaint="Cough",
                recommended_department="Pulmonology"  # OOV enum: must raise ValidationError
            )

    def test_core_slots_json_schema(self):
        schema = get_core_clinical_slots_json_schema()
        self.assertIn("properties", schema)
        self.assertIn("chief_complaint", schema["properties"])
        self.assertIn("recommended_department", schema["properties"])
        self.assertIn("Emergency", schema["properties"]["recommended_department"]["enum"])
        self.assertIn("Cardiology", schema["properties"]["recommended_department"]["enum"])
        self.assertNotIn("acuity_level", schema["properties"])  # Offloaded to deterministic engine!
        self.assertNotIn("clinical_rationale", schema["properties"])  # Offloaded!

    def test_core_slots_gbnf_generation(self):
        gbnf = generate_core_clinical_slots_gbnf()
        self.assertIn("root ::=", gbnf)
        self.assertIn("chief_complaint", gbnf)
        self.assertIn("recommended_department", gbnf)
        self.assertIn("Emergency", gbnf)


class TestCompactGCDEngine(unittest.TestCase):
    def test_fast_gcd_end_to_end(self):
        history = [
            {"role": "user", "text": "I have severe pressure in my chest and my left arm feels numb"}
        ]
        record, err, elapsed = extract_clinical_triage_fast_gcd(history, model_name="qwen3.5:4b")
        self.assertIsNone(err)
        self.assertIsNotNone(record)
        self.assertIsInstance(record, ClinicalTriageRecord)
        self.assertEqual(record.acuity_level, 2)
        self.assertTrue(record.red_flag_present)
        self.assertEqual(record.recommended_department, "Emergency")
        self.assertIn("life or organ threat", record.clinical_rationale)
        self.assertLess(elapsed, 4.0, f"Expected fast extraction, got {elapsed:.2f}s")

    def test_fast_gcd_negation(self):
        history = [
            {"role": "user", "text": "I have NO chest pain, sorry I meant my left shoulder hurts from gym"}
        ]
        record, err, elapsed = extract_clinical_triage_fast_gcd(history, model_name="qwen3.5:4b")
        self.assertIsNone(err)
        self.assertIsNotNone(record)
        self.assertFalse(record.red_flag_present)
        self.assertIn(record.acuity_level, [4, 5])
        self.assertEqual(record.recommended_department, "Orthopedics")

    def test_fast_gcd_cardiac_arrest(self):
        history = [
            {"role": "user", "text": "My father passed out and he has no pulse and is not breathing"}
        ]
        record, err, elapsed = extract_clinical_triage_fast_gcd(history, model_name="qwen3.5:4b")
        self.assertIsNone(err)
        self.assertIsNotNone(record)
        self.assertEqual(record.acuity_level, 1)
        self.assertTrue(record.red_flag_present)
        self.assertEqual(record.recommended_department, "Emergency")

    def test_fast_gcd_stroke(self):
        history = [
            {"role": "user", "text": "My wife suddenly cannot speak and the right side of her face is drooping"}
        ]
        record, err, elapsed = extract_clinical_triage_fast_gcd(history, model_name="qwen3.5:4b")
        self.assertIsNone(err)
        self.assertIsNotNone(record)
        self.assertIn(record.acuity_level, [1, 2])
        self.assertTrue(record.red_flag_present)
        self.assertEqual(record.recommended_department, "Emergency")


if __name__ == "__main__":
    unittest.main()
