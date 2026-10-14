"""
Structured clinical symptom slot extraction using Grammar-Constrained Decoding (GCD).
Grounding: Paper 18 (EMNLP 2023) & Master Plan Week 3 Day 15.

Provides:
  - extract_symptom_slots(): Backwards-compatible function returning SymptomSlots.
  - extract_clinical_record(): New function returning strict ClinicalTriageRecord.
  - Integration with DecoupledAsyncGCDWorker for zero-latency background execution.
"""

import json
import logging
import ollama
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

from src.safety.clinical_schema import (
    ClinicalTriageRecord,
    get_clinical_triage_json_schema
)
from src.safety.gcd_engine import extract_clinical_triage_gcd

logger = logging.getLogger(__name__)


@dataclass
class SymptomSlots:
    chief_complaint: Optional[str] = None
    duration: Optional[str] = None
    severity: Optional[str] = None
    associated_symptoms: List[str] = None
    red_flag_present: bool = False
    recommended_department: Optional[str] = None
    confidence: str = "low"
    acuity_level: Optional[int] = None
    clinical_rationale: Optional[str] = None
    _raw_parse_error: Optional[str] = None

    def __post_init__(self):
        if self.associated_symptoms is None:
            self.associated_symptoms = []

    def to_dict(self):
        d = asdict(self)
        d.pop("_raw_parse_error", None)
        return d


def extract_clinical_record(
    conversation_history: list,
    model_name: str = "qwen3.5:4b"
) -> Optional[ClinicalTriageRecord]:
    """Extracts strict ClinicalTriageRecord via grammar logit masking."""
    record, err, _ = extract_clinical_triage_gcd(conversation_history, model_name=model_name)
    if err:
        logger.warning(f"Clinical record extraction warning: {err}")
    return record


def extract_symptom_slots(
    conversation_history: list,
    model_name: str = "qwen3.5:4b",
    use_gcd: bool = True
) -> SymptomSlots:
    """
    Extracts clinical slots from dialogue history.
    Uses Grammar-Constrained Decoding by default to guarantee schema validity (Paper 18).
    """
    if use_gcd:
        record, err, _ = extract_clinical_triage_gcd(conversation_history, model_name=model_name)
        if record is not None:
            return SymptomSlots(
                chief_complaint=record.chief_complaint,
                duration=record.symptom_duration,
                severity=record.pain_severity,
                associated_symptoms=record.matched_red_flags,
                red_flag_present=record.red_flag_present,
                recommended_department=record.recommended_department,
                confidence="high",
                acuity_level=record.acuity_level,
                clinical_rationale=record.clinical_rationale,
            )
        # If GCD failed or model missing schema capability, record error
        if err:
            logger.warning(f"GCD failed, evaluating fallback: {err}")

    # Fallback to legacy unconstrained prompt mode if explicitly disabled or GCD unavailable
    return _extract_symptom_slots_legacy(conversation_history, model_name=model_name)


def _extract_symptom_slots_legacy(conversation_history: list, model_name: str = "qwen3.5:4b") -> SymptomSlots:
    lines = []
    for turn in conversation_history:
        speaker = "Patient" if turn.get("role") == "user" else "Assistant"
        lines.append(f"{speaker}: {turn.get('text', '')}")
    conversation_text = "\n".join(lines)

    prompt = f"""Extract structured information from this patient triage conversation.
Return ONLY a JSON object with exactly these fields, no other text:
{{
  "chief_complaint": string (the main symptom, in a few words, or null if unclear),
  "duration": string or null (how long the symptom has lasted, e.g. "3 days", or null if not mentioned),
  "severity": "mild" | "moderate" | "severe" | null (based on patient's own description),
  "associated_symptoms": array of strings (other symptoms mentioned, empty array if none),
  "red_flag_present": boolean (true if any emergency-level symptom was mentioned),
  "recommended_department": string or null (e.g. "Emergency", "General Medicine", "Orthopedics", "Pediatrics", "Cardiology"),
  "confidence": "low" | "medium" | "high"
}}

Conversation:
{conversation_text}"""

    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            think=False,
            keep_alive="24h",
            options={"num_ctx": 3072, "temperature": 0.1, "num_predict": 150},
        )
        raw_text = response.get("message", {}).get("content", "")
        parsed = json.loads(raw_text)

        return SymptomSlots(
            chief_complaint=parsed.get("chief_complaint"),
            duration=parsed.get("duration"),
            severity=parsed.get("severity"),
            associated_symptoms=parsed.get("associated_symptoms", []) or [],
            red_flag_present=bool(parsed.get("red_flag_present", False)),
            recommended_department=parsed.get("recommended_department"),
            confidence=parsed.get("confidence", "medium"),
        )
    except Exception as e:
        return SymptomSlots(confidence="low", _raw_parse_error=str(e))
