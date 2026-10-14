"""
Structured symptom slot extraction.

Runs as a SEPARATE call from the conversational response the patient hears --
extracts a structured JSON record using Ollama's format="json" mode.
"""

import json
import ollama
from dataclasses import dataclass, asdict
from typing import Optional, List
from src.safety.clinical_acuity import calculate_ktas_acuity


SLOT_SCHEMA_DESCRIPTION = """Extract structured information from this patient triage conversation.
Return ONLY a JSON object with exactly these fields, no other text:

{
  "chief_complaint": string (the main symptom, in a few words, or null if unclear),
  "duration": string or null (how long the symptom has lasted, e.g. "3 days", or null if not mentioned),
  "severity": "mild" | "moderate" | "severe" | null (based on patient's own description),
  "associated_symptoms": array of strings (other symptoms mentioned, empty array if none),
  "red_flag_present": boolean (true if any emergency-level symptom was mentioned: chest pain, difficulty breathing, severe bleeding, loss of consciousness, stroke symptoms),
  "recommended_department": string or null (e.g. "Emergency", "General Medicine", "Orthopedics", "Pediatrics", "Cardiology", or null if not yet determinable),
  "confidence": "low" | "medium" | "high" (how confident you are in this extraction given the conversation so far)
}"""


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


def _conversation_to_text(conversation_history: list) -> str:
    lines = []
    for turn in conversation_history:
        speaker = "Patient" if turn["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {turn['text']}")
    return "\n".join(lines)


def extract_symptom_slots(conversation_history: list, model_name: str = "qwen3.5:4b") -> SymptomSlots:
    conversation_text = _conversation_to_text(conversation_history)
    prompt = f"{SLOT_SCHEMA_DESCRIPTION}\n\nConversation:\n{conversation_text}"

    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            think=False,
            keep_alive="24h",
            options={"num_ctx": 3072, "temperature": 0.1, "num_predict": 120},
        )
        raw_text = response["message"]["content"]
        parsed = json.loads(raw_text)

        chief = parsed.get("chief_complaint")
        sev = parsed.get("severity")
        rf = bool(parsed.get("red_flag_present", False))
        assoc = parsed.get("associated_symptoms", []) or []
        dept = parsed.get("recommended_department")

        # Deterministic KTAS Acuity Engine (Menon & Fouad 2024 / Day 16 Master Plan)
        ktas = calculate_ktas_acuity(
            complaint_text=f"{chief or ''} {conversation_text}",
            severity=sev,
            red_flags=assoc if rf else []
        )

        return SymptomSlots(
            chief_complaint=chief,
            duration=parsed.get("duration"),
            severity=sev,
            associated_symptoms=assoc,
            red_flag_present=rf or (ktas.acuity_level <= 2 and ktas.escalate_to_staff),
            recommended_department=dept or ktas.recommended_department,
            confidence=parsed.get("confidence", "medium"),
            acuity_level=ktas.acuity_level,
            clinical_rationale=ktas.clinical_rationale,
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return SymptomSlots(confidence="low", _raw_parse_error=str(e))

