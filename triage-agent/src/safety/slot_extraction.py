"""
Unconstrained (No-GCD) symptom slot extraction.
Grounding: Negative Control Branch (Free-form generation without grammar constraint or format="json").

Features:
  - Zero schema constraints (no Ollama format="json", no GBNF grammar).
  - Unconstrained LLM text generation.
  - Regex / string extraction fallback to parse JSON if emitted in the output.
"""

import re
import json
import ollama
from dataclasses import dataclass, asdict
from typing import Optional, List


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
        speaker = "Patient" if turn.get("role") == "user" else "Assistant"
        lines.append(f"{speaker}: {turn.get('text', '')}")
    return "\n".join(lines)


def _extract_json_block(text: str) -> Optional[dict]:
    """Attempts to find and parse a JSON object from free-form LLM output."""
    # 1. Try direct parse
    try:
        return json.loads(text.strip())
    except Exception:
        pass

    # 2. Try markdown fenced block ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # 3. Try finding outermost braces { ... }
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    return None


def extract_symptom_slots(conversation_history: list, model_name: str = "qwen3.5:4b") -> SymptomSlots:
    """
    Unconstrained extraction: Calls LLM without any grammar constraint or format='json'.
    Relies purely on prompt adherence and post-hoc regex parsing.
    """
    conversation_text = _conversation_to_text(conversation_history)
    prompt = f"{SLOT_SCHEMA_DESCRIPTION}\n\nConversation:\n{conversation_text}"

    try:
        # NO format="json" or grammar constraint!
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            think=False,
            keep_alive="24h",
            options={"num_ctx": 3072, "temperature": 0.1, "num_predict": 180},
        )
        raw_text = response.get("message", {}).get("content", "")
        parsed = _extract_json_block(raw_text)

        if not parsed:
            return SymptomSlots(
                confidence="low",
                _raw_parse_error=f"Unconstrained output failed JSON parsing. Raw output: {raw_text[:80]}..."
            )

        return SymptomSlots(
            chief_complaint=parsed.get("chief_complaint"),
            duration=parsed.get("duration"),
            severity=parsed.get("severity"),
            associated_symptoms=parsed.get("associated_symptoms", []) or [],
            red_flag_present=bool(parsed.get("red_flag_present", False)),
            recommended_department=parsed.get("recommended_department"),
            confidence=parsed.get("confidence", "low"),
        )
    except Exception as e:
        return SymptomSlots(confidence="low", _raw_parse_error=str(e))
