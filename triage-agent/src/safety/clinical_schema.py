"""
Clinical Triage Extraction Schema & Grammar Definitions.
Grounding: Paper 18 (EMNLP 2023 - Grammar-Constrained Decoding) & Day 15 Master Plan.

Defines:
  1. Strict Pydantic models for structured hospital EHR triage documentation.
  2. GBNF (Grammar-Based Normal Form) generator for llama.cpp logit masking.
  3. JSON Schema generator for Ollama / Outlines constrained decoding.
  4. Bidirectional mapping to legacy SymptomSlots.
"""

from typing import Optional, List, Literal, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class ClinicalTriageRecord(BaseModel):
    """
    Standardized clinical triage record conforming to KTAS / ESI standards.
    Mathematically enforced via grammar-constrained logit masking.
    """
    model_config = ConfigDict(extra="forbid")

    chief_complaint: str = Field(
        ...,
        description="The primary clinical complaint or presenting symptom in a concise phrase"
    )
    symptom_duration: Optional[str] = Field(
        default=None,
        description="Reported duration of symptoms (e.g. '30 minutes', '2 days', null if not stated)"
    )
    pain_severity: Optional[Literal["none", "mild", "moderate", "severe"]] = Field(
        default=None,
        description="Clinical pain severity level based on patient description"
    )
    acuity_level: Literal[1, 2, 3, 4, 5] = Field(
        ...,
        description="Emergency Severity Index / KTAS Acuity tier (1=Resuscitation, 2=Emergent, 3=Urgent, 4=Less Urgent, 5=Non-Urgent)"
    )
    red_flag_present: bool = Field(
        ...,
        description="True if an acute emergency or vital life threat is detected"
    )
    matched_red_flags: List[str] = Field(
        default_factory=list,
        description="List of specific clinical red-flag triggers identified"
    )
    recommended_department: Literal[
        "Emergency",
        "General Medicine",
        "Cardiology",
        "Orthopedics",
        "Pediatrics",
        "ENT & Airway Clinic",
        "Neurology"
    ] = Field(
        ...,
        description="Clinical department referral destination"
    )
    clinical_rationale: str = Field(
        ...,
        description="Single concise sentence medical reasoning, strictly under 15 words"
    )

    def to_legacy_slots_dict(self) -> Dict[str, Any]:
        """Maps structured record to legacy SymptomSlots representation."""
        return {
            "chief_complaint": self.chief_complaint,
            "duration": self.symptom_duration,
            "severity": self.pain_severity,
            "associated_symptoms": self.matched_red_flags,
            "red_flag_present": self.red_flag_present,
            "recommended_department": self.recommended_department,
            "confidence": "high",
            "acuity_level": self.acuity_level,
            "clinical_rationale": self.clinical_rationale,
        }


def get_clinical_triage_json_schema() -> Dict[str, Any]:
    """Generates strict JSON schema for Ollama / Outlines constrained decoding."""
    return ClinicalTriageRecord.model_json_schema()


def generate_clinical_triage_gbnf() -> str:
    """
    Generates Context-Free Grammar in GBNF format for llama.cpp logit masking.
    Paper 18: Grammar-Constrained Decoding for Structured NLP Tasks Without Finetuning.
    """
    gbnf = r"""
root ::= "{" ws "\"chief_complaint\":" ws string "," ws "\"symptom_duration\":" ws (string | "null") "," ws "\"pain_severity\":" ws pain_level "," ws "\"acuity_level\":" ws acuity "," ws "\"red_flag_present\":" ws boolean "," ws "\"matched_red_flags\":" ws string_list "," ws "\"recommended_department\":" ws department "," ws "\"clinical_rationale\":" ws string ws "}"

pain_level ::= "\"none\"" | "\"mild\"" | "\"moderate\"" | "\"severe\"" | "null"
acuity ::= "1" | "2" | "3" | "4" | "5"
boolean ::= "true" | "false"
department ::= "\"Emergency\"" | "\"General Medicine\"" | "\"Cardiology\"" | "\"Orthopedics\"" | "\"Pediatrics\"" | "\"ENT & Airway Clinic\"" | "\"Neurology\""

string_list ::= "[" ws (string ("," ws string)*)? ws "]"
string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
ws ::= [ \t\n\r]*
"""
    return gbnf.strip()


class CoreClinicalSlots(BaseModel):
    """
    Compact clinical slot extraction schema.
    Reduces LLM generation token budget from ~75 tokens to ~20 tokens.
    Literature: XGrammar (MLSys 2025) & EMNLP 2023 GCD.
    """
    model_config = ConfigDict(extra="forbid")

    chief_complaint: str = Field(
        ...,
        description="The primary clinical complaint or presenting symptom in a concise phrase"
    )
    symptom_duration: Optional[str] = Field(
        default=None,
        description="Reported duration of symptoms (e.g. '30 minutes', '2 days', null if not stated)"
    )
    pain_severity: Optional[Literal["none", "mild", "moderate", "severe"]] = Field(
        default=None,
        description="Clinical pain severity level based on patient description"
    )
    recommended_department: Literal[
        "Emergency",
        "General Medicine",
        "Cardiology",
        "Orthopedics",
        "Pediatrics",
        "ENT & Airway Clinic",
        "Neurology"
    ] = Field(
        ...,
        description="Clinical department referral destination"
    )


def get_core_clinical_slots_json_schema() -> Dict[str, Any]:
    """Generates strict JSON schema for CoreClinicalSlots constrained decoding."""
    return CoreClinicalSlots.model_json_schema()


def generate_core_clinical_slots_gbnf() -> str:
    """
    Generates Context-Free Grammar in GBNF format for llama.cpp logit masking
    on CoreClinicalSlots.
    """
    gbnf = r"""
root ::= "{" ws "\"chief_complaint\":" ws string "," ws "\"symptom_duration\":" ws (string | "null") "," ws "\"pain_severity\":" ws pain_level "," ws "\"recommended_department\":" ws department ws "}"

pain_level ::= "\"none\"" | "\"mild\"" | "\"moderate\"" | "\"severe\"" | "null"
department ::= "\"Emergency\"" | "\"General Medicine\"" | "\"Cardiology\"" | "\"Orthopedics\"" | "\"Pediatrics\"" | "\"ENT & Airway Clinic\"" | "\"Neurology\""

string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""
ws ::= [ \t\n\r]*
"""
    return gbnf.strip()

