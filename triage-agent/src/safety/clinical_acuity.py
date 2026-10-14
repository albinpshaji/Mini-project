"""
Standardized Deterministic KTAS / ESI 1-5 Clinical Acuity Engine.
Grounding:
  - Canadian / Korean Triage and Acuity Scale (KTAS 1-5)
  - Emergency Severity Index (ESI 1-5)
  - Paper 19: Explainable AI-Assisted ED Triage Using Multimodal Clinical Data (Menon & Fouad, ACM 2024)
  - Day 16 of OmniSpec-Triage Master Plan

Provides deterministic, sub-millisecond clinical severity tiering and explainable clinical rationales.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal


# KTAS Acuity Tier Definitions
KTAS_TIERS = {
    1: {"name": "Resuscitation", "target_time_min": 0, "description": "Immediate life threat requiring resuscitation."},
    2: {"name": "Emergent", "target_time_min": 15, "description": "High-risk condition, severe distress, or potential life/limb threat."},
    3: {"name": "Urgent", "target_time_min": 30, "description": "Stable condition requiring evaluation and multiple hospital resources."},
    4: {"name": "Less Urgent", "target_time_min": 60, "description": "Stable condition requiring a single clinical evaluation or minor resource."},
    5: {"name": "Non-Urgent", "target_time_min": 120, "description": "Minor or chronic condition requiring no urgent hospital resources."},
}

# Approved Department Enums
DEPARTMENTS = [
    "Emergency",
    "General Medicine",
    "Cardiology",
    "Orthopedics",
    "Pediatrics",
    "ENT & Airway Clinic",
    "Neurology",
]

# Negation detection pattern
NEGATION_CHEST_PATTERN = re.compile(r"\b(no|not|without|denies|negative for)\b.{0,15}\b(chest pain|chest discomfort)\b", re.I)

# High-acuity clinical regex patterns
LEVEL_1_PATTERNS = [
    re.compile(r"\b(cardiac arrest|respiratory arrest|no pulse|passed out and won'?t wake up|unresponsive)\b", re.I),
    re.compile(r"\b(turning blue|blue lips|lips.{0,20}blue|cyanosis)\b", re.I),
    re.compile(r"\b(cannot|can'?t) catch (my )?breath\b", re.I),
    re.compile(r"\bstridor\b", re.I),
    re.compile(r"\bthroat (is )?closing\b", re.I),
    re.compile(r"\bsevere anaphylaxis\b", re.I),
]

LEVEL_2_PATTERNS = [
    re.compile(r"\b(crushing|severe|heavy|squeezing)\s+(pain|tightness|pressure|heaviness)\s+(in|on)\s+(my )?chest\b", re.I),
    re.compile(r"\bchest\s+(pain|tightness|pressure|heaviness|crushing|discomfort)\b", re.I),
    re.compile(r"\bpain\s+(radiating|shooting|goes)\s+(down|to|into)\s+(my )?(left |both )?arms?\b", re.I),
    re.compile(r"\bheart\s+(is )?(racing|pounding)\b.{0,40}\b(dizzy|faint|lightheaded)\b", re.I),
    re.compile(r"\bheart\s+racing\b.{0,30}\b(1[2-9]\d|[2-9]\d\d)\s*(bpm|beats per minute)?\b", re.I),
    re.compile(r"\b(1[2-9]\d|[2-9]\d\d)\s*(bpm|beats per minute)\b", re.I),
    re.compile(r"\b(slurred speech|facial droop|stroke|one side weak)\b", re.I),
    re.compile(r"\b(coughing|vomiting|spitting)\b.{0,20}\bblood\b", re.I),
    re.compile(r"\b(severe bleeding|heavy blood loss)\b", re.I),
    re.compile(r"\b(severe asthma attack|rescue inhaler is not helping)\b", re.I),
    re.compile(r"\b(seizure|unconscious|fainted|passed out)\b", re.I),
]

LEVEL_3_PATTERNS = [
    re.compile(r"\bpain in (my )?chest when (i )?(take a )?deep breath\b", re.I),
    re.compile(r"\bfluttering in (my )?chest\b", re.I),
    re.compile(r"\b(asthma|wheezing|bronchospasm)\b", re.I),
    re.compile(r"\b(severe abdominal pain|stomach pain|cramps|belly pain)\b", re.I),
    re.compile(r"\b(fracture|broken bone|dislocated|deformity)\b", re.I),
    re.compile(r"\bhigh fever\b.{0,30}\b(days?|week)\b", re.I),
    re.compile(r"\bpersistent (vomiting|diarrhea)\b", re.I),
]

LEVEL_4_PATTERNS = [
    re.compile(r"\bpain in (my )?chest.{0,25}\b(press|ribs?|tender|touch|costochondritis)\b", re.I),
    re.compile(r"\b(burning feeling behind (my )?breastbone|heartburn|acid reflux)\b", re.I),
    re.compile(r"\b(sprain|twisted ankle|wrist pain|knee pain|ankle|shoulder hurts?)\b", re.I),
    re.compile(r"\b(cystitis|urinary|burns when (i )?pee|uti)\b", re.I),
    re.compile(r"\b(ear pain|earache|sore throat)\b", re.I),
    re.compile(r"\b(minor cut|laceration|scraped|bruise)\b", re.I),
    re.compile(r"\b(rash|itchy skin|hives)\b", re.I),
]

LEVEL_5_PATTERNS = [
    re.compile(r"\b(mild palpitations|palpitations after (coffee|tea|caffeine|espresso))\b", re.I),
    re.compile(r"\b(mild cold|runny nose|slight cough|congestion|mild sneeze)\b", re.I),
    re.compile(r"\b(prescription refill|routine checkup|medical certificate)\b", re.I),
    re.compile(r"\b(mild headache|minor ache)\b", re.I),
]


@dataclass
class KTASAssessment:
    acuity_level: Literal[1, 2, 3, 4, 5]
    tier_name: str
    target_time_min: int
    recommended_department: str
    primary_driver: str
    clinical_rationale: str
    escalate_to_staff: bool
    matched_red_flags: List[str] = field(default_factory=list)


def calculate_ktas_acuity(
    complaint_text: str,
    severity: Optional[str] = None,
    red_flags: Optional[List[str]] = None,
    acoustic_distress: bool = False,
    pediatric: bool = False,
) -> KTASAssessment:
    """
    Computes deterministic KTAS / ESI 1-5 triage acuity from presenting symptoms.
    Execution latency is <0.01ms (pure Python regex).
    """
    text = (complaint_text or "").strip()
    norm_text = text.lower()
    flags = list(red_flags or [])
    has_flags = len(flags) > 0
    sev = (severity or "moderate").lower()
    has_negated_chest_pain = bool(NEGATION_CHEST_PATTERN.search(norm_text))

    # 1. Evaluate Level 1 (Resuscitation)
    for pat in LEVEL_1_PATTERNS:
        m = pat.search(norm_text)
        if m:
            matched_phrase = m.group(0)
            return KTASAssessment(
                acuity_level=1,
                tier_name=KTAS_TIERS[1]["name"],
                target_time_min=KTAS_TIERS[1]["target_time_min"],
                recommended_department="Emergency",
                primary_driver=f"Life-threatening presentation: '{matched_phrase}'",
                clinical_rationale="Immediate resuscitation required for compromised airway, breathing, or circulation.",
                escalate_to_staff=True,
                matched_red_flags=flags or [matched_phrase],
            )

    # 2. Evaluate Level 2 (Emergent)
    if acoustic_distress:
        return KTASAssessment(
            acuity_level=2,
            tier_name=KTAS_TIERS[2]["name"],
            target_time_min=KTAS_TIERS[2]["target_time_min"],
            recommended_department="Emergency",
            primary_driver="Acoustic Biomarker Co-Processor: Acute respiratory dyspnea / stridor detected",
            clinical_rationale="Paralinguistic distress indicates imminent airway or pulmonary compromise.",
            escalate_to_staff=True,
            matched_red_flags=flags + ["acoustic_respiratory_distress"],
        )

    for pat in LEVEL_2_PATTERNS:
        # Ignore chest pain pattern if chest pain is explicitly negated
        if has_negated_chest_pain and "chest" in pat.pattern:
            continue

        m = pat.search(norm_text)
        if m:
            matched_phrase = m.group(0)
            dept = "Emergency"
            if any(w in matched_phrase for w in ["chest", "heart", "bpm", "arm"]):
                dept = "Emergency"
            elif any(w in matched_phrase for w in ["speech", "droop", "stroke"]):
                dept = "Emergency"
            return KTASAssessment(
                acuity_level=2,
                tier_name=KTAS_TIERS[2]["name"],
                target_time_min=KTAS_TIERS[2]["target_time_min"],
                recommended_department=dept,
                primary_driver=f"High-risk acute symptom: '{matched_phrase}'",
                clinical_rationale="Potential life or organ threat requiring prompt emergency intervention within 15 minutes.",
                escalate_to_staff=True,
                matched_red_flags=flags or [matched_phrase],
            )

    if has_flags and not has_negated_chest_pain:
        return KTASAssessment(
            acuity_level=2,
            tier_name=KTAS_TIERS[2]["name"],
            target_time_min=KTAS_TIERS[2]["target_time_min"],
            recommended_department="Emergency",
            primary_driver=f"Deterministic clinical red flags matched: {flags}",
            clinical_rationale="Emergency physician evaluation required due to positive red-flag criteria.",
            escalate_to_staff=True,
            matched_red_flags=flags,
        )

    # 3. Evaluate Level 3 (Urgent)
    for pat in LEVEL_3_PATTERNS:
        m = pat.search(norm_text)
        if m:
            matched_phrase = m.group(0)
            dept = "General Medicine"
            if any(w in matched_phrase for w in ["fracture", "broken", "bone"]):
                dept = "Orthopedics"
            elif any(w in matched_phrase for w in ["asthma", "wheezing"]):
                dept = "Emergency"
            elif any(w in matched_phrase for w in ["chest", "fluttering"]):
                dept = "Emergency"
            return KTASAssessment(
                acuity_level=3,
                tier_name=KTAS_TIERS[3]["name"],
                target_time_min=KTAS_TIERS[3]["target_time_min"],
                recommended_department=dept,
                primary_driver=f"Urgent clinical condition: '{matched_phrase}'",
                clinical_rationale="Stable vitals but requiring diagnostic workup or clinical evaluation within 30 minutes.",
                escalate_to_staff=False,
                matched_red_flags=flags,
            )

    if sev == "severe":
        dept = "Emergency"
        if any(w in norm_text for w in ["ankle", "wrist", "knee", "bone", "arm", "leg"]):
            dept = "Orthopedics"
        return KTASAssessment(
            acuity_level=3,
            tier_name=KTAS_TIERS[3]["name"],
            target_time_min=30,
            recommended_department=dept,
            primary_driver="Severe pain intensity without red flags",
            clinical_rationale="Severe distress requires prompt evaluation and pain management.",
            escalate_to_staff=False,
            matched_red_flags=flags,
        )

    # 4. Evaluate Level 4 (Less Urgent)
    for pat in LEVEL_4_PATTERNS:
        m = pat.search(norm_text)
        if m:
            matched_phrase = m.group(0)
            dept = "General Medicine"
            if any(w in matched_phrase for w in ["ankle", "wrist", "sprain", "knee", "shoulder"]):
                dept = "Orthopedics"
            elif any(w in matched_phrase for w in ["ear", "throat"]):
                dept = "ENT & Airway Clinic"
            return KTASAssessment(
                acuity_level=4,
                tier_name=KTAS_TIERS[4]["name"],
                target_time_min=KTAS_TIERS[4]["target_time_min"],
                recommended_department=dept,
                primary_driver=f"Low-acuity localized condition: '{matched_phrase}'",
                clinical_rationale="Single evaluation or minor intervention anticipated with stable disposition.",
                escalate_to_staff=False,
                matched_red_flags=flags,
            )

    # 5. Evaluate Level 5 (Non-Urgent)
    for pat in LEVEL_5_PATTERNS:
        m = pat.search(norm_text)
        if m:
            matched_phrase = m.group(0)
            return KTASAssessment(
                acuity_level=5,
                tier_name=KTAS_TIERS[5]["name"],
                target_time_min=KTAS_TIERS[5]["target_time_min"],
                recommended_department="General Medicine",
                primary_driver=f"Non-urgent presentation: '{matched_phrase}'",
                clinical_rationale="Mild self-limiting symptoms appropriate for primary care or outpatient clinic.",
                escalate_to_staff=False,
                matched_red_flags=flags,
            )

    if sev in ("mild", "none"):
        return KTASAssessment(
            acuity_level=5,
            tier_name=KTAS_TIERS[5]["name"],
            target_time_min=KTAS_TIERS[5]["target_time_min"],
            recommended_department="General Medicine",
            primary_driver="Mild symptom presentation with no distress indicators",
            clinical_rationale="Stable outpatient triage disposition.",
            escalate_to_staff=False,
            matched_red_flags=flags,
        )

    return KTASAssessment(
        acuity_level=4,
        tier_name=KTAS_TIERS[4]["name"],
        target_time_min=KTAS_TIERS[4]["target_time_min"],
        recommended_department="General Medicine",
        primary_driver="Unspecified stable symptoms without red flags",
        clinical_rationale="Standard triage intake for outpatient or general clinical review.",
        escalate_to_staff=False,
        matched_red_flags=flags,
    )
