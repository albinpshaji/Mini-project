"""
Human handoff logic.

Decides when to escalate to human staff, independent of what the SLM says.
"""

import re
from dataclasses import dataclass
from src.safety.red_flag_rules import check_red_flags, RedFlagResult


EMERGENCY_OVERRIDE_MESSAGE = (
    "Based on what you've described, this needs immediate attention. "
    "Please proceed to the Emergency Department right away. "
    "I'm alerting hospital staff now."
)

MAX_TURNS_BEFORE_HANDOFF = 8

STUCK_HANDOFF_MESSAGE = (
    "I want to make sure you get the right care. "
    "Let me connect you with a member of our staff who can help directly."
)

_PLACEHOLDER_PATTERN = re.compile(r"\[[^\]]{0,60}\]")

FABRICATION_FALLBACK_MESSAGE = (
    "I don't have that specific information on hand, but our front desk "
    "staff can help you with that directly."
)


from typing import Optional, Any
from src.safety.red_flag_rules import check_red_flags, RedFlagResult

RESPIRATORY_DISTRESS_OVERRIDE_MESSAGE = (
    "I can hear that you are experiencing severe difficulty breathing. "
    "Please sit down and proceed to the Emergency Department immediately. "
    "I am alerting our emergency medical staff right now."
)


@dataclass
class HandoffDecision:
    escalate: bool
    reason: str = None
    override_message: str = None
    red_flag_category: str = None


def contains_fabricated_placeholder(text: str) -> bool:
    return bool(_PLACEHOLDER_PATTERN.search(text))


def evaluate_handoff(
    transcript: str,
    turn_count: int,
    acoustic_biomarkers: Optional[Any] = None,
    hospital_router: Optional[Any] = None,
) -> HandoffDecision:
    # 1. Paralinguistic Acoustic Distress Override (Google HeAR / Innovation 9)
    if acoustic_biomarkers is not None and getattr(acoustic_biomarkers, "acuity_boost", 0) >= 2:
        return HandoffDecision(
            escalate=True,
            reason="acoustic_respiratory_distress",
            override_message=RESPIRATORY_DISTRESS_OVERRIDE_MESSAGE,
            red_flag_category="respiratory_distress",
        )

    # 2. Text Red Flag Detection
    red_flag_result: RedFlagResult = check_red_flags(transcript)

    if red_flag_result.triggered:
        er_msg = EMERGENCY_OVERRIDE_MESSAGE
        if hospital_router is not None and getattr(hospital_router, "emergency_room", None) is not None:
            er = hospital_router.emergency_room
            er_msg = (
                f"Based on what you've described, this needs immediate attention. "
                f"Please proceed to Room {er.room_number} ({er.department}, {er.floor}) right away. "
                f"I'm alerting hospital staff now."
            )
        return HandoffDecision(
            escalate=True,
            reason="red_flag",
            override_message=er_msg,
            red_flag_category=red_flag_result.category,
        )

    # 3. Maximum Turn Stuck Gate
    if turn_count >= MAX_TURNS_BEFORE_HANDOFF:
        return HandoffDecision(
            escalate=True,
            reason="stuck",
            override_message=STUCK_HANDOFF_MESSAGE,
        )

    return HandoffDecision(escalate=False)
