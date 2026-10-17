"""
Red-flag rule-based escalation.

Runs OUTSIDE the SLM -- rule-based keyword/phrase matching on the raw
transcript, independent of whatever the model decides to say. This is a
core safety requirement: we cannot rely solely on the SLM's judgment to
catch dangerous symptoms, especially in a small quantized model under 2B params.
"""

import re
from dataclasses import dataclass, field


@dataclass
class RedFlagResult:
    triggered: bool
    matched_phrases: list = field(default_factory=list)
    category: str = None


RED_FLAG_PATTERNS = {
    "cardiac": [
        r"\bchest pain\b",
        r"\b(pain|pressure|tightness|heaviness|crushing|squeezing)\s+(in|on)\s+(my\s+)?chest\b",
        r"\bchest\s+(feels|feeling|tightness|pressure|heaviness|crushing)\b",
        r"\b(pain|radiat\w*|goes|shooting)\s+(in|down|to|into)\s+(my\s+)?(left\s+|both\s+)?arms?\b",
        r"\bheart\s+(is\s+)?(racing|pounding)\b.{0,40}\b(dizzy|faint|lightheaded)\b",
        r"\bheart\s+(is\s+)?(racing|pounding)\b.{0,30}\b\d{2,3}\s*(bpm|beats per minute)?\b",
        r"\b(1[2-9]\d|[2-9]\d\d)\s*(bpm|beats per minute)\b",
    ],
    "respiratory": [
        r"\b(can'?t|cannot)\s+catch\s+(my\s+)?breath\b",
        r"\b(can'?t|cannot)\s+breathe\b",
        r"\bdifficulty\s+breathing\b",
        r"\bshort(ness)?\s+of\s+breath\b",
        r"\bgasping\s+for\s+air\b",
        r"\bturning\s+blue\b",
        r"\blips\b.{0,25}\bblue\b",
        r"\bhard(ly)?\s+to\s+breathe\b",
        r"\bstridor\b",
        r"\b(severe asthma attack|rescue inhaler (is )?not (working|helping))\b",
    ],
    "neurological": [
        r"\b(sudden|severe)\s+(headache|confusion)\b",
        r"\bworst\s+headache\b",
        r"\bthunderclap\b",
        r"\bslurred\s+speech\b",
        r"\bface\b.{0,25}\bdroop",
        r"\b(can'?t|cannot|unable to)\s+speak\b",
        r"\b(can'?t|cannot)\s+move\s+(my|one side)\b",
        r"\b(sudden numbness|cannot hold|unable to hold|numbness.{0,25}(hand|arm|leg))\b",
        r"\b(double vision|difficulty swallowing|dysphagia)\b",
        r"\b(rigid neck|stiff neck|neck stiffness|sensitivity to (bright )?light|photophobia)\b",
        r"\blost\s+consciousness\b",
        r"\bpassed\s+out\b",
        r"\bfainted\b",
        r"\bseizure\b",
        r"\bone\s+side\s+(of|weak)",
    ],
    "bleeding": [
        r"\b(pumping|pulsatile|arterial)\s+(bright\s+red\s+)?blood\b",
        r"\bsevere\s+bleeding\b",
        r"\bwon'?t\s+stop\s+bleeding\b",
        r"\b(coughing|vomiting|spitting)\b.{0,40}\bblood\b",
        r"\b(coffee ground|dark blood|vomiting.{0,25}blood)\b",
        r"\b(black tarry|melena|blood in stool)\b",
        r"\bblood\s+in\s+(my\s+)?(stool|vomit)\b",
        r"\bheavy\s+blood\s+loss\b",
        r"\b(car accident|seatbelt sign|high speed collision)\b",
    ],
    "allergic": [
        r"\bthroat\s+(is\s+)?(swelling|closing)|swelling shut\b",
        r"\bswelling\s+(of|in)\s+(my\s+)?(face|throat|lips)\b",
        r"\banaphyla",
    ],
}

_COMPILED = {
    category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for category, patterns in RED_FLAG_PATTERNS.items()
}

_NEGATION_RE = re.compile(
    r"\b(no|not|never|without|denies|denied|negative for|neither|nor|zero|free of)\b",
    re.IGNORECASE,
)

_MILD_QUALIFIER_RE = re.compile(
    r"\b(a bit of|a bit|a little bit of|a little|mild|mildly|slight|slightly|minor|barely|not much|not too much|manageable|when i cough|when coughing|only when lying|when i press|press on|ribs?|tender to touch|costochondritis)\b",
    re.IGNORECASE,
)

_ACUTE_RESPIRATORY_RE = re.compile(
    r"\b(severe|sudden|acute|extreme|cannot|can'?t|gasping|blue|unbearable|choking|struggling)\b",
    re.IGNORECASE,
)

_ACUTE_CARDIAC_RE = re.compile(
    r"\b(severe|sudden|acute|extreme|crushing|heavy\s+(pain|chest|pressure|tightness|sweat)|squeezing|radiat\w*|arm|jaw|neck|back|dizzy|faint|sweat|sweating|unbearable|tight|pressure)\b",
    re.IGNORECASE,
)

NEGATION_WINDOW_CHARS = 45


def check_red_flags(transcript: str) -> RedFlagResult:
    matched = []
    fired_category = None

    for category, patterns in _COMPILED.items():
        for pattern in patterns:
            for match in pattern.finditer(transcript):
                start = match.start()
                end = match.end()
                prefix = transcript[max(0, start - NEGATION_WINDOW_CHARS):start].strip()
                window = transcript[max(0, start - NEGATION_WINDOW_CHARS):min(len(transcript), end + NEGATION_WINDOW_CHARS)].strip()

                # 1. Check if the symptom is explicitly negated (e.g., "no chest pain", "without shortness of breath")
                if _NEGATION_RE.search(prefix):
                    continue

                # 2. Check mild qualifiers for respiratory (e.g. "a bit shortness of breath", "mild difficulty breathing")
                if category == "respiratory":
                    is_mild = bool(_MILD_QUALIFIER_RE.search(window))
                    has_acute = bool(_ACUTE_RESPIRATORY_RE.search(transcript))
                    if is_mild and not has_acute:
                        # Non-emergency respiratory presentation (e.g. cold/cough with slight shortness of breath)
                        # Handled by Pulmonology / ENT / General Medicine via SLM rather than Emergency Room 10 override
                        continue

                # 3. Check mild qualifiers for cardiac (e.g. "slight chest pain", "mild chest pain", "chest hurts when coughing")
                if category == "cardiac":
                    is_mild = bool(_MILD_QUALIFIER_RE.search(window))
                    has_acute = bool(_ACUTE_CARDIAC_RE.search(transcript))
                    if is_mild and not has_acute:
                        # Non-emergency mild chest presentation (e.g. cough-induced chest soreness, mild twinge)
                        # Handled by SLM + HospitalRouter (Ward 3 Cardiology Room 120 or Ward 2 Pulmonology Room 45)
                        continue

                matched.append(match.group(0))
                fired_category = fired_category or category

    return RedFlagResult(
        triggered=len(matched) > 0,
        matched_phrases=matched,
        category=fired_category,
    )
