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
        r"\bchest (feels|feeling) (tight|heavy|crushing)\b",
        r"\b(crushing|squeezing) (chest|pain)\b",
        r"\b(pain|radiat\w*|goes|shooting) (in|down|to|into) (my )?(left )?arm\b",
        r"\bheart (racing|pounding) and (dizzy|faint)\b",
    ],
    "respiratory": [
        r"\b(can'?t|cannot) breathe\b",
        r"\bdifficulty breathing\b",
        r"\bshort(ness)? of breath\b",
        r"\bgasping for air\b",
        r"\bturning blue\b",
        r"\blips\b.{0,25}\bblue\b",
        r"\bhard(ly)? to breathe\b",
    ],
    "neurological": [
        r"\b(sudden|severe) (headache|confusion)\b",
        r"\bslurred speech\b",
        r"\bface\b.{0,20}\bdroop",
        r"\b(can'?t|cannot) move (my|one side)\b",
        r"\blost consciousness\b",
        r"\bpassed out\b",
        r"\bfainted\b",
        r"\bseizure\b",
        r"\bone side (of|weak)",
    ],
    "bleeding": [
        r"\bsevere bleeding\b",
        r"\bwon'?t stop bleeding\b",
        r"\bvomiting blood\b",
        r"\bblood in (my )?(stool|vomit)\b",
        r"\bheavy blood loss\b",
    ],
    "allergic": [
        r"\bthroat (is )?closing\b",
        r"\bswelling (of|in) (my )?(face|throat|lips)\b",
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
    r"\b(a bit of|a bit|a little bit of|a little|mild|mildly|slight|slightly|minor|barely|not much|not too much|manageable|when i cough|when coughing)\b",
    re.IGNORECASE,
)

_ACUTE_RESPIRATORY_RE = re.compile(
    r"\b(severe|sudden|acute|extreme|cannot|can'?t|gasping|blue|unbearable|choking|struggling)\b",
    re.IGNORECASE,
)

_ACUTE_CARDIAC_RE = re.compile(
    r"\b(severe|sudden|acute|extreme|crushing|heavy|squeezing|radiat|arm|jaw|neck|back|dizzy|faint|sweat|sweating|unbearable|tight|pressure)\b",
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
