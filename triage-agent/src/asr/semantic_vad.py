"""
Predictive Semantic VAD (Voice Activity Projection heuristic).
Collapses VAD silence timeout from 500-800ms down to 150ms upon syntactic clause closure.
Literature: Ekstedt & Skantze (VAP, Interspeech 2022).
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class SemanticVADResult:
    is_closed: bool
    recommended_silence_ms: int
    rule_matched: str


class SemanticVAD:
    """
    Syntactic closure state machine for clinical triage speech.
    Detects whether a patient's utterance is grammatically/semantically complete.
    """

    # Incomplete clausal endings: prepositions, conjunctions, hesitation fillers
    HANGING_CONNECTORS = re.compile(
        r"\b(?:and|but|or|because|so|although|while|with|in|at|on|of|for|to|about|like|if|when|as|that|um|uh|er|ah)\s*$",
        re.IGNORECASE
    )

    # Grammatical sentence completion punctuation
    SENTENCE_END = re.compile(r"[.?!]\s*$")

    # Clinical duration & temporal boundary completions (supports digits and spelled numbers)
    DURATION_CLOSURE = re.compile(
        r"\b(?:since|for)\s+(?:yesterday|this\s+morning|last\s+night|(?:one|two|three|four|five|six|seven|eight|several|\d+)\s+(?:days?|hours?|weeks?|months?))\s*$"
        r"|\b(?:all\s+day|at\s+night|this\s+morning|yesterday|earlier\s+today|an\s+hour\s+ago|last\s+week)\s*$",
        re.IGNORECASE
    )

    # Conversational closure & denials
    CONVERSATIONAL_CLOSURE = re.compile(
        r"\b(?:that\s+(?:is|was)\s+(?:all|it|everything)|that'?s\s+(?:all|it|everything)|nothing\s+else|no\s+other\s+symptoms|no\s+more)\s*$"
        r"|^(?:no|yes|nope|yeah)\s*$",
        re.IGNORECASE
    )

    # Clinical symptom + anatomical location completion (e.g. "pain in my left arm")
    SYMPTOM_LOCATION_CLOSURE = re.compile(
        r"\b(?:pain|ache|pressure|tightness|swelling|burning|numbness|rash|cut|cramp)\s+(?:in|on|around)\s+(?:my\s+)?[a-zA-Z\s]{2,20}\s*$",
        re.IGNORECASE
    )

    def __init__(self, closed_silence_ms: int = 150, open_silence_ms: int = 650):
        self.closed_silence_ms = closed_silence_ms
        self.open_silence_ms = open_silence_ms

    def evaluate(self, partial_text: str) -> SemanticVADResult:
        """
        Evaluates a partial transcript. If syntactically complete, recommends 150ms timeout.
        If trailing on a connector or incomplete, maintains full open silence timeout.
        """
        text = (partial_text or "").strip()
        words = text.split()

        if len(words) < 2:
            # Single words or empty: not enough signal to force aggressive closure
            return SemanticVADResult(
                is_closed=False,
                recommended_silence_ms=self.open_silence_ms,
                rule_matched="insufficient_words"
            )

        # 1. Negative Invariant: Check if trailing on a connector / preposition
        if self.HANGING_CONNECTORS.search(text):
            return SemanticVADResult(
                is_closed=False,
                recommended_silence_ms=self.open_silence_ms,
                rule_matched="hanging_connector"
            )

        # 2. Punctuation finality
        if self.SENTENCE_END.search(text):
            return SemanticVADResult(
                is_closed=True,
                recommended_silence_ms=self.closed_silence_ms,
                rule_matched="sentence_end_punctuation"
            )

        # 3. Conversational termination markers ("that's all", "nothing else")
        if self.CONVERSATIONAL_CLOSURE.search(text):
            return SemanticVADResult(
                is_closed=True,
                recommended_silence_ms=self.closed_silence_ms,
                rule_matched="conversational_closure"
            )

        # 4. Duration completion ([Symptom] + "since yesterday" / "for 3 days")
        if self.DURATION_CLOSURE.search(text):
            return SemanticVADResult(
                is_closed=True,
                recommended_silence_ms=self.closed_silence_ms,
                rule_matched="duration_closure"
            )

        # 5. Symptom + Anatomical Target completion
        if self.SYMPTOM_LOCATION_CLOSURE.search(text):
            return SemanticVADResult(
                is_closed=True,
                recommended_silence_ms=self.closed_silence_ms,
                rule_matched="symptom_location_closure"
            )

        # Default fallback: if clause is >= 6 words and not hanging, grant partial closure
        if len(words) >= 6:
            return SemanticVADResult(
                is_closed=True,
                recommended_silence_ms=int((self.closed_silence_ms + self.open_silence_ms) / 2),
                rule_matched="clause_length_heuristic"
            )

        return SemanticVADResult(
            is_closed=False,
            recommended_silence_ms=self.open_silence_ms,
            rule_matched="open_turn"
        )

    def is_syntactically_closed(self, partial_text: str) -> bool:
        return self.evaluate(partial_text).is_closed

    def get_silence_timeout_ms(self, partial_text: str) -> int:
        return self.evaluate(partial_text).recommended_silence_ms
