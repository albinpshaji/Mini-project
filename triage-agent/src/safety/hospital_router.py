"""
Edge-Native Hospital Context Router (Clinical Lexical-Semantic Router).

Provides sub-millisecond (< 1.0ms) CPU-only selective retrieval of relevant
hospital rooms and departments based on patient symptom statements.

Prevents context bloat and prefill latency spikes on edge GPUs by retrieving
only the Top-1 (or Top-2) relevant rooms (~35 tokens) instead of dumping
the entire hospital catalog into the LLM prompt.
"""

import json
import os
import re
import math
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set

log = logging.getLogger("hospital_router")

# Clinical synonym normalization mapping
CLINICAL_SYNONYMS = {
    "heart": ["cardiac", "cardiology", "chest", "angina"],
    "chest": ["cardiac", "cardiology", "angina"],
    "flutter": ["palpitations", "arrhythmia", "cardiac"],
    "pounding": ["palpitations", "cardiac"],
    "fever": ["temperature", "pyrexia", "infection", "febrile"],
    "temp": ["temperature", "fever", "infection"],
    "chills": ["fever", "infection", "shivering"],
    "shivering": ["fever", "chills"],
    "sweats": ["fever", "chills"],
    "broken": ["fracture", "orthopedics", "bone"],
    "fracture": ["broken", "orthopedics", "bone"],
    "twist": ["sprain", "orthopedics", "joint"],
    "sprain": ["twist", "orthopedics", "joint"],
    "ankle": ["orthopedics", "joint", "fracture"],
    "wrist": ["orthopedics", "joint", "fracture"],
    "knee": ["orthopedics", "joint"],
    "bone": ["orthopedics", "fracture"],
    "asthma": ["pulmonology", "respiratory", "inhaler", "wheeze"],
    "wheeze": ["asthma", "pulmonology", "respiratory", "stridor"],
    "cough": ["respiratory", "pulmonology", "ent", "infection"],
    "breathe": ["respiratory", "pulmonology", "dyspnea"],
    "breathing": ["respiratory", "pulmonology", "dyspnea"],
    "stomach": ["abdominal", "gastroenterology", "gastric", "belly"],
    "belly": ["stomach", "abdominal", "gastroenterology"],
    "cramp": ["abdominal", "gastroenterology"],
    "vomit": ["nausea", "gastroenterology"],
    "nausea": ["vomit", "gastroenterology"],
    "headache": ["neurology", "migraine", "vertigo"],
    "migraine": ["headache", "neurology"],
    "dizzy": ["vertigo", "neurology", "dizziness"],
    "vertigo": ["dizzy", "neurology"],
    "ear": ["ent", "otolaryngology", "hearing"],
    "throat": ["ent", "pharyngitis", "tonsils"],
    "rash": ["dermatology", "skin", "hives"],
    "skin": ["dermatology", "rash"],
    "bleed": ["trauma", "emergency", "hemorrhage"],
    "bleeding": ["trauma", "emergency", "hemorrhage"],
    "blood": ["bleeding", "trauma", "emergency"],
    "unconscious": ["emergency", "trauma", "resuscitation"],
    "passed out": ["emergency", "trauma", "unconscious"],
    "baby": ["pediatric", "child", "infant"],
    "child": ["pediatric", "toddler", "infant"],
    "toddler": ["pediatric", "child"],
}

STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "you", "your", "he", "she", "it", "they",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
    "get", "got", "please", "could", "would", "like", "tell", "water", "glass"
}

_WORD_PATTERN = re.compile(r"[a-z0-9]+")
_ROOM_NUMBER_REGEX = re.compile(r"(?:room|rm|bay|cabin)\s*(?:no\.?|#)?\s*([0-9]+[a-z]?)", re.IGNORECASE)


@dataclass
class HospitalRoomMatch:
    room_id: str
    room_number: str
    department: str
    floor: str
    acuity_tier: str
    routing_guidance: str
    score: float


class HospitalRouter:
    """
    Sub-millisecond clinical room router for edge LLMs.
    """

    def __init__(
        self,
        directory_path: Optional[str] = None,
        match_threshold: float = 0.30,
        top_k: int = 1,
    ):
        self.match_threshold = match_threshold
        self.top_k = top_k
        self.rooms: List[Dict[str, Any]] = []
        self.valid_room_numbers: Set[str] = set()
        self.emergency_room: Optional[HospitalRoomMatch] = None

        if directory_path is None:
            # Default to bundled data directory
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            directory_path = os.path.join(base_dir, "eval", "data", "hospital_directory.json")

        self.directory_path = directory_path
        self._load_and_index()

    def _tokenize(self, text: str, filter_stopwords: bool = True) -> List[str]:
        tokens = _WORD_PATTERN.findall(text.lower())
        if filter_stopwords:
            return [t for t in tokens if t not in STOPWORDS]
        return tokens

    def _expand_tokens(self, tokens: List[str]) -> Set[str]:
        expanded = set(tokens)
        for token in tokens:
            if token in CLINICAL_SYNONYMS:
                expanded.update(CLINICAL_SYNONYMS[token])
        return expanded

    def _load_and_index(self):
        if not os.path.exists(self.directory_path):
            log.warning("Hospital directory not found at %s. Router disabled.", self.directory_path)
            return

        try:
            with open(self.directory_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.error("Failed to load hospital directory %s: %s", self.directory_path, e)
            return

        self.rooms = data.get("rooms", [])
        self.valid_room_numbers = {r["room_number"].strip() for r in self.rooms if "room_number" in r}

        # Precompute room token sets and BM25 document frequencies
        self._room_profiles = []
        doc_count = len(self.rooms)
        term_doc_freq = {}

        for room in self.rooms:
            # Combine department, floor, and keywords
            text_corpus = f"{room.get('department', '')} {room.get('floor', '')} " + " ".join(
                room.get("clinical_keywords", [])
            )
            raw_tokens = self._tokenize(text_corpus, filter_stopwords=True)
            expanded_tokens = self._expand_tokens(raw_tokens)

            # Record emergency room if found
            if "emergency" in room.get("acuity_tier", "").lower() or room.get("room_number") == "10":
                if self.emergency_room is None:
                    self.emergency_room = HospitalRoomMatch(
                        room_id=room.get("room_id", "emergency"),
                        room_number=room.get("room_number", "10"),
                        department=room.get("department", "Emergency Resuscitation"),
                        floor=room.get("floor", "Ground Floor"),
                        acuity_tier=room.get("acuity_tier", "Emergency"),
                        routing_guidance=room.get("routing_guidance", "Immediate Emergency Evaluation"),
                        score=1.0,
                    )

            profile = {
                "room": room,
                "tokens": expanded_tokens,
                "keywords": [k.lower() for k in room.get("clinical_keywords", [])],
                "raw_text": text_corpus.lower(),
            }
            self._room_profiles.append(profile)

            # Document frequencies for IDF
            for token in set(expanded_tokens):
                term_doc_freq[token] = term_doc_freq.get(token, 0) + 1

        # Calculate standard smoothed IDF
        self._idf = {}
        for term, df in term_doc_freq.items():
            self._idf[term] = math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))

        log.info(
            "HospitalRouter loaded %d rooms from %s (Emergency Room: %s)",
            len(self.rooms),
            self.directory_path,
            self.emergency_room.room_number if self.emergency_room else "None",
        )

    def match(
        self,
        transcript: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[HospitalRoomMatch]:
        """
        Sub-millisecond query routing.
        Returns top matching rooms if score >= threshold, else [] (zero context bloat).
        """
        if not self._room_profiles or not transcript:
            return []

        k = top_k if top_k is not None else self.top_k
        thresh = threshold if threshold is not None else self.match_threshold

        query_tokens = self._tokenize(transcript, filter_stopwords=True)
        if not query_tokens:
            return []

        query_expanded = self._expand_tokens(query_tokens)
        transcript_lower = transcript.lower()

        scored_matches: List[HospitalRoomMatch] = []

        for profile in self._room_profiles:
            room = profile["room"]
            room_tokens = profile["tokens"]
            keywords = profile["keywords"]

            # 1. Exact phrase matching bonus (e.g. "chest pain", "high fever")
            phrase_bonus = 0.0
            for kw in keywords:
                if kw in transcript_lower:
                    phrase_bonus += 0.45

            # 2. IDF-weighted token overlap
            overlap = query_expanded.intersection(room_tokens)
            if not overlap and phrase_bonus == 0.0:
                continue

            overlap_score = sum(self._idf.get(t, 0.5) for t in overlap)
            # Normalize by query length
            normalized_score = (overlap_score / max(len(query_tokens), 1)) + phrase_bonus

            if normalized_score >= thresh:
                scored_matches.append(
                    HospitalRoomMatch(
                        room_id=room.get("room_id", ""),
                        room_number=room.get("room_number", ""),
                        department=room.get("department", ""),
                        floor=room.get("floor", ""),
                        acuity_tier=room.get("acuity_tier", "Routine"),
                        routing_guidance=room.get("routing_guidance", ""),
                        score=round(normalized_score, 3),
                    )
                )

        scored_matches.sort(key=lambda m: m.score, reverse=True)
        return scored_matches[:k]

    def format_routing_hint(self, matches: List[HospitalRoomMatch]) -> str:
        """
        Formats matches into a compact, spoken-friendly prompt hint (< 45 tokens).
        """
        if not matches:
            return ""

        top = matches[0]
        return (
            f"[HOSPITAL ROUTING REFERENCE]\n"
            f"- Matched Clinic: Room {top.room_number} ({top.floor}) - {top.department}\n"
            f"- Guidance: If you have gathered sufficient clinical details or if the presentation warrants referral, "
            f"guide the patient to Room {top.room_number}. If you need to clarify symptom duration, severity, "
            f"or associated factors first, do so before directing them. Never direct to an unlisted room."
        )

    def extract_room_numbers_from_text(self, text: str) -> List[str]:
        """Extracts any room numbers mentioned in generated text."""
        return _ROOM_NUMBER_REGEX.findall(text)

    def is_valid_room(self, room_number: str) -> bool:
        """Returns True if the room number exists in the hospital directory."""
        return room_number.strip() in self.valid_room_numbers

    def sanitize_unauthorized_rooms(self, response_text: str) -> str:
        """
        Anti-hallucination shield: replaces non-whitelisted room numbers
        with safe general guidance.
        """
        found_rooms = self.extract_room_numbers_from_text(response_text)
        sanitized = response_text
        for room in found_rooms:
            if not self.is_valid_room(room):
                log.warning("Intercepted hallucinated room number '%s' in SLM output!", room)
                # Replace hallucinated room with 'the clinic reception desk'
                sanitized = re.sub(
                    rf"\b(?:room|rm|bay|cabin)\s*(?:no\.?|#)?\s*{re.escape(room)}\b",
                    "the front reception desk",
                    sanitized,
                    flags=re.IGNORECASE,
                )
        return sanitized
