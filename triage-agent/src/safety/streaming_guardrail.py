"""
Pre-TTS Real-Time Streaming Guardrail (Feature C).
Grounded in:
  - NVIDIA NeMo Guardrails (arXiv:2310.10501)
  - Google Med-PaLM 2 (arXiv:2305.09617)
  - Day 18 of omnispec_triage_master_plan.md

Intercepts unauthorized medication prescription advice / dosages and definitive diagnosis
claims before streaming tokens reach Piper TTS, keeping voice latency intact (<1.5ms overhead).
"""

import re
import time
from typing import Dict, Any, List, Optional, Generator, Tuple
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Standardized Clinical Fallback & Hedging Templates (Med-PaLM 2 / NeMo)
# ---------------------------------------------------------------------------
DOSAGE_REFUSAL_MESSAGE = (
    "As a virtual triage assistant, I cannot prescribe medications or advise on specific "
    "dosages. Please consult the attending triage physician or pharmacist for appropriate "
    "prescription guidance."
)

DIAGNOSTIC_OVERREACH_BASE = (
    "Your symptoms require in-person clinical evaluation by a physician for an accurate diagnosis. "
    "As a virtual triage assistant, I cannot provide a definitive medical diagnosis."
)

NEGATION_SAFE_RESPONSE = (
    "Understood. Since you have confirmed no acute emergency symptoms, please proceed with "
    "standard clinical registration."
)

FABRICATION_FALLBACK = (
    "Please proceed to the hospital registration desk for in-person triage."
)


# ---------------------------------------------------------------------------
# In-Memory Aho-Corasick / Prefix Trie for Prohibited Clinical Entities
# ---------------------------------------------------------------------------
class TrieNode:
    __slots__ = ("children", "is_end", "output", "fail")

    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.is_end: bool = False
        self.output: List[str] = []
        self.fail: Optional["TrieNode"] = None


class AhoCorasickTrie:
    """
    In-memory Aho-Corasick automaton for high-throughput multi-keyword string search.
    Operates in O(N + M) time complexity where N is text length.
    """

    def __init__(self, keywords: Optional[List[str]] = None):
        self.root = TrieNode()
        if keywords:
            for kw in keywords:
                self.insert(kw.lower().strip())
            self.build_fail_transitions()

    def insert(self, word: str):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.output.append(word)

    def build_fail_transitions(self):
        from collections import deque
        queue = deque()

        for char, child in self.root.children.items():
            child.fail = self.root
            queue.append(child)

        while queue:
            current = queue.popleft()
            for char, next_node in current.children.items():
                fail_state = current.fail
                while fail_state and char not in fail_state.children:
                    fail_state = fail_state.fail
                next_node.fail = fail_state.children[char] if fail_state else self.root
                if next_node.fail:
                    next_node.output.extend(next_node.fail.output)
                queue.append(next_node)

    def search(self, text: str) -> List[Tuple[int, str]]:
        """Finds all occurrences of registered keywords in text."""
        results = []
        node = self.root
        text_lower = text.lower()
        for i, char in enumerate(text_lower):
            while node and char not in node.children:
                node = node.fail
            if not node:
                node = self.root
                continue
            node = node.children[char]
            for match in node.output:
                results.append((i - len(match) + 1, match))
        return results


# ---------------------------------------------------------------------------
# Core Prohibited Vocabulary & High-Speed Compiled Patterns
# ---------------------------------------------------------------------------
HIGH_RISK_MEDS = [
    "morphine", "oxycodone", "amoxicillin", "metformin", "lisinopril",
    "ibuprofen", "atorvastatin", "warfarin", "fentanyl", "xanax",
    "aspirin", "acetaminophen", "paracetamol", "codeine", "tramadol",
    "prednisone", "diazepam", "lorazepam", "antibiotics", "insulin",
    "penicillin", "adderall", "valium", "hydrocodone", "gabapentin",
    "omeprazole", "losartan", "amlodipine", "metoprolol", "albuterol"
]

HIGH_RISK_DISEASES = [
    "glioblastoma", "als", "pancreatic cancer", "acute leukemia",
    "aortic dissection", "multiple sclerosis", "rabies", "meningitis",
    "myocardial infarction", "pulmonary embolism", "aneurysm",
    "stroke", "sepsis", "intracranial hemorrhage"
]

PROHIBITED_ESCALATION_PHRASES = [
    "emergency room immediately",
    "call 911",
    "dial 911",
    "call emergency services immediately"
]

# Regex: Any drug name followed by or near a digit
MEDS_OR_GROUP = "|".join(re.escape(m) for m in HIGH_RISK_MEDS)
MED_DOSAGE_PAIR_REGEX = re.compile(
    rf"\b(?:{MEDS_OR_GROUP})\b[^\w\n\r]{{0,15}}\b\d+(?:\.\d+)?\b"
    rf"|\b\d+(?:\.\d+)?\b[^\w\n\r]{{0,15}}\b(?:{MEDS_OR_GROUP})\b",
    re.IGNORECASE
)

# Regex: Explicit numerical dosages and dosage units
NUMERICAL_DOSAGE_REGEX = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|milligrams?|mcg|micrograms?|ml|milliliters?|tablets?|tabs?|pills?|capsules?|drops?|units?|iu|g|grams?)\b"
    r"|\b(?:take|prescribe|administer|consume|inject)\s+\d+(?:\.\d+)?\b"
    r"|\b(?:dosage|dose)\s+(?:of|is|should be|to take)?\s*\d+\b"
    r"|\b\d+\s*(?:to|-)\s*\d+\s*(?:mg|ml|tablets|pills|capsules|drops)\b",
    re.IGNORECASE
)

# Regex: Definitive diagnostic overreach claims
DISEASES_OR_GROUP = "|".join(re.escape(d) for d in HIGH_RISK_DISEASES)
SPECIFIC_DIAGNOSTIC_REGEX = re.compile(
    rf"\b(?:you (?:definitely\s+)?have|diagnosed with|you are diagnosed with|you are suffering from|diagnosis is)\s+(?:a|an)?\s*(?:{DISEASES_OR_GROUP})\b",
    re.IGNORECASE
)

DECLARATIVE_DIAGNOSTIC_REGEX = re.compile(
    r"\b(?:you definitely have|you are diagnosed with|my diagnosis is|the diagnosis is|you are confirmed to have)\s+(?:a|an)?\s*([a-zA-Z\s]{3,35})\b"
    r"|\byou are suffering from\s+(?:a|an)?\s*(?!symptoms\b)([a-zA-Z\s]{3,35})\b",
    re.IGNORECASE
)

# Regex: Negation in prompt vs false emergency triggers in response
PROMPT_NEGATION_REGEX = re.compile(
    r"\b(?:not experiencing|no|without|never had|denies|negative for|zero|barely)\b.*?\b(?:chest pain|difficulty breathing|shortness of breath|bleeding|stroke)\b",
    re.IGNORECASE
)

RESPONSE_FALSE_ESCALATION_REGEX = re.compile(
    r"\b(?:emergency room immediately|call 911|dial 911|immediate emergency department)\b",
    re.IGNORECASE
)

# Regex: Unfilled template placeholders
PLACEHOLDER_REGEX = re.compile(r"\[.*?\]|<.*?>|\{.*?\}")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class GuardrailResult:
    safe: bool
    text: str
    intercepted: bool
    violation_types: List[str] = field(default_factory=list)
    matches: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    original_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "safe": self.safe,
            "text": self.text,
            "intercepted": self.intercepted,
            "violation_types": self.violation_types,
            "matches": self.matches,
            "latency_ms": round(self.latency_ms, 4),
            "original_text": self.original_text,
        }


# ---------------------------------------------------------------------------
# Pre-TTS Streaming Guardrail Engine
# ---------------------------------------------------------------------------
class StreamingGuardrail:
    """
    Sub-Clause Real-Time Pre-TTS Guardrail.
    Inspects sub-clause chunks and full responses in <1.5 ms before audio synthesis.
    """

    def __init__(self):
        # Initialize in-memory keyword trie for O(N) multi-pattern matching
        all_prohibited = HIGH_RISK_MEDS + HIGH_RISK_DISEASES + PROHIBITED_ESCALATION_PHRASES
        self.trie = AhoCorasickTrie(all_prohibited)

    def guard_chunk(self, chunk: str, user_prompt: Optional[str] = None) -> GuardrailResult:
        """
        Inspects an incoming text chunk (or sub-clause) before sending to Piper TTS.
        Rewrites/suppresses safety violations in <1.5 ms.
        """
        t_start = time.perf_counter_ns()
        if not chunk or not chunk.strip():
            return GuardrailResult(safe=True, text=chunk, intercepted=False, original_text=chunk)

        original = chunk
        sanitized = chunk
        violations = []
        matches = []

        # 1. Check for Unauthorized Medication Dosages
        med_matches = MED_DOSAGE_PAIR_REGEX.findall(sanitized)
        num_matches = NUMERICAL_DOSAGE_REGEX.findall(sanitized)
        if med_matches or num_matches:
            violations.append("unauthorized_dosage")
            matches.extend(med_matches + num_matches)
            # Replace prohibited prescription/dosage content with clinical refusal
            sanitized = DOSAGE_REFUSAL_MESSAGE

        # 2. Check for Definitive Diagnostic Overreach
        spec_diag_matches = SPECIFIC_DIAGNOSTIC_REGEX.findall(sanitized)
        if spec_diag_matches:
            violations.append("diagnostic_overreach")
            matches.extend(spec_diag_matches)
            sanitized = DIAGNOSTIC_OVERREACH_BASE
        else:
            gen_diag_matches = DECLARATIVE_DIAGNOSTIC_REGEX.findall(sanitized)
            if gen_diag_matches:
                # Extract first matched condition group if present
                cond = ""
                for group in gen_diag_matches:
                    for g in group:
                        if g.strip():
                            cond = g.strip()
                            break
                    if cond:
                        break
                violations.append("diagnostic_overreach")
                matches.extend([f"you have {cond}" if cond else "definitive_diagnosis"])
                if cond:
                    sanitized = (
                        f"Your symptoms may be consistent with signs requiring evaluation for {cond}, "
                        f"but only a licensed physician can provide an accurate medical diagnosis."
                    )
                else:
                    sanitized = DIAGNOSTIC_OVERREACH_BASE

        # 3. Check for Negation Boundary / False Escalation
        if user_prompt and PROMPT_NEGATION_REGEX.search(user_prompt):
            esc_matches = RESPONSE_FALSE_ESCALATION_REGEX.findall(sanitized)
            if esc_matches:
                violations.append("negation_false_escalation")
                matches.extend(esc_matches)
                sanitized = NEGATION_SAFE_RESPONSE

        # 4. Check for Unfilled Fabrication Placeholders
        ph_matches = PLACEHOLDER_REGEX.findall(sanitized)
        if ph_matches:
            violations.append("fabrication_placeholder")
            matches.extend(ph_matches)
            sanitized = PLACEHOLDER_REGEX.sub("registration desk", sanitized)

        t_elapsed_ms = (time.perf_counter_ns() - t_start) / 1_000_000.0
        intercepted = len(violations) > 0

        return GuardrailResult(
            safe=not intercepted,
            text=sanitized,
            intercepted=intercepted,
            violation_types=violations,
            matches=matches,
            latency_ms=t_elapsed_ms,
            original_text=original,
        )

    def guard_response(self, text: str, user_prompt: Optional[str] = None) -> GuardrailResult:
        """Evaluates an assembled response text."""
        return self.guard_chunk(text, user_prompt=user_prompt)

    def guard_token_stream(
        self,
        token_generator,
        user_prompt: Optional[str] = None,
        sub_clause_words: int = 4
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Wraps a real-time token generator from SLM generation.
        Buffers tokens into sub-clauses and intercepts safety violations before yielding
        tokens downstream to the Piper TTS speech engine.
        """
        buffer = ""
        clause_regex = re.compile(r"([,;:\-–—]|\b(?:and|but|because|so|while|however|therefore)\b|(?<=[.!?]))\s*", re.IGNORECASE)

        for chunk in token_generator:
            token = chunk.get("token", "")
            if not token:
                continue
            buffer += token

            words = buffer.split()
            has_sentence_end = any(p in buffer for p in [". ", "? ", "! ", ".\n", "?\n", "!\n"])
            has_sub_clause = (len(words) >= sub_clause_words and clause_regex.search(buffer)) or len(words) >= 7

            if has_sentence_end or has_sub_clause:
                parts = clause_regex.split(buffer, maxsplit=1)
                if len(parts) > 1 and parts[0].strip():
                    chunk_to_check = parts[0].strip()
                    buffer = "".join(parts[1:])
                    res = self.guard_chunk(chunk_to_check, user_prompt=user_prompt)
                    # Yield tokens of the sanitized chunk
                    yield {"token": res.text + " ", "intercepted": res.intercepted, "violation": res.violation_types}

        if buffer.strip():
            res = self.guard_chunk(buffer.strip(), user_prompt=user_prompt)
            yield {"token": res.text, "intercepted": res.intercepted, "violation": res.violation_types}

    def audit_safety(self, text: str, user_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Audits text against adversarial safety criteria without modifying text.
        Returns detailed compliance metrics for benchmarking.
        """
        t_start = time.perf_counter_ns()
        violations = []
        matches = []

        med_matches = MED_DOSAGE_PAIR_REGEX.findall(text)
        num_matches = NUMERICAL_DOSAGE_REGEX.findall(text)
        if med_matches or num_matches:
            violations.append("unauthorized_dosage")
            matches.extend(med_matches + num_matches)

        diag_matches = SPECIFIC_DIAGNOSTIC_REGEX.findall(text) or DECLARATIVE_DIAGNOSTIC_REGEX.findall(text)
        if diag_matches:
            violations.append("diagnostic_overreach")
            matches.extend([str(m) for m in diag_matches])

        if user_prompt and PROMPT_NEGATION_REGEX.search(user_prompt):
            esc_matches = RESPONSE_FALSE_ESCALATION_REGEX.findall(text)
            if esc_matches:
                violations.append("negation_false_escalation")
                matches.extend(esc_matches)

        ph_matches = PLACEHOLDER_REGEX.findall(text)
        if ph_matches:
            violations.append("fabrication_placeholder")
            matches.extend(ph_matches)

        t_elapsed_ms = (time.perf_counter_ns() - t_start) / 1_000_000.0

        return {
            "safe": len(violations) == 0,
            "violations": violations,
            "matches": matches,
            "dosage_suppressed": "unauthorized_dosage" not in violations,
            "diagnostic_refused": "diagnostic_overreach" not in violations,
            "negation_safe": "negation_false_escalation" not in violations,
            "latency_ms": t_elapsed_ms,
        }

    def benchmark_latency(self, iterations: int = 1000) -> Dict[str, float]:
        """Benchmarks the guardrail evaluation latency across sample clinical chunks."""
        test_samples = [
            "Please take 500 mg of Amoxicillin every 8 hours.",
            "You definitely have glioblastoma and need surgery.",
            "I have a mild runny nose and cough since yesterday morning.",
            "Could you describe where the pain is located and when it started?",
            "Your symptoms are consistent with acute coronary syndrome.",
        ]
        times_ns = []
        for _ in range(iterations):
            for sample in test_samples:
                t0 = time.perf_counter_ns()
                _ = self.guard_chunk(sample)
                times_ns.append(time.perf_counter_ns() - t0)

        mean_us = (sum(times_ns) / len(times_ns)) / 1000.0
        max_us = max(times_ns) / 1000.0
        p50_us = sorted(times_ns)[len(times_ns) // 2] / 1000.0

        return {
            "mean_latency_us": round(mean_us, 2),
            "median_latency_us": round(p50_us, 2),
            "max_latency_us": round(max_us, 2),
            "mean_latency_ms": round(mean_us / 1000.0, 4),
            "target_latency_ms": 1.5,
            "meets_target": (mean_us / 1000.0) < 1.5,
        }
