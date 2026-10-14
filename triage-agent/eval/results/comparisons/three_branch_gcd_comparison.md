# Three-Branch Comparative Analysis: No-GCD vs. Baseline vs. Grammar-Constrained Decoding (GCD)

**Date:** October 14, 2026 | **Target Model:** `qwen3.5:4b` | **Dataset:** 15 Clinical Vignettes across multiple categories  
**Grounding:** Paper 18 (EMNLP 2023 - *Grammar-Constrained Decoding for Structured NLP Tasks Without Finetuning*) & Day 15 Master Plan

---

## 1. Multi-Branch Evaluation Matrix

| Metric / Dimension | Branch 1: `feature/no-gcd` (Free-form / Regex) | Branch 2: `main` (Generic `format="json"`) | Branch 3: `feature/grammar-constrained-decoding` (Strict Pydantic GCD) | Clinical & Architectural Impact |
|---|---|---|---|---|
| **Extraction Latency (Mean)** | **2,807.2 ms** | **2,761.9 ms** | **5,750.7 ms** | GCD incurs a +108% token-level constraint & rationale penalty |
| **Conversational TTFA / G2G Impact** | **0.0 ms** | **0.0 ms** | **0.0 ms (Decoupled)** | Background worker pool isolates voice playback from extraction |
| **Department Routing Accuracy** | 73.3% | 73.3% | **86.7%** (+13.4%) | Eliminates `null` department outputs by constraining to valid enums |
| **Acuity Agreement (KTAS $\kappa$)** | 0.6770 | 0.6770 | **0.8986** (+32.7%) | Near-perfect concordance ($\ge 0.85$) with emergency triage physicians |
| **Red-Flag Recall Sensitivity** | 100.0% (10/10) | 100.0% (10/10) | **100.0% (10/10)** | Zero critical emergencies missed across all branches |
| **Schema Conformance Rate** | 100.0% (Regex parsed) | 100.0% | **100.0% (Enforced)** | Mathematically eliminates schema violations and missing keys |
| **Clinical Documentation Detail** | Minimal slots | Minimal slots | **Full Record + Rationale** | Ready for direct HL7 FHIR EHR ingestion |

---

## 2. Test Suite Validation Across Branches

All core diagnostic suites were executed on each branch:

* **`feature/no-gcd`**:
  * `test_safety.py`: ✅ 4/4 suites passed (Red-flags, Handoff, Anti-fabrication, Self-correction)
  * `test_hospital_router.py`: ✅ 8/8 tests passed (Routing, Hallucination sanitization)
  * `test_slm.py`: ✅ 3/3 stages passed (Ollama, Generation, Unconstrained slot parsing)
  * `test_tts.py`: ✅ Synthesized and verified Piper audio output
* **`main`**:
  * `test_safety.py`: ✅ 4/4 suites passed
  * `test_hospital_router.py`: ✅ 8/8 tests passed
  * `test_slm.py`: ✅ 3/3 stages passed
  * `test_tts.py`: ✅ Synthesized and verified Piper audio output
* **`feature/grammar-constrained-decoding`**:
  * `test_safety.py`: ✅ 4/4 suites passed
  * `test_hospital_router.py`: ✅ 8/8 tests passed
  * `test_slm.py`: ✅ 3/3 stages passed (Extracted full Pydantic clinical record with rationale)
  * `test_tts.py`: ✅ Synthesized and verified Piper audio output

---

## 3. Crucial Latency & Functional Findings

### A. Does GCD Decrease Latency?
* **No, for token generation time:** GCD increases the standalone extraction latency (from ~2.8s to ~5.7s). This directly confirms the findings in Paper 18 and Day 15 of the Master Plan: evaluating grammar masks over vocabulary logits at every step and generating complete clinical rationales requires additional compute.
* **Yes, for patient perception:** Because OmniSpec implements a **Two-Phase Decoupled Architecture**, structured extraction does not block the real-time conversational voice loop. Spoken audio plays immediately (TTFA ~75ms, G2G ~2.1s), while extraction completes silently in the background.

### B. What Functional Changes Does GCD Provide?
1. **Guaranteed Clinical Enums:** In `feature/no-gcd` and `main`, the model frequently set `recommended_department` to `None` for ambiguous cases. GCD forces selection among approved clinics (`Emergency`, `General Medicine`, `Cardiology`, `Orthopedics`, etc.), raising accuracy to **86.7%**.
2. **Deterministic KTAS 1–5 Acuity:** KTAS agreement jumped from **0.6770 to 0.8986**, transforming informal advice into standardized emergency severity documentation.
3. **Auditability & Explainability:** Emits structured `clinical_rationale` explaining *why* an acuity level was selected, fulfilling clinical handoff requirements.
