# Grammar-Constrained Decoding (GCD) — Implementation Guide

> **Branch:** `feature/no-gcd`  
> **Status:** ❌ **Completely Removed (Unconstrained Negative Control)**  
> **Literature Grounding:** Experimental baseline representing traditional LLM prompt-based extraction without Grammar-Constrained Decoding (Paper 18, EMNLP 2023).

---

## 1. What Is This Branch?

This branch represents the **unconstrained negative control** for the OmniSpec-Triage system. 

All forms of Grammar-Constrained Decoding and structured JSON parameters have been **completely stripped**:
1. **No GBNF Grammars or CFG masks.**
2. **No `format="json"` parameter in Ollama.**
3. **No Pydantic schema validation at token generation time.**

Instead, the language model outputs unconstrained natural text. Structured clinical data is extracted post-hoc using regular expressions and string search (`_extract_json_block`).

---

## 2. Key Differences in This Branch

### 1. [`src/slm/slm_engine.py`](file:///home/albin/miniproject/triage-agent/src/slm/slm_engine.py):
* The `grammar=None` parameter has been removed from `generate_stream()` and `generate()`.

### 2. [`src/safety/slot_extraction.py`](file:///home/albin/miniproject/triage-agent/src/safety/slot_extraction.py):
* `ollama.chat()` is called without any `format="json"` argument.
* Added `_extract_json_block(text: str)`:
  ```python
  def _extract_json_block(text: str) -> Optional[dict]:
      # Fallback chain: direct parse -> markdown block -> outermost braces
  ```
* If the unconstrained model emits non-JSON text or preamble, it returns `_raw_parse_error`.

---

## 3. Empirical Performance (15 Clinical Cases)

Compared against `main` and `feature/grammar-constrained-decoding`:

| Metric | `feature/no-gcd` (This Branch) | `main` (Baseline JSON) | `feature/grammar-constrained-decoding` |
|---|:---:|:---:|:---:|
| **Schema Conformance** | 100.0% (Regex parsed) | 100.0% | **100.0% (Enforced at Logits)** |
| **Department Accuracy** | **73.3%** | 73.3% | **86.7% (+13.4%)** |
| **KTAS Acuity Agreement ($\kappa$)** | **0.6770** | 0.6770 | **0.8986 (+32.7%)** |
| **Red-Flag Recall** | 100.0% | 100.0% | **100.0%** |
| **Extraction Latency** | **2,807.2 ms** | 2,761.9 ms | 5,750.7 ms |

---

## 4. Key Takeaways & Why GCD is Needed

1. **Extraction Latency is Shorter (~2.8s), but Less Informative:**
   * Because the model is unconstrained and generates fewer structured tokens (often setting fields to `null` and omitting medical rationales), raw extraction takes ~2.8s.
2. **Missing Clinical Enums (Department = `None`):**
   * In 26.7% of cases, the unconstrained model fails to choose a valid clinic and returns `null`, leaving the patient without a referral destination.
3. **Informal Acuity Scoring ($\kappa = 0.6770$):**
   * Without formal KTAS 1–5 schema constraints, the model struggles with standardized emergency triage tiering.
