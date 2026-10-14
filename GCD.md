# Grammar-Constrained Decoding (GCD) — Implementation Guide

> **Branch:** `main`  
> **Status:** ⚠️ **Pre-GCD Prototype (Generic JSON Mode Baseline)**  
> **Literature Grounding:** The original sequential baseline codebase before integrating Paper 18 (EMNLP 2023) and Day 15 Master Plan.

---

## 1. What Is This Branch?

This branch represents the **original baseline implementation** of OmniSpec-Triage.

In this branch, structured clinical documentation relies on **prompt-instructed JSON extraction** with Ollama's generic `format="json"` parameter:
* It forces the language model into JSON syntax mode, but **does not enforce an explicit schema, types, or enums**.
* The model decides which keys and values to emit based only on prompt instructions.
* If parsing fails or keys are missing, it falls back to `_raw_parse_error`.

---

## 2. Current Implementation in This Branch

1. **[`src/safety/slot_extraction.py`](file:///home/albin/miniproject/triage-agent/src/safety/slot_extraction.py):**
   * Calls `ollama.chat(format="json")` with natural language instructions (`SLOT_SCHEMA_DESCRIPTION`).
   * Lacks explicit clinical enums for departments or formal KTAS 1–5 triage tiers.
   * Defaults model parameter to `qwen2.5:1.5b-instruct` (though pipeline overrides it with `qwen3.5:4b`).

2. **[`src/slm/slm_engine.py`](file:///home/albin/miniproject/triage-agent/src/slm/slm_engine.py):**
   * Contains an unused placeholder parameter `grammar=None` in `generate_stream()` and `generate()`.

---

## 3. Empirical Performance (15 Clinical Cases)

Compared against `feature/no-gcd` and `feature/grammar-constrained-decoding`:

| Metric | `feature/no-gcd` (Free-form) | `main` (This Branch) | `feature/grammar-constrained-decoding` (Strict GCD) |
|---|:---:|:---:|:---:|
| **Schema Conformance** | 100.0% (Regex parsed) | **100.0%** | **100.0% (Enforced at Logits)** |
| **Department Accuracy** | 73.3% | **73.3%** | **86.7% (+13.4%)** |
| **KTAS Acuity Agreement ($\kappa$)** | 0.6770 | **0.6770** | **0.8986 (+32.7%)** |
| **Red-Flag Recall** | 100.0% | **100.0%** | **100.0%** |
| **Extraction Latency** | 2,807.2 ms | **2,761.9 ms** | 5,750.7 ms |

---

## 4. How This Compares to Other Branches

* **vs. `feature/no-gcd`:** `main` uses `format="json"`, avoiding raw regex parsing. However, accuracy and acuity metrics are virtually identical because neither branch enforces valid department enums or KTAS tiers.
* **vs. `feature/grammar-constrained-decoding`:** The GCD branch upgrades this system with strict Pydantic logit masking (`ClinicalTriageRecord`), GBNF grammars, physician-level KTAS scoring ($\kappa = 0.8986$), and a two-phase decoupled async worker.
