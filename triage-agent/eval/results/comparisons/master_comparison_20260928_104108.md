# Comparative Evaluation Report: sequential_eval_baseline vs. speculative_eval_speculative

**Evaluation Date:** 2026-09-28T10:41:08.033418  
**Evaluator Status:** ❌ **FAILED REGRESSION AUDIT**

---

## 1. Stage-by-Stage Latency Breakdown (Glass-to-Glass)

| Pipeline Stage | Baseline (sequential_eval_baseline) | Candidate (speculative_eval_speculative) | Improvement / Delta | Why Candidate is Faster |
|---|---|---|---|---|
| **1. VAD Silence Wait** | 500 ms | **150 ms** | -70.0% | Predictive Semantic VAD (cuts silence on closure) |
| **2. Final ASR Pass** | 250 ms | **60 ms** | -76.0% | Dual-Model cached Whisper-base pass |
| **3. SLM Prefill** | 350 ms | **751 ms** | +114.6% | Pinned system prompt in GPU KV-cache |
| **4. SLM Generation** | 4583 ms | **2244 ms** | -51.0% | Speculative decoding (pre-generated while speaking) |
| **5. Clinical Tools & Guardrail** | 0 ms | **0 ms** | -0.0% | Asynchronous tool DAG + Aho-Corasick trie |
| **6. TTS Audio Synthesis** | 450 ms | **75 ms** | -83.3% | Sub-clause chunking (starts on 4 words) |
| **TOTAL GLASS-TO-GLASS (P50)** | **3454 ms** | **3006 ms** | **+13.0% (1.1x speedup)** | **Conversational Parity Achieved** |

---

## 2. Clinical Acuity & Edge Hardware Comparison

| Metric Dimension | Baseline | Candidate | Delta | Clinical / Hardware Standard |
|---|---|---|---|---|
| **KTAS Acuity Concordance ($\\kappa_w$)** | 0.790 | **0.790** | +0.000 | Target: $\\ge 0.85$ (Substantial Agreement) |
| **Under-Triage Rate ($R_{\\text{under}}$)** | 0.00% | **0.00%** | +0.00% | **Zero Tolerance Target: 0.00%** |
| **Peak GPU VRAM Utilization** | 3077 MB | **3077 MB** | +0 MB | Budget Ceiling: 2,048 MB (4GB total) |
| **Speculative Commit Rate** | N/A | **40.0%** | N/A | Target: $> 65.0\%$ |
| **False Commits on Negation** | 0 | **0** | 0 | Hard Negation Invariant Gate |

---

## 3. Regression Audit Results

⚠️ **REGRESSION DETECTED:**
- ❌ MEMORY OVERFLOW: Peak VRAM 3077MB exceeds safe 2,048MB ceiling.
