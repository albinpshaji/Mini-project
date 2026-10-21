# Comparative Evaluation Report: sequential_eval_baseline vs. speculative_eval_speculative

**Evaluation Date:** 2026-10-21T13:27:03.477293  
**Evaluator Status:** ❌ **FAILED REGRESSION AUDIT**

---

## 1. Stage-by-Stage Latency Breakdown (Glass-to-Glass)

| Pipeline Stage | Baseline (sequential_eval_baseline) | Candidate (speculative_eval_speculative) | Improvement / Delta | Why Candidate is Faster |
|---|---|---|---|---|
| **1. VAD Silence Wait** | 500 ms | **450 ms** | -10.0% | Predictive Semantic VAD (cuts silence on closure) |
| **2. Final ASR Pass** | 250 ms | **120 ms** | -52.0% | Dual-Model cached Whisper-base pass |
| **3. SLM Prefill** | 350 ms | **141 ms** | -59.6% | Pinned system prompt in GPU KV-cache |
| **4. SLM Generation** | 3038 ms | **1626 ms** | -46.5% | Speculative decoding (pre-generated while speaking) |
| **5. Clinical Tools & Guardrail** | 0 ms | **0 ms** | -0.0% | Asynchronous tool DAG + Aho-Corasick trie |
| **6. TTS Audio Synthesis** | 450 ms | **75 ms** | -83.3% | Sub-clause chunking (starts on 4 words) |
| **TOTAL GLASS-TO-GLASS (P50)** | **3690 ms** | **2693 ms** | **+27.0% (1.4x speedup)** | **Conversational Parity Achieved** |

---

## 2. Clinical Acuity & Edge Hardware Comparison

| Metric Dimension | Baseline | Candidate | Delta | Clinical / Hardware Standard |
|---|---|---|---|---|
| **KTAS Acuity Concordance ($\\kappa_w$)** | 0.921 | **0.921** | +0.000 | Target: $\\ge 0.85$ (Substantial Agreement) |
| **Under-Triage Rate ($R_{\\text{under}}$)** | 0.00% | **0.00%** | +0.00% | **Zero Tolerance Target: 0.00%** |
| **Peak GPU VRAM Utilization** | 3055 MB | **3075 MB** | +20 MB | Budget Ceiling: 2,048 MB (4GB total) |
| **Speculative Commit Rate** | N/A | **30.0%** | N/A | Target: $> 65.0\%$ |
| **False Commits on Negation** | 0 | **0** | 0 | Hard Negation Invariant Gate |

---

## 3. Regression Audit Results

⚠️ **REGRESSION DETECTED:**
- ❌ MEMORY OVERFLOW: Peak VRAM 3075MB exceeds safe 2,048MB ceiling.
