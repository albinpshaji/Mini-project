# Comparative Evaluation Report: sequential_eval_baseline vs. speculative_eval_speculative

**Evaluation Date:** 2026-10-14T08:56:30.026073  
**Evaluator Status:** ❌ **FAILED REGRESSION AUDIT**

---

## 1. Stage-by-Stage Latency Breakdown (Glass-to-Glass)

| Pipeline Stage | Baseline (sequential_eval_baseline) | Candidate (speculative_eval_speculative) | Improvement / Delta | Why Candidate is Faster |
|---|---|---|---|---|
| **1. VAD Silence Wait** | 500 ms | **400 ms** | -20.0% | Predictive Semantic VAD (cuts silence on closure) |
| **2. Final ASR Pass** | 250 ms | **120 ms** | -52.0% | Dual-Model cached Whisper-base pass |
| **3. SLM Prefill** | 350 ms | **266 ms** | -23.9% | Pinned system prompt in GPU KV-cache |
| **4. SLM Generation** | 1882 ms | **1250 ms** | -33.6% | Speculative decoding (pre-generated while speaking) |
| **5. Clinical Tools & Guardrail** | 0 ms | **0 ms** | -0.0% | Asynchronous tool DAG + Aho-Corasick trie |
| **6. TTS Audio Synthesis** | 450 ms | **75 ms** | -83.3% | Sub-clause chunking (starts on 4 words) |
| **TOTAL GLASS-TO-GLASS (P50)** | **3060 ms** | **2403 ms** | **+21.5% (1.3x speedup)** | **Conversational Parity Achieved** |

---

## 2. Clinical Acuity & Edge Hardware Comparison

| Metric Dimension | Baseline | Candidate | Delta | Clinical / Hardware Standard |
|---|---|---|---|---|
| **KTAS Acuity Concordance ($\\kappa_w$)** | 0.667 | **0.667** | +0.000 | Target: $\\ge 0.85$ (Substantial Agreement) |
| **Under-Triage Rate ($R_{\\text{under}}$)** | 0.00% | **0.00%** | +0.00% | **Zero Tolerance Target: 0.00%** |
| **Peak GPU VRAM Utilization** | 3051 MB | **3071 MB** | +20 MB | Budget Ceiling: 2,048 MB (4GB total) |
| **Speculative Commit Rate** | N/A | **40.0%** | N/A | Target: $> 65.0\%$ |
| **False Commits on Negation** | 0 | **0** | 0 | Hard Negation Invariant Gate |

---

## 3. Regression Audit Results

⚠️ **REGRESSION DETECTED:**
- ❌ MEMORY OVERFLOW: Peak VRAM 3071MB exceeds safe 2,048MB ceiling.
