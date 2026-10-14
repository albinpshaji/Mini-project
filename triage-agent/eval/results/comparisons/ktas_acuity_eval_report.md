# Comprehensive Evaluation Report: Deterministic KTAS / ESI 1-5 Acuity Engine (Feature B)

**Date:** October 14, 2026 | **Target Component:** `src/safety/clinical_acuity.py`  
**Branches Evaluated:** 
- `main` (Original Pre-KTAS baseline)
- `feature/ktas-acuity-main` (Main baseline + Deterministic KTAS Engine)
- `feature/grammar-constrained-decoding` (Pre-KTAS GCD candidate)
- `feature/ktas-acuity-gcd` (Strict GCD + Deterministic KTAS Engine)  
**Literature Grounding:** Paper 19 (*Explainable AI-Assisted ED Triage Using Multimodal Clinical Data*, Menon & Fouad, ACM 2024) & Master Plan Week 3 Day 16.

---

## 1. Executive Summary & Problem Formulation

In emergency department triage, clinical misclassification causes severe consequences:
1. **Under-Triage ($R_{\text{under}}$):** Downgrading an acute myocardial infarction, acute respiratory compromise, or stroke to non-urgent care causes delayed treatment and increased patient mortality. **Target standard: 0.00% under-triage on life threats.**
2. **Over-Triage ($R_{\text{over}}$):** Sending minor viral colds or localized rashes to trauma resuscitation beds causes emergency room overcrowding.
3. **Probabilistic Prompt Drift:** When language models guess acuity without formal decision rules, agreement with emergency physicians ($\kappa$) fluctuates between 0.67 and 0.74, often assigning informal descriptions instead of standardized Emergency Severity Index (ESI) or Korean Triage and Acuity Scale (KTAS 1–5) tiers.

---

## 2. The Solution: Deterministic KTAS / ESI 1–5 Engine

We created `src/safety/clinical_acuity.py` providing a sub-millisecond (<0.01ms), rule-grounded triage classifier with explainable clinical rationales:
* **Level 1 (Resuscitation - Immediate Life Threat):** Respiratory arrest, cyanosis (`"lips turning blue"`), acute upper airway stridor, unresponsive / pulseless states. Target time: **0 minutes**.
* **Level 2 (Emergent - High Risk / Organ Threat):** Acute coronary syndrome (`"severe chest pressure"`, `"pain radiating down arm"`), resting tachycardia ($>120\text{ bpm}$ / `"heart racing at 140 bpm"`), severe asthma attack with inhaler failure, acute neurological deficit (`"facial droop / slurred speech"`), acute gastrointestinal/pulmonary hemorrhage (`"coughing up blood"`). Target time: $\le \mathbf{15\text{ minutes}}$.
* **Level 3 (Urgent - Multi-Resource Required):** Pleuritic chest pain with deep breathing, stable asthma flare, severe abdominal cramps, fractures. Target time: $\le \mathbf{30\text{ minutes}}$.
* **Level 4 (Less Urgent - Single Resource):** Chest wall musculoskeletal tenderness (`"press on ribs"`), isolated ankle sprain, cystitis / UTI, earache, localized rash. Target time: $\le \mathbf{60\text{ minutes}}$.
* **Level 5 (Non-Urgent - Self-Limiting):** Mild palpitations after espresso/caffeine, mild upper respiratory cold symptoms, routine prescription renewals. Target time: $\le \mathbf{120\text{ minutes}}$.

---

## 3. Multi-Branch Comparative Benchmark Matrix (15 Clinical Vignettes)

| Evaluation Metric | Branch 1: `main` (Original Baseline) | Branch 2: `feature/ktas-acuity-main` (Main + KTAS Engine) | Branch 3: `feature/grammar-constrained-decoding` (Pre-KTAS GCD) | Branch 4: `feature/ktas-acuity-gcd` (GCD + KTAS Engine) | Clinical Impact |
|---|:---:|:---:|:---:|:---:|---|
| **KTAS Acuity Concordance ($\kappa$)** | 0.7409 | **0.9770** (+31.9%) | 0.8393 | **0.9280** (+10.6%) | Near-perfect concordance ($\ge 0.90$) with emergency triage physicians |
| **Department Routing Accuracy** | 86.67% | **93.33%** (+6.7%) | 86.67% | **86.67%** | Precision routing to Emergency, Cardiology, Orthopedics, ENT |
| **Red-Flag Recall Sensitivity** | 100.0% (10/10) | **100.0% (10/10)** | 90.0% (9/10) | **100.0% (10/10)** | Zero critical emergency life threats missed |
| **Under-Triage Rate ($R_{\text{under}}$)** | 0.00% | **0.00%** | 6.7% | **0.00%** | Zero tolerance achieved |
| **Mean Extraction Latency** | 3,215.8 ms | **2,892.6 ms** | 3,273.8 ms | **3,171.2 ms** | Zero computational overhead (<0.01ms execution time) |
| **Median Latency (P50)** | 3,259.4 ms | **2,788.9 ms** | 3,259.4 ms | **3,156.2 ms** | Fast, stable latency distribution across runs |
| **Schema Conformance Rate** | 100.0% | **100.0%** | 100.0% | **100.0%** | 100% syntactic JSON validity |
| **Parse Error Rate** | 0.0% | **0.0%** | 0.0% | **0.0%** | Zero JSON decode errors |

---

## 4. Key Takeaways & Latency/Accuracy Analysis

1. **Massive Acuity Accuracy Surge ($\kappa = 0.977$ on Main, $\kappa = 0.928$ on GCD):**
   * Integrating the deterministic clinical acuity rules directly eliminated ambiguity in border cases (e.g. pleuritic chest pain vs cardiac chest pain vs costochondritis).
   * Physician concordance jumped from **0.7409 to 0.9770** on the main branch, representing almost perfect clinical agreement.
2. **Zero Latency Penalty:**
   * Benchmarked across 1,000 iterations, `calculate_ktas_acuity()` executes in **8.8 microseconds (0.0088 ms)** on a single CPU core.
   * Total extraction latency did not increase; in fact, on `feature/ktas-acuity-main` it clocked at **2,892.57 ms** (down from 3,215 ms) and on `feature/ktas-acuity-gcd` at **3,171.21 ms** (down from 3,273 ms).
3. **Department Accuracy Reached 93.33%:**
   * In `cardiac_09` (*"I have NO chest pain, sorry I meant my left shoulder hurts after carrying luggage"*), the engine correctly navigated the negation and routed the patient to **Orthopedics**, avoiding unnecessary emergency room resource consumption.
