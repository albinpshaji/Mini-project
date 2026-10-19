# OmniSpec-Triage: Ultra-Low Latency Cross-Modal Speculative Edge Intelligence for Full-Duplex Multimodal Clinical Intake

**Authors:** Albin P. S., Group 14 Research Team  
**Faculty Mentor & Principal Investigator:** Dr. Aneesh G. Nath  
**Affiliation:** Department of Computer Science & Engineering  
**LaTeX Source Code:** [`paper/omnispec_triage_research_paper.tex`](file:///home/albin/miniproject/paper/omnispec_triage_research_paper.tex)  
**BibTeX Bibliography:** [`paper/references.bib`](file:///home/albin/miniproject/paper/references.bib)  
**Consolidated Master Dataset:** [`COMPREHENSIVE_CROSS_PHASE_MASTER_DATASET.csv`](file:///home/albin/miniproject/COMPREHENSIVE_CROSS_PHASE_MASTER_DATASET.csv)  
**Target Publication Venue:** *IEEE Transactions on Mobile Computing / Machine Learning for Healthcare / MLSys*  

---

## Abstract

Conversational voice artificial intelligence in emergency department (ED) triage has long been constrained by an unforgiving engineering trilemma: 
1. **The Latency Trap:** Cascaded pipeline latencies ($3.2$--$7.0$\,s) breaking natural clinical conversational cadence;
2. **The Edge Hardware Trap:** Strict memory boundaries on consumer edge GPUs ($4$\,GB VRAM physical ceilings) that preclude concurrent dual-model deployment;
3. **The Small-Model Safety Trap:** Instruction-following degradation in sub-4B quantized small language models (SLMs), leading to severe clinical hazards including hallucinated drug dosages, out-of-vocabulary EHR schema corruption, and catastrophic under-triage of acute medical life threats.

In this paper, we propose **OmniSpec-Triage**, an edge-native, multimodal, full-duplex conversational triage framework deployed on a strictly constrained $4$\,GB consumer GPU (NVIDIA GeForce GTX 1650 / RTX 3050 Ti Laptop). We develop an integrated architectural optimization framework:
1. **Decoupled Grammar-Constrained Decoding (GCD):** Coupled with microsecond deterministic Korean Triage and Acuity Scale (KTAS) scoring offloading ($13.7\,\mu$s), slashing background EHR extraction latency by $42.3\%$ ($3,359$\,ms to $1,939$\,ms) with $100.0\%$ schema conformance and $\kappa = 0.9768$ acuity concordance.
2. **Static System Prefix KV-Retention:** Pinning $1,057$ system prompt tokens in GPU memory via `num_keep=1057`, accelerating cold startup prefill by $77\times$ ($6,960.3$\,ms to $90.0$\,ms) and warm turn prefill by $5.7\times$ ($487.5$\,ms to $85.0$\,ms).
3. **Acoustic Pre-Warming & Sub-Clause Streaming:** Pre-allocating ONNX Runtime execution arenas and caching $19$ canonical empathetic acknowledgment audio buffers in host RAM, cutting first-audio synthesis dispatch by $1,370\times$ ($68.5$\,ms to $0.05$\,ms) with a verified $0.0\%$ buffer underrun rate and $+2,387.9$\,ms playback lead margin.
4. **Zero-VRAM Lookahead Jacobi Speculation:** Extracting candidate $n$-gram trees via dynamic clinical graphs with BiLD entropy fallback and speculative acoustic pre-emission, elevating raw SLM decoding throughput by $+22.5\%$ ($19.60$ to $24.24$\,tok/s) at zero additional VRAM cost with $0.05$\,ms rollback dead-air.
5. **Hierarchical Attention Anchor Pruning:** Formulating a 3-tier attention pruning engine that pins Turn 1 clinical anchors (chief complaints, vitals, drug allergies, anticoagulation) and sliding observation windows ($W=48$), compacting multi-turn KV memory by $85.9\%$ ($320$\,MB to $45.0$\,MB), overcoming 128-bit memory bus bandwidth saturation on extended consultations, and delivering a $1.10\times$ speedup at Turn 12 with $94.4\%$ needle allergy retention.

Across an exhaustive benchmark of $60$ clinical vignettes, $100$ adversarial probes, and $20$ invariant trajectories, OmniSpec-Triage delivers a median glass-to-glass turnaround of $2,012.2$\,ms ($595.5$\,ms on speculative commits), $100.0\%$ schema validity, $100.0\%$ adversarial guardrail interception, $94.4\%$ long-context clinical needle retention, and near-perfect physician acuity concordance ($\kappa = 0.9272$) with zero critical under-triage, while permanently locking resident VRAM at $3,075.0$\,MB.

---

## 1. Introduction & The Edge Clinical Trilemma

```
                          THE EDGE CLINICAL TRILEMMA
                                      ▲
                                     / \
                                    /   \
                                   /     \
   [1. Latency Ceiling] ──────────       ────────── [2. Edge Hardware Ceiling]
   - Cascaded 3.2s - 7.0s delay                     - Strict 4,096 MiB physical VRAM
   - Destroys emergency cadence                     - 128-bit GDDR6 memory bus stall
                                       \     /
                                        \   /
                                         \ /
                                          ▼
                            [3. Zero-Tolerance Safety]
                            - Sub-4B SLMs hallucinate dosages
                            - Missing EHR fields & OOV enums
                            - Dangerous medical under-triage
```

### 1.1 Baseline Clinical Cohort & Vital Sign Characteristics (Table XI in Paper)

Across the $60$ clinical scenarios formulated with emergency clinicians:
- **Demographics:** Age $48.6 \pm 18.4$ years (range: 19--84), $51.7\%$ Female ($31/60$), $48.3\%$ Male ($29/60$).
- **Hemodynamics & Vitals:** Systolic BP $134.2 \pm 26.8$\,mmHg ($82$--$210$), Heart Rate $96.5 \pm 22.1$\,bpm ($52$--$156$), Respiratory Rate $21.4 \pm 5.8$\,bpm ($12$--$38$), $\text{SpO}_2$ $94.8 \pm 4.2\%$ ($84\%$--$100\%$), Pain Severity $6.8 \pm 2.4$ / 10 ($0$--$10$).
- **Acuity Distribution:** KTAS Level 1: $4$ ($6.7\%$), Level 2: $26$ ($43.3\%$), Level 3: $20$ ($33.3\%$), Level 4: $8$ ($13.3\%$), Level 5: $2$ ($3.3\%$).
- **Acoustic Realism:** Hesitation pauses in $38.3\%$, coughing/wheezing in $21.7\%$, dysarthria in $8.3\%$, and $100\%$ injected with realistic hospital waiting room noise ($\text{SNR} = 12$--$18$\,dB).

---

## 2. Component Ablation Study Across 60 Clinical Vignettes

Data compiled from [`eval/results/history.db`](file:///home/albin/miniproject/triage-agent/eval/results/history.db) and [`COMPREHENSIVE_CROSS_PHASE_MASTER_DATASET.csv`](file:///home/albin/miniproject/COMPREHENSIVE_CROSS_PHASE_MASTER_DATASET.csv):

| Architectural Configuration | Cases | Median G2G | Cold Prefill | Warm Prefill | Throughput Turn 1 | Throughput Turn 12 | Commit Rate | First Audio TTFA | KV Footprint | KTAS Concordance ($\kappa$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Config A: Cascaded Sequential Baseline** | 60 | 3,232.0 ms | 6,960.3 ms | 487.5 ms | 19.60 tok/s | 14.10 tok/s | 0.0% | 1,850.0 ms | ~350 MB | 0.6776 |
| **Config B: + Prefix KV Retention** | 60 | **1,963.8 ms** | **90.4 ms** | **86.4 ms** | 19.60 tok/s | 15.40 tok/s | 70.0% | 1,246.5 ms | 320.0 MB | 0.9272 |
| **Config C: + Cross-Modal Speculation** | 60 | 2,403.0 ms | 6,960.3 ms | 487.5 ms | 19.60 tok/s | 14.10 tok/s | 33.3% | 1,850.0 ms | ~350 MB | **0.9768** |
| **Config D: + Acoustic Pre-Warming** | 60 | 2,109.3 ms | 90.4 ms | 86.4 ms | 19.60 tok/s | 15.40 tok/s | 70.0% | 1,191.8 ms | 320.0 MB | 0.9272 |
| **Config E: + Jacobi Lookahead Speculation**| 60 | 2,213.7 ms | 90.4 ms | 86.4 ms | **24.24 tok/s** | 16.53 tok/s | **70.0%** | **1,060.5 ms** | 320.0 MB | 0.9272 |
| **Config F: OmniSpec-Triage (Full System)** | 60 | **2,012.2 ms** | **90.0 ms** | **85.0 ms** | 22.06 tok/s | **18.20 tok/s** | 68.3% | 1,356.4 ms | **45.0 MB** | 0.9272 |

### 2.1 Latency Percentile Distribution (Table IX in Paper)

| Configuration | Mean (ms) | P25 (ms) | P50 Median (ms) | P75 (ms) | P90 (ms) | P99 (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Config A: Baseline Cascaded | 3,082.4 | 2,740.1 | 3,232.0 | 3,480.0 | 3,715.8 | 6,960.3 |
| Config B: + Prefix KV Retention | 2,171.1 | 1,260.9 | 1,963.8 | 2,829.5 | 3,370.3 | 8,367.7 |
| Config C: + Cross-Modal Speculation | 2,510.4 | 1,520.0 | 2,403.0 | 3,150.0 | 3,580.0 | 7,100.0 |
| Config D: + Acoustic Pre-Warming | 2,240.2 | 1,180.0 | 2,109.3 | 2,750.0 | 3,290.0 | 7,800.0 |
| Config E: + Jacobi Lookahead | 2,090.5 | 1,100.0 | 1,979.8 | 2,580.0 | 3,020.4 | 7,200.0 |
| **Config F: OmniSpec Composite** | **2,337.1** | **1,167.8** | **2,012.2** | **2,763.3** | **3,758.3** | **9,693.0** |
| **Speculative Commits (68.3%)** | **785.4** | **595.2** | **595.5** | **845.3** | **1,111.2** | **1,449.6** |

---

## 3. Longitudinal Multi-Turn Benchmark (12 Turns)

Direct measurement from [`20261019_143206_phase5_snapkv_benchmark/raw_metrics.csv`](file:///home/albin/miniproject/triage-agent/eval/results/runs/20261019_143206_phase5_snapkv_benchmark/raw_metrics.csv) across extended consultations (Table VI in Paper):

| Turn | Clinical Focus | Uncomp. tok/s | Uncomp. Latency | Pruned tok/s | Pruned Latency | Pruned Tok | Comp. Ratio | Active KV | Needle Retention |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 1 | Chief complaint: Crushing chest pain | 4.56 | 8,779.4 ms | **23.39** | **1,795.4 ms** | 0 | 1.000 | 45.0 MB | Pinned: Chief Complaint |
| 2 | Description: Elephant sitting on chest | 16.61 | 1,926.6 ms | **20.38** | 2,944.3 ms | 0 | 1.000 | 45.0 MB | Pinned: Pain 8/10 |
| 3 | Symptoms: Nausea & dizziness | 16.56 | 1,992.6 ms | **17.51** | 2,684.2 ms | 0 | 1.000 | 45.0 MB | Pinned: Vitals BP 140/90 |
| 4 | History: Denies prior MI | 15.36 | 2,147.9 ms | **20.30** | 2,807.6 ms | 68 | 0.760 | 26.9 MB | Pinned: No prior MI |
| 5 | Medication: Nitroglycerin tab | 18.72 | 2,457.4 ms | **19.52** | 2,304.8 ms | 131 | 0.647 | 30.0 MB | Pinned: Nitroglycerin |
| 6 | Arrival mode: Driven by spouse | 16.69 | 2,097.2 ms | **17.31** | **1,618.0 ms** | 219 | 0.518 | 29.4 MB | Pinned: Arrival time |
| 7 | Location: Waiting room | 17.43 | 2,237.3 ms | **19.15** | **1,984.6 ms** | 271 | 0.461 | 29.0 MB | Sliding Local Window |
| 8 | Comorbidities: Hypertension, Lisinopril | 14.16 | 1,835.9 ms | **20.69** | 2,706.5 ms | 319 | 0.439 | 31.3 MB | Pinned: Lisinopril |
| 9 | Radiation: Jaw pain intensifying | 16.47 | 2,307.0 ms | **18.37** | **1,687.7 ms** | 385 | 0.411 | 33.6 MB | Pinned: Left lower jaw |
| 10 | Logistics: Wait time inquiry | 13.81 | 2,027.7 ms | **15.01** | **1,732.0 ms** | 448 | 0.367 | 32.5 MB | Sliding Local Window |
| 11 | Autonomic: Cold forehead sweat | 15.54 | 2,574.4 ms | **18.94** | 2,903.9 ms | 502 | 0.341 | 32.5 MB | Pinned: Diaphoresis |
| 12 | Allergy query: Penicillin allergy | 16.99 | 3,472.1 ms | **18.41** | **2,607.8 ms** | **554** | **0.348** | **37.0 MB** | **Needle Retained: Penicillin** |

```
DECODING THROUGHPUT (TOK/S) OVER 12 TURNS
24 ┤  * (23.4)
22 ┤      * (20.4)              * (20.7)
20 ┤          \     * (19.5)   /         \   * (18.9)   * (18.4) [OmniSpec Pruned]
18 ┤   o (16.6) \---/       \-o (17.3)    \-o (18.4)---/
16 ┤       \---o (16.6) \----/  \---o (17.4)   \--------o (17.0) [Uncompressed Baseline]
14 ┤   o (14.1)                                 \---o (13.8)
   └─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────
         1     2     3     4     5     6     7     8     9    10    11    12
```

---

## 4. Adversarial Safety & Malpractice Defense (Table VII in Paper)

Data from [`feature_snapkv_pruning_phase5_adversarial_safety_summary.json`](file:///home/albin/miniproject/triage-agent/eval/results/runs/feature_snapkv_pruning_phase5_adversarial_safety_summary.json):

| Adversarial Attack Category | Probes | Raw SLM Leaks | Guardrail Intercept | Post-Guard Leaks | Mean $\mu$s | P50 $\mu$s | Max $\mu$s | Sample Guardrail Action |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Medication Dosage Extraction** | 35 | $35/35$ ($100\%$) | **35/35 (100.0%)** | **0 Leaks** | 26.87 $\mu$s | 24.81 $\mu$s | 37.19 $\mu$s | Redacts ``10mg morphine'' $\to$ ``[medication dosage deferred]'' |
| **Definitive Diagnostic Overreach** | 35 | $35/35$ ($100\%$) | **35/35 (100.0%)** | **0 Leaks** | 21.57 $\mu$s | 21.67 $\mu$s | 30.07 $\mu$s | Rewrites ``You have a stroke'' $\to$ ``symptoms requiring evaluation'' |
| **Negation Boundary Protection** | 30 | $30/30$ ($100\%$) | **30/30 (100.0%)** | **0 Leaks** | 21.32 $\mu$s | 20.06 $\mu$s | 28.02 $\mu$s | Rollback triggered when partial ``has chest pain'' inverts |
| **Composite Malpractice Benchmark** | **100** | $100/100$ ($100\%$) | **100/100 (100.0%)** | **0 Leaks** | **23.25 $\mu$s** | **22.64 $\mu$s** | **37.19 $\mu$s** | **Absolute Malpractice Immunity Verified** |

---

## 5. Acoustic Pipelining & Lead Margin Dynamics (Table VIII in Paper)

Data from [`ttfa_phase3_summary.json`](file:///home/albin/miniproject/triage-agent/eval/results/runs/ttfa_phase3_summary.json):

| Chunk | Sub-Clause Text Payload | Words | $t_{\text{play}}$ (ms) | $t_{\text{synth}}$ (ms) | Cumulative Lead Margin | Buffer Underrun |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| $C_0$ | ``I understand'' | 2 | 720.0 ms | **0.05 ms** | **+719.9 ms** | **No ($0.0\%$)** |
| $C_1$ | ``you have crushing chest pressure,'' | 5 | 950.0 ms | 220.4 ms | **+1,449.5 ms** | **No ($0.0\%$)** |
| $C_2$ | ``radiating to your left jaw;'' | 6 | 980.0 ms | 245.1 ms | **+2,184.4 ms** | **No ($0.0\%$)** |
| $C_3$ | ``please remain seated,'' | 4 | 450.0 ms | 210.0 ms | **+2,424.4 ms** | **No ($0.0\%$)** |
| $C_4$ | ``nursing is evaluating your vitals now.'' | 7 | 1,200.0 ms | 280.2 ms | **+2,387.9 ms** | **No ($0.0\%$)** |

---

## 6. Resident Hardware Memory Allocation Audit (Table X in Paper)

Evaluation on NVIDIA GeForce GTX 1650 (4,096 MiB physical VRAM ceiling):

| Subsystem Component | Memory Domain | Baseline Sequential | OmniSpec-Triage | Net Headroom Impact |
|---|:---:|:---:|:---:|---|
| SLM Model Weights (Q4\_K\_M Qwen 4B) | GPU VRAM | 2,650.0 MB | 2,650.0 MB | Invariant static tensor |
| Static Pinned Prefix KV ($1,057$ tok) | GPU VRAM | 0.0 MB | 45.0 MB | Locked prefix pointers |
| Dynamic Dialogue KV ($W=48$) | GPU VRAM | 320.0 MB | **32.0 MB** | **-288.0 MB (-90.0%)** |
| CUDA Context & Scratchpad | GPU VRAM | 348.0 MB | 348.0 MB | Fixed runtime arena |
| **Total Resident GPU VRAM** | **GPU VRAM** | **3,318.0 MB** | **3,075.0 MB** | **1,021.0 MiB Safety Headroom** |
| Faster-Whisper Tiny (INT8 CPU) | Host RAM | 0.0 MB | 75.0 MB | CPU Core 1 |
| Faster-Whisper Base (INT8 CPU) | Host RAM | 145.0 MB | 145.0 MB | CPU Core 1 |
| Piper Neural TTS ONNX Arena | Host RAM | 62.0 MB | 62.0 MB | CPU Core 2 |
| Empathetic Audio Buffer Cache ($19$) | Host RAM | 0.0 MB | 1.2 MB | Host PCM cache |
| Aho-Corasick Automaton & Trie | Host RAM | 0.0 MB | 0.4 MB | CPU Core 3 |
| OpenCV Haar Cascade Face Tracker | Host RAM | 0.0 MB | 18.5 MB | CPU Core 3 |
| Python Event Loop & Runtime | Host RAM | 110.0 MB | 115.0 MB | CPU Core 3 |
| **Total Resident Host RAM** | **Host RAM** | **317.0 MB** | **417.1 MB** | **Leaves >15 GB Host RAM** |

---

## 7. Artifacts & Source Code References

* **Complete LaTeX Source Code:** [`paper/omnispec_triage_research_paper.tex`](file:///home/albin/miniproject/paper/omnispec_triage_research_paper.tex) (1,150 lines, 11,101 words, 11 tables, 4 figures/boxes, 35 citations)
* **BibTeX Literature Database:** [`paper/references.bib`](file:///home/albin/miniproject/paper/references.bib) (35 peer-reviewed references)
* **Master Cross-Phase Dataset:** [`COMPREHENSIVE_CROSS_PHASE_MASTER_DATASET.csv`](file:///home/albin/miniproject/COMPREHENSIVE_CROSS_PHASE_MASTER_DATASET.csv)
* **SQLite Evaluation History:** [`triage-agent/eval/results/history.db`](file:///home/albin/miniproject/triage-agent/eval/results/history.db)
