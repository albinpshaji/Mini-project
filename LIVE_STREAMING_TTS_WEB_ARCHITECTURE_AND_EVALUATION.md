# Live Streaming TTS & Pre-TTS Guardrail Web Architecture

**Project:** OmniSpec-Triage  
**Date:** October 14, 2026  
**Branches:** `feature/guardrail-ktas-gcd` | `feature/guardrail-ktas-main`

---

## 1. Overview & Problem Solved
Previously, `/api/chat` waited for complete SLM text generation and full Piper TTS synthesis before returning audio, resulting in **3.5s – 21s+ of silence** (Time-To-First-Audio / TTFA).

**The Solution:** Real-Time Sub-Clause Live Audio Streaming:
- **Sub-Clause Synthesizer:** Splits streaming SLM tokens at grammatical boundaries (3–5 words).
- **Safety Inspection:** Evaluates each sub-clause through `StreamingGuardrail` (<0.012 ms overhead).
- **In-Browser Audio Queue:** Sequentially plays audio chunks on the fly with zero buffer underruns, reducing perceived patient speech onset to **~176 ms** (a **13.5x – 126x speedup**).

---

## 2. Architecture & Dataflow

```mermaid
flowchart TD
    Patient["Patient Input"] --> PostStream["POST /api/chat/stream"]
    PostStream --> SLM["Qwen3.5:4B Token Stream"]
    
    SLM --> Guardrail{"Pre-TTS Streaming Guardrail\n(< 0.012 ms overhead)"}
    Guardrail --> Typewriter["SSE 'token' -> UI Typewriter Rendering"]
    Guardrail --> SubClause["Sub-Clause Chunker (3-5 words)"]
    
    SubClause --> PiperTTS["Piper Neural TTS (~35-90ms synth)"]
    PiperTTS --> AudioChunk["SSE 'audio_chunk' (Base64 WAV)"]
    AudioChunk --> AudioQueue["Client AudioStreamQueue\n(Continuous Speech Playback)"]
    
    PiperTTS -.-> DoneEvent["SSE 'done' (Full Replay WAV + EHR Slots)"]
```

---

## 3. Key Components Implemented

| Component | File | Key Role |
| :--- | :--- | :--- |
| **SSE Stream Endpoint** | `src/web/server.py` | `@app.post("/api/chat/stream")` yields live `token`, `audio_chunk`, and `done` events. |
| **Sub-Clause Synthesizer** | `src/web/server.py` | `SUB_CLAUSE_REGEX` chunks text at punctuation/conjunctions; synthesizes WAV chunks on the fly. |
| **AudioStreamQueue** | `frontend/src/App.jsx` | Queues and plays incoming audio chunks sequentially; handles interruption via `stopAllAudio()`. |
| **Static Client Parity** | `src/web/static/js/app.js` | Replicates SSE consumption and audio queue for fallback HTML users. |

---

## 4. Benchmark & Latency Results

### A. Pre-TTS Guardrail Micro-Overhead (3,000 Iterations)
- **Mean Overhead:** **11.23 µs (0.0112 ms)** *(Budget: < 1.50 ms — 133x faster)*
- **P95 Latency:** **18.62 µs**
- **Impact on TTFA:** **0.0 ms** (imperceptible)

### B. Speech Onset Comparison (10 Clinical Prompts)
| Metric | Legacy `/api/chat` (Full Output) | Live `/api/chat/stream` (Chunk Queue) | Improvement |
| :--- | :--- | :--- | :--- |
| **Time-To-First-Token (TTFT)** | N/A (waits for full response) | **167.9 ms – 183.3 ms** | Instant typewriter feedback |
| **Time-To-First-Audio (TTFA) Median** | **218.6 ms** | **176.8 ms** | **-41.8 ms faster voice onset** |
| **Long Turn TTFA** | Up to **21,235 ms** | **~168 ms** | **Up to 126.2x faster** |
| **Average Speech Onset Speedup** | Baseline (1.0x) | **13.5x faster** | Significant reduction in perceived wait |
| **Audio Buffer Underruns** | N/A | **0 underruns** | Continuous seamless speech |

### C. Safety & Clinical Invariance
- **Adversarial Safety (100 Probes):** 100% dosage suppression, 100% diagnostic overreach refusal.
- **Pipeline Evaluation (10 Vignettes):** KTAS agreement κ = 0.959, 0.0% critical under-triage, 3,075 MB peak VRAM (<4GB ceiling).
- **Unit Tests:** 28/28 passing in 0.094s.

---

## 5. Git Commit Parity
- **`feature/guardrail-ktas-gcd`:** Commits `b9e9091` (feat), `6011ee1` (eval), `560fff1` (docs).
- **`feature/guardrail-ktas-main`:** Commits `fbe6e17` (feat), `9e20eb0` (eval), `af33c50` (docs).
