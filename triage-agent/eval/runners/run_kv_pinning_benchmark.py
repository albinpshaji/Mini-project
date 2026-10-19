"""
Phase 2 Benchmark Runner: Static System Prompt Prefix KV-Cache Pinning (num_keep).
Literature Grounding:
  - ASPLOS 2025: Fast On-Device LLM Inference with NPUs & Prefix Caching (Xu et al.)
  - Master Plan Innovation 6: Persistent static KV-cache state preservation across turns
Compares Baseline (num_keep=0) vs. Phase 2 Candidate (num_keep=pinned_tokens).
"""

import os
import sys
import time
import json
import sqlite3
import statistics
import datetime
import ollama

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.slm.slm_engine import SLMEngine
from src.slm.prompt_templates import SYSTEM_PROMPT
from src.speculative.speculator import Speculator


def record_kv_pinning_db(summary: dict, db_path: str = "eval/results/history.db"):
    """Persists KV pinning run metrics to SQLite history.db."""
    try:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS kv_pinning_benchmark_runs (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT,
                branch TEXT,
                model_name TEXT,
                pinned_prefix_tokens INTEGER,
                mean_baseline_wall_ms REAL,
                mean_pinned_wall_ms REAL,
                mean_speedup REAL,
                speculative_restart_speedup REAL,
                summary_json_path TEXT
            )
        """)
        cur.execute("""
            INSERT OR REPLACE INTO kv_pinning_benchmark_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            summary.get("run_id", f"kv_{int(time.time())}"),
            summary["timestamp"],
            summary["branch"],
            summary["model_name"],
            summary["pinned_prefix_tokens"],
            summary["mean_baseline_wall_ms"],
            summary["mean_pinned_wall_ms"],
            summary["mean_speedup"],
            summary["speculative_restart_speedup"],
            summary.get("summary_json_path", "")
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ SQLite history.db notice: {e}")


TEST_SCENARIOS = [
    {
        "id": "cardiac_01",
        "category": "cardiac",
        "partial": "I have sudden severe pressure in my chest",
        "final": "I have sudden severe pressure in my chest spreading to my left shoulder",
    },
    {
        "id": "resp_01",
        "category": "respiratory",
        "partial": "I am having trouble catching my breath",
        "final": "I am having trouble catching my breath after walking up one flight of stairs",
    },
    {
        "id": "cold_01",
        "category": "general",
        "partial": "I have a runny nose and dry cough",
        "final": "I have a runny nose and dry cough that started three days ago",
    },
    {
        "id": "trauma_01",
        "category": "orthopedic",
        "partial": "I twisted my right ankle playing soccer",
        "final": "I twisted my right ankle playing soccer and now cannot bear weight on it",
    },
    {
        "id": "neuro_01",
        "category": "neurological",
        "partial": "I have a sudden severe headache",
        "final": "I have a sudden severe headache with blurry vision in my right eye",
    },
    {
        "id": "ent_01",
        "category": "ent",
        "partial": "My left ear has been ringing and aching",
        "final": "My left ear has been ringing and aching since yesterday morning",
    },
    {
        "id": "derm_01",
        "category": "dermatology",
        "partial": "I developed a red itchy rash on my arm",
        "final": "I developed a red itchy rash on my arm after gardening this morning",
    },
    {
        "id": "gi_01",
        "category": "gastrointestinal",
        "partial": "I have sharp cramping pain in my lower right abdomen",
        "final": "I have sharp cramping pain in my lower right abdomen for the past six hours",
    },
]


def measure_prefill(engine: SLMEngine, prompt_text: str, num_keep: int) -> dict:
    messages = engine._build_messages(prompt_text, [])
    t0 = time.perf_counter()
    res = ollama.chat(
        model=engine.model_name,
        messages=messages,
        think=False,
        keep_alive="24h",
        options={
            "num_ctx": engine.num_ctx,
            "temperature": 0.3,
            "num_predict": 1,
            "num_keep": num_keep,
        }
    )
    t1 = time.perf_counter()
    wall_clock_ms = (t1 - t0) * 1000.0
    prompt_eval_count = res.get("prompt_eval_count", 0)
    prompt_eval_duration_ms = res.get("prompt_eval_duration", 0) / 1e6
    return {
        "wall_clock_ms": wall_clock_ms,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration_ms": prompt_eval_duration_ms,
    }


def run_benchmark(model_name: str = "qwen3.5:4b", output_dir: str = "eval/results"):
    print("=" * 80)
    print(" ⚡ PHASE 2 BENCHMARK: STATIC SYSTEM PROMPT PREFIX KV-CACHE PINNING")
    print("=" * 80)

    cfg = {"model_name": model_name, "num_ctx": 3072, "temperature": 0.3, "max_new_tokens": 100}
    engine = SLMEngine(cfg)
    pinned_tokens = engine.system_prompt_tokens
    print(f"Model Name           : {model_name}")
    print(f"System Prompt Length : {len(SYSTEM_PROMPT.split())} words | {pinned_tokens} tokens")
    print(f"Test Scenarios       : {len(TEST_SCENARIOS)} clinical presentations")
    print("-" * 80)

    # 1. Warm-up
    print("Pre-warming model and baseline...")
    ollama.chat(model=model_name, messages=[{"role": "user", "content": "warmup"}], keep_alive="24h", think=False, options={"num_predict": 1})

    baseline_durations = []
    baseline_eval_durations = []
    pinned_durations = []
    pinned_eval_durations = []

    scenario_results = []

    print(f"\n{'ID':<12} | {'BASELINE (ms)':<15} | {'PINNED (ms)':<15} | {'SPEEDUP':<10} | {'PREFILL TOKENS'}")
    print("-" * 80)

    for sc in TEST_SCENARIOS:
        sc_id = sc["id"]
        full_text = sc["final"]

        # Run Baseline (num_keep = 0)
        base_res = measure_prefill(engine, full_text, num_keep=0)
        base_ms = base_res["wall_clock_ms"]
        base_eval_ms = base_res["prompt_eval_duration_ms"]
        baseline_durations.append(base_ms)
        baseline_eval_durations.append(base_eval_ms)

        # Run Phase 2 Candidate (num_keep = pinned_tokens)
        pinned_res = measure_prefill(engine, full_text, num_keep=pinned_tokens)
        pinned_ms = pinned_res["wall_clock_ms"]
        pinned_eval_ms = pinned_res["prompt_eval_duration_ms"]
        pinned_durations.append(pinned_ms)
        pinned_eval_durations.append(pinned_eval_ms)

        speedup = base_ms / max(pinned_ms, 0.001)
        print(f"{sc_id:<12} | {base_ms:<15.2f} | {pinned_ms:<15.2f} | {speedup:<10.2f}x | {pinned_res['prompt_eval_count']}")

        scenario_results.append({
            "id": sc_id,
            "category": sc["category"],
            "baseline_wall_ms": base_ms,
            "baseline_prompt_eval_ms": base_eval_ms,
            "pinned_wall_ms": pinned_ms,
            "pinned_prompt_eval_ms": pinned_eval_ms,
            "speedup": speedup,
            "prompt_eval_count": pinned_res["prompt_eval_count"],
        })

    # Speculative restart test: Partial -> Final
    spec_baseline_ms = []
    spec_pinned_ms = []
    print("\nEvaluating Speculative Partial -> Final Restart Latency:")
    for sc in TEST_SCENARIOS[:4]:
        # Partial first
        measure_prefill(engine, sc["partial"], num_keep=0)
        # Final immediately following
        b_res = measure_prefill(engine, sc["final"], num_keep=0)
        spec_baseline_ms.append(b_res["wall_clock_ms"])

        # Pinned partial first
        measure_prefill(engine, sc["partial"], num_keep=pinned_tokens)
        # Pinned final
        p_res = measure_prefill(engine, sc["final"], num_keep=pinned_tokens)
        spec_pinned_ms.append(p_res["wall_clock_ms"])

    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": "Phase 2: Static System Prompt Prefix KV-Pinning",
        "branch": "feature/kv-pinning-phase2",
        "model_name": model_name,
        "pinned_prefix_tokens": pinned_tokens,
        "num_scenarios": len(TEST_SCENARIOS),
        "mean_baseline_wall_ms": round(statistics.mean(baseline_durations), 2),
        "median_baseline_wall_ms": round(statistics.median(baseline_durations), 2),
        "mean_pinned_wall_ms": round(statistics.mean(pinned_durations), 2),
        "median_pinned_wall_ms": round(statistics.median(pinned_durations), 2),
        "mean_baseline_eval_ms": round(statistics.mean(baseline_eval_durations), 2),
        "mean_pinned_eval_ms": round(statistics.mean(pinned_eval_durations), 2),
        "mean_speedup": round(statistics.mean(baseline_durations) / statistics.mean(pinned_durations), 2),
        "speculative_restart_baseline_mean_ms": round(statistics.mean(spec_baseline_ms), 2),
        "speculative_restart_pinned_mean_ms": round(statistics.mean(spec_pinned_ms), 2),
        "speculative_restart_speedup": round(statistics.mean(spec_baseline_ms) / statistics.mean(spec_pinned_ms), 2),
        "scenarios": scenario_results,
    }

    print("\n" + "=" * 80)
    print(" 📊 SUMMARY RESULTS: PHASE 2 KV-CACHE PREFIX PINNING")
    print("=" * 80)
    print(f"Mean Baseline Prefill Time   : {summary['mean_baseline_wall_ms']} ms (Eval: {summary['mean_baseline_eval_ms']} ms)")
    print(f"Mean Pinned Prefill Time     : {summary['mean_pinned_wall_ms']} ms (Eval: {summary['mean_pinned_eval_ms']} ms)")
    print(f"Average Turn Prefill Speedup : {summary['mean_speedup']}x")
    print(f"Speculative Restart Speedup  : {summary['speculative_restart_speedup']}x ({summary['speculative_restart_baseline_mean_ms']} ms ➔ {summary['speculative_restart_pinned_mean_ms']} ms)")
    print("=" * 80)

    # Save summary JSON
    runs_dir = os.path.join(output_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    summary_path = os.path.join(runs_dir, "kv_pinning_phase2_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON saved to: {summary_path}")

    summary["summary_json_path"] = summary_path
    record_kv_pinning_db(summary)

    return summary


if __name__ == "__main__":
    run_benchmark()
