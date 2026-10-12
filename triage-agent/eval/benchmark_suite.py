"""
Comprehensive Latency & Algorithm Benchmark Suite for Speculative Triage Agent.

Measures and compares:
  - Sequential vs Speculative execution across multiple clinical categories.
  - Stage-by-stage latency breakdowns: TTFT (Time to First Token), Total Generation Time, Tokens/Sec.
  - Optional TTS synthesis latency (Time to First Audio Chunk).
  - Accuracy metrics: Commit Rate, False Commit Rate, True Rollbacks.
  - Parameter sweeps over divergence thresholds (0.5 to 0.9).
  - Exports detailed CSV and JSON reports for scientific analysis.

Usage:
  python -m eval.benchmark_suite
  python -m eval.benchmark_suite --sweep
  python -m eval.benchmark_suite --include-tts
  python -m eval.benchmark_suite --cases mild,cardiac
"""

import os
import sys
import time
import json
import csv
import argparse
import statistics
import datetime
import yaml

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.slm.slm_engine import SLMEngine
from src.speculative.speculator import Speculator
from src.speculative.divergence_detector import compute_divergence

# ---------------------------------------------------------------------------
# Comprehensive Clinical Benchmark Dataset
# Schema: (partial_text, final_text, category, ground_truth_diverged, conversation_history)
# ---------------------------------------------------------------------------
BENCHMARK_DATASET = [
    # Category: Mild / Routine Intake
    {
        "id": "mild_01",
        "category": "mild",
        "partial": "I am having a mild cold",
        "final": "I am having a mild cold",
        "ground_truth_diverged": False,
        "history": [],
        "notes": "Exact match"
    },
    {
        "id": "mild_02",
        "category": "mild",
        "partial": "I have had a mild headache since",
        "final": "I have had a mild headache since this morning",
        "ground_truth_diverged": False,
        "history": [],
        "notes": "Natural suffix extension"
    },
    {
        "id": "mild_03",
        "category": "mild",
        "partial": "Using a humidifier",
        "final": "Using a humidifier works quite well",
        "ground_truth_diverged": False,
        "history": [],
        "notes": "Predication completion"
    },
    {
        "id": "mild_04",
        "category": "mild",
        "partial": "makes my symptoms",
        "final": "makes my symptoms worse at night",
        "ground_truth_diverged": False,
        "history": [],
        "notes": "Contextual extension"
    },

    # Category: ASR Speech Artifacts & Homophones
    {
        "id": "noise_01",
        "category": "asr_noise",
        "partial": "I am having a mild cord",
        "final": "I am having a mild cold",
        "ground_truth_diverged": False,
        "history": [],
        "notes": "Homophone replacement ('cord' -> 'cold')"
    },
    {
        "id": "noise_02",
        "category": "asr_noise",
        "partial": "I am having a blog",
        "final": "I am having a blocked nose",
        "ground_truth_diverged": False,
        "history": [],
        "notes": "Truncated acoustic artifact ('blog' -> 'blocked nose')"
    },
    {
        "id": "noise_03",
        "category": "asr_noise",
        "partial": "I have had a b-",
        "final": "I have had a bad cough for three days",
        "ground_truth_diverged": False,
        "history": [],
        "notes": "Phoneme cutoff during articulation"
    },
    {
        "id": "noise_04",
        "category": "asr_noise",
        "partial": "Slight soul throat",
        "final": "Slight sore throat since yesterday",
        "ground_truth_diverged": False,
        "history": [],
        "notes": "Acoustic distortion ('soul' -> 'sore')"
    },

    # Category: Emergency & Red-Flag Presentation
    {
        "id": "redflag_01",
        "category": "emergency",
        "partial": "I have chest",
        "final": "I have chest pain and difficulty breathing",
        "ground_truth_diverged": False,
        "history": [],
        "notes": "Critical cardiac/respiratory escalation"
    },
    {
        "id": "redflag_02",
        "category": "emergency",
        "partial": "My left arm feels",
        "final": "My left arm feels numb and my face is drooping",
        "ground_truth_diverged": False,
        "history": [],
        "notes": "Neurological / stroke red flag"
    },
    {
        "id": "redflag_03",
        "category": "emergency",
        "partial": "Severe stomach",
        "final": "Severe bleeding from a deep wound on my leg",
        "ground_truth_diverged": True,
        "history": [],
        "notes": "Shift from abdominal to severe trauma bleeding"
    },

    # Category: Semantic Shift & Change of Mind (Must Rollback)
    {
        "id": "diverge_01",
        "category": "semantic_shift",
        "partial": "Using a humidifier",
        "final": "No actually nothing helped at all",
        "ground_truth_diverged": True,
        "history": [],
        "notes": "Total patient reversal"
    },
    {
        "id": "diverge_02",
        "category": "semantic_shift",
        "partial": "Hello how are you doing",
        "final": "I am having a sharp pain in my lower abdomen",
        "ground_truth_diverged": True,
        "history": [],
        "notes": "Greeting shifted to chief complaint"
    },
    {
        "id": "diverge_03",
        "category": "semantic_shift",
        "partial": "I took some aspirin yesterday",
        "final": "I am allergic to penicillin and aspirin",
        "ground_truth_diverged": True,
        "history": [],
        "notes": "Medication consumption vs drug allergy"
    },

    # Category: Negation & Clinical Modifiers
    {
        "id": "negation_01",
        "category": "negation",
        "partial": "I do have a high fever",
        "final": "I do not have a high fever",
        "ground_truth_diverged": True,
        "history": [],
        "notes": "Direct clinical negation shift"
    },
    {
        "id": "negation_02",
        "category": "negation",
        "partial": "It is on my right foot",
        "final": "Sorry I meant my left foot",
        "ground_truth_diverged": True,
        "history": [],
        "notes": "Anatomical lateral correction"
    },
]


def measure_sequential_stages(slm_engine, text, history, tts_engine=None):
    """
    Measures sequential pipeline execution breakdown:
      - TTFT (Time to First Token)
      - Total generation time
      - Token count & TPS
      - Optional TTS first-chunk latency
    """
    t_start = time.time()
    first_token_time = None
    tokens = []

    stream = slm_engine.generate_stream(text, history)
    for chunk in stream:
        token = chunk.get("token", "")
        if token:
            if first_token_time is None:
                first_token_time = time.time()
            tokens.append(token)
        if chunk.get("done", False):
            break

    t_end = time.time()
    full_response = "".join(tokens).strip()

    ttft_ms = ((first_token_time - t_start) * 1000) if first_token_time else 0.0
    total_slm_ms = (t_end - t_start) * 1000
    token_count = len(tokens)
    tps = token_count / ((t_end - first_token_time) if first_token_time and (t_end > first_token_time) else 0.001)

    tts_first_chunk_ms = None
    if tts_engine and full_response:
        t_tts_start = time.time()
        for _ in tts_engine._synthesize_chunks(full_response):
            tts_first_chunk_ms = (time.time() - t_tts_start) * 1000
            break

    return {
        "response": full_response,
        "token_count": token_count,
        "ttft_ms": ttft_ms,
        "total_slm_ms": total_slm_ms,
        "tps": tps,
        "tts_first_chunk_ms": tts_first_chunk_ms,
    }


def measure_speculative_stages(speculator, partial_text, final_text, history,
                               threshold, simulated_delay_s=1.5, tts_engine=None):
    """
    Measures speculative pipeline execution breakdown:
      - Background early speculation time
      - Divergence check time
      - Effective latency after final transcript arrives
      - Commit vs Rollback outcome
    """
    t_spec_start = time.time()
    speculator.start_speculation(partial_text, history)

    # Simulate patient speaking the remainder of the sentence
    time.sleep(simulated_delay_s)

    t_final_arrives = time.time()

    # Divergence decision
    t_div_start = time.perf_counter()
    div_result = compute_divergence(partial_text, final_text, threshold=threshold)
    div_time_us = (time.perf_counter() - t_div_start) * 1_000_000

    effective_response = ""
    committed = False

    if div_result.diverged:
        speculator.cancel()
        effective_response = speculator.engine.generate(final_text, history)
        committed = False
    else:
        cached_result = speculator.resolve(final_text)
        if cached_result is not None:
            effective_response = cached_result
            committed = True
        else:
            effective_response = speculator.engine.generate(final_text, history)
            committed = False

    t_done = time.time()
    effective_latency_ms = (t_done - t_final_arrives) * 1000

    tts_first_chunk_ms = None
    if tts_engine and effective_response:
        t_tts_start = time.time()
        for _ in tts_engine._synthesize_chunks(effective_response):
            tts_first_chunk_ms = (time.time() - t_tts_start) * 1000
            break

    return {
        "response": effective_response,
        "committed": committed,
        "similarity": div_result.similarity,
        "divergence_time_us": div_time_us,
        "effective_latency_ms": effective_latency_ms,
        "tts_first_chunk_ms": tts_first_chunk_ms,
    }


def run_benchmark_suite(config_path="config.yaml",
                        threshold=0.7,
                        simulated_delay_s=1.5,
                        include_tts=False,
                        category_filter=None,
                        output_dir="eval/results"):
    """Runs the benchmark suite, prints results, and writes CSV/JSON artifacts."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("=" * 70)
    print(" ⚡ SPECULATIVE TRIAGE PIPELINE - COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 70)
    print(f"SLM Model          : {cfg.get('slm', {}).get('model_name')}")
    print(f"Divergence Cutoff  : {threshold}")
    print(f"Simulated Speech Δ : {simulated_delay_s:.1f}s")
    print(f"Include TTS Metric : {include_tts}")
    print("=" * 70)

    slm_engine = SLMEngine(cfg["slm"])
    speculator = Speculator(cfg["slm"])

    tts_engine = None
    if include_tts:
        from src.tts.piper_engine import TTSEngine
        tts_engine = TTSEngine(cfg["tts"])

    # Filter test cases if requested
    cases = BENCHMARK_DATASET
    if category_filter:
        cats = [c.strip().lower() for c in category_filter.split(",")]
        cases = [c for c in cases if c["category"].lower() in cats]

    print(f"\nEvaluating {len(cases)} test cases across categories: "
          f"{sorted(list(set(c['category'] for c in cases)))}\n")

    records = []
    print(f"{'ID':<12} | {'CAT':<11} | {'SEQ (ms)':<9} | {'SPEC (ms)':<9} | {'STATUS':<8} | {'SPEEDUP':<8} | {'SIM':<5}")
    print("-" * 75)

    for i, item in enumerate(cases):
        seq_metrics = measure_sequential_stages(
            slm_engine, item["final"], item["history"], tts_engine=tts_engine
        )

        spec_metrics = measure_speculative_stages(
            speculator, item["partial"], item["final"], item["history"],
            threshold=threshold, simulated_delay_s=simulated_delay_s, tts_engine=tts_engine
        )

        seq_ms = seq_metrics["total_slm_ms"]
        spec_ms = spec_metrics["effective_latency_ms"]
        speedup_pct = ((seq_ms - spec_ms) / seq_ms) * 100 if seq_ms > 0 else 0.0

        status = "COMMIT" if spec_metrics["committed"] else "ROLLBACK"

        # Classification check against ground truth
        is_false_commit = spec_metrics["committed"] and item["ground_truth_diverged"]
        is_true_commit = spec_metrics["committed"] and not item["ground_truth_diverged"]
        is_true_rollback = (not spec_metrics["committed"]) and item["ground_truth_diverged"]

        print(f"{item['id']:<12} | {item['category']:<11} | {seq_ms:>8.0f}  | {spec_ms:>8.0f}  | {status:<8} | {speedup_pct:>6.1f}% | {spec_metrics['similarity']:>4.2f}")

        record = {
            "id": item["id"],
            "category": item["category"],
            "partial": item["partial"],
            "final": item["final"],
            "ground_truth_diverged": item["ground_truth_diverged"],
            "notes": item["notes"],
            "sequential_latency_ms": round(seq_ms, 1),
            "sequential_ttft_ms": round(seq_metrics["ttft_ms"], 1),
            "sequential_tokens": seq_metrics["token_count"],
            "sequential_tps": round(seq_metrics["tps"], 1),
            "speculative_latency_ms": round(spec_ms, 1),
            "speculative_similarity": round(spec_metrics["similarity"], 3),
            "speculative_committed": spec_metrics["committed"],
            "divergence_time_us": round(spec_metrics["divergence_time_us"], 1),
            "speedup_percent": round(speedup_pct, 1),
            "is_false_commit": is_false_commit,
            "is_true_commit": is_true_commit,
            "is_true_rollback": is_true_rollback,
        }
        if include_tts:
            record["seq_tts_first_chunk_ms"] = round(seq_metrics["tts_first_chunk_ms"] or 0, 1)
            record["spec_tts_first_chunk_ms"] = round(spec_metrics["tts_first_chunk_ms"] or 0, 1)

        records.append(record)

    # ---------------------------------------------------------------------------
    # Summary Statistics
    # ---------------------------------------------------------------------------
    seq_latencies = [r["sequential_latency_ms"] for r in records]
    spec_latencies = [r["speculative_latency_ms"] for r in records]
    committed_latencies = [r["speculative_latency_ms"] for r in records if r["speculative_committed"]]
    rollback_latencies = [r["speculative_latency_ms"] for r in records if not r["speculative_committed"]]

    total_commits = sum(1 for r in records if r["speculative_committed"])
    total_rollbacks = len(records) - total_commits
    false_commits = sum(1 for r in records if r["is_false_commit"])
    commit_rate = total_commits / len(records) if records else 0

    mean_seq = statistics.mean(seq_latencies) if seq_latencies else 0
    mean_spec = statistics.mean(spec_latencies) if spec_latencies else 0
    mean_committed = statistics.mean(committed_latencies) if committed_latencies else 0
    mean_rollback = statistics.mean(rollback_latencies) if rollback_latencies else 0
    overall_reduction_pct = ((mean_seq - mean_spec) / mean_seq) * 100 if mean_seq > 0 else 0

    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": cfg.get("slm", {}).get("model_name"),
        "divergence_threshold": threshold,
        "num_cases": len(records),
        "total_commits": total_commits,
        "total_rollbacks": total_rollbacks,
        "commit_rate_percent": round(commit_rate * 100, 1),
        "false_commits": false_commits,
        "mean_sequential_ms": round(mean_seq, 1),
        "median_sequential_ms": round(statistics.median(seq_latencies), 1),
        "mean_speculative_ms": round(mean_spec, 1),
        "median_speculative_ms": round(statistics.median(spec_latencies), 1),
        "mean_committed_ms": round(mean_committed, 1),
        "mean_rollback_ms": round(mean_rollback, 1),
        "overall_latency_reduction_percent": round(overall_reduction_pct, 1),
    }

    print("\n" + "=" * 70)
    print(" 📊  BENCHMARK SUMMARY & PERFORMANCE REPORT")
    print("=" * 70)
    print(f"Total Cases Tested     : {len(records)}")
    print(f"Commit Rate            : {commit_rate*100:.1f}% ({total_commits}/{len(records)})")
    print(f"False Commits (Safety) : {false_commits} {'(PERFECT 0)' if false_commits == 0 else '⚠️ CAUTION'}")
    print("-" * 70)
    print(f"Sequential Latency     : Mean = {mean_seq:.0f}ms | Median = {statistics.median(seq_latencies):.0f}ms")
    print(f"Speculative Latency    : Mean = {mean_spec:.0f}ms | Median = {statistics.median(spec_latencies):.0f}ms")
    if committed_latencies:
        print(f"Committed Speedup      : Mean Latency = {mean_committed:.0f}ms ({((mean_seq - mean_committed)/mean_seq)*100:.1f}% reduction)")
    if rollback_latencies:
        print(f"Rollback Penalty       : Mean Latency = {mean_rollback:.0f}ms (Δ = {mean_rollback - mean_seq:+.0f}ms vs baseline)")
    print(f"Net Overall Speedup    : {overall_reduction_pct:.1f}% reduction across all cases")
    print("=" * 70)

    # ---------------------------------------------------------------------------
    # Export CSV & JSON Artifacts
    # ---------------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(output_dir, f"benchmark_{stamp}.csv")
    json_file = os.path.join(output_dir, f"summary_{stamp}.json")

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n📁 Detailed CSV report exported to: {csv_file}")
    print(f"📁 Summary JSON report exported to: {json_file}")

    return summary, records


def run_threshold_sweep(thresholds=[0.5, 0.6, 0.7, 0.8, 0.9], **kwargs):
    """Runs a parameter sweep over multiple divergence thresholds to find the optimal ROC balance."""
    print("=" * 70)
    print(" 📈  DIVERGENCE THRESHOLD PARAMETER SWEEP")
    print("=" * 70)
    sweep_results = []

    for th in thresholds:
        print(f"\n--- Testing Divergence Threshold: {th} ---")
        summary, _ = run_benchmark_suite(threshold=th, **kwargs)
        sweep_results.append({
            "threshold": th,
            "commit_rate": summary["commit_rate_percent"],
            "false_commits": summary["false_commits"],
            "mean_latency_ms": summary["mean_speculative_ms"],
            "overall_reduction_pct": summary["overall_latency_reduction_percent"],
        })

    print("\n" + "=" * 70)
    print(" 🏆  SWEEP SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Threshold':<11} | {'Commit Rate':<13} | {'False Commits':<15} | {'Mean Latency':<14} | {'Net Reduction':<14}")
    print("-" * 70)
    for res in sweep_results:
        print(f"{res['threshold']:<11.2f} | {res['commit_rate']:>11.1f}% | {res['false_commits']:>13} | {res['mean_latency_ms']:>10.0f} ms | {res['overall_reduction_pct']:>11.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speculative Pipeline Benchmark Suite")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--threshold", type=float, default=0.7, help="Divergence similarity threshold (default: 0.7)")
    parser.add_argument("--sweep", action="store_true", help="Run sensitivity sweep across multiple thresholds")
    parser.add_argument("--include-tts", action="store_true", help="Measure TTS synthesis latency as well")
    parser.add_argument("--cases", default=None, help="Comma-separated category filters (e.g. 'mild,emergency')")
    parser.add_argument("--outdir", default="eval/results", help="Directory to save CSV/JSON outputs")
    parser.add_argument("--delay", type=float, default=1.5, help="Simulated speaking delay in seconds (default: 1.5)")

    args = parser.parse_args()

    if args.sweep:
        run_threshold_sweep(
            config_path=args.config,
            category_filter=args.cases,
            include_tts=args.include_tts,
            output_dir=args.outdir,
            simulated_delay_s=args.delay,
        )
    else:
        run_benchmark_suite(
            config_path=args.config,
            threshold=args.threshold,
            simulated_delay_s=args.delay,
            include_tts=args.include_tts,
            category_filter=args.cases,
            output_dir=args.outdir,
        )
