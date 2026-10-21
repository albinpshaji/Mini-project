"""
Master One-Command Benchmark & Comparative Evaluation Suite.
Runs Sequential Baseline vs Speculative Candidate, logs to SQLite DB,
and auto-generates the full thesis comparison report.

Usage:
  python -m eval.runners.run_all_evals --cases 10
"""

import os
import sys
import argparse
import datetime

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from eval.runners.run_pipeline_latency import run_pipeline_benchmark
from eval.core.regression_detector import (
    compare_runs_statistically,
    evaluate_regression_gates,
    generate_markdown_comparison_report
)


def main():
    parser = argparse.ArgumentParser(description="Master Evaluation Suite")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--dataset", default=None, help="Path to clinical dataset")
    parser.add_argument("--diverse", action="store_true", help="Run expanded 120-vignette diverse clinical benchmark suite")
    parser.add_argument("--cases", type=int, default=10, help="Number of clinical cases to benchmark")
    parser.add_argument("--delay", type=float, default=1.5, help="Simulated speaking delay in seconds")
    parser.add_argument("--tag", default="eval", help="Evaluation campaign tag")
    args = parser.parse_args()

    dataset_path = args.dataset
    if not dataset_path:
        if args.diverse:
            dataset_path = "eval/data/clinical_vignettes_120_diverse.json"
        else:
            dataset_path = "eval/data/clinical_vignettes_60.json"

    print("\n" + "=" * 80)
    print(" 🚀 STARTING OMNISPEC-TRIAGE MASTER BENCHMARK SUITE")
    print(f" Timestamp: {datetime.datetime.now().isoformat()} | Cases: {args.cases}")
    print(f" Dataset  : {dataset_path}")
    print("=" * 80)

    # 1. Run Sequential Baseline
    print("\n>>> STEP 1/2: Executing Sequential Baseline Pipeline...")
    baseline_summary = run_pipeline_benchmark(
        config_path=args.config,
        dataset_path=dataset_path,
        mode="sequential",
        max_cases=args.cases,
        tag=f"{args.tag}_baseline",
        simulated_delay_s=args.delay
    )

    # 2. Run Speculative Pipeline
    print("\n>>> STEP 2/2: Executing Speculative Pipeline...")
    candidate_summary = run_pipeline_benchmark(
        config_path=args.config,
        dataset_path=dataset_path,
        mode="speculative",
        max_cases=args.cases,
        tag=f"{args.tag}_speculative",
        simulated_delay_s=args.delay
    )

    # 3. Statistical Analysis & Report Generation
    lat_b = baseline_summary.get("latency", {}).get("raw_latencies_ms", [])
    lat_c = candidate_summary.get("latency", {}).get("raw_latencies_ms", [])
    stat_res = compare_runs_statistically(lat_b, lat_c)
    passed, violations = evaluate_regression_gates(baseline_summary, candidate_summary)

    report_md = generate_markdown_comparison_report(baseline_summary, candidate_summary)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = "eval/results/comparisons"
    os.makedirs(out_dir, exist_ok=True)
    report_file = os.path.join(out_dir, f"master_comparison_{stamp}.md")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\n" + "=" * 80)
    print(" 📊 MASTER COMPARATIVE EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Sequential Glass-to-Glass : {stat_res['baseline_mean_ms']} ms (Mean)")
    print(f"Speculative Glass-to-Glass: {stat_res['candidate_mean_ms']} ms (Mean)")
    print(f"Net Speedup Achieved      : {stat_res['speedup_percent']:+.1f}% ({baseline_summary.get('latency', {}).get('glass_to_glass_p50_ms', 1)/max(1, candidate_summary.get('latency', {}).get('glass_to_glass_p50_ms', 1)):.1f}x)")
    print(f"Statistical Significance  : p = {stat_res['p_value']} ({'Significant' if stat_res['statistically_significant'] else 'Preliminary'})")
    print(f"Speculative Commit Rate   : {candidate_summary.get('speculative_efficiency', {}).get('commit_rate_percent', 0.0):.1f}%")
    print(f"Clinical Under-Triage     : {candidate_summary.get('clinical_safety', {}).get('under_triage_rate_percent', 0.0)}% (Target: 0.00%)")
    print(f"Peak VRAM Consumption     : {candidate_summary.get('hardware', {}).get('peak_vram_mb', 0.0)} MB")
    print("-" * 80)
    print(f"📄 Comparative Report Saved : {report_file}")
    print(f"💾 Persistent Database      : eval/results/history.db")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
