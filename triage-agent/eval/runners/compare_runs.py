"""
Run Comparison & Delta Evaluation Tool.
Compares any two evaluation runs, displays a side-by-side terminal diff,
and exports an auto-generated publication report in eval/results/comparisons/.
"""

import os
import sys
import json
import argparse
import datetime

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from eval.core.regression_detector import (
    compare_runs_statistically,
    evaluate_regression_gates,
    generate_markdown_comparison_report
)
from eval.core.db import get_run_by_id


def load_run_summary(identifier: str) -> dict:
    # Check if identifier is a path to a JSON file
    if os.path.isfile(identifier):
        with open(identifier, "r", encoding="utf-8") as f:
            return json.load(f)

    # Otherwise query SQLite history.db
    db_row = get_run_by_id(identifier)
    if db_row and db_row.get("summary_json_path") and os.path.isfile(db_row["summary_json_path"]):
        with open(db_row["summary_json_path"], "r", encoding="utf-8") as f:
            return json.load(f)

    raise FileNotFoundError(f"Could not find run summary for: '{identifier}'. Pass a valid file path or run_id.")


def main():
    parser = argparse.ArgumentParser(description="Compare two benchmark runs")
    parser.add_argument("--baseline", required=True, help="Path to baseline summary.json or run_id")
    parser.add_argument("--candidate", required=True, help="Path to candidate summary.json or run_id")
    parser.add_argument("--outdir", default="eval/results/comparisons", help="Output directory for Markdown report")
    args = parser.parse_args()

    baseline = load_run_summary(args.baseline)
    candidate = load_run_summary(args.candidate)

    # Statistical test
    lat_b = baseline.get("latency", {}).get("raw_latencies_ms", [])
    lat_c = candidate.get("latency", {}).get("raw_latencies_ms", [])
    stat_res = compare_runs_statistically(lat_b, lat_c)

    # Gates check
    passed, violations = evaluate_regression_gates(baseline, candidate)

    # Generate Markdown Report
    report_md = generate_markdown_comparison_report(baseline, candidate)

    os.makedirs(args.outdir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(args.outdir, f"comparison_{stamp}.md")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\n" + "=" * 80)
    print(" ⚖️  OMNISPEC-TRIAGE COMPARATIVE EVALUATION RESULTS")
    print("=" * 80)
    print(f"Baseline Architecture  : {baseline.get('architecture')}")
    print(f"Candidate Architecture : {candidate.get('architecture')}")
    print("-" * 80)
    print(f"Baseline Glass-to-Glass: {stat_res['baseline_mean_ms']} ms (Mean)")
    print(f"Candidate Glass-to-Glass: {stat_res['candidate_mean_ms']} ms (Mean)")
    print(f"Speedup Achieved       : {stat_res['speedup_percent']:+.1f}% ({baseline.get('latency', {}).get('glass_to_glass_p50_ms', 1)/max(1, candidate.get('latency', {}).get('glass_to_glass_p50_ms', 1)):.1f}x)")
    print(f"Statistical p-value    : {stat_res['p_value']} ({'Significant p < 0.05' if stat_res['statistically_significant'] else 'Not Significant'})")
    print(f"KTAS Kappa Delta       : {candidate.get('clinical_safety', {}).get('ktas_quadratic_weighted_kappa', 0) - baseline.get('clinical_safety', {}).get('ktas_quadratic_weighted_kappa', 0):+.3f}")
    print(f"Under-Triage Rate      : {candidate.get('clinical_safety', {}).get('under_triage_rate_percent', 0.0)}% (Target: 0.00%)")
    print("-" * 80)
    if passed:
        print("🏆 VERDICT: APPROVED (All safety and latency regression gates passed)")
    else:
        print("⚠️ VERDICT: REGRESSION DETECTED")
        for v in violations:
            print(f"   ❌ {v}")
    print("=" * 80)
    print(f"📄 Full Comparative Markdown Report saved to: {out_file}\n")


if __name__ == "__main__":
    main()
