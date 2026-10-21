"""
Evaluation Harness for the Multi-Stage Clinical Invariant Gate.
Tests Innovation 1 from the OmniSpec-Triage Master Plan:
  - Polarity Invariant (Stage 1)
  - Critical Entity Invariant (Stage 2)
  - Lexical Thresholding (Stage 3)
  - Zero-Latency Overhead Verification

Generates structured JSON summary and markdown comparison tables.
"""

import os
import sys
import time
import json
import sqlite3
import argparse
import datetime
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.speculative.divergence_detector import compute_divergence, DivergenceResult


def record_invariant_db(summary: dict, db_path: str = "eval/results/history.db"):
    """Persists invariant gate run metrics to SQLite."""
    try:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS invariant_gate_runs (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT,
                branch TEXT,
                total_cases INTEGER,
                accuracy REAL,
                polarity_intercept_rate REAL,
                critical_entity_intercept_rate REAL,
                safe_commit_rate REAL,
                false_commits INTEGER,
                mean_latency_us REAL,
                summary_json_path TEXT
            )
        """)
        cur.execute("""
            INSERT OR REPLACE INTO invariant_gate_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            summary.get("run_id", f"inv_{int(time.time())}"),
            summary["timestamp"],
            summary["branch"],
            summary["total_cases"],
            summary["accuracy"],
            summary["polarity_intercept_rate"],
            summary["critical_entity_intercept_rate"],
            summary["safe_commit_rate"],
            summary["false_commits"],
            summary["mean_latency_us"],
            summary.get("summary_json_path", "")
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ SQLite history.db notice: {e}")


INVARIANT_TEST_CASES = [
    # --- Category 1: Adversarial Polarity / Negation Invariant Shifts (Must Rollback) ---
    {
        "id": "neg_01",
        "category": "polarity_shift",
        "partial": "I have pain in my chest",
        "final": "I have no pain in my chest",
        "expected_diverged": True,
        "expected_reason": "polarity_mismatch",
        "clinical_significance": "Critical: prevents false cardiac emergency commit on negated symptom"
    },
    {
        "id": "neg_02",
        "category": "polarity_shift",
        "partial": "I am having difficulty breathing",
        "final": "I am having no difficulty breathing",
        "expected_diverged": True,
        "expected_reason": "polarity_mismatch",
        "clinical_significance": "Critical: prevents false respiratory distress commit"
    },
    {
        "id": "neg_03",
        "category": "polarity_shift",
        "partial": "I have a cough and high fever",
        "final": "I have a cough but without any fever",
        "expected_diverged": True,
        "expected_reason": "polarity_mismatch",
        "clinical_significance": "Prevents assuming systemic febrile infection"
    },
    {
        "id": "neg_04",
        "category": "polarity_shift",
        "partial": "my headache has continued all day",
        "final": "my headache has stopped completely",
        "expected_diverged": True,
        "expected_reason": "polarity_mismatch",
        "clinical_significance": "Detects acute symptom resolution"
    },
    {
        "id": "neg_05",
        "category": "polarity_shift",
        "partial": "I do not have any chest discomfort",
        "final": "I now have severe chest discomfort",
        "expected_diverged": True,
        "expected_reason": "polarity_mismatch",
        "clinical_significance": "Detects sudden onset of life-threatening cardiac presentation"
    },
    {
        "id": "neg_06",
        "category": "polarity_shift",
        "partial": "patient reports no breathing trouble",
        "final": "patient has breathing trouble",
        "expected_diverged": True,
        "expected_reason": "polarity_mismatch",
        "clinical_significance": "Detects sudden transition to respiratory compromise"
    },
    {
        "id": "neg_07",
        "category": "polarity_shift",
        "partial": "I have barely any pain left",
        "final": "I have intense pain left",
        "expected_diverged": True,
        "expected_reason": "polarity_mismatch",
        "clinical_significance": "Detects degree modifier flip"
    },
    {
        "id": "neg_08",
        "category": "polarity_shift",
        "partial": "shortness of breath is present",
        "final": "denies shortness of breath",
        "expected_diverged": True,
        "expected_reason": "polarity_mismatch",
        "clinical_significance": "Detects clinical denial token"
    },
    {
        "id": "neg_09",
        "category": "polarity_shift",
        "partial": "child is coughing frequently",
        "final": "child has not stopped coughing",
        "expected_diverged": True,
        "expected_reason": "polarity_mismatch",
        "clinical_significance": "Pediatric status flip: intermittent to intractable coughing"
    },
    {
        "id": "neg_10",
        "category": "polarity_shift",
        "partial": "I took my prescribed medicine",
        "final": "I have not taken my medicine",
        "expected_diverged": True,
        "expected_reason": "polarity_mismatch",
        "clinical_significance": "Medication compliance negation flip"
    },
    {
        "id": "neg_11",
        "category": "polarity_shift",
        "partial": "there is active bleeding from the wound",
        "final": "there is zero bleeding from the wound",
        "expected_diverged": True,
        "expected_reason": "polarity_mismatch",
        "clinical_significance": "Hemorrhage cessation inversion"
    },
    {
        "id": "neg_12",
        "category": "polarity_shift",
        "partial": "the infant is responsive and crying",
        "final": "the infant is hardly responsive",
        "expected_diverged": True,
        "expected_reason": "polarity_mismatch",
        "clinical_significance": "Pediatric emergency: alert status flipped to obtunded"
    },
    {
        "id": "neg_13",
        "category": "polarity_shift",
        "partial": "my vision is blurry",
        "final": "my visual changes resolved completely",
        "expected_diverged": True,
        "expected_reason": "polarity_mismatch",
        "clinical_significance": "Ophthalmic/neurological symptom resolution flip"
    },
    {
        "id": "neg_14",
        "category": "polarity_shift",
        "partial": "I swallowed the chemical cleaner",
        "final": "I did not swallow the chemical cleaner",
        "expected_diverged": True,
        "expected_reason": "polarity_mismatch",
        "clinical_significance": "Toxic ingestion false alarm prevention"
    },
    {
        "id": "neg_15",
        "category": "polarity_shift",
        "partial": "experienced seizure at home",
        "final": "never experienced seizure at home",
        "expected_diverged": True,
        "expected_reason": "polarity_mismatch",
        "clinical_significance": "Neurological history negation flip"
    },

    # --- Category 2: Critical Anatomical & Emergency Keyword Shifts (Must Rollback) ---
    {
        "id": "crit_01",
        "category": "critical_entity_shift",
        "partial": "I have severe pain in my wrist",
        "final": "I have severe pain in my chest",
        "expected_diverged": True,
        "expected_reason": "critical_entity_mismatch",
        "clinical_significance": "Orthopedic complaint mutated to cardiac red-flag"
    },
    {
        "id": "crit_02",
        "category": "critical_entity_shift",
        "partial": "I am coughing and sneezing",
        "final": "I am coughing and spitting blood",
        "expected_diverged": True,
        "expected_reason": "critical_entity_mismatch",
        "clinical_significance": "Mild viral symptoms escalated to hemoptysis"
    },
    {
        "id": "crit_03",
        "category": "critical_entity_shift",
        "partial": "felt a strange twitch in my eye",
        "final": "had a sudden seizure this morning",
        "expected_diverged": True,
        "expected_reason": "critical_entity_mismatch",
        "clinical_significance": "Benign fasciculation mutated to neurological seizure"
    },
    {
        "id": "crit_04",
        "category": "critical_entity_shift",
        "partial": "I feel slight weakness in my hands",
        "final": "sudden facial droop and stroke symptoms",
        "expected_diverged": True,
        "expected_reason": "critical_entity_mismatch",
        "clinical_significance": "Emergent acute ischemic stroke presentation"
    },
    {
        "id": "crit_05",
        "category": "critical_entity_shift",
        "partial": "swelling on my forearm after a bite",
        "final": "swelling in my throat and severe allergy",
        "expected_diverged": True,
        "expected_reason": "critical_entity_mismatch",
        "clinical_significance": "Localized reaction progressed to anaphylaxis"
    },
    {
        "id": "crit_06",
        "category": "critical_entity_shift",
        "partial": "I have a sore neck and back",
        "final": "I have a severe injury to my head",
        "expected_diverged": True,
        "expected_reason": "critical_entity_mismatch",
        "clinical_significance": "Musculoskeletal strain mutated to acute head trauma"
    },
    {
        "id": "crit_07",
        "category": "critical_entity_shift",
        "partial": "feeling tired and drowsy",
        "final": "patient had a sudden collapse in the hallway",
        "expected_diverged": True,
        "expected_reason": "critical_entity_mismatch",
        "clinical_significance": "Fatigue mutated to syncopal collapse"
    },
    {
        "id": "crit_08",
        "category": "critical_entity_shift",
        "partial": "took extra vitamin supplements",
        "final": "suspected accidental opioid overdose",
        "expected_diverged": True,
        "expected_reason": "critical_entity_mismatch",
        "clinical_significance": "Benign ingestion mutated to toxicological overdose"
    },
    {
        "id": "crit_09",
        "category": "critical_entity_shift",
        "partial": "difficulty walking due to knee pain",
        "final": "sudden left side paralysis after waking",
        "expected_diverged": True,
        "expected_reason": "critical_entity_mismatch",
        "clinical_significance": "Orthopedic limitation mutated to hemiplegia"
    },
    {
        "id": "crit_10",
        "category": "critical_entity_shift",
        "partial": "stuffy nose and congestion",
        "final": "swelling causing severe airway obstruction",
        "expected_diverged": True,
        "expected_reason": "critical_entity_mismatch",
        "clinical_significance": "Upper respiratory cold mutated to critical airway emergency"
    },
    {
        "id": "crit_11",
        "category": "critical_entity_shift",
        "partial": "patient fell and bumped their elbow",
        "final": "patient fell and is now unconscious",
        "expected_diverged": True,
        "expected_reason": "critical_entity_mismatch",
        "clinical_significance": "Minor contusion mutated to loss of consciousness"
    },
    {
        "id": "crit_12",
        "category": "critical_entity_shift",
        "partial": "I feel a slight ache in my shoulder",
        "final": "crushing pain radiating across my chest",
        "expected_diverged": True,
        "expected_reason": "critical_entity_mismatch",
        "clinical_significance": "Deltoid ache mutated to acute myocardial infarction"
    },
    {
        "id": "crit_13",
        "category": "critical_entity_shift",
        "partial": "shallow gasping and wheezing",
        "final": "gasping for breath and turning blue",
        "expected_diverged": True,
        "expected_reason": "critical_entity_mismatch",
        "clinical_significance": "Mild wheezing mutated to hypoxic respiratory arrest"
    },
    {
        "id": "crit_14",
        "category": "critical_entity_shift",
        "partial": "I have a rash on my stomach",
        "final": "rash with swelling and severe anaphylaxis",
        "expected_diverged": True,
        "expected_reason": "critical_entity_mismatch",
        "clinical_significance": "Dermatologic rash mutated to systemic anaphylaxis"
    },
    {
        "id": "crit_15",
        "category": "critical_entity_shift",
        "partial": "cut my finger while cooking",
        "final": "deep wound with active bleeding that is serious",
        "expected_diverged": True,
        "expected_reason": "critical_entity_mismatch",
        "clinical_significance": "Minor laceration mutated to severe arterial bleeding"
    },

    # --- Category 3: Benign Matching Prefixes (Safe Speculative Commits) ---
    {
        "id": "safe_01",
        "category": "safe_commit",
        "partial": "I have had a mild fever",
        "final": "I have had a mild fever",
        "expected_diverged": False,
        "expected_reason": "passed",
        "clinical_significance": "Zero-latency commit on exact hypothesis completion"
    },
    {
        "id": "safe_02",
        "category": "safe_commit",
        "partial": "sharp pain in right ankle",
        "final": "sharp pain in right ankle",
        "expected_diverged": False,
        "expected_reason": "passed",
        "clinical_significance": "Zero-latency commit on localized orthopedic complaint"
    },
    {
        "id": "safe_03",
        "category": "safe_commit",
        "partial": "mild headache since yesterday",
        "final": "mild headache since yesterday morning",
        "expected_diverged": False,
        "expected_reason": "passed",
        "clinical_significance": "Benign temporal clause expansion with high lexical overlap"
    },
    {
        "id": "safe_04",
        "category": "safe_commit",
        "partial": "sore throat and runny nose",
        "final": "sore throat and slight runny nose",
        "expected_diverged": False,
        "expected_reason": "passed",
        "clinical_significance": "Mild modifier addition without polarity change"
    },
    {
        "id": "safe_05",
        "category": "safe_commit",
        "partial": "I fell down the stairs",
        "final": "I fell down the stairs earlier",
        "expected_diverged": False,
        "expected_reason": "passed",
        "clinical_significance": "Safe commit on trauma narrative"
    },
    {
        "id": "safe_06",
        "category": "safe_commit",
        "partial": "rash on both elbows",
        "final": "rash on both elbows and knees",
        "expected_diverged": False,
        "expected_reason": "passed",
        "clinical_significance": "Anatomical clause expansion with high lexical overlap"
    },
    {
        "id": "safe_07",
        "category": "safe_commit",
        "partial": "burning sensation when urinating",
        "final": "burning sensation when urinating today",
        "expected_diverged": False,
        "expected_reason": "passed",
        "clinical_significance": "Dysuria complaint with temporal qualifier completion"
    },
    {
        "id": "safe_08",
        "category": "safe_commit",
        "partial": "twisted knee while playing tennis",
        "final": "twisted right knee while playing tennis",
        "expected_diverged": False,
        "expected_reason": "passed",
        "clinical_significance": "Sports injury narrative with lateral specificity"
    },
    {
        "id": "safe_09",
        "category": "safe_commit",
        "partial": "child has an earache since yesterday",
        "final": "child has an earache since yesterday morning",
        "expected_diverged": False,
        "expected_reason": "passed",
        "clinical_significance": "Pediatric otalgia narrative with temporal expansion"
    },
    {
        "id": "safe_10",
        "category": "safe_commit",
        "partial": "dry cough and stuffy nose",
        "final": "dry cough and stuffy nose today",
        "expected_diverged": False,
        "expected_reason": "passed",
        "clinical_significance": "Upper respiratory viral presentation hypothesis completion"
    },

    # --- Category 4: Low-Similarity Divergences (Standard Lexical Rollback) ---
    {
        "id": "lex_01",
        "category": "low_similarity",
        "partial": "I took the bus to the hospital",
        "final": "My knee has been swelling up all week",
        "expected_diverged": True,
        "expected_reason": "low_similarity",
        "clinical_significance": "Non-clinical partial replaced by clinical complaint"
    },
    {
        "id": "lex_02",
        "category": "low_similarity",
        "partial": "Is the doctor available",
        "final": "I am experiencing severe abdominal cramps",
        "expected_diverged": True,
        "expected_reason": "low_similarity",
        "clinical_significance": "Logistical question replaced by acute pain complaint"
    },
    {
        "id": "lex_03",
        "category": "low_similarity",
        "partial": "Can I get some water please",
        "final": "My stomach has been hurting since yesterday",
        "expected_diverged": True,
        "expected_reason": "low_similarity",
        "clinical_significance": "Thirst request replaced by gastrointestinal complaint"
    },
    {
        "id": "lex_04",
        "category": "low_similarity",
        "partial": "Where is the registration desk located",
        "final": "I twisted my ankle playing soccer in the park",
        "expected_diverged": True,
        "expected_reason": "low_similarity",
        "clinical_significance": "Wayfinding inquiry replaced by musculoskeletal injury"
    },
    {
        "id": "lex_05",
        "category": "low_similarity",
        "partial": "I parked my car outside",
        "final": "My right index finger is swollen and painful",
        "expected_diverged": True,
        "expected_reason": "low_similarity",
        "clinical_significance": "Parking comment replaced by acute localized digital pain"
    },
]


def run_evaluation(branch_tag: str = "current") -> Dict[str, Any]:
    print("=" * 90)
    print(f" 🛡️  EVALUATING CLINICAL INVARIANT GATE: BRANCH = {branch_tag.upper()}")
    print("=" * 90)
    print(f"{'ID':<8} | {'CATEGORY':<23} | {'SIM':<6} | {'DIV':<6} | {'REASON':<24} | {'STATUS'}")
    print("-" * 90)

    total_cases = len(INVARIANT_TEST_CASES)
    polarity_cases = [c for c in INVARIANT_TEST_CASES if c["category"] == "polarity_shift"]
    critical_cases = [c for c in INVARIANT_TEST_CASES if c["category"] == "critical_entity_shift"]
    safe_cases = [c for c in INVARIANT_TEST_CASES if c["category"] == "safe_commit"]
    lexical_cases = [c for c in INVARIANT_TEST_CASES if c["category"] == "low_similarity"]

    correct_predictions = 0
    polarity_intercepted = 0
    critical_intercepted = 0
    safe_committed = 0
    false_commits = 0
    latencies_us = []

    results = []

    for c in INVARIANT_TEST_CASES:
        t0 = time.perf_counter_ns()
        res = compute_divergence(c["partial"], c["final"], threshold=0.70)
        t_elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
        latencies_us.append(t_elapsed_us)

        is_correct = (res.diverged == c["expected_diverged"])
        if c["expected_reason"]:
            is_correct = is_correct and (res.reason == c["expected_reason"])

        if is_correct:
            correct_predictions += 1
            status_str = "✅ PASS"
        else:
            status_str = "❌ FAIL"

        # Check for false commits (committing when we should have diverged)
        if c["expected_diverged"] and not res.diverged:
            false_commits += 1

        if c["category"] == "polarity_shift" and res.diverged and res.polarity_shifted:
            polarity_intercepted += 1

        if c["category"] == "critical_entity_shift" and res.diverged and res.entity_shifted:
            critical_intercepted += 1

        if c["category"] == "safe_commit" and not res.diverged:
            safe_committed += 1

        print(f"{c['id']:<8} | {c['category']:<23} | {res.similarity:.3f} | {str(res.diverged):<6} | {res.reason:<24} | {status_str}")

        results.append({
            "id": c["id"],
            "category": c["category"],
            "partial": c["partial"],
            "final": c["final"],
            "similarity": res.similarity,
            "diverged": res.diverged,
            "reason": res.reason,
            "passed": is_correct,
            "latency_us": t_elapsed_us,
        })

    polarity_recall = (polarity_intercepted / len(polarity_cases)) * 100.0
    critical_recall = (critical_intercepted / len(critical_cases)) * 100.0
    safe_commit_rate = (safe_committed / len(safe_cases)) * 100.0
    accuracy = (correct_predictions / total_cases) * 100.0
    mean_latency_us = sum(latencies_us) / len(latencies_us)

    summary = {
        "branch": branch_tag,
        "timestamp": datetime.datetime.now().isoformat(),
        "total_cases": total_cases,
        "accuracy": accuracy,
        "polarity_intercept_rate": polarity_recall,
        "critical_entity_intercept_rate": critical_recall,
        "safe_commit_rate": safe_commit_rate,
        "false_commits": false_commits,
        "mean_latency_us": round(mean_latency_us, 2),
        "mean_latency_ms": round(mean_latency_us / 1000.0, 4),
        "results": results,
    }

    print("-" * 90)
    print(f"📊 SUMMARY FOR {branch_tag.upper()}:")
    print(f"  • Overall Gate Accuracy           : {accuracy:.1f}% ({correct_predictions}/{total_cases})")
    print(f"  • Polarity Invariant Recall (Neg) : {polarity_recall:.1f}% ({polarity_intercepted}/{len(polarity_cases)})")
    print(f"  • Critical Entity Recall          : {critical_recall:.1f}% ({critical_intercepted}/{len(critical_cases)})")
    print(f"  • Safe Speculative Commit Rate    : {safe_commit_rate:.1f}% ({safe_committed}/{len(safe_cases)})")
    print(f"  • False Commits (Life Threats)    : {false_commits} (Target: 0)")
    print(f"  • Mean Decision Latency           : {mean_latency_us:.2f} µs ({mean_latency_us / 1000.0:.4f} ms)")
    print("=" * 90)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default="feature/grammar-constrained-decoding", help="Branch identifier")
    parser.add_argument("--save", default=None, help="Path to save output JSON")
    args = parser.parse_args()

    summary_data = run_evaluation(branch_tag=args.branch)

    clean_branch = args.branch.replace("/", "_").replace("-", "_")
    out_dir = "eval/results/runs"
    os.makedirs(out_dir, exist_ok=True)
    out_file = args.save if args.save else os.path.join(out_dir, f"{clean_branch}_invariant_gate_summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved evaluation results to {out_file}")

    summary_data["summary_json_path"] = out_file
    record_invariant_db(summary_data)
