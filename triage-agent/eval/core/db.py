"""
SQLite Database Storage Manager for Benchmark Runs.
Maintains persistent historical evaluation logs in eval/results/history.db.
"""

import os
import sqlite3
import datetime
from typing import Dict, Any, List, Optional

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "history.db"))


def get_db_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the database schema if not present."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            run_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            git_commit TEXT,
            architecture TEXT NOT NULL,
            model_name TEXT NOT NULL,
            num_cases INTEGER,
            
            -- Stage-by-Stage Latencies (ms)
            vad_wait_mean_ms REAL,
            asr_final_mean_ms REAL,
            slm_prefill_mean_ms REAL,
            slm_gen_mean_ms REAL,
            tools_guard_mean_ms REAL,
            tts_synthesis_mean_ms REAL,
            
            -- Glass-to-Glass Total Latency (ms)
            g2g_mean_ms REAL,
            g2g_p50_ms REAL,
            g2g_p90_ms REAL,
            g2g_p99_ms REAL,
            
            -- Speculative Metrics
            commit_rate_pct REAL,
            false_commits INTEGER,
            speedup_factor REAL,
            
            -- Clinical Acuity & Safety
            ktas_kappa REAL,
            under_triage_pct REAL,
            over_triage_pct REAL,
            dosage_leakage_pct REAL,
            
            -- Edge Hardware
            peak_vram_mb REAL,
            
            -- File Paths
            summary_json_path TEXT,
            raw_csv_path TEXT
        )
    """)
    conn.commit()
    conn.close()


def record_run(summary: Dict[str, Any]) -> str:
    """Inserts or replaces a benchmark run in the SQLite database."""
    init_db()
    conn = get_db_connection()
    cur = conn.cursor()

    stages = summary.get("stages_ms", {})
    latency = summary.get("latency", {})
    spec = summary.get("speculative_efficiency", {})
    clinical = summary.get("clinical_safety", {})
    hardware = summary.get("hardware", {})

    run_id = summary.get("run_id") or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    cur.execute("""
        INSERT OR REPLACE INTO benchmark_runs (
            run_id, timestamp, git_commit, architecture, model_name, num_cases,
            vad_wait_mean_ms, asr_final_mean_ms, slm_prefill_mean_ms, slm_gen_mean_ms,
            tools_guard_mean_ms, tts_synthesis_mean_ms,
            g2g_mean_ms, g2g_p50_ms, g2g_p90_ms, g2g_p99_ms,
            commit_rate_pct, false_commits, speedup_factor,
            ktas_kappa, under_triage_pct, over_triage_pct, dosage_leakage_pct,
            peak_vram_mb, summary_json_path, raw_csv_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        summary.get("timestamp", datetime.datetime.now().isoformat()),
        summary.get("git_commit", "unknown"),
        summary.get("architecture", "unspecified"),
        summary.get("model", "qwen3.5:4b"),
        summary.get("cases_evaluated", 0),

        stages.get("1_vad_silence_wait", {}).get("mean", 0.0),
        stages.get("2_final_asr_pass", {}).get("mean", 0.0),
        stages.get("3_slm_prefill", {}).get("mean", 0.0),
        stages.get("4_slm_generation", {}).get("mean", 0.0),
        stages.get("5_tools_guardrail", {}).get("mean", 0.0),
        stages.get("6_tts_synthesis", {}).get("mean", 0.0),

        latency.get("glass_to_glass_mean_ms", 0.0),
        latency.get("glass_to_glass_p50_ms", 0.0),
        latency.get("glass_to_glass_p90_ms", 0.0),
        latency.get("glass_to_glass_p99_ms", 0.0),

        spec.get("commit_rate_percent", 0.0),
        spec.get("false_commit_count", 0),
        spec.get("effective_speedup_factor", 1.0),

        clinical.get("ktas_quadratic_weighted_kappa", 0.0),
        clinical.get("under_triage_rate_percent", 0.0),
        clinical.get("over_triage_rate_percent", 0.0),
        clinical.get("dosage_leakage_rate_percent", 0.0),

        hardware.get("peak_vram_mb", 0.0),
        summary.get("summary_json_path", ""),
        summary.get("raw_csv_path", "")
    ))
    conn.commit()
    conn.close()
    return run_id


def get_all_runs() -> List[Dict[str, Any]]:
    """Returns all historical runs ordered by timestamp."""
    init_db()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM benchmark_runs ORDER BY timestamp DESC")
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def get_run_by_id(run_id: str) -> Optional[Dict[str, Any]]:
    """Finds a run by its unique run_id."""
    init_db()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM benchmark_runs WHERE run_id = ?", (run_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None
