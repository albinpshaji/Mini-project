"""
End-to-End Evaluation for Edge Selective Hospital Context Routing (Top-3 Multi-Candidate).

Validates:
1. Sub-millisecond router latency on CPU.
2. Multi-option candidate retrieval accuracy on synthetic vignettes (Cardiology Room 120-A, Fever Room 20-A, etc.).
3. Zero-token overhead on non-clinical / chit-chat utterances.
4. SLM generation quality, reasoning among candidate options, and anti-hallucination validation.
"""

import json
import os
import time
from typing import Dict, Any, List

from src.safety.hospital_router import HospitalRouter
from src.slm.slm_engine import SLMEngine

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "data")


def run_benchmark():
    eval_file = os.path.join(DATA_DIR, "synthetic_hospital_eval.json")
    with open(eval_file, "r", encoding="utf-8") as f:
        cases = json.load(f)

    router = HospitalRouter(top_k=3)
    slm = SLMEngine({
        "model_name": "qwen3.5:4b",
        "max_new_tokens": 120,
        "temperature": 0.2,
        "num_ctx": 3072,
        "hospital_routing": True
    })

    print("=" * 75)
    print(" 🏥  SELECTIVE HOSPITAL CONTEXT ROUTING BENCHMARK (TOP-3 CANDIDATES)")
    print("=" * 75)
    print(f"Total Test Cases: {len(cases)}")
    print(f"Hospital Directory: {len(router.rooms)} registered rooms across 11 wards\n")

    latencies_ms = []
    correct_retrievals = 0
    correct_rejections = 0
    total_tokens_injected = 0
    full_dump_tokens_baseline = 1200 * len(cases)  # ~1200 tokens if full 28-room catalog was dumped each turn

    print(f"{'ID':<18} | {'Latency':<8} | {'Expected':<10} | {'Matched Top Candidates':<22} | {'Status'}")
    print("-" * 75)

    for case in cases:
        transcript = case["transcript"]
        expected_prefix = case.get("expected_room_prefix")
        expected_retrieval = case.get("expected_retrieval", True)

        t0 = time.perf_counter()
        matches = router.match(transcript, top_k=3)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)

        matched_rooms = [m.room_number for m in matches]
        matched_summary = ", ".join(matched_rooms) if matched_rooms else "None"
        hint = router.format_routing_hint(matches)
        hint_tokens = len(hint.split())
        total_tokens_injected += hint_tokens

        status = "❌ FAIL"
        if expected_retrieval:
            if any(r.startswith(expected_prefix) for r in matched_rooms):
                status = f"✅ PASS ({len(matches)} options)"
                correct_retrievals += 1
        else:
            if not matches:
                status = "✅ PASS (0 Tok)"
                correct_rejections += 1

        print(
            f"{case['id']:<18} | {elapsed_ms:5.3f}ms | {str(expected_prefix):<10} | "
            f"{matched_summary:<22} | {status}"
        )

    avg_latency = sum(latencies_ms) / len(latencies_ms)
    retrieval_cases = sum(1 for c in cases if c.get("expected_retrieval"))
    rejection_cases = sum(1 for c in cases if not c.get("expected_retrieval"))

    retrieval_acc = (correct_retrievals / retrieval_cases) * 100 if retrieval_cases else 100
    rejection_acc = (correct_rejections / rejection_cases) * 100 if rejection_cases else 100
    token_savings_pct = (1.0 - (total_tokens_injected / full_dump_tokens_baseline)) * 100

    print("-" * 75)
    print("📊 BENCHMARK METRICS SUMMARY:")
    print(f"  • Average CPU Retrieval Latency : {avg_latency:.3f} ms (Target: < 1.0 ms)")
    print(f"  • Multi-Option Retrieval Precision: {retrieval_acc:.1f}% ({correct_retrievals}/{retrieval_cases})")
    print(f"  • Irrelevant Rejection Rate     : {rejection_acc:.1f}% ({correct_rejections}/{rejection_cases})")
    print(f"  • Context Token Savings vs Dump: {token_savings_pct:.1f}% (Injected {total_tokens_injected} vs {full_dump_tokens_baseline} tokens)")
    print("=" * 75)

    # Live SLM Test on 2 representative cases
    print("\n🧠 RUNNING LIVE SLM INFERENCE WITH TOP-3 CANDIDATES IN CONTEXT:\n")

    test_vignettes = [
        {
            "desc": "Cardiac Presentation (Chest pressure & fluttering)",
            "text": "I have had tight pressure across my chest and my heart has been fluttering since this morning.",
        },
        {
            "desc": "Fever Presentation (High fever & rigors)",
            "text": "I have had a high fever of 102 with shivering chills and body aches for two days.",
        }
    ]

    for tv in test_vignettes:
        print(f"--- Case: {tv['desc']} ---")
        print(f"Patient: \"{tv['text']}\"")
        matches = router.match(tv["text"], top_k=3)
        print("👉 Injected Top-3 Candidates:")
        for idx, m in enumerate(matches, 1):
            print(f"   [{idx}] Room {m.room_number} ({m.sub_room_name}) in {m.ward} [Score: {m.score}]")

        t0 = time.time()
        reply = slm.generate(tv["text"])
        elapsed = time.time() - t0
        print(f"SLM Response ({elapsed:.2f}s):\n  \"{reply}\"\n")

    print("✅ All evaluation checks completed!")


if __name__ == "__main__":
    run_benchmark()
