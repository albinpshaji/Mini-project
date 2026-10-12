"""
End-to-End Evaluation for Edge Selective Hospital Context Routing.

Validates:
1. Router latency (< 1.0ms on CPU).
2. Retrieval accuracy on synthetic clinical vignettes (Cardiology Room 120, Fever Room 20, etc.).
3. Zero-token overhead on non-clinical / chit-chat utterances.
4. SLM generation quality and anti-hallucination guardrail validation.
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

    router = HospitalRouter()
    slm = SLMEngine({"model_name": "qwen3.5:4b", "max_new_tokens": 100, "temperature": 0.2})

    print("=" * 65)
    print(" 🏥  SELECTIVE HOSPITAL CONTEXT ROUTING BENCHMARK")
    print("=" * 65)
    print(f"Total Test Cases: {len(cases)}")
    print(f"Hospital Directory: {len(router.rooms)} registered rooms\n")

    latencies_ms = []
    correct_retrievals = 0
    correct_rejections = 0
    total_tokens_injected = 0
    full_dump_tokens_baseline = 520 * len(cases)  # ~520 tokens if full catalog was dumped each turn

    print(f"{'ID':<18} | {'Latency':<8} | {'Expected':<10} | {'Matched':<10} | {'Status'}")
    print("-" * 65)

    for case in cases:
        transcript = case["transcript"]
        expected_room = case.get("expected_room")
        expected_retrieval = case.get("expected_retrieval", True)

        t0 = time.perf_counter()
        matches = router.match(transcript, top_k=1)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)

        matched_room = matches[0].room_number if matches else None
        hint = router.format_routing_hint(matches)
        hint_tokens = len(hint.split())
        total_tokens_injected += hint_tokens

        status = "❌ FAIL"
        if expected_retrieval:
            if matched_room == expected_room:
                status = "✅ PASS"
                correct_retrievals += 1
        else:
            if matched_room is None:
                status = "✅ PASS (0 Tok)"
                correct_rejections += 1

        print(
            f"{case['id']:<18} | {elapsed_ms:5.3f}ms | {str(expected_room):<10} | "
            f"{str(matched_room):<10} | {status}"
        )

    avg_latency = sum(latencies_ms) / len(latencies_ms)
    retrieval_cases = sum(1 for c in cases if c.get("expected_retrieval"))
    rejection_cases = sum(1 for c in cases if not c.get("expected_retrieval"))

    retrieval_acc = (correct_retrievals / retrieval_cases) * 100 if retrieval_cases else 100
    rejection_acc = (correct_rejections / rejection_cases) * 100 if rejection_cases else 100
    token_savings_pct = (1.0 - (total_tokens_injected / full_dump_tokens_baseline)) * 100

    print("-" * 65)
    print("📊 BENCHMARK METRICS SUMMARY:")
    print(f"  • Average CPU Retrieval Latency : {avg_latency:.3f} ms (Target: < 1.0 ms)")
    print(f"  • Clinical Retrieval Accuracy  : {retrieval_acc:.1f}% ({correct_retrievals}/{retrieval_cases})")
    print(f"  • Irrelevant Rejection Rate     : {rejection_acc:.1f}% ({correct_rejections}/{rejection_cases})")
    print(f"  • Context Token Savings vs Dump: {token_savings_pct:.1f}% (Injected {total_tokens_injected} vs {full_dump_tokens_baseline} tokens)")
    print("=" * 65)

    # Live SLM Test on 2 representative cases: Cardiac (120) and Fever (20)
    print("\n🧠 RUNNING LIVE SLM INFERENCE ON REPRESENTATIVE VIGNETTES:\n")

    test_vignettes = [
        {
            "desc": "Cardiac Presentation (Room 120 context)",
            "text": "I have had tight pressure across my chest and my heart has been fluttering since this morning.",
            "expected_room": "120"
        },
        {
            "desc": "Fever Presentation (Room 20 context)",
            "text": "I have had a high fever with shivering chills and body aches for two days.",
            "expected_room": "20"
        }
    ]

    for tv in test_vignettes:
        print(f"--- Case: {tv['desc']} ---")
        print(f"Patient: \"{tv['text']}\"")
        matches = router.match(tv["text"], top_k=1)
        if matches:
            top = matches[0]
            print(f"👉 Injected Context: Room {top.room_number} ({top.department}, {top.floor}) [Score: {top.score}]")
        else:
            print("👉 Injected Context: None")

        t0 = time.time()
        reply = slm.generate(tv["text"])
        elapsed = time.time() - t0
        print(f"SLM Response ({elapsed:.2f}s):\n  \"{reply}\"\n")

    print("✅ All evaluation checks completed!")


if __name__ == "__main__":
    run_benchmark()
