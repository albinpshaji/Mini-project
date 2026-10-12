"""
Master Diagnostics Runner for Triage Agent.

Runs individual component diagnostics and prints a consolidated health report:
  1. Microphone & VAD Capture
  2. Safety Guardrails & Red-Flag Rules
  3. Small Language Model (Ollama) & Slot Extraction
  4. Piper Text-to-Speech Engine
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_safety import run_safety_test
from tests.test_mic import run_mic_test
from tests.test_slm import run_slm_test
from tests.test_tts import run_tts_test


def main():
    print("=" * 65)
    print(" 🏥  SPECULATIVE TRIAGE AGENT - COMPONENT DIAGNOSTICS SUITE")
    print("=" * 65)
    print("This suite tests each pipeline module independently.\n")

    results = {}

    # 1. Safety Layer (fast, unit tests)
    print("\n>>> TEST 1: Clinical Safety & Guardrails")
    results["Safety Layer"] = run_safety_test()

    # 2. SLM & Extraction
    print("\n>>> TEST 2: Ollama SLM & Slot Extraction")
    results["Ollama SLM"] = run_slm_test()

    # 3. Audio / Microphone
    print("\n>>> TEST 3: Microphone & Voice Activity Detection")
    results["Microphone"] = run_mic_test(duration_s=3)

    # 4. Text to Speech
    print("\n>>> TEST 4: Piper TTS Audio Playback")
    results["Piper TTS"] = run_tts_test()

    # Summary Report
    print("\n" + "=" * 65)
    print(" 📊  FINAL DIAGNOSTIC REPORT")
    print("=" * 65)
    all_ok = True
    for component, passed in results.items():
        icon = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_ok = False
        print(f"  {icon} : {component}")

    print("=" * 65)
    if all_ok:
        print("🎉 ALL SYSTEMS GO! The agent is fully ready to run.")
        print("   Start the live agent with: python -m src.pipeline")
    else:
        print("⚠️  One or more components failed. Check the details above.")
    print("=" * 65)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
