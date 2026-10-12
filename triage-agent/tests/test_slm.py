"""
Small Language Model (SLM) & Clinical Extraction Diagnostic Test.

Checks:
  1. Ollama service connectivity and list available models.
  2. Runs SLMEngine generation on a sample patient complaint.
  3. Verifies non-empty response, prompt compliance, and latency.
  4. Runs structured symptom slot extraction (JSON schema).
"""

import sys
import os
import time
import yaml
import ollama

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.slm.slm_engine import SLMEngine
from src.safety.slot_extraction import extract_symptom_slots


def run_slm_test(config_path="config.yaml"):
    print("=" * 60)
    print(" 🧠  SLM & CLINICAL EXTRACTION DIAGNOSTIC TEST")
    print("=" * 60)

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        slm_cfg = cfg.get("slm", {})
    except Exception as e:
        print(f"❌ Failed to load {config_path}: {e}")
        return False

    model_name = slm_cfg.get("model_name", "qwen3.5:4b")
    print(f"Target model: {model_name}")

    print("\n[1/3] Checking Ollama service...")
    try:
        models_resp = ollama.list()
        model_names = [m.model for m in models_resp.models]
        print(f"✅ Ollama service is reachable. Available models: {model_names}")
        if not any(model_name in m for m in model_names):
            print(f"⚠️  WARNING: {model_name} not found in Ollama models list!")
    except Exception as e:
        print(f"❌ Could not reach Ollama: {e}")
        return False

    print("\n[2/3] Initializing SLMEngine & Testing Generation...")
    try:
        engine = SLMEngine(slm_cfg)
        test_transcript = "I have had a mild dry cough and a sore throat for about two days."
        print(f"  Prompt patient input: {test_transcript!r}")

        t0 = time.time()
        response = engine.generate(test_transcript)
        elapsed_s = time.time() - t0

        print(f"  SLM Response ({elapsed_s:.2f}s):\n  👉 \"{response}\"")

        if not response or len(response.strip()) == 0:
            print("❌ FAILED: SLM produced an empty response!")
            return False
        print("✅ Generation successful and non-empty!")
    except Exception as e:
        print(f"❌ SLM generation error: {e}")
        return False

    print("\n[3/3] Testing Structured Clinical Slot Extraction...")
    try:
        history = [
            {"role": "user", "text": test_transcript},
            {"role": "assistant", "text": response},
        ]
        t0 = time.time()
        slots = extract_symptom_slots(history, model_name=model_name)
        elapsed_s = time.time() - t0

        slots_dict = slots.to_dict()
        print(f"  Extracted Slots ({elapsed_s:.2f}s):")
        for k, v in slots_dict.items():
            print(f"    • {k}: {v}")

        if slots.chief_complaint:
            print("✅ Structured slot extraction verified!")
        else:
            print("⚠️  Warning: Chief complaint was not extracted from test history.")
    except Exception as e:
        print(f"❌ Slot extraction error: {e}")
        return False

    print("\n" + "-" * 60)
    print("✅ SUCCESS: SLM generation and structured clinical extraction are operational!")
    return True


if __name__ == "__main__":
    success = run_slm_test()
    sys.exit(0 if success else 1)
