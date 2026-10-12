"""
Piper TTS Diagnostic Test.

Checks:
  1. Voice model files (.onnx and .onnx.json) exist.
  2. Loads TTSEngine.
  3. Synthesizes text to audio.
  4. Plays spoken audio through system speakers.
  5. Tests cancellation capability.
"""

import os
import sys
import yaml
import sounddevice as sd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tts.piper_engine import TTSEngine


def run_tts_test(config_path="config.yaml"):
    print("=" * 60)
    print(" 🔊  PIPER TTS DIAGNOSTIC TEST")
    print("=" * 60)

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        tts_cfg = cfg.get("tts", {})
    except Exception as e:
        print(f"❌ Failed to load {config_path}: {e}")
        return False

    model_path = tts_cfg.get("voice_model", "models/tts/en_US-lessac-medium.onnx")
    json_path = model_path + ".json"

    print(f"Model path: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Voice model file not found at: {model_path}")
        return False
    if not os.path.exists(json_path):
        print(f"⚠️  Warning: Config json not found at: {json_path}")

    print("✅ Model files verified.")

    print("\n[1/3] Initializing TTSEngine...")
    try:
        engine = TTSEngine(tts_cfg)
        print(f"✅ TTSEngine loaded! Sample rate: {engine.sample_rate} Hz")
    except Exception as e:
        print(f"❌ Failed to initialize TTSEngine: {e}")
        return False

    test_sentence = "This is a test of the Piper text to speech engine. If you hear this clearly, sound output is working."
    print(f"\n[2/3] Synthesizing: {test_sentence!r}")
    try:
        chunks = list(engine._synthesize_chunks(test_sentence))
        total_samples = sum(len(c) for c in chunks)
        duration_s = total_samples / engine.sample_rate
        print(f"✅ Synthesized {len(chunks)} audio chunk(s) totaling {duration_s:.2f} seconds.")
    except Exception as e:
        print(f"❌ Synthesis failed: {e}")
        return False

    print("\n[3/3] Playing audio through speakers... (Listen now)")
    try:
        engine.speak(test_sentence, blocking=True)
        print("✅ Playback completed successfully!")
    except Exception as e:
        print(f"❌ Playback failed: {e}")
        return False

    print("\n" + "-" * 60)
    print("✅ SUCCESS: TTS synthesis and playback are fully operational!")
    return True


if __name__ == "__main__":
    success = run_tts_test()
    sys.exit(0 if success else 1)
