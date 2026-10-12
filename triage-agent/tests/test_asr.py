"""
Streaming ASR (Whisper) Diagnostic Test.

Checks:
  1. Loads faster-whisper models (tiny & base).
  2. Initializes StreamingASR with microphone input.
  3. Records live for 5 seconds and captures speech events (partial & final).
"""

import sys
import os
import time
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.asr.streaming_asr import StreamingASR


def run_asr_test(duration_s=5, config_path="config.yaml"):
    print("=" * 60)
    print(" 🎙️  STREAMING ASR (FASTER-WHISPER) DIAGNOSTIC TEST")
    print("=" * 60)

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        asr_cfg = cfg.get("asr", {})
    except Exception as e:
        print(f"❌ Failed to load {config_path}: {e}")
        return False

    print(f"ASR Configuration: model={asr_cfg.get('model_size')}, final_model={asr_cfg.get('final_model_size')}")
    print(f"Input device: {asr_cfg.get('input_device', 'System Default')}")

    events = []
    def on_event(event):
        events.append(event)
        event_type = event.get("type", "unknown").upper()
        text = event.get("text", "")
        print(f"  👉 [{event_type}] \"{text}\"")

    print("\n[1/2] Initializing Streaming ASR models...")
    try:
        asr = StreamingASR(asr_cfg, on_event=on_event)
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize StreamingASR: {e}")
        return False

    print(f"\n[2/2] Starting ASR stream for {duration_s} seconds...")
    print("  👉 PLEASE SPEAK A SENTENCE INTO YOUR MICROPHONE NOW (e.g. 'I have a mild fever'):\n")

    try:
        asr.start()
        for i in range(duration_s, 0, -1):
            time.sleep(1)
            print(f"  Listening... ({i}s left)", end="\r", flush=True)
        asr.stop()
        print("  Finished listening!                 ")
    except Exception as e:
        print(f"❌ ASR runtime error: {e}")
        return False

    print("\n" + "-" * 60)
    if events:
        print(f"✅ SUCCESS: Captured {len(events)} transcription event(s) from your voice!")
        return True
    else:
        print("⚠️  No speech was recognized during the window.")
        print("   (Ensure your microphone is close and speak clearly at normal volume).")
        return True


if __name__ == "__main__":
    success = run_asr_test()
    sys.exit(0 if success else 1)
