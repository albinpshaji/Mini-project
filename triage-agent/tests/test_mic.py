"""
Microphone & Audio Capture Diagnostic Test.

Checks:
  1. Default audio input device & available channels.
  2. Records 3 seconds of audio from the configured/default device.
  3. Measures Peak Amplitude and RMS volume levels.
  4. Runs WebRTC VAD (Voice Activity Detection) to verify speech frame detection.
"""

import sys
import time
import yaml
import numpy as np
import sounddevice as sd
import webrtcvad


def run_mic_test(duration_s=3, config_path="config.yaml"):
    print("=" * 60)
    print(" 🎙️  MICROPHONE & VAD DIAGNOSTIC TEST")
    print("=" * 60)

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        configured_dev = cfg.get("asr", {}).get("input_device", None)
    except Exception:
        configured_dev = None

    print(f"Configured input_device in {config_path}: {configured_dev}")
    default_dev = sd.default.device[0]
    print(f"System default input device index: {default_dev}")

    dev_to_use = configured_dev if configured_dev is not None else default_dev
    dev_info = sd.query_devices(dev_to_use)
    print(f"Using device: [{dev_to_use}] {dev_info['name']}")
    print(f"Max input channels: {dev_info['max_input_channels']}")
    print(f"Default sample rate: {dev_info['default_samplerate']} Hz")

    sample_rate = 16000
    frame_ms = 30
    frame_samples = int(sample_rate * frame_ms / 1000)

    print(f"\n[1/2] Recording {duration_s} seconds of audio... PLEASE SPEAK INTO THE MIC.")
    for i in range(duration_s, 0, -1):
        print(f"  Recording in progress... {i}s remaining", end="\r", flush=True)
        time.sleep(1)
    print("  Recording complete!                          ")

    try:
        rec = sd.rec(
            int(sample_rate * duration_s),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            device=configured_dev,
        )
        sd.wait()
    except Exception as e:
        print(f"\n❌ FAILED to record audio: {e}")
        return False

    peak = int(np.max(np.abs(rec)))
    rms = float(np.sqrt(np.mean(rec.astype(float) ** 2)))

    print(f"\n[2/2] Analyzing Audio Signal:")
    print(f"  • Peak Amplitude : {peak} / 32767")
    print(f"  • RMS Energy     : {rms:.2f}")

    vad = webrtcvad.Vad(2)
    total_frames = len(rec) // frame_samples
    speech_frames = 0

    for i in range(total_frames):
        chunk = rec[i * frame_samples : (i + 1) * frame_samples].tobytes()
        if vad.is_speech(chunk, sample_rate):
            speech_frames += 1

    speech_pct = (speech_frames / total_frames) * 100 if total_frames > 0 else 0
    print(f"  • VAD Speech Frames: {speech_frames}/{total_frames} ({speech_pct:.1f}%)")

    print("\n" + "-" * 60)
    if peak < 300:
        print("⚠️  WARNING: Audio level is extremely low or silent.")
        print("   Check if your microphone is physically muted or volume is set too low in system settings.")
        return False
    elif speech_frames == 0:
        print("⚠️  WARNING: Sound was detected, but no voice/speech was classified by VAD.")
        print("   Try speaking louder or closer to the microphone.")
        return True
    else:
        print("✅ SUCCESS: Microphone is capturing clear audio and voice is detected by VAD!")
        return True


if __name__ == "__main__":
    success = run_mic_test()
    sys.exit(0 if success else 1)
