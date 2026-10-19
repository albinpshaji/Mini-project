"""
Unit Tests for Phase 3: GLM-4-Voice Acoustic Pre-Warming & Pipelined TTFA.
Tests:
  1. AcousticPrewarmer initialization and ONNX graph pre-warming.
  2. Empathetic acknowledgment audio cache generation and WAV format validity.
  3. Fast-path canonical opener matching (exact, prefix, and regex).
  4. Clinical safety validation on pre-rendered audio buffers.
  5. Pipelined audio synthesis timing (T_synth < T_playback guarantee).
  6. TTSEngine prewarmer integration.
"""

import os
import sys
import time
import wave
import io
import unittest
import numpy as np
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tts.piper_engine import TTSEngine, AcousticPrewarmer, PrewarmedAudioChunk
from src.safety.streaming_guardrail import StreamingGuardrail
from piper import PiperVoice


class TestAcousticPrewarming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
        with open(config_path) as f:
            cls.cfg = yaml.safe_load(f)
        cls.tts_cfg = cls.cfg.get("tts", {})
        voice_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", cls.tts_cfg.get("voice_model", "models/tts/en_US-lessac-medium.onnx")))
        if not os.path.exists(voice_model_path):
            voice_model_path = cls.tts_cfg.get("voice_model")
        cls.voice = PiperVoice.load(str(voice_model_path))
        cls.prewarmer = AcousticPrewarmer(cls.voice)
        cls.guardrail = StreamingGuardrail()

    def test_01_prewarmer_initialization(self):
        """Verify AcousticPrewarmer initializes with valid sample rate and canonical openers."""
        self.assertIsNotNone(self.prewarmer.voice)
        self.assertIn(self.prewarmer.sample_rate, [22050, 16000, 24000])
        self.assertGreater(len(self.prewarmer.CANONICAL_OPENERS), 5)
        self.assertIn("understand", self.prewarmer.CANONICAL_OPENERS)
        self.assertIn("noted", self.prewarmer.CANONICAL_OPENERS)
        self.assertIn("hear_you", self.prewarmer.CANONICAL_OPENERS)

    def test_02_onnx_graph_warm_up(self):
        """Verify warm_up_engine pre-allocates ONNX execution graph and completes quickly."""
        t_warmup = self.prewarmer.warm_up_engine()
        self.assertIsInstance(t_warmup, float)
        self.assertGreater(t_warmup, 0.0)
        # Subsequent synthesis should now be warm
        t0 = time.perf_counter()
        _ = list(self.voice.synthesize("Test phrase."))
        warm_latency = (time.perf_counter() - t0) * 1000.0
        # Warm latency should be under 150 ms
        self.assertLess(warm_latency, 150.0)

    def test_03_empathetic_audio_cache_generation(self):
        """Verify build_cache pre-renders all canonical openers into valid 16-bit WAV buffers."""
        cache = self.prewarmer.build_cache()
        self.assertEqual(len(cache), len(self.prewarmer.CANONICAL_OPENERS))

        for key, chunk in cache.items():
            self.assertIsInstance(chunk, PrewarmedAudioChunk)
            self.assertEqual(chunk.key, key)
            self.assertGreater(len(chunk.wav_bytes), 100)
            self.assertGreater(chunk.duration_s, 0.2)
            self.assertGreater(len(chunk.pcm_int16), 1000)
            self.assertEqual(chunk.pcm_int16.dtype, np.int16)

            # Validate WAV header and audio parameters
            with wave.open(io.BytesIO(chunk.wav_bytes), "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getsampwidth(), 2)
                self.assertEqual(wf.getframerate(), self.prewarmer.sample_rate)
                n_frames = wf.getnframes()
                self.assertAlmostEqual(chunk.duration_s, n_frames / self.prewarmer.sample_rate, places=2)

    def test_04_fast_path_opener_matching(self):
        """Verify exact, prefix, and regex matching for clinical openers."""
        self.prewarmer.build_cache()

        # Exact match
        chunk1 = self.prewarmer.match_opener("I understand. Let me help you with that.")
        self.assertIsNotNone(chunk1)
        self.assertIn("understand", chunk1.key)

        # Prefix variations
        chunk2 = self.prewarmer.match_opener("I understand your concern, let me note that down.")
        self.assertIsNotNone(chunk2)
        self.assertIn("understand", chunk2.key)

        chunk3 = self.prewarmer.match_opener("I have noted that down.")
        self.assertIsNotNone(chunk3)
        self.assertIn("noted", chunk3.key)

        chunk4 = self.prewarmer.match_opener("I hear you, please take a deep breath.")
        self.assertIsNotNone(chunk4)
        self.assertIn("hear_you", chunk4.key)

        chunk5 = self.prewarmer.match_opener("Thank you for sharing that information.")
        self.assertIsNotNone(chunk5)
        self.assertIn("thank_you", chunk5.key)

        chunk6 = self.prewarmer.match_opener("Please stay calm, I am here to help you.")
        self.assertIsNotNone(chunk6)
        self.assertIn("stay_calm", chunk6.key)

        # Non-matching presentation (should return None and use normal synthesis)
        chunk_none = self.prewarmer.match_opener("My knee has been swelling since yesterday.")
        self.assertIsNone(chunk_none)

    def test_05_guardrail_safety_on_prewarmed_openers(self):
        """Verify that every prewarmed acknowledgment passes the safety guardrail with 0 violations."""
        cache = self.prewarmer.build_cache()
        for key, chunk in cache.items():
            guarded = self.guardrail.guard_chunk(chunk.text)
            self.assertTrue(guarded.safe, f"Opener '{key}' was flagged as unsafe!")
            self.assertFalse(guarded.intercepted, f"Opener '{key}' was falsely intercepted by guardrail!")
            self.assertEqual(guarded.violation_types, [])
            self.assertEqual(guarded.text, chunk.text)

    def test_06_pipelined_synthesis_timing_guarantee(self):
        """
        Verify the mathematical zero-underrun guarantee:
        Synthesis of subsequent clause (T_synth) must be strictly less than playback duration (T_play)
        of the initial pre-warmed opener.
        """
        cache = self.prewarmer.build_cache()
        opener = cache["understand"]
        t_play_opener = opener.duration_s  # Typically ~1.2 to 1.8 seconds

        subsequent_clause = "Could you tell me how long the pain has been present?"
        t0 = time.perf_counter()
        chunks = [c.audio_float_array for c in self.voice.synthesize(subsequent_clause)]
        t_synth_next = time.perf_counter() - t0

        self.assertGreater(len(chunks), 0)
        # T_synth should be under 0.15s (150ms), while T_play is > 0.8s
        self.assertLess(t_synth_next, t_play_opener,
                        f"Buffer underrun risk! T_synth ({t_synth_next:.3f}s) >= T_play ({t_play_opener:.3f}s)")
        # Margin should be at least 0.4s ahead of playback
        lead_margin_s = t_play_opener - t_synth_next
        self.assertGreater(lead_margin_s, 0.4, f"Lead margin {lead_margin_s:.3f}s is too narrow (< 0.4s)!")

    def test_07_tts_engine_prewarmer_integration(self):
        """Verify TTSEngine embeds prewarmer and engine.warm_up() executes cleanly."""
        engine = TTSEngine(self.tts_cfg, guardrail=self.guardrail)
        self.assertTrue(hasattr(engine, "prewarmer"))
        self.assertIsInstance(engine.prewarmer, AcousticPrewarmer)

        lat = engine.warm_up()
        self.assertIsInstance(lat, float)
        self.assertGreater(len(engine.prewarmer.cache), 5)


if __name__ == "__main__":
    unittest.main()
