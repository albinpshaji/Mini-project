"""
Unit Test Suite for Phase 2: Static System Prompt Prefix KV-Cache Pinning (num_keep).
Literature Grounding:
  - ASPLOS 2025: Fast On-Device LLM Inference with NPUs & Prefix Caching (Xu et al.)
  - Master Plan Innovation 6: Persistent static KV-cache state preservation across turns
"""

import unittest
from unittest.mock import patch, MagicMock
from src.slm.slm_engine import SLMEngine, _PREFIX_TOKEN_CACHE
from src.slm.prompt_templates import SYSTEM_PROMPT
from src.speculative.speculator import Speculator


class TestKVPinningPhase2(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "model_name": "qwen3.5:4b",
            "num_ctx": 3072,
            "temperature": 0.3,
            "max_new_tokens": 100,
            "hospital_routing": True,
        }

    def test_prefix_token_calculation(self):
        """Verify that system_prompt_tokens is computed and bounded within valid range."""
        engine = SLMEngine(self.cfg)
        self.assertIsInstance(engine.system_prompt_tokens, int)
        # 778 words translates to ~1000 - 1100 BPE tokens in Qwen tokenizer
        self.assertGreater(engine.system_prompt_tokens, 700)
        self.assertLess(engine.system_prompt_tokens, 1500)

    def test_prefix_token_cache(self):
        """Verify that prefix token calculation is cached across instances."""
        engine1 = SLMEngine(self.cfg)
        tokens1 = engine1.system_prompt_tokens
        cache_key = (engine1.model_name, len(SYSTEM_PROMPT))
        self.assertIn(cache_key, _PREFIX_TOKEN_CACHE)
        self.assertEqual(_PREFIX_TOKEN_CACHE[cache_key], tokens1)

        engine2 = SLMEngine(self.cfg)
        self.assertEqual(engine2.system_prompt_tokens, tokens1)

    def test_speculator_exposes_system_prompt_tokens(self):
        """Verify that Speculator exposes system_prompt_tokens property from underlying engine."""
        speculator = Speculator(self.cfg)
        self.assertIsInstance(speculator.system_prompt_tokens, int)
        self.assertEqual(speculator.system_prompt_tokens, speculator.engine.system_prompt_tokens)

    @patch("src.slm.slm_engine.ollama.chat")
    def test_generate_stream_passes_num_keep(self, mock_chat):
        """Verify that generate_stream injects num_keep into ollama.chat options."""
        mock_chat.return_value = iter([
            {"message": {"content": "Hello"}, "done": True}
        ])

        engine = SLMEngine(self.cfg)
        # Consume the generator
        chunks = list(engine.generate_stream("I have a mild headache"))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["token"], "Hello")
        self.assertTrue(chunks[0]["done"])

        # Check call arguments
        self.assertTrue(mock_chat.called)
        call_kwargs = mock_chat.call_args[1]
        self.assertIn("options", call_kwargs)
        options = call_kwargs["options"]
        self.assertIn("num_keep", options)
        self.assertEqual(options["num_keep"], engine.system_prompt_tokens)
        self.assertEqual(options["num_ctx"], 3072)
        self.assertEqual(options["temperature"], 0.3)

    def test_live_generation_with_num_keep(self):
        """Verify that live SLM generation with num_keep produces valid clinical guidance."""
        engine = SLMEngine(self.cfg)
        response = engine.generate("I have had a mild runny nose and sneezing for two days.")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 10)
        # Should not lead to emergency or panic for mild cold
        self.assertNotIn("call 911 immediately", response.lower())


if __name__ == "__main__":
    unittest.main()
