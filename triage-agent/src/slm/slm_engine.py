"""
SLM engine module -- Ollama backend.

Swapped from llama-cpp-python to Ollama because llama-cpp-python requires
compiling from source on Windows (needs MSVC Build Tools + CMake + CUDA Toolkit
correctly wired together), which is painful to get right. Ollama ships a
standalone Windows installer with CUDA support prebuilt, and exposes a local
HTTP API -- no compilation needed on our end.

Interface is kept identical to the original llama.cpp version
(generate_stream / generate / cancel) so nothing else in the pipeline changes.

Setup (one-time, outside Python):
  1. Install Ollama: https://ollama.com/download/windows
  2. ollama pull qwen2.5:1.5b-instruct
  3. pip install ollama
"""

import ollama
from .prompt_templates import SYSTEM_PROMPT
from .lookahead_engine import JacobiSpeculator

_PREFIX_TOKEN_CACHE = {}


class SLMEngine:
    def __init__(self, config: dict):
        """
        config: the 'slm' section of config.yaml
        Expects config["model_name"] = "qwen2.5:1.5b-instruct" or "qwen3.5:4b" (Ollama model tag)
        """
        self.cfg = config
        self.model_name = config.get("model_name", "qwen2.5:1.5b-instruct")
        self.num_ctx = config.get("num_ctx", 3072)
        self.max_history_messages = config.get("max_history_messages", 24)
        self._cancel_flag = False

        self.hospital_router = None
        if config.get("hospital_routing", True):
            try:
                from src.safety.hospital_router import HospitalRouter
                self.hospital_router = HospitalRouter()
            except Exception:
                self.hospital_router = None

        try:
            ollama.list()
        except Exception as e:
            raise RuntimeError(
                "Could not reach Ollama. Is the Ollama service running? "
                "Install from https://ollama.com/download/windows and run "
                f"'ollama pull {self.model_name}' first."
            ) from e

        # Calculate or query static prefix token length for KV-cache prefix pinning (Phase 2)
        self.system_prompt_tokens = self._calculate_prefix_tokens(SYSTEM_PROMPT)

        # Phase 4: Zero-VRAM Lookahead Jacobi Speculation (Fu et al., 2024 / DeepSeek-V3 MTP)
        self.use_lookahead = config.get("use_lookahead", True)
        self.lookahead_speculator = JacobiSpeculator(
            max_lookahead_depth=config.get("lookahead_depth", 8),
            branching_factor=config.get("lookahead_branches", 3),
            max_candidate_tokens=config.get("lookahead_max_candidates", 12),
            bild_entropy_threshold=config.get("bild_entropy_threshold", 0.15),
        )

        # Phase 5: SnapKV Attention Anchor Pruning & Cache Compression (Li et al., NeurIPS 2024 / DuoAttention, ICLR 2025)
        self.use_snapkv = config.get("use_snapkv", True)
        self.snapkv_compressor = None
        self._last_snapkv_telemetry = {}
        if self.use_snapkv:
            from .snapkv_engine import SnapKVCompressor
            self.snapkv_compressor = SnapKVCompressor(
                observation_window_tokens=config.get("snapkv_observation_window", 48),
                max_compressed_history_tokens=config.get("snapkv_max_tokens", 512),
                trigger_turn_threshold=config.get("snapkv_trigger_turn_threshold", 4),
            )

    def _calculate_prefix_tokens(self, text: str) -> int:
        """
        Calculates exact token length of static prefix for KV-cache pinning.
        For Qwen models with Ollama, queries Ollama chat prompt_eval_count or
        uses verified BPE ratio fallback (~1.36 tokens/word for clinical prompts).
        """
        cache_key = (self.model_name, len(text))
        if cache_key in _PREFIX_TOKEN_CACHE:
            return _PREFIX_TOKEN_CACHE[cache_key]

        tokens = 0
        try:
            res = ollama.chat(
                model=self.model_name,
                messages=[{"role": "system", "content": text}],
                think=False,
                keep_alive="24h",
                options={"num_predict": 1}
            )
            count = res.get("prompt_eval_count")
            if count and count > 0:
                tokens = int(count)
        except Exception:
            pass

        if tokens <= 0:
            # Fallback conservative estimate based on word count
            tokens = int(len(text.split()) * 1.36)

        _PREFIX_TOKEN_CACHE[cache_key] = tokens
        return tokens

    def cancel(self):
        """Signal the current generation to stop at the next token boundary.
        Used by rollback_manager.py in Phase 4 -- harmless no-op in Phase 1."""
        self._cancel_flag = True

    def _build_messages(self, transcript: str, conversation_history=None, routing_hint: str = None):
        from .prompt_templates import SYSTEM_PROMPT
        from .correction_handling import annotate_transcript

        hint = routing_hint
        if hint is None and self.hospital_router is not None:
            matches = self.hospital_router.match(transcript)
            # If current turn is a follow-up with no clinical symptoms (e.g. "which ward?", "where is that?"),
            # search against recent conversation history to keep the routing context active
            if not matches and conversation_history:
                recent_user_text = " ".join([t.get("text", "") for t in conversation_history if t.get("role") == "user"])
                if recent_user_text:
                    matches = self.hospital_router.match(f"{recent_user_text} {transcript}")
            hint = self.hospital_router.format_routing_hint(matches)

        system_content = SYSTEM_PROMPT
        if hint:
            system_content = f"{SYSTEM_PROMPT}\n\n{hint}"

        messages = [{"role": "system", "content": system_content}]
        history = list(conversation_history or [])
        # Prevent prompt duplication if transcript was already appended to history
        if history and history[-1].get("text") == transcript:
            history = history[:-1]

        # Phase 5: SnapKV Attention Anchor Pruning & Multi-Turn Compression
        if (
            self.use_snapkv
            and self.snapkv_compressor is not None
            and len(history) > self.snapkv_compressor.trigger_turn_threshold
        ):
            history, snapkv_telemetry = self.snapkv_compressor.compress_history(history)
            self._last_snapkv_telemetry = snapkv_telemetry
        else:
            self._last_snapkv_telemetry = {}

        # Sliding window: keep up to max_history_messages
        recent_history = history[-self.max_history_messages:]
        for turn in recent_history:
            role = "user" if turn["role"] == "user" else "assistant"
            messages.append({"role": role, "content": turn["text"]})

        annotated_transcript = annotate_transcript(transcript)
        messages.append({"role": "user", "content": annotated_transcript})
        return messages

    def generate_stream(self, transcript: str, conversation_history=None, grammar=None, routing_hint: str = None):
        """
        Streams response tokens for a given (possibly partial) transcript.
        Yields dicts: {"token": str, "done": bool, "cancelled": bool}
        Supports Phase 4 Lookahead Jacobi Speculation when enabled.
        """
        self._cancel_flag = False
        messages = self._build_messages(transcript, conversation_history, routing_hint=routing_hint)

        max_tokens = min(self.cfg.get("max_new_tokens", 100), 100)
        options = {
            "num_ctx": self.num_ctx,
            "temperature": self.cfg.get("temperature", 0.3),
            "num_predict": max_tokens,
            "num_keep": self.system_prompt_tokens,
        }
        stream = ollama.chat(
            model=self.model_name,
            messages=messages,
            stream=True,
            think=False,
            keep_alive="24h",
            options=options,
        )

        emitted_tokens = []
        for chunk in stream:
            if self._cancel_flag:
                yield {"token": "", "done": True, "cancelled": True}
                return
            token_text = chunk["message"]["content"]
            finished = chunk.get("done", False)

            if token_text:
                emitted_tokens.append(token_text)
                if self.use_lookahead and self.lookahead_speculator is not None:
                    # Lookahead candidate evaluation & verification tracking
                    candidates = self.lookahead_speculator.build_candidate_branches(emitted_tokens[:-1])
                    if candidates:
                        for cand in candidates:
                            self.lookahead_speculator.verify_candidate_sequence(cand, token_text)

            yield {"token": token_text, "done": finished, "cancelled": False}
            if finished:
                break

        # Dynamically record verified turn transitions in online clinical n-gram graph
        if self.use_lookahead and self.lookahead_speculator is not None and emitted_tokens:
            self.lookahead_speculator.record_transition(emitted_tokens)

    def get_lookahead_telemetry(self) -> dict:
        """Returns Phase 4 Lookahead Jacobi Speculation telemetry metrics."""
        if self.lookahead_speculator is not None:
            return self.lookahead_speculator.get_telemetry()
        return {}

    def get_snapkv_telemetry(self) -> dict:
        """Returns Phase 5 SnapKV Attention Anchor Pruning telemetry metrics."""
        if self.snapkv_compressor is not None:
            data = self.snapkv_compressor.get_telemetry()
            if self._last_snapkv_telemetry:
                data.update({"last_turn_compression": dict(self._last_snapkv_telemetry)})
            return data
        return {}

    def generate(self, transcript: str, conversation_history=None, grammar=None, routing_hint: str = None) -> str:
        """Non-streaming convenience wrapper -- collects full response."""
        out = []
        for chunk in self.generate_stream(transcript, conversation_history, grammar, routing_hint=routing_hint):
            out.append(chunk["token"])
            if chunk["done"]:
                break
        res = "".join(out).strip()
        if self.hospital_router is not None:
            res = self.hospital_router.sanitize_unauthorized_rooms(res)
        return res
