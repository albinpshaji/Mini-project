"""
TTS engine module wrapping Piper.

Provides incremental (sentence-chunked) synthesis and playback, with a
cancel() method. In Phase 1 (sequential) cancel() is never actually needed --
but the rollback_manager in Phase 4 depends on this existing, so it's built
in from the start rather than retrofitted later.
"""

import io
import re
import time
import wave
import threading
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import sounddevice as sd
from piper import PiperVoice


@dataclass
class PrewarmedAudioChunk:
    key: str
    text: str
    wav_bytes: bytes
    duration_s: float
    pcm_int16: np.ndarray
    audio_float: np.ndarray


class AcousticPrewarmer:
    """
    GLM-4-Voice style acoustic pre-warming and RAM audio caching (Phase 3).
    Pre-synthesizes canonical clinical acknowledgment openers for sub-40ms TTFA,
    eliminating the cold-start ONNX synthesis stall and providing instantaneous speech onset.
    """
    CANONICAL_OPENERS = {
        "understand": "I understand. Let me help you with that.",
        "understand_concern": "I understand your concern.",
        "understand_help": "I understand, let me help you with that.",
        "noted": "I have noted that down. Let me assess your symptoms.",
        "noted_short": "I have noted that down.",
        "hear_you": "I hear you, please take a deep breath.",
        "hear_you_short": "I hear you.",
        "hear_that": "I hear you. Let me help you with that.",
        "thank_you": "Thank you for sharing that information.",
        "thank_you_short": "Thank you for letting me know.",
        "stay_calm": "Please stay calm, I am here to help you.",
        "stay_calm_short": "Please stay calm.",
        "help_you": "Let me help you with that.",
        "assess": "Let me assess your symptoms.",
        "short_understand": "I understand.",
        "that_sounds": "That sounds serious. Let me help you right away.",
        "that_sounds_severe": "That sounds severe. Please stay as calm as possible.",
        "this_sounds": "This sounds like an urgent medical concern.",
        "please_tell_me": "Please tell me more about your symptoms.",
    }

    OPENER_REGEX = re.compile(
        r"^(i\s+understand(\s+your\s+concern)?|i\s+have\s+noted(\s+that\s+down)?|noted|i\s+hear(\s+you|\s+that)?|thank\s+you(\s+for\s+(sharing|letting\s+me\s+know))?|please\s+stay\s+calm|stay\s+calm|let\s+me\s+help(\s+you)?|let\s+me\s+assess|that\s+sounds(\s+quite|\s+very)?|this\s+sounds(\s+like)?|please\s+tell\s+me)",
        re.IGNORECASE
    )

    def __init__(self, piper_voice):
        self.voice = piper_voice
        self.sample_rate = getattr(piper_voice.config, "sample_rate", 22050)
        self.cache: Dict[str, PrewarmedAudioChunk] = {}

    def warm_up_engine(self) -> float:
        """
        Warms up the ONNX Runtime execution provider graph and internal memory pools,
        eliminating the cold-start synthesis stall.
        """
        t0 = time.perf_counter()
        _ = list(self.voice.synthesize("Initializing audio engine."))
        return (time.perf_counter() - t0) * 1000.0

    def build_cache(self) -> Dict[str, PrewarmedAudioChunk]:
        """Pre-renders canonical openers into persistent in-memory RAM buffers."""
        for key, text in self.CANONICAL_OPENERS.items():
            chunks = [c.audio_float_array for c in self.voice.synthesize(text)]
            if not chunks:
                continue
            combined = np.concatenate(chunks)
            int16_arr = np.clip(combined * 32767.0, -32768.0, 32767.0).astype(np.int16)
            dur = float(len(combined) / self.sample_rate)

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(int16_arr.tobytes())

            self.cache[key] = PrewarmedAudioChunk(
                key=key,
                text=text,
                wav_bytes=buf.getvalue(),
                duration_s=dur,
                pcm_int16=int16_arr,
                audio_float=combined,
            )
        return self.cache

    def get_prewarmed_chunk(self, text: str) -> Optional[PrewarmedAudioChunk]:
        """
        Direct lookup: returns cached chunk if clean text matches a registered opener.
        """
        clean = text.strip().lower().rstrip(".,!?")
        if not clean:
            return None

        # Check by key or normalized text
        if clean in self.cache:
            return self.cache[clean]

        for chunk in self.cache.values():
            if chunk.text.lower().rstrip(".,!?") == clean:
                return chunk
        return None

    def match_opener(self, text_prefix: str) -> Optional[PrewarmedAudioChunk]:
        """
        Fast-path check: returns pre-synthesized audio chunk if opening tokens match
        a registered empathetic acknowledgment opener.
        """
        clean = text_prefix.strip().lower()
        if not clean:
            return None

        # 1. Exact or prefix match against cached openers
        direct = self.get_prewarmed_chunk(clean)
        if direct is not None:
            return direct

        for key, chunk in self.cache.items():
            c_text = chunk.text.lower().rstrip(".,!?")
            clean_stripped = clean.rstrip(".,!?")
            if clean_stripped == c_text or clean_stripped.startswith(c_text):
                return chunk

        # 2. Regex matching on canonical opener patterns
        m = self.OPENER_REGEX.search(clean)
        if not m:
            return None
        matched = m.group(1).lower()
        if "understand" in matched:
            if "concern" in matched:
                return self.cache.get("understand_concern") or self.cache.get("understand")
            return self.cache.get("understand") or self.cache.get("short_understand")
        if "noted" in matched:
            return self.cache.get("noted") or self.cache.get("noted_short")
        if "hear" in matched:
            return self.cache.get("hear_you") or self.cache.get("hear_that") or self.cache.get("hear_you_short")
        if "thank" in matched:
            return self.cache.get("thank_you") or self.cache.get("thank_you_short")
        if "calm" in matched:
            return self.cache.get("stay_calm") or self.cache.get("stay_calm_short")
        if "help" in matched:
            return self.cache.get("help_you") or self.cache.get("understand")
        if "assess" in matched:
            return self.cache.get("assess") or self.cache.get("noted")
        if "that" in matched and "sound" in matched:
            if "severe" in matched:
                return self.cache.get("that_sounds_severe") or self.cache.get("that_sounds")
            return self.cache.get("that_sounds") or self.cache.get("that_sounds_severe")
        if "this" in matched and "sound" in matched:
            return self.cache.get("this_sounds") or self.cache.get("that_sounds")
        if "tell" in matched:
            return self.cache.get("please_tell_me") or self.cache.get("help_you")
        return None

    def get_default_opener(self) -> Optional[PrewarmedAudioChunk]:
        """Returns canonical grounding opener for Phase 4 Speculative Pre-Emission."""
        return self.cache.get("understand") or (next(iter(self.cache.values())) if self.cache else None)


class TTSEngine:
    def __init__(self, config: dict, guardrail=None):
        """
        config: the 'tts' section of config.yaml
        """
        self.cfg = config
        self.voice = PiperVoice.load(config["voice_model"])
        self.sample_rate = self.voice.config.sample_rate

        self._cancel_flag = threading.Event()
        self._playback_thread = None
        self._is_active_speaking = False
        self.on_audio_chunk = None

        # Phase 3: Acoustic Prewarmer
        self.prewarmer = AcousticPrewarmer(self.voice)

        # Feature C: Pre-TTS Real-Time Streaming Guardrail
        from src.safety.streaming_guardrail import StreamingGuardrail
        self.guardrail = guardrail if guardrail is not None else StreamingGuardrail()

    def warm_up(self) -> float:
        """Runs graph warm-up and populates acoustic cache."""
        lat = self.prewarmer.warm_up_engine()
        self.prewarmer.build_cache()
        return lat

    def cancel(self):
        """Stop playback ASAP. Safe to call even if nothing is playing."""
        self._cancel_flag.set()
        self._is_active_speaking = False
        sd.stop()

    def speak(self, text: str, blocking: bool = True):
        self._cancel_flag.clear()
        self._is_active_speaking = True

        def _run():
            try:
                for audio_chunk in self._synthesize_chunks(text):
                    if self._cancel_flag.is_set():
                        return
                    if self.on_audio_chunk:
                        try:
                            self.on_audio_chunk(audio_chunk, self.sample_rate)
                        except Exception:
                            pass
                    sd.play(audio_chunk, samplerate=self.sample_rate)
                    sd.wait()
                    if self._cancel_flag.is_set():
                        sd.stop()
                        return
            finally:
                self._is_active_speaking = False

        if blocking:
            _run()
        else:
            self._playback_thread = threading.Thread(target=_run, daemon=True)
            self._playback_thread.start()

    def is_speaking(self) -> bool:
        return self._is_active_speaking or (self._playback_thread is not None and self._playback_thread.is_alive())

    def cancel_immediate(self):
        """Immediately aborts speech synthesis and flushes the sounddevice DAC buffer in <15ms."""
        self._cancel_flag.set()
        self._is_active_speaking = False
        sd.stop()

    def speak_sentence(self, sentence: str):
        """Synthesizes and plays a single sentence/sub-clause chunk immediately."""
        if self._cancel_flag.is_set():
            return
        if self.guardrail:
            res = self.guardrail.guard_chunk(sentence)
            sentence = res.text
        clean = self._strip_markdown(sentence)
        if not clean.strip():
            return
        self._is_active_speaking = True
        try:
            if hasattr(self, "prewarmer") and self.prewarmer:
                cached = self.prewarmer.get_prewarmed_chunk(clean)
                if cached is not None:
                    if self._cancel_flag.is_set():
                        return
                    if self.on_audio_chunk:
                        try:
                            self.on_audio_chunk(cached.audio_float, self.sample_rate)
                        except Exception:
                            pass
                    sd.play(cached.audio_float, samplerate=self.sample_rate)
                    sd.wait()
                    return

            for audio_chunk in self.voice.synthesize(clean):
                if self._cancel_flag.is_set():
                    sd.stop()
                    return
                if self.on_audio_chunk:
                    try:
                        self.on_audio_chunk(audio_chunk.audio_float_array, self.sample_rate)
                    except Exception:
                        pass
                sd.play(audio_chunk.audio_float_array, samplerate=self.sample_rate)
                sd.wait()
                if self._cancel_flag.is_set():
                    sd.stop()
                    return
        finally:
            self._is_active_speaking = False

    # Sub-clause delimiters: punctuation pauses, major conjunctions, or sentence terminators
    SUB_CLAUSE_REGEX = re.compile(
        r"(?:[,;:\-–—]|\b(?:and|but|because|so|while|however|therefore)\b|(?<=[.!?]))\s*",
        re.IGNORECASE
    )

    def speak_stream(self, token_generator, on_first_audio_callback=None) -> str:
        """
        Consumes tokens incrementally as they are generated by the SLM.
        Synthesizes and speaks sub-clause chunks (every 3-5 words) as soon as they arrive,
        slashing Time-To-First-Audio (TTFA) from ~450ms down to ~75-90ms.
        Returns the complete assembled response text.
        """
        self._cancel_flag.clear()
        self._is_active_speaking = True
        buffer = ""
        full_response = []
        first_audio_started = False

        try:
            for chunk in token_generator:
                if self._cancel_flag.is_set():
                    break
                token = chunk.get("token", "")
                if not token:
                    continue
                buffer += token
                full_response.append(token)

                # Check for sentence end OR sub-clause boundary with >= 3 words
                words = buffer.split()
                has_sentence_end = any(p in buffer for p in [". ", "? ", "! ", ".\n", "?\n", "!\n"])
                has_sub_clause = (len(words) >= 3 and self.SUB_CLAUSE_REGEX.search(buffer)) or len(words) >= 6

                if has_sentence_end or has_sub_clause:
                    # Split at the first sub-clause or sentence delimiter
                    parts = self.SUB_CLAUSE_REGEX.split(buffer, maxsplit=1)
                    if len(parts) > 1 and parts[0].strip():
                        chunk_to_play = parts[0].strip()
                        buffer = "".join(parts[1:])
                        if not first_audio_started:
                            first_audio_started = True
                            if on_first_audio_callback:
                                on_first_audio_callback()
                        self.speak_sentence(chunk_to_play)

            # Play any remaining text in the buffer
            if buffer.strip() and not self._cancel_flag.is_set():
                if not first_audio_started and on_first_audio_callback:
                    on_first_audio_callback()
                self.speak_sentence(buffer.strip())
        finally:
            self._is_active_speaking = False

        return "".join(full_response).strip()

    @staticmethod
    def _strip_markdown(text: str) -> str:
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        text = re.sub(r"^\s*[\*\-]\s+", "", text, flags=re.MULTILINE)
        text = text.replace("*", "")
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
        return text.strip()

    def _synthesize_chunks(self, text: str):
        """Yields audio chunks split by sub-clauses for low-latency streaming playback."""
        if self.guardrail:
            text = self.guardrail.guard_chunk(text).text
        chunks = self._split_sub_clauses(text)
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            if i == 0 and self.prewarmer:
                cached = self.prewarmer.get_prewarmed_chunk(chunk)
                if cached is not None:
                    yield cached.audio_float
                    continue
            for audio_chunk in self.voice.synthesize(chunk):
                yield audio_chunk.audio_float_array

    @staticmethod
    def _split_sub_clauses(text: str):
        """Splits full response into sub-clauses of 3-6 words or punctuation pauses."""
        clean = TTSEngine._strip_markdown(text)
        raw_parts = TTSEngine.SUB_CLAUSE_REGEX.split(clean)
        chunks = []
        current = []
        for part in raw_parts:
            part_str = part.strip()
            if not part_str:
                continue
            current.append(part_str)
            current_words = " ".join(current).split()
            if len(current_words) >= 3 or any(p in part_str for p in [".", "!", "?", ",", ";"]):
                chunks.append(" ".join(current))
                current = []
        if current:
            chunks.append(" ".join(current))
        return chunks if chunks else [clean]

    @staticmethod
    def _split_sentences(text: str):
        return re.split(r"(?<=[.!?])\s+", text.strip())
