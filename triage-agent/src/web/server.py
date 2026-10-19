"""
OmniSpec-Triage Web Backend Server.
Provides REST and WebSocket endpoints for mobile phone and desktop browser interaction.
Supports:
1. Live text chat with conversational clinical triage reasoning.
2. Mobile browser audio recording and upload with faster-whisper streaming transcription.
3. Multimodal paralinguistic acoustic biomarker extraction (<3ms DSP co-processor).
4. Real-time deterministic clinical red-flag detection and emergency handoffs.
5. In-browser Piper neural TTS audio playback.
6. Asynchronous structured EHR clinical record extraction (JSON slots).
7. Live edge hardware and latency telemetry.
"""

import os
import io
import re
import json
import time
import base64
import wave
import uuid
import socket
import logging
import asyncio
import threading
from typing import List, Dict, Any, Optional
from pathlib import Path

import av
import yaml
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Internal subsystem imports
from src.safety.red_flag_rules import check_red_flags
from src.safety.handoff import evaluate_handoff, contains_fabricated_placeholder, FABRICATION_FALLBACK_MESSAGE
from src.safety.slot_extraction import extract_symptom_slots, SymptomSlots
from src.slm.slm_engine import SLMEngine
from src.slm.correction_handling import annotate_transcript
from src.audio.acoustic_biomarkers import AcousticBiomarkerCoProcessor, AcousticBiomarkerResult
from src.scheduler.async_scheduler import ResourceScheduler
from src.speculative.speculator import Speculator
from src.speculative.divergence_detector import compute_divergence
from faster_whisper import WhisperModel
from piper import PiperVoice

import collections

# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
STATIC_DIR = Path(__file__).resolve().parent / "static"
REACT_DIST_DIR = ROOT_DIR / "frontend" / "dist"
SERVER_LOG_FILE = ROOT_DIR / "server.log"

log_history = collections.deque(maxlen=300)

class BufferingLogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            log_history.append(msg)
        except Exception:
            pass

# Configure root logger with console, file (server.log), and in-memory buffer
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
root_logger = logging.getLogger()

file_handler = logging.FileHandler(SERVER_LOG_FILE, mode="a", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
root_logger.addHandler(file_handler)

buf_handler = BufferingLogHandler()
buf_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
root_logger.addHandler(buf_handler)

log = logging.getLogger("web_server")

# Load config
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

app = FastAPI(title="OmniSpec-Triage Live Web Interface", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models & engines (lazy or preloaded)
slm_engine: Optional[SLMEngine] = None
whisper_model: Optional[WhisperModel] = None
whisper_partial: Optional[WhisperModel] = None
speculator: Optional[Speculator] = None
piper_voice: Optional[PiperVoice] = None
biomarker_coprocessor: Optional[AcousticBiomarkerCoProcessor] = None
resource_scheduler: Optional[ResourceScheduler] = None
streaming_guardrail: Optional[Any] = None

# Audio cache for generated TTS audio
audio_cache: Dict[str, bytes] = {}

# Active conversation state
class ConversationSession:
    def __init__(self):
        self.history: List[Dict[str, str]] = []
        self.turn_count: int = 0
        self.latest_slots: Dict[str, Any] = SymptomSlots().to_dict()
        self.active_red_flag: Optional[str] = None
        self.last_biomarkers: Optional[Dict[str, Any]] = None
        self.last_telemetry: Dict[str, Any] = {}
        self.speculative_enabled: bool = True
        self.divergence_threshold: float = float(cfg.get("speculative", {}).get("divergence_threshold", 0.70))

    def reset(self):
        self.history = []
        self.turn_count = 0
        self.latest_slots = SymptomSlots().to_dict()
        self.active_red_flag = None
        self.last_biomarkers = None
        self.last_telemetry = {}

session = ConversationSession()


def get_lan_ip() -> str:
    """Discovers host LAN IP for phone connection."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def initialize_engines():
    """Initializes models upon startup."""
    global slm_engine, whisper_model, whisper_partial, speculator, piper_voice, biomarker_coprocessor, resource_scheduler, streaming_guardrail
    
    log.info("Initializing OmniSpec-Triage AI Engines...")
    
    # 0. Pre-TTS Real-Time Streaming Guardrail (Feature C)
    from src.safety.streaming_guardrail import StreamingGuardrail
    streaming_guardrail = StreamingGuardrail()
    
    # 1. Resource scheduler
    resource_scheduler = ResourceScheduler(cfg.get("pipeline", {}))
    
    # 2. Acoustic Biomarker co-processor (16kHz DSP)
    biomarker_coprocessor = AcousticBiomarkerCoProcessor(sample_rate=16000)
    
    # 3. Whisper ASR: dual-model setup (tiny for partial hypotheses, base for final)
    asr_cfg = cfg.get("asr", {})
    model_name = asr_cfg.get("final_model_size") or asr_cfg.get("model_size", "base")
    whisper_device = asr_cfg.get("device", "cpu")
    whisper_compute = asr_cfg.get("compute_type", "int8")
    log.info(f"Loading faster-whisper final ({model_name}) on {whisper_device} [{whisper_compute}]...")
    whisper_model = WhisperModel(model_name, device=whisper_device, compute_type=whisper_compute)
    
    log.info("Loading faster-whisper partial (tiny) for low-latency speculative hypothesis streaming...")
    whisper_partial = WhisperModel("tiny", device="cpu", compute_type="int8")
    
    # 4. Piper Neural TTS
    tts_cfg = cfg.get("tts", {})
    voice_path = ROOT_DIR / tts_cfg.get("voice_model", "models/tts/en_US-lessac-medium.onnx")
    if not voice_path.exists():
        log.warning(f"Voice model not found at {voice_path}, checking relative path...")
        voice_path = Path(tts_cfg.get("voice_model"))
    log.info(f"Loading Piper voice from {voice_path}...")
    piper_voice = PiperVoice.load(str(voice_path))
    
    # 5. SLM Reasoning Engine (Ollama) & Speculator
    slm_cfg = cfg.get("slm", {})
    log.info(f"Connecting to Ollama model {slm_cfg.get('model_name')}...")
    slm_engine = SLMEngine(slm_cfg)
    speculator = Speculator(slm_cfg)
    
    # Pre-warm model into GPU memory with keep_alive='24h' and pin static system prompt prefix (Phase 2)
    try:
        import ollama
        from src.slm.prompt_templates import SYSTEM_PROMPT
        log.info("Pre-warming SLM into GPU VRAM (pinning system prompt with num_keep)...")
        ollama.chat(
            model=slm_cfg.get('model_name', 'qwen3.5:4b'),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "hello"}
            ],
            keep_alive="24h",
            think=False,
            options={
                "num_ctx": slm_cfg.get("num_ctx", 3072),
                "num_predict": 1,
                "num_keep": slm_engine.system_prompt_tokens,  # ⚡ Pinned on boot
            }
        )
        log.info(f"SLM system prompt prefix ({slm_engine.system_prompt_tokens} tokens) successfully pinned in GPU KV-cache!")
    except Exception as e:
        log.warning(f"Could not pre-warm SLM: {e}")
    
    log.info("All engines initialized successfully (Speculative execution ready)!")


def decode_audio_to_16k_pcm(audio_bytes: bytes, filename_hint: str = "audio.webm") -> np.ndarray:
    """
    Decodes audio bytes (WebM, MP4, WAV, OGG, etc.) from mobile browsers
    and resamples to 16kHz float32 mono using faster-whisper's decode_audio.
    """
    if not audio_bytes or len(audio_bytes) < 64:
        return np.zeros(0, dtype=np.float32)

    import tempfile
    from faster_whisper.audio import decode_audio

    ext = Path(filename_hint).suffix or ".webm"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name

        audio = decode_audio(temp_path, sampling_rate=16000)
        return audio
    except Exception as e:
        log.warning(f"faster-whisper decode_audio failed ({e}), attempting PyAV fallback...")
        try:
            container = av.open(io.BytesIO(audio_bytes))
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
            frames = []
            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    frames.append(resampled.to_ndarray())
            for resampled in resampler.resample(None):
                frames.append(resampled.to_ndarray())
            if frames:
                return np.concatenate(frames, axis=1).squeeze().astype(np.float32)
        except Exception as e2:
            log.error(f"PyAV fallback also failed: {e2}")
        return np.zeros(0, dtype=np.float32)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def synthesize_speech_wav(text: str) -> tuple[bytes, float]:
    """
    Synthesizes speech using Piper and returns (wav_bytes, duration_seconds).
    """
    global piper_voice, streaming_guardrail
    if piper_voice is None:
        raise RuntimeError("Piper voice model not loaded.")
        
    if streaming_guardrail is not None:
        text = streaming_guardrail.guard_chunk(text).text
        
    sample_rate = piper_voice.config.sample_rate
    chunks = []
    
    # Strip basic markdown before synthesis
    clean_text = text.replace("**", "").replace("*", "").replace("#", "")
    for chunk in piper_voice.synthesize(clean_text):
        chunks.append(chunk.audio_float_array)
        
    if not chunks:
        return b"", 0.0
        
    audio_combined = np.concatenate(chunks)
    duration_s = float(len(audio_combined) / sample_rate)
    
    # Convert float32 [-1, 1] to int16 PCM
    audio_int16 = np.clip(audio_combined * 32767.0, -32768.0, 32767.0).astype(np.int16)
    
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
        
    return wav_buf.getvalue(), duration_s


SUB_CLAUSE_REGEX = re.compile(
    r"(?:[,;:\-–—]|\b(?:and|but|because|so|while|however|therefore)\b|(?<=[.!?]))\s*",
    re.IGNORECASE
)


def synthesize_chunk_audio_wav(text: str) -> tuple[bytes, float, np.ndarray]:
    """
    Synthesizes a sub-clause chunk using Piper.
    Returns (wav_bytes, duration_seconds, int16_array).
    """
    global piper_voice, streaming_guardrail
    if piper_voice is None or not text.strip():
        return b"", 0.0, np.array([], dtype=np.int16)
        
    if streaming_guardrail is not None:
        text = streaming_guardrail.guard_chunk(text).text
        
    sample_rate = piper_voice.config.sample_rate
    clean_text = text.replace("**", "").replace("*", "").replace("#", "").strip()
    if not clean_text:
        return b"", 0.0, np.array([], dtype=np.int16)
        
    chunks = []
    for chunk in piper_voice.synthesize(clean_text):
        chunks.append(chunk.audio_float_array)
        
    if not chunks:
        return b"", 0.0, np.array([], dtype=np.int16)
        
    audio_combined = np.concatenate(chunks)
    duration_s = float(len(audio_combined) / sample_rate)
    audio_int16 = np.clip(audio_combined * 32767.0, -32768.0, 32767.0).astype(np.int16)
    
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
        
    return wav_buf.getvalue(), duration_s, audio_int16


class ChatRequest(BaseModel):
    message: str


@app.on_event("startup")
def on_startup():
    try:
        initialize_engines()
    except Exception as e:
        log.error(f"Failed to initialize engines on startup: {e}")


@app.get("/")
@app.head("/")
def get_index():
    if (REACT_DIST_DIR / "index.html").exists():
        resp = FileResponse(REACT_DIST_DIR / "index.html")
    else:
        resp = FileResponse(STATIC_DIR / "index.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.get("/api/status")
def get_status():
    vram_status = resource_scheduler.check_budget() if resource_scheduler else {}
    return {
        "status": "online",
        "lan_ip": get_lan_ip(),
        "port": 8000,
        "slm_model": cfg.get("slm", {}).get("model_name", "qwen3.5:4b"),
        "tts_voice": Path(cfg.get("tts", {}).get("voice_model", "")).name,
        "turn_count": session.turn_count,
        "active_red_flag": session.active_red_flag,
        "latest_slots": session.latest_slots,
        "last_telemetry": session.last_telemetry,
        "last_biomarkers": session.last_biomarkers,
        "speculative_enabled": session.speculative_enabled,
        "divergence_threshold": session.divergence_threshold,
        "vram": vram_status,
    }


class SettingsRequest(BaseModel):
    speculative_enabled: Optional[bool] = None
    divergence_threshold: Optional[float] = None


@app.post("/api/settings")
def update_settings(req: SettingsRequest):
    """Allows turning Speculative Execution ON or OFF directly from the frontend."""
    if req.speculative_enabled is not None:
        session.speculative_enabled = req.speculative_enabled
        mode_str = "⚡ SPECULATIVE CMSPE (Accelerated)" if session.speculative_enabled else "SEQUENTIAL BASELINE"
        log.info(f"⚡ [PIPELINE MODE] User toggled mode -> {mode_str}")
    if req.divergence_threshold is not None:
        session.divergence_threshold = req.divergence_threshold
        log.info(f"⚡ [SPECULATIVE] Divergence threshold set to {session.divergence_threshold}")
    return {
        "status": "ok",
        "speculative_enabled": session.speculative_enabled,
        "divergence_threshold": session.divergence_threshold,
    }


@app.get("/api/hospital/directory")
def get_hospital_directory():
    """Returns the loaded hospital rooms and emergency room for UI inspection."""
    if slm_engine and slm_engine.hospital_router:
        return {
            "status": "ok",
            "rooms": slm_engine.hospital_router.rooms,
            "emergency_room": slm_engine.hospital_router.emergency_room.__dict__ if slm_engine.hospital_router.emergency_room else None,
        }
    return {"status": "disabled", "rooms": []}


@app.get("/api/logs")
def get_logs(limit: int = 80):
    """Returns recent server terminal logs and latest turn latency telemetry."""
    vram_status = resource_scheduler.check_budget() if resource_scheduler else {}
    logs_slice = list(log_history)[-limit:]
    return {
        "logs": logs_slice,
        "latest_telemetry": session.last_telemetry,
        "turn": session.turn_count,
        "vram": vram_status,
    }


@app.post("/api/reset")
def reset_conversation():
    session.reset()
    log.info("Conversation session reset by user.")
    return {"status": "reset", "message": "New patient intake started."}


@app.get("/api/audio/{audio_id}")
def get_audio(audio_id: str):
    if audio_id not in audio_cache:
        raise HTTPException(status_code=404, detail="Audio expired or not found")
    return Response(content=audio_cache[audio_id], media_type="audio/wav")


@app.post("/api/chat")
def handle_chat_turn(req: ChatRequest):
    """Handles text-based conversational turn from the live chat UI."""
    t_start = time.time()
    user_text = req.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    log.info(f"Incoming text from patient: {user_text!r}")
    session.turn_count += 1
    session.history.append({"role": "user", "text": user_text})

    # 1. Deterministic clinical safety evaluation
    t_safety_start = time.time()
    handoff_decision = evaluate_handoff(
        user_text,
        session.turn_count,
        acoustic_biomarkers=None,
        hospital_router=slm_engine.hospital_router if slm_engine else None,
    )
    t_safety_end = time.time()

    is_emergency = handoff_decision.escalate
    override_message = handoff_decision.override_message
    red_flag_category = handoff_decision.red_flag_category
    
    if is_emergency:
        session.active_red_flag = red_flag_category or handoff_decision.reason
        agent_response = override_message
        t_slm_duration = 0.0
    else:
        # 2. SLM conversational generation
        t_slm_start = time.time()
        agent_response = slm_engine.generate(user_text, session.history)
        t_slm_duration = time.time() - t_slm_start

        # Post-hoc anti-fabrication check
        if contains_fabricated_placeholder(agent_response):
            log.warning("Placeholder fabrication detected in SLM response; substituting safe fallback.")
            agent_response = FABRICATION_FALLBACK_MESSAGE

    session.history.append({"role": "assistant", "text": agent_response})

    # 3. Piper TTS speech synthesis
    t_tts_start = time.time()
    wav_bytes, duration_s = synthesize_speech_wav(agent_response)
    t_tts_duration = time.time() - t_tts_start

    # Cache audio and create Base64 Data URL for instant mobile playback
    audio_id = str(uuid.uuid4())
    audio_cache[audio_id] = wav_bytes
    audio_base64 = f"data:audio/wav;base64,{base64.b64encode(wav_bytes).decode('utf-8')}"

    # 4. Asynchronous structured slot extraction (background thread so it never delays patient audio)
    def _extract_slots_bg(history_copy):
        try:
            recent = history_copy[-6:]
            extracted = extract_symptom_slots(recent, model_name=cfg.get("slm", {}).get("model_name", "qwen3.5:4b"))
            session.latest_slots = extracted.to_dict()
        except Exception as e:
            log.warning(f"Background slot extraction notice: {e}")

    import threading
    threading.Thread(target=_extract_slots_bg, args=(list(session.history),), daemon=True).start()
    t_extract_duration = 0.0

    total_latency_ms = int((time.time() - t_start) * 1000)
    ttfa_ms = int(t_tts_duration * 1000)  # Time to ready audio

    # Hardware status
    vram_status = resource_scheduler.check_budget() if resource_scheduler else {}

    spec_info = {
        "enabled": session.speculative_enabled,
        "committed": False,
        "similarity": 1.0,
        "status": "EMERGENCY_OVERRIDE" if is_emergency else ("TEXT_DIRECT" if session.speculative_enabled else "SEQUENTIAL")
    }

    telemetry = {
        "turn": session.turn_count,
        "total_latency_ms": total_latency_ms,
        "slm_latency_ms": int(t_slm_duration * 1000),
        "tts_latency_ms": int(t_tts_duration * 1000),
        "slot_extract_ms": int(t_extract_duration * 1000),
        "audio_duration_s": round(duration_s, 2),
        "is_emergency": is_emergency,
        "red_flag_category": red_flag_category,
        "speculative": spec_info,
    }
    session.last_telemetry = telemetry

    vram_used = vram_status.get('used_mb') or 0
    vram_tot = vram_status.get('total_mb') or 4096
    pipe_desc = "⚡ CMSPE Speculative Active" if session.speculative_enabled else "Sequential Baseline"
    log.info(
        f"\n{'='*70}\n"
        f"📊 TURN #{session.turn_count} LATENCY PROFILE (Text Input)\n"
        f"{'-'*70}\n"
        f"  ⚡ Pipeline Mode:        {pipe_desc}\n"
        f"  🧠 SLM Reasoning:        {int(t_slm_duration * 1000):>6} ms\n"
        f"  🔊 Piper TTS Synthesis:   {int(t_tts_duration * 1000):>6} ms\n"
        f"  📋 EHR Slot Extraction:  {int(t_extract_duration * 1000):>6} ms (Async)\n"
        f"{'-'*70}\n"
        f"  ⏱️ TOTAL GLASS-TO-GLASS: {total_latency_ms:>6} ms\n"
        f"  🖥️ GPU VRAM USAGE:       {vram_used:>6} MB / {vram_tot} MB\n"
        f"{'='*70}"
    )

    return {
        "user_message": user_text,
        "agent_response": agent_response,
        "audio_id": audio_id,
        "audio_url": f"/api/audio/{audio_id}",
        "audio_base64": audio_base64,
        "audio_duration_s": duration_s,
        "is_emergency": is_emergency,
        "red_flag_category": red_flag_category,
        "slots": session.latest_slots,
        "speculative": spec_info,
        "telemetry": telemetry,
        "vram": vram_status,
    }


def stream_chat_generator(user_text: str):
    """
    Server-Sent Events (SSE) generator for live streaming text and real-time sub-clause Piper TTS audio.
    Yields JSON lines formatted as `data: {...}\n\n`.
    """
    t_start = time.time()
    session.turn_count += 1
    session.history.append({"role": "user", "text": user_text})

    # 1. Deterministic clinical safety evaluation
    handoff_decision = evaluate_handoff(
        user_text,
        session.turn_count,
        acoustic_biomarkers=None,
        hospital_router=slm_engine.hospital_router if slm_engine else None,
    )
    is_emergency = handoff_decision.escalate
    override_message = handoff_decision.override_message
    red_flag_category = handoff_decision.red_flag_category

    sample_rate = piper_voice.config.sample_rate if piper_voice else 22050

    if is_emergency:
        session.active_red_flag = red_flag_category or handoff_decision.reason
        session.history.append({"role": "assistant", "text": override_message})

        # Send full emergency override tokens
        yield f"data: {json.dumps({'type': 'token', 'token': override_message})}\n\n"

        # Synthesize audio chunk
        wav_bytes, dur, _ = synthesize_chunk_audio_wav(override_message)
        audio_id = str(uuid.uuid4())
        audio_cache[audio_id] = wav_bytes
        audio_b64 = f"data:audio/wav;base64,{base64.b64encode(wav_bytes).decode('utf-8')}" if wav_bytes else ""
        yield f"data: {json.dumps({'type': 'audio_chunk', 'chunk_index': 0, 'audio_base64': audio_b64, 'duration': dur, 'text': override_message})}\n\n"

        total_lat_ms = int((time.time() - t_start) * 1000)
        yield f"data: {json.dumps({'type': 'done', 'agent_response': override_message, 'audio_base64': audio_b64, 'is_emergency': True, 'red_flag_category': red_flag_category, 'total_latency_ms': total_lat_ms})}\n\n"
        return

    # 2. SLM conversational generation with live streaming tokens & sub-clause TTS
    token_stream = slm_engine.generate_stream(user_text, session.history)
    if streaming_guardrail is not None:
        token_stream = streaming_guardrail.guard_token_stream(token_stream, user_prompt=user_text)

    buffer = ""
    full_response_parts = []
    chunk_index = 0
    all_int16_chunks = []

    for chunk in token_stream:
        tok = chunk.get("token", "")
        if not tok:
            continue

        buffer += tok
        full_response_parts.append(tok)

        # Stream token text to client immediately for live typewriter rendering
        yield f"data: {json.dumps({'type': 'token', 'token': tok})}\n\n"

        # Check sub-clause boundary (>= 3 words and punctuation/conjunction, or >= 6 words, or sentence end)
        words = buffer.split()
        has_sentence_end = any(p in buffer for p in [". ", "? ", "! ", ".\n", "?\n", "!\n"])
        has_sub_clause = (len(words) >= 3 and SUB_CLAUSE_REGEX.search(buffer)) or len(words) >= 6

        if has_sentence_end or has_sub_clause:
            clause_text = None
            parts = SUB_CLAUSE_REGEX.split(buffer, maxsplit=1)
            if len(parts) > 1 and parts[0].strip():
                clause_text = parts[0].strip()
                buffer = "".join(parts[1:])
            elif len(words) >= 6:
                clause_text = " ".join(words[:5])
                buffer = " ".join(words[5:])

            if clause_text:
                # Synthesize sub-clause audio chunk on the fly
                wav_bytes, dur, int16_arr = synthesize_chunk_audio_wav(clause_text)
                if wav_bytes:
                    if len(int16_arr) > 0:
                        all_int16_chunks.append(int16_arr)
                    audio_b64 = f"data:audio/wav;base64,{base64.b64encode(wav_bytes).decode('utf-8')}"
                    yield f"data: {json.dumps({'type': 'audio_chunk', 'chunk_index': chunk_index, 'audio_base64': audio_b64, 'duration': dur, 'text': clause_text})}\n\n"
                    chunk_index += 1

    # Remaining buffer audio
    if buffer.strip():
        clause_text = buffer.strip()
        wav_bytes, dur, int16_arr = synthesize_chunk_audio_wav(clause_text)
        if wav_bytes:
            if len(int16_arr) > 0:
                all_int16_chunks.append(int16_arr)
            audio_b64 = f"data:audio/wav;base64,{base64.b64encode(wav_bytes).decode('utf-8')}"
            yield f"data: {json.dumps({'type': 'audio_chunk', 'chunk_index': chunk_index, 'audio_base64': audio_b64, 'duration': dur, 'text': clause_text})}\n\n"
            chunk_index += 1

    assembled_response = "".join(full_response_parts).strip()
    if contains_fabricated_placeholder(assembled_response):
        log.warning("Placeholder fabrication detected in SLM response; substituting safe fallback.")
        assembled_response = FABRICATION_FALLBACK_MESSAGE

    session.history.append({"role": "assistant", "text": assembled_response})

    # Assemble complete concatenated WAV for persistent history replay
    full_audio_base64 = ""
    if all_int16_chunks:
        full_audio_int16 = np.concatenate(all_int16_chunks)
        full_buf = io.BytesIO()
        with wave.open(full_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(full_audio_int16.tobytes())
        full_wav_bytes = full_buf.getvalue()
        full_audio_id = str(uuid.uuid4())
        audio_cache[full_audio_id] = full_wav_bytes
        full_audio_base64 = f"data:audio/wav;base64,{base64.b64encode(full_wav_bytes).decode('utf-8')}"

    # Asynchronous structured slot extraction (background thread so it never delays phone response)
    def _extract_slots_bg(history_copy):
        try:
            recent = history_copy[-6:]
            extracted = extract_symptom_slots(recent, model_name=cfg.get("slm", {}).get("model_name", "qwen3.5:4b"))
            session.latest_slots = extracted.to_dict()
        except Exception as e:
            log.warning(f"Background slot extraction notice: {e}")

    threading.Thread(target=_extract_slots_bg, args=(list(session.history),), daemon=True).start()

    total_lat_ms = int((time.time() - t_start) * 1000)
    vram_status = resource_scheduler.check_budget() if resource_scheduler else {}

    yield f"data: {json.dumps({'type': 'done', 'agent_response': assembled_response, 'audio_base64': full_audio_base64, 'is_emergency': False, 'slots': session.latest_slots, 'total_latency_ms': total_lat_ms, 'vram': vram_status})}\n\n"


@app.post("/api/chat/stream")
def handle_chat_stream(req: ChatRequest):
    """Handles text-based conversational turn with real-time SSE token and audio chunk streaming."""
    user_text = req.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    return StreamingResponse(stream_chat_generator(user_text), media_type="text/event-stream")


def process_audio_pipeline(audio_bytes: bytes, filename_hint: str = "speech.webm") -> dict:
    """
    Core multimodal voice processing pipeline shared by REST and WebSocket Live Talk:
    1. Decodes audio to 16kHz float32 PCM.
    2. Extracts paralinguistic acoustic biomarkers (Google HeAR inspired).
    3. Triggers speculative CMSPE partial hypothesis with fast Whisper tiny.
    4. Transcribes full utterance with faster-whisper base.
    5. Evaluates deterministic clinical safety shield (red flags, acoustic distress, handoff).
    6. Resolves speculative draft vs fresh SLM generation.
    7. Synthesizes Piper neural TTS audio.
    8. Dispatches asynchronous EHR slot extraction in background.
    """
    t_start = time.time()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file received.")

    # 1. Decode audio to 16kHz float32 PCM using faster-whisper decode_audio
    t_decode_start = time.time()
    audio_pcm = decode_audio_to_16k_pcm(audio_bytes, filename_hint=filename_hint)
    t_decode_duration = time.time() - t_decode_start
    audio_duration_patient = len(audio_pcm) / 16000.0

    if len(audio_pcm) < 1600:  # < 100ms
        raise HTTPException(status_code=400, detail="Audio recording too brief to process.")

    # 2. Extract Paralinguistic Acoustic Biomarkers (Google HeAR inspired)
    t_bio_start = time.time()
    biomarkers = biomarker_coprocessor.analyze(audio_pcm)
    t_bio_duration = time.time() - t_bio_start
    session.last_biomarkers = biomarkers.to_dict()

    if biomarkers.acuity_boost > 0:
        log.warning(f"🚨 Acoustic distress alert: {biomarkers.clinical_rationale} (+{biomarkers.acuity_boost})")

    # Soft AGC / Peak Normalization to dramatically expand hearing range for quiet/distant phone mic audio
    max_amp = float(np.max(np.abs(audio_pcm))) if len(audio_pcm) > 0 else 0.0
    if 0.001 < max_amp < 0.6:
        gain = min(0.75 / max_amp, 4.0)
        audio_pcm = audio_pcm * gain
        log.info(f"🎙️ Acoustic AGC boost applied: {gain:.2f}x (Peak: {max_amp:.3f} -> {max_amp * gain:.3f})")

    # 3. Speculative Partial Pipeline Trigger (CMSPE Innovation)
    partial_text = ""
    speculative_info = {
        "enabled": session.speculative_enabled,
        "committed": False,
        "similarity": 0.0,
        "status": "sequential"
    }

    if session.speculative_enabled and len(audio_pcm) >= 12800 and whisper_partial is not None and speculator is not None:
        # Transcribe first ~65% of audio using ultra-fast tiny Whisper model
        partial_pcm = audio_pcm[: int(len(audio_pcm) * 0.65)]
        try:
            partial_segs, _ = whisper_partial.transcribe(
                partial_pcm, language="en", beam_size=1, condition_on_previous_text=False
            )
            partial_text = " ".join(seg.text for seg in partial_segs).strip()
            if len(partial_text.split()) >= 3:
                log.info(f"⚡ [SPECULATIVE CMSPE] Early partial hypothesis emitted: {partial_text!r}")
                speculator.start_speculation(partial_text, session.history)
        except Exception as e:
            log.warning(f"Speculation partial trigger error: {e}")

    # 4. Transcribe full final audio via faster-whisper base model
    t_asr_start = time.time()
    segments, _ = whisper_model.transcribe(
        audio_pcm,
        language="en",
        beam_size=2,
        condition_on_previous_text=False,
        initial_prompt="Hospital triage intake: patient describing symptoms, fever, body temperature, duration, pain."
    )
    transcribed_text = " ".join(seg.text for seg in segments).strip()
    t_asr_duration = time.time() - t_asr_start

    if not transcribed_text:
        transcribed_text = partial_text or "[Inaudible or silent audio]"

    log.info(f"🎙️ Transcribed phone speech: {transcribed_text!r} ({t_asr_duration:.2f}s)")
    session.turn_count += 1
    session.history.append({"role": "user", "text": transcribed_text})

    # 5. Deterministic Clinical Safety Layer
    handoff_decision = evaluate_handoff(
        transcribed_text,
        session.turn_count,
        acoustic_biomarkers=biomarkers,
        hospital_router=slm_engine.hospital_router if slm_engine else None,
    )

    is_emergency = handoff_decision.escalate
    override_message = handoff_decision.override_message
    red_flag_category = handoff_decision.red_flag_category

    if is_emergency:
        if speculator is not None:
            speculator.cancel()
        session.active_red_flag = red_flag_category or handoff_decision.reason
        agent_response = override_message
        t_slm_duration = 0.0
        speculative_info["status"] = "EMERGENCY_OVERRIDE"
    else:
        # 6. SLM Reasoning: Speculative Resolve vs Fresh Pass
        t_slm_start = time.time()
        committed = False
        spec_text = None
        sim = 0.0

        if session.speculative_enabled and partial_text and speculator is not None and (speculator.is_active() or speculator.has_result()):
            div = compute_divergence(partial_text, transcribed_text, threshold=session.divergence_threshold)
            sim = div.similarity
            if not div.diverged:
                spec_text = speculator.resolve(transcribed_text, wait_timeout_s=1.5)
                if spec_text:
                    agent_response = spec_text
                    committed = True
                    speculative_info.update({"committed": True, "similarity": round(sim, 2), "status": "COMMITTED"})
                    log.info(f"⚡ [SPECULATIVE CMSPE] COMMITTED! (Similarity: {sim:.2f} >= {session.divergence_threshold}) -> Zero perceived wait!")
                else:
                    speculator.cancel()
            else:
                speculator.cancel()
                log.info(f"🔄 [SPECULATIVE CMSPE] DIVERGED (Similarity: {sim:.2f} < {session.divergence_threshold}) -> Rollback to fresh pass")
                speculative_info.update({"committed": False, "similarity": round(sim, 2), "status": "ROLLED_BACK"})

        if not committed:
            if session.speculative_enabled and not partial_text:
                speculative_info["status"] = "SEQUENTIAL_FALLBACK"
            agent_response = slm_engine.generate(transcribed_text, session.history)

        t_slm_duration = time.time() - t_slm_start

        if contains_fabricated_placeholder(agent_response):
            log.warning("Placeholder fabrication detected in SLM response; substituting safe fallback.")
            agent_response = FABRICATION_FALLBACK_MESSAGE

    session.history.append({"role": "assistant", "text": agent_response})

    # 7. Piper TTS Synthesis
    t_tts_start = time.time()
    wav_bytes, duration_s = synthesize_speech_wav(agent_response)
    t_tts_duration = time.time() - t_tts_start

    audio_id = str(uuid.uuid4())
    audio_cache[audio_id] = wav_bytes
    audio_base64 = f"data:audio/wav;base64,{base64.b64encode(wav_bytes).decode('utf-8')}"

    # 8. Asynchronous Structured Slot Extraction (background thread so it never delays phone response)
    def _extract_slots_voice_bg(history_copy):
        try:
            recent = history_copy[-6:]
            extracted = extract_symptom_slots(recent, model_name=cfg.get("slm", {}).get("model_name", "qwen3.5:4b"))
            session.latest_slots = extracted.to_dict()
        except Exception as e:
            log.warning(f"Background slot extraction notice: {e}")

    import threading
    threading.Thread(target=_extract_slots_voice_bg, args=(list(session.history),), daemon=True).start()
    t_extract_duration = 0.0

    total_latency_ms = int((time.time() - t_start) * 1000)
    vram_status = resource_scheduler.check_budget() if resource_scheduler else {}

    telemetry = {
        "turn": session.turn_count,
        "total_latency_ms": total_latency_ms,
        "asr_latency_ms": int(t_asr_duration * 1000),
        "slm_latency_ms": int(t_slm_duration * 1000),
        "tts_latency_ms": int(t_tts_duration * 1000),
        "slot_extract_ms": int(t_extract_duration * 1000),
        "patient_audio_s": round(audio_duration_patient, 2),
        "agent_audio_s": round(duration_s, 2),
        "is_emergency": is_emergency,
        "red_flag_category": red_flag_category,
        "speculative": speculative_info,
    }
    session.last_telemetry = telemetry

    vram_used = vram_status.get('used_mb') or 0
    vram_tot = vram_status.get('total_mb') or 4096
    spec_status_desc = f"{speculative_info['status']} (Sim: {speculative_info['similarity']:.2f})" if speculative_info.get("enabled") else "OFF (Sequential Baseline)"
    log.info(
        f"\n{'='*70}\n"
        f"📊 TURN #{session.turn_count} LATENCY PROFILE (Voice Input / Live Talk)\n"
        f"{'-'*70}\n"
        f"  ⚡ Pipeline Mode:        {spec_status_desc}\n"
        f"  🎙️ Whisper ASR Decode:   {int(t_asr_duration * 1000):>6} ms (Audio: {audio_duration_patient:.2f}s)\n"
        f"  🫁 Paralinguistic Biomarkers: {int(t_bio_duration * 1000):>3} ms\n"
        f"  🧠 SLM Reasoning:        {int(t_slm_duration * 1000):>6} ms\n"
        f"  🔊 Piper TTS Synthesis:   {int(t_tts_duration * 1000):>6} ms\n"
        f"  📋 EHR Slot Extraction:  {int(t_extract_duration * 1000):>6} ms (Async)\n"
        f"{'-'*70}\n"
        f"  ⏱️ TOTAL GLASS-TO-GLASS: {total_latency_ms:>6} ms\n"
        f"  🖥️ GPU VRAM USAGE:       {vram_used:>6} MB / {vram_tot} MB\n"
        f"{'='*70}"
    )

    return {
        "transcribed_text": transcribed_text,
        "agent_response": agent_response,
        "audio_id": audio_id,
        "audio_url": f"/api/audio/{audio_id}",
        "audio_base64": audio_base64,
        "audio_duration_s": duration_s,
        "is_emergency": is_emergency,
        "red_flag_category": red_flag_category,
        "biomarkers": session.last_biomarkers,
        "slots": session.latest_slots,
        "speculative": speculative_info,
        "telemetry": telemetry,
        "vram": vram_status,
    }


@app.post("/api/voice")
async def handle_voice_turn(audio_file: UploadFile = File(...)):
    """
    Handles uploaded audio from phone browser (via Web Audio recorder or native phone mic).
    Decodes audio, extracts paralinguistic biomarkers, transcribes via faster-whisper,
    evaluates clinical safety, generates response, synthesizes TTS, and extracts EHR slots.
    """
    audio_bytes = await audio_file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file received.")

    log.info(f"Incoming voice file from phone: {len(audio_bytes)} bytes ({audio_file.content_type})")
    result = await asyncio.to_thread(process_audio_pipeline, audio_bytes, audio_file.filename or "speech.webm")
    return result


@app.post("/api/interrupt")
def handle_interrupt():
    """Cancels speculative background generation upon patient voice barge-in."""
    global speculator
    if speculator is not None:
        speculator.cancel()
    log.info("⚡ [VOICE BARGE-IN] Patient interrupted triage agent. Speculator generation cancelled.")
    return {"status": "interrupted"}


@app.websocket("/ws/live")
async def websocket_live_endpoint(websocket: WebSocket):
    """
    Full-duplex WebSocket endpoint for Live Talk (ChatGPT Live style).
    Receives voice utterances, processes them via the multi-stage pipeline, and streams
    real-time state updates, transcripts, paralinguistic biomarkers, and neural TTS audio.
    """
    await websocket.accept()
    log.info("🎙️ [LIVE TALK] WebSocket client connected for continuous hands-free session.")
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
            elif msg_type == "interrupt":
                if speculator is not None:
                    speculator.cancel()
                log.info("⚡ [LIVE TALK BARGE-IN] Patient interrupted via WebSocket.")
                await websocket.send_json({"type": "interrupted"})
            elif msg_type in ("chat", "text"):
                text_msg = data.get("message") or data.get("text", "")
                if text_msg:
                    await websocket.send_json({"type": "state", "status": "processing"})
                    for sse_event in stream_chat_generator(text_msg):
                        if sse_event.startswith("data: "):
                            ev_data = json.loads(sse_event[6:].strip())
                            await websocket.send_json(ev_data)
            elif msg_type in ("audio", "voice", "voice_chunk"):
                raw_b64 = data.get("audio_base64", "")
                if "," in raw_b64:
                    raw_b64 = raw_b64.split(",", 1)[1]
                if not raw_b64:
                    continue
                audio_bytes = base64.b64decode(raw_b64)
                await websocket.send_json({"type": "state", "status": "processing"})
                result = await asyncio.to_thread(process_audio_pipeline, audio_bytes, "live.webm")
                await websocket.send_json({"type": "response", **result})
    except WebSocketDisconnect:
        log.info("🎙️ [LIVE TALK] WebSocket client disconnected normally.")
    except Exception as e:
        log.warning(f"🎙️ [LIVE TALK] WebSocket connection ended: {e}")


# Mount static assets
if (REACT_DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(REACT_DIST_DIR / "assets")), name="react_assets")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
