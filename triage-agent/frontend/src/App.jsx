import React, { useState, useEffect, useRef } from 'react';
import {
  Zap,
  Mic,
  MicOff,
  Send,
  Square,
  FileText,
  Activity,
  Terminal,
  RotateCcw,
  Volume2,
  VolumeX,
  AlertTriangle,
  X,
  Copy,
  Trash2,
  Cpu,
  Heart,
  ChevronRight,
  ShieldCheck,
  Stethoscope,
  Upload,
  Radio,
  PhoneOff,
  MessageSquare
} from 'lucide-react';

/* ==========================================================================
   VoiceOrb Component - ChatGPT Live Style Organic Reactive Audio Orb
   ========================================================================== */
function VoiceOrb({ state, audioLevel, isEmergency }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animId;
    let phase = 0;

    const render = () => {
      phase += 0.035;
      const width = canvas.width;
      const height = canvas.height;
      const cx = width / 2;
      const cy = height / 2;

      ctx.clearRect(0, 0, width, height);

      // Dynamic color palette based on active conversational state
      let color1, color2, color3, glowColor, midGlowColor;
      if (isEmergency) {
        color1 = 'rgba(239, 68, 68, 0.95)';
        color2 = 'rgba(220, 38, 38, 0.8)';
        color3 = 'rgba(254, 202, 202, 0.5)';
        glowColor = 'rgba(239, 68, 68, 0.55)';
        midGlowColor = 'rgba(239, 68, 68, 0.18)';
      } else if (state === 'speaking') {
        color1 = 'rgba(168, 85, 247, 0.95)'; // violet
        color2 = 'rgba(99, 102, 241, 0.85)';  // indigo
        color3 = 'rgba(6, 182, 212, 0.75)';   // cyan
        glowColor = 'rgba(168, 85, 247, 0.55)';
        midGlowColor = 'rgba(168, 85, 247, 0.18)';
      } else if (state === 'analyzing') {
        color1 = 'rgba(245, 158, 11, 0.95)';  // amber
        color2 = 'rgba(14, 165, 233, 0.85)';  // cyan
        color3 = 'rgba(139, 92, 246, 0.7)';   // violet
        glowColor = 'rgba(245, 158, 11, 0.5)';
        midGlowColor = 'rgba(245, 158, 11, 0.18)';
      } else if (state === 'interrupted') {
        color1 = 'rgba(251, 146, 60, 0.95)';  // orange
        color2 = 'rgba(239, 68, 68, 0.75)';
        color3 = 'rgba(254, 215, 170, 0.5)';
        glowColor = 'rgba(251, 146, 60, 0.55)';
        midGlowColor = 'rgba(251, 146, 60, 0.18)';
      } else if (state === 'muted') {
        color1 = 'rgba(100, 116, 139, 0.6)';
        color2 = 'rgba(71, 85, 105, 0.5)';
        color3 = 'rgba(148, 163, 184, 0.3)';
        glowColor = 'rgba(100, 116, 139, 0.2)';
        midGlowColor = 'rgba(100, 116, 139, 0.08)';
      } else {
        // Listening / idle ambient breathing
        color1 = 'rgba(14, 165, 233, 0.95)'; // cyan
        color2 = 'rgba(16, 185, 129, 0.85)'; // emerald
        color3 = 'rgba(56, 189, 248, 0.65)';  // sky
        glowColor = 'rgba(14, 165, 233, 0.5)';
        midGlowColor = 'rgba(14, 165, 233, 0.18)';
      }

      const baseR = Math.min(width, height) * 0.2;
      const boost = Math.min(audioLevel * 65, width * 0.18);
      const r = baseR + boost;

      // 1. Ambient Outer Halo
      const haloGrad = ctx.createRadialGradient(cx, cy, r * 0.3, cx, cy, r * 1.85);
      haloGrad.addColorStop(0, glowColor);
      haloGrad.addColorStop(0.5, midGlowColor);
      haloGrad.addColorStop(1, 'transparent');
      ctx.fillStyle = haloGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, r * 1.85, 0, Math.PI * 2);
      ctx.fill();

      // 2. Harmonic Fluid Deformed Waveform Boundary
      ctx.save();
      ctx.beginPath();
      const points = 64;
      for (let i = 0; i <= points; i++) {
        const theta = (i / points) * Math.PI * 2;
        const speed = state === 'speaking' ? 3.8 : (state === 'analyzing' ? 5.2 : 2.2);
        const amp = state === 'speaking' ? 14 : (state === 'listening' ? 7 + audioLevel * 22 : 5);
        const wave = Math.sin(theta * 4 + phase * speed) * amp
                   + Math.cos(theta * 2 - phase * (speed * 0.7)) * (amp * 0.5);
        const curR = r + wave;
        const x = cx + Math.cos(theta) * curR;
        const y = cy + Math.sin(theta) * curR;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();

      const coreGrad = ctx.createRadialGradient(cx - r * 0.25, cy - r * 0.25, r * 0.1, cx, cy, r * 1.25);
      coreGrad.addColorStop(0, color3);
      coreGrad.addColorStop(0.35, color1);
      coreGrad.addColorStop(1, color2);
      ctx.fillStyle = coreGrad;
      ctx.shadowColor = glowColor;
      ctx.shadowBlur = 30 + audioLevel * 35;
      ctx.fill();
      ctx.restore();

      // 3. Orbital Celestial Rings
      const ringRot = phase * (state === 'analyzing' ? 2.8 : 0.9);
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(ringRot);
      ctx.beginPath();
      ctx.ellipse(0, 0, r * 1.35, r * 0.65, Math.PI / 4, 0, Math.PI * 2);
      ctx.strokeStyle = color3;
      ctx.lineWidth = 1.8;
      ctx.shadowColor = color3;
      ctx.shadowBlur = 10;
      ctx.stroke();

      ctx.beginPath();
      ctx.ellipse(0, 0, r * 1.45, r * 0.55, -Math.PI / 4, 0, Math.PI * 2);
      ctx.strokeStyle = color2;
      ctx.lineWidth = 1.2;
      ctx.stroke();
      ctx.restore();

      animId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animId);
  }, [state, audioLevel, isEmergency]);

  return (
    <div className={`live-orb-wrapper ${state} ${isEmergency ? 'emergency' : ''}`}>
      <div className="live-orb-ambient-glow" />
      <canvas ref={canvasRef} width={280} height={280} className="live-orb-canvas" />
    </div>
  );
}

export default function App() {
  // App State
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      role: 'assistant',
      text: 'Hello, I am your virtual triage assistant. Please describe what brings you in today — where do you feel discomfort, how severe is it, and when did it start?',
      audioBase64: null,
      isEmergency: false,
      speculative: null,
      time: 'Just now',
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [speculativeEnabled, setSpeculativeEnabled] = useState(true);
  const [autoPlay, setAutoPlay] = useState(true);

  // Modals & Drawers
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [isTerminalOpen, setIsTerminalOpen] = useState(false);
  const [activeEmergency, setActiveEmergency] = useState(null);

  // Clinical & Telemetry Data
  const [slots, setSlots] = useState({
    chief_complaint: null,
    duration: null,
    severity: null,
    associated_symptoms: [],
    recommended_department: null,
    confidence: 'low',
    red_flag_present: false,
  });
  const [biomarkers, setBiomarkers] = useState(null);
  const [telemetry, setTelemetry] = useState({});
  const [vram, setVram] = useState({ used_mb: 0, total_mb: 4096 });
  const [lanIp, setLanIp] = useState('localhost');
  const [logs, setLogs] = useState([]);
  const [playingAudioId, setPlayingAudioId] = useState(null);

  // Recording State
  const [isRecording, setIsRecording] = useState(false);
  const [recDuration, setRecDuration] = useState(0);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recTimerRef = useRef(null);

  const chatEndRef = useRef(null);
  const audioElementRef = useRef(new Audio());
  const fileInputRef = useRef(null);

  // Live Talk Mode State (ChatGPT Live Full-Duplex Style)
  const [isLiveTalkOpen, setIsLiveTalkOpen] = useState(() => {
    try {
      return new URLSearchParams(window.location.search).get('live') === '1';
    } catch {
      return false;
    }
  });
  const [liveState, setLiveState] = useState('listening'); // 'listening' | 'analyzing' | 'speaking' | 'interrupted' | 'muted'
  const [isMicMuted, setIsMicMuted] = useState(false);
  const [liveSubtitles, setLiveSubtitles] = useState({ user: '', assistant: '' });
  const [liveAudioLevel, setLiveAudioLevel] = useState(0);
  const [liveSessionSeconds, setLiveSessionSeconds] = useState(0);
  const [isLiveTranscriptVisible, setIsLiveTranscriptVisible] = useState(false);

  // Live Talk Audio & WebSocket Refs
  const liveWsRef = useRef(null);
  const liveAudioContextRef = useRef(null);
  const liveAnalyserRef = useRef(null);
  const liveMediaStreamRef = useRef(null);
  const liveMediaRecorderRef = useRef(null);
  const liveAudioChunksRef = useRef([]);
  const liveSilenceTimerRef = useRef(null);
  const liveSpeechStartTimeRef = useRef(0);
  const isSpeakingUserRef = useRef(false);
  const liveTtsAudioRef = useRef(new Audio());
  const liveSessionTimerRef = useRef(null);
  const liveAnimFrameRef = useRef(null);
  const liveStateRef = useRef('listening');
  const isMicMutedRef = useRef(false);

  // Keep state refs synchronized for 60fps RAF loop
  useEffect(() => {
    liveStateRef.current = liveState;
  }, [liveState]);

  useEffect(() => {
    isMicMutedRef.current = isMicMuted;
  }, [isMicMuted]);

  // Scroll to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isProcessing]);

  // Initial Status Fetch
  useEffect(() => {
    fetchStatus();
  }, []);

  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        if (data.speculative_enabled !== undefined) {
          setSpeculativeEnabled(data.speculative_enabled);
        }
        if (data.vram) setVram(data.vram);
        if (data.lan_ip) setLanIp(data.lan_ip);
        if (data.latest_slots && data.latest_slots.chief_complaint) {
          setSlots(data.latest_slots);
        }
        if (data.last_biomarkers) setBiomarkers(data.last_biomarkers);
        if (data.last_telemetry) setTelemetry(data.last_telemetry);
      }
    } catch (e) {
      console.warn('Initial status fetch error:', e);
    }
  };

  // Toggle Speculative Mode
  const toggleSpeculativeMode = async (targetValue) => {
    const nextVal = targetValue !== undefined ? targetValue : !speculativeEnabled;
    try {
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speculative_enabled: nextVal }),
      });
      if (res.ok) {
        const data = await res.json();
        setSpeculativeEnabled(data.speculative_enabled);
      }
    } catch (e) {
      console.error('Failed to toggle speculative mode:', e);
    }
  };

  // Reset Session
  const handleReset = async () => {
    if (!window.confirm('Start a new patient intake session?')) return;
    try {
      await fetch('/api/reset', { method: 'POST' });
      setMessages([
        {
          id: 'welcome-' + Date.now(),
          role: 'assistant',
          text: 'Hello, I am your virtual triage assistant. Please describe what brings you in today — where do you feel discomfort, how severe is it, and when did it start?',
          audioBase64: null,
          isEmergency: false,
          speculative: null,
          time: 'Just now',
        },
      ]);
      setActiveEmergency(null);
      setSlots({
        chief_complaint: null,
        duration: null,
        severity: null,
        associated_symptoms: [],
        recommended_department: null,
        confidence: 'low',
        red_flag_present: false,
      });
      setBiomarkers(null);
      setTelemetry({});
    } catch (e) {
      console.error('Reset error:', e);
    }
  };

  // Log Polling when Terminal Open
  useEffect(() => {
    let interval = null;
    if (isTerminalOpen) {
      fetchLogs();
      interval = setInterval(fetchLogs, 1500);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isTerminalOpen]);

  const fetchLogs = async () => {
    try {
      const res = await fetch('/api/logs?limit=100');
      if (res.ok) {
        const data = await res.json();
        if (data.logs) setLogs(data.logs);
        if (data.latest_telemetry) setTelemetry(data.latest_telemetry);
        if (data.vram) setVram(data.vram);
      }
    } catch (e) {
      console.warn('Fetch logs error:', e);
    }
  };

  // Audio Playback
  const playAudio = (audioBase64, msgId) => {
    if (!audioBase64) return;
    try {
      audioElementRef.current.src = audioBase64;
      audioElementRef.current.play().then(() => {
        setPlayingAudioId(msgId);
      }).catch((e) => {
        console.warn('Audio play restricted by browser policy:', e);
      });
      audioElementRef.current.onended = () => {
        setPlayingAudioId(null);
      };
    } catch (e) {
      console.error('Audio playback error:', e);
    }
  };

  // Send Text Turn
  const handleSendText = async (textToSend) => {
    const text = (textToSend || inputText).trim();
    if (!text || isProcessing) return;

    setInputText('');
    const userMsg = {
      id: 'user-' + Date.now(),
      role: 'user',
      text,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsProcessing(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      processAssistantTurn(data);
    } catch (e) {
      console.error('Chat error:', e);
      setMessages((prev) => [
        ...prev,
        {
          id: 'err-' + Date.now(),
          role: 'assistant',
          text: 'I encountered an issue contacting the clinical model. Please ensure Ollama is active.',
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
      ]);
    } finally {
      setIsProcessing(false);
    }
  };

  // Send Voice Turn
  const sendVoiceTurn = async (audioBlob) => {
    if (isProcessing) return;

    const userMsgId = 'voice-' + Date.now();
    setMessages((prev) => [
      ...prev,
      {
        id: userMsgId,
        role: 'user',
        text: '🎙️ [Processing phone speech recording...]',
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ]);
    setIsProcessing(true);

    try {
      const formData = new FormData();
      const ext = audioBlob.type.includes('mp4') ? 'mp4' : 'webm';
      formData.append('audio_file', audioBlob, `speech_turn.${ext}`);

      const res = await fetch('/api/voice', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${res.status}`);
      }

      const data = await res.json();

      // Update the user message with transcribed speech
      setMessages((prev) =>
        prev.map((m) =>
          m.id === userMsgId
            ? { ...m, text: `🎙️ "${data.transcribed_text}"` }
            : m
        )
      );

      processAssistantTurn(data);
    } catch (e) {
      console.error('Voice turn error:', e);
      setMessages((prev) => [
        ...prev,
        {
          id: 'err-' + Date.now(),
          role: 'assistant',
          text: `Voice processing error: ${e.message}. Please try again or type symptoms.`,
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
      ]);
    } finally {
      setIsProcessing(false);
    }
  };

  // Process Server Response
  const processAssistantTurn = (data) => {
    const assistantMsgId = 'asst-' + Date.now();
    const assistantMsg = {
      id: assistantMsgId,
      role: 'assistant',
      text: data.agent_response,
      audioBase64: data.audio_base64,
      isEmergency: data.is_emergency,
      redFlagCategory: data.red_flag_category,
      speculative: data.speculative,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, assistantMsg]);

    if (data.is_emergency) {
      setActiveEmergency({
        category: data.red_flag_category || 'EMERGENCY ESCALATION',
        message: data.agent_response,
      });
    }

    if (data.slots) setSlots(data.slots);
    if (data.biomarkers) setBiomarkers(data.biomarkers);
    if (data.telemetry) setTelemetry(data.telemetry);
    if (data.vram) setVram(data.vram);

    // Auto-play speech on phone
    if (autoPlay && data.audio_base64) {
      playAudio(data.audio_base64, assistantMsgId);
    }
  };

  // Start In-Browser Microphone Recording
  const startRecording = async () => {
    try {
      audioChunksRef.current = [];
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      let mimeType = 'audio/webm;codecs=opus';
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = MediaRecorder.isTypeSupported('audio/mp4') ? 'audio/mp4' : '';
      }

      const mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        if (audioChunksRef.current.length > 0) {
          const blob = new Blob(audioChunksRef.current, {
            type: mimeType || 'audio/webm',
          });
          sendVoiceTurn(blob);
        }
      };

      mediaRecorder.start(250);
      setIsRecording(true);
      setRecDuration(0);

      recTimerRef.current = setInterval(() => {
        setRecDuration((d) => d + 1);
      }, 1000);
    } catch (e) {
      console.warn('Microphone error:', e);
      alert('Unable to access microphone directly. Tap the upload button to record using your phone’s native voice recorder!');
    }
  };

  const stopRecording = () => {
    if (recTimerRef.current) clearInterval(recTimerRef.current);
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  };

  const cancelRecording = () => {
    if (recTimerRef.current) clearInterval(recTimerRef.current);
    audioChunksRef.current = [];
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  };

  // Native Phone File Input
  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      sendVoiceTurn(file);
    }
    e.target.value = '';
  };

  const formatTimer = (secs) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  // ==========================================================================
  // Live Talk Mode Functions (ChatGPT Live Hands-Free Full-Duplex Engine)
  // ==========================================================================

  const openLiveTalk = () => {
    setIsLiveTalkOpen(true);
    setLiveSessionSeconds(0);
    setLiveSubtitles({ user: '', assistant: '' });
    if (liveSessionTimerRef.current) clearInterval(liveSessionTimerRef.current);
    liveSessionTimerRef.current = setInterval(() => {
      setLiveSessionSeconds((s) => s + 1);
    }, 1000);
    startLiveAudioCapture();
  };

  const closeLiveTalk = () => {
    setIsLiveTalkOpen(false);
    if (liveSessionTimerRef.current) clearInterval(liveSessionTimerRef.current);
    if (liveSilenceTimerRef.current) clearTimeout(liveSilenceTimerRef.current);
    if (liveAnimFrameRef.current) cancelAnimationFrame(liveAnimFrameRef.current);

    if (liveTtsAudioRef.current) {
      liveTtsAudioRef.current.pause();
      liveTtsAudioRef.current.currentTime = 0;
    }

    if (liveMediaStreamRef.current) {
      liveMediaStreamRef.current.getTracks().forEach((track) => track.stop());
      liveMediaStreamRef.current = null;
    }

    if (liveAudioContextRef.current) {
      liveAudioContextRef.current.close().catch(() => {});
      liveAudioContextRef.current = null;
    }

    if (liveWsRef.current) {
      liveWsRef.current.close();
      liveWsRef.current = null;
    }

    isSpeakingUserRef.current = false;
    setLiveState('listening');
  };

  const toggleLiveMic = () => {
    const next = !isMicMuted;
    setIsMicMuted(next);
    if (liveMediaStreamRef.current) {
      liveMediaStreamRef.current.getAudioTracks().forEach((t) => {
        t.enabled = !next;
      });
    }
    if (next) setLiveState('muted');
    else setLiveState('listening');
  };

  const interruptLiveSpeech = () => {
    if (liveTtsAudioRef.current) {
      liveTtsAudioRef.current.pause();
      liveTtsAudioRef.current.currentTime = 0;
    }
    if (liveWsRef.current && liveWsRef.current.readyState === WebSocket.OPEN) {
      liveWsRef.current.send(JSON.stringify({ type: 'interrupt' }));
    }
    fetch('/api/interrupt', { method: 'POST' }).catch(() => {});
    setLiveState('interrupted');
    setTimeout(() => {
      setLiveState('listening');
    }, 350);
  };

  const connectLiveWebSocket = () => {
    try {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${proto}//${window.location.host}/ws/live`;
      const ws = new WebSocket(wsUrl);
      liveWsRef.current = ws;

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'response') {
            handleLiveServerResponse(msg);
          } else if (msg.type === 'state') {
            if (msg.status === 'processing') {
              setLiveState('analyzing');
            }
          }
        } catch (e) {
          console.warn('WS message parse error:', e);
        }
      };

      ws.onerror = (e) => {
        console.warn('Live Talk WS error, will fallback to HTTP:', e);
      };
    } catch (e) {
      console.warn('WS setup error:', e);
    }
  };

  const handleLiveAudioUpload = async (blob) => {
    setLiveState('analyzing');
    if (liveWsRef.current && liveWsRef.current.readyState === WebSocket.OPEN) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64data = reader.result;
        liveWsRef.current.send(JSON.stringify({
          type: 'voice',
          audio_base64: base64data,
        }));
      };
      reader.readAsDataURL(blob);
    } else {
      try {
        const formData = new FormData();
        const ext = blob.type.includes('mp4') ? 'mp4' : 'webm';
        formData.append('audio_file', blob, `live_speech.${ext}`);
        const res = await fetch('/api/voice', { method: 'POST', body: formData });
        if (res.ok) {
          const data = await res.json();
          handleLiveServerResponse(data);
        } else {
          throw new Error('HTTP error');
        }
      } catch (err) {
        console.error('Live voice HTTP fallback error:', err);
        setLiveState('listening');
      }
    }
  };

  const handleLiveServerResponse = (data) => {
    setLiveSubtitles({
      user: data.transcribed_text,
      assistant: data.agent_response,
    });

    const userMsgId = 'user-live-' + Date.now();
    const asstMsgId = 'asst-live-' + Date.now();
    setMessages((prev) => [
      ...prev,
      {
        id: userMsgId,
        role: 'user',
        text: `🎙️ "${data.transcribed_text}"`,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
      {
        id: asstMsgId,
        role: 'assistant',
        text: data.agent_response,
        audioBase64: data.audio_base64,
        isEmergency: data.is_emergency,
        redFlagCategory: data.red_flag_category,
        speculative: data.speculative,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ]);

    if (data.is_emergency) {
      setActiveEmergency({
        category: data.red_flag_category || 'EMERGENCY ESCALATION',
        message: data.agent_response,
      });
    }

    if (data.slots) setSlots(data.slots);
    if (data.biomarkers) setBiomarkers(data.biomarkers);
    if (data.telemetry) setTelemetry(data.telemetry);
    if (data.vram) setVram(data.vram);

    // Play synthesized voice and animate the VoiceOrb
    if (data.audio_base64) {
      setLiveState('speaking');
      liveTtsAudioRef.current.src = data.audio_base64;
      liveTtsAudioRef.current.play().catch((e) => console.warn('Play error:', e));
      liveTtsAudioRef.current.onended = () => {
        if (liveStateRef.current !== 'muted') {
          setLiveState('listening');
        }
      };
    } else {
      setLiveState('listening');
    }
  };

  const startLiveAudioCapture = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      liveMediaStreamRef.current = stream;

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioCtx({ sampleRate: 16000 });
      liveAudioContextRef.current = ctx;
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.7;
      source.connect(analyser);
      liveAnalyserRef.current = analyser;

      let mimeType = 'audio/webm;codecs=opus';
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = MediaRecorder.isTypeSupported('audio/mp4') ? 'audio/mp4' : '';
      }

      const createRecorder = () => {
        const mr = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
        mr.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) {
            liveAudioChunksRef.current.push(e.data);
          }
        };
        mr.onstop = () => {
          if (liveAudioChunksRef.current.length > 0) {
            const blob = new Blob(liveAudioChunksRef.current, {
              type: mimeType || 'audio/webm',
            });
            liveAudioChunksRef.current = [];
            handleLiveAudioUpload(blob);
          }
        };
        return mr;
      };

      liveMediaRecorderRef.current = createRecorder();
      connectLiveWebSocket();

      const timeData = new Uint8Array(analyser.frequencyBinCount);
      let smoothRms = 0;

      const vadLoop = () => {
        if (!liveAudioContextRef.current) return;
        analyser.getByteTimeDomainData(timeData);

        let sum = 0;
        for (let i = 0; i < timeData.length; i++) {
          const v = (timeData[i] - 128) / 128;
          sum += v * v;
        }
        const rms = Math.sqrt(sum / timeData.length);
        smoothRms = smoothRms * 0.7 + rms * 0.3;
        setLiveAudioLevel(smoothRms);

        const curState = liveStateRef.current;
        const muted = isMicMutedRef.current;

        // Barge-in check: If AI is speaking and user vocalizes loudly
        if (curState === 'speaking' && !muted && smoothRms > 0.08) {
          interruptLiveSpeech();
        }

        // VAD in listening mode
        if (curState === 'listening' && !muted) {
          const speechThreshold = 0.032;
          if (smoothRms >= speechThreshold) {
            if (!isSpeakingUserRef.current) {
              isSpeakingUserRef.current = true;
              liveSpeechStartTimeRef.current = Date.now();
              liveAudioChunksRef.current = [];
              if (liveSilenceTimerRef.current) clearTimeout(liveSilenceTimerRef.current);
              try {
                if (liveMediaRecorderRef.current.state === 'inactive') {
                  liveMediaRecorderRef.current.start(200);
                }
              } catch (e) {}
            } else {
              if (liveSilenceTimerRef.current) {
                clearTimeout(liveSilenceTimerRef.current);
                liveSilenceTimerRef.current = null;
              }
            }
          } else if (isSpeakingUserRef.current) {
            if (!liveSilenceTimerRef.current) {
              liveSilenceTimerRef.current = setTimeout(() => {
                const speechDuration = Date.now() - liveSpeechStartTimeRef.current;
                if (speechDuration >= 450) {
                  setLiveState('analyzing');
                  try {
                    if (liveMediaRecorderRef.current && liveMediaRecorderRef.current.state === 'recording') {
                      liveMediaRecorderRef.current.stop();
                    }
                  } catch (e) {}
                }
                isSpeakingUserRef.current = false;
                liveSilenceTimerRef.current = null;
              }, 800);
            }
          }
        }

        liveAnimFrameRef.current = requestAnimationFrame(vadLoop);
      };

      liveAnimFrameRef.current = requestAnimationFrame(vadLoop);
      setLiveState('listening');
    } catch (err) {
      console.error('Failed to start Live Audio:', err);
      alert('Could not access microphone for Live Talk. Please grant microphone permissions in your browser.');
      setIsLiveTalkOpen(false);
    }
  };

  return (
    <div className="app-root">
      {/* Background glow effects */}
      <div className="bg-ambient">
        <div className="ambient-glow-1"></div>
        <div className="ambient-glow-2"></div>
        <div className="ambient-grid"></div>
      </div>

      {/* Top Header */}
      <header className="top-header">
        <div className="header-brand">
          <div className="brand-icon">
            <Stethoscope size={20} />
          </div>
          <div className="brand-meta">
            <h1 className="brand-title">
              <span className="brand-title-full">OmniSpec-Triage</span>
              <span className="brand-title-compact">OmniSpec</span>
            </h1>
            <span className="brand-sub">Clinical Intake AI</span>
          </div>
        </div>

        <div className="header-actions">
          {/* Live Talk Button (ChatGPT Live Full-Duplex Voice) */}
          <button
            type="button"
            className={`btn-live-talk-head ${isLiveTalkOpen ? 'active' : ''}`}
            onClick={openLiveTalk}
            title="Enter Hands-Free Live Voice Mode (Like ChatGPT Live)"
          >
            <span className="live-talk-pulse"></span>
            <Radio size={14} className="live-radio-icon" />
            <span className="btn-live-talk-label-full">LIVE TALK</span>
            <span className="btn-live-talk-label-compact">LIVE</span>
          </button>

          {/* Speculative Streaming Pill Toggle (Thumb friendly on Redmi 13C) */}
          <button
            type="button"
            className={`btn-spec-pill ${speculativeEnabled ? 'active' : 'inactive'}`}
            onClick={() => toggleSpeculativeMode()}
            title={
              speculativeEnabled
                ? 'Speculative Streaming is ON (CMSPE). Tap to switch to Sequential Baseline.'
                : 'Sequential Baseline is ON. Tap to activate Speculative Streaming (CMSPE).'
            }
          >
            <Zap size={14} className="spec-icon-spin" />
            <span className="spec-label-full">{speculativeEnabled ? 'SPEC: ON' : 'SEQ: OFF'}</span>
            <span className="spec-label-compact">{speculativeEnabled ? 'SPEC' : 'SEQ'}</span>
          </button>

          {/* Terminal & Latency Stream Button */}
          <button
            type="button"
            className="btn-head-icon"
            onClick={() => setIsTerminalOpen(true)}
            title="Live Terminal Logs & Latency Profile"
          >
            <Terminal size={17} />
            <span className="head-badge" style={{ background: '#10B981' }}>LOGS</span>
          </button>

          {/* Clinical EHR Drawer Button */}
          <button
            type="button"
            className="btn-head-icon"
            onClick={() => setIsDrawerOpen(true)}
            title="Clinical EHR & Telemetry"
          >
            <Activity size={18} />
            <span className="head-badge" style={{ background: '#0EA5E9' }}>EHR</span>
          </button>

          {/* Reset Session */}
          <button
            type="button"
            className="btn-head-icon"
            onClick={handleReset}
            title="Start New Patient Session"
          >
            <RotateCcw size={16} />
          </button>
        </div>
      </header>

      {/* Emergency Red-Flag Banner */}
      {activeEmergency && (
        <section className="emergency-strip">
          <div className="emergency-info">
            <AlertTriangle size={20} color="#FCA5A5" />
            <div>
              <strong style={{ fontSize: '0.85rem' }}>EMERGENCY CLINICAL OVERRIDE</strong>
              <div style={{ fontSize: '0.72rem', opacity: 0.9 }}>
                Immediate physician intervention required.
              </div>
            </div>
          </div>
          <span className="emergency-badge">{activeEmergency.category}</span>
        </section>
      )}

      {/* Main Workspace */}
      <main className="main-workspace">
        {/* Chat Feed & Input */}
        <section className="chat-container">
          <div className="chat-feed">
            {messages.map((m) => (
              <div key={m.id} className={`msg-row ${m.role} ${m.isEmergency ? 'emergency' : ''}`}>
                <div className="msg-avatar">
                  {m.role === 'user' ? <Heart size={16} /> : <Stethoscope size={16} />}
                </div>
                <div className="msg-bubble">
                  <div className="msg-meta-row">
                    <span className="msg-author">
                      {m.role === 'user'
                        ? 'Patient'
                        : m.isEmergency
                        ? '🚨 Emergency Directive'
                        : 'Triage Nurse AI'}
                    </span>
                    <span>{m.time}</span>
                  </div>
                  <div>{m.text}</div>
                  {m.role === 'assistant' && (
                    <div className="msg-actions-row">
                      {m.audioBase64 && (
                        <button
                          type="button"
                          className={`btn-play-voice ${playingAudioId === m.id ? 'playing' : ''}`}
                          onClick={() => playAudio(m.audioBase64, m.id)}
                        >
                          <Volume2 size={13} />
                          <span>{playingAudioId === m.id ? 'Speaking...' : 'Play Voice'}</span>
                        </button>
                      )}

                      {/* Speculation status badge */}
                      {m.isEmergency ? (
                        <span className="badge-pill emergency">🚨 Emergency Override</span>
                      ) : m.speculative?.committed ? (
                        <span className="badge-pill committed">
                          ⚡ Speculation Committed (0ms SLM wait, Sim: {Math.round((m.speculative.similarity || 0.8) * 100)}%)
                        </span>
                      ) : m.speculative?.status === 'ROLLED_BACK' ? (
                        <span className="badge-pill diverged">
                          🔄 Speculation Rolled Back (Sim: {Math.round((m.speculative.similarity || 0.5) * 100)}%)
                        </span>
                      ) : m.speculative?.enabled ? (
                        <span className="badge-pill committed">⚡ CMSPE Speculative</span>
                      ) : (
                        <span className="badge-pill">Sequential Baseline</span>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {isProcessing && (
              <div className="typing-row">
                <span className="pulse-dot"></span>
                <span>Triage nurse analyzing symptoms...</span>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Quick Prompt Chips */}
          <div className="quick-chips-bar">
            <button
              type="button"
              className="prompt-pill urgent"
              onClick={() => handleSendText('I have crushing chest pain and left arm numbness.')}
            >
              💔 Crushing Chest Pain (Emergency)
            </button>
            <button
              type="button"
              className="prompt-pill urgent"
              onClick={() => handleSendText('I am gasping for air and having severe difficulty breathing.')}
            >
              🫁 Severe Difficulty Breathing
            </button>
            <button
              type="button"
              className="prompt-pill"
              onClick={() => handleSendText('I have had a throbbing migraine for 2 days.')}
            >
              🤕 Throbbing Migraine (2 days)
            </button>
            <button
              type="button"
              className="prompt-pill"
              onClick={() => handleSendText('I have a mild runny nose and cough for 3 days.')}
            >
              🤒 Cold & Cough (3 days)
            </button>
            <button
              type="button"
              className="prompt-pill"
              onClick={() => handleSendText('Actually, I meant the pain started 3 weeks ago, not yesterday.')}
            >
              🔄 "I meant..." (Correction)
            </button>
          </div>

          {/* Audio Recording Bar Overlay */}
          {isRecording && (
            <div className="recording-bar">
              <div className="rec-pulse-group">
                <span className="rec-dot"></span>
                <span style={{ fontSize: '0.82rem', fontWeight: 600 }}>Listening to speech...</span>
                <span className="rec-timer">{formatTimer(recDuration)}</span>
              </div>
              <div className="rec-actions">
                <button type="button" className="btn-rec-cancel" onClick={cancelRecording}>
                  Cancel
                </button>
                <button type="button" className="btn-rec-done" onClick={stopRecording}>
                  <span>Done</span>
                  <ChevronRight size={14} />
                </button>
              </div>
            </div>
          )}

          {/* Chat Input Bar */}
          <footer className="chat-input-bar">
            <div className="input-controls">
              {/* Native Voice File Upload (Fallback for mobile HTTP) */}
              <label className="btn-input-icon" title="Native Phone Voice Input">
                <Upload size={18} />
                <input
                  type="file"
                  ref={fileInputRef}
                  accept="audio/*"
                  capture="microphone"
                  style={{ display: 'none' }}
                  onChange={handleFileChange}
                />
              </label>

              {/* Text Input */}
              <div className="input-field-box">
                <textarea
                  className="input-textarea"
                  rows={1}
                  placeholder="Type or speak symptoms (e.g. fever, chest pain)..."
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSendText();
                    }
                  }}
                />
              </div>

              {/* Hands-Free Live Talk Button (ChatGPT Live Style) */}
              <button
                type="button"
                className="btn-live-talk-trigger"
                onClick={openLiveTalk}
                title="Start Hands-Free Live Talk (ChatGPT Live)"
              >
                <Radio size={16} />
                <span>Live</span>
              </button>

              {/* In-Browser Push to Talk Mic Button */}
              <button
                type="button"
                className={`btn-mic-trigger ${isRecording ? 'recording' : ''}`}
                onClick={isRecording ? stopRecording : startRecording}
                title="Tap to speak into phone microphone"
              >
                <Mic size={20} />
              </button>

              {/* Send Button */}
              <button
                type="button"
                className="btn-send-trigger"
                onClick={() => handleSendText()}
                disabled={!inputText.trim() || isProcessing}
                title="Send"
              >
                <Send size={18} />
              </button>
            </div>

            <div className="input-options-row">
              <span>💡 Tap <strong>LIVE TALK</strong> for hands-free conversational voice, or <strong>Mic</strong> to push-to-talk.</span>
              <label className="auto-play-toggle">
                <input
                  type="checkbox"
                  checked={autoPlay}
                  onChange={(e) => setAutoPlay(e.target.checked)}
                />
                <span>Auto-play voice</span>
              </label>
            </div>
          </footer>
        </section>

        {/* Clinical EHR Drawer */}
        {isDrawerOpen && (
          <div
            className="drawer-backdrop"
            onClick={() => setIsDrawerOpen(false)}
          />
        )}
        <aside className={`clinical-drawer ${isDrawerOpen ? 'open' : ''}`}>
          <div className="drawer-header">
            <div className="drawer-title-group">
              <Activity size={18} color="#0EA5E9" />
              <span>Clinical EHR & Pipeline</span>
            </div>
            <button
              type="button"
              className="btn-close-drawer"
              onClick={() => setIsDrawerOpen(false)}
            >
              <X size={20} />
            </button>
          </div>

          <div className="drawer-body">
            {/* Pipeline Execution Mode Card */}
            <div className="drawer-card" style={{ borderColor: 'rgba(14, 165, 233, 0.3)' }}>
              <div className="card-title-row">
                <span className="card-label">Execution Pipeline</span>
                <span className={`card-badge ${speculativeEnabled ? 'active' : ''}`}>
                  {speculativeEnabled ? '⚡ CMSPE Active' : 'Sequential Baseline'}
                </span>
              </div>
              <div className="switch-row">
                <div className="switch-label-group">
                  <span className="switch-main-text">Speculative Streaming</span>
                  <span className="switch-sub-text">
                    Parallel SLM reasoning on early partial ASR hypothesis (~80ms tiny Whisper).
                  </span>
                </div>
                <label className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={speculativeEnabled}
                    onChange={(e) => toggleSpeculativeMode(e.target.checked)}
                  />
                  <span className="toggle-track"></span>
                </label>
              </div>
              <div style={{ fontSize: '0.7rem', color: '#94A3B8' }}>
                Rollback threshold: <strong>Similarity &ge; 70%</strong>
              </div>
            </div>

            {/* Structured EHR Extraction Slots */}
            <div className="drawer-card">
              <div className="card-title-row">
                <span className="card-label">Structured Intake (JSON)</span>
                <span className="card-badge">
                  Confidence: {(slots.confidence || 'Low').toUpperCase()}
                </span>
              </div>
              <div className="slot-grid">
                <div className="slot-box">
                  <span className="slot-name">Chief Complaint</span>
                  <span className="slot-val">{slots.chief_complaint || 'None recorded'}</span>
                </div>
                <div className="slot-box">
                  <span className="slot-name">Duration</span>
                  <span className="slot-val">{slots.duration || '—'}</span>
                </div>
                <div className="slot-box">
                  <span className="slot-name">Severity</span>
                  <span className="slot-val">{(slots.severity || 'Pending').toUpperCase()}</span>
                </div>
                <div className="slot-box">
                  <span className="slot-name">Unit / Dept</span>
                  <span className="slot-val">{slots.recommended_department || 'Triage Intake'}</span>
                </div>
              </div>

              <div>
                <span style={{ fontSize: '0.65rem', color: '#64748B' }}>Associated Symptoms:</span>
                <div className="slot-tags-container" style={{ marginTop: '4px' }}>
                  {slots.associated_symptoms && slots.associated_symptoms.length > 0 ? (
                    slots.associated_symptoms.map((s, idx) => (
                      <span key={idx} className="slot-chip">{s}</span>
                    ))
                  ) : (
                    <span style={{ fontSize: '0.72rem', color: '#64748B', fontStyle: 'italic' }}>
                      None documented
                    </span>
                  )}
                </div>
              </div>

              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '7px 10px',
                  borderRadius: '8px',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  background: slots.red_flag_present
                    ? 'rgba(239, 68, 68, 0.15)'
                    : 'rgba(16, 185, 129, 0.12)',
                  color: slots.red_flag_present ? '#FCA5A5' : '#10B981',
                  border: `1px solid ${slots.red_flag_present ? 'rgba(239, 68, 68, 0.35)' : 'rgba(16, 185, 129, 0.25)'}`,
                }}
              >
                {slots.red_flag_present ? (
                  <>
                    <AlertTriangle size={15} />
                    <span>Red-Flag Condition Active</span>
                  </>
                ) : (
                  <>
                    <ShieldCheck size={15} />
                    <span>Safety Baseline Clear</span>
                  </>
                )}
              </div>
            </div>

            {/* Paralinguistic Biomarkers Card */}
            <div className="drawer-card">
              <div className="card-title-row">
                <span className="card-label">Acoustic Biomarkers (&lt;3ms DSP)</span>
                <span className="card-badge" style={{ color: (biomarkers?.acuity_boost || 0) > 0 ? '#EF4444' : '#10B981' }}>
                  Acuity: {(biomarkers?.acuity_boost || 0) > 0 ? `+${biomarkers.acuity_boost} Elevated` : 'Normal'}
                </span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '0.74rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#94A3B8' }}>Speech-to-Pause ($R_p$):</span>
                  <span style={{ fontFamily: 'monospace', color: '#0EA5E9' }}>
                    {biomarkers ? biomarkers.speech_to_pause_ratio.toFixed(2) : '1.00'}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#94A3B8' }}>Wheeze Band Power:</span>
                  <span style={{ fontFamily: 'monospace', color: '#0EA5E9' }}>
                    {biomarkers ? biomarkers.wheeze_band_power_ratio.toFixed(2) : '0.00'}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: '#94A3B8' }}>Inspiratory/Expiratory (I/E):</span>
                  <span style={{ fontFamily: 'monospace', color: '#0EA5E9' }}>
                    {biomarkers ? biomarkers.ie_ratio.toFixed(2) : '0.33'}
                  </span>
                </div>
              </div>
              <p style={{ fontSize: '0.7rem', color: '#64748B', fontStyle: 'italic' }}>
                {biomarkers?.clinical_rationale || 'Awaiting patient voice input...'}
              </p>
            </div>

            {/* Hardware Telemetry Card */}
            <div className="drawer-card">
              <div className="card-title-row">
                <span className="card-label">Edge GPU & Latency</span>
                <span className="card-badge">NVIDIA GTX 1650</span>
              </div>
              <div className="slot-grid">
                <div className="slot-box">
                  <span className="slot-name">Whisper ASR</span>
                  <span className="slot-val" style={{ color: '#06B6D4' }}>
                    {telemetry.asr_latency_ms ? `${telemetry.asr_latency_ms} ms` : '-- ms'}
                  </span>
                </div>
                <div className="slot-box">
                  <span className="slot-name">SLM Reasoning</span>
                  <span className="slot-val" style={{ color: '#818CF8' }}>
                    {telemetry.slm_latency_ms !== undefined ? `${telemetry.slm_latency_ms} ms` : '-- ms'}
                  </span>
                </div>
                <div className="slot-box">
                  <span className="slot-name">Piper TTS</span>
                  <span className="slot-val" style={{ color: '#0EA5E9' }}>
                    {telemetry.tts_latency_ms ? `${telemetry.tts_latency_ms} ms` : '-- ms'}
                  </span>
                </div>
                <div className="slot-box">
                  <span className="slot-name">VRAM Budget</span>
                  <span className="slot-val" style={{ color: '#F59E0B' }}>
                    {vram.used_mb} MB / {vram.total_mb} MB
                  </span>
                </div>
              </div>
            </div>

            {/* Phone Access Guide */}
            <div className="drawer-card">
              <span className="card-label">Phone USB / LAN Access</span>
              <p style={{ fontSize: '0.74rem', color: '#94A3B8', lineHeight: 1.45 }}>
                Connected via USB debugging (<code>adb reverse tcp:8000 tcp:8000</code>):<br />
                <span style={{ color: '#0EA5E9', fontFamily: 'monospace' }}>http://localhost:8000</span>
              </p>
            </div>
          </div>
        </aside>
      </main>

      {/* Live Terminal Modal / Bottom Sheet */}
      {isTerminalOpen && (
        <div className="terminal-overlay">
          <div className="terminal-backdrop" onClick={() => setIsTerminalOpen(false)}></div>
          <div className="terminal-box">
            <div className="terminal-box-header">
              <div className="terminal-title">
                <span className="term-dot red"></span>
                <span className="term-dot yellow"></span>
                <span className="term-dot green"></span>
                <span>Backend Terminal & Latency Stream</span>
              </div>
              <div className="terminal-btn-group">
                <button
                  type="button"
                  className="btn-term"
                  onClick={() => {
                    navigator.clipboard.writeText(logs.join('\n'));
                    alert('Terminal logs copied to clipboard!');
                  }}
                >
                  <Copy size={13} style={{ display: 'inline', marginRight: '4px' }} />
                  Copy
                </button>
                <button
                  type="button"
                  className="btn-term"
                  onClick={() => setLogs([])}
                >
                  <Trash2 size={13} style={{ display: 'inline', marginRight: '4px' }} />
                  Clear
                </button>
                <button
                  type="button"
                  className="btn-term"
                  onClick={() => setIsTerminalOpen(false)}
                >
                  <X size={15} />
                </button>
              </div>
            </div>

            {/* Latency Chips */}
            <div className="terminal-chips-grid">
              <div className="term-chip">
                <span className="chip-label">Total Latency</span>
                <span className="chip-val total">
                  {telemetry.total_latency_ms ? `${telemetry.total_latency_ms} ms` : '-- ms'}
                </span>
              </div>
              <div className="term-chip">
                <span className="chip-label">Speculation</span>
                <span className="chip-val spec">
                  {telemetry.speculative?.committed
                    ? '⚡ COMMITTED'
                    : telemetry.speculative?.status === 'ROLLED_BACK'
                    ? 'ROLLED BACK'
                    : speculativeEnabled
                    ? '⚡ ACTIVE'
                    : 'OFF (SEQ)'}
                </span>
              </div>
              <div className="term-chip">
                <span className="chip-label">Whisper ASR</span>
                <span className="chip-val asr">
                  {telemetry.asr_latency_ms ? `${telemetry.asr_latency_ms} ms` : '-- ms'}
                </span>
              </div>
              <div className="term-chip">
                <span className="chip-label">SLM Reasoning</span>
                <span className="chip-val slm">
                  {telemetry.slm_latency_ms !== undefined ? `${telemetry.slm_latency_ms} ms` : '-- ms'}
                </span>
              </div>
              <div className="term-chip">
                <span className="chip-label">Piper TTS</span>
                <span className="chip-val tts">
                  {telemetry.tts_latency_ms ? `${telemetry.tts_latency_ms} ms` : '-- ms'}
                </span>
              </div>
              <div className="term-chip">
                <span className="chip-label">GPU VRAM</span>
                <span className="chip-val vram">
                  {vram.used_mb} MB
                </span>
              </div>
            </div>

            {/* Terminal Body */}
            <div className="terminal-logs-viewport">
              <pre className="terminal-pre">
                {logs.length > 0 ? logs.join('\n') : 'Connecting to live server log buffer...'}
              </pre>
            </div>
          </div>
        </div>
      )}

      {/* ChatGPT Live Style Full-Screen / Modal Live Voice Interface */}
      {isLiveTalkOpen && (
        <div className="live-talk-modal-overlay">
          <div className="live-talk-box">
            {/* Top Session Bar */}
            <div className="live-talk-topbar">
              <div className="live-session-info">
                <span className="live-pulsing-badge">
                  <span className="live-dot-ping"></span>
                  LIVE
                </span>
                <span className="live-timer">{formatTimer(liveSessionSeconds)}</span>
              </div>

              <div className="live-spec-badge">
                {speculativeEnabled ? (
                  <span className="spec-indicator-pill active">
                    <Zap size={13} className="spec-icon-spin" />
                    <span className="spec-label-full">CMSPE 0ms Speculation</span>
                    <span className="spec-label-compact">0ms CMSPE</span>
                  </span>
                ) : (
                  <span className="spec-indicator-pill">
                    <span className="spec-label-full">Sequential Baseline</span>
                    <span className="spec-label-compact">Sequential</span>
                  </span>
                )}
              </div>

              <button
                type="button"
                className="btn-live-close"
                onClick={closeLiveTalk}
                title="Exit Live Talk Mode"
              >
                <X size={18} />
              </button>
            </div>

            {/* Emergency Alert Banner inside Live Talk */}
            {activeEmergency && (
              <div className="live-emergency-strip">
                <AlertTriangle size={18} color="#EF4444" />
                <span>EMERGENCY CLINICAL OVERRIDE: {activeEmergency.category}</span>
              </div>
            )}

            {/* Main Center Stage: Organic Glowing Voice Orb */}
            <div className="live-talk-stage">
              <VoiceOrb
                state={liveState}
                audioLevel={liveAudioLevel}
                isEmergency={!!activeEmergency}
              />

              {/* State Indicator Badge */}
              <div className={`live-state-pill ${liveState} ${activeEmergency ? 'emergency' : ''}`}>
                {liveState === 'listening' && (
                  <>
                    <span className="state-pulse-dot cyan"></span>
                    <span>Listening to you... (Speak naturally)</span>
                  </>
                )}
                {liveState === 'analyzing' && (
                  <>
                    <span className="state-pulse-dot amber"></span>
                    <span>Analyzing symptoms & clinical safety...</span>
                  </>
                )}
                {liveState === 'speaking' && (
                  <>
                    <span className="state-pulse-dot purple"></span>
                    <span>Triage Nurse Speaking... (You can interrupt)</span>
                  </>
                )}
                {liveState === 'interrupted' && (
                  <>
                    <span className="state-pulse-dot orange"></span>
                    <span>Interrupted — Listening...</span>
                  </>
                )}
                {liveState === 'muted' && (
                  <>
                    <span className="state-pulse-dot gray"></span>
                    <span>Microphone Muted</span>
                  </>
                )}
              </div>

              {/* Real-Time Audio Frequency Bars */}
              <div className="live-spectrum-row">
                {Array.from({ length: 24 }).map((_, i) => {
                  const h = liveState === 'muted'
                    ? 3
                    : Math.max(3, Math.min(36, Math.sin(i * 0.45 + liveAudioLevel * 8) * (liveAudioLevel * 50) + 4));
                  return (
                    <span
                      key={i}
                      className={`spectrum-bar ${liveState}`}
                      style={{ height: `${h}px` }}
                    />
                  );
                })}
              </div>

              {/* Live Closed-Captions Card */}
              <div className="live-subtitles-card">
                <div className="subtitles-patient">
                  <span className="sub-tag">YOU</span>
                  <p>{liveSubtitles.user || 'Speak symptoms aloud (e.g. "I have a high fever and headache")'}</p>
                </div>
                {liveSubtitles.assistant && (
                  <div className="subtitles-assistant">
                    <span className="sub-tag asst">NURSE</span>
                    <p>{liveSubtitles.assistant}</p>
                  </div>
                )}
              </div>
            </div>

            {/* Live Talk Controls Footer */}
            <div className="live-talk-controls">
              {/* Mute Mic */}
              <button
                type="button"
                className={`btn-live-action ${isMicMuted ? 'muted' : ''}`}
                onClick={toggleLiveMic}
                title={isMicMuted ? 'Unmute Microphone' : 'Mute Microphone'}
              >
                <div className="live-action-icon-circle">
                  {isMicMuted ? <MicOff size={20} /> : <Mic size={20} />}
                </div>
                <span>{isMicMuted ? 'Unmute' : 'Mute'}</span>
              </button>

              {/* Voice Interruption / Stop Speech */}
              <button
                type="button"
                className={`btn-live-action interrupt ${liveState === 'speaking' ? 'active-interrupt' : ''}`}
                onClick={interruptLiveSpeech}
                disabled={liveState !== 'speaking'}
                title="Interrupt Nurse Speech Immediately"
              >
                <div className="live-action-icon-circle">
                  <VolumeX size={20} />
                </div>
                <span>Interrupt</span>
              </button>

              {/* Transcript / Chat Drawer Toggle */}
              <button
                type="button"
                className={`btn-live-action ${isLiveTranscriptVisible ? 'active' : ''}`}
                onClick={() => setIsLiveTranscriptVisible(!isLiveTranscriptVisible)}
                title="Toggle Full Conversation Transcript"
              >
                <div className="live-action-icon-circle">
                  <MessageSquare size={20} />
                </div>
                <span>Chat</span>
              </button>

              {/* End Live Talk Session */}
              <button
                type="button"
                className="btn-live-action end-call"
                onClick={closeLiveTalk}
                title="End Live Talk Session"
              >
                <div className="live-action-icon-circle end-icon">
                  <PhoneOff size={20} />
                </div>
                <span>End</span>
              </button>
            </div>

            {/* Optional Overlay Slide-up Transcript inside Live Talk */}
            {isLiveTranscriptVisible && (
              <div className="live-transcript-slideup">
                <div className="slideup-handle-bar">
                  <span className="slideup-pill"></span>
                </div>
                <div className="slideup-header">
                  <h4>Live Intake Transcript ({messages.length} turns)</h4>
                  <button type="button" onClick={() => setIsLiveTranscriptVisible(false)}>
                    <X size={18} />
                  </button>
                </div>
                <div className="slideup-feed">
                  {messages.map((m) => (
                    <div key={m.id} className={`slideup-msg ${m.role}`}>
                      <strong>{m.role === 'user' ? 'Patient:' : 'Nurse:'}</strong>
                      <span>{m.text}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
