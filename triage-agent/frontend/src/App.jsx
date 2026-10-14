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
   VoiceOrb Component - Minimalist Precision Audio Indicator
   ========================================================================== */
function VoiceOrb({ state, analyserRef, audioLevel, isEmergency, isUserSpeaking }) {
  const canvasRef = useRef(null);
  const stateRef = useRef(state);
  stateRef.current = state;
  const isEmergencyRef = useRef(isEmergency);
  isEmergencyRef.current = isEmergency;
  const audioLevelRef = useRef(audioLevel);
  audioLevelRef.current = audioLevel;
  const isUserSpeakingRef = useRef(isUserSpeaking);
  isUserSpeakingRef.current = isUserSpeaking;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animId;
    let phase = 0;
    let smoothAmp = 0;
    const freqData = new Uint8Array(32);

    const render = () => {
      animId = requestAnimationFrame(render);
      phase += 0.05;
      const width = canvas.width;
      const height = canvas.height;
      const cx = width / 2;
      const cy = height / 2;

      ctx.clearRect(0, 0, width, height);

      const curState = stateRef.current;
      const curEmergency = isEmergencyRef.current;
      const curSpeaking = isUserSpeakingRef.current;
      const curLevel = audioLevelRef.current || 0;

      // Solid minimalist clinical state color (flat, no gradient)
      let stateColor = '#10b981'; // default listening: emerald
      if (curEmergency) {
        stateColor = '#ef4444';
      } else if (curState === 'speaking') {
        stateColor = '#3b82f6';
      } else if (curState === 'analyzing') {
        stateColor = '#f59e0b';
      } else if (curState === 'interrupted') {
        stateColor = '#f97316';
      } else if (curState === 'muted') {
        stateColor = '#71717a';
      }

      // Compute speech volume amplitude
      let targetAmp = 0;
      let hasFreq = false;

      if (curState === 'muted') {
        targetAmp = 0;
      } else if (analyserRef?.current && curState === 'listening') {
        analyserRef.current.getByteFrequencyData(freqData);
        let sum = 0;
        // Human voice dominant bands: bin 1 to bin 14 (~250Hz - ~3500Hz)
        for (let b = 1; b <= 14; b++) {
          sum += freqData[b];
        }
        const rawAmp = sum / (14 * 255);
        if (rawAmp > 0.012) {
          hasFreq = true;
          // Non-linear scaling so normal conversational voice reaches ~0.5 - 0.9
          targetAmp = Math.min(1.0, Math.pow(rawAmp * 2.4, 0.72));
        }
      }

      if (!hasFreq) {
        if (curState === 'speaking') {
          targetAmp = 0.4 + Math.sin(phase * 4) * 0.25;
        } else if (curState === 'listening') {
          targetAmp = curSpeaking ? Math.max(0.45, curLevel) : curLevel * 0.8;
        }
      }

      // Smooth attack/decay for snappy, organic movement
      smoothAmp += (targetAmp - smoothAmp) * 0.28;

      // 1. Static guide hairline ring
      const baseR = 52;
      ctx.beginPath();
      ctx.arc(cx, cy, baseR + 24, 0, Math.PI * 2);
      ctx.strokeStyle = '#27272a';
      ctx.lineWidth = 1;
      ctx.stroke();

      // 2. Active responsive voice ripple ring (expands dynamically up to 82px)
      const dynamicR = baseR + smoothAmp * 28;
      ctx.beginPath();
      if (smoothAmp > 0.08 && (curState === 'listening' || curState === 'speaking')) {
        const points = 36;
        for (let p = 0; p <= points; p++) {
          const theta = (p / points) * Math.PI * 2;
          const ripple = Math.sin(theta * 6 + phase * 5) * (smoothAmp * 4.5);
          const r = dynamicR + ripple;
          const x = cx + Math.cos(theta) * r;
          const y = cy + Math.sin(theta) * r;
          if (p === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.closePath();
      } else {
        ctx.arc(cx, cy, dynamicR, 0, Math.PI * 2);
      }
      ctx.strokeStyle = stateColor;
      ctx.lineWidth = smoothAmp > 0.1 || curSpeaking || curState === 'speaking' || curEmergency ? 2.5 : 1.5;
      ctx.stroke();

      // 3. Center Solid Indicator Hub
      ctx.beginPath();
      ctx.arc(cx, cy, 32, 0, Math.PI * 2);
      ctx.fillStyle = '#141417';
      ctx.fill();
      ctx.strokeStyle = smoothAmp > 0.1 || curSpeaking ? stateColor : '#27272a';
      ctx.lineWidth = smoothAmp > 0.1 || curSpeaking ? 1.5 : 1;
      ctx.stroke();

      // 4. Center 5 Minimalist Audio Waveform Bars (jumps up to 30px with voice)
      const numBars = 5;
      const barWidth = 3;
      const barGap = 4;
      const totalBarWidth = numBars * barWidth + (numBars - 1) * barGap;
      const startX = cx - totalBarWidth / 2;

      for (let i = 0; i < numBars; i++) {
        const x = startX + i * (barWidth + barGap);
        const distFromCenter = Math.abs(i - 2);
        const centerFactor = 1 - distFromCenter * 0.18;
        let barH = 5;

        if (curState === 'muted') {
          barH = 4;
        } else if (hasFreq && curState === 'listening') {
          // Direct harmonic bin response
          const binIdx = [1, 3, 5, 8, 12][i];
          const binVal = freqData[binIdx] / 255;
          barH = 5 + Math.pow(binVal * 2.0, 0.72) * 26 * centerFactor;
        } else if (curState === 'speaking') {
          barH = 7 + Math.sin(phase * 4 + i * 0.85) * 14 * centerFactor;
        } else if (curState === 'analyzing') {
          barH = 6 + Math.sin(phase * 6 + i * 1.2) * 7;
        } else if (curState === 'listening') {
          barH = 5 + smoothAmp * 24 * centerFactor + Math.sin(phase * 2 + i * 0.4) * 2;
        }

        barH = Math.max(4, Math.min(32, barH));

        ctx.fillStyle = stateColor;
        ctx.beginPath();
        if (ctx.roundRect) {
          ctx.roundRect(x, cy - barH / 2, barWidth, barH, 2);
        } else {
          ctx.rect(x, cy - barH / 2, barWidth, barH);
        }
        ctx.fill();
      }
    };

    render();
    return () => cancelAnimationFrame(animId);
  }, []);

  return (
    <div className={`live-orb-wrapper ${state} ${isEmergency ? 'emergency' : ''}`}>
      <canvas ref={canvasRef} width={240} height={240} className="live-orb-canvas" />
    </div>
  );
}

function LiveSpectrumBars({ analyserRef, liveState, isUserSpeaking, audioLevel }) {
  const containerRef = useRef(null);
  const liveStateRef = useRef(liveState);
  liveStateRef.current = liveState;
  const isUserSpeakingRef = useRef(isUserSpeaking);
  isUserSpeakingRef.current = isUserSpeaking;
  const audioLevelRef = useRef(audioLevel);
  audioLevelRef.current = audioLevel;

  useEffect(() => {
    let animId;
    let phase = 0;
    const freqData = new Uint8Array(32);
    const container = containerRef.current;
    if (!container) return;
    const barEls = container.querySelectorAll('.spectrum-bar');

    const updateBars = () => {
      animId = requestAnimationFrame(updateBars);
      phase += 0.08;
      const curState = liveStateRef.current;
      const curSpeaking = isUserSpeakingRef.current;
      const curLevel = audioLevelRef.current || 0;

      if (curState === 'muted') {
        for (let i = 0; i < barEls.length; i++) {
          barEls[i].style.height = '3px';
        }
        return;
      }

      let hasFreq = false;
      if (analyserRef?.current && curState === 'listening') {
        analyserRef.current.getByteFrequencyData(freqData);
        let maxVal = 0;
        for (let i = 0; i < 24; i++) {
          if (freqData[i] > maxVal) maxVal = freqData[i];
        }
        if (maxVal > 8) {
          hasFreq = true;
          for (let i = 0; i < barEls.length; i++) {
            const bin = Math.min(31, Math.floor(i * 1.1) + 1);
            const raw = freqData[bin] / 255;
            const boost = i < 16 ? 1.6 : 1.1;
            const h = Math.max(3, Math.min(32, Math.pow(raw * boost, 0.72) * 32));
            barEls[i].style.height = `${Math.round(h)}px`;
          }
        }
      }

      if (!hasFreq) {
        if (curState === 'speaking') {
          for (let i = 0; i < barEls.length; i++) {
            const wave = Math.sin(phase * 3.5 + i * 0.45) * 0.5 + 0.5;
            const centerFactor = 1 - Math.abs(i - 11.5) / 14;
            const h = Math.max(3, Math.min(28, wave * 24 * centerFactor + 4));
            barEls[i].style.height = `${Math.round(h)}px`;
          }
        } else if (curState === 'analyzing') {
          for (let i = 0; i < barEls.length; i++) {
            const wave = Math.sin(phase * 5 + i * 0.6) * 5 + 6;
            barEls[i].style.height = `${Math.round(Math.max(3, wave))}px`;
          }
        } else if (curState === 'listening') {
          const amp = curSpeaking ? Math.max(0.45, curLevel) : curLevel;
          for (let i = 0; i < barEls.length; i++) {
            const centerFactor = 1 - Math.abs(i - 11.5) / 14;
            const wave = Math.sin(phase * 3 + i * 0.5) * 0.35 + 0.65;
            const h = Math.max(3, Math.min(30, amp * 28 * wave * centerFactor + 3));
            barEls[i].style.height = `${Math.round(h)}px`;
          }
        } else {
          for (let i = 0; i < barEls.length; i++) {
            barEls[i].style.height = '3px';
          }
        }
      }
    };

    updateBars();
    return () => cancelAnimationFrame(animId);
  }, []);

  return (
    <div ref={containerRef} className="live-spectrum-row">
      {Array.from({ length: 24 }).map((_, i) => (
        <span key={i} className={`spectrum-bar ${liveState}`} />
      ))}
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

  // Live Streaming Audio Queue Refs
  const autoPlayRef = useRef(autoPlay);
  useEffect(() => {
    autoPlayRef.current = autoPlay;
  }, [autoPlay]);

  const audioStreamQueueRef = useRef([]);
  const isAudioQueuePlayingRef = useRef(false);

  const stopAllAudio = () => {
    audioStreamQueueRef.current = [];
    isAudioQueuePlayingRef.current = false;
    if (audioElementRef.current) {
      audioElementRef.current.pause();
      audioElementRef.current.currentTime = 0;
    }
    setPlayingAudioId(null);
  };

  const playNextStreamChunk = () => {
    if (audioStreamQueueRef.current.length === 0) {
      isAudioQueuePlayingRef.current = false;
      setPlayingAudioId(null);
      return;
    }
    isAudioQueuePlayingRef.current = true;
    const chunk = audioStreamQueueRef.current.shift();
    if (audioElementRef.current) {
      audioElementRef.current.src = chunk.audioBase64;
      setPlayingAudioId(chunk.msgId);
      audioElementRef.current.play().catch((e) => {
        console.warn('Audio chunk playback warning:', e);
        playNextStreamChunk();
      });
      audioElementRef.current.onended = () => {
        playNextStreamChunk();
      };
    }
  };

  const enqueueAudioChunk = (audioBase64, msgId) => {
    if (!audioBase64) return;
    audioStreamQueueRef.current.push({ audioBase64, msgId });
    if (!isAudioQueuePlayingRef.current) {
      playNextStreamChunk();
    }
  };

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
  const [isUserSpeaking, setIsUserSpeaking] = useState(false);
  const [liveSessionSeconds, setLiveSessionSeconds] = useState(0);
  const [isLiveTranscriptVisible, setIsLiveTranscriptVisible] = useState(false);

  // Live Talk Audio & WebSocket Refs
  const liveWsRef = useRef(null);
  const liveAudioContextRef = useRef(null);
  const liveAnalyserRef = useRef(null);
  const liveMediaStreamRef = useRef(null);
  const liveMediaRecorderRef = useRef(null);
  const liveProcessorRef = useRef(null);
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
    stopAllAudio();
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
    stopAllAudio();
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

  // Send Text Turn with Live Streaming Tokens and Real-Time Sub-Clause Audio
  const handleSendText = async (textToSend) => {
    const text = (textToSend || inputText).trim();
    if (!text || isProcessing) return;

    // Stop any ongoing audio playback
    stopAllAudio();

    setInputText('');
    const userMsg = {
      id: 'user-' + Date.now(),
      role: 'user',
      text,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    const asstMsgId = 'asst-' + Date.now();
    const asstMsg = {
      id: asstMsgId,
      role: 'assistant',
      text: '',
      audioBase64: '',
      isEmergency: false,
      redFlagCategory: null,
      speculative: null,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      isStreaming: true,
    };

    setMessages((prev) => [...prev, userMsg, asstMsg]);
    setIsProcessing(true);

    try {
      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let accumulatedText = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data: ')) continue;
          const jsonStr = trimmed.slice(6);
          try {
            const data = JSON.parse(jsonStr);

            if (data.type === 'token') {
              accumulatedText += data.token;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === asstMsgId
                    ? { ...m, text: accumulatedText }
                    : m
                )
              );
            } else if (data.type === 'audio_chunk') {
              // Enqueue audio chunk for live playback as LLM outputs words
              if (autoPlayRef.current && data.audio_base64) {
                enqueueAudioChunk(data.audio_base64, asstMsgId);
              }
            } else if (data.type === 'done') {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === asstMsgId
                    ? {
                        ...m,
                        text: data.agent_response || accumulatedText,
                        audioBase64: data.audio_base64,
                        isEmergency: data.is_emergency,
                        redFlagCategory: data.red_flag_category,
                        isStreaming: false,
                      }
                    : m
                )
              );

              if (data.is_emergency) {
                setActiveEmergency({
                  category: data.red_flag_category || 'EMERGENCY ESCALATION',
                  message: data.agent_response,
                });
              }
              if (data.slots) setSlots(data.slots);
              if (data.vram) setVram(data.vram);
              if (data.total_latency_ms) {
                setTelemetry((prev) => ({
                  ...prev,
                  total_latency_ms: data.total_latency_ms,
                  slm_latency_ms: data.total_latency_ms,
                }));
              }
            }
          } catch (pe) {
            console.warn('SSE parse error:', pe);
          }
        }
      }
    } catch (e) {
      console.error('Streaming chat error, attempting standard fallback:', e);
      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        setMessages((prev) =>
          prev.map((m) =>
            m.id === asstMsgId
              ? {
                  ...m,
                  text: data.agent_response,
                  audioBase64: data.audio_base64,
                  isEmergency: data.is_emergency,
                  redFlagCategory: data.red_flag_category,
                  speculative: data.speculative,
                  isStreaming: false,
                }
              : m
          )
        );
        processAssistantTurn(data);
      } catch (fallbackErr) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === asstMsgId
              ? {
                  ...m,
                  text: 'I encountered an issue contacting the clinical model. Please ensure Ollama is active.',
                  isStreaming: false,
                }
              : m
          )
        );
      }
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

    if (liveProcessorRef.current) {
      try {
        liveProcessorRef.current.disconnect();
        liveProcessorRef.current.onaudioprocess = null;
      } catch {}
      liveProcessorRef.current = null;
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
    setIsUserSpeaking(false);
    setLiveAudioLevel(0);
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
    isSpeakingUserRef.current = false;
    setIsUserSpeaking(false);
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
        const ext = blob.type.includes('wav') ? 'wav' : (blob.type.includes('mp4') ? 'mp4' : 'webm');
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
      analyser.fftSize = 64;
      analyser.smoothingTimeConstant = 0.4;
      source.connect(analyser);
      liveAnalyserRef.current = analyser;

      connectLiveWebSocket();

      // Lossless 16-bit Mono PCM WAV Encoder
      const encodeWavBlob = (samples, sampleRate) => {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);
        const writeString = (offset, string) => {
          for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
          }
        };
        writeString(0, 'RIFF');
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true); // PCM
        view.setUint16(22, 1, true); // Mono
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(36, 'data');
        view.setUint32(40, samples.length * 2, true);

        let offset = 44;
        for (let i = 0; i < samples.length; i++, offset += 2) {
          const s = Math.max(-1, Math.min(1, samples[i]));
          view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        }
        return new Blob([buffer], { type: 'audio/wav' });
      };

      // 500ms pre-roll circular buffer to eliminate onset clipping
      const sampleRate = ctx.sampleRate || 16000;
      const preRollCapacity = Math.floor(sampleRate * 0.5); // ~500ms
      const preRollRing = new Float32Array(preRollCapacity);
      let preRollIndex = 0;
      let preRollCount = 0;
      let accumulatedSpeech = [];
      let smoothRms = 0;
      let noiseFloor = 0.006; // Adaptive ambient baseline

      // Low-latency PCM streaming & Voice Activity Processor
      const bufferSize = 2048;
      const processor = ctx.createScriptProcessor(bufferSize, 1, 1);
      liveProcessorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        const curState = liveStateRef.current;
        const muted = isMicMutedRef.current;

        // 1. Calculate Frame RMS
        let sum = 0;
        for (let i = 0; i < inputData.length; i++) {
          sum += inputData[i] * inputData[i];
        }
        const rms = Math.sqrt(sum / inputData.length);
        smoothRms = smoothRms * 0.65 + rms * 0.35;

        // 2. Compute normalized speech volume dynamics (0.0 to 1.0)
        const ambientBase = Math.max(0.003, noiseFloor);
        const speechSignal = Math.max(0, smoothRms - ambientBase * 0.7);
        const normalizedSpeechLevel = Math.min(1.0, Math.pow(speechSignal * 24, 0.75));
        setLiveAudioLevel(normalizedSpeechLevel);

        // 3. Track ambient noise floor adaptively
        noiseFloor = noiseFloor * 0.995 + rms * 0.005;

        // 4. Barge-in check: If agent is speaking and patient vocalizes
        if (curState === 'speaking' && !muted && smoothRms > 0.06) {
          interruptLiveSpeech();
          return;
        }

        // 5. Voice Activity Detection & Recording (High sensitivity + Pre-roll)
        if (curState === 'listening' && !muted) {
          // Dynamic dual-threshold (Hysteresis): sensitive onset, gentle hold
          const speechStartThreshold = Math.max(0.011, Math.min(0.022, noiseFloor * 2.2));
          const speechHoldThreshold = Math.max(0.006, Math.min(0.014, noiseFloor * 1.3));

          if (!isSpeakingUserRef.current) {
            // Buffer into pre-roll ring buffer while waiting
            for (let i = 0; i < inputData.length; i++) {
              preRollRing[preRollIndex] = inputData[i];
              preRollIndex = (preRollIndex + 1) % preRollCapacity;
              if (preRollCount < preRollCapacity) preRollCount++;
            }

            // Check if patient started speaking
            if (smoothRms >= speechStartThreshold) {
              isSpeakingUserRef.current = true;
              setIsUserSpeaking(true);
              liveSpeechStartTimeRef.current = Date.now();
              if (liveSilenceTimerRef.current) {
                clearTimeout(liveSilenceTimerRef.current);
                liveSilenceTimerRef.current = null;
              }

              // Unroll pre-roll ring buffer (previous 500ms before speech) to eliminate onset clipping
              const preRollChunk = new Float32Array(preRollCount);
              const startIdx = preRollCount < preRollCapacity ? 0 : preRollIndex;
              for (let i = 0; i < preRollCount; i++) {
                preRollChunk[i] = preRollRing[(startIdx + i) % preRollCapacity];
              }
              accumulatedSpeech = [preRollChunk, new Float32Array(inputData)];
            }
          } else {
            // Currently accumulating speech
            accumulatedSpeech.push(new Float32Array(inputData));

            if (smoothRms >= speechHoldThreshold) {
              // Voice active: keep reset timer
              if (liveSilenceTimerRef.current) {
                clearTimeout(liveSilenceTimerRef.current);
                liveSilenceTimerRef.current = null;
              }
            } else {
              // Silence detected: initiate clinical end-of-turn window (1300ms)
              if (!liveSilenceTimerRef.current) {
                liveSilenceTimerRef.current = setTimeout(() => {
                  const speechDuration = Date.now() - liveSpeechStartTimeRef.current;
                  if (speechDuration >= 400 && accumulatedSpeech.length > 0) {
                    setLiveState('analyzing');
                    // Concatenate all float32 speech frames
                    let totalLen = 0;
                    for (let c of accumulatedSpeech) totalLen += c.length;
                    const fullPcm = new Float32Array(totalLen);
                    let offset = 0;
                    for (let c of accumulatedSpeech) {
                      fullPcm.set(c, offset);
                      offset += c.length;
                    }
                    accumulatedSpeech = [];
                    // Generate pristine WAV blob and dispatch
                    const wavBlob = encodeWavBlob(fullPcm, sampleRate);
                    handleLiveAudioUpload(wavBlob);
                  }
                  isSpeakingUserRef.current = false;
                  setIsUserSpeaking(false);
                  liveSilenceTimerRef.current = null;
                  preRollCount = 0;
                  preRollIndex = 0;
                }, 1300); // 1300ms natural clinical pause tolerance
              }
            }
          }
        }
      };

      source.connect(processor);
      // Connect to a silent dummy destination to keep ScriptProcessor running
      const dummyGain = ctx.createGain();
      dummyGain.gain.value = 0;
      processor.connect(dummyGain);
      dummyGain.connect(ctx.destination);

      setLiveState('listening');
    } catch (err) {
      console.error('Failed to start Live Audio:', err);
      alert('Could not access microphone for Live Talk. Please grant microphone permissions in your browser.');
      setIsLiveTalkOpen(false);
    }
  };

  return (
    <div className="app-root">
      {/* Top Header */}
      <header className="top-header">
        <div className="header-brand">
          <div className="brand-icon">
            <Stethoscope size={18} />
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
          {/* Live Talk Button */}
          <button
            type="button"
            className={`btn-live-talk-head ${isLiveTalkOpen ? 'active' : ''}`}
            onClick={openLiveTalk}
            title="Enter Hands-Free Live Voice Mode"
          >
            <span className="live-talk-pulse"></span>
            <Radio size={13} className="live-radio-icon" />
            <span className="btn-live-talk-label-full">LIVE TALK</span>
            <span className="btn-live-talk-label-compact">LIVE</span>
          </button>

          {/* Speculative Streaming Pill Toggle */}
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
            <Zap size={13} />
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
                  <div>
                    {m.text}
                    {m.isStreaming && (
                      <span className="inline-block w-1.5 h-3.5 ml-1 bg-emerald-400 animate-pulse align-middle" />
                    )}
                  </div>
                  {m.role === 'assistant' && (
                    <div className="msg-actions-row">
                      {(m.audioBase64 || playingAudioId === m.id) && (
                        <button
                          type="button"
                          className={`btn-play-voice ${playingAudioId === m.id ? 'playing' : ''}`}
                          onClick={() => {
                            if (playingAudioId === m.id) {
                              stopAllAudio();
                            } else if (m.audioBase64) {
                              playAudio(m.audioBase64, m.id);
                            }
                          }}
                        >
                          <Volume2 size={13} className={playingAudioId === m.id ? 'animate-pulse' : ''} />
                          <span>{playingAudioId === m.id ? 'Speaking live...' : 'Play Voice'}</span>
                        </button>
                      )}

                      {/* Speculation status badge */}
                      {m.isEmergency ? (
                        <span className="badge-pill emergency">🚨 Emergency Override</span>
                      ) : m.isStreaming ? (
                        <span className="badge-pill committed animate-pulse">
                          ⚡ Streaming Tokens & TTS
                        </span>
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

            {isProcessing && !messages.some((m) => m.isStreaming) && (
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

            {/* Main Center Stage: Minimalist Clinical Voice Hub */}
            <div className="live-talk-stage">
              <VoiceOrb
                state={liveState}
                analyserRef={liveAnalyserRef}
                audioLevel={liveAudioLevel}
                isEmergency={!!activeEmergency}
                isUserSpeaking={isUserSpeaking}
              />

              {/* State Indicator Badge */}
              <div className={`live-state-pill ${liveState} ${activeEmergency ? 'emergency' : ''}`}>
                {liveState === 'listening' && (
                  isUserSpeaking || liveAudioLevel > 0.15 ? (
                    <>
                      <span className="state-pulse-dot cyan active-hearing"></span>
                      <span style={{ color: '#34D399', fontWeight: 700 }}>Hearing you... (Recording speech)</span>
                    </>
                  ) : (
                    <>
                      <span className="state-pulse-dot cyan"></span>
                      <span>Listening... (Speak naturally)</span>
                    </>
                  )
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
              <LiveSpectrumBars
                analyserRef={liveAnalyserRef}
                liveState={liveState}
                isUserSpeaking={isUserSpeaking}
                audioLevel={liveAudioLevel}
              />

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
