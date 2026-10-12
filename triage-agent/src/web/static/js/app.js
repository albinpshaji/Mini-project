/**
 * OmniSpec-Triage Client Application Logic
 * Integrates live text chat, voice streaming, EHR slot updates, and Piper TTS audio.
 */

document.addEventListener("DOMContentLoaded", () => {
  // --- Element Selectors ---
  const chatFeed = document.getElementById("chat-feed");
  const userInput = document.getElementById("user-input");
  const btnSend = document.getElementById("btn-send");
  const btnMic = document.getElementById("btn-mic");
  const btnReset = document.getElementById("btn-reset-chat");
  const btnToggleDrawer = document.getElementById("btn-toggle-drawer");
  const btnCloseDrawer = document.getElementById("btn-close-drawer");
  const clinicalDrawer = document.getElementById("clinical-drawer");
  const emergencyBanner = document.getElementById("emergency-banner");
  const emergencyCategory = document.getElementById("emergency-category");
  const recordingOverlay = document.getElementById("recording-overlay");
  const btnFinishRec = document.getElementById("btn-finish-rec");
  const btnCancelRec = document.getElementById("btn-cancel-rec");
  const toggleAutoPlay = document.getElementById("toggle-auto-play");
  const ttsAudioPlayer = document.getElementById("tts-audio-player");
  const guideIpBox = document.getElementById("guide-ip-box");

  // EHR Record Elements
  const slotComplaint = document.getElementById("slot-complaint");
  const slotDuration = document.getElementById("slot-duration");
  const slotSeverity = document.getElementById("slot-severity");
  const slotDept = document.getElementById("slot-dept");
  const slotTags = document.getElementById("slot-associated-tags");
  const slotConfidence = document.getElementById("slot-confidence");
  const safetyIndicatorBox = document.getElementById("safety-indicator-box");
  const safetyIndicatorText = document.getElementById("safety-indicator-text");

  // Biomarkers & Telemetry
  const bioPause = document.getElementById("bio-pause-ratio");
  const bioWheeze = document.getElementById("bio-wheeze-ratio");
  const bioIe = document.getElementById("bio-ie-ratio");
  const bioRationale = document.getElementById("bio-rationale");
  const bioBoost = document.getElementById("biomarker-boost-pill");
  const telTtfa = document.getElementById("tel-ttfa");
  const telSlm = document.getElementById("tel-slm");
  const telAsr = document.getElementById("tel-asr");
  const telVram = document.getElementById("tel-vram");
  const vramText = document.getElementById("vram-text");

  // Terminal & Latency Console Elements
  const btnToggleTerminal = document.getElementById("btn-toggle-terminal");
  const terminalModal = document.getElementById("terminal-modal");
  const terminalBackdrop = document.getElementById("terminal-backdrop");
  const btnCloseTerminal = document.getElementById("btn-close-terminal");
  const btnClearLogs = document.getElementById("btn-clear-logs");
  const btnCopyLogs = document.getElementById("btn-copy-logs");
  const toggleTermScroll = document.getElementById("toggle-term-scroll");
  const terminalLogsPre = document.getElementById("terminal-logs-pre");
  const terminalBody = document.getElementById("terminal-body");
  const liveLatTotal = document.getElementById("live-lat-total");
  const liveLatAsr = document.getElementById("live-lat-asr");
  const liveLatSlm = document.getElementById("live-lat-slm");
  const liveLatTts = document.getElementById("live-lat-tts");
  const liveLatVram = document.getElementById("live-lat-vram");

  // Speculative Streaming Controls
  const btnToggleSpec = document.getElementById("btn-toggle-spec");
  const specStatusText = document.getElementById("spec-status-text");
  const drawerSpecToggle = document.getElementById("drawer-spec-toggle");
  const drawerPipelineBadge = document.getElementById("drawer-pipeline-badge");
  const chipLiveSpec = document.getElementById("chip-live-spec");
  const liveLatSpec = document.getElementById("live-lat-spec");
  let speculativeEnabled = true;

  let logPollInterval = null;

  let isProcessing = false;
  let activeAudioBtn = null;

  // --- Initialize Audio Recorder ---
  const recorder = new TriageAudioRecorder({
    onRecordingStart: () => {
      btnMic.classList.add("recording");
      recordingOverlay.classList.remove("hidden");
    },
    onRecordingStop: () => {
      btnMic.classList.remove("recording");
      recordingOverlay.classList.add("hidden");
    },
    onAudioReady: async (audioBlob) => {
      await sendVoiceMessage(audioBlob);
    },
    onError: (err) => {
      alert(`Microphone Notice: ${err.message}`);
    },
  });

  // --- Microphone Controls ---
  btnMic.addEventListener("click", () => {
    if (recorder.isRecording) {
      recorder.stop();
    } else {
      recorder.start();
    }
  });

  if (btnFinishRec) {
    btnFinishRec.addEventListener("click", () => recorder.stop());
  }
  if (btnCancelRec) {
    btnCancelRec.addEventListener("click", () => recorder.cancel());
  }

  // --- Drawer Controls ---
  btnToggleDrawer.addEventListener("click", () => {
    clinicalDrawer.classList.toggle("drawer-open");
  });
  if (btnCloseDrawer) {
    btnCloseDrawer.addEventListener("click", () => {
      clinicalDrawer.classList.remove("drawer-open");
    });
  }

  // --- Speculative Streaming Toggle Logic ---
  async function setSpeculativeMode(enabled) {
    try {
      const res = await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ speculative_enabled: enabled }),
      });
      if (res.ok) {
        const data = await res.json();
        updateSpeculativeUI(data.speculative_enabled);
      }
    } catch (e) {
      console.warn("Failed to update speculative mode:", e);
    }
  }

  function updateSpeculativeUI(enabled) {
    speculativeEnabled = !!enabled;
    if (btnToggleSpec) {
      if (speculativeEnabled) {
        btnToggleSpec.className = "btn-spec-toggle active";
        btnToggleSpec.title = "Speculative Streaming Active (CMSPE). Tap to switch to Sequential Baseline.";
        if (specStatusText) specStatusText.textContent = "SPEC: ON";
      } else {
        btnToggleSpec.className = "btn-spec-toggle inactive";
        btnToggleSpec.title = "Sequential Baseline Active. Tap to switch to Speculative Streaming (CMSPE).";
        if (specStatusText) specStatusText.textContent = "SEQ: OFF";
      }
    }
    if (drawerSpecToggle) {
      drawerSpecToggle.checked = speculativeEnabled;
    }
    if (drawerPipelineBadge) {
      drawerPipelineBadge.className = `badge-pipeline ${speculativeEnabled ? "active" : "inactive"}`;
      drawerPipelineBadge.textContent = speculativeEnabled ? "⚡ Speculative: Active" : "Sequential Baseline";
    }
    if (chipLiveSpec && liveLatSpec) {
      chipLiveSpec.className = `lat-chip spec ${speculativeEnabled ? "active" : "inactive"}`;
      liveLatSpec.textContent = speculativeEnabled ? "⚡ ACTIVE" : "OFF (SEQ)";
    }
  }

  if (btnToggleSpec) {
    btnToggleSpec.addEventListener("click", () => {
      setSpeculativeMode(!speculativeEnabled);
    });
  }
  if (drawerSpecToggle) {
    drawerSpecToggle.addEventListener("change", (e) => {
      setSpeculativeMode(e.target.checked);
    });
  }

  // --- Reset Session ---
  btnReset.addEventListener("click", async () => {
    if (!confirm("Start a new patient triage session?")) return;
    try {
      await fetch("/api/reset", { method: "POST" });
      chatFeed.innerHTML = "";
      emergencyBanner.classList.add("hidden");
      resetSlotDisplay();
      addAssistantMessage(
        "Hello, I am your virtual triage assistant. Please describe what brings you in today — where do you feel discomfort, how severe is it, and when did it start?"
      );
    } catch (e) {
      console.error("Error resetting session:", e);
    }
  });

  // --- Quick Prompt Chips ---
  document.querySelectorAll(".prompt-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const text = chip.getAttribute("data-text");
      if (text && !isProcessing) {
        userInput.value = text;
        sendMessage();
      }
    });
  });

  // --- Auto-Expanding Textarea & Enter Key ---
  userInput.addEventListener("input", () => {
    userInput.style.height = "auto";
    userInput.style.height = Math.min(userInput.scrollHeight, 120) + "px";
  });

  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  btnSend.addEventListener("click", sendMessage);

  // --- Send Text Message Turn ---
  async function sendMessage() {
    const text = userInput.value.trim();
    if (!text || isProcessing) return;

    userInput.value = "";
    userInput.style.height = "auto";

    addUserMessage(text);
    const typingIndicator = addTypingIndicator();
    setProcessing(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const data = await response.json();
      removeTypingIndicator(typingIndicator);

      handleAssistantTurn(data);
    } catch (err) {
      console.error("Chat error:", err);
      removeTypingIndicator(typingIndicator);
      addAssistantMessage(
        "I encountered a connection issue communicating with the local clinical reasoning model. Please ensure Ollama is running and try again."
      );
    } finally {
      setProcessing(false);
    }
  }

  // --- Send Voice Turn from Mobile Phone ---
  async function sendVoiceMessage(audioBlob) {
    if (isProcessing) return;

    addUserMessage("🎙️ [Speaking on Microphone...]");
    const typingIndicator = addTypingIndicator();
    setProcessing(true);

    try {
      const formData = new FormData();
      const ext = audioBlob.type.includes("mp4") ? "mp4" : "webm";
      formData.append("audio_file", audioBlob, `speech_input.${ext}`);

      const response = await fetch("/api/voice", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned ${response.status}`);
      }

      const data = await response.json();
      removeTypingIndicator(typingIndicator);

      // Update the last user message text with real transcript
      const lastUserMsg = chatFeed.querySelector(".msg-user:last-of-type .msg-text");
      if (lastUserMsg) {
        lastUserMsg.innerHTML = `<span style="opacity: 0.75; font-size: 0.85em;">🎙️ Transcribed:</span><br>"${escapeHtml(data.transcribed_text)}"`;
      }

      handleAssistantTurn(data);
    } catch (err) {
      console.error("Voice processing error:", err);
      removeTypingIndicator(typingIndicator);
      addAssistantMessage(
        `Unable to process speech recording: ${err.message}. Please try speaking again or use text chat.`
      );
    } finally {
      setProcessing(false);
    }
  }

  // --- Process & Render Assistant Response ---
  function handleAssistantTurn(data) {
    const msgCard = addAssistantMessage(data.agent_response, data.is_emergency, data.audio_base64, data.speculative);

    // Emergency Banner
    if (data.is_emergency) {
      emergencyBanner.classList.remove("hidden");
      emergencyCategory.textContent = (data.red_flag_category || "EMERGENCY ESCALATION").toUpperCase();
      safetyIndicatorBox.className = "safety-indicator-box danger";
      safetyIndicatorText.textContent = `RED-FLAG TRIGGERED (${data.red_flag_category || "Urgent"})`;
    }

    // Auto-Play Voice Audio on Phone
    if (toggleAutoPlay.checked && data.audio_base64) {
      playAudio(data.audio_base64, msgCard.querySelector(".btn-audio-play"));
    }

    // Update Clinical EHR Slots
    if (data.slots) {
      updateSlotDisplay(data.slots);
    }

    // Update Paralinguistic Biomarkers
    if (data.biomarkers) {
      updateBiomarkerDisplay(data.biomarkers);
    }

    // Update Telemetry
    if (data.telemetry) {
      updateTelemetryDisplay(data.telemetry, data.vram);
    }
  }

  // --- Chat Feed Rendering Helpers ---
  function addUserMessage(text) {
    const card = document.createElement("div");
    card.className = "msg-card msg-user";
    card.innerHTML = `
      <div class="msg-avatar">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18">
          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
          <circle cx="12" cy="7" r="4"/>
        </svg>
      </div>
      <div class="msg-bubble">
        <div class="msg-meta">
          <span class="msg-author">Patient</span>
          <span class="msg-time">${formatTime()}</span>
        </div>
        <div class="msg-text">${escapeHtml(text)}</div>
      </div>
    `;
    chatFeed.appendChild(card);
    scrollToBottom();
    return card;
  }

  function addAssistantMessage(text, isEmergency = false, audioBase64 = null, speculativeInfo = null) {
    const card = document.createElement("div");
    card.className = `msg-card msg-assistant ${isEmergency ? "msg-emergency" : ""}`;

    const playBtnHtml = audioBase64
      ? `<button type="button" class="btn-audio-play" title="Listen to response">
           <svg viewBox="0 0 24 24" fill="currentColor" width="13" height="13">
             <polygon points="5 3 19 12 5 21 5 3"/>
           </svg>
           <span>Play Voice</span>
         </button>`
      : "";

    let specPillHtml = "";
    if (isEmergency) {
      specPillHtml = `<span class="telemetry-pill emergency">🚨 Emergency Override</span>`;
    } else if (speculativeInfo && speculativeInfo.enabled) {
      if (speculativeInfo.committed) {
        specPillHtml = `<span class="telemetry-pill spec-committed">⚡ Speculation Committed (Sim: ${Math.round(speculativeInfo.similarity * 100)}%)</span>`;
      } else if (speculativeInfo.status === "DIVERGED") {
        specPillHtml = `<span class="telemetry-pill spec-diverged">🔄 Speculation Rolled Back (Sim: ${Math.round(speculativeInfo.similarity * 100)}%)</span>`;
      } else {
        specPillHtml = `<span class="telemetry-pill spec-committed">⚡ Speculative CMSPE</span>`;
      }
    } else {
      specPillHtml = `<span class="telemetry-pill">Sequential Baseline</span>`;
    }

    card.innerHTML = `
      <div class="msg-avatar" style="${isEmergency ? 'color: var(--red-emergency); border-color: var(--red-emergency);' : ''}">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18">
          <circle cx="12" cy="7" r="4"/>
          <path d="M5.5 21v-2a6.5 6.5 0 0 1 13 0v2"/>
        </svg>
      </div>
      <div class="msg-bubble">
        <div class="msg-meta">
          <span class="msg-author">${isEmergency ? "🚨 Emergency Triage Directive" : "Triage Nurse AI"}</span>
          <span class="msg-time">${formatTime()}</span>
        </div>
        <div class="msg-text">${escapeHtml(text)}</div>
        <div class="msg-actions">
          ${playBtnHtml}
          ${specPillHtml}
        </div>
      </div>
    `;

    const playBtn = card.querySelector(".btn-audio-play");
    if (playBtn && audioBase64) {
      playBtn.addEventListener("click", () => playAudio(audioBase64, playBtn));
    }

    chatFeed.appendChild(card);
    scrollToBottom();
    return card;
  }

  function addTypingIndicator() {
    const card = document.createElement("div");
    card.className = "msg-card msg-assistant msg-typing";
    card.innerHTML = `
      <div class="msg-avatar">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18">
          <circle cx="12" cy="7" r="4"/>
          <path d="M5.5 21v-2a6.5 6.5 0 0 1 13 0v2"/>
        </svg>
      </div>
      <div class="msg-bubble">
        <div class="msg-text" style="color: var(--teal-primary); font-size: 0.85rem;">
          <span class="status-pulse" style="display: inline-block; vertical-align: middle; margin-right: 6px;"></span>
          Nurse reasoning clinical intake...
        </div>
      </div>
    `;
    chatFeed.appendChild(card);
    scrollToBottom();
    return card;
  }

  function removeTypingIndicator(el) {
    if (el && el.parentNode) {
      el.parentNode.removeChild(el);
    }
  }

  // --- Audio Player Controller ---
  function playAudio(audioDataUri, btn) {
    if (!audioDataUri) return;

    if (activeAudioBtn && activeAudioBtn !== btn) {
      activeAudioBtn.classList.remove("playing");
      activeAudioBtn.querySelector("span").textContent = "Play Voice";
    }

    ttsAudioPlayer.src = audioDataUri;
    ttsAudioPlayer.play().then(() => {
      if (btn) {
        btn.classList.add("playing");
        btn.querySelector("span").textContent = "Speaking...";
        activeAudioBtn = btn;
      }
    }).catch((e) => {
      console.warn("Audio auto-play prevented by browser policy (requires user tap):", e);
    });

    ttsAudioPlayer.onended = () => {
      if (btn) {
        btn.classList.remove("playing");
        btn.querySelector("span").textContent = "Replay Voice";
      }
      activeAudioBtn = null;
    };
  }

  // --- Clinical EHR Slot Updates ---
  function updateSlotDisplay(slots) {
    slotComplaint.textContent = slots.chief_complaint || "Evaluating...";
    slotDuration.textContent = slots.duration || "Not specified";
    slotSeverity.textContent = (slots.severity || "Pending").toUpperCase();
    slotDept.textContent = slots.recommended_department || "Assessment in progress";
    slotConfidence.textContent = `Confidence: ${(slots.confidence || "Low").toUpperCase()}`;

    // Associated Symptoms Tags
    slotTags.innerHTML = "";
    if (slots.associated_symptoms && slots.associated_symptoms.length > 0) {
      slots.associated_symptoms.forEach((s) => {
        const tag = document.createElement("span");
        tag.className = "slot-tag";
        tag.textContent = s;
        slotTags.appendChild(tag);
      });
    } else {
      slotTags.innerHTML = `<span class="empty-tag">None documented</span>`;
    }

    if (slots.red_flag_present) {
      safetyIndicatorBox.className = "safety-indicator-box danger";
      safetyIndicatorText.textContent = "High-Acuity Red-Flag Present";
    } else {
      safetyIndicatorBox.className = "safety-indicator-box safe";
      safetyIndicatorText.textContent = "No active red-flag triggers";
    }
  }

  function resetSlotDisplay() {
    slotComplaint.textContent = "None recorded yet";
    slotDuration.textContent = "—";
    slotSeverity.textContent = "—";
    slotDept.textContent = "Intake Pending";
    slotConfidence.textContent = "Confidence: Low";
    slotTags.innerHTML = `<span class="empty-tag">None listed</span>`;
    safetyIndicatorBox.className = "safety-indicator-box safe";
    safetyIndicatorText.textContent = "No active red-flag triggers";
  }

  // --- Paralinguistic Biomarkers Updates ---
  function updateBiomarkerDisplay(bio) {
    bioPause.textContent = bio.speech_to_pause_ratio.toFixed(2);
    bioWheeze.textContent = bio.wheeze_band_power_ratio.toFixed(2);
    bioIe.textContent = bio.ie_ratio.toFixed(2);
    bioRationale.textContent = bio.clinical_rationale || "Normal respiratory pattern.";

    if (bio.acuity_boost > 0) {
      bioBoost.textContent = `Acuity: +${bio.acuity_boost} (Elevated)`;
      bioBoost.style.color = "var(--red-emergency)";
    } else {
      bioBoost.textContent = "Acuity: Normal";
      bioBoost.style.color = "var(--text-secondary)";
    }
  }

  // --- Hardware Telemetry Updates ---
  function updateTelemetryDisplay(tel, vram) {
    telTtfa.textContent = `${tel.tts_latency_ms || "--"} ms`;
    telSlm.textContent = `${tel.slm_latency_ms || "--"} ms`;
    telAsr.textContent = `${tel.asr_latency_ms || "--"} ms`;

    if (vram && vram.used_mb) {
      telVram.textContent = `${(vram.used_mb / 1024).toFixed(1)} GB`;
      vramText.textContent = `VRAM: ${vram.used_mb}MB / ${vram.total_mb}MB`;
    }

    // Also update live latency chips in terminal window
    if (liveLatTotal && tel.total_latency_ms) liveLatTotal.textContent = `${tel.total_latency_ms} ms`;
    if (liveLatAsr && tel.asr_latency_ms) liveLatAsr.textContent = `${tel.asr_latency_ms} ms`;
    if (liveLatSlm && tel.slm_latency_ms) liveLatSlm.textContent = `${tel.slm_latency_ms} ms`;
    if (liveLatTts && tel.tts_latency_ms) liveLatTts.textContent = `${tel.tts_latency_ms} ms`;
    if (liveLatVram && vram && vram.used_mb) liveLatVram.textContent = `${vram.used_mb} MB`;

    if (chipLiveSpec && liveLatSpec) {
      if (tel.speculative) {
        if (tel.speculative.committed) {
          chipLiveSpec.className = "lat-chip spec active";
          liveLatSpec.textContent = "⚡ COMMITTED";
        } else if (tel.speculative.status === "DIVERGED") {
          chipLiveSpec.className = "lat-chip spec inactive";
          liveLatSpec.textContent = "ROLLED BACK";
        } else if (tel.speculative.status === "EMERGENCY_OVERRIDE") {
          chipLiveSpec.className = "lat-chip spec active";
          liveLatSpec.textContent = "OVERRIDE";
        } else {
          chipLiveSpec.className = `lat-chip spec ${tel.speculative.enabled ? "active" : "inactive"}`;
          liveLatSpec.textContent = tel.speculative.enabled ? "⚡ ACTIVE" : "OFF (SEQ)";
        }
      }
    }
  }

  // --- Initial System Health Fetch ---
  async function fetchStatus() {
    try {
      const res = await fetch("/api/status");
      if (res.ok) {
        const data = await res.json();
        if (guideIpBox && data.lan_ip) {
          guideIpBox.textContent = `http://${data.lan_ip}:8000`;
        }
        if (data.vram && data.vram.used_mb) {
          vramText.textContent = `VRAM: ${data.vram.used_mb}MB / ${data.vram.total_mb}MB`;
        }
        if (data.latest_slots && data.latest_slots.chief_complaint) {
          updateSlotDisplay(data.latest_slots);
        }
        if (data.speculative_enabled !== undefined) {
          updateSpeculativeUI(data.speculative_enabled);
        }
      }
    } catch (e) {
      console.warn("Status check failed:", e);
    }
  }
  fetchStatus();

  // --- Terminal & Latency Console Logic ---
  async function fetchLogs() {
    try {
      const res = await fetch("/api/logs?limit=120");
      if (!res.ok) return;
      const data = await res.json();

      if (terminalLogsPre && data.logs) {
        terminalLogsPre.textContent = data.logs.join("\n");
        if (toggleTermScroll && toggleTermScroll.checked && terminalBody) {
          terminalBody.scrollTop = terminalBody.scrollHeight;
        }
      }

      if (data.latest_telemetry) {
        const tel = data.latest_telemetry;
        if (liveLatTotal && tel.total_latency_ms) liveLatTotal.textContent = `${tel.total_latency_ms} ms`;
        if (liveLatAsr && tel.asr_latency_ms) liveLatAsr.textContent = `${tel.asr_latency_ms} ms`;
        if (liveLatSlm && tel.slm_latency_ms) liveLatSlm.textContent = `${tel.slm_latency_ms} ms`;
        if (liveLatTts && tel.tts_latency_ms) liveLatTts.textContent = `${tel.tts_latency_ms} ms`;
      }
      if (data.vram && data.vram.used_mb && liveLatVram) {
        liveLatVram.textContent = `${data.vram.used_mb} MB`;
      }
    } catch (err) {
      console.warn("Log fetch error:", err);
    }
  }

  function openTerminal() {
    if (!terminalModal) return;
    terminalModal.classList.remove("hidden");
    fetchLogs();
    if (!logPollInterval) {
      logPollInterval = setInterval(fetchLogs, 1500);
    }
  }

  function closeTerminal() {
    if (!terminalModal) return;
    terminalModal.classList.add("hidden");
    if (logPollInterval) {
      clearInterval(logPollInterval);
      logPollInterval = null;
    }
  }

  if (btnToggleTerminal) {
    btnToggleTerminal.addEventListener("click", openTerminal);
  }
  if (btnCloseTerminal) {
    btnCloseTerminal.addEventListener("click", closeTerminal);
  }
  if (terminalBackdrop) {
    terminalBackdrop.addEventListener("click", closeTerminal);
  }
  if (btnClearLogs && terminalLogsPre) {
    btnClearLogs.addEventListener("click", () => {
      terminalLogsPre.textContent = "Logs cleared in viewer.";
    });
  }
  if (btnCopyLogs && terminalLogsPre) {
    btnCopyLogs.addEventListener("click", () => {
      navigator.clipboard.writeText(terminalLogsPre.textContent).then(() => {
        const orig = btnCopyLogs.textContent;
        btnCopyLogs.textContent = "Copied!";
        setTimeout(() => { btnCopyLogs.textContent = orig; }, 1200);
      });
    });
  }

  // --- Utilities ---
  function setProcessing(val) {
    isProcessing = val;
    btnSend.disabled = val;
    btnMic.disabled = val;
  }

  function scrollToBottom() {
    chatFeed.scrollTop = chatFeed.scrollHeight;
  }

  function formatTime() {
    const d = new Date();
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
