/**
 * Audio Recorder Module for OmniSpec-Triage
 * Supports both:
 * 1. In-Browser Web Audio API / MediaRecorder with live visualizer waveform
 * 2. Native Mobile OS Voice Recorder (<input type="file" capture="microphone">)
 */

class TriageAudioRecorder {
  constructor(options = {}) {
    this.onRecordingStart = options.onRecordingStart || (() => {});
    this.onRecordingStop = options.onRecordingStop || (() => {});
    this.onAudioReady = options.onAudioReady || (() => {});
    this.onError = options.onError || ((err) => console.error(err));

    this.mediaStream = null;
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.audioContext = null;
    this.analyser = null;
    this.animFrameId = null;

    this.isRecording = false;
    this.startTime = 0;
    this.timerInterval = null;

    this.canvas = document.getElementById("waveform-canvas");
    this.canvasCtx = this.canvas ? this.canvas.getContext("2d") : null;
    this.timerEl = document.getElementById("recording-timer");

    this._initNativeFileInput();
  }

  _initNativeFileInput() {
    const fileInput = document.getElementById("native-audio-file");
    if (!fileInput) return;

    fileInput.addEventListener("change", async (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;

      console.log("Native phone audio file selected:", file.name, file.size, file.type);
      this.onAudioReady(file);
      // Reset input value so the same file can be re-recorded if needed
      fileInput.value = "";
    });
  }

  async start() {
    if (this.isRecording) return;

    // Check if getUserMedia is supported
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      this.onError(
        new Error(
          "In-browser microphone is not directly supported over plain HTTP on this mobile browser. " +
          "Please tap the Paperclip/Microphone upload button next to the input to record using your phone's native voice recorder!"
        )
      );
      return;
    }

    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      this.audioChunks = [];
      
      // Determine optimal mimeType supported by this mobile/desktop browser
      const mimeTypes = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/mp4",
        "audio/ogg;codecs=opus",
        "audio/wav",
      ];
      let selectedMime = "";
      for (const mime of mimeTypes) {
        if (MediaRecorder.isTypeSupported(mime)) {
          selectedMime = mime;
          break;
        }
      }

      const recorderOptions = selectedMime ? { mimeType: selectedMime } : {};
      this.mediaRecorder = new MediaRecorder(this.mediaStream, recorderOptions);

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.mediaRecorder.onstop = () => {
        const mimeType = this.mediaRecorder.mimeType || "audio/webm";
        const audioBlob = new Blob(this.audioChunks, { type: mimeType });
        console.log(`Audio recording captured: ${audioBlob.size} bytes (${mimeType})`);
        this.onAudioReady(audioBlob);
        this._cleanupStream();
      };

      this.mediaRecorder.start(250); // Emit chunks every 250ms
      this.isRecording = true;
      this.startTime = Date.now();

      this._startTimer();
      this._startWaveformVisualizer();
      this.onRecordingStart();

    } catch (err) {
      console.warn("Failed to acquire microphone stream:", err);
      let userMsg = err.message;
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        userMsg = "Microphone access was denied. Please allow microphone permission in your phone browser settings, or use the Native Voice Upload button.";
      } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        userMsg = "No microphone hardware detected on this device.";
      }
      this.onError(new Error(userMsg));
    }
  }

  stop() {
    if (!this.isRecording) return;
    this.isRecording = false;
    this._stopTimer();
    this._stopWaveformVisualizer();

    if (this.mediaRecorder && this.mediaRecorder.state !== "inactive") {
      this.mediaRecorder.stop();
    }
    this.onRecordingStop();
  }

  cancel() {
    if (!this.isRecording) return;
    this.isRecording = false;
    this._stopTimer();
    this._stopWaveformVisualizer();

    if (this.mediaRecorder && this.mediaRecorder.state !== "inactive") {
      this.mediaRecorder.stop();
    }
    this.audioChunks = [];
    this._cleanupStream();
    this.onRecordingStop();
  }

  _cleanupStream() {
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((t) => t.stop());
      this.mediaStream = null;
    }
  }

  _startTimer() {
    this._stopTimer();
    if (this.timerEl) this.timerEl.textContent = "00:00";

    this.timerInterval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
      const mins = String(Math.floor(elapsed / 60)).padStart(2, "0");
      const secs = String(elapsed % 60).padStart(2, "0");
      if (this.timerEl) {
        this.timerEl.textContent = `${mins}:${secs}`;
      }
      // Safety auto-stop after 25 seconds
      if (elapsed >= 25) {
        this.stop();
      }
    }, 200);
  }

  _stopTimer() {
    if (this.timerInterval) {
      clearInterval(this.timerInterval);
      this.timerInterval = null;
    }
  }

  _startWaveformVisualizer() {
    if (!this.canvas || !this.mediaStream) return;

    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.audioContext = new AudioCtx();
      const source = this.audioContext.createMediaStreamSource(this.mediaStream);
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 64;
      source.connect(this.analyser);

      const bufferLength = this.analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      const draw = () => {
        if (!this.isRecording) return;
        this.animFrameId = requestAnimationFrame(draw);

        this.analyser.getByteFrequencyData(dataArray);

        const width = this.canvas.width;
        const height = this.canvas.height;
        this.canvasCtx.clearRect(0, 0, width, height);

        const barWidth = (width / bufferLength) * 1.5;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
          const barHeight = (dataArray[i] / 255) * height;

          // Gradient from cyan to medical teal
          const grad = this.canvasCtx.createLinearGradient(0, height, 0, height - barHeight);
          grad.addColorStop(0, "rgba(14, 165, 233, 0.4)");
          grad.addColorStop(1, "rgba(6, 182, 212, 1.0)");

          this.canvasCtx.fillStyle = grad;
          this.canvasCtx.fillRect(x, height - barHeight, barWidth - 1, barHeight);

          x += barWidth;
        }
      };

      draw();
    } catch (e) {
      console.warn("Waveform visualizer setup skipped:", e);
    }
  }

  _stopWaveformVisualizer() {
    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
    if (this.audioContext) {
      try {
        this.audioContext.close();
      } catch (e) {}
      this.audioContext = null;
    }
    if (this.canvasCtx && this.canvas) {
      this.canvasCtx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }
}
