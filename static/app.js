const sourceSelect = document.querySelector("#source");
const refreshSourcesBtn = document.querySelector("#refreshSourcesBtn");
const sessionNameInput = document.querySelector("#sessionName");
const browserInputSelect = document.querySelector("#browserInput");
const asrBackendInput = document.querySelector("#asrBackend");
const mlxModelInput = document.querySelector("#mlxModel");
const preloadBtn = document.querySelector("#preloadBtn");
const refineBtn = document.querySelector("#refineBtn");
const summaryBtn = document.querySelector("#summaryBtn");
const startBtn = document.querySelector("#startBtn");
const stopBtn = document.querySelector("#stopBtn");
const statusEl = document.querySelector("#status");
const savedPathEl = document.querySelector("#savedPath");
const liveTranscriptEl = document.querySelector("#liveTranscript");
const englishTranscriptEl = document.querySelector("#englishTranscript");
const chineseTranscriptEl = document.querySelector("#chineseTranscript");
const systemLogEl = document.querySelector("#systemLog");
const textSizeInput = document.querySelector("#textSize");
const includeMicInput = document.querySelector("#includeMic");
const autoSummaryInput = document.querySelector("#autoSummary");
const ollamaModelInput = document.querySelector("#ollamaModel");
const summaryOllamaModelInput = document.querySelector("#summaryOllamaModel");
const finalOllamaModelInput = document.querySelector("#finalOllamaModel");
let latestEventId = 0;
let mediaRecorder = null;
let captureStream = null;
let recorderStream = null;
let micStream = null;
let audioContext = null;
let sessionId = null;
let chunkIndex = 0;
let recordingActive = false;
let segmentTimer = null;
let recorderMimeType = "audio/webm";
let recorderChunkMs = 3000;
let activeUploads = 0;
let liveChunkCount = 0;
let completeParagraphCount = 0;
let activeEnglishParagraph = null;
let activeChineseParagraph = null;

function waitForRecorderStop(recorder) {
  if (!recorder || recorder.state === "inactive") {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    recorder.addEventListener("stop", resolve, {once: true});
    recorder.stop();
  });
}

function setPlaceholder(target, message) {
  target.innerHTML = "";
  const div = document.createElement("div");
  div.className = "placeholder";
  div.textContent = message;
  target.appendChild(div);
}

function addSystem(message, error = false) {
  const div = document.createElement("div");
  div.className = error ? "system error" : "system";
  div.textContent = message;
  systemLogEl.appendChild(div);
  systemLogEl.scrollTop = systemLogEl.scrollHeight;
}

function addCaption(event) {
  if (liveChunkCount === 0) {
    liveTranscriptEl.innerHTML = "";
  }
  liveChunkCount += 1;
  const article = document.createElement("article");
  article.className = "entry";
  article.innerHTML = `
    <div class="meta">${escapeHtml(event.timestamp || event.time || "")}</div>
    <div class="english">${escapeHtml(event.english || "")}</div>
  `;
  liveTranscriptEl.appendChild(article);
  liveTranscriptEl.scrollTop = liveTranscriptEl.scrollHeight;
  if (event.transcript) {
    savedPathEl.textContent = event.transcript;
  }
  if (event.polished && event.polished.chinese) {
    addPolishedTranslation(event.polished);
  }
}

function addPolishedTranslation(polished) {
  if (completeParagraphCount === 0) {
    englishTranscriptEl.innerHTML = "";
    chineseTranscriptEl.innerHTML = "";
  }
  completeParagraphCount += 1;
  appendToLanguageParagraph(englishTranscriptEl, "english", polished.english || "");
  appendToLanguageParagraph(chineseTranscriptEl, "chinese", polished.chinese || "");
}

function appendToLanguageParagraph(container, language, text) {
  const cleanText = String(text || "").trim();
  if (!cleanText) {
    return;
  }
  const activeKey = language === "english" ? "activeEnglishParagraph" : "activeChineseParagraph";
  let paragraph = language === "english" ? activeEnglishParagraph : activeChineseParagraph;
  const shouldStartParagraph = !paragraph || paragraph.textContent.length > 1400;
  if (shouldStartParagraph) {
    paragraph = document.createElement("p");
    paragraph.className = `complete-paragraph ${language}`;
    container.appendChild(paragraph);
    if (activeKey === "activeEnglishParagraph") {
      activeEnglishParagraph = paragraph;
    } else {
      activeChineseParagraph = paragraph;
    }
  }
  paragraph.textContent = joinParagraphText(paragraph.textContent, cleanText);
  container.scrollTop = container.scrollHeight;
}

function joinParagraphText(existing, next) {
  const current = String(existing || "").trim();
  if (!current) {
    return next;
  }
  if (/[\s([{“"']$/.test(current) || /^[,.;:!?，。！？；：)]/.test(next)) {
    return `${current}${next}`;
  }
  return `${current} ${next}`;
}

function applyTextSize() {
  document.documentElement.style.setProperty("--caption-size", `${textSizeInput.value}px`);
}

function slugifySessionName(value) {
  return String(value || "")
    .trim()
    .replace(/[^A-Za-z0-9\u4e00-\u9fa5._-]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 60);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function loadSources() {
  try {
    sourceSelect.innerHTML = "";
    const loading = document.createElement("option");
    loading.textContent = "Loading audio sources...";
    loading.value = "";
    sourceSelect.appendChild(loading);

    const res = await fetch("/api/sources", {cache: "no-store"});
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const sources = await res.json();
    sourceSelect.innerHTML = "";
    for (const source of sources) {
      const option = document.createElement("option");
      option.value = source.id;
      option.textContent = source.label;
      sourceSelect.appendChild(option);
    }
    if (sources.some((source) => source.id === "browser_system")) {
      sourceSelect.value = "browser_system";
    }
    if (!sources.length) {
      addSystem("No audio sources found. Check Windows microphone/privacy settings, then click Refresh.", true);
    } else {
      statusEl.textContent = `${sources.length} audio source(s) found`;
    }
  } catch (error) {
    sourceSelect.innerHTML = "";
    const option = document.createElement("option");
    option.textContent = "Cannot load sources";
    option.value = "";
    sourceSelect.appendChild(option);
    addSystem(`Cannot load audio sources. Is the launcher window still open? ${error}`, true);
  }
}

async function loadBrowserInputs() {
  browserInputSelect.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Default input";
  browserInputSelect.appendChild(placeholder);

  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    addSystem("Browser cannot enumerate audio input devices.", true);
    return;
  }

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const inputs = devices.filter((device) => device.kind === "audioinput");
    for (const input of inputs) {
      const option = document.createElement("option");
      option.value = input.deviceId;
      option.textContent = input.label || `Audio input ${browserInputSelect.length}`;
      browserInputSelect.appendChild(option);
    }
    if (!inputs.length) {
      addSystem("No browser audio inputs found. Allow microphone permission, then click Refresh.", true);
    }
  } catch (error) {
    addSystem(`Cannot list browser audio inputs until microphone permission is granted: ${error.message || error}`);
  }
}

async function start() {
  liveChunkCount = 0;
  completeParagraphCount = 0;
  activeEnglishParagraph = null;
  activeChineseParagraph = null;
  setPlaceholder(liveTranscriptEl, "Waiting for live English...");
  setPlaceholder(englishTranscriptEl, "Waiting for repaired English...");
  setPlaceholder(chineseTranscriptEl, "等待更完整的中文段落...");
  systemLogEl.innerHTML = "";
  savedPathEl.textContent = "";
  if (!sourceSelect.value) {
    addSystem("Choose an audio source first. Click Refresh if the list is empty.", true);
    return;
  }
  try {
    const timestamp = new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-");
    const sessionSlug = slugifySessionName(sessionNameInput.value);
    sessionId = sessionSlug ? `${timestamp}_${sessionSlug}` : timestamp;
    chunkIndex = 0;
    if (sourceSelect.value === "browser_system") {
      captureStream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        }
      });
      const audioTracks = captureStream.getAudioTracks();
      if (!audioTracks.length) {
        throw new Error("No audio track selected. In Chrome's share dialog, enable Share audio.");
      }
      if (includeMicInput.checked) {
        micStream = await getSelectedMicStream();
        recorderStream = mixAudioStreams(captureStream, micStream);
      } else {
        recorderStream = new MediaStream(audioTracks);
      }
    } else {
      captureStream = await getSelectedMicStream();
      recorderStream = captureStream;
      await loadBrowserInputs();
    }

    recorderMimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : "audio/webm";
    recorderChunkMs = Number(document.querySelector("#chunkSeconds").value) * 1000;
    recordingActive = true;
    startRecorderSegment();
    startBtn.disabled = true;
    stopBtn.disabled = false;
    statusEl.textContent = "recording";
    addSystem("Recording through browser capture. First transcript may take a little longer while Whisper loads.");
  } catch (error) {
    recordingActive = false;
    if (captureStream) {
      for (const track of captureStream.getTracks()) {
        track.stop();
      }
    }
    if (micStream) {
      for (const track of micStream.getTracks()) {
        track.stop();
      }
    }
    if (audioContext) {
      await audioContext.close();
    }
    captureStream = null;
    micStream = null;
    audioContext = null;
    recorderStream = null;
    addSystem(`Cannot start browser capture: ${error.message || error}`, true);
    startBtn.disabled = false;
    stopBtn.disabled = true;
  }
}

async function getSelectedMicStream() {
  const selectedDevice = browserInputSelect.value;
  return navigator.mediaDevices.getUserMedia({
    audio: selectedDevice
      ? {
          deviceId: {exact: selectedDevice},
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      : {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
  });
}

function mixAudioStreams(systemStream, microphoneStream) {
  audioContext = new AudioContext();
  const destination = audioContext.createMediaStreamDestination();

  const systemSource = audioContext.createMediaStreamSource(new MediaStream(systemStream.getAudioTracks()));
  const micSource = audioContext.createMediaStreamSource(new MediaStream(microphoneStream.getAudioTracks()));

  const systemGain = audioContext.createGain();
  const micGain = audioContext.createGain();
  systemGain.gain.value = 1.0;
  micGain.gain.value = 1.0;

  systemSource.connect(systemGain).connect(destination);
  micSource.connect(micGain).connect(destination);

  return destination.stream;
}

async function preloadWhisper() {
  preloadBtn.disabled = true;
  statusEl.textContent = "preloading whisper";
  addSystem("Preloading Whisper model...");
  try {
    const res = await fetch("/api/preload", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        asr_backend: asrBackendInput.value,
        whisper_model: document.querySelector("#whisperModel").value,
        mlx_model: mlxModelInput.value,
        device: document.querySelector("#device").value,
        compute_type: document.querySelector("#computeType").value,
        ollama_model: ollamaModelInput.value,
        summary_ollama_model: summaryOllamaModelInput.value,
        final_ollama_model: finalOllamaModelInput.value,
        ollama_host: "http://localhost:11434"
      })
    });
    const data = await res.json();
    if (!data.ok) {
      addSystem(data.error || "Preload failed.", true);
      statusEl.textContent = "preload failed";
      return;
    }
    addSystem(
      `Preload ready: ASR ${data.asr_backend || "unknown"} / ${data.model} (${data.device}, ${data.compute_type}); ` +
      `Ollama ${((data.ollama_models || []).join(", ") || "none")} in ${data.elapsed_seconds}s`
    );
    statusEl.textContent = "whisper ready";
  } catch (error) {
    addSystem(`Preload failed: ${error.message || error}`, true);
    statusEl.textContent = "preload failed";
  } finally {
    preloadBtn.disabled = false;
  }
}

async function stop() {
  try {
    recordingActive = false;
    if (segmentTimer) {
      clearTimeout(segmentTimer);
      segmentTimer = null;
    }
    await waitForRecorderStop(mediaRecorder);
    mediaRecorder = null;
    if (captureStream) {
      for (const track of captureStream.getTracks()) {
        track.stop();
      }
    }
    if (micStream) {
      for (const track of micStream.getTracks()) {
        track.stop();
      }
    }
    if (audioContext) {
      await audioContext.close();
    }
    captureStream = null;
    micStream = null;
    audioContext = null;
    recorderStream = null;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    statusEl.textContent = "stopped";
    addSystem("Stopped.");
    await waitForPendingChunks();
    await flushTranslation();
    if (autoSummaryInput.checked) {
      generateSummary(true);
    }
  } catch (error) {
    addSystem(`Cannot stop cleanly. You can close the launcher window. ${error}`, true);
  }
}

async function flushTranslation() {
  if (!sessionId) {
    return;
  }
  try {
    await fetch("/api/flush_translation", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        session_id: sessionId,
        ollama_model: ollamaModelInput.value,
        ollama_host: "http://localhost:11434",
        translate: document.querySelector("#translate").checked
      })
    });
  } catch (error) {
    addSystem(`Could not flush final translation chunk: ${error.message || error}`, true);
  }
}

async function waitForPendingChunks(timeoutMs = 90000) {
  const started = Date.now();
  await new Promise((resolve) => setTimeout(resolve, 800));
  while (activeUploads > 0 && Date.now() - started < timeoutMs) {
    statusEl.textContent = `waiting for ${activeUploads} chunk(s)`;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
}

async function refineTranscript() {
  if (!sessionId) {
    addSystem("No active session to refine yet.", true);
    return null;
  }
  refineBtn.disabled = true;
  statusEl.textContent = "creating paragraph transcript";
  addSystem("Creating paragraph transcript...");
  try {
    const res = await fetch("/api/refine_session", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        session_id: sessionId,
        ollama_model: ollamaModelInput.value,
        ollama_host: "http://localhost:11434"
      })
    });
    const data = await res.json();
    if (!data.ok) {
      addSystem(data.error || "Paragraph transcript failed.", true);
      statusEl.textContent = "paragraph transcript failed";
      return null;
    }
    savedPathEl.textContent = data.output;
    addSystem(`Paragraph transcript saved: ${data.output}`);
    statusEl.textContent = "paragraph transcript ready";
    return data.output;
  } catch (error) {
    addSystem(`Paragraph transcript failed: ${error.message || error}`, true);
    statusEl.textContent = "paragraph transcript failed";
    return null;
  } finally {
    refineBtn.disabled = false;
  }
}

async function generateSummary(fromStop = false) {
  if (!sessionId) {
    addSystem("No active session to summarize yet.", true);
    return;
  }
  summaryBtn.disabled = true;
  statusEl.textContent = "generating summary";
  addSystem("Generating meeting summary...");
  try {
    const res = await fetch("/api/summarize_session", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        session_id: sessionId,
        ollama_model: summaryOllamaModelInput.value,
        final_ollama_model: finalOllamaModelInput.value,
        ollama_host: "http://localhost:11434"
      })
    });
    const data = await res.json();
    if (!data.ok) {
      addSystem(data.error || "Summary failed.", true);
      statusEl.textContent = "summary failed";
      return;
    }
    savedPathEl.textContent = data.output;
    addSystem(`Summary saved: ${data.output}`);
    if (data.detailed_output) {
      addSystem(`Detailed review saved: ${data.detailed_output}`);
    }
    if (data.output_zh) {
      addSystem(`Chinese summary saved: ${data.output_zh}`);
    }
    if (data.detailed_output_zh) {
      addSystem(`Chinese detailed review saved: ${data.detailed_output_zh}`);
    }
    if (data.source) {
      addSystem(`Summary source: ${data.source}`);
    }
    if (data.ollama_model || data.final_ollama_model) {
      addSystem(`Models: chunks=${data.ollama_model || "unknown"}, final=${data.final_ollama_model || "unknown"}`);
    }
    statusEl.textContent = "summary ready";
  } catch (error) {
    addSystem(`Summary failed: ${error.message || error}`, true);
    statusEl.textContent = "summary failed";
  } finally {
    summaryBtn.disabled = false;
  }
}

function startRecorderSegment() {
  if (!recordingActive || !recorderStream) {
    return;
  }

  const chunks = [];
  mediaRecorder = new MediaRecorder(recorderStream, {mimeType: recorderMimeType});
  mediaRecorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) {
      chunks.push(event.data);
    }
  };
  mediaRecorder.onstop = () => {
    segmentTimer = null;
    if (chunks.length) {
      const blob = new Blob(chunks, {type: recorderMimeType});
      sendChunk(blob);
    }
    if (recordingActive && recorderStream) {
      setTimeout(startRecorderSegment, 150);
    }
  };
  mediaRecorder.start();
  segmentTimer = setTimeout(() => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.stop();
    }
  }, recorderChunkMs);
}

async function sendChunk(blob) {
  activeUploads += 1;
  const form = new FormData();
  form.append("audio", blob, `chunk_${chunkIndex++}.webm`);
  form.append("session_id", sessionId);
  form.append("asr_backend", asrBackendInput.value);
  form.append("whisper_model", document.querySelector("#whisperModel").value);
  form.append("mlx_model", mlxModelInput.value);
  form.append("device", document.querySelector("#device").value);
  form.append("compute_type", document.querySelector("#computeType").value);
  form.append("translate", document.querySelector("#translate").checked ? "true" : "false");
  form.append("ollama_model", ollamaModelInput.value);
  form.append("ollama_host", "http://localhost:11434");

  try {
    statusEl.textContent = "transcribing";
    const res = await fetch("/api/transcribe_chunk", {method: "POST", body: form});
    const data = await res.json();
    if (!data.ok) {
      addSystem(data.error || "Chunk transcription failed.", true);
      return;
    }
    if (data.english || data.chinese) {
      addCaption(data);
      statusEl.textContent = `recording (${data.asr_backend || "asr"}, ${data.device})`;
    } else {
      statusEl.textContent = "recording (silence)";
    }
  } catch (error) {
    addSystem(`Cannot transcribe chunk: ${error.message || error}`, true);
    statusEl.textContent = "server disconnected";
  } finally {
    activeUploads = Math.max(0, activeUploads - 1);
  }
}

async function pollEvents() {
  try {
    const res = await fetch(`/api/events?since=${latestEventId}`, {cache: "no-store"});
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const data = await res.json();
    latestEventId = data.latest || latestEventId;
    for (const event of data.events || []) {
    if (event.kind === "caption") {
      addCaption(event);
    } else if (event.kind === "polished") {
      if (event.polished) {
        addPolishedTranslation(event.polished);
      }
      if (event.transcript) {
        savedPathEl.textContent = event.transcript;
      }
    } else if (event.kind === "error") {
      addSystem(event.message, true);
      startBtn.disabled = false;
      stopBtn.disabled = true;
      statusEl.textContent = "error";
    } else if (event.kind === "status") {
      statusEl.textContent = event.message || "status";
      addSystem(event.message || "status");
      if ((event.message || "").startsWith("Stopped.")) {
        startBtn.disabled = false;
        stopBtn.disabled = true;
      }
    }
    }
  } catch (error) {
    statusEl.textContent = "server disconnected";
  } finally {
    setTimeout(pollEvents, 1000);
  }
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);
preloadBtn.addEventListener("click", preloadWhisper);
refineBtn.addEventListener("click", refineTranscript);
summaryBtn.addEventListener("click", generateSummary);
refreshSourcesBtn.addEventListener("click", loadSources);
refreshSourcesBtn.addEventListener("click", loadBrowserInputs);
textSizeInput.addEventListener("input", applyTextSize);
applyTextSize();
setPlaceholder(liveTranscriptEl, "Live English will appear here.");
setPlaceholder(englishTranscriptEl, "Repaired English will appear here.");
setPlaceholder(chineseTranscriptEl, "修复后的中文会显示在这里。");
loadSources();
loadBrowserInputs();
addSystem("connected");
pollEvents();
