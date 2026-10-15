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
let captureStream = null;
let recorderStream = null;
let micStream = null;
let audioContext = null;
let sessionId = null;
let chunkIndex = 0;
let recordingActive = false;
let captureNode = null;
let captureSource = null;
let captureParts = [];
let captureSampleCount = 0;
let samplesPerChunk = 0;
let captureConfig = null;
let captureFlushed = null;
let finalQueued = false;
let uploadQueue = [];
let uploading = false;
let uploadFailure = null;
let lastCaptionIndex = -1;
let draftCaptionEl = null;
let liveChunkCount = 0;
let completeParagraphCount = 0;
let activeEnglishParagraph = null;
let activeChineseParagraph = null;

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
  if (liveChunkCount === 0 && (event.english || event.draft)) {
    liveTranscriptEl.innerHTML = "";
  }
  if (event.draft !== undefined) {
    draftCaptionEl?.remove();
    draftCaptionEl = null;
  }
  for (const [text, draft] of [[event.english, false], [event.draft, true]]) {
    if (!text) continue;
    liveChunkCount += 1;
    const article = document.createElement("article");
    article.className = draft ? "entry draft" : "entry";
    article.innerHTML = `
      <div class="meta">${draft ? "Draft" : escapeHtml(event.timestamp || event.time || "")}</div>
      <div class="english">${escapeHtml(text)}</div>
    `;
    liveTranscriptEl.insertBefore(article, draftCaptionEl);
    if (draft) draftCaptionEl = article;
  }
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
  if (recordingActive || uploadQueue.length) return;
  liveChunkCount = 0;
  completeParagraphCount = 0;
  activeEnglishParagraph = null;
  activeChineseParagraph = null;
  draftCaptionEl = null;
  lastCaptionIndex = -1;
  setPlaceholder(liveTranscriptEl, "Waiting for live English...");
  setPlaceholder(englishTranscriptEl, "Waiting for repaired English...");
  setPlaceholder(chineseTranscriptEl, "等待更完整的中文段落...");
  systemLogEl.innerHTML = "";
  savedPathEl.textContent = "";
  if (!sourceSelect.value) {
    addSystem("Choose an audio source first. Click Refresh if the list is empty.", true);
    return;
  }
  startBtn.disabled = true;
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

    captureConfig = {
      session_id: sessionId,
      asr_backend: asrBackendInput.value,
      whisper_model: document.querySelector("#whisperModel").value,
      mlx_model: mlxModelInput.value,
      device: document.querySelector("#device").value,
      compute_type: document.querySelector("#computeType").value,
      translate: document.querySelector("#translate").checked ? "true" : "false",
      ollama_model: ollamaModelInput.value,
      ollama_host: "http://localhost:11434"
    };
    captureParts = [];
    captureSampleCount = 0;
    finalQueued = false;
    await startAudioCapture();
    recordingActive = true;
    for (const track of captureStream.getTracks()) {
      track.addEventListener("ended", () => { if (recordingActive) stop(); }, {once: true});
    }
    startBtn.disabled = true;
    stopBtn.disabled = false;
    statusEl.textContent = "recording";
    addSystem("Recording through browser capture. First transcript may take a little longer while Whisper loads.");
  } catch (error) {
    recordingActive = false;
    await releaseCapture();
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
  audioContext = new AudioContext({sampleRate: 16000});
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
  stopBtn.disabled = true;
  try {
    recordingActive = false;
    if (!finalQueued) {
      if (captureNode) {
        await new Promise((resolve, reject) => {
          const timeout = setTimeout(() => reject(new Error("Audio capture did not finish flushing.")), 5000);
          captureFlushed = () => { clearTimeout(timeout); resolve(); };
          captureNode.port.postMessage("flush");
        });
      }
      enqueueAudio(takeAudio(captureSampleCount), true);
      finalQueued = true;
      await releaseCapture();
    }
    await waitForPendingChunks();
    await flushTranslation();
    statusEl.textContent = "stopped";
    addSystem("Stopped. Final subtitles saved.");
    startBtn.disabled = false;
    if (autoSummaryInput.checked) {
      generateSummary(true);
    }
  } catch (error) {
    stopBtn.disabled = false;
    addSystem(`Finishing the session failed. Keep this tab open and click Stop again after the connection recovers. ${error.message || error}`, true);
  }
}

async function flushTranslation() {
  if (!sessionId) {
    return;
  }
  const response = await fetch("/api/flush_translation", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        session_id: sessionId,
        ollama_model: captureConfig?.ollama_model || ollamaModelInput.value,
        ollama_host: "http://localhost:11434",
        translate: captureConfig ? captureConfig.translate === "true" : document.querySelector("#translate").checked
      })
  });
  const data = await response.json();
  if (!data.ok) throw new Error(data.error || "Could not flush the final translation.");
}

async function waitForPendingChunks(timeoutMs = 90000) {
  const started = Date.now();
  while (uploadQueue.length && Date.now() - started < timeoutMs) {
    statusEl.textContent = `saving ${uploadQueue.length} audio chunk(s)`;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  if (uploadQueue.length) throw new Error(uploadFailure?.message || "Audio upload has not finished.");
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

async function startAudioCapture() {
  audioContext ||= new AudioContext({sampleRate: 16000});
  await audioContext.resume();
  const seconds = Number(document.querySelector("#chunkSeconds").value);
  if (!Number.isFinite(seconds) || seconds < 1 || seconds > 20) throw new Error("Subtitle refresh must be 1–20 seconds.");
  samplesPerChunk = Math.round(audioContext.sampleRate * seconds);
  await audioContext.audioWorklet.addModule("/static/audio-capture.js");
  captureSource = audioContext.createMediaStreamSource(recorderStream);
  captureNode = new AudioWorkletNode(audioContext, "caption-audio-capture", {outputChannelCount: [1]});
  captureNode.port.onmessage = ({data}) => {
    if (data.type === "flushed") {
      captureFlushed?.();
      captureFlushed = null;
      return;
    }
    captureParts.push(data.samples);
    captureSampleCount += data.samples.length;
    while (recordingActive && captureSampleCount >= samplesPerChunk) {
      enqueueAudio(takeAudio(samplesPerChunk));
    }
  };
  captureSource.connect(captureNode).connect(audioContext.destination);
}

async function releaseCapture() {
  captureNode?.disconnect();
  captureSource?.disconnect();
  for (const stream of [captureStream, micStream, recorderStream]) {
    for (const track of stream?.getTracks() || []) track.stop();
  }
  if (audioContext) await audioContext.close();
  captureNode = captureSource = captureStream = micStream = recorderStream = audioContext = null;
}

function takeAudio(count) {
  const audio = new Float32Array(count);
  let offset = 0;
  while (offset < count) {
    const part = captureParts[0];
    const length = Math.min(part.length, count - offset);
    audio.set(part.subarray(0, length), offset);
    if (length === part.length) captureParts.shift();
    else captureParts[0] = part.subarray(length);
    offset += length;
  }
  captureSampleCount -= count;
  return audio;
}

function audioWav(samples, rate) {
  const bytes = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(bytes);
  for (const [offset, text] of [[0, "RIFF"], [8, "WAVE"], [12, "fmt "], [36, "data"]]) {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  }
  for (const [offset, value] of [[4, bytes.byteLength - 8], [16, 16], [24, rate], [28, rate * 2], [40, samples.length * 2]]) {
    view.setUint32(offset, value, true);
  }
  for (const [offset, value] of [[20, 1], [22, 1], [32, 2], [34, 16]]) view.setUint16(offset, value, true);
  samples.forEach((sample, i) => {
    const value = Math.max(-1, Math.min(1, sample));
    view.setInt16(44 + i * 2, value * (value < 0 ? 32768 : 32767), true);
  });
  return new Blob([bytes], {type: "audio/wav"});
}

function enqueueAudio(samples, final = false) {
  // ponytail: unsent audio stays in this tab; use IndexedDB if recording through long outages is needed.
  uploadQueue.push({blob: samples.length ? audioWav(samples, audioContext.sampleRate) : null, index: chunkIndex++, final});
  drainAudioQueue();
}

async function drainAudioQueue() {
  if (uploading) return;
  uploading = true;
  try {
    while (uploadQueue.length) {
      try {
        await sendChunk(uploadQueue[0]);
        uploadQueue.shift();
        if (uploadFailure) addSystem("Audio upload recovered.");
        uploadFailure = null;
      } catch (error) {
        if (!uploadFailure) addSystem(`Audio upload paused; keeping audio in this tab and retrying. ${error.message || error}`, true);
        uploadFailure = error;
        setTimeout(drainAudioQueue, 2000);
        break;
      }
    }
  } finally {
    uploading = false;
  }
}

async function sendChunk(item) {
  const form = new FormData();
  if (item.blob) form.append("audio", item.blob, `audio_${item.index}.wav`);
  for (const [key, value] of Object.entries(captureConfig)) form.append(key, value);
  form.append("chunk_index", item.index);
  form.append("final", String(item.final));
  const res = await fetch("/api/transcribe_stream", {method: "POST", body: form, signal: AbortSignal.timeout(120000)});
  const data = await res.json();
  if (!data.ok) throw new Error(data.error || "Streaming transcription failed.");
  if (data.session_id === sessionId && data.chunk_index > lastCaptionIndex) {
    lastCaptionIndex = data.chunk_index;
    addCaption(data);
    if (recordingActive) statusEl.textContent = `recording (${data.asr_backend || "asr"}, ${data.device})`;
  }
}

async function pollEvents() {
  try {
    const res = await fetch(`/api/events?since=${latestEventId}`, {cache: "no-store"});
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const data = await res.json();
    latestEventId = data.latest ?? latestEventId;
    for (const event of data.events || []) {
    if (event.session_id && event.session_id !== sessionId) continue;
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
window.addEventListener("beforeunload", event => {
  if (recordingActive || uploadQueue.length || captureSampleCount) {
    event.preventDefault();
    event.returnValue = "";
  }
});
