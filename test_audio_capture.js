// Run with: node test_audio_capture.js
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

async function main() {
  let Processor;
  const packets = [];
  const worklet = vm.createContext({
    AudioWorkletProcessor: class { constructor() { this.port = {postMessage: (data, transfer) => packets.push(structuredClone(data, {transfer}))}; } },
    registerProcessor: (_, processor) => { Processor = processor; }
  });
  vm.runInContext(fs.readFileSync("static/audio-capture.js", "utf8"), worklet);
  const processor = new Processor();
  const expected = [];
  for (let offset = 0; offset < 20003; offset += 128) {
    const left = Float32Array.from({length: Math.min(128, 20003 - offset)}, (_, i) => Math.sin((offset + i) / 31) * 0.7);
    const right = Float32Array.from(left, x => x * 0.5);
    expected.push(...left.map((x, i) => (x + right[i]) / 2));
    assert.equal(processor.process([[left, right]]), true);
  }
  processor.port.onmessage({data: "flush"});
  assert.equal(packets.at(-1).type, "flushed");
  const actual = packets.filter(p => p.type === "audio").flatMap(p => Array.from(p.samples));
  assert.deepEqual(actual, expected); // Every sample, including the final partial packet, is captured once.
  processor.port.onmessage({data: "flush"});
  assert.equal(packets.filter(p => p.type === "audio").reduce((sum,p) => sum+p.samples.length,0), expected.length);
  assert.equal(processor.process([[]]), false);

  const live = {children: [], set innerHTML(value) { this.children = []; },
    insertBefore(node, before) { this.children.splice(before ? this.children.indexOf(before) : this.children.length, 0, node); }};
  const app = vm.createContext({
    document: {querySelector: selector => selector === "#liveTranscript" ? live : {},
      createElement: () => ({remove() { live.children.splice(live.children.indexOf(this), 1); }})},
    Blob, FormData, AbortSignal, setTimeout, clearTimeout,
    Float32Array, ArrayBuffer, DataView, console
  });
  vm.runInContext(fs.readFileSync("static/app.js", "utf8").split('startBtn.addEventListener("click", start);')[0], app);
  const taken = vm.runInContext(`
    captureParts = [new Float32Array([1,2,3]), new Float32Array([4,5])];
    captureSampleCount = 5;
    [...takeAudio(4), ...takeAudio(1)];
  `, app);
  assert.deepEqual(Array.from(taken), [1,2,3,4,5]);
  assert.equal(vm.runInContext("captureSampleCount", app), 0);
  const wav = await vm.runInContext("audioWav(new Float32Array([-1,0,1]), 48000).arrayBuffer()", app);
  const view = new DataView(wav);
  assert.equal(view.getUint32(24, true), 48000); // Use the actual AudioContext rate, not an assumed rate.
  assert.equal(view.getUint32(40, true), 6);
  assert.deepEqual([44,46,48].map(i => view.getInt16(i, true)), [-32768,0,32767]);
  vm.runInContext(`
    addCaption({english: "", draft: "configure"});
    addCaption({english: "I need", draft: "configurable"});
  `, app);
  assert.equal(live.children.length, 2);
  assert.ok(live.children[1].innerHTML.includes("configurable"));
  vm.runInContext('addCaption({english: "configurable", draft: ""})', app);
  assert.equal(live.children.length, 2);
  assert.ok(live.children.every(node => node.className === "entry"));

  const attempts = [];
  app.fetch = async (_, options) => {
    const index = Number(options.body.get("chunk_index"));
    attempts.push(index);
    if (attempts.length === 1) throw new Error("temporary connection failure");
    return {json: async () => ({ok: true, session_id: "test", chunk_index: index, english: "", draft: ""})};
  };
  await vm.runInContext(`
    addSystem = () => {}; addCaption = () => {};
    captureConfig = {session_id: "test"}; sessionId = "test"; audioContext = {sampleRate: 16000};
    uploadQueue = [{blob: null, index: 0, final: false}, {blob: null, index: 1, final: true}];
    drainAudioQueue();
  `, app);
  assert.equal(vm.runInContext("uploadQueue.length", app), 2);
  await vm.runInContext("drainAudioQueue()", app);
  assert.deepEqual(attempts, [0,0,1]);
  assert.equal(vm.runInContext("uploadQueue.length", app), 0);

  let capture;
  app.AudioWorkletNode = class {
    constructor() {
      capture = new Processor();
      this.port = {onmessage: null, postMessage: data => capture.port.onmessage({data})};
      capture.port.postMessage = (data, transfer) => {
        const copy = structuredClone(data, {transfer});
        queueMicrotask(() => this.port.onmessage({data: copy}));
      };
    }
    connect() { return this; }
    disconnect() {}
  };
  const uploads = [];
  const finishEvents = [];
  app.finishEvents = finishEvents;
  app.fetch = async (_, options) => {
    const form = options.body;
    const wav = new DataView(await form.get("audio").arrayBuffer());
    uploads.push({index: Number(form.get("chunk_index")), final: form.get("final"), samples: wav.getUint32(40,true)/2});
    if (form.get("final") === "true") finishEvents.push("final-saved");
    return {json: async () => ({ok: true, session_id: "test", chunk_index: uploads.at(-1).index, english: "", draft: ""})};
  };
  vm.runInContext(`
    document.querySelector = () => ({value: "3"});
    audioContext = {sampleRate: 48000, resume: async () => {}, close: async () => {},
      audioWorklet: {addModule: async () => {}},
      createMediaStreamSource: () => ({connect: node => node, disconnect() {}})};
    captureStream = recorderStream = {getTracks: () => []};
    captureParts = []; captureSampleCount = 0; chunkIndex = 0; finalQueued = false;
    recordingActive = true; autoSummaryInput.checked = true;
    flushTranslation = async () => { if (uploadQueue.length) throw new Error("Audio still pending"); finishEvents.push("translation"); };
    generateSummary = () => finishEvents.push("summary");
  `, app);
  await vm.runInContext("startAudioCapture()", app);
  for (let offset = 0; offset < 360197; offset += 128) {
    capture.process([[new Float32Array(Math.min(128,360197-offset))]]);
  }
  await new Promise(resolve => queueMicrotask(resolve));
  await vm.runInContext("stop()", app);
  assert.deepEqual(uploads.map(u => u.index), [0,1,2]);
  assert.deepEqual(uploads.map(u => u.final), ["false","false","true"]);
  assert.equal(uploads.reduce((sum,u) => sum + u.samples, 0), 360197);
  assert.equal(vm.runInContext("uploadQueue.length", app), 0);
  assert.deepEqual(finishEvents, ["final-saved", "translation", "summary"]);
  console.log("PASS: continuous samples, draft replacement, ordered retries, and Start-to-Stop audio flush without lost or repeated samples.");
}

main().catch(error => { console.error(error); process.exitCode = 1; });
