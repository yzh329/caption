class CaptionAudioCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.samples = new Float32Array(1024);
    this.used = 0;
    this.closed = false;
    this.port.onmessage = ({data}) => {
      if (data === "flush") {
        if (this.used) {
          const samples = this.samples.slice(0, this.used);
          this.port.postMessage({type: "audio", samples}, [samples.buffer]);
        }
        this.used = 0;
        this.closed = true;
        this.port.postMessage({type: "flushed"});
      }
    };
  }

  process(inputs) {
    if (this.closed) return false;
    const channels = inputs[0];
    if (!channels.length) return true;
    for (let i = 0; i < channels[0].length; i++) {
      let sample = 0;
      for (const channel of channels) sample += channel[i] / channels.length;
      this.samples[this.used++] = sample;
      if (this.used === this.samples.length) {
        this.port.postMessage({type: "audio", samples: this.samples}, [this.samples.buffer]);
        this.samples = new Float32Array(1024);
        this.used = 0;
      }
    }
    return true; // Output stays silent; capturing the microphone must not play it through the speakers.
  }
}

registerProcessor("caption-audio-capture", CaptionAudioCapture);
