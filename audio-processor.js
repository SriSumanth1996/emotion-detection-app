// static/audio-processor.js
class AudioProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input.length > 0) {
      const audioData = input[0];
      const int16Array = new Int16Array(audioData.length);
      
      // Convert Float32 to Int16
      for (let i = 0; i < audioData.length; i++) {
        int16Array[i] = Math.max(-32768, Math.min(32767, audioData[i] * 32767));
      }
      
      this.port.postMessage({
        type: 'audio_chunk',
        data: int16Array.buffer
      }, [int16Array.buffer]);
    }
    return true;
  }
}
registerProcessor('audio-processor', AudioProcessor);
