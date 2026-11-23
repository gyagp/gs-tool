import type { InferenceSession } from 'onnxruntime-web';
import { foregroundFromMask, type ForegroundResult } from './foreground';

let sessionPromise: Promise<InferenceSession> | undefined;

/** U²-Net small, executed locally with a self-hosted WASM runtime and model. */
async function getSession() {
  if (!sessionPromise) sessionPromise = (async () => {
    const ort = await import('onnxruntime-web/wasm');
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = '/inference/';
    return ort.InferenceSession.create('/models/u2netp.onnx', { executionProviders: ['wasm'] });
  })().catch(error => { sessionPromise = undefined; throw error; });
  return sessionPromise;
}

export async function segmentSubject(rgba: Uint8ClampedArray, width: number, height: number): Promise<ForegroundResult> {
  const ort = await import('onnxruntime-web/wasm');
  const session = await getSession();
  const size = 320, plane = size * size;
  const input = new Float32Array(3 * plane);
  const mean = [.485, .456, .406], deviation = [.229, .224, .225];
  // Bilinear sampling matches the model's training preprocessing and preserves small caps.
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const sx = Math.max(0, Math.min(width - 1, (x + .5) * width / size - .5));
    const sy = Math.max(0, Math.min(height - 1, (y + .5) * height / size - .5));
    const x0 = Math.floor(sx), y0 = Math.floor(sy), x1 = Math.min(width - 1, x0 + 1), y1 = Math.min(height - 1, y0 + 1);
    const fx = sx - x0, fy = sy - y0;
    for (let c = 0; c < 3; c++) {
      const top = rgba[(y0 * width + x0) * 4 + c] * (1 - fx) + rgba[(y0 * width + x1) * 4 + c] * fx;
      const bottom = rgba[(y1 * width + x0) * 4 + c] * (1 - fx) + rgba[(y1 * width + x1) * 4 + c] * fx;
      input[c * plane + y * size + x] = ((top * (1 - fy) + bottom * fy) / 255 - mean[c]) / deviation[c];
    }
  }
  const result = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, size, size]) });
  try {
    const prediction = result[session.outputNames[0]].data as Float32Array;
    let min = Infinity, max = -Infinity;
    for (const p of prediction) { min = Math.min(min, p); max = Math.max(max, p); }
    const mask = new Uint8Array(width * height);
    if (max - min > .01) for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const sx = Math.min(size - 1, Math.floor((x + .5) * size / width));
      const sy = Math.min(size - 1, Math.floor((y + .5) * size / height));
      mask[y * width + x] = prediction[sy * size + sx] > min + (max - min) * .5 ? 255 : 0;
    }
    return foregroundFromMask(mask, width, height);
  } finally {
    for (const tensor of Object.values(result)) tensor.dispose();
  }
}
