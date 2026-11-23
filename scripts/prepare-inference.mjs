import { copyFile, mkdir } from 'node:fs/promises';

const target = new URL('../public/inference/', import.meta.url);
await mkdir(target, { recursive: true });
for (const name of ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
  await copyFile(new URL('../node_modules/onnxruntime-web/dist/' + name, import.meta.url), new URL(name, target));
}
await copyFile(new URL('../node_modules/onnxruntime-web/README.md', import.meta.url), new URL('README.md', target));
await copyFile(new URL('../public/models/ONNXRUNTIME_LICENSE.txt', import.meta.url), new URL('LICENSE.txt', target));
