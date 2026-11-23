import { defineConfig } from 'vite';

export default defineConfig({
  resolve: {
    conditions: ['onnxruntime-web-use-extern-wasm', 'module', 'browser', 'development|production']
  },
  server: {
    watch: { ignored: ['**/artifacts/**'] }
  },
  build: {
    target: 'es2022',
    sourcemap: false,
    minify: 'esbuild'
  },
  worker: {
    format: 'es'
  }
});
