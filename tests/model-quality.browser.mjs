// Calibrated source-view validation of the actual web renderer, without uploading
// a person's model or putting private media in a public static directory.
import assert from 'node:assert/strict';
import { readFile, mkdir } from 'node:fs/promises';
import { resolve, extname } from 'node:path';
import { createServer } from 'node:http';
import { build } from 'vite';
import { chromium } from 'playwright-core';

const root = resolve(process.env.GS_RECONSTRUCTION || 'artifacts/reconstruction');
const model = process.env.GS_TEST_MODEL;
if (!model) throw new Error('Set GS_TEST_MODEL to a local trained PLY.');
const assets = resolve('artifacts/model-quality-app');
const output = resolve(root, 'browser-validation');
await mkdir(output, { recursive: true });
await build({ configFile: false, logLevel: 'error', build: { outDir: assets, emptyOutDir: true,
  rollupOptions: { input: resolve('tests/model-quality.html') } } });
const server = createServer(async (req, res) => {
  const path = resolve(assets, '.' + new URL(req.url, 'http://localhost').pathname);
  if (!path.startsWith(assets + '/'.replace('/', process.platform === 'win32' ? '\\' : '/'))) { res.writeHead(403).end(); return; }
  try {
    const bytes = await readFile(path);
    res.setHeader('content-type', extname(path) === '.js' ? 'text/javascript' : 'text/html');
    res.end(bytes);
  } catch { res.writeHead(404).end(); }
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const browser = await chromium.launch({ channel: 'msedge', headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 481, height: 729 } });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  // Fulfill in the test driver; the model never reaches any HTTP server.
  await page.route('**/private-test-model.ply', r => r.fulfill({ contentType: 'application/octet-stream', path: resolve(model) }));
  await page.goto(`http://127.0.0.1:${server.address().port}/tests/model-quality.html`);
  await page.evaluate(() => window.loadModel());
  const cameras = JSON.parse(await readFile(resolve(root, 'viewer-cameras.json'), 'utf8'));
  for (const index of [0, 16, 40, 64, 88, 112].filter(i => i < cameras.length)) {
    const camera = cameras[index];
    const k = camera.intrinsics;
    assert.ok(Math.abs(k[0][0] - k[1][1]) < 1e-6 && Math.abs(k[0][2] - camera.width / 2) < 1e-6 && Math.abs(k[1][2] - camera.height / 2) < 1e-6, 'symmetric calibration required by orbit camera');
    await page.setViewportSize({ width: camera.width, height: camera.height });
    await page.evaluate(p => { window.pose = p; }, camera);
    await page.waitForTimeout(1000);
    await page.locator('#imported-canvas').screenshot({ path: resolve(output, camera.name) });
  }
  assert.deepEqual(errors, []);
  console.log('PASS calibrated web renders saved to', output);
} finally { await browser.close(); await new Promise(r => server.close(r)); }
