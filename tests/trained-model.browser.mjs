import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { chromium } from 'playwright-core';

const model = process.env.GS_TEST_MODEL;
if (!model) throw new Error('Set GS_TEST_MODEL to a local trained PLY (never uploaded).');
const base = process.env.GS_VERIFY_URL || 'http://127.0.0.1:4173';
const output = resolve('artifacts/trained-model-regression');
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ channel: 'msedge', headless: true });
const errors = [], uploads = [], requests = [];
try {
  const page = await browser.newPage({ viewport: { width: 390, height: 844 }, isMobile: true, locale: 'zh-CN', deviceScaleFactor: 2 });
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'warning' || m.type() === 'error') console.log(m.type(), m.text().slice(0,2000)); });
  page.on('dialog', async d => { console.log('dialog', d.message()); await d.dismiss(); });
  page.on('request', r => {
    requests.push(r.url());
    if (!['GET', 'HEAD'].includes(r.method())) uploads.push(r.url());
  });
  await page.route('https://fonts.googleapis.com/**', r => r.abort());
  await page.goto(base, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => document.querySelector('#ui-import-btn')?.textContent?.length > 0);
  assert.ok(!requests.some(r => r.includes('importedRenderer')), 'renderer stays lazy before model import');
  await page.locator('#input-import-splat').setInputFiles(model);
  await page.locator('#imported-canvas').waitFor({ state: 'visible', timeout: 120000 });
  await page.waitForFunction(() => Number(document.querySelector('#imported-canvas').dataset.splats) > 0);
  await page.locator('#btn-turntable').click();
  await page.locator('#btn-reset-cam').click();
  const stats = await page.locator('#imported-canvas').evaluate(c => ({ ...c.dataset }));
  const header = (await readFile(model)).subarray(0, 16384).toString().split('end_header')[0];
  assert.equal(Number(stats.splats), Number(header.match(/element vertex (\d+)/)[1]), 'no point decimation');
  if (header.includes('f_rest_44')) assert.equal(Number(stats.sh), 3, 'all degree-three SH retained');
  assert.equal(await page.locator('#slider-scale').isDisabled(), true);
  assert.equal(await page.locator('#btn-export-splat').isDisabled(), true);
  assert.ok(['float16', 'rgba8'].includes(stats.compositing));
  const viewport = page.locator('.splat-viewport-wrap');
  await viewport.scrollIntoViewIfNeeded();
  const box = await viewport.boundingBox();
  for (let angle = 0; angle < 12; angle++) {
    // Let asynchronous GPU sorting settle before each visual comparison.
    await page.waitForTimeout(600);
    await viewport.screenshot({ path: `${output}/orbit-${String(angle).padStart(2, '0')}.png` });
    await page.mouse.move(box.x + box.width * .7, box.y + box.height * .6);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * .7 - 30 / .42, box.y + box.height * .6, { steps: 4 });
    await page.mouse.up();
  }
  const downloadEvent = page.waitForEvent('download');
  await page.locator('#btn-export-ply').click();
  const download = await downloadEvent;
  const hash = data => createHash('sha256').update(data).digest('hex');
  assert.equal(hash(await readFile(await download.path())), hash(await readFile(model)), 'lossless original PLY export');
  await page.locator('#ui-mode-2').click();
  await page.locator('#imported-canvas').waitFor({ state: 'hidden' });
  assert.equal(await page.locator('#slider-scale').isDisabled(), false);
  await page.locator('#ui-mode-0').click();
  await page.locator('#imported-canvas').waitFor({ state: 'visible' });
  const restoration = await page.locator('#imported-canvas').evaluate(c => {
    const extension = c.getContext('webgl2').getExtension('WEBGL_lose_context');
    if (!extension) return false;
    extension.loseContext(); setTimeout(() => extension.restoreContext(), 150); return true;
  });
  if (restoration) await page.waitForFunction(() => {
    const c = document.querySelector('#imported-canvas');
    return c?.style.display === 'block' && !c.dataset.error && !c.getContext('webgl2').isContextLost();
  }, {}, { timeout: 60000 });
  // A second load must replace GPU resources and preserve full data.
  const previousCanvas = await page.locator('#imported-canvas').elementHandle();
  await page.locator('#input-import-splat').setInputFiles(model);
  await page.waitForFunction(() => document.querySelector('#busy-overlay').classList.contains('hidden'));
  await page.waitForFunction(() => document.querySelectorAll('#imported-canvas').length === 1 && document.querySelector('#imported-canvas').style.display === 'block');
  assert.equal(await previousCanvas.evaluate(c => c.isConnected), false, 'previous model resources removed');
  assert.equal(await page.locator('#imported-canvas').getAttribute('data-splats'), stats.splats);
  assert.deepEqual(errors, []); assert.deepEqual(uploads, []);
  const report = { base, stats, errors, uploads, requests: [...new Set(requests)].filter(r => !r.startsWith('blob:')) };
  await writeFile(`${output}/report.json`, JSON.stringify(report, null, 2));
  console.log('PASS', JSON.stringify({ ...stats, losslessExport: true, uploads, errors }));

  const fallback = await browser.newPage({ viewport: { width: 390, height: 844 }, isMobile: true, locale: 'en-US' });
  fallback.on('dialog', d => d.dismiss());
  await fallback.addInitScript(() => {
    const getContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function(type, ...args) {
      return type === 'webgl2' ? null : getContext.call(this, type, ...args);
    };
  });
  await fallback.route('https://fonts.googleapis.com/**', r => r.abort());
  await fallback.goto(base, { waitUntil: 'domcontentloaded' });
  await fallback.waitForFunction(() => document.querySelector('#ui-import-btn')?.textContent?.length > 0);
  await fallback.locator('#input-import-splat').setInputFiles(model);
  await fallback.waitForFunction(() => /simplified/i.test(document.querySelector('#viewer-method').textContent));
  assert.equal(await fallback.locator('#imported-canvas').count(), 0);
  console.log('PASS no WebGL2 shows an explicit simplified preview');
  await fallback.close();
} finally { await browser.close(); }
