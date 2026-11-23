import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { chromium } from 'playwright-core';

// Explicit opt-in local media. Never bundle, upload, or commit a person's video.
const video = process.env.GS_TEST_VIDEO;
if (!video) throw new Error('Set GS_TEST_VIDEO to a local orbit video. Start npm run preview first.');
const base = process.env.GS_VERIFY_URL || 'http://127.0.0.1:4173';
const output = resolve('artifacts/video-regression');
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ channel: process.env.GS_BROWSER_CHANNEL || 'msedge', headless: true });
const report = [];
try {
  for (const detail of [false, true]) {
    const page = await browser.newPage({ viewport: { width: 390, height: 844 }, isMobile: true, locale: 'zh-CN' });
    const errors = [], uploads = [];
    page.on('pageerror', e => errors.push(e.message));
    page.on('request', r => { if (!['GET', 'HEAD'].includes(r.method())) uploads.push(r.url()); });
    await page.route('https://fonts.googleapis.com/**', r => r.abort());
    await page.goto(base, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => document.querySelector('#ui-video-capture')?.textContent?.length > 0);
    assert.equal(await page.locator('#input-capture-video').getAttribute('capture'), null);
    assert.equal(await page.locator('#input-record-video').getAttribute('capture'), 'environment');
    assert.equal(await page.locator('#input-round-bottle').isVisible(), true);
    await page.locator('#input-video-detail').setChecked(detail);
    const start = Date.now();
    await page.locator('#input-capture-video').setInputFiles(video);
    await page.waitForFunction(() => document.querySelector('#busy-overlay').classList.contains('hidden'), {}, { timeout: 300000 });
    const count = Number((await page.locator('#badge-capture-count').textContent()).split('/')[0]);
    assert.ok(count >= (detail ? 12 : 6) && count <= (detail ? 24 : 8), `selected ${count}`);
    assert.equal(await page.locator('#render-stage.active').isVisible(), true);
    await page.locator('#surface-canvas').waitFor({ state: 'visible' });
    await page.locator('#btn-turntable').click(); // stop automatic motion for reproducible inspection
    await page.locator('#btn-reset-cam').click();
    const sizes = [], sourceViews = new Set();
    const viewport = page.locator('.splat-viewport-wrap');
    await viewport.scrollIntoViewIfNeeded();
    const box = await viewport.boundingBox();
    for (let angle = 0; angle < 12; angle++) {
      const source = await page.locator('#surface-canvas').evaluate(c => ({ view: c.dataset.sourceView, time: c.dataset.sourceTime, size: c.dataset.textureSize }));
      assert.ok(source.view !== undefined && source.size);
      sourceViews.add(source.view); sizes.push(source.size);
      await viewport.screenshot({ path: `${output}/${detail ? 'detail' : 'quick'}-${String(angle).padStart(2, '0')}.png` });
      await page.mouse.move(box.x + box.width * .35, box.y + box.height * .6);
      await page.mouse.down();
      await page.mouse.move(box.x + box.width * .35 - 30 / .42, box.y + box.height * .6, { steps: 4 });
      await page.mouse.up();
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    }
    assert.ok(sourceViews.size >= 6, 'rotation selects multiple complete source images');
    if (detail) assert.ok(sizes.some(size => Number(size.split('×')[1]) > 384), 'original-resolution texture survives');
    assert.equal(await page.locator('#slider-scale').isDisabled(), true);
    await page.locator('#ui-mode-2').click();
    await page.locator('#surface-canvas').waitFor({ state: 'hidden' });
    assert.equal(await page.locator('#slider-scale').isDisabled(), false);
    await page.locator('#ui-mode-0').click();
    await page.locator('#surface-canvas').waitFor({ state: 'visible' });

    // Resource restoration must bring the source textures back after a WebGL loss.
    const restoration = await page.locator('#surface-canvas').evaluate(c => {
      const extension = c.getContext('webgl2').getExtension('WEBGL_lose_context');
      if (!extension) return false;
      extension.loseContext(); setTimeout(() => extension.restoreContext(), 150); return true;
    });
    if (restoration) {
      await page.waitForFunction(() => document.querySelector('#surface-canvas').style.display === 'block' && !document.querySelector('#surface-canvas').getContext('webgl2').isContextLost());
    }
    await page.locator('#tab-capture').click();
    assert.equal(await page.locator('#capture-filmstrip img').count(), count);
    assert.equal(await page.locator('.preset-card.selected').getAttribute('data-preset'), detail ? 'video_detail' : 'video');
    await page.locator('#input-round-bottle').uncheck();
    await page.locator('#tab-render').click();
    assert.equal(await page.locator('#surface-canvas').isVisible(), false);
    assert.equal(await page.locator('#surface-canvas').getAttribute('data-source-view'), null);
    assert.deepEqual(errors, []); assert.deepEqual(uploads, []);
    const result = { detail, count, elapsedMs: Date.now() - start, textureSizes: [...new Set(sizes)], sourceViews: sourceViews.size };
    report.push(result); console.log('PASS', JSON.stringify(result));
    await page.close();
  }
  // Force the common no-WebGL2 path; it must stay usable and identify its limits.
  const fallback = await browser.newPage({ viewport: { width: 390, height: 844 }, isMobile: true, locale: 'en-US' });
  const fallbackErrors = [];
  fallback.on('pageerror', e => fallbackErrors.push(e.message));
  await fallback.addInitScript(() => {
    const getContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function(type, ...args) {
      return type === 'webgl2' ? null : getContext.call(this, type, ...args);
    };
  });
  await fallback.route('https://fonts.googleapis.com/**', r => r.abort());
  await fallback.goto(base, { waitUntil: 'domcontentloaded' });
  await fallback.waitForFunction(() => document.querySelector('#ui-video-capture')?.textContent?.length > 0);
  await fallback.locator('#input-video-detail').uncheck();
  await fallback.locator('#input-capture-video').setInputFiles(video);
  await fallback.waitForFunction(() => document.querySelector('#busy-overlay').classList.contains('hidden'), {}, { timeout: 300000 });
  await fallback.waitForFunction(() => /fallback/i.test(document.querySelector('#viewer-method').textContent));
  assert.equal(await fallback.locator('#surface-canvas').isVisible(), false);
  assert.equal(await fallback.locator('#slider-scale').isDisabled(), false);
  assert.equal(await fallback.locator('#render-stage.active').isVisible(), true);
  assert.deepEqual(fallbackErrors, []);
  console.log('PASS unavailable WebGL2 uses an explicitly labelled point fallback');
  await fallback.close();
  await writeFile(`${output}/report.json`, JSON.stringify(report, null, 2));
} finally { await browser.close(); }
