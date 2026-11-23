import assert from 'node:assert/strict';
import test from 'node:test';
import { build } from 'vite';

// Reuse Vite's TS compilation; no additional test/runtime dependencies are needed.
const bundle = await build({
  configFile: false,
  logLevel: 'silent',
  build: { ssr: 'tests/capture-api.ts', write: false, minify: false, target: 'node22' }
});
const api = await import(`data:text/javascript;base64,${Buffer.from(bundle.output[0].code).toString('base64')}`);
const { extractForeground, getCaptureDimensions, CAPTURE_PRESETS, buildSlotsForPreset,
  evaluateCoverage, reconstructSplatsFromCaptures, reconstructBottleFromCaptures, ReconstructionError, selectDistinctFrame } = api;

function bottlePixels() {
  const width = 160, height = 288;
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let color = y < 225 ? [45, 60, 90] : [165, 130, 100];
      const radius = y < 28 || y > 260 ? 0 : y <= 65 ? 10 : y < 85 ? 10 + (y - 65) * 0.85 : 27;
      if (radius && Math.abs(x - 80) <= radius) {
        color = y <= 45 ? [210, 42, 36] : [20, 170, 75];
        if (y >= 120 && y <= 190) color = [245, 245, 242];
      }
      if (x >= 8 && x <= 10 && y >= 80 && y <= 82) color = [210, 42, 36];
      rgba.set([...color, 255], (y * width + x) * 4);
    }
  }
  return { width, height, rgba };
}

function capturedBottleSlots() {
  const pixels = bottlePixels();
  const foreground = extractForeground(pixels.rgba, pixels.width, pixels.height);
  return buildSlotsForPreset(CAPTURE_PRESETS.minimal).map(slot => ({
    ...slot, captured: true, photo: {
      source: 'real', slotIndex: slot.index, azimuthDeg: slot.azimuthDeg, elevationDeg: slot.elevationDeg,
      workWidth: pixels.width, workHeight: pixels.height, rgba: pixels.rgba, foregroundMask: foreground.mask,
      verdict: 'good', sharpnessScore: 90, timestamp: slot.index, dataUrl: `original-${slot.index}`
    }
  }));
}

test('portrait and landscape resizing keep the full frame and never force a square', () => {
  assert.deepEqual(getCaptureDimensions(900, 1600, 1280), { width: 720, height: 1280 });
  assert.deepEqual(getCaptureDimensions(900, 1600, 384), { width: 216, height: 384 });
  assert.deepEqual(getCaptureDimensions(1600, 900, 384), { width: 384, height: 216 });
  assert.deepEqual(getCaptureDimensions(100, 200, 384), { width: 100, height: 200 });
});

test('cutout keeps the cap, base and pale label, and removes wall, table and isolated noise', () => {
  const { rgba, width, height } = bottlePixels();
  const foreground = extractForeground(rgba, width, height);
  assert.equal(foreground.usable, true);
  for (const [x, y] of [[80, 30], [80, 150], [80, 259]]) assert.equal(foreground.mask[y * width + x], 255);
  for (const [x, y] of [[0, 0], [20, 100], [20, 250], [9, 81]]) assert.equal(foreground.mask[y * width + x], 0);
  assert.ok(foreground.bounds.height > foreground.bounds.width * 3);
});

test('blank photos do not receive an invented ellipse or substitute sphere', async () => {
  const rgba = new Uint8ClampedArray(64 * 64 * 4).fill(255);
  const foreground = extractForeground(rgba, 64, 64);
  assert.equal(foreground.usable, false);
  assert.equal(foreground.mask.some(Boolean), false);
  const slots = capturedBottleSlots().map(slot => ({ ...slot, photo: { ...slot.photo, rgba, workWidth: 64, workHeight: 64 } }));
  await assert.rejects(reconstructSplatsFromCaptures(slots, 48, 'standard'), error => error instanceof ReconstructionError && error.code === 'foreground');
});

test('a thin blended backdrop seam does not attach the bottle to the frame edges', () => {
  const { rgba, width, height } = bottlePixels();
  for (let x = 0; x < width; x++) {
    if (Math.abs(x - 80) > 27) rgba.set([105, 95, 95, 255], (224 * width + x) * 4);
  }
  const foreground = extractForeground(rgba, width, height);
  assert.equal(foreground.usable, true);
  assert.equal(foreground.mask[224 * width + 20], 0);
  assert.equal(foreground.mask[224 * width + 80], 255);
});

test('eight bottle silhouettes produce an elongated surface inside the volume, not a solid cube', async () => {
  const cloud = await reconstructSplatsFromCaptures(capturedBottleSlots(), 48, 'standard');
  assert.ok(cloud.count > 100);
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < cloud.positions.length; i++) {
    assert.ok(Number.isFinite(cloud.positions[i]));
    min[i % 3] = Math.min(min[i % 3], cloud.positions[i]);
    max[i % 3] = Math.max(max[i % 3], cloud.positions[i]);
  }
  const spans = max.map((value, i) => value - min[i]);
  assert.ok(spans[1] > spans[0] * 1.8, JSON.stringify(spans));
  assert.ok(spans[1] > spans[2] * 1.8, JSON.stringify(spans));
  assert.ok(spans.every(span => span < 1.4), JSON.stringify(spans));
  assert.ok(cloud.colors.some((value, i) => i % 4 === 0 && value > 0.7));
});

test('8-to-16 upgrade keeps original pixels, masks and camera metadata; clear creates empty slots', () => {
  const original = capturedBottleSlots();
  const upgraded = buildSlotsForPreset(CAPTURE_PRESETS.express, original);
  assert.equal(upgraded.filter(slot => slot.captured).length, 8);
  assert.equal(upgraded.filter(slot => !slot.captured).length, 8);
  for (const slot of upgraded.filter(slot => slot.captured)) {
    const before = original.find(previous => previous.azimuthDeg === slot.azimuthDeg && previous.elevationDeg === slot.elevationDeg);
    assert.equal(slot.photo.rgba, before.photo.rgba);
    assert.equal(slot.photo.foregroundMask, before.photo.foregroundMask);
    assert.equal(slot.photo.source, 'real');
    assert.equal(slot.photo.slotIndex, slot.index);
  }
  assert.equal(buildSlotsForPreset(CAPTURE_PRESETS.express).some(slot => slot.captured), false);
});

test('eight unusable photos do not announce that the capture is ready', () => {
  const slots = capturedBottleSlots();
  slots[0].photo.verdict = 'bad';
  const report = evaluateCoverage(CAPTURE_PRESETS.minimal, slots, 0, 'en');
  assert.equal(report.isReadyToRender, false);
  assert.match(report.missingSummaryText, /retaking/);
});

test('video keyframes prioritize quality while rejecting repeated views and near-identical noise', () => {
  const frame = (value, score, time) => ({ bin: 0, fingerprint: new Uint8Array(24).fill(value), score, time });
  const previous = frame(40, 80, 1);
  const duplicate = frame(41, 100, 2);
  const clearer = frame(90, 90, 3);
  const blurry = frame(120, 25, 4);
  assert.equal(selectDistinctFrame([duplicate, blurry, clearer], [previous]), clearer);
  assert.equal(selectDistinctFrame([duplicate], [previous]), null);
  assert.equal(selectDistinctFrame([blurry, clearer], []), clearer);
});

test('six distinct video sectors can preview without pretending a top ring was filmed', () => {
  const slots = buildSlotsForPreset(CAPTURE_PRESETS.video);
  const photo = capturedBottleSlots()[0].photo;
  slots.slice(0, 6).forEach(slot => { slot.captured = true; slot.photo = { ...photo, slotIndex: slot.index }; });
  const report = evaluateCoverage(CAPTURE_PRESETS.video, slots, 0, 'en');
  assert.equal(report.isReadyToRender, true);
  assert.equal(report.minimumViableCount, 6);
  assert.equal(report.topRingCapturedCount, 0);
  assert.equal(report.ringProgress.length, 1);
  assert.ok(slots.every(slot => slot.elevationDeg === 0));
});

test('one view or views from only one side never become an extruded solid', async () => {
  const slots = capturedBottleSlots();
  for (const views of [slots.slice(0, 1), slots.slice(0, 4).map((slot, i) => ({ ...slot, azimuthDeg: i * 20 }))]) {
    await assert.rejects(reconstructSplatsFromCaptures(views, 48, 'standard'), error => error instanceof ReconstructionError && error.code === 'views');
  }
});

test('round-bottle fitting preserves measured neck/body proportions and captured colors', async () => {
  const photo = capturedBottleSlots()[0].photo;
  const slots = buildSlotsForPreset(CAPTURE_PRESETS.video).map(slot => ({ ...slot, captured: true, photo: { ...photo } }));
  const cloud = await reconstructBottleFromCaptures(slots);
  assert.ok(cloud.count > 40000);
  let neck = 0, body = 0;
  for (let i = 0; i < cloud.count; i++) {
    const [x,y,z] = cloud.positions.subarray(i*3,i*3+3);
    assert.ok(Number.isFinite(x+y+z));
    const radius = Math.hypot(x,z);
    if (y>.45&&y<.52) neck=Math.max(neck,radius);
    if (Math.abs(y)<.1) body=Math.max(body,radius);
  }
  assert.ok(neck>0&&neck<body*.65,JSON.stringify({neck,body}));
  assert.ok(cloud.colors.some((v,i)=>i%4===0&&v>.9),'captured pale label survives');
  await assert.rejects(reconstructBottleFromCaptures(slots.slice(0,1)),e=>e instanceof ReconstructionError&&e.code==='views');
});

test('reconstruction uses the inspected mask even when subject colors also occur in the background', async () => {
  const slots = capturedBottleSlots().map(slot => ({ ...slot, photo: { ...slot.photo, rgba: new Uint8ClampedArray(slot.photo.rgba.length).fill(255) } }));
  const cloud = await reconstructSplatsFromCaptures(slots,48,'standard');
  assert.ok(cloud.count>100);
  assert.ok(cloud.colors.every(v=>Number.isFinite(v)));
});
