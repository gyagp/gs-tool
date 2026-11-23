import type { CaptureSlot } from '../capture/coverage';
import { foregroundFromMask } from '../capture/foreground';
import { normalToTangentQuat, packSplatsForGPU, ReconstructionError, type SplatCloudData } from './reconstruct';
import type { TexturedSurfaceData } from './surface';
import { sampleCaptureColor } from '../capture/colorSampling';

interface BottleView {
  slot: CaptureSlot;
  cx: number; cy: number; cos: number; sin: number;
  top: number; height: number;
  left: Float32Array; right: Float32Array;
}

const ROWS = 256;
const median = (a: number[]) => { a.sort((x, y) => x - y); return a[Math.floor(a.length / 2)]; };

/** Bilinear source sampling uses preserved texture pixels, not the segmentation thumbnail. */
export function sampleBottleColor(view: BottleView, t: number, delta: number): number[] {
  const photo = view.slot.photo!;
  const row = Math.max(0, Math.min(ROWS - 1, t * (ROWS - 1))), y0 = Math.floor(row), y1 = Math.min(ROWS - 1, y0 + 1), f = row - y0;
  const left = view.left[y0] * (1 - f) + view.left[y1] * f;
  const right = view.right[y0] * (1 - f) + view.right[y1] * f;
  const u = (left + right) / 2 + Math.sin(delta) * (right - left) / 2 * .98;
  const v = view.top + t * (view.height - 1);
  let x = view.cx + u * view.cos + v * view.sin, y = view.cy - u * view.sin + v * view.cos;
  return sampleCaptureColor(photo, x, y);
}

function buildTexturedSurface(views: BottleView[], radii: Float32Array, height: number): TexturedSurfaceData {
  // A single captured view covers the visible surface at a time. Unknown poses make
  // a static strip atlas tear labels; preserving complete views avoids that failure.
  const textures = views.map(view => {
    const photo = view.slot.photo!, source = photo.texture;
    const pixelsPerWorkPixel = source ? source.height / source.sourceHeight : 1;
    const textureHeight = Math.max(1, Math.min(1920, Math.ceil(view.height * pixelsPerWorkPixel)));
    const textureWidth = Math.max(1, Math.min(1024, Math.ceil(Math.max(...view.right.map((r, i) => r - view.left[i] + 1)) * pixelsPerWorkPixel)));
    const rgba = new Uint8ClampedArray(textureWidth * textureHeight * 4);
    for (let x = 0; x < textureWidth; x++) for (let y = 0; y < textureHeight; y++) {
      const color = sampleBottleColor(view, (y + .5) / textureHeight, Math.asin((x + .5) / textureWidth * 2 - 1));
      const p = (y * textureWidth + x) * 4;
      rgba[p] = color[0] * 255; rgba[p + 1] = color[1] * 255; rgba[p + 2] = color[2] * 255; rgba[p + 3] = 255;
    }
    return { azimuthRad: view.slot.azimuthDeg * Math.PI / 180, timeSeconds: photo.videoTimeSeconds,
      width: textureWidth, height: textureHeight, rgba };
  });
  const sides = 192, rows = ROWS;
  const positions = new Float32Array((rows + 1) * (sides + 1) * 3);
  const normals = new Float32Array(positions.length), uvs = new Float32Array((rows + 1) * (sides + 1) * 2);
  const indices: number[] = [];
  for (let y = 0; y <= rows; y++) for (let a = 0; a <= sides; a++) {
    const t = y / rows, theta = a / sides * 2 * Math.PI;
    const i = y * (sides + 1) + a, row = Math.min(ROWS - 1, Math.round(t * (ROWS - 1)));
    // Close only the two measured silhouette endpoints; hidden cap/base details remain unknown.
    const r = y === 0 || y === rows ? 0 : radii[row];
    positions.set([r * Math.sin(theta), .05 + height * (.5 - t), r * Math.cos(theta)], i * 3);
    const slope = (radii[Math.min(ROWS - 1, row + 1)] - radii[Math.max(0, row - 1)]) / (2 * height / ROWS);
    const length = Math.hypot(1, slope);
    normals.set([Math.sin(theta) / length, slope / length, Math.cos(theta) / length], i * 3);
    uvs.set([a / sides, t], i * 2);
    if (y < rows && a < sides) { const j = i + sides + 1; indices.push(i, j, i + 1, i + 1, j, j + 1); }
  }
  return { positions, normals, uvs, indices: new Uint32Array(indices), views: textures };
}

/** Observed silhouette in an upright coordinate system; no color-based re-segmentation. */
function prepareView(slot: CaptureSlot): BottleView {
  const photo = slot.photo!;
  const { mask, usable } = foregroundFromMask(photo.foregroundMask, photo.workWidth, photo.workHeight);
  if (!usable) throw new ReconstructionError('foreground');
  const width = photo.workWidth;
  let count = 0, cx = 0, cy = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) { count++; cx += i % width; cy += Math.floor(i / width); }
  cx /= count; cy /= count;
  let xx = 0, yy = 0, xy = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) {
    const dx = i % width - cx, dy = Math.floor(i / width) - cy;
    xx += dx * dx; yy += dy * dy; xy += dx * dy;
  }
  if (yy < xx * 2) throw new ReconstructionError('geometry');
  const angle = .5 * Math.atan2(2 * xy, yy - xx), cos = Math.cos(angle), sin = Math.sin(angle);
  let top = Infinity, bottom = -Infinity;
  for (let i = 0; i < mask.length; i++) if (mask[i]) {
    const v = (i % width - cx) * sin + (Math.floor(i / width) - cy) * cos;
    top = Math.min(top, v); bottom = Math.max(bottom, v);
  }
  const height = bottom - top + 1;
  const left = new Float32Array(ROWS).fill(Infinity), right = new Float32Array(ROWS).fill(-Infinity);
  for (let i = 0; i < mask.length; i++) if (mask[i]) {
    const dx = i % width - cx, dy = Math.floor(i / width) - cy;
    const u = dx * cos - dy * sin, v = dx * sin + dy * cos;
    const row = Math.min(ROWS - 1, Math.floor((v - top) / height * ROWS));
    left[row] = Math.min(left[row], u); right[row] = Math.max(right[row], u);
  }
  // Rows narrower than one source pixel occur after resampling. Interpolate only between observations.
  for (let row = 0; row < ROWS; row++) if (!Number.isFinite(left[row])) {
    let before = row - 1, after = row + 1;
    while (before >= 0 && !Number.isFinite(left[before])) before--;
    while (after < ROWS && !Number.isFinite(left[after])) after++;
    if (before < 0) { left[row] = left[after]; right[row] = right[after]; continue; }
    if (after === ROWS) { left[row] = left[before]; right[row] = right[before]; continue; }
    const t = (row - before) / (after - before);
    left[row] = left[before] * (1 - t) + left[after] * t;
    right[row] = right[before] * (1 - t) + right[after] * t;
  }
  // Suppress pixel-scale mask stair steps before mapping the original-resolution
  // image. The low-resolution mask should not introduce wavy label lettering.
  const smooth = (values: Float32Array) => values.map((_, row) => median(Array.from({ length: 9 }, (_, i) => values[Math.max(0, Math.min(ROWS - 1, row + i - 4))])));
  return { slot, cx, cy, cos, sin, top, height, left: smooth(left), right: smooth(right) };
}

/**
 * Explicit round-bottle prior: median measured radius profile, revolved around its axis.
 * This is a fitted surface with captured colors, not SfM or a trained 3DGS reconstruction.
 * Camera azimuth still follows the assumed steady video orbit; unseen geometry is not measured.
 */
export async function reconstructBottleFromCaptures(
  slots: CaptureSlot[], onProgress?: (stage: string, pct: number) => void
): Promise<SplatCloudData> {
  const start = performance.now();
  const captured = slots.filter(s => s.captured && s.photo);
  const azimuths = [...new Set(captured.map(s => s.azimuthDeg))].sort((a, b) => a - b);
  const gap = azimuths.length < 2 ? 360 : Math.max(...azimuths.map((a, i) => (azimuths[i + 1] ?? azimuths[0] + 360) - a));
  if (captured.length < 6 || gap > 150) throw new ReconstructionError('views');
  onProgress?.('Fitting measured bottle silhouettes (round-bottle assumption)…', 15);
  await new Promise(resolve => setTimeout(resolve, 10));
  const views = captured.map(prepareView);
  const worldHeight = 1.3;
  const radii = new Float32Array(ROWS);
  for (let y = 0; y < ROWS; y++) {
    radii[y] = median(views.map(v => (v.right[y] - v.left[y] + 1) / v.height * worldHeight / 2));
  }
  const smooth = radii.map((_, y) => median(Array.from({ length: 5 }, (_, dy) => radii[Math.max(0, Math.min(ROWS - 1, y + dy - 2))])));
  const maxRadius = Math.max(...smooth);
  if (maxRadius < .035 || maxRadius > worldHeight * .28) throw new ReconstructionError('geometry');

  onProgress?.('Sampling original bottle colors without averaging opposing labels…', 50);
  await new Promise(resolve => setTimeout(resolve, 10));
  const positions: number[] = [], scales: number[] = [], quats: number[] = [], colors: number[] = [];
  const sides = 192, verticalStep = worldHeight / ROWS;
  for (let y = 0; y < ROWS; y++) {
    const radius = smooth[y];
    const slope = (smooth[Math.min(ROWS - 1, y + 1)] - smooth[Math.max(0, y - 1)]) / (2 * verticalStep);
    for (let a = 0; a < sides; a++) {
      const theta = a / sides * Math.PI * 2;
      const sin = Math.sin(theta), cos = Math.cos(theta);
      let chosen: BottleView | undefined, delta = 0, bestWeight = -1;
      for (const view of views) {
        const d = theta - view.slot.azimuthDeg * Math.PI / 180;
        const weight = Math.cos(d);
        if (weight > bestWeight) { bestWeight = weight; chosen = view; delta = d; }
      }
      const view = chosen!, photo = view.slot.photo!;
      const center = (view.left[y] + view.right[y]) / 2;
      const u = center + Math.sin(delta) * (view.right[y] - view.left[y]) / 2 * .98;
      const v = view.top + (y + .5) / ROWS * view.height;
      const px = Math.max(0, Math.min(photo.workWidth - 1, Math.round(view.cx + u * view.cos + v * view.sin)));
      const py = Math.max(0, Math.min(photo.workHeight - 1, Math.round(view.cy - u * view.sin + v * view.cos)));
      if (!photo.foregroundMask[py * photo.workWidth + px]) continue;
      positions.push(radius * sin, .05 + worldHeight * (.5 - (y + .5) / ROWS), radius * cos);
      const scale = Math.max(verticalStep, 2 * Math.PI * radius / sides) * .68;
      scales.push(scale, scale, verticalStep * .22);
      quats.push(...normalToTangentQuat(sin, slope, cos));
      const color = sampleBottleColor(view, (y + .5) / ROWS, delta);
      colors.push(color[0], color[1], color[2], .97);
    }
  }
  if (positions.length < ROWS * sides * .6 * 3) throw new ReconstructionError('geometry');
  const result = packSplatsForGPU(new Float32Array(positions), new Float32Array(scales), new Float32Array(quats), new Float32Array(colors), positions.length / 3);
  result.surface = buildTexturedSurface(views, smooth, worldHeight);
  result.buildTimeMs = performance.now() - start;
  onProgress?.('Round-bottle approximation ready; camera directions remain estimated', 100);
  return result;
}
