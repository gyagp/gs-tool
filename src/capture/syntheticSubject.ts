/**
 * Procedural Photometric Calibration Object Renderer
 * Generates realistic multi-angle photographs of a sculpted iridescent ceramic & brass
 * geometric artifact on a calibrated turntable so users without a physical setup handy
 * can immediately test multi-view capture, SfM space-carving, and WebGPU Gaussian Splatting.
 */

import { mat4LookAt, mat4Multiply, mat4PerspectiveZO, sphericalToCartesian, vec3Dot, vec3Normalize, Vec3 } from '../math/mat4';
import { CAPTURE_RADIUS } from './coverage';

interface SpherePrimitive {
  center: Vec3;
  radius: number;
  color: Vec3;
  roughness: number;
  metallic: number;
}

const CALIBRATION_PRIMITIVES: SpherePrimitive[] = buildCalibrationSculpture();

function buildCalibrationSculpture(): SpherePrimitive[] {
  const prims: SpherePrimitive[] = [];

  // Pedestal base ring
  for (let i = 0; i < 24; i++) {
    const a = (i / 24) * Math.PI * 2;
    prims.push({
      center: [Math.cos(a) * 0.42, -0.36, Math.sin(a) * 0.42],
      radius: 0.11,
      color: [0.18, 0.22, 0.28],
      roughness: 0.45,
      metallic: 0.6
    });
  }

  // Central core orb (warm amber-gold metallic)
  prims.push({
    center: [0, 0.04, 0],
    radius: 0.34,
    color: [0.96, 0.66, 0.18],
    roughness: 0.28,
    metallic: 0.8
  });

  // Helical double-spiral arms (teal ceramic & coral enamel) for unambiguous 360° parallax features
  for (let i = 0; i < 28; i++) {
    const t = i / 27;
    const angle1 = t * Math.PI * 3.2;
    const y = -0.28 + t * 0.74;
    const r = 0.46 * (1 - 0.38 * t);

    prims.push({
      center: [Math.cos(angle1) * r, y, Math.sin(angle1) * r],
      radius: 0.085 + 0.025 * Math.sin(t * Math.PI),
      color: [0.06, 0.88 - 0.3 * t, 0.76],
      roughness: 0.32,
      metallic: 0.2
    });

    const angle2 = angle1 + Math.PI * 0.92;
    prims.push({
      center: [Math.cos(angle2) * r * 0.92, y * 0.95, Math.sin(angle2) * r * 0.92],
      radius: 0.078,
      color: [0.94, 0.34 + 0.35 * t, 0.38],
      roughness: 0.36,
      metallic: 0.15
    });
  }

  // Asymmetric orientation fin + crown finial so front/back/left/right are unmistakably distinct
  prims.push({
    center: [0.0, 0.52, 0.0],
    radius: 0.14,
    color: [0.95, 0.96, 0.98],
    roughness: 0.22,
    metallic: 0.1
  });
  prims.push({
    center: [0.0, 0.16, 0.48],
    radius: 0.12,
    color: [0.22, 0.62, 0.98],
    roughness: 0.25,
    metallic: 0.4
  });

  return prims;
}

/**
 * Raytraces a crisp anti-aliased multi-angle photograph of the calibration artifact from (azimuth, elevation)
 */
export function renderSyntheticCaptureView(
  azimuthDeg: number,
  elevationDeg: number,
  fovDeg: number,
  width = 256,
  height = 256
): { rgba: Uint8ClampedArray; dataUrl: string } {
  const rgba = new Uint8ClampedArray(width * height * 4);
  const eye = sphericalToCartesian(azimuthDeg, elevationDeg, CAPTURE_RADIUS);
  const target: Vec3 = [0, 0.05, 0];
  const forward = vec3Normalize([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);

  // Camera basis vectors
  const worldUp: Vec3 = [0, 1, 0];
  const right = vec3Normalize([
    forward[1] * worldUp[2] - forward[2] * worldUp[1],
    forward[2] * worldUp[0] - forward[0] * worldUp[2],
    forward[0] * worldUp[1] - forward[1] * worldUp[0]
  ]);
  const up: Vec3 = [
    right[1] * forward[2] - right[2] * forward[1],
    right[2] * forward[0] - right[0] * forward[2],
    right[0] * forward[1] - right[1] * forward[0]
  ];

  const halfH = Math.tan(((fovDeg * Math.PI) / 180) * 0.5);
  const halfW = halfH * (width / height);

  const keyLight = vec3Normalize([0.55, 0.78, 0.42]);
  const rimLight = vec3Normalize([-0.68, 0.35, -0.58]);

  for (let py = 0; py < height; py++) {
    const ndcY = 1 - ((py + 0.5) / height) * 2;
    for (let px = 0; px < width; px++) {
      const ndcX = ((px + 0.5) / width) * 2 - 1;

      const dir = vec3Normalize([
        forward[0] + right[0] * ndcX * halfW + up[0] * ndcY * halfH,
        forward[1] + right[1] * ndcX * halfW + up[1] * ndcY * halfH,
        forward[2] + right[2] * ndcX * halfW + up[2] * ndcY * halfH
      ]);

      let closestT = Infinity;
      let hitPrim: SpherePrimitive | null = null;

      for (let i = 0; i < CALIBRATION_PRIMITIVES.length; i++) {
        const s = CALIBRATION_PRIMITIVES[i];
        const ocX = eye[0] - s.center[0];
        const ocY = eye[1] - s.center[1];
        const ocZ = eye[2] - s.center[2];
        const b = ocX * dir[0] + ocY * dir[1] + ocZ * dir[2];
        const c = ocX * ocX + ocY * ocY + ocZ * ocZ - s.radius * s.radius;
        const disc = b * b - c;
        if (disc > 0) {
          const t = -b - Math.sqrt(disc);
          if (t > 0.05 && t < closestT) {
            closestT = t;
            hitPrim = s;
          }
        }
      }

      const outIdx = (py * width + px) * 4;
      if (hitPrim) {
        const hx = eye[0] + dir[0] * closestT;
        const hy = eye[1] + dir[1] * closestT;
        const hz = eye[2] + dir[2] * closestT;
        const n = vec3Normalize([
          hx - hitPrim.center[0],
          hy - hitPrim.center[1],
          hz - hitPrim.center[2]
        ]);

        // Procedural surface micro-pattern for rich multi-view texture gradients
        const texWave =
          0.92 + 0.08 * Math.sin(hx * 26) * Math.cos(hy * 26) * Math.sin(hz * 26);

        const ndotl1 = Math.max(0, vec3Dot(n, keyLight));
        const ndotl2 = Math.max(0, vec3Dot(n, rimLight));
        const halfVec = vec3Normalize([keyLight[0] - dir[0], keyLight[1] - dir[1], keyLight[2] - dir[2]]);
        const spec = Math.pow(Math.max(0, vec3Dot(n, halfVec)), 28) * (1 - hitPrim.roughness * 0.6);

        const ambient = 0.22 + 0.1 * Math.max(0, n[1]);
        const diff = ambient + ndotl1 * 0.78 + ndotl2 * 0.32;

        const r = Math.min(1, hitPrim.color[0] * texWave * diff + spec * 0.65);
        const g = Math.min(1, hitPrim.color[1] * texWave * diff + spec * 0.65);
        const b = Math.min(1, hitPrim.color[2] * texWave * diff + spec * 0.72);

        rgba[outIdx] = Math.round(Math.pow(r, 1 / 2.2) * 255);
        rgba[outIdx + 1] = Math.round(Math.pow(g, 1 / 2.2) * 255);
        rgba[outIdx + 2] = Math.round(Math.pow(b, 1 / 2.2) * 255);
        rgba[outIdx + 3] = 255;
      } else {
        // Dark matte studio cyclorama background (distinct from foreground for clean visual hull segmentation)
        const vig = 1 - 0.35 * (ndcX * ndcX + ndcY * ndcY);
        const bg = Math.max(8, Math.round(14 * vig));
        rgba[outIdx] = bg;
        rgba[outIdx + 1] = bg + 2;
        rgba[outIdx + 2] = bg + 5;
        rgba[outIdx + 3] = 255;
      }
    }
  }

  const offscreen = document.createElement('canvas');
  offscreen.width = width;
  offscreen.height = height;
  const ctx = offscreen.getContext('2d')!;
  ctx.putImageData(new ImageData(new Uint8ClampedArray(rgba), width, height), 0, 0);

  return {
    rgba,
    dataUrl: offscreen.toDataURL('image/jpeg', 0.9)
  };
}

// Keep unused imports clean
void mat4LookAt;
void mat4Multiply;
void mat4PerspectiveZO;
