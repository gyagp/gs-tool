/**
 * Zero-allocation column-major 4x4 matrix, Vec3, Quaternion, and Z-order Morton utilities
 * Optimized for high-frequency WebGPU uniform updates & PlayCanvas-style spatial locality.
 */

export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number]; // [x, y, z, w]

export function vec3Sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function vec3Add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function vec3Scale(a: Vec3, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}

export function vec3Dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function vec3Cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  ];
}

export function vec3Len(a: Vec3): number {
  return Math.hypot(a[0], a[1], a[2]);
}

export function vec3Normalize(a: Vec3): Vec3 {
  const len = Math.hypot(a[0], a[1], a[2]);
  if (len < 1e-8) return [0, 0, 1];
  return [a[0] / len, a[1] / len, a[2] / len];
}

export function mat4Identity(out = new Float32Array(16)): Float32Array {
  out.fill(0);
  out[0] = 1;
  out[5] = 1;
  out[10] = 1;
  out[15] = 1;
  return out;
}

/**
 * Right-handed lookAt view matrix (camera looks along -Z in view space, WebGPU compatible)
 */
export function mat4LookAt(eye: Vec3, target: Vec3, up: Vec3, out = new Float32Array(16)): Float32Array {
  let zx = eye[0] - target[0];
  let zy = eye[1] - target[1];
  let zz = eye[2] - target[2];
  let len = Math.hypot(zx, zy, zz);
  if (len < 1e-8) {
    zz = 1;
  } else {
    zx /= len;
    zy /= len;
    zz /= len;
  }

  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;
  len = Math.hypot(xx, xy, xz);
  if (len < 1e-8) {
    xx = 1;
    xy = 0;
    xz = 0;
  } else {
    xx /= len;
    xy /= len;
    xz /= len;
  }

  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  out[0] = xx;
  out[1] = yx;
  out[2] = zx;
  out[3] = 0;

  out[4] = xy;
  out[5] = yy;
  out[6] = zy;
  out[7] = 0;

  out[8] = xz;
  out[9] = yz;
  out[10] = zz;
  out[11] = 0;

  out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
  out[15] = 1;

  return out;
}

/**
 * WebGPU [0, 1] depth perspective projection matrix (column-major)
 */
export function mat4PerspectiveZO(fovYRad: number, aspect: number, near: number, far: number, out = new Float32Array(16)): Float32Array {
  const f = 1.0 / Math.tan(fovYRad * 0.5);
  const nf = 1.0 / (near - far);

  out.fill(0);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = far * nf;
  out[11] = -1;
  out[14] = far * near * nf;
  return out;
}

export function mat4Multiply(a: Float32Array, b: Float32Array, out = new Float32Array(16)): Float32Array {
  const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
  const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
  const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
  const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

  for (let c = 0; c < 4; c++) {
    const idx = c * 4;
    const b0 = b[idx], b1 = b[idx + 1], b2 = b[idx + 2], b3 = b[idx + 3];
    out[idx]     = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[idx + 1] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[idx + 2] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[idx + 3] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
  }
  return out;
}

/**
 * Spherical coordinates (azimuth deg, elevation deg, radius) to 3D Cartesian eye position (Y-up)
 */
export function sphericalToCartesian(azimuthDeg: number, elevationDeg: number, radius: number): Vec3 {
  const az = (azimuthDeg * Math.PI) / 180;
  const el = (elevationDeg * Math.PI) / 180;
  const cosEl = Math.cos(el);
  return [
    radius * cosEl * Math.sin(az),
    radius * Math.sin(el),
    radius * cosEl * Math.cos(az)
  ];
}

/**
 * IEEE-754 float32 -> float16 bit packing for packed GPU splat storage
 */
const _f32Buf = new Float32Array(1);
const _u32Buf = new Uint32Array(_f32Buf.buffer);

export function floatToHalf(val: number): number {
  _f32Buf[0] = val;
  const x = _u32Buf[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 127 + 15;
  let mantissa = x & 0x007fffff;

  if (exp <= 0) {
    if (exp < -10) return sign;
    mantissa = (mantissa | 0x00800000) >>> (1 - exp);
    return sign | (mantissa >>> 13);
  } else if (exp === 0xff - 127 + 15) {
    return sign | 0x7c00 | (mantissa ? 0x0200 : 0);
  } else if (exp > 30) {
    return sign | 0x7c00;
  }
  return sign | (exp << 10) | (mantissa >>> 13);
}

export function packHalf2x16(x: number, y: number): number {
  return (floatToHalf(x) & 0xffff) | ((floatToHalf(y) & 0xffff) << 16);
}

export function packUnorm4x8(r: number, g: number, b: number, a: number): number {
  const cr = Math.max(0, Math.min(255, Math.round(r * 255)));
  const cg = Math.max(0, Math.min(255, Math.round(g * 255)));
  const cb = Math.max(0, Math.min(255, Math.round(b * 255)));
  const ca = Math.max(0, Math.min(255, Math.round(a * 255)));
  return (cr | (cg << 8) | (cb << 16) | (ca << 24)) >>> 0;
}

export function packSnorm4x8Quat(qx: number, qy: number, qz: number, qw: number): number {
  const len = Math.hypot(qx, qy, qz, qw) || 1;
  const s = qw < 0 ? -1 / len : 1 / len;
  const cx = Math.max(-127, Math.min(127, Math.round(qx * s * 127))) & 0xff;
  const cy = Math.max(-127, Math.min(127, Math.round(qy * s * 127))) & 0xff;
  const cz = Math.max(-127, Math.min(127, Math.round(qz * s * 127))) & 0xff;
  const cw = Math.max(-127, Math.min(127, Math.round(qw * s * 127))) & 0xff;
  return (cx | (cy << 8) | (cz << 16) | (cw << 24)) >>> 0;
}

/**
 * 30-bit Z-order Morton code (10 bits per axis) for PlayCanvas-style spatial cache locality
 */
function expandBits10(v: number): number {
  let x = Math.max(0, Math.min(1023, v | 0));
  x = (x | (x << 16)) & 0x030000ff;
  x = (x | (x << 8)) & 0x0300f00f;
  x = (x | (x << 4)) & 0x030c30c3;
  x = (x | (x << 2)) & 0x09249249;
  return x >>> 0;
}

export function computeMortonOrder(positions: Float32Array, count: number): Uint32Array {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (let i = 0; i < count; i++) {
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }

  const invX = 1023 / Math.max(1e-6, maxX - minX);
  const invY = 1023 / Math.max(1e-6, maxY - minY);
  const invZ = 1023 / Math.max(1e-6, maxZ - minZ);

  const codes = new Uint32Array(count);
  const order = new Uint32Array(count);
  for (let i = 0; i < count; i++) {
    const mx = expandBits10((positions[i * 3] - minX) * invX);
    const my = expandBits10((positions[i * 3 + 1] - minY) * invY);
    const mz = expandBits10((positions[i * 3 + 2] - minZ) * invZ);
    codes[i] = (mx | (my << 1) | (mz << 2)) >>> 0;
    order[i] = i;
  }

  // Fast 8-bit LSD radix sort on CPU (4 passes over 32-bit Morton codes, O(N) zero-GC)
  const tmpOrder = new Uint32Array(count);
  const counts = new Uint32Array(256);

  for (let shift = 0; shift < 30; shift += 8) {
    counts.fill(0);
    const src = (shift & 8) === 0 ? order : tmpOrder;
    const dst = (shift & 8) === 0 ? tmpOrder : order;

    for (let i = 0; i < count; i++) {
      counts[(codes[src[i]] >>> shift) & 0xff]++;
    }
    let sum = 0;
    for (let b = 0; b < 256; b++) {
      const c = counts[b];
      counts[b] = sum;
      sum += c;
    }
    for (let i = 0; i < count; i++) {
      const idx = src[i];
      const bucket = (codes[idx] >>> shift) & 0xff;
      dst[counts[bucket]++] = idx;
    }
  }

  return order;
}
