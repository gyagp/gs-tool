/**
 * PlayCanvas-Inspired High-Performance WebGPU WGSL Shaders for 3D Gaussian Splatting
 *
 * Key optimizations adopted from PlayCanvas Engine & SuperSplat:
 * 1. Packed 32-byte input splats + Compact 16-byte Projected Splat Cache (`vec4<u32>`):
 *    - word0: pack2x16float(ndc.x, ndc.y)
 *    - word1: pack2x16float(axis1.x, axis1.y) (screen-space major eigenvector in pixels)
 *    - word2: pack2x16float(axis2.x, axis2.y) (screen-space minor eigenvector in pixels)
 *    - word3: RGBA8 premultiplied/compensated color + mip-splat AA opacity
 * 2. Exact 2D Covariance Eigenvector Decomposition + Low-Pass 0.3px^2 & Mip-AA Compensation.
 * 3. Frustum + Near/Far + Contribution Mass (`alpha * 2pi * sqrt(det)`) Culling in Compute.
 * 4. Atomic Survivor Compaction (`atomicAdd(&counter[0], 1u)`) + 16-bit Log-Spaced Sort Keys.
 * 5. GPU-Driven Indirect Dispatch & Batched 128-Splat Instanced Indirect Draw (`drawIndexedIndirect`)
 *    with zero CPU readback stalls.
 * 6. Portable Adreno-Safe 4-bit Multi-Pass Radix Sort (`countOneBits` 256-thread bitmask ranking).
 */

export const INSTANCE_SPLATS = 128;
export const PROJECTOR_WG_SIZE = 256;
export const RADIX_WG_SIZE = 256;
export const RADIX_ELTS_PER_THREAD = 4;
export const RADIX_ELTS_PER_WG = RADIX_WG_SIZE * RADIX_ELTS_PER_THREAD; // 1024

export const PROJECTOR_WGSL = /* wgsl */ `
struct CameraUniforms {
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
  viewport: vec2<f32>,      // width, height in pixels
  focal: vec2<f32>,         // fx, fy in pixels
  camPos: vec3<f32>,
  splatCount: u32,
  nearClip: f32,
  farClip: f32,
  splatScale: f32,
  minContribution: f32,
  renderMode: u32,          // 0 = Normal 3DGS, 1 = Ellipsoid Rings, 2 = Depth Heatmap, 3 = Splat Centers
  sortKeyMinBits: i32,
  sortKeyScale: f32,
  pad0: u32,
};

@group(0) @binding(0) var<uniform> uCam: CameraUniforms;
@group(0) @binding(1) var<storage, read> rawSplats: array<vec4<u32>>;     // 2 x vec4<u32> (32B) per splat
@group(0) @binding(2) var<storage, read_write> projCache: array<vec4<u32>>; // 16B per visible/cached splat
@group(0) @binding(3) var<storage, read_write> sortKeys: array<u32>;
@group(0) @binding(4) var<storage, read_write> sortValues: array<u32>;
@group(0) @binding(5) var<storage, read_write> survivorCount: array<atomic<u32>>;

const MAX_SORT_KEY: u32 = 65535u; // 16-bit log-spaced depth key (4 passes of 4-bit radix sort!)
const LOWPASS_FILTER: f32 = 0.3;
const AA_FILTER: f32 = 0.1;

fn quatToMat3(q: vec4<f32>) -> mat3x3<f32> {
  let x = q.x; let y = q.y; let z = q.z; let w = q.w;
  let x2 = x + x; let y2 = y + y; let z2 = z + z;
  let xx = x * x2; let xy = x * y2; let xz = x * z2;
  let yy = y * y2; let yz = y * z2; let zz = z * z2;
  let wx = w * x2; let wy = w * y2; let wz = w * z2;
  return mat3x3<f32>(
    vec3<f32>(1.0 - (yy + zz), xy + wz, xz - wy),
    vec3<f32>(xy - wz, 1.0 - (xx + zz), yz + wx),
    vec3<f32>(xz + wy, yz - wx, 1.0 - (xx + yy))
  );
}

fn turboColormap(tIn: f32) -> vec3<f32> {
  let t = clamp(tIn, 0.0, 1.0);
  return clamp(
    vec3<f32>(
      0.13572 + t * (4.61539 + t * (-42.6603 + t * (132.131 + t * (-152.942 + t * 59.286)))),
      0.09140 + t * (2.19418 + t * (4.84296 + t * (-14.1850 + t * (4.27729 + t * 2.82956)))),
      0.10667 + t * (12.6419 + t * (-60.5820 + t * (110.362 + t * (-89.9031 + t * 27.3482))))
    ),
    vec3<f32>(0.0),
    vec3<f32>(1.0)
  );
}

fn logDepthSortKey(distSq: f32) -> u32 {
  let bits = bitcast<i32>(max(distSq, 1e-4)) - uCam.sortKeyMinBits;
  let norm = clamp(f32(bits) * uCam.sortKeyScale, 0.0, f32(MAX_SORT_KEY));
  // Invert so farthest splats receive smallest keys -> drawn first (back-to-front blending)
  return MAX_SORT_KEY - u32(norm);
}

@compute @workgroup_size(${PROJECTOR_WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= uCam.splatCount) {
    return;
  }

  let w0 = rawSplats[idx * 2u];
  let w1 = rawSplats[idx * 2u + 1u];

  let worldPos = vec3<f32>(bitcast<f32>(w0.x), bitcast<f32>(w0.y), bitcast<f32>(w0.z));
  let viewPos4 = uCam.view * vec4<f32>(worldPos, 1.0);
  let depth = -viewPos4.z;

  if (depth <= uCam.nearClip || depth >= uCam.farClip) {
    return;
  }

  let clipPos = uCam.proj * viewPos4;
  let ndc = clipPos.xy / max(clipPos.w, 1e-6);

  // Broad frustum guard before covariance math
  if (abs(ndc.x) > 1.38 || abs(ndc.y) > 1.38) {
    return;
  }

  let rgba = unpack4x8unorm(w1.z);
  if (rgba.a < 0.006) {
    return;
  }

  let qRaw = unpack4x8snorm(w0.w);
  let qNorm = normalize(qRaw);
  let sxy = unpack2x16float(w1.x);
  let szw = unpack2x16float(w1.y);
  let scale = vec3<f32>(sxy.x, sxy.y, szw.x) * uCam.splatScale;

  // Build 3D covariance Sigma = R * S^2 * R^T
  let rot = quatToMat3(qNorm);
  let m = mat3x3<f32>(rot[0] * scale.x, rot[1] * scale.y, rot[2] * scale.z);
  let cov3d = m * transpose(m);

  // EWA Splatting Jacobian J * W * cov3d * W^T * J^T
  let limX = 1.3 * (uCam.viewport.x * 0.5) / uCam.focal.x;
  let limY = 1.3 * (uCam.viewport.y * 0.5) / uCam.focal.y;
  let tx = clamp(viewPos4.x / depth, -limX, limX) * depth;
  let ty = clamp(viewPos4.y / depth, -limY, limY) * depth;
  let invZ = 1.0 / depth;
  let invZ2 = invZ * invZ;

  let J = mat3x3<f32>(
    vec3<f32>(uCam.focal.x * invZ, 0.0, 0.0),
    vec3<f32>(0.0, uCam.focal.y * invZ, 0.0),
    vec3<f32>(-uCam.focal.x * tx * invZ2, -uCam.focal.y * ty * invZ2, 0.0)
  );

  let W = mat3x3<f32>(
    uCam.view[0].xyz,
    uCam.view[1].xyz,
    uCam.view[2].xyz
  );

  let T = J * W;
  let cov2dMat = T * cov3d * transpose(T);

  // PlayCanvas Mip-Splatting AA filter & low-pass dilation
  let rawC00 = cov2dMat[0][0];
  let rawC01 = cov2dMat[0][1];
  let rawC11 = cov2dMat[1][1];

  let detOrig = max(0.0, (rawC00 + AA_FILTER) * (rawC11 + AA_FILTER) - rawC01 * rawC01);
  let c00 = rawC00 + LOWPASS_FILTER;
  let c01 = rawC01;
  let c11 = rawC11 + LOWPASS_FILTER;
  let detBlur = max(1e-8, c00 * c11 - c01 * c01);

  let aaFactor = sqrt(detOrig / detBlur);
  let finalAlpha = clamp(rgba.a * aaFactor, 0.0, 0.995);

  // Contribution culling (PlayCanvas SuperSplat rule: alpha * 2*pi*sqrt(det) < minContribution)
  if (finalAlpha * 6.2831853 * sqrt(detBlur) < uCam.minContribution) {
    return;
  }

  // Exact 2x2 symmetric eigendecomposition -> screen-space major/minor axes in pixels
  let mid = 0.5 * (c00 + c11);
  let halfDiff = 0.5 * (c00 - c11);
  let disc = sqrt(max(0.0, halfDiff * halfDiff + c01 * c01));
  let lambda1 = max(0.05, mid + disc);
  let lambda2 = max(0.05, mid - disc);

  let eigVec = vec2<f32>(c01, lambda1 - c00);
  let eigLen = length(eigVec);
  let dir = select(vec2<f32>(1.0, 0.0), eigVec / eigLen, eigLen > 1e-7);

  let maxRad = min(768.0, min(uCam.viewport.x, uCam.viewport.y));
  // 2 * sqrt(2 * lambda) covers exp(-4) = 1.8% Gaussian tail (PlayCanvas tighter quad footprint!)
  var len1 = 2.828427 * sqrt(lambda1);
  var len2 = 2.828427 * sqrt(lambda2);

  if (uCam.renderMode == 3u) {
    // Point-cloud centers inspection mode
    len1 = 2.6;
    len2 = 2.6;
  }

  let radScale = min(1.0, maxRad / max(len1, 1e-4));
  let axis1 = dir * (len1 * radScale);
  let axis2 = vec2<f32>(dir.y, -dir.x) * (len2 * radScale);

  // Exact pixel-extent screen culling
  let extent = abs(axis1) + abs(axis2);
  let centerPx = (ndc * 0.5 + 0.5) * uCam.viewport;
  if (centerPx.x + extent.x < 0.0 || centerPx.x - extent.x > uCam.viewport.x ||
      centerPx.y + extent.y < 0.0 || centerPx.y - extent.y > uCam.viewport.y) {
    return;
  }

  var outRgb = rgba.rgb;
  var outAlpha = finalAlpha;
  if (uCam.renderMode == 2u) {
    let normD = clamp((depth - 0.6) / 3.8, 0.0, 1.0);
    outRgb = turboColormap(1.0 - normD);
    outAlpha = max(finalAlpha, 0.75);
  } else if (uCam.renderMode == 3u) {
    outAlpha = 0.95;
  }

  // Claim compact survivor slot
  let slot = atomicAdd(&survivorCount[0], 1u);

  // Radial squared distance log-sort key (prevents popping during camera rotation)
  let delta = worldPos - uCam.camPos;
  let distSq = dot(delta, delta);
  sortKeys[slot] = logDepthSortKey(distSq);
  sortValues[slot] = slot;

  // Store 16-byte compact projected splat directly indexed by compact slot
  projCache[slot] = vec4<u32>(
    pack2x16float(ndc),
    pack2x16float(axis1),
    pack2x16float(axis2),
    pack4x8unorm(vec4<f32>(outRgb, outAlpha))
  );
}
`;

/**
 * Single-thread GPU indirect argument writer (Zero CPU readback!)
 * Writes:
 *  - indirectDrawArgs: [indexCount = INSTANCE_SPLATS * 6, instanceCount = ceil(count / 128), 0, 0, 0]
 *  - indirectSortDispatch: [ceil(count / 1024), 1, 1]
 *  - sortUniformsBuf: [workgroupCount, elementCount]
 */
export const INDIRECT_ARGS_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> survivorCount: array<u32>;
@group(0) @binding(1) var<storage, read_write> indirectDrawArgs: array<u32>;
@group(0) @binding(2) var<storage, read_write> indirectSortDispatch: array<u32>;
@group(0) @binding(3) var<storage, read_write> sortParams: array<u32>; // [workgroupCount, elementCount, blockSumCount, pad]

@compute @workgroup_size(1)
fn main() {
  let count = survivorCount[0];
  let instCount = (count + ${INSTANCE_SPLATS}u - 1u) / ${INSTANCE_SPLATS}u;

  // DrawIndexedIndirect structure (5 x u32)
  indirectDrawArgs[0] = ${INSTANCE_SPLATS * 6}u;
  indirectDrawArgs[1] = instCount;
  indirectDrawArgs[2] = 0u;
  indirectDrawArgs[3] = 0u;
  indirectDrawArgs[4] = 0u;

  let wgCount = max(1u, (count + ${RADIX_ELTS_PER_WG}u - 1u) / ${RADIX_ELTS_PER_WG}u);
  indirectSortDispatch[0] = wgCount;
  indirectSortDispatch[1] = 1u;
  indirectSortDispatch[2] = 1u;

  sortParams[0] = wgCount;
  sortParams[1] = count;
  sortParams[2] = wgCount * 16u;
  sortParams[3] = 0u;
}
`;

/**
 * PlayCanvas Portable 4-Bit Radix Sort (Histogram + Blelloch Scan + Bitmask Popcount Scatter)
 * Sorts 16-bit log depth keys in just 4 passes with GPU-driven indirect dispatch!
 */
export const RADIX_HISTOGRAM_WGSL = /* wgsl */ `
struct PassUniform {
  bitShift: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> uPass: PassUniform;
@group(0) @binding(1) var<storage, read> sortParams: array<u32>; // [wgCount, elementCount, totalBins, 0]
@group(0) @binding(2) var<storage, read> inputKeys: array<u32>;
@group(0) @binding(3) var<storage, read_write> blockSums: array<u32>;

var<workgroup> histogram: array<atomic<u32>, 16>;

@compute @workgroup_size(${RADIX_WG_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) tid: u32
) {
  let wgId = wid.x;
  let wgCount = sortParams[0];
  let elementCount = sortParams[1];
  let baseIdx = wgId * ${RADIX_ELTS_PER_WG}u;

  if (tid < 16u) {
    atomicStore(&histogram[tid], 0u);
  }
  workgroupBarrier();

  for (var r = 0u; r < ${RADIX_ELTS_PER_THREAD}u; r++) {
    let gid = baseIdx + r * ${RADIX_WG_SIZE}u + tid;
    let valid = gid < elementCount && wgId < wgCount;
    // Adreno-safe unconditional atomicAdd
    let key = select(0u, inputKeys[select(0u, gid, valid)], valid);
    let digit = (key >> uPass.bitShift) & 15u;
    atomicAdd(&histogram[digit], select(0u, 1u, valid));
  }
  workgroupBarrier();

  if (tid < 16u && wgId < wgCount) {
    blockSums[tid * wgCount + wgId] = atomicLoad(&histogram[tid]);
  }
}
`;

/**
 * Hierarchical Parallel Exclusive Prefix Sum over `16 * wgCount` histogram bins
 * Supports up to 16 * 1024 = 16,384 bins -> up to 16.7 Million splats!
 */
export const RADIX_SCAN_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> sortParams: array<u32>; // [wgCount, elementCount, totalBins, 0]
@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;
@group(0) @binding(2) var<storage, read_write> superBlockSums: array<u32>;

const SCAN_THREADS: u32 = 256u;
const ITEMS_PER_BLOCK: u32 = 512u;

var<workgroup> temp: array<u32, 512>;
var<workgroup> superTemp: array<u32, 64>;

@compute @workgroup_size(256)
fn scanBlocks(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) tid: u32
) {
  let totalBins = sortParams[2];
  let blockId = wid.x;
  let base = blockId * ITEMS_PER_BLOCK;
  let ai = tid;
  let bi = tid + SCAN_THREADS;

  temp[ai] = select(0u, blockSums[base + ai], base + ai < totalBins);
  temp[bi] = select(0u, blockSums[base + bi], base + bi < totalBins);

  var offset = 1u;
  for (var d = ITEMS_PER_BLOCK >> 1u; d > 0u; d >>= 1u) {
    workgroupBarrier();
    if (tid < d) {
      let i1 = offset * (2u * tid + 1u) - 1u;
      let i2 = offset * (2u * tid + 2u) - 1u;
      temp[i2] += temp[i1];
    }
    offset <<= 1u;
  }

  if (tid == 0u) {
    superBlockSums[blockId] = temp[ITEMS_PER_BLOCK - 1u];
    temp[ITEMS_PER_BLOCK - 1u] = 0u;
  }

  for (var d = 1u; d < ITEMS_PER_BLOCK; d <<= 1u) {
    offset >>= 1u;
    workgroupBarrier();
    if (tid < d) {
      let i1 = offset * (2u * tid + 1u) - 1u;
      let i2 = offset * (2u * tid + 2u) - 1u;
      let t = temp[i1];
      temp[i1] = temp[i2];
      temp[i2] += t;
    }
  }
  workgroupBarrier();

  if (base + ai < totalBins) {
    blockSums[base + ai] = temp[ai];
  }
  if (base + bi < totalBins) {
    blockSums[base + bi] = temp[bi];
  }
}

@compute @workgroup_size(32)
fn scanSuperBlocks(@builtin(local_invocation_index) tid: u32) {
  let totalBins = sortParams[2];
  let numBlocks = (totalBins + ITEMS_PER_BLOCK - 1u) / ITEMS_PER_BLOCK;

  superTemp[tid * 2u] = select(0u, superBlockSums[tid * 2u], tid * 2u < numBlocks);
  superTemp[tid * 2u + 1u] = select(0u, superBlockSums[tid * 2u + 1u], tid * 2u + 1u < numBlocks);

  var offset = 1u;
  for (var d = 32u; d > 0u; d >>= 1u) {
    workgroupBarrier();
    if (tid < d) {
      let i1 = offset * (2u * tid + 1u) - 1u;
      let i2 = offset * (2u * tid + 2u) - 1u;
      superTemp[i2] += superTemp[i1];
    }
    offset <<= 1u;
  }

  if (tid == 0u) {
    superTemp[63u] = 0u;
  }

  for (var d = 1u; d < 64u; d <<= 1u) {
    offset >>= 1u;
    workgroupBarrier();
    if (tid < d) {
      let i1 = offset * (2u * tid + 1u) - 1u;
      let i2 = offset * (2u * tid + 2u) - 1u;
      let t = superTemp[i1];
      superTemp[i1] = superTemp[i2];
      superTemp[i2] += t;
    }
  }
  workgroupBarrier();

  if (tid * 2u < numBlocks) {
    superBlockSums[tid * 2u] = superTemp[tid * 2u];
  }
  if (tid * 2u + 1u < numBlocks) {
    superBlockSums[tid * 2u + 1u] = superTemp[tid * 2u + 1u];
  }
}

@compute @workgroup_size(256)
fn addSuperBlockSums(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) tid: u32
) {
  let totalBins = sortParams[2];
  let blockId = wid.x;
  if (blockId == 0u) {
    return;
  }
  let base = blockId * ITEMS_PER_BLOCK;
  let addVal = superBlockSums[blockId];
  if (base + tid < totalBins) {
    blockSums[base + tid] += addVal;
  }
  if (base + tid + SCAN_THREADS < totalBins) {
    blockSums[base + tid + SCAN_THREADS] += addVal;
  }
}
`;

export const RADIX_SCATTER_WGSL = /* wgsl */ `
struct PassUniform {
  bitShift: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> uPass: PassUniform;
@group(0) @binding(1) var<storage, read> sortParams: array<u32>;
@group(0) @binding(2) var<storage, read> inputKeys: array<u32>;
@group(0) @binding(3) var<storage, read> inputVals: array<u32>;
@group(0) @binding(4) var<storage, read> prefixBlockSum: array<u32>;
@group(0) @binding(5) var<storage, read_write> outputKeys: array<u32>;
@group(0) @binding(6) var<storage, read_write> outputVals: array<u32>;

// 16 digits * 8 u32 words (256 bits per digit) = 128 atomic u32s
var<workgroup> digitMasks: array<atomic<u32>, 128>;
var<workgroup> digitOffsets: array<u32, 16>;

@compute @workgroup_size(${RADIX_WG_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) tid: u32
) {
  let wgId = wid.x;
  let wgCount = sortParams[0];
  let elementCount = sortParams[1];
  let baseIdx = wgId * ${RADIX_ELTS_PER_WG}u;

  let wordIdx = tid >> 5u;
  let bitIdx = tid & 31u;

  if (tid < 16u) {
    digitOffsets[tid] = 0u;
  }
  if (tid < 128u) {
    atomicStore(&digitMasks[tid], 0u);
  }
  workgroupBarrier();

  for (var round = 0u; round < ${RADIX_ELTS_PER_THREAD}u; round++) {
    let gid = baseIdx + round * ${RADIX_WG_SIZE}u + tid;
    let valid = gid < elementCount && wgId < wgCount;

    let safeGid = select(0u, gid, valid);
    let key = select(0u, inputKeys[safeGid], valid);
    let val = select(0u, inputVals[safeGid], valid);
    let digit = select(0u, (key >> uPass.bitShift) & 15u, valid);

    atomicOr(&digitMasks[digit * 8u + wordIdx], select(0u, 1u << bitIdx, valid));
    workgroupBarrier();

    if (valid) {
      let maskBase = digit * 8u;
      var localRank = digitOffsets[digit];
      for (var w = 0u; w < wordIdx; w++) {
        localRank += countOneBits(atomicLoad(&digitMasks[maskBase + w]));
      }
      localRank += countOneBits(atomicLoad(&digitMasks[maskBase + wordIdx]) & ((1u << bitIdx) - 1u));

      let dstPos = prefixBlockSum[digit * wgCount + wgId] + localRank;
      outputKeys[dstPos] = key;
      outputVals[dstPos] = val;
    }

    if (round < ${RADIX_ELTS_PER_THREAD - 1}u) {
      workgroupBarrier();
      if (tid < 16u) {
        var c = 0u;
        for (var w = 0u; w < 8u; w++) {
          let mIdx = tid * 8u + w;
          c += countOneBits(atomicLoad(&digitMasks[mIdx]));
          atomicStore(&digitMasks[mIdx], 0u);
        }
        digitOffsets[tid] += c;
      }
      workgroupBarrier();
    }
  }
}
`;

/**
 * PlayCanvas 128-Splat Batched Instanced Render Shader
 * Reads sorted indices + 16-byte compact `projCache` entries, unpacks eigenvectors,
 * and evaluates smooth exp(-4 * r^2) Gaussian falloff with optional ellipsoid ring visualization.
 */
export const SPLAT_RENDER_WGSL = /* wgsl */ `
struct RenderUniforms {
  invViewport2: vec2<f32>,  // 2.0 / width, 2.0 / height
  renderMode: u32,          // 0 = Normal, 1 = Ellipsoid Rings, 2 = Depth, 3 = Centers
  pad0: u32,
};

@group(0) @binding(0) var<uniform> uRender: RenderUniforms;
@group(0) @binding(1) var<storage, read> survivorCount: array<u32>;
@group(0) @binding(2) var<storage, read> sortedIndices: array<u32>;
@group(0) @binding(3) var<storage, read> projCache: array<vec4<u32>>;

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) @interpolate(flat) packedRgba: u32,
};

const DISCARD_POS = vec4<f32>(0.0, 0.0, 2.0, 1.0);
const EXP_NEG4: f32 = 0.0183156389; // exp(-4.0)
const INV_NORM: f32 = 1.01865736;   // 1.0 / (1.0 - exp(-4.0))

@vertex
fn vsMain(
  @location(0) cornerAndSubIdx: vec3<f32>,
  @builtin(instance_index) instIdx: u32
) -> VertexOut {
  var out: VertexOut;
  let order = instIdx * ${INSTANCE_SPLATS}u + u32(cornerAndSubIdx.z);
  let visibleCount = survivorCount[0];

  if (order >= visibleCount) {
    out.position = DISCARD_POS;
    out.uv = vec2<f32>(0.0);
    out.packedRgba = 0u;
    return out;
  }

  let cacheSlot = sortedIndices[order];
  let entry = projCache[cacheSlot];

  let ndc = unpack2x16float(entry.x);
  let axis1 = unpack2x16float(entry.y);
  let axis2 = unpack2x16float(entry.z);

  let corner = cornerAndSubIdx.xy; // in [-1, 1]^2
  let pixelOffset = corner.x * axis1 + corner.y * axis2;
  let ndcOffset = pixelOffset * uRender.invViewport2;

  out.position = vec4<f32>(ndc + ndcOffset, 0.5, 1.0);
  out.uv = corner;
  out.packedRgba = entry.w;
  return out;
}

@fragment
fn fsMain(in: VertexOut) -> @location(0) vec4<f32> {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) {
    discard;
  }

  let color = unpack4x8unorm(in.packedRgba);

  if (uRender.renderMode == 3u) {
    // Crisp splat centers dot mode
    let core = smoothstep(1.0, 0.45, r2);
    let a = color.a * core;
    return vec4<f32>(color.rgb * a, a);
  }

  // PlayCanvas truncated Gaussian kernel: (exp(-4 * r^2) - exp(-4)) / (1 - exp(-4))
  let gauss = (exp(-4.0 * r2) - EXP_NEG4) * INV_NORM;
  var alpha = color.a * gauss;
  var rgb = color.rgb;

  if (uRender.renderMode == 1u) {
    // Ellipsoid covariance ring overlay
    let ringBand = smoothstep(0.68, 0.80, r2) * (1.0 - smoothstep(0.88, 1.0, r2));
    alpha = max(alpha * 0.42, ringBand * 0.88);
    rgb = mix(rgb, vec3<f32>(0.0, 0.94, 0.78), ringBand * 0.72);
  }

  if (alpha < 0.0039) {
    discard;
  }

  // Premultiplied alpha output
  return vec4<f32>(rgb * alpha, alpha);
}
`;

/**
 * Reference Ground-Plane Calibration Grid Shader for spatial orientation in the WebGPU viewport
 */
export const GRID_WGSL = /* wgsl */ `
struct GridUniforms {
  viewProj: mat4x4<f32>,
  camPos: vec3<f32>,
  showGrid: f32,
};

@group(0) @binding(0) var<uniform> uGrid: GridUniforms;

struct GridOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) worldXZ: vec2<f32>,
};

@vertex
fn vsGrid(@builtin(vertex_index) vid: u32) -> GridOut {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-3.5, -3.5), vec2<f32>( 3.5, -3.5), vec2<f32>( 3.5,  3.5),
    vec2<f32>(-3.5, -3.5), vec2<f32>( 3.5,  3.5), vec2<f32>(-3.5,  3.5)
  );
  let xz = corners[vid];
  var out: GridOut;
  if (uGrid.showGrid < 0.5) {
    out.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    out.worldXZ = vec2<f32>(0.0);
    return out;
  }
  let world = vec4<f32>(xz.x, -0.46, xz.y, 1.0);
  out.pos = uGrid.viewProj * world;
  out.worldXZ = xz;
  return out;
}

@fragment
fn fsGrid(in: GridOut) -> @location(0) vec4<f32> {
  let coord = in.worldXZ * 4.0;
  let deriv = fwidth(coord);
  let grid = abs(fract(coord - 0.5) - 0.5) / max(deriv, vec2<f32>(1e-4));
  let line = min(grid.x, grid.y);
  let lineAlpha = 1.0 - min(line, 1.0);

  let dist = length(in.worldXZ);
  let ring = 1.0 - clamp(abs(dist - 0.85) / max(fwidth(dist) * 1.5, 0.005), 0.0, 1.0);
  let fade = clamp(1.0 - dist / 3.2, 0.0, 1.0);

  let col = mix(vec3<f32>(0.12, 0.22, 0.30), vec3<f32>(0.0, 0.94, 0.78), ring);
  let a = max(lineAlpha * 0.22, ring * 0.45) * fade * fade;
  if (a < 0.004) {
    discard;
  }
  return vec4<f32>(col * a, a);
}
`;
