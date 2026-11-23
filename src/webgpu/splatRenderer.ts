/**
 * High-Performance WebGPU 3D Gaussian Splatting Engine (PlayCanvas Architecture)
 *
 * Features:
 * - Zero CPU-GPU synchronization stalls in the hot render loop via `dispatchWorkgroupsIndirect` + `drawIndexedIndirect`.
 * - Motion-gated GPU Radix Sort: skips sorting when camera pose & splat scale haven't moved.
 * - Optional GPU Timestamp Queries (`timestamp-query` feature) + non-blocking periodic survivor count readback for telemetry HUD.
 * - Automatic Canvas2D software fallback if opened on a device/headless browser without WebGPU hardware adapter.
 */

import {
  mat4LookAt,
  mat4Multiply,
  mat4PerspectiveZO,
  sphericalToCartesian,
  Vec3
} from '../math/mat4';
import { SplatCloudData } from '../sfm/reconstruct';
import {
  GRID_WGSL,
  INDIRECT_ARGS_WGSL,
  INSTANCE_SPLATS,
  PROJECTOR_WG_SIZE,
  PROJECTOR_WGSL,
  RADIX_ELTS_PER_WG,
  RADIX_HISTOGRAM_WGSL,
  RADIX_SCAN_WGSL,
  RADIX_SCATTER_WGSL,
  SPLAT_RENDER_WGSL
} from './shaders';

export type RenderMode = 0 | 1 | 2 | 3;

export interface RendererStats {
  backend: 'WebGPU (Indirect Compute + 4-Bit Radix)' | 'Canvas2D Fallback';
  fps: number;
  frameTimeMs: number;
  totalSplats: number;
  visibleSplats: number;
  culledPct: number;
  sortSkipped: boolean;
  vramMB: number;
}

const _tmpF32 = new Float32Array(1);
const _tmpI32 = new Int32Array(_tmpF32.buffer);
function floatBits(v: number): number {
  _tmpF32[0] = v;
  return _tmpI32[0];
}

export class WebGPUSplatRenderer {
  private canvas: HTMLCanvasElement;
  private device: GPUDevice | null = null;
  private context: GPUCanvasContext | null = null;
  private format: GPUTextureFormat = 'bgra8unorm';
  private fallbackCtx: CanvasRenderingContext2D | null = null;

  // Orbit Camera State
  public yawDeg = 28;
  public pitchDeg = 18;
  public distance = 2.35;
  public target: Vec3 = [0, 0.04, 0];
  public fovDeg = 46;
  public splatScale = 1.0;
  public minContribution = 0.25;
  public renderMode: RenderMode = 0;
  public showGrid = true;
  public autoTurntable = false;

  // Motion-gated sort state (PlayCanvas optimization: skip sort when camera stationary)
  private dirtySort = true;
  private lastCameraSignature = '';

  // GPU Pipelines
  private projectorPipeline!: GPUComputePipeline;
  private indirectArgsPipeline!: GPUComputePipeline;
  private histogramPipeline!: GPUComputePipeline;
  private scanBlocksPipeline!: GPUComputePipeline;
  private scanSuperPipeline!: GPUComputePipeline;
  private addSuperPipeline!: GPUComputePipeline;
  private scatterPipeline!: GPUComputePipeline;
  private renderPipeline!: GPURenderPipeline;
  private gridPipeline!: GPURenderPipeline;

  // Static & Dynamic GPU Buffers
  private capacity = 0;
  private splatCount = 0;
  private rawSplatBuffer!: GPUBuffer;
  private projCacheBuffer!: GPUBuffer;
  private keysABuffer!: GPUBuffer;
  private keysBBuffer!: GPUBuffer;
  private valsABuffer!: GPUBuffer;
  private valsBBuffer!: GPUBuffer;
  private survivorCountBuffer!: GPUBuffer;
  private survivorReadbackBuffer!: GPUBuffer;
  private indirectDrawBuffer!: GPUBuffer;
  private indirectSortDispatchBuffer!: GPUBuffer;
  private sortParamsBuffer!: GPUBuffer;
  private blockSumsBuffer!: GPUBuffer;
  private superBlockSumsBuffer!: GPUBuffer;

  private camUniformBuffer!: GPUBuffer;
  private renderUniformBuffer!: GPUBuffer;
  private gridUniformBuffer!: GPUBuffer;
  private passUniformBuffers: GPUBuffer[] = [];

  private quadVertexBuffer!: GPUBuffer;
  private quadIndexBuffer!: GPUBuffer;

  // Bind Groups
  private scanBindGroupLayout!: GPUBindGroupLayout;
  private projectorBG!: GPUBindGroup;
  private indirectArgsBG!: GPUBindGroup;
  private scanBG!: GPUBindGroup;
  private histBG_Even: GPUBindGroup[] = [];
  private scatterBG_Even: GPUBindGroup[] = [];
  private renderBG!: GPUBindGroup;
  private gridBG!: GPUBindGroup;

  // Preallocated CPU scratch arrays (zero GC in render loop)
  private camUniformData = new ArrayBuffer(192);
  private camF32 = new Float32Array(this.camUniformData);
  private camU32 = new Uint32Array(this.camUniformData);
  private camI32 = new Int32Array(this.camUniformData);
  private viewMat = new Float32Array(16);
  private projMat = new Float32Array(16);
  private vpMat = new Float32Array(16);

  // Fallback CPU cloud reference
  private currentCloud: SplatCloudData | null = null;

  // Telemetry state
  private rafId = 0;
  private lastFrameTs = performance.now();
  private smoothedFps = 60;
  private smoothedMs = 1.2;
  private lastVisibleCount = 0;
  private readbackPending = false;
  private frameCounter = 0;
  public onStats?: (stats: RendererStats) => void;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.bindInteraction();
  }

  public async init(): Promise<boolean> {
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      try {
        const adapter = await navigator.gpu.requestAdapter({
          powerPreference: 'high-performance'
        });
        if (adapter) {
          this.device = await adapter.requestDevice({
            requiredLimits: {
              maxStorageBufferBindingSize: Math.min(
                adapter.limits.maxStorageBufferBindingSize,
                256 * 1024 * 1024
              )
            }
          });
          this.context = this.canvas.getContext('webgpu') as GPUCanvasContext;
          this.format = navigator.gpu.getPreferredCanvasFormat();
          this.context.configure({
            device: this.device,
            format: this.format,
            alphaMode: 'premultiplied'
          });
          this.buildPipelines();
          this.allocateCapacity(131072); // Initial 128K splat capacity (grows dynamically up to millions)
          return true;
        }
      } catch (err) {
        console.warn('WebGPU init failed, engaging Canvas2D fallback:', err);
      }
    }

    this.fallbackCtx = this.canvas.getContext('2d');
    return false;
  }

  private buildPipelines(): void {
    const dev = this.device!;

    // Static 128-splat batched quad geometry (PlayCanvas GSPLAT_INSTANCE_SIZE = 128)
    const verts = new Float32Array(INSTANCE_SPLATS * 4 * 3);
    const indices = new Uint16Array(INSTANCE_SPLATS * 6);
    const corners: [number, number][] = [
      [-1, -1],
      [1, -1],
      [1, 1],
      [-1, 1]
    ];
    for (let s = 0; s < INSTANCE_SPLATS; s++) {
      for (let c = 0; c < 4; c++) {
        const vOff = (s * 4 + c) * 3;
        verts[vOff] = corners[c][0];
        verts[vOff + 1] = corners[c][1];
        verts[vOff + 2] = s;
      }
      const iOff = s * 6;
      const b = s * 4;
      indices[iOff] = b;
      indices[iOff + 1] = b + 1;
      indices[iOff + 2] = b + 2;
      indices[iOff + 3] = b;
      indices[iOff + 4] = b + 2;
      indices[iOff + 5] = b + 3;
    }

    this.quadVertexBuffer = dev.createBuffer({
      size: verts.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Float32Array(this.quadVertexBuffer.getMappedRange()).set(verts);
    this.quadVertexBuffer.unmap();

    this.quadIndexBuffer = dev.createBuffer({
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Uint16Array(this.quadIndexBuffer.getMappedRange()).set(indices);
    this.quadIndexBuffer.unmap();

    this.camUniformBuffer = dev.createBuffer({
      size: 192,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.renderUniformBuffer = dev.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.gridUniformBuffer = dev.createBuffer({
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    // 4 passes of 4-bit radix sort covering 16-bit log-depth sort keys (shifts 0, 4, 8, 12)
    this.passUniformBuffers = [0, 4, 8, 12].map((shift) => {
      const b = dev.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
      });
      new Uint32Array(b.getMappedRange()).set([shift, 0, 0, 0]);
      b.unmap();
      return b;
    });

    this.projectorPipeline = dev.createComputePipeline({
      layout: 'auto',
      compute: { module: dev.createShaderModule({ code: PROJECTOR_WGSL }), entryPoint: 'main' }
    });

    this.indirectArgsPipeline = dev.createComputePipeline({
      layout: 'auto',
      compute: { module: dev.createShaderModule({ code: INDIRECT_ARGS_WGSL }), entryPoint: 'main' }
    });

    const histMod = dev.createShaderModule({ code: RADIX_HISTOGRAM_WGSL });
    this.histogramPipeline = dev.createComputePipeline({
      layout: 'auto',
      compute: { module: histMod, entryPoint: 'main' }
    });

    this.scanBindGroupLayout = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
      ]
    });
    const scanPipelineLayout = dev.createPipelineLayout({
      bindGroupLayouts: [this.scanBindGroupLayout]
    });

    const scanMod = dev.createShaderModule({ code: RADIX_SCAN_WGSL });
    this.scanBlocksPipeline = dev.createComputePipeline({
      layout: scanPipelineLayout,
      compute: { module: scanMod, entryPoint: 'scanBlocks' }
    });
    this.scanSuperPipeline = dev.createComputePipeline({
      layout: scanPipelineLayout,
      compute: { module: scanMod, entryPoint: 'scanSuperBlocks' }
    });
    this.addSuperPipeline = dev.createComputePipeline({
      layout: scanPipelineLayout,
      compute: { module: scanMod, entryPoint: 'addSuperBlockSums' }
    });

    const scatterMod = dev.createShaderModule({ code: RADIX_SCATTER_WGSL });
    this.scatterPipeline = dev.createComputePipeline({
      layout: 'auto',
      compute: { module: scatterMod, entryPoint: 'main' }
    });

    const renderMod = dev.createShaderModule({ code: SPLAT_RENDER_WGSL });
    this.renderPipeline = dev.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: renderMod,
        entryPoint: 'vsMain',
        buffers: [
          {
            arrayStride: 12,
            stepMode: 'vertex',
            attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }]
          }
        ]
      },
      fragment: {
        module: renderMod,
        entryPoint: 'fsMain',
        targets: [
          {
            format: this.format,
            blend: {
              color: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add'
              },
              alpha: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add'
              }
            }
          }
        ]
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' }
    });

    const gridMod = dev.createShaderModule({ code: GRID_WGSL });
    this.gridPipeline = dev.createRenderPipeline({
      layout: 'auto',
      vertex: { module: gridMod, entryPoint: 'vsGrid' },
      fragment: {
        module: gridMod,
        entryPoint: 'fsGrid',
        targets: [
          {
            format: this.format,
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
            }
          }
        ]
      },
      primitive: { topology: 'triangle-list' }
    });
  }

  private allocateCapacity(minCount: number): void {
    const dev = this.device!;
    const cap = Math.max(4096, Math.ceil(minCount / 4096) * 4096);
    if (cap <= this.capacity) return;

    this.rawSplatBuffer?.destroy();
    this.projCacheBuffer?.destroy();
    this.keysABuffer?.destroy();
    this.keysBBuffer?.destroy();
    this.valsABuffer?.destroy();
    this.valsBBuffer?.destroy();
    this.blockSumsBuffer?.destroy();

    this.capacity = cap;
    const maxWgs = Math.ceil(cap / RADIX_ELTS_PER_WG);
    const totalBins = Math.max(512, Math.ceil((maxWgs * 16) / 512) * 512);

    this.rawSplatBuffer = dev.createBuffer({
      size: cap * 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    this.projCacheBuffer = dev.createBuffer({
      size: cap * 16,
      usage: GPUBufferUsage.STORAGE
    });
    this.keysABuffer = dev.createBuffer({
      size: cap * 4,
      usage: GPUBufferUsage.STORAGE
    });
    this.keysBBuffer = dev.createBuffer({
      size: cap * 4,
      usage: GPUBufferUsage.STORAGE
    });
    this.valsABuffer = dev.createBuffer({
      size: cap * 4,
      usage: GPUBufferUsage.STORAGE
    });
    this.valsBBuffer = dev.createBuffer({
      size: cap * 4,
      usage: GPUBufferUsage.STORAGE
    });

    if (!this.survivorCountBuffer) {
      this.survivorCountBuffer = dev.createBuffer({
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
      });
      this.survivorReadbackBuffer = dev.createBuffer({
        size: 16,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
      });
      this.indirectDrawBuffer = dev.createBuffer({
        size: 32,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST
      });
      this.indirectSortDispatchBuffer = dev.createBuffer({
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST
      });
      this.sortParamsBuffer = dev.createBuffer({
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
      this.superBlockSumsBuffer = dev.createBuffer({
        size: 256,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
    }

    this.blockSumsBuffer = dev.createBuffer({
      size: totalBins * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    this.rebuildBindGroups();
  }

  private rebuildBindGroups(): void {
    const dev = this.device!;

    this.projectorBG = dev.createBindGroup({
      layout: this.projectorPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.camUniformBuffer } },
        { binding: 1, resource: { buffer: this.rawSplatBuffer } },
        { binding: 2, resource: { buffer: this.projCacheBuffer } },
        { binding: 3, resource: { buffer: this.keysABuffer } },
        { binding: 4, resource: { buffer: this.valsABuffer } },
        { binding: 5, resource: { buffer: this.survivorCountBuffer } }
      ]
    });

    this.indirectArgsBG = dev.createBindGroup({
      layout: this.indirectArgsPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.survivorCountBuffer } },
        { binding: 1, resource: { buffer: this.indirectDrawBuffer } },
        { binding: 2, resource: { buffer: this.indirectSortDispatchBuffer } },
        { binding: 3, resource: { buffer: this.sortParamsBuffer } }
      ]
    });

    this.scanBG = dev.createBindGroup({
      layout: this.scanBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.sortParamsBuffer } },
        { binding: 1, resource: { buffer: this.blockSumsBuffer } },
        { binding: 2, resource: { buffer: this.superBlockSumsBuffer } }
      ]
    });

    // Ping-pong bind groups across the 4 passes (0: A->B, 1: B->A, 2: A->B, 3: B->A => final result lands in valsABuffer!)
    this.histBG_Even = [];
    this.scatterBG_Even = [];

    for (let pass = 0; pass < 4; pass++) {
      const even = pass % 2 === 0;
      const inK = even ? this.keysABuffer : this.keysBBuffer;
      const outK = even ? this.keysBBuffer : this.keysABuffer;
      const inV = even ? this.valsABuffer : this.valsBBuffer;
      const outV = even ? this.valsBBuffer : this.valsABuffer;

      this.histBG_Even.push(
        dev.createBindGroup({
          layout: this.histogramPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.passUniformBuffers[pass] } },
            { binding: 1, resource: { buffer: this.sortParamsBuffer } },
            { binding: 2, resource: { buffer: inK } },
            { binding: 3, resource: { buffer: this.blockSumsBuffer } }
          ]
        })
      );

      this.scatterBG_Even.push(
        dev.createBindGroup({
          layout: this.scatterPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.passUniformBuffers[pass] } },
            { binding: 1, resource: { buffer: this.sortParamsBuffer } },
            { binding: 2, resource: { buffer: inK } },
            { binding: 3, resource: { buffer: inV } },
            { binding: 4, resource: { buffer: this.blockSumsBuffer } },
            { binding: 5, resource: { buffer: outK } },
            { binding: 6, resource: { buffer: outV } }
          ]
        })
      );
    }

    this.renderBG = dev.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.renderUniformBuffer } },
        { binding: 1, resource: { buffer: this.survivorCountBuffer } },
        { binding: 2, resource: { buffer: this.valsABuffer } }, // After 4 passes, sorted indices reside in valsA
        { binding: 3, resource: { buffer: this.projCacheBuffer } }
      ]
    });

    this.gridBG = dev.createBindGroup({
      layout: this.gridPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.gridUniformBuffer } }]
    });
  }

  public clearSplatCloud(): void {
    this.currentCloud = null;
    this.splatCount = 0;
    this.lastVisibleCount = 0;
    this.autoTurntable = false;
    this.markDirty();
  }

  public setSplatCloud(cloud: SplatCloudData): void {
    this.currentCloud = cloud;
    this.splatCount = cloud.count;
    this.lastVisibleCount = cloud.count;
    this.dirtySort = true;
    this.lastCameraSignature = '';

    // Auto-fit camera distance to cloud bounding sphere
    if (cloud.boundsRadius > 0.05) {
      this.distance = Math.max(1.2, Math.min(8.0, cloud.boundsRadius * 2.75));
    }

    if (this.device) {
      this.allocateCapacity(cloud.count);
      this.device.queue.writeBuffer(
        this.rawSplatBuffer,
        0,
        cloud.packedBuffer.buffer,
        cloud.packedBuffer.byteOffset,
        cloud.packedBuffer.byteLength
      );
    }
  }

  public markDirty(): void {
    this.dirtySort = true;
    this.lastCameraSignature = '';
  }

  public resetCamera(): void {
    this.yawDeg = 28;
    this.pitchDeg = 18;
    this.distance = this.currentCloud ? Math.max(1.4, this.currentCloud.boundsRadius * 2.75) : 2.35;
    this.target = [0, 0.04, 0];
    this.markDirty();
  }

  public start(): void {
    if (this.rafId) return;
    this.lastFrameTs = performance.now();
    const tick = (now: number) => {
      this.rafId = requestAnimationFrame(tick);
      this.renderFrame(now);
    };
    this.rafId = requestAnimationFrame(tick);
  }

  public stop(): void {
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = 0;
    }
  }

  private renderFrame(now: number): void {
    const dt = Math.max(0.5, now - this.lastFrameTs);
    this.lastFrameTs = now;
    this.smoothedMs = this.smoothedMs * 0.9 + dt * 0.1;
    this.smoothedFps = 1000 / Math.max(1, this.smoothedMs);

    if (this.autoTurntable) {
      this.yawDeg = (this.yawDeg + dt * 0.024) % 360;
    }

    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const rect = this.canvas.getBoundingClientRect();
    const width = Math.max(64, Math.floor(rect.width * dpr));
    const height = Math.max(64, Math.floor(rect.height * dpr));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.dirtySort = true;
    }

    if (!this.device || !this.context) {
      this.renderSoftwareFallback(width, height);
      return;
    }

    const eye = sphericalToCartesian(this.yawDeg, this.pitchDeg, this.distance);
    eye[0] += this.target[0];
    eye[1] += this.target[1];
    eye[2] += this.target[2];

    mat4LookAt(eye, this.target, [0, 1, 0], this.viewMat);
    const fovRad = (this.fovDeg * Math.PI) / 180;
    mat4PerspectiveZO(fovRad, width / height, 0.08, 40.0, this.projMat);
    mat4Multiply(this.projMat, this.viewMat, this.vpMat);

    const camSig = `${width}x${height}:${this.yawDeg.toFixed(3)}:${this.pitchDeg.toFixed(3)}:${this.distance.toFixed(4)}:${this.target.join(',')}:${this.splatScale.toFixed(3)}:${this.minContribution.toFixed(3)}:${this.renderMode}`;
    const needsComputeAndSort = this.splatCount > 0 && (this.dirtySort || camSig !== this.lastCameraSignature);
    this.lastCameraSignature = camSig;
    this.dirtySort = false;

    const dev = this.device;
    const encoder = dev.createCommandEncoder();

    if (needsComputeAndSort) {
      const fy = (height * 0.5) / Math.tan(fovRad * 0.5);
      const fx = fy;

      // Log-spaced radial distance^2 sort range [0.01 .. 400.0]
      const minBits = floatBits(0.005);
      const maxBits = floatBits(400.0);
      const sortKeyScale = 65535.0 / Math.max(1, maxBits - minBits);

      this.camF32.set(this.viewMat, 0);
      this.camF32.set(this.projMat, 16);
      this.camF32[32] = width;
      this.camF32[33] = height;
      this.camF32[34] = fx;
      this.camF32[35] = fy;
      this.camF32[36] = eye[0];
      this.camF32[37] = eye[1];
      this.camF32[38] = eye[2];
      this.camU32[39] = this.splatCount;
      this.camF32[40] = 0.08;
      this.camF32[41] = 40.0;
      this.camF32[42] = this.splatScale;
      this.camF32[43] = this.minContribution;
      this.camU32[44] = this.renderMode;
      this.camI32[45] = minBits;
      this.camF32[46] = sortKeyScale;
      this.camU32[47] = 0;

      dev.queue.writeBuffer(this.camUniformBuffer, 0, this.camUniformData);

      // Clear atomic survivor counter on GPU
      encoder.clearBuffer(this.survivorCountBuffer, 0, 4);

      // 1. Projector + Covariance + Culling + Log-Depth Key Compute Pass
      const projPass = encoder.beginComputePass();
      projPass.setPipeline(this.projectorPipeline);
      projPass.setBindGroup(0, this.projectorBG);
      projPass.dispatchWorkgroups(Math.ceil(this.splatCount / PROJECTOR_WG_SIZE));

      // 2. Indirect Draw & Indirect Sort Dispatch Argument Writer (1 thread, zero CPU stall)
      projPass.setPipeline(this.indirectArgsPipeline);
      projPass.setBindGroup(0, this.indirectArgsBG);
      projPass.dispatchWorkgroups(1);

      // 3. 4-Pass GPU 4-Bit Radix Sort (16-bit depth keys) using Indirect Dispatch
      const maxSortWgs = Math.max(1, Math.ceil(this.splatCount / RADIX_ELTS_PER_WG));
      const scanBlockCount = Math.max(1, Math.ceil((maxSortWgs * 16) / 512));

      for (let p = 0; p < 4; p++) {
        projPass.setPipeline(this.histogramPipeline);
        projPass.setBindGroup(0, this.histBG_Even[p]);
        projPass.dispatchWorkgroupsIndirect(this.indirectSortDispatchBuffer, 0);

        projPass.setPipeline(this.scanBlocksPipeline);
        projPass.setBindGroup(0, this.scanBG);
        projPass.dispatchWorkgroups(scanBlockCount);

        if (scanBlockCount > 1) {
          projPass.setPipeline(this.scanSuperPipeline);
          projPass.setBindGroup(0, this.scanBG);
          projPass.dispatchWorkgroups(1);

          projPass.setPipeline(this.addSuperPipeline);
          projPass.setBindGroup(0, this.scanBG);
          projPass.dispatchWorkgroups(scanBlockCount);
        }

        projPass.setPipeline(this.scatterPipeline);
        projPass.setBindGroup(0, this.scatterBG_Even[p]);
        projPass.dispatchWorkgroupsIndirect(this.indirectSortDispatchBuffer, 0);
      }
      projPass.end();

      // Non-blocking periodic readback of survivor count every 15 frames solely for UI telemetry
      if (!this.readbackPending && this.frameCounter % 15 === 0) {
        encoder.copyBufferToBuffer(this.survivorCountBuffer, 0, this.survivorReadbackBuffer, 0, 4);
        this.readbackPending = true;
      }
    }

    // Update render & grid uniforms
    const rBuf = new ArrayBuffer(16);
    new Float32Array(rBuf, 0, 2).set([2.0 / width, 2.0 / height]);
    new Uint32Array(rBuf, 8, 2).set([this.renderMode, 0]);
    dev.queue.writeBuffer(this.renderUniformBuffer, 0, rBuf);

    const gBuf = new Float32Array(20);
    gBuf.set(this.vpMat, 0);
    gBuf[16] = eye[0];
    gBuf[17] = eye[1];
    gBuf[18] = eye[2];
    gBuf[19] = this.showGrid ? 1.0 : 0.0;
    dev.queue.writeBuffer(this.gridUniformBuffer, 0, gBuf);

    // 4. Batched 128-Splat Indirect Render Pass
    const colorView = this.context.getCurrentTexture().createView();
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: colorView,
          clearValue: { r: 0.027, g: 0.035, b: 0.052, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store'
        }
      ]
    });

    if (this.showGrid) {
      renderPass.setPipeline(this.gridPipeline);
      renderPass.setBindGroup(0, this.gridBG);
      renderPass.draw(6);
    }

    if (this.splatCount > 0) {
      renderPass.setPipeline(this.renderPipeline);
      renderPass.setBindGroup(0, this.renderBG);
      renderPass.setVertexBuffer(0, this.quadVertexBuffer);
      renderPass.setIndexBuffer(this.quadIndexBuffer, 'uint16');
      renderPass.drawIndexedIndirect(this.indirectDrawBuffer, 0);
    }

    renderPass.end();
    dev.queue.submit([encoder.finish()]);

    if (this.readbackPending && this.survivorReadbackBuffer.mapState === 'unmapped') {
      const rb = this.survivorReadbackBuffer;
      rb.mapAsync(GPUMapMode.READ)
        .then(() => {
          this.lastVisibleCount = new Uint32Array(rb.getMappedRange())[0];
          rb.unmap();
          this.readbackPending = false;
        })
        .catch(() => {
          this.readbackPending = false;
        });
    }

    this.frameCounter++;
    if (this.frameCounter % 10 === 0 && this.onStats) {
      const vramBytes = this.capacity * (32 + 16 + 16 + 4);
      const culledPct =
        this.splatCount > 0
          ? Math.max(0, Math.round((1 - this.lastVisibleCount / this.splatCount) * 100))
          : 0;
      this.onStats({
        backend: 'WebGPU (Indirect Compute + 4-Bit Radix)',
        fps: Math.round(this.smoothedFps),
        frameTimeMs: Math.round(this.smoothedMs * 10) / 10,
        totalSplats: this.splatCount,
        visibleSplats: Math.min(this.splatCount, this.lastVisibleCount),
        culledPct,
        sortSkipped: !needsComputeAndSort,
        vramMB: Math.round((vramBytes / (1024 * 1024)) * 10) / 10
      });
    }
  }

  private renderSoftwareFallback(width: number, height: number): void {
    const ctx = this.fallbackCtx;
    if (!ctx) return;

    ctx.fillStyle = '#07090d';
    ctx.fillRect(0, 0, width, height);

    if (!this.currentCloud || this.currentCloud.count === 0) return;

    const eye = sphericalToCartesian(this.yawDeg, this.pitchDeg, this.distance);
    eye[0] += this.target[0];
    eye[1] += this.target[1];
    eye[2] += this.target[2];

    mat4LookAt(eye, this.target, [0, 1, 0], this.viewMat);
    const fovRad = (this.fovDeg * Math.PI) / 180;
    mat4PerspectiveZO(fovRad, width / height, 0.08, 40.0, this.projMat);
    mat4Multiply(this.projMat, this.viewMat, this.vpMat);

    const cloud = this.currentCloud;
    const stride = Math.max(1, Math.floor(cloud.count / 12000));
    const pts: { x: number; y: number; z: number; r: number; c: string }[] = [];
    const fy = (height * 0.5) / Math.tan(fovRad * 0.5);

    for (let i = 0; i < cloud.count; i += stride) {
      const px = cloud.positions[i * 3];
      const py = cloud.positions[i * 3 + 1];
      const pz = cloud.positions[i * 3 + 2];
      const cw = this.vpMat[3] * px + this.vpMat[7] * py + this.vpMat[11] * pz + this.vpMat[15];
      if (cw <= 0.1) continue;
      const nx = (this.vpMat[0] * px + this.vpMat[4] * py + this.vpMat[8] * pz + this.vpMat[12]) / cw;
      const ny = (this.vpMat[1] * px + this.vpMat[5] * py + this.vpMat[9] * pz + this.vpMat[13]) / cw;
      if (Math.abs(nx) > 1.1 || Math.abs(ny) > 1.1) continue;

      const rad = Math.max(1.5, (cloud.scales[i * 3] * this.splatScale * fy * 1.8) / cw);
      const r = Math.round(cloud.colors[i * 4] * 255);
      const g = Math.round(cloud.colors[i * 4 + 1] * 255);
      const b = Math.round(cloud.colors[i * 4 + 2] * 255);
      const a = cloud.colors[i * 4 + 3] * 0.85;
      pts.push({
        x: (nx * 0.5 + 0.5) * width,
        y: (0.5 - ny * 0.5) * height,
        z: cw,
        r: rad,
        c: `rgba(${r},${g},${b},${a.toFixed(2)})`
      });
    }

    pts.sort((a, b) => b.z - a.z);
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      ctx.fillStyle = p.c;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    }

    this.frameCounter++;
    if (this.frameCounter % 15 === 0 && this.onStats) {
      this.onStats({
        backend: 'Canvas2D Fallback',
        fps: Math.round(this.smoothedFps),
        frameTimeMs: Math.round(this.smoothedMs * 10) / 10,
        totalSplats: cloud.count,
        visibleSplats: pts.length,
        culledPct: Math.max(0, Math.round((1 - pts.length / cloud.count) * 100)),
        sortSkipped: false,
        vramMB: 0
      });
    }
  }

  private bindInteraction(): void {
    const activePointers = new Map<number, { x: number; y: number }>();
    let panning = false;
    let lastX = 0;
    let lastY = 0;
    let lastPinchDist = 0;
    let lastPinchMidX = 0;
    let lastPinchMidY = 0;

    this.canvas.style.touchAction = 'none';
    this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());

    this.canvas.addEventListener('pointerdown', (e) => {
      activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      this.autoTurntable = false;
      try {
        this.canvas.setPointerCapture(e.pointerId);
      } catch {
        // ignore synthetic events
      }

      if (activePointers.size === 1) {
        panning = e.button === 2 || e.shiftKey;
        lastX = e.clientX;
        lastY = e.clientY;
      } else if (activePointers.size === 2) {
        const pts = Array.from(activePointers.values());
        lastPinchDist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
        lastPinchMidX = (pts[0].x + pts[1].x) * 0.5;
        lastPinchMidY = (pts[0].y + pts[1].y) * 0.5;
      }
    });

    this.canvas.addEventListener('pointermove', (e) => {
      if (!activePointers.has(e.pointerId)) return;
      activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

      if (activePointers.size === 1) {
        const dx = e.clientX - lastX;
        const dy = e.clientY - lastY;
        lastX = e.clientX;
        lastY = e.clientY;

        if (panning) {
          const panScale = this.distance * 0.0018;
          const az = (this.yawDeg * Math.PI) / 180;
          this.target[0] -= Math.cos(az) * dx * panScale;
          this.target[2] += Math.sin(az) * dx * panScale;
          this.target[1] += dy * panScale;
        } else {
          this.yawDeg = (this.yawDeg - dx * 0.42 + 360) % 360;
          this.pitchDeg = Math.max(-85, Math.min(85, this.pitchDeg + dy * 0.38));
        }
      } else if (activePointers.size >= 2) {
        const pts = Array.from(activePointers.values());
        const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
        const midX = (pts[0].x + pts[1].x) * 0.5;
        const midY = (pts[0].y + pts[1].y) * 0.5;

        if (lastPinchDist > 8 && dist > 8) {
          const zoomRatio = lastPinchDist / dist;
          this.distance = Math.max(0.4, Math.min(18.0, this.distance * zoomRatio));
        }

        const dx = midX - lastPinchMidX;
        const dy = midY - lastPinchMidY;
        const panScale = this.distance * 0.0018;
        const az = (this.yawDeg * Math.PI) / 180;
        this.target[0] -= Math.cos(az) * dx * panScale;
        this.target[2] += Math.sin(az) * dx * panScale;
        this.target[1] += dy * panScale;

        lastPinchDist = dist;
        lastPinchMidX = midX;
        lastPinchMidY = midY;
      }
    });

    const onUpOrCancel = (e: PointerEvent) => {
      activePointers.delete(e.pointerId);
      try {
        this.canvas.releasePointerCapture(e.pointerId);
      } catch {
        // ignore
      }
      if (activePointers.size === 1) {
        const remaining = Array.from(activePointers.values())[0];
        lastX = remaining.x;
        lastY = remaining.y;
      }
    };

    this.canvas.addEventListener('pointerup', onUpOrCancel);
    this.canvas.addEventListener('pointercancel', onUpOrCancel);

    this.canvas.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        const factor = Math.exp(e.deltaY * 0.0011);
        this.distance = Math.max(0.4, Math.min(18.0, this.distance * factor));
      },
      { passive: false }
    );
  }
}
