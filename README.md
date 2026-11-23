# 3DGS Studio — Guided Multi-Angle Capture & WebGPU Gaussian Splatting

**Live App:** [https://gs.gyagp.workers.dev](https://gs.gyagp.workers.dev)

A bilingual (**中文 / English**), responsive web toolkit for **3D Gaussian Splatting (3DGS)** that guides anyone step-by-step through multi-angle object photography and renders splats directly in the browser via a PlayCanvas-inspired **WebGPU** compute & indirect-draw pipeline.

---

## Key Features / 核心特性

### 1. Step-by-Step Photo Coach & Instant Shot Quality Check (引导式环拍与实时照片质检)
- **Clear Multi-Angle Presets**:
  - **Quick Scan (16 Photos · 2 Rings)** — Fast tabletop scan.
  - **Balanced 3DGS (28 Photos · 3 Rings)** — Recommended eye-level, 45° overhead, and low undercut rings (~78% view overlap).
  - **Dense Studio (48 Photos · 4 Rings)** — High-density capture for complex geometry.
- **Human-Friendly Instructions**: Converts raw azimuth/elevation degrees into intuitive **clock-face positions** (e.g., *6 o'clock Front*, *3 o'clock Right*) and camera tilt directions.
- **Missing-Angle Compass Checklist**: Real-time 8-direction + overhead (>30°) coverage tracker showing exactly which sectors are covered and which are still missing.
- **Instant Photo Quality Inspector**: Evaluates every new shot for **Sharpness** (Laplacian focus score), **Exposure**, **Subject Framing/Centering**, and **Duplicate Angle Similarity**, telling you immediately if a shot is good or needs a one-tap retake.
- **Responsive Multi-Device Layout**: Adapts seamlessly across mobile phones (375px+ with pinch-zoom & two-finger pan), tablets, laptops, and 4K monitors.

### 2. PlayCanvas-Class WebGPU 3DGS Engine (高性能 WebGPU 渲染引擎)
- **30-bit Morton Z-Order Spatial Sorting** for coherent GPU cache reads.
- **Compact 32B Splat Storage + 16B Projected Cache** (`f16` screen-space covariance eigenvectors + Mip-Splatting AA compensation).
- **Zero CPU-GPU Readback Stalls**: Frustum + contribution culling, 16-bit log-spaced radial depth keys, portable **4-bit GPU Radix Sort** (`countOneBits` bitmask ranking), `dispatchWorkgroupsIndirect`, and 128-splat batched `drawIndexedIndirect`.
- **Motion-Gated Sort**: Automatically caches sorted order when the camera is stationary (`0 ms` sort overhead).
- **Standard `.PLY` & `.SPLAT` Import/Export**: Compatible with PlayCanvas SuperSplat, Blender, Unreal, and Postshot.

---

## Local Development & Deployment

```bash
# Install dependencies
npm install

# Start local dev server
npm run dev

# Typecheck & build production bundle
npm run build

# Deploy to Cloudflare Workers (https://gs.gyagp.workers.dev)
npm run deploy
```
