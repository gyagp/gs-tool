# 3DGS Studio — Guided Multi-Angle Capture & WebGPU Gaussian Splatting

**Live App:** [https://gs.gyagp.workers.dev](https://gs.gyagp.workers.dev)

A bilingual (**中文 / English**), responsive web toolkit for **3D Gaussian Splatting (3DGS)** that guides anyone step-by-step through multi-angle object photography and renders splats directly in the browser via a PlayCanvas-inspired **WebGPU** compute & indirect-draw pipeline.

Reconstruction is an **experimental approximation**, not camera-pose estimation (SfM) or trained 3DGS. Photo mode uses a visual hull; video defaults to a round-bottle fit from measured silhouettes and captured colors. Both assume the guided/video camera directions. Reflections, transparent bottles, and label text may not reconstruct faithfully.

---

## Key Features / 核心特性

### 1. Step-by-Step Photo Coach & Instant Shot Quality Check (引导式环拍与实时照片质检)
- **Clear Multi-Angle Presets**:
  - **8-Shot Preview (Default · 2 Rings)** — Four side views and four raised views for a rough preview of simple objects; 71% fewer photos than the previous 28-shot default. More views may be needed for complex shapes.
  - **Quick Scan (16 Photos · 2 Rings)** — Add detail to the 8-shot plan, reusing all eight photos and taking only eight more.
  - **28-Shot Full Coverage (3 Rings)** — Optional eye-level, 45° overhead, and low undercut rings (~78% view overlap).
  - **Dense Studio (48 Photos · 4 Rings)** — High-density capture for complex geometry.
- **Human-Friendly Instructions**: Converts raw azimuth/elevation degrees into intuitive **clock-face positions** (e.g., *6 o'clock Front*, *3 o'clock Right*) and camera tilt directions.
- **Missing-Angle Compass Checklist**: Real-time 8-direction + overhead (>30°) coverage tracker showing exactly which sectors are covered and which are still missing.
- **Instant Photo Quality Inspector**: Evaluates every new shot for **Sharpness** (Laplacian focus score), **Exposure**, **Subject Framing/Centering**, and **Duplicate Angle Similarity**, telling you immediately if a shot is good or needs a one-tap retake.
- **Responsive Multi-Device Layout**: Adapts across mobile phones (320px+ with pinch-zoom & two-finger pan), tablets, laptops, and 4K monitors.

### Mobile Capture / 手机拍摄

- **从相册选择视频 / Choose Video from Library** is the primary mobile action. It uses a video file picker without a `capture` attribute so existing videos remain selectable. **录制环绕视频 / Record Orbit Video** is a separate action using the native camera capture hint where supported. Fix the bottle in place and walk one full orbit at a steady height, distance, and speed (10–20 seconds recommended; 2–90 seconds accepted). Processing stays on the device. Eight temporal sections provide 48 candidate frames; focus, exposure, foreground quality, and normalized subject-image similarity select up to eight distinct keyframes. Six usable sectors trigger an approximate preview; fewer frames remain available for inspection and manual completion. Stationary or repeated footage is not padded with duplicate frames.
- **Local foreground segmentation:** video keyframe candidates use the small U²-Net saliency model through ONNX Runtime WASM to isolate the subject in cluttered rooms. The model and runtime are self-hosted and loaded only when processing video. The first use needs a network connection to load them; video pixels are never uploaded. Models can still omit caps or select background objects: inspect the cutouts. Round-bottle selection tries alternative frames when the silhouette proportions change unexpectedly. Cancel remains available while loading the model.
- **Round-bottle video fit (default):** compensate image roll and framing changes, combine measured radius profiles robustly across views, and sample captured colors from the closest viewing direction. This produces a smooth surface of revolution and avoids averaging opposing labels together. It assumes a round, upright bottle; switch it off under Capture Settings for other objects. The preview visibly identifies the fitting method. Same-height video does not measure cap/base geometry or hidden recesses, and label seams and texture distortion remain possible.
- Video camera directions assume a uniform full orbit. They are **not recovered camera poses**, and image differences do not prove new viewing angles or a complete orbit. Video import does not add SfM or 3DGS training; same-height footage does not capture overhead views. Cancel processing with the on-screen button. Browser codec support varies; MP4 / WebM are suggested. Existing captures must be explicitly cleared before starting a video session. See [third-party notices](THIRD_PARTY_NOTICES.md) for model/runtime attribution.
- The mobile capture page keeps the viewfinder first and the capture controls fixed above the phone's bottom safe area. Settings, coverage, and shooting tips expand on demand.
- Tap **开启相机拍摄 / Open Camera**, allow camera access, then tap **拍下当前角度 / Take Photo** for each guided angle. Live preview requires HTTPS (or localhost).
- **系统相机拍照 / System Camera** opens the phone's native camera picker where supported; **从相册添加 / Add Photos** imports existing photos. Browsers decide whether to offer a camera or file picker.
- Simulated photos are available separately in **拍摄设置与覆盖进度 / Capture Settings & Coverage**.
- After the initial eight views, **查看 3D 预览 / View 3D Preview** is available above the viewfinder. **补拍至 16 张 / Add Detail to 16 Photos** keeps those eight captures. Other preset changes reuse matching camera angles when the field of view is the same; unmatched captures are not transferred.
- Real photos retain their full aspect ratio (up to 1280px for display; 384px for processing). Inspect **查看抠图 / Check cutout**: only the bottle should remain against the checkerboard. Use a matte solid backdrop contrasting with both the bottle and label. Border-connected background is removed, and the subject is normalized to fit the approximate reconstruction volume.
- Unusable masks and empty geometry produce a visible error while preserving photos. They do not produce replacement ellipses or spheres. Reconstruction uses the same inspected mask shown in the cutout preview. Demo and real captures cannot be combined, and changing captures clears the previous model to avoid displaying a stale result.

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

# Test foreground extraction, bottle geometry, photo sizing, and capture reuse
npm test

# Deploy to Cloudflare Workers (https://gs.gyagp.workers.dev)
npm run deploy
```
