# Local foreground segmentation

- U²-Net / U²-Net small: Xuebin Qin et al., https://github.com/xuebinqin/U-2-Net (Apache-2.0).
- `public/models/u2netp.onnx`: ONNX conversion distributed by rembg (MIT), https://github.com/danielgatis/rembg/releases/tag/v0.0.0.
  SHA-256: `309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8`.
- ONNX Runtime Web: Microsoft, https://github.com/microsoft/onnxruntime (MIT). Its JavaScript and WASM runtime are copied from the pinned npm dependency during development/build.

The model and runtime are served with the app. Image/video pixels stay in the browser; inference does not call an external service. The model predicts salient foreground and can select an incorrect object or miss details. Inspect the cutouts before relying on the result.

# Imported Gaussian model viewer

- Spark 2.4.0: World Labs Technologies, Inc., https://github.com/sparkjsdev/spark (MIT). Bundled lazily with its local workers/WASM for full Gaussian and spherical harmonics rendering. License: [public/licenses/spark-MIT.txt](public/licenses/spark-MIT.txt).
- Three.js 0.187.0: https://github.com/mrdoob/three.js (MIT). License: [public/licenses/three-MIT.txt](public/licenses/three-MIT.txt).

# Optional local reconstruction tools (not bundled in the website)

- COLMAP / pycolmap 4.2.1: https://github.com/colmap/colmap (BSD-3-Clause; see upstream third-party licenses).
- Brush 0.3.0: Arthur Brussee and contributors, https://github.com/ArthurBrussee/brush (Apache-2.0). Setup downloads the official Windows binary and verifies its SHA-256; it is not redistributed in this repository.
- Python dependencies installed into a separate ignored virtual environment: NumPy (BSD), SciPy (BSD), OpenCV (Apache-2.0), ONNX Runtime (MIT). The local script reuses the attributed U²-Net model above.

All reconstruction media and downloaded tool binaries remain outside the deployment bundle. Local training makes no hosted inference calls.
