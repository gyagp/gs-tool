# Local foreground segmentation

- U²-Net / U²-Net small: Xuebin Qin et al., https://github.com/xuebinqin/U-2-Net (Apache-2.0).
- `public/models/u2netp.onnx`: ONNX conversion distributed by rembg (MIT), https://github.com/danielgatis/rembg/releases/tag/v0.0.0.
  SHA-256: `309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8`.
- ONNX Runtime Web: Microsoft, https://github.com/microsoft/onnxruntime (MIT). Its JavaScript and WASM runtime are copied from the pinned npm dependency during development/build.

The model and runtime are served with the app. Image/video pixels stay in the browser; inference does not call an external service. The model predicts salient foreground and can select an incorrect object or miss details. Inspect the cutouts before relying on the result.
