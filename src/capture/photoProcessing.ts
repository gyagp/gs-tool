export function getCaptureDimensions(width: number, height: number, longest: number) {
  const scale = Math.min(1, longest / Math.max(width, height));
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
}

export interface CaptureTexture {
  width: number;
  height: number;
  rgba: Uint8ClampedArray;
  /** Crop rectangle in the low-resolution work image's coordinates. */
  x: number; y: number; sourceWidth: number; sourceHeight: number;
}

/** Bound per-view RGBA memory as well as edge length for wide 4K phone footage. */
export function getTextureDimensions(width: number, height: number, longest = 1920, maxPixels = 512 * 1024) {
  const scale = Math.min(1, longest / Math.max(width, height), Math.sqrt(maxPixels / (width * height)));
  return { width: Math.max(1, Math.floor(width * scale)), height: Math.max(1, Math.floor(height * scale)) };
}

/** Preserve source pixels for texturing; segmentation stays small and independent. */
export function captureSubjectTexture(
  source: CanvasImageSource, sourceWidth: number, sourceHeight: number,
  workWidth: number, workHeight: number,
  bounds: { x: number; y: number; width: number; height: number },
  longest = 1920
): CaptureTexture {
  const x = Math.max(0, bounds.x - 2), y = Math.max(0, bounds.y - 2);
  const width = Math.min(workWidth, bounds.x + bounds.width + 2) - x;
  const height = Math.min(workHeight, bounds.y + bounds.height + 2) - y;
  const sx = sourceWidth / workWidth, sy = sourceHeight / workHeight;
  const dimensions = getTextureDimensions(width * sx, height * sy, longest);
  const canvas = document.createElement('canvas');
  canvas.width = dimensions.width; canvas.height = dimensions.height;
  const context = canvas.getContext('2d')!;
  context.drawImage(source, x * sx, y * sy, width * sx, height * sy, 0, 0, canvas.width, canvas.height);
  return { ...dimensions, rgba: context.getImageData(0, 0, canvas.width, canvas.height).data,
    x, y, sourceWidth: width, sourceHeight: height };
}

/** Keep the entire phone photo and its aspect ratio; only resize, never center-crop. */
export function prepareCaptureImage(source: CanvasImageSource, width: number, height: number) {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d')!;
  const resize = (longest: number) => {
    const dimensions = getCaptureDimensions(width, height, longest);
    canvas.width = dimensions.width;
    canvas.height = dimensions.height;
    context.drawImage(source, 0, 0, canvas.width, canvas.height);
  };
  resize(1280);
  const dataUrl = canvas.toDataURL('image/jpeg', 0.94);
  resize(384);
  return { dataUrl, width: canvas.width, height: canvas.height, rgba: context.getImageData(0, 0, canvas.width, canvas.height).data };
}
