export function getCaptureDimensions(width: number, height: number, longest: number) {
  const scale = Math.min(1, longest / Math.max(width, height));
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
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
