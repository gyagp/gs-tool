import type { CapturedPhoto } from './coverage';

/** Preserve full-resolution colors independently of the small foreground/geometry mask. */
export function sampleCaptureColor(photo: Pick<CapturedPhoto, 'texture' | 'rgba' | 'workWidth' | 'workHeight'>, x: number, y: number): number[] {
  const texture = photo.texture;
  if (texture) {
    x = (x - texture.x + .5) / texture.sourceWidth * texture.width - .5;
    y = (y - texture.y + .5) / texture.sourceHeight * texture.height - .5;
  }
  const width = texture?.width ?? photo.workWidth, height = texture?.height ?? photo.workHeight;
  const rgba = texture?.rgba ?? photo.rgba;
  x = Math.max(0, Math.min(width - 1, x)); y = Math.max(0, Math.min(height - 1, y));
  const ix = Math.floor(x), iy = Math.floor(y), jx = Math.min(width - 1, ix + 1), jy = Math.min(height - 1, iy + 1), fx = x - ix, fy = y - iy;
  return [0, 1, 2].map(c => {
    const top = rgba[(iy * width + ix) * 4 + c] * (1 - fx) + rgba[(iy * width + jx) * 4 + c] * fx;
    const bottom = rgba[(jy * width + ix) * 4 + c] * (1 - fx) + rgba[(jy * width + jx) * 4 + c] * fx;
    return (top * (1 - fy) + bottom * fy) / 255;
  });
}
