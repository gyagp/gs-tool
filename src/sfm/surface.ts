export interface TexturedSurfaceData {
  positions: Float32Array;
  normals: Float32Array;
  uvs: Float32Array;
  indices: Uint32Array;
  /** One complete source view per texture. Do not stitch uncertain camera poses. */
  views: SurfaceTextureView[];
}

export interface SurfaceTextureView {
  azimuthRad: number;
  timeSeconds?: number;
  width: number;
  height: number;
  rgba: Uint8ClampedArray;
}

/** Cyclic distance also supports reverse orbits and yaw values outside one turn. */
export function nearestTextureView(views: Pick<SurfaceTextureView, 'azimuthRad'>[], azimuthRad: number): number {
  let best = -1, distance = Infinity;
  views.forEach((view, index) => {
    const delta = Math.abs(Math.atan2(Math.sin(azimuthRad - view.azimuthRad), Math.cos(azimuthRad - view.azimuthRad)));
    if (delta < distance) { distance = delta; best = index; }
  });
  return best;
}
