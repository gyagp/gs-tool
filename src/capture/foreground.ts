export interface ForegroundResult {
  mask: Uint8Array;
  bounds: { x: number; y: number; width: number; height: number } | null;
  usable: boolean;
}

/** Remove border-connected background; keep the main foreground component. */
export function extractForeground(rgba: Uint8ClampedArray, width: number, height: number): ForegroundResult {
  const total = width * height;
  const mask = new Uint8Array(total);
  const samples: number[][] = [];
  const sample = (x: number, y: number) => {
    const p = (y * width + x) * 4;
    samples.push([rgba[p], rgba[p + 1], rgba[p + 2]]);
  };
  // Include thin backdrop/table transition rows introduced by video compression.
  for (let x = 0; x < width; x++) { sample(x, 0); sample(x, height - 1); }
  for (let y = 0; y < height; y++) { sample(0, y); sample(width - 1, y); }

  // Several border colors handle a backdrop/table seam without averaging it into a false color.
  const centers = [samples[0].slice()];
  for (let k = 1; k < 4; k++) {
    let farthest = samples[0], farthestDistance = 0;
    for (const color of samples) {
      const distance = Math.min(...centers.map(c => color.reduce((sum, value, i) => sum + (value - c[i]) ** 2, 0)));
      if (distance > farthestDistance) { farthestDistance = distance; farthest = color; }
    }
    if (farthestDistance < 24 ** 2) break;
    centers.push(farthest.slice());
  }
  for (let iteration = 0; iteration < 5; iteration++) {
    const sums = centers.map(() => [0, 0, 0, 0]);
    for (const color of samples) {
      let nearest = 0, distance = Infinity;
      centers.forEach((c, i) => {
        const d = color.reduce((sum, value, channel) => sum + (value - c[channel]) ** 2, 0);
        if (d < distance) { distance = d; nearest = i; }
      });
      for (let c = 0; c < 3; c++) sums[nearest][c] += color[c];
      sums[nearest][3]++;
    }
    centers.forEach((center, i) => {
      if (sums[i][3]) for (let c = 0; c < 3; c++) center[c] = sums[i][c] / sums[i][3];
    });
  }

  const background = new Uint8Array(total);
  const queue = new Int32Array(total);
  let head = 0, tail = 0;
  const enqueueBackground = (index: number) => {
    if (background[index]) return;
    const p = index * 4;
    const matches = rgba[p + 3] < 128 || centers.some(c =>
      (rgba[p] - c[0]) ** 2 + (rgba[p + 1] - c[1]) ** 2 + (rgba[p + 2] - c[2]) ** 2 < 38 ** 2
    );
    if (matches) { background[index] = 1; queue[tail++] = index; }
  };
  for (let x = 0; x < width; x++) { enqueueBackground(x); enqueueBackground((height - 1) * width + x); }
  for (let y = 0; y < height; y++) { enqueueBackground(y * width); enqueueBackground(y * width + width - 1); }
  const neighbors = (index: number, visit: (neighbor: number) => void) => {
    const x = index % width, y = Math.floor(index / width);
    if (x > 0) visit(index - 1);
    if (x + 1 < width) visit(index + 1);
    if (y > 0) visit(index - width);
    if (y + 1 < height) visit(index + width);
  };
  while (head < tail) neighbors(queue[head++], enqueueBackground);

  for (let i = 0; i < total; i++) mask[i] = background[i] ? 0 : 255;
  return foregroundFromMask(mask, width, height);
}

/** Validate an observed mask and retain its main component, without inventing a silhouette. */
export function foregroundFromMask(input: Uint8Array, width: number, height: number): ForegroundResult {
  const total = width * height;
  if (input.length !== total) return { mask: new Uint8Array(total), bounds: null, usable: false };
  const mask = new Uint8Array(total);
  const queue = new Int32Array(total);
  let head = 0, tail = 0;
  const neighbors = (index: number, visit: (neighbor: number) => void) => {
    const x = index % width, y = Math.floor(index / width);
    if (x > 0) visit(index - 1);
    if (x + 1 < width) visit(index + 1);
    if (y > 0) visit(index - width);
    if (y + 1 < height) visit(index + width);
  };

  // Small disconnected background patches should not become geometry around the bottle.
  const visited = new Uint8Array(total);
  let best: number[] = [], bestScore = 0;
  for (let i = 0; i < total; i++) {
    if (!input[i] || visited[i]) continue;
    head = 0; tail = 0; queue[tail++] = i; visited[i] = 1;
    let sumX = 0, sumY = 0;
    while (head < tail) {
      const index = queue[head++];
      sumX += index % width; sumY += Math.floor(index / width);
      neighbors(index, neighbor => {
        if (input[neighbor] && !visited[neighbor]) { visited[neighbor] = 1; queue[tail++] = neighbor; }
      });
    }
    const offset = Math.hypot(sumX / tail / width - 0.5, sumY / tail / height - 0.5);
    const score = tail * Math.max(0.1, 1 - offset);
    if (score > bestScore) { bestScore = score; best = Array.from(queue.subarray(0, tail)); }
  }

  let minX = width, minY = height, maxX = -1, maxY = -1;
  for (const index of best) {
    mask[index] = 255;
    const x = index % width, y = Math.floor(index / width);
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  const bounds = best.length ? { x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1 } : null;
  const touchesBorder = minX <= 1 || minY <= 1 || maxX >= width - 2 || maxY >= height - 2;
  const ratio = best.length / total;
  return { mask, bounds, usable: !!bounds && ratio >= 0.015 && ratio <= 0.75 && !touchesBorder };
}
