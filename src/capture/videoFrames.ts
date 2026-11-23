import { extractForeground, ForegroundResult } from './foreground';
import { inspectCapturedPhoto } from './coverage';
import { prepareCaptureImage } from './photoProcessing';

export interface ScoredVideoFrame {
  bin: number;
  time: number;
  score: number;
  fingerprint: Uint8Array;
}

export interface VideoFrame extends ScoredVideoFrame {
  image: ReturnType<typeof prepareCaptureImage>;
}

export class VideoFrameError extends Error {
  constructor(public readonly code: 'decode' | 'duration' | 'no_frames') { super(code); }
}

export function selectDistinctFrame<T extends ScoredVideoFrame>(candidates: T[], selected: ScoredVideoFrame[]): T | null {
  return candidates.slice().sort((a, b) => b.score - a.score || a.time - b.time).find(candidate =>
    selected.every(previous => {
      let difference = 0;
      for (let i = 0; i < candidate.fingerprint.length; i++) difference += Math.abs(candidate.fingerprint[i] - previous.fingerprint[i]);
      return difference / candidate.fingerprint.length >= 3;
    })
  ) ?? null;
}

function fingerprint(image: VideoFrame['image'], foreground: ForegroundResult): Uint8Array {
  const bounds = foreground.bounds!;
  const result = new Uint8Array(16 * 24 * 3);
  for (let y = 0; y < 24; y++) for (let x = 0; x < 16; x++) {
    const sx = Math.min(image.width - 1, Math.floor(bounds.x + (x + 0.5) * bounds.width / 16));
    const sy = Math.min(image.height - 1, Math.floor(bounds.y + (y + 0.5) * bounds.height / 24));
    const index = sy * image.width + sx;
    if (foreground.mask[index]) for (let c = 0; c < 3; c++) result[(y * 16 + x) * 3 + c] = image.rgba[index * 4 + c];
  }
  return result;
}

function mediaEvent(video: HTMLVideoElement, name: string, signal: AbortSignal, start: () => void): Promise<void> {
  return new Promise((resolve, reject) => {
    const finish = (error?: unknown) => {
      clearTimeout(timeout);
      video.removeEventListener(name, onReady);
      video.removeEventListener('error', onError);
      signal.removeEventListener('abort', onAbort);
      error ? reject(error) : resolve();
    };
    const onReady = () => finish();
    const onError = () => finish(new VideoFrameError('decode'));
    const onAbort = () => finish(new DOMException('Cancelled', 'AbortError'));
    const timeout = setTimeout(onError, 15000);
    video.addEventListener(name, onReady, { once: true });
    video.addEventListener('error', onError, { once: true });
    signal.addEventListener('abort', onAbort, { once: true });
    if (signal.aborted) { onAbort(); return; }
    try { start(); } catch { onError(); }
  });
}

/** Local-only temporal sampling: camera angles remain estimates from a uniform full orbit. */
export async function extractVideoFrames(
  file: File,
  signal: AbortSignal,
  onProgress: (completed: number, total: number) => void
): Promise<{ frames: VideoFrame[]; sampled: number; duration: number }> {
  const video = document.createElement('video');
  video.preload = 'auto'; video.muted = true; video.playsInline = true; video.hidden = true;
  document.body.append(video);
  const url = URL.createObjectURL(file);
  const seek = async (time: number) => {
    if (Math.abs(video.currentTime - time) > 0.001 || video.readyState < 2) {
      await mediaEvent(video, 'seeked', signal, () => { video.currentTime = time; });
    }
  };
  try {
    await mediaEvent(video, 'loadedmetadata', signal, () => { video.src = url; video.load(); });
    // Some locally recorded WebM files reveal their duration only after an end seek.
    if (!Number.isFinite(video.duration)) await seek(1e6);
    const duration = video.duration;
    if (!Number.isFinite(duration) || duration < 2 || duration > 90) throw new VideoFrameError('duration');
    const frames: VideoFrame[] = [];
    const sampled = 48;
    for (let bin = 0; bin < 8; bin++) {
      const candidates: VideoFrame[] = [];
      for (let sample = 0; sample < 6; sample++) {
        if (signal.aborted) throw new DOMException('Cancelled', 'AbortError');
        const time = (bin + (sample + 0.5) / 6) * duration / 8;
        await seek(time);
        const image = prepareCaptureImage(video, video.videoWidth, video.videoHeight);
        const foreground = extractForeground(image.rgba, image.width, image.height);
        if (foreground.usable) {
          const quality = inspectCapturedPhoto(image.rgba, image.width, image.height);
          if (quality.verdict !== 'bad') candidates.push({
            bin, time, image, fingerprint: fingerprint(image, foreground),
            score: quality.sharpnessScore * 0.65 + quality.exposureScore * 0.25 + (1 - quality.centerOffset) * 10
          });
        }
        onProgress(bin * 6 + sample + 1, sampled);
      }
      const selected = selectDistinctFrame(candidates, frames);
      if (selected) frames.push(selected);
    }
    if (!frames.length) throw new VideoFrameError('no_frames');
    return { frames, sampled, duration };
  } finally {
    video.pause(); video.removeAttribute('src'); video.load(); video.remove(); URL.revokeObjectURL(url);
  }
}
