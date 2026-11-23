/**
 * Interactive 2D/3D Geodesic Capture Dome Visualizer
 * Renders orbital camera rings, frustums, captured vs missing stations, blind-spot sectors,
 * and allows clicking/dragging to inspect or select any capture angle.
 */

import { CaptureSlot, CAPTURE_RADIUS } from './coverage';
import { mat4LookAt, mat4Multiply, mat4PerspectiveZO, sphericalToCartesian } from '../math/mat4';
import { Locale } from '../i18n/translations';

export class CaptureDomeVisualizer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private slots: CaptureSlot[] = [];
  private selectedIndex = 0;
  private recommendedIndex = -1;
  private locale: Locale = 'zh';
  private orbitYaw = 28;
  private orbitPitch = 24;
  private isDragging = false;
  private dragMoved = false;
  private lastX = 0;
  private lastY = 0;
  private projectedScreenCoords: { x: number; y: number; z: number; idx: number }[] = [];
  public onSelectSlot?: (slotIndex: number) => void;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.canvas.style.touchAction = 'none';
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D canvas context unavailable');
    this.ctx = ctx;
    this.bindEvents();
  }

  public setState(slots: CaptureSlot[], selectedIndex: number, recommendedIndex: number, locale: Locale = 'zh'): void {
    this.slots = slots;
    this.selectedIndex = selectedIndex;
    this.recommendedIndex = recommendedIndex;
    this.locale = locale;
    this.render();
  }

  public focusAngle(azimuthDeg: number, elevationDeg: number): void {
    this.orbitYaw = azimuthDeg + 18;
    this.orbitPitch = Math.max(-20, Math.min(68, elevationDeg + 14));
    this.render();
  }

  private bindEvents(): void {
    this.canvas.addEventListener('pointerdown', (e) => {
      this.isDragging = true;
      this.dragMoved = false;
      this.lastX = e.clientX;
      this.lastY = e.clientY;
      this.canvas.setPointerCapture(e.pointerId);
    });

    this.canvas.addEventListener('pointermove', (e) => {
      if (!this.isDragging) return;
      const dx = e.clientX - this.lastX;
      const dy = e.clientY - this.lastY;
      if (Math.hypot(dx, dy) > 3) this.dragMoved = true;
      this.orbitYaw = (this.orbitYaw - dx * 0.45 + 360) % 360;
      this.orbitPitch = Math.max(-25, Math.min(78, this.orbitPitch + dy * 0.35));
      this.lastX = e.clientX;
      this.lastY = e.clientY;
      this.render();
    });

    this.canvas.addEventListener('pointerup', (e) => {
      if (!this.isDragging) return;
      this.isDragging = false;
      this.canvas.releasePointerCapture(e.pointerId);

      if (!this.dragMoved) {
        const rect = this.canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const px = (e.clientX - rect.left) * dpr;
        const py = (e.clientY - rect.top) * dpr;

        let bestIdx = -1;
        let bestDist = 26 * dpr;
        for (const pt of this.projectedScreenCoords) {
          const d = Math.hypot(pt.x - px, pt.y - py);
          if (d < bestDist) {
            bestDist = d;
            bestIdx = pt.idx;
          }
        }
        if (bestIdx >= 0 && this.onSelectSlot) {
          this.onSelectSlot(bestIdx);
        }
      }
    });
  }

  public render(): void {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(240, Math.floor(rect.width * dpr));
    const h = Math.max(200, Math.floor(rect.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }

    const ctx = this.ctx;
    ctx.clearRect(0, 0, w, h);

    // Subtle radial optical grid backdrop
    const grad = ctx.createRadialGradient(w * 0.5, h * 0.52, 10, w * 0.5, h * 0.52, w * 0.55);
    grad.addColorStop(0, 'rgba(0, 240, 200, 0.08)');
    grad.addColorStop(0.6, 'rgba(12, 18, 28, 0.45)');
    grad.addColorStop(1, 'rgba(7, 9, 13, 0.92)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    const eye = sphericalToCartesian(this.orbitYaw, this.orbitPitch, 6.4);
    const view = mat4LookAt(eye, [0, 0.1, 0], [0, 1, 0]);
    const proj = mat4PerspectiveZO((42 * Math.PI) / 180, w / h, 0.1, 30);
    const vp = mat4Multiply(proj, view);

    const project = (x: number, y: number, z: number): { x: number; y: number; w: number } | null => {
      const cx = vp[0] * x + vp[4] * y + vp[8] * z + vp[12];
      const cy = vp[1] * x + vp[5] * y + vp[9] * z + vp[13];
      const cw = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
      if (cw <= 0.05) return null;
      return {
        x: (cx / cw * 0.5 + 0.5) * w,
        y: (0.5 - cy / cw * 0.5) * h,
        w: cw
      };
    };

    // Draw turntable ground calibration disc
    ctx.strokeStyle = 'rgba(130, 155, 185, 0.16)';
    ctx.lineWidth = 1 * dpr;
    for (const r of [0.85, 1.7, CAPTURE_RADIUS]) {
      ctx.beginPath();
      let started = false;
      for (let deg = 0; deg <= 360; deg += 6) {
        const rad = (deg * Math.PI) / 180;
        const p = project(Math.sin(rad) * r, -0.45, Math.cos(rad) * r);
        if (!p) continue;
        if (!started) {
          ctx.moveTo(p.x, p.y);
          started = true;
        } else {
          ctx.lineTo(p.x, p.y);
        }
      }
      ctx.stroke();
    }

    // Draw elevation rings
    const uniqueElevs = Array.from(new Set(this.slots.map((s) => s.elevationDeg)));
    for (const el of uniqueElevs) {
      ctx.strokeStyle = 'rgba(0, 240, 200, 0.22)';
      ctx.setLineDash([4 * dpr, 4 * dpr]);
      ctx.beginPath();
      let started = false;
      for (let az = 0; az <= 360; az += 5) {
        const pos = sphericalToCartesian(az, el, CAPTURE_RADIUS);
        const p = project(pos[0], pos[1], pos[2]);
        if (!p) continue;
        if (!started) {
          ctx.moveTo(p.x, p.y);
          started = true;
        } else {
          ctx.lineTo(p.x, p.y);
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw central target wireframe prism representing subject
    const subjectCenter = project(0, 0.05, 0);
    if (subjectCenter) {
      const boxR = 0.45;
      const corners = [
        [-boxR, -0.4, -boxR], [boxR, -0.4, -boxR], [boxR, -0.4, boxR], [-boxR, -0.4, boxR],
        [-boxR, 0.5, -boxR], [boxR, 0.5, -boxR], [boxR, 0.5, boxR], [-boxR, 0.5, boxR]
      ].map((c) => project(c[0], c[1], c[2]));

      const edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
      ];
      ctx.strokeStyle = 'rgba(255, 176, 32, 0.55)';
      ctx.lineWidth = 1.3 * dpr;
      for (const [a, b] of edges) {
        const pa = corners[a];
        const pb = corners[b];
        if (pa && pb) {
          ctx.beginPath();
          ctx.moveTo(pa.x, pa.y);
          ctx.lineTo(pb.x, pb.y);
          ctx.stroke();
        }
      }
    }

    // Sort camera stations back-to-front for crisp painter's order
    this.projectedScreenCoords = [];
    const items = this.slots
      .map((slot) => {
        const p = project(slot.position[0], slot.position[1], slot.position[2]);
        return p ? { slot, p } : null;
      })
      .filter((x): x is { slot: CaptureSlot; p: { x: number; y: number; w: number } } => x !== null)
      .sort((a, b) => b.p.w - a.p.w);

    for (const { slot, p } of items) {
      this.projectedScreenCoords.push({ x: p.x, y: p.y, z: p.w, idx: slot.index });

      const isSelected = slot.index === this.selectedIndex;
      const isRec = slot.index === this.recommendedIndex && !slot.captured;
      const scale = Math.max(0.55, Math.min(1.35, 5.8 / p.w));
      const radius = (isSelected ? 8.5 : isRec ? 7.5 : 5.6) * dpr * scale;

      // Draw optical sightline ray for active/captured camera
      if (isSelected || slot.captured) {
        const targetPt = project(0, 0.05, 0);
        if (targetPt) {
          ctx.strokeStyle = isSelected
            ? 'rgba(255, 176, 32, 0.65)'
            : 'rgba(0, 240, 200, 0.14)';
          ctx.lineWidth = (isSelected ? 1.6 : 0.9) * dpr;
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(targetPt.x, targetPt.y);
          ctx.stroke();
        }
      }

      // Outer halo for recommended or selected station
      if (isSelected || isRec) {
        ctx.strokeStyle = isSelected ? '#ffb020' : '#00f0c8';
        ctx.lineWidth = 1.8 * dpr;
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius + 4.5 * dpr, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Camera frustum node
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      if (slot.captured) {
        const verdict = slot.photo?.verdict ?? 'good';
        ctx.fillStyle = verdict === 'good' ? '#18e28c' : verdict === 'warn' ? '#ffb020' : '#ff4d5e';
        ctx.fill();
        ctx.strokeStyle = isSelected ? '#ffffff' : '#042822';
        ctx.lineWidth = (isSelected ? 2.0 : 1.4) * dpr;
        ctx.stroke();
      } else if (isSelected) {
        ctx.fillStyle = '#00f0c8';
        ctx.fill();
      } else if (isRec) {
        ctx.fillStyle = 'rgba(0, 240, 200, 0.28)';
        ctx.fill();
        ctx.strokeStyle = '#00f0c8';
        ctx.lineWidth = 1.4 * dpr;
        ctx.stroke();
      } else {
        ctx.fillStyle = 'rgba(24, 34, 48, 0.88)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(138, 164, 196, 0.45)';
        ctx.lineWidth = 1.1 * dpr;
        ctx.stroke();
      }
    }

    // Compass azimuth labels
    const cardinals: [string, number][] =
      this.locale === 'zh'
        ? [
            ['正前 (6点)', 0],
            ['正右 (3点)', 90],
            ['正后 (12点)', 180],
            ['正左 (9点)', 270]
          ]
        : [
            ['FRONT (6h)', 0],
            ['RIGHT (3h)', 90],
            ['BACK (12h)', 180],
            ['LEFT (9h)', 270]
          ];
    ctx.font = `600 ${10.5 * dpr}px "Noto Sans SC", "Plus Jakarta Sans", sans-serif`;
    ctx.textAlign = 'center';
    for (const [label, az] of cardinals) {
      const pos = sphericalToCartesian(az, -10, CAPTURE_RADIUS + 0.48);
      const p = project(pos[0], pos[1], pos[2]);
      if (p) {
        ctx.fillStyle = 'rgba(168, 190, 215, 0.72)';
        ctx.fillText(label, p.x, p.y);
      }
    }
  }
}
