/**
 * Multi-angle Photometric Capture Rig, Real-Time Shot Quality Inspector & Missing-Angle Coach
 * Supports full Chinese ('zh') & English ('en') localization, 8-sector compass checklist,
 * blur/exposure/centering/duplicate detection, and beginner-friendly step-by-step guidance.
 */

import { sphericalToCartesian, Vec3 } from '../math/mat4';
import { Locale, OCTANT_LABELS } from '../i18n/translations';

export type CapturePresetId = 'minimal' | 'express' | 'standard' | 'studio';
export type ShotVerdict = 'good' | 'warn' | 'bad';

export interface RingSpec {
  id: string;
  label: Record<Locale, string>;
  elevationDeg: number;
  count: number;
  phaseOffsetDeg?: number;
  purpose: Record<Locale, string>;
}

export interface CapturePreset {
  id: CapturePresetId;
  name: Record<Locale, string>;
  badge: Record<Locale, string>;
  totalPhotos: number;
  recommendedFovDeg: number;
  angularStepDeg: number;
  estimatedOverlapPct?: number;
  rings: RingSpec[];
  useCase: Record<Locale, string>;
}

export interface CaptureSlot {
  index: number;
  ringId: string;
  azimuthDeg: number;
  elevationDeg: number;
  position: Vec3;
  captured: boolean;
  photo?: CapturedPhoto;
}

export interface ShotIssue {
  code: 'blurry' | 'too_dark' | 'too_bright' | 'too_small' | 'off_center' | 'duplicate';
  severity: 'warn' | 'bad';
  message: Record<Locale, string>;
}

export interface CapturedPhoto {
  slotIndex: number;
  azimuthDeg: number;
  elevationDeg: number;
  width: number;
  height: number;
  dataUrl: string;
  rgba: Uint8ClampedArray;
  workWidth: number;
  workHeight: number;
  sharpnessScore: number;   // 0 - 100
  exposureScore: number;    // 0 - 100
  meanLuma: number;         // 0 - 255
  foregroundRatio: number;  // 0 - 1
  centerOffset: number;     // 0 (perfect center) - 1 (at edge)
  similarityToPrev: number; // 0 - 1 (1 = identical duplicate photo)
  verdict: ShotVerdict;
  issues: ShotIssue[];
  summaryText: Record<Locale, string>;
  timestamp: number;
}

export interface SectorStatus {
  octantIndex: number;
  label: string;
  covered: boolean;
  capturedCount: number;
  firstMissingSlotIndex: number;
}

export interface CoverageReport {
  preset: CapturePreset;
  capturedCount: number;
  goodCount: number;
  warnOrBadCount: number;
  requiredCount: number;
  completionPct: number;
  minimumViableCount: number;
  isReadyToRender: boolean;
  isOptimalQuality: boolean;
  sectors: SectorStatus[];
  topRingCovered: boolean;
  topRingCapturedCount: number;
  firstMissingTopSlotIndex: number;
  ringProgress: {
    ringId: string;
    label: string;
    elevationDeg: number;
    captured: number;
    total: number;
  }[];
  nextRecommendedSlot: CaptureSlot | null;
  missingSummaryText: string;
  averageSharpness: number;
}

export const CAPTURE_PRESETS: Record<CapturePresetId, CapturePreset> = {
  minimal: {
    id: 'minimal',
    name: { zh: '8 张极速预览（默认）', en: '8-Shot Preview (Default)' },
    badge: { zh: '8 张 · 2 圈', en: '8 Photos · 2 Rings' },
    totalPhotos: 8,
    recommendedFovDeg: 48,
    angularStepDeg: 90,
    useCase: {
      zh: '先拍前后左右 4 张，再举高俯拍 4 张。适合简单物体的粗略预览；细节不足可保留照片补拍至 16 张。',
      en: '4 side views + 4 raised views for a rough preview of simple objects. Keep these photos and add 8 more if needed.'
    },
    rings: [
      {
        id: 'equator',
        label: { zh: '第 1 圈：前后左右 (+12°)', en: 'Ring 1: Four Sides (+12°)' },
        elevationDeg: 12,
        count: 4,
        purpose: {
          zh: '在物体正前、右侧、正后、左侧各拍一张，先记录整体轮廓。',
          en: 'One photo each from the front, right, back, and left for the overall silhouette.'
        }
      },
      {
        id: 'upper',
        label: { zh: '第 2 圈：四个俯拍角度 (+45°)', en: 'Ring 2: Four Raised Views (+45°)' },
        elevationDeg: 45,
        count: 4,
        // A subset of the 16/28-shot upper ring, so these captures can be reused.
        phaseOffsetDeg: 22.5,
        purpose: {
          zh: '举高手机，按引导从四个斜向角度俯拍，补充顶面信息。',
          en: 'Raise the phone and follow four offset angles to include the top surface.'
        }
      }
    ]
  },
  express: {
    id: 'express',
    name: { zh: '快速尝鲜模式', en: 'Quick Scan' },
    badge: { zh: '16 张 · 2 圈', en: '16 Photos · 2 Rings' },
    totalPhotos: 16,
    recommendedFovDeg: 48,
    angularStepDeg: 45,
    estimatedOverlapPct: 68,
    useCase: {
      zh: '比 8 张方案补充更多侧面和顶面角度；从 8 张升级时只需再拍 8 张。',
      en: 'More side and top views. Upgrading from 8-Shot Preview keeps all 8 photos; take only 8 more.'
    },
    rings: [
      {
        id: 'equator',
        label: { zh: '第 1 圈：水平平视圈 (+12°)', en: 'Ring 1: Eye-Level (+12°)' },
        elevationDeg: 12,
        count: 8,
        purpose: {
          zh: '每隔 45° 拍一张，快速勾勒物体四周一圈的轮廓与表面颜色',
          en: 'Every 45° around the object to capture main silhouette and side colors'
        }
      },
      {
        id: 'upper',
        label: { zh: '第 2 圈：斜向下俯拍圈 (+45°)', en: 'Ring 2: High Oblique (+45°)' },
        elevationDeg: 45,
        count: 8,
        purpose: {
          zh: '举高手机斜向下拍，补全物体顶面细节并提供上下视差',
          en: 'Raise camera looking down to capture the top surface and vertical depth'
        }
      }
    ]
  },
  standard: {
    id: 'standard',
    name: { zh: '28 张完整覆盖', en: '28-Shot Full Coverage' },
    badge: { zh: '28 张 · 3 圈', en: '28 Photos · 3 Rings' },
    totalPhotos: 28,
    recommendedFovDeg: 48,
    angularStepDeg: 30,
    estimatedOverlapPct: 78,
    useCase: {
      zh: '画质与耗时最佳平衡！包含平视、俯拍、微仰 3 圈，大幅减少漂浮噪点',
      en: 'Best balance of speed & crispness! 3 rings prevent holes under rims and chins'
    },
    rings: [
      {
        id: 'equator',
        label: { zh: '第 1 圈：主力水平圈 (+12°)', en: 'Ring 1: Primary Eye-Level (+12°)' },
        elevationDeg: 12,
        count: 12,
        purpose: {
          zh: '像看表盘一样每小时方向拍一张（每 30° 一张），保证相邻照片 78% 重叠度',
          en: '12 clock-face positions (every 30°) for rock-solid ~78% feature overlap'
        }
      },
      {
        id: 'upper',
        label: { zh: '第 2 圈：高角度俯拍圈 (+45°)', en: 'Ring 2: 45° Overhead (+45°)' },
        elevationDeg: 45,
        count: 8,
        purpose: {
          zh: '封住物体顶部结构，防止 3D 高斯椭球沿视线方向拉长变形',
          en: 'Locks top geometry and prevents elongated needle-shaped splats'
        }
      },
      {
        id: 'low',
        label: { zh: '第 3 圈：低角度微仰圈 (-12°)', en: 'Ring 3: Low Undercut (-12°)' },
        elevationDeg: -12,
        count: 8,
        purpose: {
          zh: '稍微蹲低向上仰拍，专门补齐下巴、底座边缘与阴影死角',
          en: 'Slightly below horizon to fill lower shadow gaps and undercuts'
        }
      }
    ]
  },
  studio: {
    id: 'studio',
    name: { zh: '精细棚拍模式', en: 'Dense Studio Quality' },
    badge: { zh: '48 张 · 4 圈', en: '48 Photos · 4 Rings' },
    totalPhotos: 48,
    recommendedFovDeg: 45,
    angularStepDeg: 22.5,
    estimatedOverlapPct: 86,
    useCase: {
      zh: '适合有镂空、凹槽、复杂纹理的雕塑或潮玩，提供最高密度 3D 高斯细节',
      en: 'For complex concavities, intricate carvings, and archival-grade splat density'
    },
    rings: [
      {
        id: 'equator',
        label: { zh: '第 1 圈：加密水平圈 (+8°)', en: 'Ring 1: Dense Equator (+8°)' },
        elevationDeg: 8,
        count: 16,
        purpose: {
          zh: '每 22.5° 拍一张，超高重叠度还原细微表面肌理',
          en: 'Every 22.5° for ultra-high overlap on fine surface micro-textures'
        }
      },
      {
        id: 'mid_high',
        label: { zh: '第 2 圈：中斜俯拍圈 (+35°)', en: 'Ring 2: Mid Oblique (+35°)' },
        elevationDeg: 35,
        count: 12,
        purpose: {
          zh: '平滑衔接侧面与顶面过渡区域',
          en: 'Smoothly bridges side silhouettes and upper surfaces'
        }
      },
      {
        id: 'zenith',
        label: { zh: '第 3 圈：正顶封口圈 (+65°)', en: 'Ring 3: Top Zenith (+65°)' },
        elevationDeg: 65,
        count: 8,
        purpose: {
          zh: '高处往下俯视，彻底拍清顶部凹陷内部',
          en: 'Steep overhead closure for top cavities and concavities'
        }
      },
      {
        id: 'low',
        label: { zh: '第 4 圈：底部仰拍圈 (-15°)', en: 'Ring 4: Low Base (-15°)' },
        elevationDeg: -15,
        count: 12,
        purpose: {
          zh: '扫清底部悬挑结构死角',
          en: 'Eliminates lower overhang blind spots'
        }
      }
    ]
  }
};

export const CAPTURE_RADIUS = 2.6;

export function getRingSpec(preset: CapturePreset, ringId: string): RingSpec {
  return preset.rings.find((r) => r.id === ringId) || preset.rings[0];
}

export function buildSlotsForPreset(preset: CapturePreset, previousSlots: readonly CaptureSlot[] = []): CaptureSlot[] {
  const slots: CaptureSlot[] = [];
  let idx = 0;

  for (let r = 0; r < preset.rings.length; r++) {
    const ring = preset.rings[r];
    const phaseOffset = ring.phaseOffsetDeg ?? (r % 2) * (180 / ring.count);
    for (let i = 0; i < ring.count; i++) {
      const azimuthDeg = (i * (360 / ring.count) + phaseOffset) % 360;
      const position = sphericalToCartesian(azimuthDeg, ring.elevationDeg, CAPTURE_RADIUS);
      const index = idx++;
      const roundedAzimuth = Math.round(azimuthDeg * 10) / 10;
      const previous = previousSlots.find((slot) =>
        slot.captured && slot.photo && slot.azimuthDeg === roundedAzimuth && slot.elevationDeg === ring.elevationDeg
      );
      slots.push({
        index,
        ringId: ring.id,
        azimuthDeg: roundedAzimuth,
        elevationDeg: ring.elevationDeg,
        position,
        captured: Boolean(previous),
        photo: previous?.photo ? { ...previous.photo, slotIndex: index } : undefined
      });
    }
  }
  return slots;
}

/**
 * Inspects a newly captured/uploaded photo against quality & framing rules + optional previous photo similarity
 */
export function inspectCapturedPhoto(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  previousPhoto?: CapturedPhoto | null
): {
  sharpnessScore: number;
  exposureScore: number;
  meanLuma: number;
  foregroundRatio: number;
  centerOffset: number;
  similarityToPrev: number;
  verdict: ShotVerdict;
  issues: ShotIssue[];
  summaryText: Record<Locale, string>;
} {
  const gray = new Float32Array(width * height);
  let lumaSum = 0;
  let clippedDark = 0;
  let clippedBright = 0;

  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    const y = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
    gray[i] = y;
    lumaSum += y;
    if (y < 14) clippedDark++;
    if (y > 244) clippedBright++;
  }

  const meanLuma = lumaSum / gray.length;

  // 1. Sharpness via 3x3 Laplacian variance (central 76% region)
  let lapSum = 0;
  let lapSqSum = 0;
  let count = 0;
  const x0 = Math.floor(width * 0.12);
  const x1 = Math.ceil(width * 0.88);
  const y0 = Math.floor(height * 0.12);
  const y1 = Math.ceil(height * 0.88);

  for (let y = Math.max(1, y0); y < Math.min(height - 1, y1); y++) {
    const row = y * width;
    for (let x = Math.max(1, x0); x < Math.min(width - 1, x1); x++) {
      const idx = row + x;
      const lap =
        -4 * gray[idx] +
        gray[idx - 1] +
        gray[idx + 1] +
        gray[idx - width] +
        gray[idx + width];
      lapSum += lap;
      lapSqSum += lap * lap;
      count++;
    }
  }

  const meanLap = lapSum / Math.max(1, count);
  const variance = Math.max(0, lapSqSum / Math.max(1, count) - meanLap * meanLap);
  const sharpnessScore = Math.min(100, Math.round(Math.sqrt(variance) * 3.8));

  // 2. Exposure balance
  const clipFraction = (clippedDark + clippedBright) / gray.length;
  const centerDist = Math.abs(meanLuma - 122) / 122;
  const exposureScore = Math.max(
    8,
    Math.min(100, Math.round((1 - centerDist * 0.52 - clipFraction * 0.6) * 100))
  );

  // 3. Foreground subject size & centroid offset
  let borderR = 0, borderG = 0, borderB = 0, borderN = 0;
  for (let x = 0; x < width; x++) {
    const top = x * 4;
    const bot = ((height - 1) * width + x) * 4;
    borderR += rgba[top] + rgba[bot];
    borderG += rgba[top + 1] + rgba[bot + 1];
    borderB += rgba[top + 2] + rgba[bot + 2];
    borderN += 2;
  }
  borderR /= borderN;
  borderG /= borderN;
  borderB /= borderN;

  let fgPixels = 0;
  let sumX = 0;
  let sumY = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const p = (y * width + x) * 4;
      const dist = Math.hypot(rgba[p] - borderR, rgba[p + 1] - borderG, rgba[p + 2] - borderB);
      if (dist > 26 || rgba[p + 3] < 250) {
        fgPixels++;
        sumX += x;
        sumY += y;
      }
    }
  }

  const foregroundRatio = fgPixels / (width * height);
  let centerOffset = 0;
  if (fgPixels > 20) {
    const cx = sumX / fgPixels / width;
    const cy = sumY / fgPixels / height;
    centerOffset = Math.min(1, Math.hypot(cx - 0.5, cy - 0.5) * 2);
  }

  // 4. Similarity against previous shot (detects accidental duplicate without stepping)
  let similarityToPrev = 0;
  if (previousPhoto && previousPhoto.workWidth === width && previousPhoto.workHeight === height) {
    let sad = 0;
    const prev = previousPhoto.rgba;
    const step = 16;
    let samples = 0;
    for (let p = 0; p < rgba.length; p += step) {
      sad +=
        Math.abs(rgba[p] - prev[p]) +
        Math.abs(rgba[p + 1] - prev[p + 1]) +
        Math.abs(rgba[p + 2] - prev[p + 2]);
      samples++;
    }
    const meanChannelDiff = sad / Math.max(1, samples * 3);
    similarityToPrev = Math.max(0, 1 - meanChannelDiff / 36);
  }

  const issues: ShotIssue[] = [];

  if (sharpnessScore < 18) {
    issues.push({
      code: 'blurry',
      severity: 'bad',
      message: {
        zh: '画面明显手抖模糊，请拿稳手机对焦在物体上重拍',
        en: 'Image is blurry — hold steady, tap to focus on the object, and retake'
      }
    });
  } else if (sharpnessScore < 32) {
    issues.push({
      code: 'blurry',
      severity: 'warn',
      message: {
        zh: '边缘略微发虚，建议停稳 0.5 秒后再按快门',
        en: 'Slightly soft focus — pause briefly before pressing the shutter'
      }
    });
  }

  if (meanLuma < 26 && foregroundRatio < 0.08) {
    issues.push({
      code: 'too_dark',
      severity: 'bad',
      message: {
        zh: '画面过暗几乎看不清物体，请增加环境照明',
        en: 'Frame is far too dark — add room lighting on the subject'
      }
    });
  } else if (meanLuma > 236 && clippedBright > gray.length * 0.55) {
    issues.push({
      code: 'too_bright',
      severity: 'warn',
      message: {
        zh: '高光有些过曝发白，建议避开强光直射',
        en: 'Highlights are overexposed — avoid harsh direct spotlight'
      }
    });
  }

  if (foregroundRatio < 0.045) {
    issues.push({
      code: 'too_small',
      severity: 'warn',
      message: {
        zh: '物体在画面中太小或与背景融为一体，请靠近一点拍摄',
        en: 'Object looks very small in frame — move closer so it fills ~2/3 of the circle'
      }
    });
  } else if (centerOffset > 0.42 && foregroundRatio < 0.75) {
    issues.push({
      code: 'off_center',
      severity: 'warn',
      message: {
        zh: '物体偏离中心较多，尽量把物体对准圆环正中央',
        en: 'Subject is off-center — keep the object inside the center guide ring'
      }
    });
  }

  if (similarityToPrev > 0.94) {
    issues.push({
      code: 'duplicate',
      severity: 'warn',
      message: {
        zh: '这张和上一张几乎一模一样！请记得绕着物体挪动一小步再拍',
        en: 'Almost identical to your previous photo! Remember to step ~30° around the object'
      }
    });
  }

  let verdict: ShotVerdict = 'good';
  if (issues.some((i) => i.severity === 'bad')) {
    verdict = 'bad';
  } else if (issues.length > 0) {
    verdict = 'warn';
  }

  let summaryText: Record<Locale, string>;
  if (verdict === 'good') {
    summaryText = {
      zh: `清晰度优 (${sharpnessScore}分)、亮度适中、主体居中饱满，非常棒！`,
      en: `Sharp (${sharpnessScore}/100), well-lit, and nicely centered for 3DGS!`
    };
  } else {
    summaryText = {
      zh: issues.map((i) => `• ${i.message.zh}`).join('  '),
      en: issues.map((i) => `• ${i.message.en}`).join('  ')
    };
  }

  return {
    sharpnessScore,
    exposureScore,
    meanLuma,
    foregroundRatio,
    centerOffset,
    similarityToPrev,
    verdict,
    issues,
    summaryText
  };
}

/**
 * Evaluates overall multi-angle coverage, exact missing compass sectors, and next step instructions
 */
export function evaluateCoverage(
  preset: CapturePreset,
  slots: CaptureSlot[],
  currentSelectionIndex: number,
  locale: Locale
): CoverageReport {
  const capturedSlots = slots.filter((s) => s.captured);
  const capturedCount = capturedSlots.length;
  const goodCount = capturedSlots.filter((s) => s.photo?.verdict === 'good').length;
  const warnOrBadCount = capturedCount - goodCount;

  const requiredCount = slots.length;
  const completionPct = Math.round((capturedCount / Math.max(1, requiredCount)) * 100);
  const minimumViableCount = Math.max(8, Math.ceil(requiredCount * 0.5));

  const ringProgress = preset.rings.map((ring) => {
    const rSlots = slots.filter((s) => s.ringId === ring.id);
    const done = rSlots.filter((s) => s.captured).length;
    return {
      ringId: ring.id,
      label: ring.label[locale],
      elevationDeg: ring.elevationDeg,
      captured: done,
      total: rSlots.length
    };
  });

  const labels = OCTANT_LABELS[locale];
  const sectors: SectorStatus[] = [];

  for (let oct = 0; oct < 8; oct++) {
    const centerAz = oct * 45;
    const inSector = slots.filter((s) => {
      let diff = Math.abs(s.azimuthDeg - centerAz) % 360;
      if (diff > 180) diff = 360 - diff;
      return diff <= 26;
    });
    const capInSector = inSector.filter((s) => s.captured);
    const firstMissing = inSector.find((s) => !s.captured)?.index ?? 0;

    sectors.push({
      octantIndex: oct,
      label: labels[oct],
      covered: capInSector.length > 0,
      capturedCount: capInSector.length,
      firstMissingSlotIndex: firstMissing
    });
  }

  const topSlots = slots.filter((s) => s.elevationDeg >= 30);
  const topCaptured = topSlots.filter((s) => s.captured).length;
  const topRingCovered = topCaptured >= 3;
  const firstMissingTopSlotIndex = topSlots.find((s) => !s.captured)?.index ?? 0;

  // Sequential walking order so the user walks naturally around each ring one step at a time
  let nextRecommendedSlot: CaptureSlot | null = null;
  const uncaptured = slots.filter((s) => !s.captured);

  if (uncaptured.length > 0) {
    for (let offset = 0; offset < slots.length; offset++) {
      const idx = (currentSelectionIndex + offset) % slots.length;
      if (!slots[idx].captured) {
        nextRecommendedSlot = slots[idx];
        break;
      }
    }
  }

  const missingSectors = sectors.filter((s) => !s.covered);
  const isReadyToRender = capturedCount >= minimumViableCount && missingSectors.length <= 2 && topRingCovered;
  const isOptimalQuality = capturedCount === requiredCount && warnOrBadCount === 0;

  let missingSummaryText = '';
  if (locale === 'zh') {
    if (capturedCount === 0) {
      missingSummaryText = `当前方案推荐 ${requiredCount} 张：按引导拍四周和顶部，可先预览，再按需补拍。`;
    } else if (uncaptured.length === 0) {
      missingSummaryText =
        warnOrBadCount > 0
          ? `全部 ${requiredCount} 个机位已拍完！其中有 ${warnOrBadCount} 张提示小瑕疵，可直接生成 3D 或点击黄框重拍。`
          : `已拍齐 ${requiredCount} 个推荐角度！可先生成预览，细节不足时再补拍。`;
    } else {
      const parts: string[] = [];
      if (missingSectors.length > 0) {
        parts.push(
          `水平方向还缺【${missingSectors
            .slice(0, 3)
            .map((s) => s.label)
            .join('、')}】${missingSectors.length > 3 ? `等 ${missingSectors.length} 个方位` : ''}`
        );
      }
      if (!topRingCovered) {
        parts.push(`顶部俯视还差 ${Math.max(1, 3 - topCaptured)} 张`);
      }
      if (parts.length === 0) {
        parts.push(`8 个主方向均已覆盖！还可补拍剩余 ${uncaptured.length} 个加密机位让细节更锐利`);
      }
      missingSummaryText = parts.join('；') + '。';
    }
  } else {
    if (capturedCount === 0) {
      missingSummaryText = `This plan recommends ${requiredCount} side and top views. Preview first, then add angles if needed.`;
    } else if (uncaptured.length === 0) {
      missingSummaryText =
        warnOrBadCount > 0
          ? `All ${requiredCount} stations captured (${warnOrBadCount} flagged with minor warnings). Ready to reconstruct!`
          : `All ${requiredCount} recommended angles captured! Preview now and add views if details need improvement.`;
    } else {
      const parts: string[] = [];
      if (missingSectors.length > 0) {
        parts.push(
          `Missing directions: ${missingSectors
            .slice(0, 3)
            .map((s) => s.label)
            .join(', ')}${missingSectors.length > 3 ? ` (+${missingSectors.length - 3} more)` : ''}`
        );
      }
      if (!topRingCovered) {
        parts.push(`Need ${Math.max(1, 3 - topCaptured)} more overhead (>30°) shots`);
      }
      if (parts.length === 0) {
        parts.push(`All 8 compass directions covered! ${uncaptured.length} optional fill-in angles left`);
      }
      missingSummaryText = parts.join(' · ') + '.';
    }
  }

  const averageSharpness =
    capturedSlots.length > 0
      ? Math.round(
          capturedSlots.reduce((acc, s) => acc + (s.photo?.sharpnessScore ?? 80), 0) /
            capturedSlots.length
        )
      : 0;

  return {
    preset,
    capturedCount,
    goodCount,
    warnOrBadCount,
    requiredCount,
    completionPct,
    minimumViableCount,
    isReadyToRender,
    isOptimalQuality,
    sectors,
    topRingCovered,
    topRingCapturedCount: topCaptured,
    firstMissingTopSlotIndex,
    ringProgress,
    nextRecommendedSlot,
    missingSummaryText,
    averageSharpness
  };
}
