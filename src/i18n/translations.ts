/**
 * Bilingual (Chinese / English) Internationalization & Human-Friendly Angle Helper
 * Designed so beginners immediately understand where to stand, how to tilt the camera,
 * what angles are still missing, and whether their latest photo is sharp & well-framed.
 */

export type Locale = 'zh' | 'en';

export function detectInitialLocale(): Locale {
  try {
    const saved = localStorage.getItem('gs_locale');
    if (saved === 'zh' || saved === 'en') return saved;
  } catch {
    // ignore storage errors
  }
  const nav = (typeof navigator !== 'undefined' ? navigator.language : 'zh').toLowerCase();
  return nav.startsWith('zh') ? 'zh' : 'en';
}

export function saveLocale(locale: Locale): void {
  try {
    localStorage.setItem('gs_locale', locale);
  } catch {
    // ignore
  }
}

/**
 * Converts raw Azimuth (0..360°) into intuitive clock-face + compass direction
 */
export function describeAzimuth(azDeg: number, locale: Locale): {
  clock: string;
  direction: string;
} {
  const norm = ((azDeg % 360) + 360) % 360;
  // 0° = 6 o'clock (standing directly in front of object), 90° = 3 o'clock (right side), 180° = 12 o'clock (back), 270° = 9 o'clock (left)
  const clockHours = [6, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7];
  const hourIdx = Math.round(norm / 30) % 12;
  const hour = clockHours[hourIdx];

  const octIdx = Math.round(norm / 45) % 8;
  const dirsZh = ['正前方', '右前方', '正右侧', '右后方', '正后方', '左后方', '正左侧', '左前方'];
  const dirsEn = ['Front', 'Front-Right', 'Right Side', 'Back-Right', 'Back', 'Back-Left', 'Left Side', 'Front-Left'];

  return {
    clock: locale === 'zh' ? `${hour} 点钟方向` : `${hour} o'clock`,
    direction: locale === 'zh' ? dirsZh[octIdx] : dirsEn[octIdx]
  };
}

/**
 * Converts Elevation angle into plain-language camera height & tilt advice
 */
export function describeElevation(elDeg: number, locale: Locale): string {
  if (locale === 'zh') {
    if (elDeg < 0) return `略低于物体，向上微仰拍 (${elDeg}°)`;
    if (elDeg <= 18) return `与物体同高，水平直视拍摄 (+${elDeg}°)`;
    if (elDeg <= 48) return `举高手机/相机，斜向下 45° 俯拍 (+${elDeg}°)`;
    return `从正上方往下高角度俯拍 (+${elDeg}°)`;
  } else {
    if (elDeg < 0) return `Slightly below object, tilt camera gently up (${elDeg}°)`;
    if (elDeg <= 18) return `Eye-level with object, shoot straight ahead (+${elDeg}°)`;
    if (elDeg <= 48) return `Raise camera above object, tilt ~45° down (+${elDeg}°)`;
    return `High top-down overhead view (+${elDeg}°)`;
  }
}

export const OCTANT_LABELS: Record<Locale, string[]> = {
  zh: ['正前(6点)', '右前(4-5点)', '正右(3点)', '右后(1-2点)', '正后(12点)', '左后(10-11点)', '正左(9点)', '左前(7-8点)'],
  en: ['Front (6h)', 'Front-R (5h)', 'Right (3h)', 'Back-R (2h)', 'Back (12h)', 'Back-L (10h)', 'Left (9h)', 'Front-L (7h)']
};

export const UI_STRINGS = {
  zh: {
    appTitle: '3DGS 智能环拍与重建工作台',
    appSubtitle: '一步步引导拍摄 · 实时照片质检 · WebGPU 高速高斯泼溅渲染',
    step1Tab: '第 1 步：引导环拍采集',
    step2Tab: '第 2 步：WebGPU 3D 预览与导出',
    importBtn: '导入 .PLY / .SPLAT',
    gpuReady: 'WebGPU 硬件加速已就绪',
    gpuFallback: '兼容画布模式 (当前浏览器未开启 WebGPU)',
    demoBannerTitle: '手边暂时没有拍摄物体？',
    demoBannerDesc: '一键自动模拟全套多角度环拍照片，立即体验 WebGPU 3D 高斯重建效果！',
    demoBannerBtn: '⚡ 一键演示完整环拍与 3D 重建',

    // Left column: Wizard & Missing checklist
    chooseModeTitle: '1. 选择拍摄精细度',
    missingTitle: '2. 当前还缺哪些角度？',
    compassSub: '8 个水平方位 + 顶部俯视覆盖检测（绿勾=已拍齐，红框=缺失）',
    topCapLabel: '顶部俯拍圈 (>30°)',
    topCapDone: '已覆盖',
    topCapMissing: '还需补拍',
    ringProgressTitle: '各高度圈拍摄进度',
    generateBtnReady: '✨ 照片已达标！立即生成 WebGPU 3D 模型 →',
    generateBtnEarly: '生成 3D 预览（建议继续补齐缺失方位效果更好）',
    generateHintMin: (min: number, cur: number) =>
      cur === 0
        ? `建议至少拍摄 ${min} 张不同角度照片（或点击上方“一键演示”）`
        : cur < min
          ? `再拍 ${min - cur} 张即可达到最低 3D 重建标准`
          : '已满足重建基线！补齐全部点位可消除边缘噪点',

    // Center column: Step-by-step coach & latest photo feedback
    nextActionBadge: '下一步拍摄指引',
    allDoneBadge: '🎉 全部推荐机位已拍齐！',
    standAtLabel: '站位方向',
    cameraHeightLabel: '相机高度',
    framingTip: '将物体放在虚线圆圈正中央（约占画面 2/3），保持背景干净、光线均匀',
    btnStartCam: '📷 开启摄像头实拍',
    btnStopCam: '⏹ 关闭摄像头',
    btnShutterLive: '📸 拍下当前角度',
    btnShutterSim: '🎨 模拟拍下此角度',
    btnUploadPhotos: '📁 上传手机/相机照片',
    btnResetAll: '🔄 清空重拍',

    // Latest Shot Quality Feedback Card
    qcWaitingTitle: '实时照片质检（拍完立刻告诉你行不行）',
    qcWaitingDesc: '每拍一张或上传一张照片，系统会自动检测【清晰度】、【曝光亮度】、【主体居中度】以及【是否与上一张太相似】。',
    qcGoodTitle: '✅ 这张拍得很好！已自动切到下一个角度',
    qcWarnTitle: '⚠️ 这张有些小瑕疵，建议重拍以提升 3D 精度',
    qcBadTitle: '❌ 这张不合格，强烈建议重拍该角度',
    qcMetricSharp: '清晰度',
    qcMetricExp: '曝光度',
    qcMetricSubject: '主体占比',
    qcRetakeBtn: '↩ 重拍刚才这张',
    qcKeepBtn: '继续拍下一张 →',

    // Right column: 3D Dome
    domeTitle: '3D 环拍机位球（直观查看已拍/未拍）',
    domeHint: '拖动旋转球体 · 点击任意圆点可指定补拍该机位',
    legendDone: '已拍合格',
    legendWarn: '有瑕疵',
    legendCurrent: '当前目标',
    legendMissing: '待拍摄',
    tipsCardTitle: '💡 拍出高质量 3DGS 的 3 个秘诀',
    tipsList: [
      '<strong>绕物走动，别转物体</strong>：尽量人绕着静物走一圈拍；若用转盘，请务必使用纯黑或纯白无纹理背景布。',
      '<strong>相邻两张重叠 70%</strong>：每次只挪动一小步（约 30°），不要跨度太大，否则算法无法拼接深度。',
      '<strong>避开强反光与透明物</strong>：玻璃、镜面金属会随视角变化产生高光漂移，哑光/漫反射材质重建最逼真。'
    ],

    // Render Stage
    hudTitle: 'WebGPU 实时性能监控',
    hudFps: '渲染帧率',
    hudTotal: '总高斯椭球数',
    hudVisible: '当前可见数',
    hudSort: 'GPU 深度基数排序',
    hudVram: '显存占用',
    sortCached: '已缓存 (相机静止 0 开销)',
    sortActive: '4 趟 GPU 间接基数排序',
    culledSuffix: '已剔除',
    splatsUnit: '个高斯点',
    viewportControlsHint: '左键拖动：360° 旋转观察 · 右键/Shift+拖动：平移 · 滚轮：缩放远近',

    viewModeTitle: '显示模式切换',
    modeSplats: '真实 3D 高斯渲染',
    modeRings: '椭球协方差轮廓',
    modeDepth: '深度远近热力图',
    modeCenters: '高斯中心点云',

    gpuParamsTitle: '画质与显示调节',
    paramScale: '高斯点尺寸缩放',
    paramCull: '微小透明点剔除阈值',
    paramFov: '相机透视广角 (FOV)',
    autoOrbitOn: '自动旋转展示：开',
    autoOrbitOff: '自动旋转展示：关',
    gridOn: '地面参考网格：开',
    gridOff: '地面参考网格：关',
    resetCamBtn: '重置最佳观察视角',

    exportTitle: '导出模型 / 继续补拍',
    exportPlyBtn: '⬇️ 下载标准 .PLY 模型 (兼容 SuperSplat/Blender)',
    exportSplatBtn: '⬇️ 下载轻量 .SPLAT 模型 (网页极速加载)',
    backToCaptureBtn: '← 返回拍摄台补拍更多角度',

    proBenchTitle: 'WebGPU 极限性能压测',
    proBenchDesc: '生成 25 万个各向异性高斯椭球，测试 PlayCanvas 同款 GPU 间接排序与渲染帧率。',
    proBenchBtn: '🚀 运行 250,000 高斯点压测',

    busySimTitle: '正在自动模拟多角度环拍...',
    busyReconTitle: '正在用多视角照片重建 3D 高斯模型...',
    busyImportTitle: '正在解析并优化 3DGS 模型文件...',
    busyBenchTitle: '正在生成 25 万高斯点压测场景...'
  },
  en: {
    appTitle: '3DGS Guided Capture & WebGPU Studio',
    appSubtitle: 'Step-by-Step Photo Coach · Instant Shot Quality Check · PlayCanvas-Class WebGPU',
    step1Tab: 'Step 1: Guided Photo Capture',
    step2Tab: 'Step 2: WebGPU 3D Viewer & Export',
    importBtn: 'Import .PLY / .SPLAT',
    gpuReady: 'WebGPU Hardware Accelerated',
    gpuFallback: 'Canvas2D Fallback (WebGPU Unavailable)',
    demoBannerTitle: "Don't have an object to photograph right now?",
    demoBannerDesc: 'Simulate a complete multi-angle turntable shoot in 1 click to experience instant WebGPU 3D Gaussian Splatting!',
    demoBannerBtn: '⚡ 1-Click Demo Shoot & 3D Render',

    chooseModeTitle: '1. Choose Scan Quality',
    missingTitle: '2. What Angles Are Still Missing?',
    compassSub: '8 horizontal directions + overhead ring checklist (Green = covered, Red = missing)',
    topCapLabel: 'Overhead Ring (>30°)',
    topCapDone: 'Covered',
    topCapMissing: 'Still Needed',
    ringProgressTitle: 'Elevation Ring Progress',
    generateBtnReady: '✨ Coverage Ready! Reconstruct 3D in WebGPU →',
    generateBtnEarly: 'Preview 3D Now (Capturing missing red sectors recommended)',
    generateHintMin: (min: number, cur: number) =>
      cur === 0
        ? `Capture at least ${min} angles around your object (or click 1-Click Demo above)`
        : cur < min
          ? `Take ${min - cur} more photos to reach baseline 3D reconstruction quality`
          : 'Baseline reached! Fill remaining angles for cleaner edges & zero floaters',

    nextActionBadge: 'NEXT SHOT INSTRUCTION',
    allDoneBadge: '🎉 ALL RECOMMENDED ANGLES CAPTURED!',
    standAtLabel: 'Where to Stand',
    cameraHeightLabel: 'Camera Height & Tilt',
    framingTip: 'Keep the object centered inside the dashed circle (~2/3 of frame) with steady lighting',
    btnStartCam: '📷 Start Live Camera',
    btnStopCam: '⏹ Stop Camera',
    btnShutterLive: '📸 Take Photo',
    btnShutterSim: '🎨 Simulate This Shot',
    btnUploadPhotos: '📁 Upload Phone/Camera Photos',
    btnResetAll: '🔄 Start Over',

    qcWaitingTitle: 'Instant Photo Quality Inspector',
    qcWaitingDesc: 'Every time you snap or upload a photo, we check Sharpness, Exposure, Subject Framing, and Duplicate Angle risk immediately.',
    qcGoodTitle: '✅ Great Shot! Automatically advanced to next angle',
    qcWarnTitle: '⚠️ Usable, but has minor issues — retake recommended',
    qcBadTitle: '❌ Poor Photo Quality — Please retake this angle',
    qcMetricSharp: 'Sharpness',
    qcMetricExp: 'Exposure',
    qcMetricSubject: 'Subject Size',
    qcRetakeBtn: '↩ Retake Previous Shot',
    qcKeepBtn: 'Continue Next →',

    domeTitle: '3D Capture Dome (Drag to Inspect)',
    domeHint: 'Drag to orbit · Click any station dot to inspect or retake that angle',
    legendDone: 'Good Shot',
    legendWarn: 'Needs Care',
    legendCurrent: 'Current Target',
    legendMissing: 'Uncaptured',
    tipsCardTitle: '💡 3 Quick Tips for Crisp 3D Gaussian Splats',
    tipsList: [
      '<strong>Walk around the object</strong>: Move your camera around a stationary object. If using a turntable, use a plain matte black/white backdrop.',
      '<strong>Keep ~70% overlap</strong>: Step gently (~30° per shot) rather than making huge jumps so multi-view depth locks cleanly.',
      '<strong>Avoid mirrors & clear glass</strong>: Matte, textured surfaces reconstruct sharpest; strong specular glare can cause floating artifacts.'
    ],

    hudTitle: 'WebGPU Live Telemetry',
    hudFps: 'Frame Rate',
    hudTotal: 'Total Gaussians',
    hudVisible: 'Visible Survivors',
    hudSort: 'GPU Radix Sort',
    hudVram: 'GPU Memory',
    sortCached: 'Cached (0 ms · Camera Still)',
    sortActive: '4-Pass GPU Indirect Radix',
    culledSuffix: 'culled',
    splatsUnit: 'Splats',
    viewportControlsHint: 'Left-Drag: Orbit 360° · Right-Drag / Shift+Drag: Pan · Scroll: Zoom',

    viewModeTitle: 'Display Mode',
    modeSplats: '3D Gaussian Splats',
    modeRings: 'Ellipsoid Rings',
    modeDepth: 'Depth Heatmap',
    modeCenters: 'Point Centers',

    gpuParamsTitle: 'Visual & GPU Controls',
    paramScale: 'Splat Size Multiplier',
    paramCull: 'Tiny Splat Cull Threshold',
    paramFov: 'Camera Field of View',
    autoOrbitOn: 'Auto-Rotate: ON',
    autoOrbitOff: 'Auto-Rotate: OFF',
    gridOn: 'Ground Grid: ON',
    gridOff: 'Ground Grid: OFF',
    resetCamBtn: 'Reset Camera Framing',

    exportTitle: 'Export / Add More Photos',
    exportPlyBtn: '⬇️ Download Standard .PLY (SuperSplat / Blender)',
    exportSplatBtn: '⬇️ Download Compact .SPLAT (Fast Web Viewer)',
    backToCaptureBtn: '← Back to Capture Studio to Add Angles',

    proBenchTitle: 'WebGPU 250K Stress Benchmark',
    proBenchDesc: 'Test PlayCanvas-grade 4-bit GPU Radix Sort & indirect instancing at 250,000 Gaussians.',
    proBenchBtn: '🚀 Run 250,000 Splat Benchmark',

    busySimTitle: 'Simulating Multi-Angle Photo Shoot...',
    busyReconTitle: 'Reconstructing 3D Gaussian Splats...',
    busyImportTitle: 'Importing & Morton-Sorting 3DGS File...',
    busyBenchTitle: 'Generating 250K Splat Stress Scene...'
  }
} as const;
