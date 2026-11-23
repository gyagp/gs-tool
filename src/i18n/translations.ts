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
    appTitleCompact: '3DGS 环拍工作台',
    captureTabShort: '① 拍摄采集',
    renderTabShort: '② 3D 预览',
    importShort: '导入模型',
    captureSettings: '拍摄设置与覆盖进度',
    captureHelp: '机位球与拍摄技巧',
    btnVideoCapture: '🎞️ 从相册选择视频',
    btnVideoRecord: '🎥 录制环绕视频',
    videoHint: '固定物体，手机保持高度和距离，缓慢绕完整一圈（建议 10–20 秒）。让主体尽量占满画面，减少反光和手抖。',
    btnCancelVideo: '取消视频处理',
    videoBusyTitle: '正在本机筛选清晰且不同的视频帧…',
    videoLocalModelHint: '首次使用需载入本地分割模型与运行库；视频不会上传。',
    roundBottleOption: '圆瓶（其他物体请关闭）',
    videoDetailOption: '24 帧细节（下次导入生效）',
    roundBottleResult: '圆瓶轮廓拟合 · 单视角原纹理预览，切换可能跳变；点云导出不含此贴图。非精确重建。',
    hullResult: '轮廓近似 · 相机方向为引导估算值',
    qualityTitle: '想获得更逼真的效果？',
    qualityHint: '手机导入视频生成快速轮廓预览。高保真重建需在电脑上从更多原始帧恢复真实机位并训练模型，再导入 PLY 在这里转动查看。适用于静止物体，不限于圆瓶；视频和模型保留在本机。',
    qualityGuide: '查看电脑重建步骤',
    importResult: '原始高斯模型 · 完整点数、椭球形状与视角相关颜色；诊断模式仅显示基础颜色',
    importFallback: '当前为简化点预览，缺少视角相关颜色。完整模型显示需要 WebGL2。导出 PLY 保留原文件。',
    importContextLost: '显卡上下文已中断，正在等待浏览器恢复；可重新导入原模型。',
    videoModelError: '本地分割模型未能载入，请检查网络后重试。视频没有上传，也没有生成替代模型。',
    videoPlanHint: '细节优先最多保留 24 帧，快速模式最多 8 帧；方向按均匀环绕估算，不包含顶面采集，增加帧数不能保证精确重建。',
    videoNeedsEmptySession: '视频将单独建立一组照片。请先“清空重拍”，以保留当前照片不被覆盖；相机正在开启时请稍候。',
    videoDurationError: '请选择 2–90 秒的视频，建议用 10–20 秒缓慢绕瓶子完整一圈。',
    videoDecodeError: '当前浏览器无法读取这段视频，请尝试 MP4 / WebM，或用系统相机重新录制。',
    videoNoFrames: '未选出可用画面。请使用有反差的纯色背景，瓶子保持不动，手机平稳绕行，避免模糊和长时间停在同一角度。',
    videoNeedMoreViews: (minimum: number) => `当前视频选出的不同画面不足 ${minimum} 个，暂不构建。已选画面保留，可按引导补拍。`,
    needMoreViews: '当前角度不足以约束瓶身，暂不生成实体。请补齐四周至少 4 个分散方向；单张照片或只拍一侧无法可靠重建。',
    videoCancelled: '已取消视频处理，没有替换现有照片。',
    videoSelected: (count: number, sampled = 48, minimum = 6) => `已从 ${sampled} 个候选画面选出 ${count} 个关键帧。方向仍为估算值，请检查抠图；${count >= minimum ? '正在构建近似预览。' : `不足 ${minimum} 个，可按引导补拍。`}`,
    reconstructionNote: '当前为实验性轮廓近似：按引导机位估算，无法保证标签、反光和透明瓶身的真实 3D 还原。先检查抠图是否只剩瓶子。',
    foregroundTitle: '查看抠图：应只剩瓶子，不含背景',
    foregroundHint: '棋盘格代表已去掉的背景。若桌面、墙面也在里面，或瓶身缺失，请换成与瓶身、标签颜色不同的哑光纯色背景重拍。',
    backgroundUnclear: '未能可靠分离瓶子与背景。请保留完整瓶口和瓶底，换用与瓶身、标签颜色不同的纯色背景，并查看抠图后重拍。透明或强反光瓶身可能无法正确分离。',
    noPhotos: '请先拍摄或上传你的饮料瓶照片。演示物体需单独点击“一键演示”。',
    cannotMixDemo: '实拍照片与演示图片不能混合重建。请先使用“清空重拍”，再开始另一组拍摄。',
    reconstructionFailed: '这些照片暂时无法得到可靠的瓶身轮廓，未生成替代模型。照片已保留，请先检查各张抠图，并保持取景高度、距离和方向与引导一致。',
    emptyViewer: '暂无当前物体的 3D 结果。请拍摄后生成预览，或导入模型。',
    btnPreviewCaptures: '查看 3D 预览',
    btnAddDetail: '补拍至 16 张',
    previewReady: (count: number) => `已拍 ${count} 张，可先预览；细节不足再补拍。`,
    appSubtitle: '引导拍摄 · 抠图检查 · 实验性轮廓预览与 3DGS 模型查看',
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
    generateBtnReady: '根据已拍照片生成近似轮廓预览 →',
    generateBtnEarly: '生成 3D 预览（建议继续补齐缺失方位效果更好）',
    generateHintMin: (min: number, cur: number, ready: boolean) =>
      cur === 0
        ? `先拍 ${min} 张起步，覆盖四周与顶部（也可用“一键演示”体验）`
        : cur < min
          ? `建议再拍 ${min - cur} 张，并覆盖四周与顶部；也可先查看已拍照片的预览`
          : ready ? '可以先生成预览，不必拍满更多机位；细节不足再按需补拍' : '张数已够起步，请优先补拍缺失方向与顶部',

    // Center column: Step-by-step coach & latest photo feedback
    nextActionBadge: '下一步拍摄指引',
    allDoneBadge: '🎉 全部推荐机位已拍齐！',
    standAtLabel: '站位方向',
    cameraHeightLabel: '相机高度',
    framingTip: '完整保留瓶口和瓶底、四周留白；使用与瓶身和标签有反差的哑光纯色背景',
    btnStartCam: '📷 相机预览',
    btnShutterReady: '📸 开启相机拍摄',
    btnCameraStarting: '正在开启相机…',
    btnNativeCamera: '📷 系统相机拍照',
    cameraUnavailable: '无法开启相机预览。请允许浏览器使用相机，并通过 HTTPS 打开页面；也可点击“系统相机拍照”或从相册添加照片。',
    photoReadError: '无法读取这张照片，请重试或选择 JPG / PNG 照片。',
    btnStopCam: '⏹ 关闭摄像头',
    btnShutterLive: '📸 拍下当前角度',
    btnShutterSim: '🎨 模拟拍下此角度',
    btnUploadPhotos: '📁 从相册添加',
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
    modeSplats: '高斯点表面显示',
    modeTexture: '视频视角贴图（近似）',
    textureFallback: '当前无法显示视频贴图，已回退为点云近似预览。',
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
    appTitleCompact: '3DGS Capture Studio',
    captureTabShort: '① Capture',
    renderTabShort: '② 3D Preview',
    importShort: 'Import',
    captureSettings: 'Capture Settings & Coverage',
    captureHelp: 'Camera Positions & Tips',
    btnVideoCapture: '🎞️ Choose Video from Library',
    btnVideoRecord: '🎥 Record Orbit Video',
    videoHint: 'Keep the object fixed and orbit once at a steady height and distance (10–20 seconds). Fill the frame; reduce glare and camera shake.',
    btnCancelVideo: 'Cancel Video Processing',
    videoBusyTitle: 'Selecting clear, distinct video frames on this device…',
    videoLocalModelHint: 'Loading the local segmentation model/runtime on first use. Your video stays on this device.',
    roundBottleOption: 'Round bottle (off for other shapes)',
    videoDetailOption: 'Detail: 24 frames (next import)',
    roundBottleResult: 'Round-bottle silhouette fit · one source view at a time; switching may pop. Point exports exclude these textures. Approximate geometry.',
    hullResult: 'Silhouette approximation · assumed guided camera directions',
    qualityTitle: 'Looking for a more realistic result?',
    qualityHint: 'Video import on your phone creates a quick silhouette preview. For higher fidelity, use a computer to recover real cameras from more source frames and train a model, then import its PLY here. Works with static subjects beyond bottles; media stays local.',
    qualityGuide: 'Computer reconstruction guide',
    importResult: 'Original Gaussians · full count, ellipsoid shapes and view-dependent color; diagnostic modes show base color only',
    importFallback: 'Simplified point preview without view-dependent color. Full fidelity requires WebGL2. PLY export preserves the original file.',
    importContextLost: 'Graphics context interrupted. Waiting for browser recovery; you can also reimport the original model.',
    videoModelError: 'The local segmentation model could not load. Check the connection and retry. No video was uploaded or replacement geometry generated.',
    videoPlanHint: 'Detailed video keeps up to 24 frames; quick mode keeps up to 8. Directions assume a steady full orbit. More frames do not guarantee accurate geometry; top views are not included.',
    videoNeedsEmptySession: 'Video starts a separate photo session. Use Start Over first so current photos are not overwritten; wait if the camera is opening.',
    videoDurationError: 'Choose a 2–90 second video. A slow, complete 10–20 second orbit is recommended.',
    videoDecodeError: 'This browser could not decode the video. Try MP4 / WebM or record again with the system camera.',
    videoNoFrames: 'No usable frames found. Use a contrasting solid backdrop, keep the bottle fixed, orbit steadily, and avoid blur or staying at one angle.',
    videoNeedMoreViews: (minimum: number) => `Fewer than ${minimum} distinct video views are available. Selected frames are kept; add missing guided views.`,
    needMoreViews: 'Not enough surrounding views to constrain the bottle. Capture at least 4 spread-out directions. A single photo or views from only one side cannot reconstruct a reliable surface.',
    videoCancelled: 'Video processing cancelled. Existing photos were not replaced.',
    videoSelected: (count: number, sampled = 48, minimum = 6) => `${count} keyframes selected from ${sampled} samples. Directions remain estimates. Check the cutouts; ${count >= minimum ? 'building an approximate preview.' : `fewer than ${minimum} views found; add missing guided shots.`}`,
    reconstructionNote: 'Experimental silhouette approximation using assumed guided camera poses. Labels, reflections, and transparent bottles may not reconstruct accurately. Check each cutout first.',
    foregroundTitle: 'Check cutout: bottle only, no background',
    foregroundHint: 'The checkerboard is removed background. If the table/wall remains, or the bottle is missing, retake against a matte solid color contrasting with the bottle and label.',
    backgroundUnclear: 'Could not reliably separate the bottle from its background. Keep the cap and base in frame, use a contrasting solid backdrop, and inspect the cutout before retaking. Clear or shiny bottles may not separate correctly.',
    noPhotos: 'Take or upload photos of your bottle first. The demo object is available separately via 1-Click Demo.',
    cannotMixDemo: 'Real photos and demo images cannot be mixed. Use Start Over before starting the other capture session.',
    reconstructionFailed: 'These photos did not produce a reliable bottle surface. No substitute model was generated. Your photos are kept; check every cutout and follow the guided camera height, distance, and direction.',
    emptyViewer: 'No 3D result for the current subject. Capture and generate a preview, or import a model.',
    btnPreviewCaptures: 'View 3D Preview',
    btnAddDetail: 'Add Detail to 16 Photos',
    previewReady: (count: number) => `${count} photos captured. Preview first; add views if needed.`,
    appSubtitle: 'Guided Capture · Cutout Inspection · Experimental Silhouettes & 3DGS Viewer',
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
    generateBtnReady: 'Generate Approximate Silhouette Preview →',
    generateBtnEarly: 'Preview 3D Now (Capturing missing red sectors recommended)',
    generateHintMin: (min: number, cur: number, ready: boolean) =>
      cur === 0
        ? `Start with ${min} side and top views (or try the 1-Click Demo)`
        : cur < min
          ? `${min - cur} more photos recommended, including side and top views. You can also preview now.`
          : ready ? 'Preview now; further angles are optional if details need improvement.' : 'Enough photos to start; prioritize missing directions and top views.',

    nextActionBadge: 'NEXT SHOT INSTRUCTION',
    allDoneBadge: '🎉 ALL RECOMMENDED ANGLES CAPTURED!',
    standAtLabel: 'Where to Stand',
    cameraHeightLabel: 'Camera Height & Tilt',
    framingTip: 'Keep the whole cap and base in frame with a margin; use a matte backdrop contrasting with the bottle and label',
    btnStartCam: '📷 Preview Camera',
    btnShutterReady: '📸 Open Camera',
    btnCameraStarting: 'Opening camera…',
    btnNativeCamera: '📷 System Camera',
    cameraUnavailable: 'Live preview is unavailable. Allow camera access and open this page over HTTPS, or use System Camera / Add Photos instead.',
    photoReadError: 'Unable to read this photo. Try again or choose a JPG / PNG image.',
    btnStopCam: '⏹ Stop Camera',
    btnShutterLive: '📸 Take Photo',
    btnShutterSim: '🎨 Simulate This Shot',
    btnUploadPhotos: '📁 Add Photos',
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
    modeSplats: 'Gaussian Surface Display',
    modeTexture: 'Video View Texture (Approximate)',
    textureFallback: 'Video textures are unavailable; showing the approximate point-cloud fallback.',
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
