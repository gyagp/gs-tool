import './styles/main.css';
import {
  buildSlotsForPreset,
  CAPTURE_PRESETS,
  CapturedPhoto,
  CapturePresetId,
  CaptureSlot,
  evaluateCoverage,
  getRingSpec,
  inspectCapturedPhoto
} from './capture/coverage';
import { CaptureDomeVisualizer } from './capture/domeVisualizer';
import { renderSyntheticCaptureView } from './capture/syntheticSubject';
import {
  describeAzimuth,
  describeElevation,
  detectInitialLocale,
  Locale,
  saveLocale,
  UI_STRINGS
} from './i18n/translations';
import { packSplatsForGPU, reconstructSplatsFromCaptures, SplatCloudData } from './sfm/reconstruct';
import { RenderMode, RendererStats, WebGPUSplatRenderer } from './webgpu/splatRenderer';
import { exportToPlyBinary, exportToSplatBinary, importSplatFile } from './io/plySplatIo';

class GuidedGSplatApp {
  private locale: Locale = detectInitialLocale();
  private presetId: CapturePresetId = 'minimal';
  private slots: CaptureSlot[] = [];
  private selectedSlotIdx = 0;
  private lastCapturedSlotIdx: number | null = null;
  private activeTab: 'capture' | 'render' = 'capture';

  private stream: MediaStream | null = null;
  private cameraLive = false;
  private cameraStarting = false;

  private domeVis!: CaptureDomeVisualizer;
  private renderer!: WebGPUSplatRenderer;
  private currentCloud: SplatCloudData | null = null;
  private webgpuSupported = false;
  private latestStats: RendererStats | null = null;

  public async mount(root: HTMLElement): Promise<void> {
    root.innerHTML = this.buildShellHTML();
    this.slots = buildSlotsForPreset(CAPTURE_PRESETS[this.presetId]);

    const domeCanvas = document.getElementById('dome-canvas') as HTMLCanvasElement;
    this.domeVis = new CaptureDomeVisualizer(domeCanvas);
    this.domeVis.onSelectSlot = (idx) => {
      this.selectedSlotIdx = idx;
      this.refreshAllUI();
      this.renderCurrentPreviewSlot();
    };

    const gpuCanvas = document.getElementById('webgpu-canvas') as HTMLCanvasElement;
    this.renderer = new WebGPUSplatRenderer(gpuCanvas);
    this.webgpuSupported = await this.renderer.init();

    this.renderer.onStats = (stats) => {
      this.latestStats = stats;
      this.updateTelemetryHUD();
    };

    this.bindEvents();
    this.refreshAllUI();
    this.renderCurrentPreviewSlot();

    const ro = new ResizeObserver(() => {
      if (this.activeTab === 'capture') {
        this.domeVis.render();
      } else {
        this.renderer.markDirty();
      }
    });
    ro.observe(domeCanvas.parentElement || domeCanvas);
    ro.observe(gpuCanvas.parentElement || gpuCanvas);

    window.addEventListener('resize', () => {
      if (this.activeTab === 'capture') {
        this.domeVis.render();
      } else {
        this.renderer.markDirty();
      }
    });
  }

  private t() {
    return UI_STRINGS[this.locale];
  }

  private buildShellHTML(): string {
    return `
      <header class="topbar">
        <div class="brand">
          <div class="brand-mark">
            <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="#00f0c8" stroke-width="2">
              <circle cx="12" cy="12" r="9" stroke-dasharray="3 2"/>
              <circle cx="12" cy="12" r="3.5" fill="#ffb020" stroke="none"/>
            </svg>
          </div>
          <div>
            <div class="brand-title" id="ui-app-title"></div>
            <div class="brand-title mobile-only" id="ui-app-title-compact"></div>
            <div class="brand-sub" id="ui-app-sub"></div>
          </div>
        </div>

        <nav class="mode-tabs" role="tablist">
          <button class="mode-tab active" id="tab-capture" data-tab="capture">
            <span class="desktop-only" id="ui-tab-step1"></span>
            <span class="mobile-only" id="ui-tab-capture-short"></span>
            <span class="tab-badge" id="badge-capture-count">0 / ${CAPTURE_PRESETS[this.presetId].totalPhotos}</span>
          </button>
          <button class="mode-tab" id="tab-render" data-tab="render">
            <span class="desktop-only" id="ui-tab-step2"></span>
            <span class="mobile-only" id="ui-tab-render-short"></span>
            <span class="tab-badge" id="badge-splat-count">0</span>
          </button>
        </nav>

        <div class="topbar-actions">
          <div class="lang-switch" role="group" aria-label="Language">
            <button class="lang-btn" id="lang-zh" data-lang="zh">中文</button>
            <button class="lang-btn" id="lang-en" data-lang="en">EN</button>
          </div>

          <div class="gpu-pill">
            <span class="status-dot" id="gpu-status-dot"></span>
            <span id="gpu-status-text">WebGPU</span>
          </div>

          <label class="btn">
            <span class="desktop-only" id="ui-import-btn"></span>
            <span class="mobile-only" id="ui-import-short"></span>
            <input type="file" id="input-import-splat" accept=".ply,.splat" hidden />
          </label>
        </div>
      </header>

      <main class="workspace">
        <!-- STAGE 1: STEP-BY-STEP CAPTURE STUDIO -->
        <section id="capture-stage" class="stage-view active">
          <!-- Left Column: Preset + Missing Directions Checklist + Generate Button -->
          <aside class="panel">
            <details class="responsive-details" id="capture-settings" open>
              <summary><span id="ui-capture-settings"></span><span class="tab-badge" id="settings-capture-count"></span></summary>
            <div class="panel-section">
              <div class="demo-banner">
                <div class="demo-banner-title" id="ui-demo-title"></div>
                <div class="demo-banner-desc" id="ui-demo-desc"></div>
                <button class="btn btn-amber" id="btn-synth-all" style="width:100%;">
                  <span id="ui-demo-btn"></span>
                </button>
                <button class="btn" id="btn-simulate-shot" style="width:100%; margin-top:8px;"></button>
              </div>
            </div>

            <div class="panel-section">
              <div class="section-header">
                <span class="section-title" id="ui-choose-mode"></span>
              </div>
              <div class="preset-grid" id="preset-cards"></div>
            </div>

            <div class="panel-section">
              <div class="section-header">
                <span class="section-title" id="ui-missing-title"></span>
              </div>
              <div class="missing-summary-banner" id="missing-summary-text"></div>
              <div style="font-size:11px; color:var(--text-muted); margin-bottom:7px;" id="ui-compass-sub"></div>
              <div class="sector-grid" id="sector-checklist"></div>
            </div>

            <div class="panel-section">
              <div class="section-header">
                <span class="section-title" id="ui-ring-progress-title"></span>
              </div>
              <div class="ring-list" id="ring-breakdown"></div>
            </div>

            <div class="panel-section">
              <button class="btn btn-primary" id="btn-reconstruct" style="width:100%; padding:11px 14px; font-size:13.5px;"></button>
              <div style="margin-top:7px; font-size:11.5px; color:var(--text-secondary); text-align:center;" id="ui-generate-hint"></div>
            </div>
            </details>
          </aside>

          <!-- Center Column: Step-by-Step Coach + Viewfinder + Instant Shot Quality Inspector + Filmstrip -->
          <div class="viewfinder-col">
            <div class="coach-banner">
              <div>
                <span class="coach-step-tag" id="coach-badge"></span>
                <div class="coach-main-instruction" id="coach-where"></div>
                <div class="coach-sub-instruction" id="coach-height"></div>
              </div>
              <div style="text-align:right; font-family:var(--font-mono); font-size:12px; color:var(--text-secondary);" id="coach-counter"></div>
            </div>
            <div class="capture-next-actions hidden" id="capture-next-actions" role="status">
              <span id="ui-preview-ready"></span>
              <div class="capture-next-buttons">
                <button class="btn btn-primary" id="btn-preview-captures"></button>
                <button class="btn" id="btn-add-detail"></button>
              </div>
            </div>
            <p class="camera-feedback hidden" id="camera-feedback" role="status" aria-live="polite"></p>

            <div class="viewfinder-stage">
              <video id="camera-video" class="hidden" autoplay playsinline muted></video>
              <canvas id="preview-photo-canvas" width="512" height="512"></canvas>
              <div class="reticle-center-ring"></div>
              <div class="framing-tip-pill" id="ui-framing-tip"></div>
            </div>

            <div class="shutter-bar">
              <div class="shutter-group">
                <label class="btn" id="native-camera-label">
                  <span id="ui-native-camera"></span>
                  <input type="file" id="input-camera-photo" accept="image/*" capture="environment" hidden />
                </label>
                <button class="btn btn-primary btn-shutter-big" id="btn-shutter"></button>
                <label class="btn">
                  <span id="ui-upload-btn"></span>
                  <input type="file" id="input-upload-photos" accept="image/*" multiple hidden />
                </label>
              </div>
            </div>

            <div class="capture-tools">
              <button class="btn" id="btn-toggle-webcam"></button>
              <button class="btn" id="btn-clear-captures"></button>
            </div>
            <!-- Real-time Newly Captured Photo Quality Inspector -->
            <div class="qc-card" id="qc-card">
              <div class="qc-left">
                <img id="qc-thumb" class="qc-thumb" style="display:none;" alt="Last shot" />
                <div>
                  <div class="qc-title" id="qc-title"></div>
                  <div class="qc-desc" id="qc-desc"></div>
                  <div class="qc-metrics-pills" id="qc-metrics"></div>
                </div>
              </div>
              <div id="qc-actions" style="display:none; flex-shrink:0;">
                <button class="btn btn-amber" id="btn-retake-last"></button>
              </div>
            </div>

            <div class="filmstrip" id="capture-filmstrip"></div>
          </div>

          <!-- Right Column: 3D Capture Dome + Beginner Tips -->
          <aside class="panel">
            <details class="responsive-details" id="capture-help" open>
              <summary><span id="ui-capture-help"></span></summary>
            <div class="panel-section" style="padding-bottom:8px;">
              <div class="section-header" style="margin-bottom:3px;">
                <span class="section-title" id="ui-dome-title"></span>
              </div>
              <div style="font-size:11.5px; color:var(--text-secondary);" id="ui-dome-hint"></div>
            </div>

            <div class="dome-canvas-wrap">
              <canvas id="dome-canvas"></canvas>
              <div class="dome-legend" id="ui-dome-legend"></div>
            </div>

            <div class="panel-section">
              <div class="section-header">
                <span class="section-title" id="ui-tips-title"></span>
              </div>
              <ul class="tips-list" id="ui-tips-list"></ul>
            </div>
            </details>
          </aside>
        </section>

        <!-- STAGE 2: WEBGPU 3DGS VIEWER -->
        <section id="render-stage" class="stage-view">
          <div class="splat-viewport-wrap">
            <canvas id="webgpu-canvas"></canvas>

            <div class="perf-hud">
              <div class="perf-hud-title" id="ui-hud-title"></div>
              <div class="perf-grid">
                <span class="perf-key" id="ui-hud-fps"></span>
                <span class="perf-val" id="stat-fps">-- FPS</span>
                <span class="perf-key" id="ui-hud-total"></span>
                <span class="perf-val" id="stat-total">0</span>
                <span class="perf-key" id="ui-hud-visible"></span>
                <span class="perf-val" id="stat-visible">0</span>
                <span class="perf-key" id="ui-hud-sort"></span>
                <span class="perf-val" id="stat-sort">--</span>
                <span class="perf-key" id="ui-hud-vram"></span>
                <span class="perf-val" id="stat-vram">0 MB</span>
              </div>
            </div>

            <div class="viewport-hint" id="ui-viewport-hint"></div>
          </div>

          <aside class="panel">
            <div class="panel-section">
              <div class="section-header">
                <span class="section-title" id="ui-view-mode-title"></span>
              </div>
              <div class="segmented-modes" id="render-mode-group">
                <button class="seg-btn active" data-rmode="0" id="ui-mode-0"></button>
                <button class="seg-btn" data-rmode="1" id="ui-mode-1"></button>
                <button class="seg-btn" data-rmode="2" id="ui-mode-2"></button>
                <button class="seg-btn" data-rmode="3" id="ui-mode-3"></button>
              </div>
            </div>

            <div class="panel-section">
              <div class="section-header">
                <span class="section-title" id="ui-gpu-params-title"></span>
              </div>

              <div class="control-row">
                <div class="control-label">
                  <span id="ui-param-scale"></span>
                  <span id="val-scale">1.00x</span>
                </div>
                <input type="range" id="slider-scale" min="0.2" max="2.5" step="0.02" value="1.0" />
              </div>

              <div class="control-row">
                <div class="control-label">
                  <span id="ui-param-cull"></span>
                  <span id="val-cull">0.25</span>
                </div>
                <input type="range" id="slider-cull" min="0.0" max="3.0" step="0.05" value="0.25" />
              </div>

              <div class="control-row">
                <div class="control-label">
                  <span id="ui-param-fov"></span>
                  <span id="val-fov">46°</span>
                </div>
                <input type="range" id="slider-fov" min="24" max="80" step="1" value="46" />
              </div>

              <div style="display:flex; gap:8px; margin-top:10px;">
                <button class="btn" id="btn-turntable" style="flex:1;"></button>
                <button class="btn" id="btn-toggle-grid" style="flex:1;"></button>
              </div>
              <button class="btn" id="btn-reset-cam" style="width:100%; margin-top:8px;"></button>
            </div>

            <div class="panel-section">
              <div class="section-header">
                <span class="section-title" id="ui-export-title"></span>
              </div>
              <div style="display:flex; flex-direction:column; gap:8px;">
                <button class="btn btn-primary" id="btn-export-ply"></button>
                <button class="btn" id="btn-export-splat"></button>
                <button class="btn btn-amber" id="btn-back-to-capture"></button>
              </div>
            </div>

            <div class="panel-section">
              <div class="section-header">
                <span class="section-title" id="ui-bench-title"></span>
              </div>
              <div style="font-size:11.5px; color:var(--text-secondary); margin-bottom:9px;" id="ui-bench-desc"></div>
              <button class="btn" id="btn-benchmark-250k" style="width:100%;"></button>
            </div>
          </aside>
        </section>
      </main>

      <div class="busy-overlay hidden" id="busy-overlay">
        <div class="busy-card">
          <div class="busy-title" id="busy-title"></div>
          <div class="busy-stage" id="busy-stage"></div>
          <div class="progress-track" style="height:7px;">
            <div class="progress-fill" id="busy-bar" style="width:15%"></div>
          </div>
        </div>
      </div>
    `;
  }

  private bindEvents(): void {
    const compactLayout = window.matchMedia('(max-width: 1024px)');
    const updateDetails = () => {
      document.querySelectorAll<HTMLDetailsElement>('.responsive-details').forEach((details) => {
        details.open = !compactLayout.matches;
      });
      this.domeVis.render();
    };
    updateDetails();
    compactLayout.addEventListener('change', updateDetails);
    document.getElementById('capture-help')!.addEventListener('toggle', () => this.domeVis.render());

    // Language switcher
    document.querySelectorAll('.lang-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const lang = (btn as HTMLElement).dataset.lang as Locale;
        this.locale = lang;
        saveLocale(lang);
        this.refreshAllUI();
        this.renderCurrentPreviewSlot();
      });
    });

    // Mode tabs
    document.querySelectorAll('.mode-tab').forEach((btn) => {
      btn.addEventListener('click', () => {
        const tab = (btn as HTMLElement).dataset.tab as 'capture' | 'render';
        this.switchTab(tab);
      });
    });

    // Camera & Shutter
    document.getElementById('btn-toggle-webcam')!.addEventListener('click', () => {
      this.toggleWebcam();
    });

    document.getElementById('btn-shutter')!.addEventListener('click', () => {
      if (this.cameraLive) {
        this.captureFromWebcam();
      } else {
        void this.toggleWebcam();
      }
    });

    document.getElementById('btn-simulate-shot')!.addEventListener('click', () => {
      this.captureSyntheticAt(this.selectedSlotIdx, true);
    });

    // Retake last photo button
    document.getElementById('btn-retake-last')!.addEventListener('click', () => {
      if (this.lastCapturedSlotIdx !== null) {
        const slot = this.slots[this.lastCapturedSlotIdx];
        if (slot) {
          slot.captured = false;
          slot.photo = undefined;
          this.selectedSlotIdx = this.lastCapturedSlotIdx;
          this.lastCapturedSlotIdx = null;
          this.domeVis.focusAngle(slot.azimuthDeg, slot.elevationDeg);
          this.refreshAllUI();
          this.renderCurrentPreviewSlot();
        }
      }
    });

    // Upload photos
    for (const id of ['input-upload-photos', 'input-camera-photo']) {
      const uploadInput = document.getElementById(id) as HTMLInputElement;
      uploadInput.addEventListener('change', async () => {
        if (!uploadInput.files?.length) return;
        try {
          await this.handleUploadedPhotos(Array.from(uploadInput.files));
          this.setCameraFeedback('');
        } catch {
          this.setCameraFeedback(this.t().photoReadError);
        } finally {
          uploadInput.value = '';
        }
      });
    }

    // 1-Click Full Demo
    document.getElementById('btn-synth-all')!.addEventListener('click', async () => {
      await this.simulateFullDomeAndRender();
    });

    // Reset all captures
    document.getElementById('btn-clear-captures')!.addEventListener('click', () => {
      this.slots = buildSlotsForPreset(CAPTURE_PRESETS[this.presetId]);
      this.selectedSlotIdx = 0;
      this.lastCapturedSlotIdx = null;
      this.refreshAllUI();
      this.renderCurrentPreviewSlot();
    });

    // Reconstruct
    document.getElementById('btn-reconstruct')!.addEventListener('click', async () => {
      await this.runReconstruction();
    });
    document.getElementById('btn-preview-captures')!.addEventListener('click', async () => {
      await this.runReconstruction();
    });
    document.getElementById('btn-add-detail')!.addEventListener('click', () => {
      this.selectPreset('express');
    });

    // Render Mode Buttons
    document.querySelectorAll('#render-mode-group .seg-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#render-mode-group .seg-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const mode = parseInt((btn as HTMLElement).dataset.rmode || '0', 10) as RenderMode;
        this.renderer.renderMode = mode;
        this.renderer.markDirty();
      });
    });

    // Sliders
    const scaleSlider = document.getElementById('slider-scale') as HTMLInputElement;
    scaleSlider.addEventListener('input', () => {
      const v = parseFloat(scaleSlider.value);
      this.renderer.splatScale = v;
      document.getElementById('val-scale')!.textContent = `${v.toFixed(2)}x`;
      this.renderer.markDirty();
    });

    const cullSlider = document.getElementById('slider-cull') as HTMLInputElement;
    cullSlider.addEventListener('input', () => {
      const v = parseFloat(cullSlider.value);
      this.renderer.minContribution = v;
      document.getElementById('val-cull')!.textContent = v.toFixed(2);
      this.renderer.markDirty();
    });

    const fovSlider = document.getElementById('slider-fov') as HTMLInputElement;
    fovSlider.addEventListener('input', () => {
      const v = parseFloat(fovSlider.value);
      this.renderer.fovDeg = v;
      document.getElementById('val-fov')!.textContent = `${v}°`;
      this.renderer.markDirty();
    });

    document.getElementById('btn-turntable')!.addEventListener('click', () => {
      this.renderer.autoTurntable = !this.renderer.autoTurntable;
      this.refreshStaticTexts();
    });

    document.getElementById('btn-toggle-grid')!.addEventListener('click', () => {
      this.renderer.showGrid = !this.renderer.showGrid;
      this.refreshStaticTexts();
    });

    document.getElementById('btn-reset-cam')!.addEventListener('click', () => {
      this.renderer.resetCamera();
    });

    document.getElementById('btn-back-to-capture')!.addEventListener('click', () => {
      this.switchTab('capture');
    });

    document.getElementById('btn-export-ply')!.addEventListener('click', () => {
      if (!this.currentCloud) return;
      this.downloadBlob(exportToPlyBinary(this.currentCloud), `3dgs-model-${this.currentCloud.count}.ply`);
    });

    document.getElementById('btn-export-splat')!.addEventListener('click', () => {
      if (!this.currentCloud) return;
      this.downloadBlob(exportToSplatBinary(this.currentCloud), `3dgs-model-${this.currentCloud.count}.splat`);
    });

    const importInput = document.getElementById('input-import-splat') as HTMLInputElement;
    importInput.addEventListener('change', async () => {
      const file = importInput.files?.[0];
      if (!file) return;
      this.setBusy(true, this.t().busyImportTitle, file.name, 55);
      try {
        await new Promise((r) => setTimeout(r, 20));
        const cloud = await importSplatFile(file);
        this.loadCloudIntoViewer(cloud);
      } catch (err) {
        alert(`${(err as Error).message}`);
      } finally {
        this.setBusy(false);
        importInput.value = '';
      }
    });

    document.getElementById('btn-benchmark-250k')!.addEventListener('click', async () => {
      await this.generateStressBenchmark(250000);
    });
  }

  private switchTab(tab: 'capture' | 'render'): void {
    this.activeTab = tab;
    document.querySelectorAll('.mode-tab').forEach((b) => {
      b.classList.toggle('active', (b as HTMLElement).dataset.tab === tab);
    });
    document.getElementById('capture-stage')!.classList.toggle('active', tab === 'capture');
    document.getElementById('render-stage')!.classList.toggle('active', tab === 'render');

    if (tab === 'render') {
      this.renderer.start();
      this.renderer.markDirty();
    } else {
      this.renderer.stop();
      this.domeVis.render();
    }
  }

  private selectPreset(id: CapturePresetId): void {
    if (id === this.presetId) return;
    const compatibleFov = CAPTURE_PRESETS[this.presetId].recommendedFovDeg === CAPTURE_PRESETS[id].recommendedFovDeg;
    const previousSlots = compatibleFov ? this.slots : [];
    this.presetId = id;
    this.slots = buildSlotsForPreset(CAPTURE_PRESETS[id], previousSlots);
    this.selectedSlotIdx = this.slots.find((slot) => !slot.captured)?.index ?? 0;
    const latest = this.slots.filter((slot) => slot.photo).sort((a, b) => b.photo!.timestamp - a.photo!.timestamp)[0];
    this.lastCapturedSlotIdx = latest?.index ?? null;
    this.refreshAllUI();
    this.renderCurrentPreviewSlot();
  }

  private refreshStaticTexts(): void {
    const s = this.t();
    document.documentElement.lang = this.locale === 'zh' ? 'zh-CN' : 'en';

    document.getElementById('lang-zh')!.classList.toggle('active', this.locale === 'zh');
    document.getElementById('lang-en')!.classList.toggle('active', this.locale === 'en');

    document.getElementById('ui-app-title')!.textContent = s.appTitle;
    document.getElementById('ui-app-title-compact')!.textContent = s.appTitleCompact;
    document.getElementById('ui-app-sub')!.textContent = s.appSubtitle;
    document.getElementById('ui-tab-step1')!.textContent = s.step1Tab;
    document.getElementById('ui-tab-step2')!.textContent = s.step2Tab;
    document.getElementById('ui-tab-capture-short')!.textContent = s.captureTabShort;
    document.getElementById('ui-tab-render-short')!.textContent = s.renderTabShort;
    document.getElementById('ui-import-btn')!.textContent = s.importBtn;
    document.getElementById('ui-import-short')!.textContent = s.importShort;
    document.getElementById('ui-capture-settings')!.textContent = s.captureSettings;
    document.getElementById('ui-capture-help')!.textContent = s.captureHelp;
    document.getElementById('btn-preview-captures')!.textContent = s.btnPreviewCaptures;
    document.getElementById('btn-add-detail')!.textContent = s.btnAddDetail;

    const gpuDot = document.getElementById('gpu-status-dot')!;
    const gpuText = document.getElementById('gpu-status-text')!;
    gpuDot.classList.toggle('warn', !this.webgpuSupported);
    gpuText.textContent = this.webgpuSupported ? s.gpuReady : s.gpuFallback;

    document.getElementById('ui-demo-title')!.textContent = s.demoBannerTitle;
    document.getElementById('ui-demo-desc')!.textContent = s.demoBannerDesc;
    document.getElementById('ui-demo-btn')!.textContent = s.demoBannerBtn;

    document.getElementById('ui-choose-mode')!.textContent = s.chooseModeTitle;
    document.getElementById('ui-missing-title')!.textContent = s.missingTitle;
    document.getElementById('ui-compass-sub')!.textContent = s.compassSub;
    document.getElementById('ui-ring-progress-title')!.textContent = s.ringProgressTitle;

    document.getElementById('ui-framing-tip')!.textContent = s.framingTip;
    document.getElementById('btn-toggle-webcam')!.textContent = this.cameraLive ? s.btnStopCam : s.btnStartCam;
    const shutter = document.getElementById('btn-shutter') as HTMLButtonElement;
    shutter.textContent = this.cameraStarting ? s.btnCameraStarting : this.cameraLive ? s.btnShutterLive : s.btnShutterReady;
    shutter.disabled = this.cameraStarting;
    (document.getElementById('btn-toggle-webcam') as HTMLButtonElement).disabled = this.cameraStarting;
    document.getElementById('btn-simulate-shot')!.textContent = s.btnShutterSim;
    document.getElementById('ui-native-camera')!.textContent = s.btnNativeCamera;
    document.getElementById('ui-upload-btn')!.textContent = s.btnUploadPhotos;
    document.getElementById('btn-clear-captures')!.textContent = s.btnResetAll;
    document.getElementById('btn-retake-last')!.textContent = s.qcRetakeBtn;

    document.getElementById('ui-dome-title')!.textContent = s.domeTitle;
    document.getElementById('ui-dome-hint')!.textContent = s.domeHint;
    document.getElementById('ui-dome-legend')!.innerHTML = `
      <span><i class="legend-dot" style="background:#18e28c"></i>${s.legendDone}</span>
      <span><i class="legend-dot" style="background:#ffb020"></i>${s.legendWarn}</span>
      <span><i class="legend-dot" style="background:#00f0c8"></i>${s.legendCurrent}</span>
      <span><i class="legend-dot" style="border:1.5px solid #8ea4be"></i>${s.legendMissing}</span>
    `;

    document.getElementById('ui-tips-title')!.textContent = s.tipsCardTitle;
    document.getElementById('ui-tips-list')!.innerHTML = s.tipsList.map((li) => `<li>${li}</li>`).join('');

    // Stage 2 labels
    document.getElementById('ui-hud-title')!.textContent = s.hudTitle;
    document.getElementById('ui-hud-fps')!.textContent = s.hudFps;
    document.getElementById('ui-hud-total')!.textContent = s.hudTotal;
    document.getElementById('ui-hud-visible')!.textContent = s.hudVisible;
    document.getElementById('ui-hud-sort')!.textContent = s.hudSort;
    document.getElementById('ui-hud-vram')!.textContent = s.hudVram;
    document.getElementById('ui-viewport-hint')!.textContent = s.viewportControlsHint;

    document.getElementById('ui-view-mode-title')!.textContent = s.viewModeTitle;
    document.getElementById('ui-mode-0')!.textContent = s.modeSplats;
    document.getElementById('ui-mode-1')!.textContent = s.modeRings;
    document.getElementById('ui-mode-2')!.textContent = s.modeDepth;
    document.getElementById('ui-mode-3')!.textContent = s.modeCenters;

    document.getElementById('ui-gpu-params-title')!.textContent = s.gpuParamsTitle;
    document.getElementById('ui-param-scale')!.textContent = s.paramScale;
    document.getElementById('ui-param-cull')!.textContent = s.paramCull;
    document.getElementById('ui-param-fov')!.textContent = s.paramFov;

    document.getElementById('btn-turntable')!.textContent = this.renderer.autoTurntable
      ? s.autoOrbitOn
      : s.autoOrbitOff;
    document.getElementById('btn-toggle-grid')!.textContent = this.renderer.showGrid
      ? s.gridOn
      : s.gridOff;
    document.getElementById('btn-reset-cam')!.textContent = s.resetCamBtn;

    document.getElementById('ui-export-title')!.textContent = s.exportTitle;
    document.getElementById('btn-export-ply')!.textContent = s.exportPlyBtn;
    document.getElementById('btn-export-splat')!.textContent = s.exportSplatBtn;
    document.getElementById('btn-back-to-capture')!.textContent = s.backToCaptureBtn;

    document.getElementById('ui-bench-title')!.textContent = s.proBenchTitle;
    document.getElementById('ui-bench-desc')!.textContent = s.proBenchDesc;
    document.getElementById('btn-benchmark-250k')!.textContent = s.proBenchBtn;

    const count = this.currentCloud ? this.currentCloud.count : 0;
    document.getElementById('badge-splat-count')!.textContent = `${count.toLocaleString()} ${s.splatsUnit}`;
    this.updateTelemetryHUD();
  }

  private refreshAllUI(): void {
    this.refreshStaticTexts();
    const s = this.t();
    const preset = CAPTURE_PRESETS[this.presetId];
    const report = evaluateCoverage(preset, this.slots, this.selectedSlotIdx, this.locale);

    // Top capture count badge
    document.getElementById('badge-capture-count')!.textContent = `${report.capturedCount} / ${report.requiredCount}`;
    document.getElementById('settings-capture-count')!.textContent = `${report.capturedCount} / ${report.requiredCount}`;
    document.getElementById('capture-next-actions')!.classList.toggle('hidden', !report.isReadyToRender);
    document.getElementById('ui-preview-ready')!.textContent = s.previewReady(report.capturedCount);
    (document.getElementById('btn-add-detail') as HTMLButtonElement).hidden = this.presetId !== 'minimal';

    // 1. Preset Cards
    const presetContainer = document.getElementById('preset-cards')!;
    presetContainer.innerHTML = (Object.keys(CAPTURE_PRESETS) as CapturePresetId[])
      .map((key) => {
        const p = CAPTURE_PRESETS[key];
        const sel = key === this.presetId ? 'selected' : '';
        return `
          <button class="preset-card ${sel}" data-preset="${key}">
            <div class="preset-top">
              <span class="preset-name">${p.name[this.locale]}</span>
              <span class="preset-count">${p.badge[this.locale]}</span>
            </div>
            <div class="preset-desc">${p.useCase[this.locale]}</div>
          </button>
        `;
      })
      .join('');

    presetContainer.querySelectorAll('.preset-card').forEach((card) => {
      card.addEventListener('click', () => {
        this.selectPreset((card as HTMLElement).dataset.preset as CapturePresetId);
      });
    });

    // 2. Missing Angles Summary + Interactive 8-Sector Checklist
    const summaryEl = document.getElementById('missing-summary-text')!;
    summaryEl.textContent = report.missingSummaryText;
    summaryEl.classList.toggle('all-ready', report.isReadyToRender);

    const sectorGrid = document.getElementById('sector-checklist')!;
    const sectorButtons = report.sectors
      .map((sec) => {
        const cls = sec.covered ? 'covered' : 'missing';
        const icon = sec.covered ? '✓' : '○';
        return `
          <button class="sector-pill ${cls}" data-jump="${sec.firstMissingSlotIndex}">
            <span>${sec.label}</span>
            <span>${icon}</span>
          </button>
        `;
      })
      .join('');

    const topCls = report.topRingCovered ? 'covered' : 'missing';
    const topStatus = report.topRingCovered
      ? `✓ ${s.topCapDone} (${report.topRingCapturedCount})`
      : `○ ${s.topCapMissing} (${report.topRingCapturedCount}/3+)`;

    sectorGrid.innerHTML =
      sectorButtons +
      `
      <button class="sector-pill top-span ${topCls}" data-jump="${report.firstMissingTopSlotIndex}">
        <span>${s.topCapLabel}</span>
        <span>${topStatus}</span>
      </button>
    `;

    sectorGrid.querySelectorAll('.sector-pill').forEach((pill) => {
      pill.addEventListener('click', () => {
        const jumpIdx = parseInt((pill as HTMLElement).dataset.jump || '0', 10);
        if (this.slots[jumpIdx]) {
          this.selectedSlotIdx = jumpIdx;
          this.domeVis.focusAngle(this.slots[jumpIdx].azimuthDeg, this.slots[jumpIdx].elevationDeg);
          this.refreshAllUI();
          this.renderCurrentPreviewSlot();
        }
      });
    });

    // 3. Ring Progress
    const ringsEl = document.getElementById('ring-breakdown')!;
    ringsEl.innerHTML = report.ringProgress
      .map((rp) => {
        const pct = Math.round((rp.captured / Math.max(1, rp.total)) * 100);
        const doneClass = rp.captured === rp.total ? 'complete' : '';
        return `
          <div class="ring-item">
            <div class="ring-row">
              <span>${rp.label}</span>
              <span class="ring-nums ${doneClass}">${rp.captured} / ${rp.total}</span>
            </div>
            <div class="progress-track">
              <div class="progress-fill" style="width:${pct}%"></div>
            </div>
          </div>
        `;
      })
      .join('');

    // Reconstruct Button
    const reconBtn = document.getElementById('btn-reconstruct')!;
    reconBtn.textContent = report.isReadyToRender ? s.generateBtnReady : s.generateBtnEarly;
    document.getElementById('ui-generate-hint')!.textContent = s.generateHintMin(
      report.minimumViableCount,
      report.capturedCount,
      report.isReadyToRender
    );

    // 4. Step-by-Step Coach Banner
    const activeSlot = this.slots[this.selectedSlotIdx] || this.slots[0];
    const azDesc = describeAzimuth(activeSlot.azimuthDeg, this.locale);
    const elDesc = describeElevation(activeSlot.elevationDeg, this.locale);
    const ringSpec = getRingSpec(preset, activeSlot.ringId);

    if (report.capturedCount === report.requiredCount) {
      document.getElementById('coach-badge')!.textContent = s.allDoneBadge;
    } else {
      document.getElementById('coach-badge')!.textContent = `${s.nextActionBadge} · ${ringSpec.label[this.locale]}`;
    }

    document.getElementById('coach-where')!.textContent =
      `${s.standAtLabel}：${azDesc.direction}（${azDesc.clock} / ${activeSlot.azimuthDeg}°）`;
    document.getElementById('coach-height')!.textContent = `${s.cameraHeightLabel}：${elDesc}`;
    document.getElementById('coach-counter')!.innerHTML =
      `#${activeSlot.index + 1} / ${this.slots.length}<br/><span style="color:var(--accent-cyan)">${report.completionPct}%</span>`;

    // 5. Real-Time Photo Quality Inspector Card (shows feedback on the photo just taken, or currently inspected captured photo)
    const inspectedSlot =
      this.slots[this.selectedSlotIdx]?.photo
        ? this.slots[this.selectedSlotIdx]
        : this.lastCapturedSlotIdx !== null && this.slots[this.lastCapturedSlotIdx]?.photo
          ? this.slots[this.lastCapturedSlotIdx]
          : null;

    const qcCard = document.getElementById('qc-card')!;
    const qcThumb = document.getElementById('qc-thumb') as HTMLImageElement;
    const qcTitle = document.getElementById('qc-title')!;
    const qcDesc = document.getElementById('qc-desc')!;
    const qcMetrics = document.getElementById('qc-metrics')!;
    const qcActions = document.getElementById('qc-actions')!;

    qcCard.classList.remove('good', 'warn', 'bad');

    if (inspectedSlot && inspectedSlot.photo) {
      const p = inspectedSlot.photo;
      qcCard.classList.add(p.verdict);
      qcThumb.src = p.dataUrl;
      qcThumb.style.display = 'block';

      qcTitle.textContent =
        p.verdict === 'good'
          ? s.qcGoodTitle
          : p.verdict === 'warn'
            ? s.qcWarnTitle
            : s.qcBadTitle;
      qcDesc.textContent = p.summaryText[this.locale];
      qcMetrics.innerHTML = `
        <span>${s.qcMetricSharp}: <strong>${p.sharpnessScore}</strong>/100</span>
        <span>·</span>
        <span>${s.qcMetricExp}: <strong>${p.exposureScore}</strong>/100</span>
        <span>·</span>
        <span>${s.qcMetricSubject}: <strong>${Math.round(p.foregroundRatio * 100)}%</strong></span>
      `;
      this.lastCapturedSlotIdx = inspectedSlot.index;
      qcActions.style.display = 'block';
    } else {
      qcThumb.style.display = 'none';
      qcTitle.textContent = s.qcWaitingTitle;
      qcDesc.textContent = s.qcWaitingDesc;
      qcMetrics.innerHTML = '';
      qcActions.style.display = 'none';
    }

    // 6. Filmstrip
    const stripEl = document.getElementById('capture-filmstrip')!;
    stripEl.innerHTML = this.slots
      .map((slot) => {
        const vCls = slot.photo ? `captured-${slot.photo.verdict}` : '';
        const selCls = slot.index === this.selectedSlotIdx ? 'selected' : '';
        const badgeIcon = slot.photo
          ? slot.photo.verdict === 'good'
            ? '✅'
            : slot.photo.verdict === 'warn'
              ? '⚠️'
              : '❌'
          : '';
        const inner = slot.photo
          ? `<img src="${slot.photo.dataUrl}" alt="#${slot.index + 1}" /><span class="thumb-badge">${badgeIcon}</span>`
          : `<span style="font-family:var(--font-mono); font-size:11px; color:var(--text-muted);">#${slot.index + 1}</span>`;
        return `
          <div class="thumb-slot ${vCls} ${selCls}" data-slot="${slot.index}">
            ${inner}
            <div class="thumb-label">${Math.round(slot.azimuthDeg)}°/${slot.elevationDeg}°</div>
          </div>
        `;
      })
      .join('');

    stripEl.querySelectorAll('.thumb-slot').forEach((el) => {
      el.addEventListener('click', () => {
        this.selectedSlotIdx = parseInt((el as HTMLElement).dataset.slot || '0', 10);
        const target = this.slots[this.selectedSlotIdx];
        this.domeVis.focusAngle(target.azimuthDeg, target.elevationDeg);
        this.refreshAllUI();
        this.renderCurrentPreviewSlot();
      });
    });

    // 7. 3D Dome
    this.domeVis.setState(
      this.slots,
      this.selectedSlotIdx,
      report.nextRecommendedSlot ? report.nextRecommendedSlot.index : -1,
      this.locale
    );
  }

  private updateTelemetryHUD(): void {
    if (!this.latestStats) return;
    const s = this.t();
    const st = this.latestStats;
    document.getElementById('stat-fps')!.textContent = `${st.fps} FPS (${st.frameTimeMs.toFixed(1)} ms)`;
    document.getElementById('stat-total')!.textContent = st.totalSplats.toLocaleString();
    document.getElementById('stat-visible')!.textContent = `${st.visibleSplats.toLocaleString()} (-${st.culledPct}% ${s.culledSuffix})`;
    document.getElementById('stat-sort')!.textContent = st.sortSkipped ? s.sortCached : s.sortActive;
    document.getElementById('stat-vram')!.textContent = `${st.vramMB} MB`;
  }

  private setCameraFeedback(message: string): void {
    const feedback = document.getElementById('camera-feedback')!;
    feedback.textContent = message;
    feedback.classList.toggle('hidden', !message);
  }

  private async toggleWebcam(): Promise<void> {
    if (this.cameraStarting) return;
    const video = document.getElementById('camera-video') as HTMLVideoElement;
    const previewCanvas = document.getElementById('preview-photo-canvas') as HTMLCanvasElement;

    if (this.cameraLive) {
      this.stream?.getTracks().forEach((t) => t.stop());
      this.stream = null;
      video.srcObject = null;
      this.cameraLive = false;
      this.setCameraFeedback('');
      video.classList.add('hidden');
      previewCanvas.classList.remove('hidden');
      this.refreshAllUI();
      this.renderCurrentPreviewSlot();
      return;
    }

    // Keep the native camera picker available on phones without live camera support.
    if (!navigator.mediaDevices?.getUserMedia) {
      this.setCameraFeedback(this.t().cameraUnavailable);
      (document.getElementById('input-camera-photo') as HTMLInputElement).click();
      return;
    }

    this.cameraStarting = true;
    this.setCameraFeedback('');
    this.refreshStaticTexts();
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 960 } },
        audio: false
      });
      video.srcObject = this.stream;
      await video.play();
      this.cameraLive = true;
      video.classList.remove('hidden');
      previewCanvas.classList.add('hidden');
    } catch {
      this.stream?.getTracks().forEach((track) => track.stop());
      this.stream = null;
      video.srcObject = null;
      this.setCameraFeedback(this.t().cameraUnavailable);
    } finally {
      this.cameraStarting = false;
      this.refreshAllUI();
    }
  }

  private captureFromWebcam(): void {
    const video = document.getElementById('camera-video') as HTMLVideoElement;
    if (!video.videoWidth) return;

    const workSize = 256;
    const off = document.createElement('canvas');
    off.width = workSize;
    off.height = workSize;
    const ctx = off.getContext('2d')!;

    const side = Math.min(video.videoWidth, video.videoHeight);
    const sx = (video.videoWidth - side) * 0.5;
    const sy = (video.videoHeight - side) * 0.5;
    ctx.drawImage(video, sx, sy, side, side, 0, 0, workSize, workSize);

    const imgData = ctx.getImageData(0, 0, workSize, workSize);
    this.storeCaptureInSlot(this.selectedSlotIdx, imgData.data, workSize, workSize, off.toDataURL('image/jpeg', 0.88));
    this.advanceAfterCapture();
  }

  private async handleUploadedPhotos(files: File[]): Promise<void> {
    const workSize = 256;
    const off = document.createElement('canvas');
    off.width = workSize;
    off.height = workSize;
    const ctx = off.getContext('2d')!;

    for (const file of files) {
      const bitmap = await createImageBitmap(file);
      const side = Math.min(bitmap.width, bitmap.height);
      const sx = (bitmap.width - side) * 0.5;
      const sy = (bitmap.height - side) * 0.5;
      ctx.clearRect(0, 0, workSize, workSize);
      ctx.drawImage(bitmap, sx, sy, side, side, 0, 0, workSize, workSize);
      bitmap.close();

      const imgData = ctx.getImageData(0, 0, workSize, workSize);
      this.storeCaptureInSlot(
        this.selectedSlotIdx,
        imgData.data,
        workSize,
        workSize,
        off.toDataURL('image/jpeg', 0.88)
      );
      this.advanceAfterCapture(false);
    }

    this.refreshAllUI();
    this.renderCurrentPreviewSlot();
  }

  private captureSyntheticAt(slotIdx: number, advance = true): void {
    const slot = this.slots[slotIdx];
    if (!slot) return;
    const preset = CAPTURE_PRESETS[this.presetId];
    const workSize = 224;
    const { rgba, dataUrl } = renderSyntheticCaptureView(
      slot.azimuthDeg,
      slot.elevationDeg,
      preset.recommendedFovDeg,
      workSize,
      workSize
    );
    this.storeCaptureInSlot(slotIdx, rgba, workSize, workSize, dataUrl);
    if (advance) {
      this.advanceAfterCapture();
    }
  }

  private storeCaptureInSlot(
    slotIdx: number,
    rgba: Uint8ClampedArray,
    w: number,
    h: number,
    dataUrl: string
  ): void {
    const prevPhoto =
      this.lastCapturedSlotIdx !== null && this.lastCapturedSlotIdx !== slotIdx
        ? this.slots[this.lastCapturedSlotIdx]?.photo
        : null;

    const qc = inspectCapturedPhoto(rgba, w, h, prevPhoto);
    const slot = this.slots[slotIdx];

    const photo: CapturedPhoto = {
      slotIndex: slotIdx,
      azimuthDeg: slot.azimuthDeg,
      elevationDeg: slot.elevationDeg,
      width: w,
      height: h,
      dataUrl,
      rgba,
      workWidth: w,
      workHeight: h,
      sharpnessScore: qc.sharpnessScore,
      exposureScore: qc.exposureScore,
      meanLuma: qc.meanLuma,
      foregroundRatio: qc.foregroundRatio,
      centerOffset: qc.centerOffset,
      similarityToPrev: qc.similarityToPrev,
      verdict: qc.verdict,
      issues: qc.issues,
      summaryText: qc.summaryText,
      timestamp: Date.now()
    };

    slot.captured = true;
    slot.photo = photo;
    this.lastCapturedSlotIdx = slotIdx;
  }

  private advanceAfterCapture(refresh = true): void {
    const lastSlot = this.lastCapturedSlotIdx !== null ? this.slots[this.lastCapturedSlotIdx] : null;
    // If photo was severely blurry/dark ('bad'), stay on the same slot so user can immediately retake!
    if (lastSlot?.photo?.verdict !== 'bad') {
      const preset = CAPTURE_PRESETS[this.presetId];
      const report = evaluateCoverage(preset, this.slots, this.selectedSlotIdx, this.locale);
      if (report.nextRecommendedSlot) {
        this.selectedSlotIdx = report.nextRecommendedSlot.index;
        this.domeVis.focusAngle(
          report.nextRecommendedSlot.azimuthDeg,
          report.nextRecommendedSlot.elevationDeg
        );
      }
    }

    if (refresh) {
      this.refreshAllUI();
      this.renderCurrentPreviewSlot();
    }
  }

  private renderCurrentPreviewSlot(): void {
    if (this.cameraLive) return;
    const canvas = document.getElementById('preview-photo-canvas') as HTMLCanvasElement;
    const ctx = canvas.getContext('2d')!;
    const slot = this.slots[this.selectedSlotIdx];

    ctx.fillStyle = '#070a10';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // If current slot has a photo, show it; otherwise if we just took a photo on the previous slot, show a subtle split/preview or target guide
    if (slot?.photo) {
      const img = new Image();
      img.onload = () => ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      img.src = slot.photo.dataUrl;
    } else if (slot) {
      const azDesc = describeAzimuth(slot.azimuthDeg, this.locale);
      ctx.save();
      ctx.strokeStyle = 'rgba(0, 240, 200, 0.22)';
      ctx.lineWidth = 1.6;
      ctx.translate(canvas.width / 2, canvas.height / 2);
      for (let r = 48; r <= 148; r += 34) {
        ctx.beginPath();
        ctx.ellipse(0, 0, r, r * 0.68, (slot.azimuthDeg * Math.PI) / 180, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.fillStyle = '#f1f6fc';
      ctx.font = '600 16px "Plus Jakarta Sans", "Noto Sans SC", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        this.locale === 'zh'
          ? `准备拍摄第 #${slot.index + 1} 张：${azDesc.direction} (${azDesc.clock})`
          : `Ready for Shot #${slot.index + 1}: ${azDesc.direction} (${azDesc.clock})`,
        0,
        -6
      );
      ctx.fillStyle = '#ffb020';
      ctx.font = '500 13px "Plus Jakarta Sans", "Noto Sans SC", sans-serif';
      ctx.fillText(describeElevation(slot.elevationDeg, this.locale), 0, 20);
      ctx.restore();
    }
  }

  private async simulateFullDomeAndRender(): Promise<void> {
    const s = this.t();
    this.setBusy(true, s.busySimTitle, '0%', 5);
    const total = this.slots.length;
    for (let i = 0; i < total; i++) {
      if (!this.slots[i].captured) {
        this.captureSyntheticAt(i, false);
      }
      if (i % 4 === 0) {
        const pct = Math.round(((i + 1) / total) * 45);
        this.setBusy(
          true,
          s.busySimTitle,
          `${i + 1} / ${total} (${this.slots[i].azimuthDeg}°, ${this.slots[i].elevationDeg}°)`,
          pct
        );
        await new Promise((r) => setTimeout(r, 8));
      }
    }
    this.refreshAllUI();
    this.renderCurrentPreviewSlot();
    await this.runReconstruction();
  }

  private async runReconstruction(): Promise<void> {
    const capturedCount = this.slots.filter((s) => s.captured).length;
    if (capturedCount === 0) {
      await this.simulateFullDomeAndRender();
      return;
    }

    const s = this.t();
    const preset = CAPTURE_PRESETS[this.presetId];
    this.setBusy(true, s.busyReconTitle, '...', 10);

    try {
      const cloud = await reconstructSplatsFromCaptures(
        this.slots,
        preset.recommendedFovDeg,
        'high',
        (stage, pct) => {
          this.setBusy(true, s.busyReconTitle, stage, pct);
        }
      );
      this.loadCloudIntoViewer(cloud);
    } finally {
      this.setBusy(false);
    }
  }

  private loadCloudIntoViewer(cloud: SplatCloudData): void {
    this.currentCloud = cloud;
    this.renderer.setSplatCloud(cloud);
    this.renderer.autoTurntable = true;
    this.refreshStaticTexts();
    this.switchTab('render');
  }

  private async generateStressBenchmark(count: number): Promise<void> {
    this.setBusy(this.locale === 'zh' || true, this.t().busyBenchTitle, '250,000 Gaussians', 40);
    await new Promise((r) => setTimeout(r, 20));

    const positions = new Float32Array(count * 3);
    const scales = new Float32Array(count * 3);
    const quats = new Float32Array(count * 4);
    const colors = new Float32Array(count * 4);

    for (let i = 0; i < count; i++) {
      const t = (i / count) * Math.PI * 2 * 7;
      const p = 3;
      const q = 4;
      const rTube = 0.22;
      const rCore = 0.62 + 0.24 * Math.cos(q * t);

      const cx = rCore * Math.cos(p * t);
      const cy = 0.28 * Math.sin(q * t);
      const cz = rCore * Math.sin(p * t);

      const ringAngle = (i * 2.3999632) % (Math.PI * 2);
      const radial = rTube * (0.65 + 0.35 * (((i * 7919) % 100) / 100));
      const nx = Math.cos(ringAngle) * Math.cos(p * t);
      const ny = Math.sin(ringAngle);
      const nz = Math.cos(ringAngle) * Math.sin(p * t);

      positions[i * 3] = cx + nx * radial;
      positions[i * 3 + 1] = cy + ny * radial;
      positions[i * 3 + 2] = cz + nz * radial;

      scales[i * 3] = 0.0085;
      scales[i * 3 + 1] = 0.0065;
      scales[i * 3 + 2] = 0.0028;

      quats[i * 4] = 0;
      quats[i * 4 + 1] = 0;
      quats[i * 4 + 2] = 0;
      quats[i * 4 + 3] = 1;

      colors[i * 4] = 0.5 + 0.48 * Math.sin(t * 0.5);
      colors[i * 4 + 1] = 0.5 + 0.48 * Math.sin(t * 0.5 + 2.09);
      colors[i * 4 + 2] = 0.5 + 0.48 * Math.sin(t * 0.5 + 4.18);
      colors[i * 4 + 3] = 0.86;
    }

    const cloud = packSplatsForGPU(positions, scales, quats, colors, count, true);
    this.setBusy(false);
    this.loadCloudIntoViewer(cloud);
  }

  private setBusy(visible: boolean, title = '', stage = '', pct = 0): void {
    const el = document.getElementById('busy-overlay')!;
    el.classList.toggle('hidden', !visible);
    if (visible) {
      document.getElementById('busy-title')!.textContent = title;
      document.getElementById('busy-stage')!.textContent = stage;
      document.getElementById('busy-bar')!.style.width = `${pct}%`;
    }
  }

  private downloadBlob(blob: Blob, filename: string): void {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1500);
  }
}

const app = new GuidedGSplatApp();
app.mount(document.getElementById('app')!);
