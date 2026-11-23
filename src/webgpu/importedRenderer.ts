import * as THREE from 'three';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';
import type { Vec3 } from '../math/mat4';

/** Lazy, full Gaussian rendering of original bytes, including degree 1–3 SH.
 * No source photos, point decimation or remote model URLs are used here.
 */
export class ImportedRenderer {
  private canvas = document.createElement('canvas');
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(46, 1, .01, 1000);
  private spark: SparkRenderer;
  private mesh?: SplatMesh;
  private target?: THREE.WebGLRenderTarget;
  private compositeScene = new THREE.Scene();
  private compositeCamera = new THREE.Camera();
  private compositeQuad?: THREE.Mesh<THREE.PlaneGeometry, THREE.ShaderMaterial>;
  private disposed = false;
  private resourcesReleased = false;
  private lost = false;

  constructor(parent: HTMLElement) {
    this.canvas.id = 'imported-canvas';
    this.canvas.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;display:none';
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: false, alpha: false });
    this.renderer.setClearColor(0x07090d);
    // Spark checks outputColorSpace itself. An sRGB canvas keeps trained display
    // colors; a linear output would visibly darken the juice and label.
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    // Match the standard/Brush training footprint: +0.3 covariance without
    // mip-opacity compensation, and camera-depth (rather than radial) sorting.
    this.spark = new SparkRenderer({ renderer: this.renderer, enableLod: false, accumExtSplats: true,
      preBlurAmount: .3, blurAmount: 0, sortRadial: false, maxStdDev: Math.sqrt(12), minAlpha: 1 / 255 });
    this.scene.add(this.spark);
    // Hundreds of translucent layers can accumulate rounding error when blended
    // directly into an 8-bit canvas. Match training with float RGB accumulation.
    if (this.renderer.extensions.has('EXT_color_buffer_float')) {
      this.target = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, depthBuffer: false });
      this.compositeQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
        depthTest: false, depthWrite: false, uniforms: { source: { value: this.target.texture } },
        vertexShader: 'varying vec2 vUv; void main(){vUv=uv; gl_Position=vec4(position.xy,0.,1.);}',
        fragmentShader: 'uniform sampler2D source; varying vec2 vUv; void main(){gl_FragColor=vec4(texture2D(source,vUv).rgb,1.);}'
      }));
      this.compositeScene.add(this.compositeQuad);
      const beforeRender = this.spark.onBeforeRender;
      this.spark.onBeforeRender = (...args) => {
        beforeRender.apply(this.spark, args);
        // Blend display RGB, as in standard Gaussian training. Spark otherwise
        // converts to linear RGB whenever rendering to an offscreen target.
        this.spark.uniforms.encodeLinear.value = false;
      };
    }
    this.canvas.dataset.compositing = this.target ? 'float16' : 'rgba8';
    this.canvas.addEventListener('webglcontextlost', event => {
      event.preventDefault();
      this.lost = true;
      this.canvas.dataset.error = 'context-lost';
      // Old GL handles are already invalid. Release them while the context is
      // lost, before Three restores a new generation of its resource caches.
      this.releaseResources();
    });
    this.canvas.addEventListener('webglcontextrestored', () => {
      // Reload from original bytes so workers/textures are recreated together.
      if (!this.disposed) this.canvas.dispatchEvent(new CustomEvent('modelreload', { bubbles: true }));
    });
    parent.insertBefore(this.canvas, parent.children[1] ?? null);
  }

  async load(file: File): Promise<void> {
    const bytes = await file.arrayBuffer();
    if (this.disposed || this.lost) throw new Error('Graphics context interrupted during model load');
    const mesh = new SplatMesh({ fileBytes: bytes, fileName: file.name, extSplats: true,
      enableLod: false, nonLod: true, lod: false });
    this.mesh = mesh;
    await mesh.initialized;
    if (this.disposed || this.lost) { mesh.dispose(); throw new Error('Graphics context interrupted during model load'); }
    const header = new TextDecoder().decode(bytes.slice(0, 16384)).split('end_header')[0];
    if (!file.name.toLowerCase().endsWith('.ply') || !header.includes('HELIOS')) {
      mesh.quaternion.set(1, 0, 0, 0); // COLMAP camera coordinates → viewer Y-up
    }
    this.scene.add(mesh);
    this.canvas.dataset.splats = String(mesh.numSplats);
    this.canvas.dataset.sh = String(mesh.splats?.getNumSh() ?? 0);
  }

  render(eye: Vec3, target: Vec3, fov: number, width: number, height: number, up: Vec3 = [0, 1, 0]): void {
    this.canvas.style.display = 'block';
    if (this.lost || this.disposed) return;
    if (this.canvas.width !== width || this.canvas.height !== height) this.renderer.setSize(width, height, false);
    this.camera.fov = fov; this.camera.aspect = width / height;
    this.camera.up.set(...up); this.camera.position.set(...eye); this.camera.lookAt(...target); this.camera.updateProjectionMatrix();
    if (this.target) {
      if (this.target.width !== width || this.target.height !== height) this.target.setSize(width, height);
      this.renderer.setRenderTarget(this.target);
      this.renderer.render(this.scene, this.camera);
      this.renderer.setRenderTarget(null);
      this.renderer.render(this.compositeScene, this.compositeCamera);
    } else this.renderer.render(this.scene, this.camera);
  }

  hide() { this.canvas.style.display = 'none'; }
  get count() { return this.mesh?.numSplats ?? 0; }
  get contextLost() { return this.lost; }
  private releaseResources() {
    if (this.resourcesReleased) return;
    this.resourcesReleased = true;
    this.mesh?.removeFromParent(); this.mesh?.dispose();
    this.target?.dispose(); this.compositeQuad?.geometry.dispose(); this.compositeQuad?.material.dispose();
    this.spark.dispose(); this.renderer.dispose();
  }
  dispose() {
    this.disposed = true;
    this.releaseResources(); this.renderer.forceContextLoss();
    this.canvas.remove();
  }
}
