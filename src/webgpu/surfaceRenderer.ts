import { nearestTextureView, type TexturedSurfaceData } from '../sfm/surface';

/** Ordinary textured mesh preview: independent of Gaussian point-budget fallbacks. */
export class SurfaceRenderer {
  private canvas = document.createElement('canvas');
  private gl: WebGL2RenderingContext | null;
  private program: WebGLProgram | null = null;
  private vao: WebGLVertexArrayObject | null = null;
  private textures: WebGLTexture[] = [];
  private surface?: TexturedSurfaceData;
  private buffers: WebGLBuffer[] = [];
  private count = 0;
  private failed = false;

  constructor(parent: HTMLElement) {
    this.canvas.id = 'surface-canvas';
    this.canvas.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;display:none';
    parent.insertBefore(this.canvas, parent.children[1] ?? null);
    this.gl = this.canvas.getContext('webgl2', { alpha: false, antialias: true });
    this.canvas.addEventListener('webglcontextlost', event => { event.preventDefault(); this.failed = true; this.hide(); });
    this.canvas.addEventListener('webglcontextrestored', () => {
      const surface = this.surface;
      this.gl = this.canvas.getContext('webgl2'); this.failed = false; this.program = null;
      this.buffers = []; this.textures = []; this.vao = null;
      this.setSurface(surface);
    });
  }

  private initialize(): boolean {
    const gl = this.gl;
    if (!gl || this.failed) return false;
    const shader = (type: number, source: string) => {
      const s = gl.createShader(type)!; gl.shaderSource(s, source); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) { gl.deleteShader(s); throw new Error('Surface shader failed'); }
      return s;
    };
    try {
      const vertex = shader(gl.VERTEX_SHADER, `#version 300 es
        layout(location=0) in vec3 position; layout(location=1) in vec2 uv;
        uniform mat4 vp; out vec2 texcoord;
        void main(){ gl_Position=vp*vec4(position,1.0); texcoord=uv; }`);
      const fragment = shader(gl.FRAGMENT_SHADER, `#version 300 es
        precision highp float; in vec2 texcoord; uniform sampler2D image; uniform float sourceAngle; out vec4 color;
        void main(){
          float delta=texcoord.x*6.28318530718-sourceAngle;
          color=texture(image,vec2(0.5+0.5*sin(delta),texcoord.y));
        }`);
      this.program = gl.createProgram()!; gl.attachShader(this.program, vertex); gl.attachShader(this.program, fragment); gl.linkProgram(this.program);
      gl.deleteShader(vertex); gl.deleteShader(fragment);
      if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw new Error('Surface link failed');
      return true;
    } catch { this.failed = true; this.hide(); return false; }
  }

  setSurface(surface?: TexturedSurfaceData): boolean {
    this.clear();
    this.surface = surface;
    const gl = this.gl;
    if (!surface?.views.length || !gl || this.failed || (!this.program && !this.initialize())) return false;
    this.vao = gl.createVertexArray(); gl.bindVertexArray(this.vao);
    const attribute = (location: number, size: number, data: Float32Array) => {
      const buffer = gl.createBuffer()!; this.buffers.push(buffer); gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(location); gl.vertexAttribPointer(location, size, gl.FLOAT, false, 0, 0);
    };
    attribute(0, 3, surface.positions); attribute(1, 2, surface.uvs);
    const indices = gl.createBuffer()!; this.buffers.push(indices); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indices); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, surface.indices, gl.STATIC_DRAW);
    for (const view of surface.views) {
      const texture = gl.createTexture()!; this.textures.push(texture); gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, view.width, view.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, view.rgba);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR); gl.generateMipmap(gl.TEXTURE_2D);
    }
    gl.bindVertexArray(null); this.count = surface.indices.length;
    return true;
  }

  render(vp: Float32Array, width: number, height: number, azimuthRad: number): boolean {
    const gl = this.gl;
    if (!gl || !this.program || !this.vao || !this.count || this.failed || !this.surface) { this.hide(); return false; }
    const index = nearestTextureView(this.surface.views, azimuthRad), view = this.surface.views[index];
    this.canvas.dataset.sourceView = String(index);
    this.canvas.dataset.sourceTime = view.timeSeconds?.toFixed(3) ?? '';
    this.canvas.dataset.textureSize = `${view.width}×${view.height}`;
    this.canvas.style.display = 'block';
    if (this.canvas.width !== width || this.canvas.height !== height) { this.canvas.width = width; this.canvas.height = height; }
    gl.viewport(0, 0, width, height); gl.clearColor(7 / 255, 9 / 255, 13 / 255, 1); gl.clearDepth(1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.disable(gl.CULL_FACE);
    gl.useProgram(this.program); gl.uniformMatrix4fv(gl.getUniformLocation(this.program, 'vp'), false, vp);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.textures[index]); gl.uniform1i(gl.getUniformLocation(this.program, 'image'), 0);
    gl.uniform1f(gl.getUniformLocation(this.program, 'sourceAngle'), view.azimuthRad);
    gl.bindVertexArray(this.vao); gl.drawElements(gl.TRIANGLES, this.count, gl.UNSIGNED_INT, 0); gl.bindVertexArray(null);
    return true;
  }

  hide() { this.canvas.style.display = 'none'; }
  get ready() { return !!this.gl && !!this.program && !!this.vao && this.count > 0 && !this.failed; }
  get vramMB() {
    return this.surface ? (this.surface.positions.byteLength + this.surface.uvs.byteLength + this.surface.indices.byteLength + this.surface.views.reduce((sum, view) => sum + view.rgba.byteLength * 4 / 3, 0)) / 1048576 : 0;
  }
  clear() {
    const gl = this.gl;
    if (gl) { for (const b of this.buffers) gl.deleteBuffer(b); if (this.vao) gl.deleteVertexArray(this.vao); for (const texture of this.textures) gl.deleteTexture(texture); }
    this.buffers = []; this.vao = null; this.textures = []; this.surface = undefined; this.count = 0;
    delete this.canvas.dataset.sourceView; delete this.canvas.dataset.sourceTime; delete this.canvas.dataset.textureSize;
    this.hide();
  }
}
