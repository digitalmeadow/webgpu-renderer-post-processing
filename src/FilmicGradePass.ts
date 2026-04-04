import shader from "./FilmicGradePass.wgsl?raw";
import { PostPass, PostPassContext } from "@digitalmeadow/webgpu-renderer";

export interface FilmicGradePassOptions {
  /**
   * Lift: shifts the black point per RGB channel.
   * Positive values add colour to shadows.
   * Super 16 default: slight cool push in shadows [0.0, 0.01, 0.02].
   */
  lift?: [number, number, number];

  /**
   * Gamma: power curve on midtones per RGB channel.
   * 1.0 = neutral. <1.0 brightens, >1.0 darkens.
   * Default: neutral [1.0, 1.0, 1.0].
   */
  gamma?: [number, number, number];

  /**
   * Gain: scales the white point per RGB channel.
   * Super 16 default: slightly warm highlights [1.05, 1.0, 0.95].
   */
  gain?: [number, number, number];

  /**
   * Saturation: 1.0 = neutral, 0.0 = greyscale.
   * Super 16 default: 0.85 (slightly desaturated, filmic).
   */
  saturation?: number;

  /**
   * S-curve contrast strength. 0.0 = no effect, 1.0 = full sigmoid.
   * Super 16 default: 0.4 (gentle film contrast).
   */
  contrast?: number;

  /**
   * Vignette intensity. 0.0 = none, 1.0 = strong corner darkening.
   * Super 16 default: 0.4 (subtle lens falloff).
   */
  vignette?: number;
}

// Uniform buffer layout (bytes):
//   lift   vec4<f32>  → 16 bytes  (offset 0)
//   gamma  vec4<f32>  → 16 bytes  (offset 16)
//   gain   vec4<f32>  → 16 bytes  (offset 32)
//   saturation f32    →  4 bytes  (offset 48)
//   contrast   f32    →  4 bytes  (offset 52)
//   vignette   f32    →  4 bytes  (offset 56)
//   padding    f32    →  4 bytes  (offset 60)
// Total: 64 bytes
const UNIFORMS_SIZE = 64;

export class FilmicGradePass extends PostPass {
  private device: GPUDevice;

  private pipeline: GPURenderPipeline;
  private sampler: GPUSampler;
  private uniformsBuffer: GPUBuffer;
  private bindGroupLayout: GPUBindGroupLayout;

  private options: Required<FilmicGradePassOptions>;

  constructor(device: GPUDevice, options: FilmicGradePassOptions = {}) {
    super();
    this.device = device;

    // Super 16 film defaults
    this.options = {
      lift: options.lift ?? [0.0, 0.01, 0.02], // cool shadow push
      gamma: options.gamma ?? [1.0, 1.0, 1.0], // neutral midtones
      gain: options.gain ?? [1.05, 1.0, 0.95], // warm highlights
      saturation: options.saturation ?? 0.85, // slightly desaturated
      contrast: options.contrast ?? 0.4, // gentle S-curve
      vignette: options.vignette ?? 0.4, // subtle lens falloff
    };

    const shaderModule = device.createShaderModule({
      label: "Filmic Grade Pass Shader",
      code: shader,
    });

    this.sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });

    this.bindGroupLayout = device.createBindGroupLayout({
      label: "Filmic Grade Pass Bind Group Layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: "filtering" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "float", viewDimension: "2d" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
      ],
    });

    this.uniformsBuffer = device.createBuffer({
      label: "Filmic Grade Pass Uniforms",
      size: UNIFORMS_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout],
    });

    this.pipeline = device.createRenderPipeline({
      label: "Filmic Grade Pass Pipeline",
      layout: pipelineLayout,
      vertex: { module: shaderModule, entryPoint: "vs_main" },
      fragment: {
        module: shaderModule,
        entryPoint: "fs_main",
        targets: [{ format: "rgba16float" }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  // Individual property setters for runtime tweaking
  set lift(value: [number, number, number]) {
    this.options.lift = value;
  }
  get lift(): [number, number, number] {
    return this.options.lift;
  }

  set gamma(value: [number, number, number]) {
    this.options.gamma = value;
  }
  get gamma(): [number, number, number] {
    return this.options.gamma;
  }

  set gain(value: [number, number, number]) {
    this.options.gain = value;
  }
  get gain(): [number, number, number] {
    return this.options.gain;
  }

  set saturation(value: number) {
    this.options.saturation = value;
  }
  get saturation(): number {
    return this.options.saturation;
  }

  set contrast(value: number) {
    this.options.contrast = value;
  }
  get contrast(): number {
    return this.options.contrast;
  }

  set vignette(value: number) {
    this.options.vignette = value;
  }
  get vignette(): number {
    return this.options.vignette;
  }

  private writeUniforms(): void {
    const data = new Float32Array(16); // 64 bytes / 4 = 16 floats
    const o = this.options;

    // lift   (vec4, w = 0)
    data[0] = o.lift[0];
    data[1] = o.lift[1];
    data[2] = o.lift[2];
    data[3] = 0;
    // gamma  (vec4, w = 0)
    data[4] = o.gamma[0];
    data[5] = o.gamma[1];
    data[6] = o.gamma[2];
    data[7] = 0;
    // gain   (vec4, w = 0)
    data[8] = o.gain[0];
    data[9] = o.gain[1];
    data[10] = o.gain[2];
    data[11] = 0;
    // saturation, contrast, vignette, padding
    data[12] = o.saturation;
    data[13] = o.contrast;
    data[14] = o.vignette;
    data[15] = 0;

    this.device.queue.writeBuffer(this.uniformsBuffer, 0, data);
  }

  render(
    input: GPUTextureView,
    output: GPUTextureView,
    _context: PostPassContext,
  ): void {
    this.writeUniforms();

    const bindGroup = this.device.createBindGroup({
      label: "Filmic Grade Pass Bind Group",
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: input },
        { binding: 2, resource: { buffer: this.uniformsBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder({
      label: "Filmic Grade Pass Encoder",
    });

    const pass = commandEncoder.beginRenderPass({
      label: "Filmic Grade Render Pass",
      colorAttachments: [
        {
          view: output,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();

    this.device.queue.submit([commandEncoder.finish()]);
  }

  resize(_width: number, _height: number): void {
    // No internal render targets to resize
  }
}
