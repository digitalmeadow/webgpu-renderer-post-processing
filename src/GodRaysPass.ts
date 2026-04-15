import shader from "./GodRaysPass.wgsl?raw";
import { PostPass, PostPassContext } from "@digitalmeadow/webgpu-renderer";

export interface GodRaysPassOptions {
  /** Number of ray march samples. Higher = better quality but slower. Range: 16-128. Default: 64 */
  numSamples?: number;

  /** Fog/atmosphere density. Range: 0.0-1.0. Default: 0.8 */
  density?: number;

  /** Per-sample brightness contribution. Range: 0.0-2.0. Default: 0.3 */
  weight?: number;

  /** Light decay along ray. Lower = faster falloff. Range: 0.8-1.0. Default: 0.95 */
  decay?: number;

  /** Final brightness multiplier. Range: 0.0-2.0. Default: 1.0 */
  exposure?: number;

  /** Maximum ray distance in screen space. Range: 0.5-2.0. Default: 1.0 */
  maxRayLength?: number;
}

export class GodRaysPass extends PostPass {
  private device: GPUDevice;

  private pipeline: GPURenderPipeline;
  private sampler: GPUSampler;
  private uniformsBuffer: GPUBuffer;
  private bindGroupLayout: GPUBindGroupLayout;

  private options: Required<GodRaysPassOptions>;

  constructor(
    device: GPUDevice,
    cameraBindGroupLayout: GPUBindGroupLayout,
    lightingBindGroupLayout: GPUBindGroupLayout,
    options: GodRaysPassOptions = {},
  ) {
    super();
    this.device = device;

    this.options = {
      numSamples: options.numSamples ?? 64,
      density: options.density ?? 0.8,
      weight: options.weight ?? 0.3,
      decay: options.decay ?? 0.95,
      exposure: options.exposure ?? 1.0,
      maxRayLength: options.maxRayLength ?? 1.0,
    };

    const shaderModule = device.createShaderModule({
      label: "God Rays Pass Shader",
      code: shader,
    });

    this.sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });

    this.bindGroupLayout = device.createBindGroupLayout({
      label: "God Rays Pass Bind Group Layout",
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
      label: "God Rays Pass Uniforms",
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [
        this.bindGroupLayout,
        cameraBindGroupLayout,
        lightingBindGroupLayout,
      ],
    });

    this.pipeline = device.createRenderPipeline({
      label: "God Rays Pass Pipeline",
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

  get numSamples(): number {
    return this.options.numSamples;
  }

  set numSamples(value: number) {
    this.options.numSamples = Math.max(1, Math.floor(value));
  }

  get density(): number {
    return this.options.density;
  }

  set density(value: number) {
    this.options.density = value;
  }

  get weight(): number {
    return this.options.weight;
  }

  set weight(value: number) {
    this.options.weight = value;
  }

  get decay(): number {
    return this.options.decay;
  }

  set decay(value: number) {
    this.options.decay = value;
  }

  get exposure(): number {
    return this.options.exposure;
  }

  set exposure(value: number) {
    this.options.exposure = value;
  }

  get maxRayLength(): number {
    return this.options.maxRayLength;
  }

  set maxRayLength(value: number) {
    this.options.maxRayLength = value;
  }

  render(
    input: GPUTextureView,
    output: GPUTextureView,
    context: PostPassContext,
  ): void {
    // Update uniforms buffer
    this.device.queue.writeBuffer(
      this.uniformsBuffer,
      0,
      new Int32Array([this.options.numSamples]),
    );
    this.device.queue.writeBuffer(
      this.uniformsBuffer,
      4,
      new Float32Array([
        this.options.density,
        this.options.weight,
        this.options.decay,
        this.options.exposure,
        this.options.maxRayLength,
        0, // padding
        0, // padding
      ]),
    );

    // Create bind group for group 0 (scene texture and uniforms)
    const bindGroup = this.device.createBindGroup({
      label: "God Rays Pass Bind Group",
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: input },
        { binding: 2, resource: { buffer: this.uniformsBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder({
      label: "God Rays Pass Encoder",
    });

    const pass = commandEncoder.beginRenderPass({
      label: "God Rays Render Pass",
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
    pass.setBindGroup(0, bindGroup); // Group 0: scene texture and uniforms
    pass.setBindGroup(1, context.cameraBindGroup); // Group 1: camera uniforms
    pass.setBindGroup(2, context.lightingBindGroup); // Group 2: lighting uniforms
    pass.draw(3);
    pass.end();

    this.device.queue.submit([commandEncoder.finish()]);
  }

  resize(_width: number, _height: number): void {
    // No internal render targets to resize
  }
}
