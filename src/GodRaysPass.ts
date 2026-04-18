import shader from "./GodRaysPass.wgsl?raw";
import { PostPass, PostPassContext } from "@digitalmeadow/webgpu-renderer";

export interface GodRaysPassOptions {
  numSamples?: number;
  intensity?: number;
  decay?: number;
  maxRayDistance?: number;
  sunRadius?: number;
  angleFalloff?: number;
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
      numSamples: options.numSamples ?? 32,
      intensity: options.intensity ?? 1.0,
      decay: options.decay ?? 0.95,
      maxRayDistance: options.maxRayDistance ?? 1.0,
      sunRadius: options.sunRadius ?? 0.1,
      angleFalloff: options.angleFalloff ?? 0.8,
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
          texture: { sampleType: "depth", viewDimension: "2d" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
      ],
    });

    this.uniformsBuffer = device.createBuffer({
      label: "God Rays Pass Uniforms",
      size: 24,
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
  get intensity(): number {
    return this.options.intensity;
  }
  set intensity(value: number) {
    this.options.intensity = value;
  }
  get decay(): number {
    return this.options.decay;
  }
  set decay(value: number) {
    this.options.decay = value;
  }
  get maxRayDistance(): number {
    return this.options.maxRayDistance;
  }
  set maxRayDistance(value: number) {
    this.options.maxRayDistance = value;
  }
  get sunRadius(): number {
    return this.options.sunRadius;
  }
  set sunRadius(value: number) {
    this.options.sunRadius = Math.max(0.0, value);
  }
  get angleFalloff(): number {
    return this.options.angleFalloff;
  }
  set angleFalloff(value: number) {
    this.options.angleFalloff = Math.max(0.0, Math.min(1.0, value));
  }

  render(
    input: GPUTextureView,
    output: GPUTextureView,
    context: PostPassContext,
  ): void {
    this.device.queue.writeBuffer(
      this.uniformsBuffer,
      0,
      new Int32Array([this.options.numSamples]),
    );
    this.device.queue.writeBuffer(
      this.uniformsBuffer,
      4,
      new Float32Array([
        this.options.intensity,
        this.options.decay,
        this.options.maxRayDistance,
        this.options.sunRadius,
        this.options.angleFalloff,
      ]),
    );

    const bindGroup = this.device.createBindGroup({
      label: "God Rays Pass Bind Group",
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: input },
        { binding: 2, resource: context.geometryBuffer.depthView },
        { binding: 3, resource: { buffer: this.uniformsBuffer } },
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
    pass.setBindGroup(0, bindGroup);
    pass.setBindGroup(1, context.cameraBindGroup);
    pass.setBindGroup(2, context.lightingBindGroup);
    pass.draw(3);
    pass.end();

    this.device.queue.submit([commandEncoder.finish()]);
  }

  resize(width: number, height: number): void {
    // No internal render targets to resize
  }
}
