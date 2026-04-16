import shader from "./GodRaysPass.wgsl?raw";
import { PostPass, PostPassContext } from "@digitalmeadow/webgpu-renderer";

export interface GodRaysPassOptions {
  numSamples?: number;
  density?: number;
  weight?: number;
  decay?: number;
  exposure?: number;
  maxRayLength?: number;
  occlusionSmoothness?: number;
}

export class GodRaysPass extends PostPass {
  private device: GPUDevice;
  private pipeline: GPURenderPipeline;
  private sampler: GPUSampler;
  private comparisonSampler: GPUSampler; // For occlusion depth comparison with bilinear filtering
  private uniformsBuffer: GPUBuffer;
  private bindGroupLayout: GPUBindGroupLayout;
  private occlusionView: GPUTextureView | null;
  private dummyOcclusionTexture: GPUTexture | null = null;
  private options: Required<GodRaysPassOptions>;

  constructor(
    device: GPUDevice,
    cameraBindGroupLayout: GPUBindGroupLayout,
    lightingBindGroupLayout: GPUBindGroupLayout,
    occlusionTextureView: GPUTextureView | null,
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
      occlusionSmoothness: options.occlusionSmoothness ?? 0.01,
    };

    // Use provided occlusion texture or create a dummy 1x1 white texture
    if (occlusionTextureView) {
      this.occlusionView = occlusionTextureView;
    } else {
      // Create dummy 1x1 texture with max depth (no occlusion)
      this.dummyOcclusionTexture = device.createTexture({
        label: "God Rays Dummy Occlusion Texture",
        size: [1, 1],
        format: "depth32float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      this.occlusionView = this.dummyOcclusionTexture.createView();
      // Write max depth value (1.0)
      const depthData = new Float32Array([1.0]);
      device.queue.writeTexture(
        { texture: this.dummyOcclusionTexture },
        depthData,
        { bytesPerRow: 4 },
        [1, 1],
      );
    }

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

    // Comparison sampler for hardware-filtered occlusion testing
    this.comparisonSampler = device.createSampler({
      compare: "less", // Enables depth comparison mode
      magFilter: "linear", // Bilinear filtering on comparison results
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
        {
          binding: 4,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "depth", viewDimension: "2d" },
        },
        {
          binding: 5,
          visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: "comparison" },
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
  get occlusionSmoothness(): number {
    return this.options.occlusionSmoothness;
  }
  set occlusionSmoothness(value: number) {
    this.options.occlusionSmoothness = Math.max(0.0001, value);
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
        this.options.density,
        this.options.weight,
        this.options.decay,
        this.options.exposure,
        this.options.maxRayLength,
        this.options.occlusionSmoothness,
        0,
      ]),
    );

    const bindGroup = this.device.createBindGroup({
      label: "God Rays Pass Bind Group",
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: input },
        { binding: 2, resource: this.occlusionView! },
        { binding: 3, resource: { buffer: this.uniformsBuffer } },
        { binding: 4, resource: context.geometryBuffer.depthView },
        { binding: 5, resource: this.comparisonSampler },
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
