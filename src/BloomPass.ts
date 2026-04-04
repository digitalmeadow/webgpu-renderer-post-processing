import shader from "./BloomPass.wgsl?raw";
import { PostPass, PostPassContext } from "@digitalmeadow/webgpu-renderer";

export interface BloomPassOptions {
  threshold?: number;
  radius?: number;
  intensity?: number;
  iterations?: number;
  scale?: number;
}

export class BloomPass extends PostPass {
  private device: GPUDevice;

  private thresholdPipeline: GPURenderPipeline;
  private blurPipeline: GPURenderPipeline;
  private compositePipeline: GPURenderPipeline;

  private sampler: GPUSampler;
  private uniformsBuffer: GPUBuffer;
  private uniformsBindGroup: GPUBindGroup;
  private bindGroupLayout: GPUBindGroupLayout;

  private thresholdTexture: GPUTexture;
  private thresholdView: GPUTextureView;

  private blurTextureA: GPUTexture;
  private blurViewA: GPUTextureView;
  private blurTextureB: GPUTexture;
  private blurViewB: GPUTextureView;

  private options: Required<BloomPassOptions>;
  private lastWidth: number = 0;
  private lastHeight: number = 0;

  constructor(device: GPUDevice, options: BloomPassOptions = {}) {
    super();
    this.device = device;

    this.options = {
      threshold: options.threshold ?? 0.5,
      radius: options.radius ?? 8,
      intensity: options.intensity ?? 1.0,
      iterations: options.iterations ?? 2,
      scale: options.scale ?? 0.5,
    };

    const shaderModule = device.createShaderModule({ code: shader });

    this.sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });

    this.bindGroupLayout = device.createBindGroupLayout({
      label: "Bloom Pass Bind Group Layout",
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
          texture: { sampleType: "float", viewDimension: "2d" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "float", viewDimension: "2d" },
        },
        {
          binding: 5,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "float", viewDimension: "2d" },
        },
      ],
    });

    this.uniformsBuffer = device.createBuffer({
      label: "Bloom Pass Uniforms",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.uniformsBindGroup = null as unknown as GPUBindGroup;

    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout],
    });

    this.thresholdPipeline = device.createRenderPipeline({
      label: "Bloom Threshold Pipeline",
      layout: pipelineLayout,
      vertex: { module: shaderModule, entryPoint: "vs_main" },
      fragment: {
        module: shaderModule,
        entryPoint: "fs_threshold",
        targets: [{ format: "rgba16float" }],
      },
      primitive: { topology: "triangle-list" },
    });

    this.blurPipeline = device.createRenderPipeline({
      label: "Bloom Blur Pipeline",
      layout: pipelineLayout,
      vertex: { module: shaderModule, entryPoint: "vs_main" },
      fragment: {
        module: shaderModule,
        entryPoint: "fs_blur",
        targets: [{ format: "rgba16float" }],
      },
      primitive: { topology: "triangle-list" },
    });

    this.compositePipeline = device.createRenderPipeline({
      label: "Bloom Composite Pipeline",
      layout: pipelineLayout,
      vertex: { module: shaderModule, entryPoint: "vs_main" },
      fragment: {
        module: shaderModule,
        entryPoint: "fs_composite",
        targets: [{ format: "rgba16float" }],
      },
      primitive: { topology: "triangle-list" },
    });

    this.thresholdTexture = null as unknown as GPUTexture;
    this.thresholdView = null as unknown as GPUTextureView;
    this.blurTextureA = null as unknown as GPUTexture;
    this.blurViewA = null as unknown as GPUTextureView;
    this.blurTextureB = null as unknown as GPUTexture;
    this.blurViewB = null as unknown as GPUTextureView;
  }

  private createRenderTargets(width: number, height: number): void {
    const scaledWidth = Math.floor(width * this.options.scale);
    const scaledHeight = Math.floor(height * this.options.scale);

    if (this.lastWidth === scaledWidth && this.lastHeight === scaledHeight)
      return;

    this.lastWidth = scaledWidth;
    this.lastHeight = scaledHeight;

    if (this.thresholdTexture) this.thresholdTexture.destroy();
    if (this.blurTextureA) this.blurTextureA.destroy();
    if (this.blurTextureB) this.blurTextureB.destroy();

    const createTexture = (label: string, w: number, h: number) =>
      this.device.createTexture({
        label,
        size: [w, h],
        format: "rgba16float",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.RENDER_ATTACHMENT |
          GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.COPY_DST,
      });

    this.thresholdTexture = createTexture(
      "Bloom Threshold Texture",
      scaledWidth,
      scaledHeight,
    );
    this.thresholdView = this.thresholdTexture.createView();

    this.blurTextureA = createTexture(
      "Bloom Blur Texture A",
      scaledWidth,
      scaledHeight,
    );
    this.blurViewA = this.blurTextureA.createView();

    this.blurTextureB = createTexture(
      "Bloom Blur Texture B",
      scaledWidth,
      scaledHeight,
    );
    this.blurViewB = this.blurTextureB.createView();
  }

  render(
    input: GPUTextureView,
    output: GPUTextureView,
    context: PostPassContext,
  ): void {
    const { width, height } = context;
    this.createRenderTargets(width, height);

    this.device.queue.writeBuffer(
      this.uniformsBuffer,
      0,
      new Float32Array([
        this.options.threshold,
        this.options.radius,
        this.options.intensity,
        0,
      ]),
    );

    const commandEncoder = this.device.createCommandEncoder();

    const gbufferView = context.geometryBuffer.metalRoughnessView;
    const emissiveView = context.geometryBuffer.emissiveView;

    const gbufferBindGroup = this.device.createBindGroup({
      label: "Bloom G-Buffer Bind Group",
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: gbufferView },
        { binding: 2, resource: input },
        { binding: 3, resource: { buffer: this.uniformsBuffer } },
        { binding: 4, resource: input },
        { binding: 5, resource: emissiveView },
      ],
    });

    const thresholdPass = commandEncoder.beginRenderPass({
      label: "Bloom Threshold Pass",
      colorAttachments: [
        {
          view: this.thresholdView,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    thresholdPass.setPipeline(this.thresholdPipeline);
    thresholdPass.setBindGroup(0, gbufferBindGroup);
    thresholdPass.draw(3);
    thresholdPass.end();

    let readView = this.thresholdView;
    let writeView = this.blurViewA;

    for (let i = 0; i < this.options.iterations; i++) {
      const blurBindGroup = this.device.createBindGroup({
        label: `Bloom Blur Bind Group ${i}`,
        layout: this.bindGroupLayout,
        entries: [
          { binding: 0, resource: this.sampler },
          { binding: 1, resource: gbufferView },
          { binding: 2, resource: readView },
          { binding: 3, resource: { buffer: this.uniformsBuffer } },
          { binding: 4, resource: input },
          { binding: 5, resource: emissiveView },
        ],
      });

      const blurPass = commandEncoder.beginRenderPass({
        label: `Bloom Blur Pass ${i}`,
        colorAttachments: [
          {
            view: writeView,
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });

      blurPass.setPipeline(this.blurPipeline);
      blurPass.setBindGroup(0, blurBindGroup);
      blurPass.draw(3);
      blurPass.end();

      const temp = readView;
      readView = writeView;
      writeView = temp;
    }

    const compositeBindGroup = this.device.createBindGroup({
      label: "Bloom Composite Bind Group",
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: gbufferView },
        { binding: 2, resource: readView },
        { binding: 3, resource: { buffer: this.uniformsBuffer } },
        { binding: 4, resource: input },
        { binding: 5, resource: emissiveView },
      ],
    });

    const compositePass = commandEncoder.beginRenderPass({
      label: "Bloom Composite Pass",
      colorAttachments: [
        {
          view: output,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    compositePass.setPipeline(this.compositePipeline);
    compositePass.setBindGroup(0, compositeBindGroup);
    compositePass.draw(3);
    compositePass.end();

    this.device.queue.submit([commandEncoder.finish()]);
  }

  resize(width: number, height: number): void {
    this.createRenderTargets(width, height);
  }
}
