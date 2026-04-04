import shader from "./TonemapPass.wgsl?raw";
import { PostPass, PostPassContext } from "@digitalmeadow/webgpu-renderer";

export interface TonemapPassOptions {
  /** Scene exposure multiplier applied before tonemapping. Default: 1.0 */
  exposure?: number;
}

export class TonemapPass extends PostPass {
  private device: GPUDevice;

  private pipeline: GPURenderPipeline;
  private sampler: GPUSampler;
  private uniformsBuffer: GPUBuffer;
  private bindGroupLayout: GPUBindGroupLayout;

  private options: Required<TonemapPassOptions>;

  constructor(device: GPUDevice, options: TonemapPassOptions = {}) {
    super();
    this.device = device;

    this.options = {
      exposure: options.exposure ?? 1.0,
    };

    const shaderModule = device.createShaderModule({
      label: "Tonemap Pass Shader",
      code: shader,
    });

    this.sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });

    this.bindGroupLayout = device.createBindGroupLayout({
      label: "Tonemap Pass Bind Group Layout",
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
      label: "Tonemap Pass Uniforms",
      size: 16, // 4 x f32 (exposure + 3 padding)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout],
    });

    this.pipeline = device.createRenderPipeline({
      label: "Tonemap Pass Pipeline",
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

  get exposure(): number {
    return this.options.exposure;
  }

  set exposure(value: number) {
    this.options.exposure = value;
  }

  render(
    input: GPUTextureView,
    output: GPUTextureView,
    _context: PostPassContext,
  ): void {
    this.device.queue.writeBuffer(
      this.uniformsBuffer,
      0,
      new Float32Array([this.options.exposure, 0, 0, 0]),
    );

    const bindGroup = this.device.createBindGroup({
      label: "Tonemap Pass Bind Group",
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: input },
        { binding: 2, resource: { buffer: this.uniformsBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder({
      label: "Tonemap Pass Encoder",
    });

    const pass = commandEncoder.beginRenderPass({
      label: "Tonemap Render Pass",
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
