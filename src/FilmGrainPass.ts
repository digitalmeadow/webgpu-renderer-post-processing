import shader from "./FilmGrainPass.wgsl?raw";
import { PostPass, PostPassContext } from "@digitalmeadow/webgpu-renderer";

export interface FilmGrainPassOptions {
  intensity?: number;
}

export class FilmGrainPass extends PostPass {
  private device: GPUDevice;

  private pipeline: GPURenderPipeline;
  private sampler: GPUSampler;
  private uniformsBuffer: GPUBuffer;
  private bindGroupLayout: GPUBindGroupLayout;

  private options: Required<FilmGrainPassOptions>;
  private time: number = 0;
  private lastWidth: number = 0;
  private lastHeight: number = 0;

  constructor(device: GPUDevice, options: FilmGrainPassOptions = {}) {
    super();
    this.device = device;

    this.options = {
      intensity: options.intensity ?? 0.15,
    };

    const shaderModule = device.createShaderModule({ code: shader });

    this.sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });

    this.bindGroupLayout = device.createBindGroupLayout({
      label: "Film Grain Pass Bind Group Layout",
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
      label: "Film Grain Pass Uniforms",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout],
    });

    this.pipeline = device.createRenderPipeline({
      label: "Film Grain Pass Pipeline",
      layout: pipelineLayout,
      vertex: { module: shaderModule, entryPoint: "vs_main" },
      fragment: {
        module: shaderModule,
        entryPoint: "fs_main",
        targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  render(
    input: GPUTextureView,
    output: GPUTextureView,
    context: PostPassContext,
  ): void {
    this.time += 0.016;

    this.device.queue.writeBuffer(
      this.uniformsBuffer,
      0,
      new Float32Array([
        this.options.intensity,
        this.time,
        0,
        0,
      ]),
    );

    const bindGroup = this.device.createBindGroup({
      label: "Film Grain Bind Group",
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: input },
        { binding: 2, resource: { buffer: this.uniformsBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();

    const pass = commandEncoder.beginRenderPass({
      label: "Film Grain Pass",
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

  resize(width: number, height: number): void {
    // No render targets to resize for this pass
  }
}
