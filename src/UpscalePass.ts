import shader from "./UpscalePass.wgsl?raw";
import { PostPass, PostPassContext } from "@digitalmeadow/webgpu-renderer";

export interface UpscalePassOptions {
  /** Sampling filter. "linear" for smooth upscaling, "nearest" for pixelated. Default: "linear" */
  filter?: "linear" | "nearest";
}

/**
 * UpscalePass bridges low-resolution and high-resolution post-processing chains.
 * It samples from the low-res input and renders to the high-res output.
 * This pass should be the FIRST pass added via addHighResPostPass().
 */
export class UpscalePass extends PostPass {
  private device: GPUDevice;

  private pipeline: GPURenderPipeline;
  private sampler: GPUSampler;
  private bindGroupLayout: GPUBindGroupLayout;

  private options: Required<UpscalePassOptions>;

  constructor(device: GPUDevice, options: UpscalePassOptions = {}) {
    super();
    this.device = device;

    this.options = {
      filter: options.filter ?? "linear",
    };

    const shaderModule = device.createShaderModule({
      label: "Upscale Pass Shader",
      code: shader,
    });

    this.sampler = device.createSampler({
      magFilter: this.options.filter,
      minFilter: this.options.filter,
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });

    this.bindGroupLayout = device.createBindGroupLayout({
      label: "Upscale Pass Bind Group Layout",
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
      ],
    });

    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout],
    });

    this.pipeline = device.createRenderPipeline({
      label: "Upscale Pass Pipeline",
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

  render(
    input: GPUTextureView,
    output: GPUTextureView,
    context: PostPassContext,
  ): void {
    const bindGroup = this.device.createBindGroup({
      label: "Upscale Pass Bind Group",
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: input },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder({
      label: "Upscale Pass Encoder",
    });

    const pass = commandEncoder.beginRenderPass({
      label: "Upscale Render Pass",
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
}
