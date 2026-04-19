import shader from "./CRTPass.wgsl?raw";
import { PostPass, PostPassContext } from "@digitalmeadow/webgpu-renderer";

export interface CRTPassOptions {
  /** Size of RGB cell in pixels (R+G+B together). Default: 3.0 (1px per phosphor) */
  maskSize?: number;
  /** Number of colors for dithering effect. Default: 16.0 */
  colorNum?: number;
  /** Intensity of the RGB mask effect (0-1). Default: 0.5 */
  maskIntensity?: number;
  /** Whether to blend the mask or multiply it. Default: true */
  blending?: boolean;
  /** Border intensity of the mask cells (0-1). Default: 0.9 */
  maskBorder?: number;
}

// https://mini.gmshaders.com/p/gm-shaders-mini-crt
export class CRTPass extends PostPass {
  private device: GPUDevice;

  private pipeline: GPURenderPipeline;
  private sampler: GPUSampler;
  private uniformsBuffer: GPUBuffer;
  private bindGroupLayout: GPUBindGroupLayout;

  private options: Required<CRTPassOptions>;
  private lastWidth: number = 0;
  private lastHeight: number = 0;

  constructor(device: GPUDevice, options: CRTPassOptions = {}) {
    super();
    this.device = device;

    this.options = {
      maskSize: options.maskSize ?? 3.0,
      colorNum: options.colorNum ?? 16.0,
      maskIntensity: options.maskIntensity ?? 0.5,
      blending: options.blending ?? true,
      maskBorder: options.maskBorder ?? 0.9,
    };

    const shaderModule = device.createShaderModule({
      label: "CRT Pass Shader",
      code: shader,
    });

    this.sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });

    this.bindGroupLayout = device.createBindGroupLayout({
      label: "CRT Pass Bind Group Layout",
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
      label: "CRT Pass Uniforms",
      size: 32, // 8 x f32 (maskSize, colorNum, maskIntensity, maskBorder, resolution.xy, blending, padding)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout],
    });

    this.pipeline = device.createRenderPipeline({
      label: "CRT Pass Pipeline",
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
    // Update resolution if changed
    if (
      context.width !== this.lastWidth ||
      context.height !== this.lastHeight
    ) {
      this.lastWidth = context.width;
      this.lastHeight = context.height;
    }

    // Update uniforms
    this.device.queue.writeBuffer(
      this.uniformsBuffer,
      0,
      new Float32Array([
        this.options.maskSize,
        this.options.colorNum,
        this.options.maskIntensity,
        this.options.maskBorder,
        this.lastWidth,
        this.lastHeight,
      ]),
    );

    this.device.queue.writeBuffer(
      this.uniformsBuffer,
      24, // offset for blending (u32 at byte 24)
      new Uint32Array([this.options.blending ? 1 : 0, 0]), // blending + padding
    );

    const bindGroup = this.device.createBindGroup({
      label: "CRT Pass Bind Group",
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: input },
        { binding: 2, resource: { buffer: this.uniformsBuffer } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder({
      label: "CRT Pass Encoder",
    });

    const pass = commandEncoder.beginRenderPass({
      label: "CRT Render Pass",
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
    this.lastWidth = width;
    this.lastHeight = height;
  }
}
