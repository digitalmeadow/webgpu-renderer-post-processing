import shader from "./DirectionalVolumetricPass.wgsl?raw";
import { PostPass, PostPassContext } from "@digitalmeadow/webgpu-renderer";

export interface DirectionalVolumetricPassOptions {
  numSamples?: number;
  maxMarchDistance?: number;
  density?: number;
  intensity?: number;
  viewAngleFalloff?: number;
  viewAnglePower?: number;
}

export class DirectionalVolumetricPass extends PostPass {
  private device: GPUDevice;
  private pipeline: GPURenderPipeline;
  private sampler: GPUSampler;
  private comparisonSampler: GPUSampler;
  private uniformsBuffer: GPUBuffer;
  private bindGroupLayout: GPUBindGroupLayout;
  private options: Required<DirectionalVolumetricPassOptions>;

  constructor(
    device: GPUDevice,
    cameraBindGroupLayout: GPUBindGroupLayout,
    lightingBindGroupLayout: GPUBindGroupLayout,
    options: DirectionalVolumetricPassOptions = {},
  ) {
    super();
    this.device = device;

    // Set default options
    this.options = {
      numSamples: options.numSamples ?? 32,
      maxMarchDistance: options.maxMarchDistance ?? 200.0,
      density: options.density ?? 0.1,
      intensity: options.intensity ?? 1.0,
      viewAngleFalloff: options.viewAngleFalloff ?? 0.7,
      viewAnglePower: options.viewAnglePower ?? 2.0,
    };

    const shaderModule = device.createShaderModule({ code: shader });

    // Create linear sampler for scene texture
    this.sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });

    // Create comparison sampler for shadow maps
    this.comparisonSampler = device.createSampler({
      compare: "less",
      magFilter: "linear",
      minFilter: "linear",
    });

    // Create bind group layout for scene input (group 0)
    this.bindGroupLayout = device.createBindGroupLayout({
      label: "Directional Volumetric Pass Bind Group Layout",
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
      ],
    });

    // Create uniforms bind group layout (group 3)
    const uniformsBindGroupLayout = device.createBindGroupLayout({
      label: "Directional Volumetric Pass Uniforms Layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
      ],
    });

    // Create uniforms buffer (24 bytes: u32, 5 × f32)
    // Layout: num_samples(u32), max_march_distance(f32), density(f32), intensity(f32), view_angle_falloff(f32), view_angle_power(f32)
    this.uniformsBuffer = device.createBuffer({
      label: "Directional Volumetric Pass Uniforms",
      size: 24,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Uniforms will be written at the start of each render() call

    // Create pipeline with additive blending for volumetric light
    this.pipeline = device.createRenderPipeline({
      label: "Directional Volumetric Pass Pipeline",
      layout: device.createPipelineLayout({
        bindGroupLayouts: [
          this.bindGroupLayout, // Group 0: Scene input
          cameraBindGroupLayout, // Group 1: Camera
          lightingBindGroupLayout, // Group 2: Lighting + shadows
          uniformsBindGroupLayout, // Group 3: Pass uniforms
        ],
      }),
      vertex: {
        module: shaderModule,
        entryPoint: "vs_main",
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fs_main",
        targets: [
          {
            format: "rgba16float",
            // Additive blending: result = src + dst
            blend: {
              color: {
                srcFactor: "one",
                dstFactor: "one",
                operation: "add",
              },
              alpha: {
                srcFactor: "one",
                dstFactor: "one",
                operation: "add",
              },
            },
          },
        ],
      },
      primitive: {
        topology: "triangle-list",
      },
    });
  }

  render(
    input: GPUTextureView,
    output: GPUTextureView,
    context: PostPassContext,
  ): void {
    // Write uniforms to buffer
    const uniformsArray = new ArrayBuffer(24);
    const uniformsView = new DataView(uniformsArray);
    uniformsView.setUint32(0, this.options.numSamples, true); // u32
    uniformsView.setFloat32(4, this.options.maxMarchDistance, true); // f32
    uniformsView.setFloat32(8, this.options.density, true); // f32
    uniformsView.setFloat32(12, this.options.intensity, true); // f32
    uniformsView.setFloat32(16, this.options.viewAngleFalloff, true); // f32
    uniformsView.setFloat32(20, this.options.viewAnglePower, true); // f32
    this.device.queue.writeBuffer(this.uniformsBuffer, 0, uniformsArray);

    // Create bind group for scene input (group 0)
    const sceneBindGroup = this.device.createBindGroup({
      label: "Directional Volumetric Pass Scene Bind Group",
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: input },
        { binding: 2, resource: context.geometryBuffer.depthView },
      ],
    });

    // Create bind group for uniforms (group 3)
    const uniformsBindGroup = this.device.createBindGroup({
      label: "Directional Volumetric Pass Uniforms Bind Group",
      layout: this.pipeline.getBindGroupLayout(3),
      entries: [{ binding: 0, resource: { buffer: this.uniformsBuffer } }],
    });

    // Create command encoder
    const commandEncoder = this.device.createCommandEncoder({
      label: "Directional Volumetric Pass Encoder",
    });

    // Render pass
    const renderPass = commandEncoder.beginRenderPass({
      label: "Directional Volumetric Pass",
      colorAttachments: [
        {
          view: output,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    renderPass.setPipeline(this.pipeline);
    renderPass.setBindGroup(0, sceneBindGroup);
    renderPass.setBindGroup(1, context.cameraBindGroup);
    renderPass.setBindGroup(2, context.lightingBindGroup);
    renderPass.setBindGroup(3, uniformsBindGroup);
    renderPass.draw(3); // Full-screen triangle
    renderPass.end();

    // Submit command buffer
    this.device.queue.submit([commandEncoder.finish()]);
  }

  resize(_width: number, _height: number): void {
    // No internal render targets to resize
  }
}
