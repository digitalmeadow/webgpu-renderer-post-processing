struct TonemapUniforms {
    exposure: f32,
    padding: f32,
    padding2: f32,
    padding3: f32,
}

@group(0) @binding(0) var inputSampler: sampler;
@group(0) @binding(1) var scene_texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> tonemap_uniforms: TonemapUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv_coords: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );

    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    output.uv_coords = positions[vertex_index] * 0.5 + 0.5;
    output.uv_coords.y = 1.0 - output.uv_coords.y;

    return output;
}

// ACES RRT+ODT approximation (Stephen Hill / Narkowicz fit)
// Input: scene-linear HDR RGB. Output: SDR linear RGB in [0, 1].
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let hdr = textureSample(scene_texture, inputSampler, in.uv_coords).rgb;

    // Apply exposure before tonemapping
    let exposed = hdr * tonemap_uniforms.exposure;

    // ACES tonemapping: HDR linear → SDR linear (no gamma encoding here)
    let tonemapped = aces_tonemap(exposed);

    return vec4<f32>(tonemapped, 1.0);
}
