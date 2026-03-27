struct BloomUniforms {
    threshold: f32,
    radius: f32,
    intensity: f32,
    padding: f32,
}

@group(0) @binding(0) var inputSampler: sampler;
@group(0) @binding(1) var gbuffer_metallic_roughness: texture_2d<f32>;
@group(0) @binding(2) var blurred_texture: texture_2d<f32>;
@group(0) @binding(3) var<uniform> bloom_uniforms: BloomUniforms;
@group(0) @binding(4) var scene_texture: texture_2d<f32>;

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

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn get_emissive_intensity() -> f32 {
    let metal_rough = textureSample(gbuffer_metallic_roughness, inputSampler, vec2<f32>(0.0, 0.0));
    return metal_rough.a;
}

@fragment
fn fs_threshold(in: VertexOutput) -> @location(0) vec4<f32> {
    let scene_color = textureSample(scene_texture, inputSampler, in.uv_coords);
    let metal_rough = textureSample(gbuffer_metallic_roughness, inputSampler, in.uv_coords);
    let emissive_intensity = metal_rough.a;

    let lum = luminance(scene_color.rgb);
    let threshold = bloom_uniforms.threshold;

    var intensity = 0.0;
    if (emissive_intensity > 0.0) {
        intensity = emissive_intensity;
    } else if (lum > threshold) {
        intensity = lum - threshold;
    }

    if (intensity > 0.0) {
        return vec4<f32>(scene_color.rgb * intensity, 1.0);
    }

    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs_blur(in: VertexOutput) -> @location(0) vec4<f32> {
    let texture_size = vec2<f32>(textureDimensions(blurred_texture));
    let radius = bloom_uniforms.radius;

    var result = vec3<f32>(0.0);
    var total_weight = 0.0;

    let start = vec2<i32>(-i32(radius));
    let end = vec2<i32>(i32(radius) + 1);

    for (var y: i32 = start.y; y < end.y; y++) {
        for (var x: i32 = start.x; x < end.x; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) / texture_size;
            let sample_color = textureSample(blurred_texture, inputSampler, in.uv_coords + offset);

            let dist = length(vec2<f32>(f32(x), f32(y)));
            let weight = 1.0 - smoothstep(0.0, radius, dist);

            result += sample_color.rgb * weight;
            total_weight += weight;
        }
    }

    return vec4<f32>(result / max(total_weight, 0.0001), 1.0);
}

@fragment
fn fs_composite(in: VertexOutput) -> @location(0) vec4<f32> {
    let scene_color = textureSample(scene_texture, inputSampler, in.uv_coords);
    let bloom_color = textureSample(blurred_texture, inputSampler, in.uv_coords);

    let final_color = scene_color.rgb + bloom_color.rgb * bloom_uniforms.intensity;

    return vec4<f32>(final_color, 1.0);
}
