// God Rays (Volumetric Light Scattering) Post-Processing Pass
// Uses radial blur ray marching from light source to simulate atmospheric scattering

struct GodRaysUniforms {
    num_samples: i32,
    density: f32,
    weight: f32,
    decay: f32,
    exposure: f32,
    max_ray_length: f32,
    padding: f32,
    padding2: f32,
}

struct CameraUniforms {
    view_matrix: mat4x4<f32>,
    projection_matrix: mat4x4<f32>,
    view_projection_matrix: mat4x4<f32>,
    view_matrix_inverse: mat4x4<f32>,
    projection_matrix_inverse: mat4x4<f32>,
    position: vec4<f32>,
    near: f32,
    far: f32,
}

const MAX_DIRECTIONAL_LIGHTS: u32 = 4;

struct LightDirectionalUniforms {
    view_projection_matrices: array<mat4x4<f32>, 3>,
    cascade_splits: vec4<f32>,
    direction: vec4<f32>,
    color: vec4<f32>,
    active_view_projection_index: u32,
}

struct LightDirectionalUniformsArray {
    lights: array<LightDirectionalUniforms, MAX_DIRECTIONAL_LIGHTS>,
    light_count: u32,
}

// Group 0: Scene textures, sampler, and god rays uniforms
@group(0) @binding(0) var inputSampler: sampler;
@group(0) @binding(1) var scene_texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms: GodRaysUniforms;

// Group 1: Camera uniforms
@group(1) @binding(0) var<uniform> camera: CameraUniforms;

// Group 2: Lighting uniforms (directional lights buffer only)
@group(2) @binding(1) var<uniform> lights: LightDirectionalUniformsArray;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv_coords: vec2<f32>,
}

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

// Calculate luminance from RGB
fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Project directional light direction to screen space
// Since directional lights are "at infinity", we create a point far away in the light direction
fn project_light_to_screen() -> vec2<f32> {
    // Get the first directional light if available
    if (lights.light_count == 0u) {
        // No lights - return center of screen as fallback
        return vec2<f32>(0.5, 0.5);
    }
    
    let light = lights.lights[0];
    let light_direction = normalize(light.direction.xyz);
    
    // Create a point "at infinity" in the direction TOWARD the light
    // (opposite of light direction since light.direction points FROM light TO surface)
    let far_distance = 10000.0;
    let light_world_pos = camera.position.xyz - light_direction * far_distance;
    
    // Project to clip space
    let light_clip = camera.view_projection_matrix * vec4<f32>(light_world_pos, 1.0);
    
    // Convert to NDC
    var light_ndc = light_clip.xyz / light_clip.w;
    
    // Convert NDC to UV coordinates (0 to 1)
    var light_screen = vec2<f32>(
        light_ndc.x * 0.5 + 0.5,
        -light_ndc.y * 0.5 + 0.5  // Flip Y
    );
    
    return light_screen;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Early exit if no lights
    if (lights.light_count == 0u) {
        return textureSample(scene_texture, inputSampler, in.uv_coords);
    }
    
    let screen_pos = in.uv_coords;
    let light_screen_pos = project_light_to_screen();
    
    // Calculate ray from current pixel to light source
    let ray_vector = light_screen_pos - screen_pos;
    let ray_distance = length(ray_vector);
    
    // Limit ray length to avoid excessive marching
    let effective_ray_length = min(ray_distance, uniforms.max_ray_length);
    
    // Calculate step size
    let num_samples_f = f32(uniforms.num_samples);
    let step_vector = normalize(ray_vector) * (effective_ray_length / num_samples_f);
    
    // Ray march toward the light
    var illumination = 0.0;
    var sample_pos = screen_pos;
    
    for (var i: i32 = 0; i < uniforms.num_samples; i++) {
        sample_pos += step_vector;
        
        // Sample the scene color at this position
        let scene_sample = textureSample(scene_texture, inputSampler, sample_pos);
        
        // Calculate brightness contribution
        let brightness = luminance(scene_sample.rgb);
        
        // Accumulate with exponential decay
        let decay_factor = pow(uniforms.decay, f32(i));
        illumination += brightness * uniforms.weight * decay_factor;
    }
    
    // Apply density and exposure
    let god_rays_intensity = illumination * uniforms.density * uniforms.exposure;
    
    // Get light color (use first directional light)
    let light_color = lights.lights[0].color.rgb * lights.lights[0].color.a;
    
    // Apply light color to god rays
    let god_rays_color = vec3<f32>(god_rays_intensity) * light_color;
    
    // Sample original scene color
    let scene_color = textureSample(scene_texture, inputSampler, screen_pos);
    
    // Additive blend
    let final_color = scene_color.rgb + god_rays_color;
    
    return vec4<f32>(final_color, 1.0);
}
