// God Rays (Volumetric Light Scattering) Post-Processing Pass
// Uses radial blur ray marching from light source with occlusion testing

struct GodRaysUniforms {
    num_samples: i32,
    density: f32,
    weight: f32,
    decay: f32,
    exposure: f32,
    max_ray_length: f32,
    occlusion_smoothness: f32,
    padding: f32,
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

// Group 0: Scene textures, sampler, occlusion texture, scene depth, and god rays uniforms
@group(0) @binding(0) var inputSampler: sampler;
@group(0) @binding(1) var scene_texture: texture_2d<f32>;
@group(0) @binding(2) var occlusion_texture: texture_depth_2d;
@group(0) @binding(3) var<uniform> uniforms: GodRaysUniforms;
@group(0) @binding(4) var scene_depth_texture: texture_depth_2d;
@group(0) @binding(5) var comparisonSampler: sampler_comparison;

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

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Reconstruct world position from screen UV and depth
fn screen_to_world(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let view_pos = camera.projection_matrix_inverse * vec4<f32>(ndc, depth * 2.0 - 1.0, 1.0);
    let world_pos = camera.view_matrix_inverse * vec4<f32>(view_pos.xyz / view_pos.w, 1.0);
    return world_pos.xyz;
}

// Project world position to light space for occlusion sampling
fn world_to_light_space(world_pos: vec3<f32>) -> vec2<f32> {
    let light = lights.lights[0];
    let light_space = light.view_projection_matrices[0] * vec4<f32>(world_pos, 1.0);
    let light_ndc = light_space.xyz / light_space.w;
    return vec2<f32>(light_ndc.x * 0.5 + 0.5, light_ndc.y * 0.5 + 0.5);
}

fn project_light_to_screen() -> vec2<f32> {
    if (lights.light_count == 0u) {
        return vec2<f32>(0.5, 0.5);
    }
    
    let light = lights.lights[0];
    let light_direction = normalize(light.direction.xyz);
    let far_distance = 10000.0;
    let light_world_pos = camera.position.xyz - light_direction * far_distance;
    
    let light_clip = camera.view_projection_matrix * vec4<f32>(light_world_pos, 1.0);
    var light_ndc = light_clip.xyz / light_clip.w;
    
    var light_screen = vec2<f32>(
        light_ndc.x * 0.5 + 0.5,
        -light_ndc.y * 0.5 + 0.5
    );
    
    return light_screen;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (lights.light_count == 0u) {
        return textureSample(scene_texture, inputSampler, in.uv_coords);
    }
    
    let screen_pos = in.uv_coords;
    let light_screen_pos = project_light_to_screen();
    
    let ray_vector = light_screen_pos - screen_pos;
    let ray_distance = length(ray_vector);
    let effective_ray_length = min(ray_distance, uniforms.max_ray_length);
    
    let num_samples_f = f32(uniforms.num_samples);
    let step_vector = normalize(ray_vector) * (effective_ray_length / num_samples_f);
    
    var illumination = 0.0;
    var sample_pos = screen_pos;
    
    let light = lights.lights[0];
    
    for (var i: i32 = 0; i < uniforms.num_samples; i++) {
        sample_pos += step_vector;
        
        // 1. Sample scene depth (depth textures require textureLoad, not textureSample)
        // MUST be in uniform control flow - sample first, check bounds later
        let depth_texture_size = textureDimensions(scene_depth_texture);
        let depth_texel_coords = vec2<i32>(sample_pos * vec2<f32>(depth_texture_size));
        let scene_depth = textureLoad(scene_depth_texture, depth_texel_coords, 0);
        
        // 2. Reconstruct 3D world position from screen UV + depth
        let world_pos = screen_to_world(sample_pos, scene_depth);
        
        // 3. Project to light space for occlusion testing
        let light_uv = world_to_light_space(world_pos);
        
        // 4. Calculate light space depth of current world position
        let light_space_pos = light.view_projection_matrices[0] * vec4<f32>(world_pos, 1.0);
        let light_ndc_depth = (light_space_pos.z / light_space_pos.w) * 0.5 + 0.5;
        
        // 5. Hardware-filtered occlusion test using comparison sampler
        // textureSampleCompare performs: sample 4 texels → compare each → bilinear filter results
        // MUST be in uniform control flow - sample first, check bounds later
        let bias = 0.001;
        let clamped_uv = clamp(light_uv, vec2(0.0), vec2(1.0));
        
        // Returns percentage of samples passing depth test (0.0-1.0)
        // Already bilinearly filtered by hardware - smooth transitions automatically!
        let occlusion_comparison_result = textureSampleCompare(
            occlusion_texture,
            comparisonSampler,
            clamped_uv,
            light_ndc_depth - bias
        );
        
        // 6. Check bounds AFTER sampling (enables uniform control flow)
        let in_screen_bounds = sample_pos.x >= 0.0 && sample_pos.x <= 1.0 && 
                                sample_pos.y >= 0.0 && sample_pos.y <= 1.0;
        let in_light_bounds = light_uv.x >= 0.0 && light_uv.x <= 1.0 && 
                               light_uv.y >= 0.0 && light_uv.y <= 1.0;
        
        // INVERTED LOGIC: God rays appear where light is BLOCKED, not where it's visible
        // occlusion_comparison_result: 1.0 = visible (no occlusion), 0.0 = occluded (in shadow)
        // We want rays where occluded, so invert: 1.0 - result
        let shadow_factor = 1.0 - occlusion_comparison_result;
        let occlusion_factor = smoothstep(0.0, uniforms.occlusion_smoothness, shadow_factor);
        
        // 7. Atmospheric scattering parameters
        // Decay: disabled for debugging (set to 1.0 for uniform intensity along rays)
        let decay_factor = pow(uniforms.decay, f32(i));
        
        // Atmospheric thickness: disabled for debugging (set to 1.0 for full intensity)
        // When enabled, fades rays near geometry surfaces
        let distance_from_geometry = scene_depth;
        // let atmospheric_thickness = smoothstep(0.995, 1.0, distance_from_geometry);
        let atmospheric_thickness = 1.0; // DEBUGGING: Full intensity, no depth-based falloff
        
        // Combine atmospheric thickness with smooth occlusion gradient
        let contribution = atmospheric_thickness * occlusion_factor;
        
        // Use select() to mask out invalid samples (preserves uniform control flow)
        let valid_contribution = select(0.0, contribution, in_screen_bounds && in_light_bounds);
        
        // Accumulate valid contributions
        illumination += valid_contribution * uniforms.weight * decay_factor;
    }
    
    let god_rays_intensity = illumination * uniforms.density * uniforms.exposure;
    let light_color = light.color.rgb * light.color.a;
    let god_rays_color = vec3<f32>(god_rays_intensity) * light_color;
    let scene_color = textureSample(scene_texture, inputSampler, screen_pos);
    let final_color = scene_color.rgb + god_rays_color;
    
    return vec4<f32>(final_color, 1.0);
}
