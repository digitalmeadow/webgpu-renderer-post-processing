// God Rays (Volumetric Light Scattering) Post-Processing Pass
// Uses radial blur ray marching from light source with occlusion testing

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

// Group 0: Scene textures, sampler, occlusion texture, scene depth, and god rays uniforms
@group(0) @binding(0) var inputSampler: sampler;
@group(0) @binding(1) var scene_texture: texture_2d<f32>;
@group(0) @binding(2) var occlusion_texture: texture_depth_2d;
@group(0) @binding(3) var<uniform> uniforms: GodRaysUniforms;
@group(0) @binding(4) var scene_depth_texture: texture_depth_2d;

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
        
        // Clamp sample position to screen bounds
        if (sample_pos.x < 0.0 || sample_pos.x > 1.0 || sample_pos.y < 0.0 || sample_pos.y > 1.0) {
            continue;
        }
        
        // 1. Load scene depth at this screen position
        let scene_tex_size = textureDimensions(scene_depth_texture);
        let depth_coord = vec2<i32>(sample_pos * vec2<f32>(scene_tex_size));
        let scene_depth = textureLoad(scene_depth_texture, depth_coord, 0);
        
        // Skip sky pixels (depth = 1.0 means no geometry)
        // Sky should contribute to god rays since light travels through atmosphere
        let is_sky = scene_depth >= 0.9999;
        
        // 2. Reconstruct 3D world position from screen UV + depth
        let world_pos = screen_to_world(sample_pos, scene_depth);
        
        // 3. Project to light space for occlusion testing
        let light_uv = world_to_light_space(world_pos);
        
        // 4. Sample occlusion depth
        let occlusion_size = textureDimensions(occlusion_texture);
        let clamped_uv = clamp(light_uv, vec2(0.0), vec2(1.0));
        let occlusion_coord = vec2<i32>(clamped_uv * vec2<f32>(occlusion_size));
        let occlusion_depth = textureLoad(occlusion_texture, occlusion_coord, 0);
        
        // 5. Check if within valid bounds
        let in_bounds = light_uv.x >= 0.0 && light_uv.x <= 1.0 && 
                         light_uv.y >= 0.0 && light_uv.y <= 1.0;
        
        // 6. Calculate light space depth of current world position
        let light = lights.lights[0];
        let light_space_pos = light.view_projection_matrices[0] * vec4<f32>(world_pos, 1.0);
        let light_ndc_depth = (light_space_pos.z / light_space_pos.w) * 0.5 + 0.5;
        
        // 7. Occlusion test: world position should be at or in front of occlusion surface
        // Add small bias to prevent self-shadowing artifacts
        let bias = 0.001;
        let is_occluded = in_bounds && (occlusion_depth - bias < light_ndc_depth) && !is_sky;
        
        // 8. Atmospheric scattering contribution (only in empty space)
        let decay_factor = pow(uniforms.decay, f32(i));
        
        // Only contribute atmospheric light in sky areas that aren't occluded
        // This eliminates geometry ghosting by not sampling scene colors
        if (is_sky && !is_occluded) {
            // Pure atmospheric scattering - no scene color sampling
            let atmospheric_intensity = 1.0;
            illumination += atmospheric_intensity * uniforms.weight * decay_factor;
        }
    }
    
    let god_rays_intensity = illumination * uniforms.density * uniforms.exposure;
    let light_color = light.color.rgb * light.color.a;
    let god_rays_color = vec3<f32>(god_rays_intensity) * light_color;
    let scene_color = textureSample(scene_texture, inputSampler, screen_pos);
    let final_color = scene_color.rgb + god_rays_color;
    
    return vec4<f32>(final_color, 1.0);
}
