// God Rays (Volumetric Light Scattering) Post-Processing Pass
// Simplified screen-space radial blur from sun position with depth-based occlusion

struct GodRaysUniforms {
    num_samples: i32,
    intensity: f32,
    decay: f32,
    max_ray_distance: f32,
    sun_radius: f32,
    angle_falloff: f32,
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

// Group 0: Scene textures, sampler, depth, and god rays uniforms
@group(0) @binding(0) var inputSampler: sampler;
@group(0) @binding(1) var scene_texture: texture_2d<f32>;
@group(0) @binding(2) var scene_depth_texture: texture_depth_2d;
@group(0) @binding(3) var<uniform> uniforms: GodRaysUniforms;

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
    // Early exit if no lights
    if (lights.light_count == 0u) {
        return textureSample(scene_texture, inputSampler, in.uv_coords);
    }
    
    let light = lights.lights[0];
    let screen_pos = in.uv_coords;
    let sun_screen_pos = project_light_to_screen();
    
    // Calculate ray from pixel toward sun
    let ray_vector = sun_screen_pos - screen_pos;
    let ray_distance = length(ray_vector);
    let ray_dir = normalize(ray_vector);
    
    // Limit ray marching distance
    let effective_ray_distance = min(ray_distance, uniforms.max_ray_distance);
    let step_size = effective_ray_distance / f32(uniforms.num_samples);
    
    // Calculate viewing angle falloff
    // When looking toward sun (in world space), rays are brighter
    let camera_forward = normalize((camera.view_matrix_inverse * vec4<f32>(0.0, 0.0, 1.0, 0.0)).xyz);
    let light_dir = normalize(-light.direction.xyz);
    let view_alignment = max(0.0, dot(camera_forward, light_dir)); // 0 = perpendicular, 1 = facing sun
    let angle_factor = mix(1.0 - uniforms.angle_falloff, 1.0, view_alignment);
    
    // Calculate distance falloff (brighter near sun position)
    let distance_from_sun = ray_distance;
    let distance_factor = smoothstep(uniforms.sun_radius * 2.0, uniforms.sun_radius, distance_from_sun);
    
    // Ray march accumulation
    var accumulated_light = 0.0;
    var current_pos = screen_pos;
    
    for (var i: i32 = 0; i < uniforms.num_samples; i++) {
        current_pos += ray_dir * step_size;
        
        // Skip if outside screen bounds
        if (current_pos.x < 0.0 || current_pos.x > 1.0 || 
            current_pos.y < 0.0 || current_pos.y > 1.0) {
            continue;
        }
        
        // Sample scene depth
        let depth_coords = vec2<i32>(current_pos * vec2<f32>(textureDimensions(scene_depth_texture)));
        let depth = textureLoad(scene_depth_texture, depth_coords, 0);
        
        // Consider sky/far geometry as unoccluded (contributes to god rays)
        let is_sky = depth >= 0.9999;
        let occlusion = select(0.0, 1.0, is_sky);
        
        // Apply exponential decay along ray
        let decay_factor = pow(uniforms.decay, f32(i));
        
        // Accumulate contribution
        accumulated_light += occlusion * decay_factor;
    }
    
    // Normalize by sample count and apply parameters
    let normalized_light = accumulated_light / f32(uniforms.num_samples);
    let god_rays_intensity = normalized_light * uniforms.intensity * angle_factor * distance_factor;
    
    // Apply light color
    let god_rays_color = god_rays_intensity * light.color.rgb * light.color.a;
    
    // Additive blend with scene
    let scene_color = textureSample(scene_texture, inputSampler, screen_pos);
    return vec4<f32>(scene_color.rgb + god_rays_color, 1.0);
}
