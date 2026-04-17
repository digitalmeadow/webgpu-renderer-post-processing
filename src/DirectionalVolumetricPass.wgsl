// DirectionalVolumetricPass.wgsl
// True volumetric lighting with world-space ray marching toward directional light
// Samples shadow maps along ray to determine lit vs shadowed regions

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv_coords: vec2<f32>,
};

// ═══════════════════════════════════════════════════════════════════════════════
// BINDINGS
// ═══════════════════════════════════════════════════════════════════════════════

// Group 0: Scene input (texture + depth)
@group(0) @binding(0) var sceneSampler: sampler;
@group(0) @binding(1) var sceneTexture: texture_2d<f32>;
@group(0) @binding(2) var sceneDepth: texture_depth_2d;

// Group 1: Camera uniforms
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

@group(1) @binding(0) var<uniform> camera: CameraUniforms;

// Group 2: Lighting uniforms + shadow texture
const MAX_DIRECTIONAL_LIGHTS: u32 = 4;

struct LightDirectionalUniforms {
    view_projection_matrices: array<mat4x4<f32>, 3>,
    cascade_splits: vec4<f32>,
    direction: vec4<f32>,
    color: vec4<f32>,
    active_view_projection_index: u32,
};

struct LightDirectionalUniformsArray {
    lights: array<LightDirectionalUniforms, MAX_DIRECTIONAL_LIGHTS>,
    light_count: u32,
};

@group(2) @binding(0) var shadowSampler: sampler_comparison;
@group(2) @binding(1) var<uniform> lights: LightDirectionalUniformsArray;
@group(2) @binding(2) var shadowTexture: texture_depth_2d_array;

// Group 3: Pass-specific uniforms
struct VolumetricUniforms {
    num_samples: u32,
    max_march_distance: f32,
    density: f32,
    intensity: f32,
    view_angle_falloff: f32,
    view_angle_power: f32,
}

@group(3) @binding(0) var<uniform> uniforms: VolumetricUniforms;

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS (from LightingPass.wgsl)
// ═══════════════════════════════════════════════════════════════════════════════

// Reconstruct view-space position from depth buffer
fn position_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc_x = uv.x * 2.0 - 1.0;
    let ndc_y = (1.0 - uv.y) * 2.0 - 1.0;
    let ndc_z = depth;

    let clip_pos = vec4<f32>(ndc_x, ndc_y, ndc_z, 1.0);
    let view_pos = camera.projection_matrix_inverse * clip_pos;

    return view_pos.xyz / view_pos.w;
}

// Select cascade based on view-space depth
fn select_cascade(view_space_z: f32, splits: vec4<f32>) -> u32 {
    // Convert negative view-space Z to positive depth distance
    let depth = abs(view_space_z);
    
    if depth < splits.y {
        return 0u;
    } else if depth < splits.z {
        return 1u;
    } else {
        return 2u;
    }
}

// Simplified shadow fetch (single sample, no Vogel disk for performance)
fn fetch_shadow_simple(
    light_index: u32,
    cascade_id: u32,
    world_pos: vec3<f32>,
    light: LightDirectionalUniforms
) -> f32 {
    // Transform to shadow space
    let shadow_matrix = light.view_projection_matrices[cascade_id];
    let shadow_coords = shadow_matrix * vec4<f32>(world_pos, 1.0);
    
    if (shadow_coords.w <= 0.0) {
        return 1.0;
    }
    
    // Calculate UV and depth
    let flip_correction = vec2<f32>(0.5, -0.5);
    let proj_correction = 1.0 / shadow_coords.w;
    let light_local = shadow_coords.xy * flip_correction * proj_correction + vec2<f32>(0.5, 0.5);
    let depth = shadow_coords.z * proj_correction;
    
    // Return fully lit for fragments outside the light frustum
    if (light_local.x < 0.0 || light_local.x > 1.0 || 
        light_local.y < 0.0 || light_local.y > 1.0 || 
        depth < 0.0 || depth > 1.0) {
        return 1.0;
    }
    
    // Layer index: light0 uses cascades 0,1,2; light1 uses 3,4,5; etc.
    let layer_index = light_index * 3u + cascade_id;
    
    // Single shadow sample (no soft shadows for performance)
    return textureSampleCompareLevel(
        shadowTexture,
        shadowSampler,
        light_local,
        i32(layer_index),
        depth
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// VERTEX SHADER
// ═══════════════════════════════════════════════════════════════════════════════

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Full-screen triangle
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

// ═══════════════════════════════════════════════════════════════════════════════
// FRAGMENT SHADER - RAY MARCHING
// ═══════════════════════════════════════════════════════════════════════════════

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample scene color and depth
    let scene_color = textureSample(sceneTexture, sceneSampler, in.uv_coords);
    let depth = textureLoad(sceneDepth, vec2<i32>(in.uv_coords * vec2<f32>(textureDimensions(sceneDepth))), 0);
    
    // No lights? Return scene as-is
    if (lights.light_count == 0u) {
        return scene_color;
    }
    
    // Get first directional light
    let light = lights.lights[0];
    
    // Get camera position
    let camera_pos = camera.position.xyz;
    
    // Ray setup: march from camera toward pixel
    var world_pos: vec3<f32>;
    var ray_length: f32;
    
    if (depth >= 0.9999) {
        // Skybox/far plane: march to max distance in the direction of the pixel
        let view_pos = position_from_depth(in.uv_coords, 0.999); // Use near-far plane
        let far_world_pos = (camera.view_matrix_inverse * vec4<f32>(view_pos, 1.0)).xyz;
        let ray_dir = normalize(far_world_pos - camera_pos);
        world_pos = camera_pos + ray_dir * uniforms.max_march_distance;
        ray_length = uniforms.max_march_distance;
    } else {
        // Geometry: reconstruct world position from depth
        let view_pos = position_from_depth(in.uv_coords, depth);
        world_pos = (camera.view_matrix_inverse * vec4<f32>(view_pos, 1.0)).xyz;
        
        // March to minimum of (distance to pixel, max march distance)
        let distance_to_pixel = distance(camera_pos, world_pos);
        ray_length = min(distance_to_pixel, uniforms.max_march_distance);
    }
    
    // Ray direction from camera to pixel
    let ray_dir = normalize(world_pos - camera_pos);
    let step_size = ray_length / f32(uniforms.num_samples);
    
    // Calculate view-dependent density falloff
    // When viewing perpendicular to light (side-on), god rays are brightest
    // When viewing along light direction (toward/away from sun), reduce brightness
    let light_dir = normalize(-light.direction.xyz);
    let alignment = abs(dot(ray_dir, light_dir)); // 0 = perpendicular, 1 = parallel
    let perpendicularity = 1.0 - alignment;
    let falloff_curve = pow(perpendicularity, uniforms.view_angle_power);
    let view_factor = mix(1.0 - uniforms.view_angle_falloff, 1.0, falloff_curve);
    let adjusted_density = uniforms.density * view_factor;
    
    // Ray march accumulation
    var accumulated_fog = 0.0;
    
    for (var i = 0u; i < uniforms.num_samples; i++) {
        let t = f32(i) * step_size;
        let sample_pos = camera_pos + ray_dir * t;
        
        // Determine cascade for this sample position
        let sample_view_pos = (camera.view_matrix * vec4<f32>(sample_pos, 1.0)).xyz;
        let cascade = select_cascade(sample_view_pos.z, light.cascade_splits);
        
        // Sample shadow map at this position
        let shadow = fetch_shadow_simple(0u, cascade, sample_pos, light);
        
        // Accumulate if lit (shadow > 0.5 means not in shadow)
        // Use shadow value directly (already 0.0-1.0 from comparison sampler)
        let falloff = exp(-t * 0.01);
        accumulated_fog += shadow * adjusted_density * falloff;
    }
    
    // Calculate volumetric light contribution
    let volumetric_light = accumulated_fog * uniforms.intensity * light.color.rgb * light.color.a;
    
    // Apply volumetric lighting as additive (brightens lit areas)
    // The blend state will handle the actual blending
    let result = scene_color.rgb + volumetric_light;
    
    return vec4<f32>(result, scene_color.a);
}
