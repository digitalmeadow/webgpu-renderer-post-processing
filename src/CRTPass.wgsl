struct CRTUniforms {
    maskSize: f32,
    colorNum: f32,
    maskIntensity: f32,
    maskBorder: f32,
    resolution: vec2<f32>,
    blending: u32,
    padding: u32,
}

@group(0) @binding(0) var inputSampler: sampler;
@group(0) @binding(1) var inputTexture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> crt_uniforms: CRTUniforms;

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

// Bayer matrix for ordered dithering
const bayerMatrix8x8 = array<f32, 64>(
    0.0/ 64.0, 48.0/ 64.0, 12.0/ 64.0, 60.0/ 64.0,  3.0/ 64.0, 51.0/ 64.0, 15.0/ 64.0, 63.0/ 64.0,
   32.0/ 64.0, 16.0/ 64.0, 44.0/ 64.0, 28.0/ 64.0, 35.0/ 64.0, 19.0/ 64.0, 47.0/ 64.0, 31.0/ 64.0,
    8.0/ 64.0, 56.0/ 64.0,  4.0/ 64.0, 52.0/ 64.0, 11.0/ 64.0, 59.0/ 64.0,  7.0/ 64.0, 55.0/ 64.0,
   40.0/ 64.0, 24.0/ 64.0, 36.0/ 64.0, 20.0/ 64.0, 43.0/ 64.0, 27.0/ 64.0, 39.0/ 64.0, 23.0/ 64.0,
    2.0/ 64.0, 50.0/ 64.0, 14.0/ 64.0, 62.0/ 64.0,  1.0/ 64.0, 49.0/ 64.0, 13.0/ 64.0, 61.0/ 64.0,
   34.0/ 64.0, 18.0/ 64.0, 46.0/ 64.0, 30.0/ 64.0, 33.0/ 64.0, 17.0/ 64.0, 45.0/ 64.0, 29.0/ 64.0,
   10.0/ 64.0, 58.0/ 64.0,  6.0/ 64.0, 54.0/ 64.0,  9.0/ 64.0, 57.0/ 64.0,  5.0/ 64.0, 53.0/ 64.0,
   42.0/ 64.0, 26.0/ 64.0, 38.0/ 64.0, 22.0/ 64.0, 41.0/ 64.0, 25.0/ 64.0, 37.0/ 64.0, 21.0 / 64.0
);

fn dither(uv: vec2<f32>, color: vec3<f32>) -> vec3<f32> {
    let x = i32(uv.x * crt_uniforms.resolution.x) % 8;
    let y = i32(uv.y * crt_uniforms.resolution.y) % 8;
    let threshold = bayerMatrix8x8[y * 8 + x];
    
    var result = color + threshold * 0.6;
    result.r = floor(result.r * (crt_uniforms.colorNum - 1.0) + 0.5) / (crt_uniforms.colorNum - 1.0);
    result.g = floor(result.g * (crt_uniforms.colorNum - 1.0) + 0.5) / (crt_uniforms.colorNum - 1.0);
    result.b = floor(result.b * (crt_uniforms.colorNum - 1.0) + 0.5) / (crt_uniforms.colorNum - 1.0);
    
    return result;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // CRT mask implementation - matches original article exactly
    // Step 1: Compute pixel coordinates from texture resolution
    let pixel = in.uv_coords * crt_uniforms.resolution;
    
    // Step 2: RGB cell and subcell coordinates
    let coord = pixel / crt_uniforms.maskSize;
    let subcoord = coord * vec2<f32>(3.0, 1.0);
    
    // Step 3: Offset for staggering every other cell
    let cell_offset = vec2<f32>(0.0, fract(floor(coord.x) * 0.5));
    
    // Step 4: Compute the RGB color index from 0 to 2
    let ind = floor(subcoord.x) % 3.0;
    
    // Step 5: Convert that value to an RGB color (multiplied to maintain brightness)
    var mask_color = vec3<f32>(
        select(0.0, 1.0, ind == 0.0),
        select(0.0, 1.0, ind == 1.0),
        select(0.0, 1.0, ind == 2.0)
    ) * 3.0;
    
    // Step 6: Signed subcell UVs (ranging from -1 to +1)
    let cell_uv = fract(subcoord + cell_offset) * 2.0 - 1.0;
    
    // Step 7: X and Y borders
    let border = 1.0 - cell_uv * cell_uv * crt_uniforms.maskBorder;
    
    // Step 8: Blend x and y mask borders
    mask_color = mask_color * border.x * border.y;
    
    // Step 9: Pixel coordinates rounded to the nearest cell
    let mask_coord = floor(coord + cell_offset) * crt_uniforms.maskSize;
    let mask_uv = mask_coord / crt_uniforms.resolution;
    
    // Step 10: Sample texture at cell position
    var color = textureSample(inputTexture, inputSampler, mask_uv);
    
    // Step 11: Apply dithering
    color = vec4<f32>(dither(mask_uv, color.rgb), color.a);
    
    // Step 12: Apply mask (blending or multiply)
    if (crt_uniforms.blending != 0u) {
        color = vec4<f32>(color.rgb * (1.0 + (mask_color - 1.0) * crt_uniforms.maskIntensity), color.a);
    } else {
        color = vec4<f32>(color.rgb * mask_color, color.a);
    }
    
    return color;
}
