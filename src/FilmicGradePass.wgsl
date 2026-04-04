// Filmic Grade Pass
// Applies a Super 16 film-inspired colour grade to SDR input.
// Operations in order:
//   1. Lift / Gamma / Gain  (DaVinci Resolve-style colour correction)
//   2. Saturation
//   3. S-curve contrast
//   4. Vignette

struct FilmicGradeUniforms {
    // Lift: raises the black point per-channel. Small positive values add colour to shadows.
    lift:  vec4<f32>, // xyz = RGB lift, w = unused

    // Gamma: power curve applied to midtones. 1.0 = neutral. <1 brightens, >1 darkens.
    gamma: vec4<f32>, // xyz = RGB gamma, w = unused

    // Gain: scales highlights per-channel. 1.0 = neutral. Warm highlights = R>1, B<1.
    gain:  vec4<f32>, // xyz = RGB gain, w = unused

    // Saturation: 1.0 = neutral, 0.0 = greyscale, >1.0 = boosted.
    saturation: f32,

    // S-curve contrast strength. 0.0 = none, 1.0 = full film-style contrast curve.
    contrast: f32,

    // Vignette intensity. 0.0 = none, 1.0 = strong darkening at edges.
    vignette: f32,

    padding: f32,
};

@group(0) @binding(0) var inputSampler: sampler;
@group(0) @binding(1) var scene_texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> grade: FilmicGradeUniforms;

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

// Rec.709 luminance coefficients
fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Lift / Gamma / Gain
// Lift shifts the black point, Gamma adjusts midtones via power curve,
// Gain scales the white point. This matches DaVinci Resolve's primary colour wheels.
fn lift_gamma_gain(color: vec3<f32>, lift: vec3<f32>, gamma: vec3<f32>, gain: vec3<f32>) -> vec3<f32> {
    // Apply gain first (scales whites)
    var c = color * gain;
    // Apply lift (shifts blacks — additive offset before gamma)
    c = c + lift * (1.0 - c);
    // Apply gamma (power curve on midtones). Guard against zero/negative.
    c = pow(max(c, vec3<f32>(0.0)), 1.0 / max(gamma, vec3<f32>(0.001)));
    return clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
}

// Saturation adjustment
fn adjust_saturation(color: vec3<f32>, saturation: f32) -> vec3<f32> {
    let lum = luminance(color);
    return mix(vec3<f32>(lum), color, saturation);
}

// S-curve contrast
// Maps [0,1] through a smooth sigmoid centred at 0.5.
// Strength 0 = identity, strength 1 = strong film-style contrast.
fn scurve_contrast(color: vec3<f32>, strength: f32) -> vec3<f32> {
    // Compute the sigmoid: smoothstep has the right shape for this.
    // We lerp between the identity and the curve based on strength.
    let curved = smoothstep(vec3<f32>(0.0), vec3<f32>(1.0), color);
    return mix(color, curved, strength);
}

// Radial vignette
// UV is in [0,1]. Darkens corners using a smooth falloff.
fn vignette(color: vec3<f32>, uv: vec2<f32>, intensity: f32) -> vec3<f32> {
    // Distance from centre, normalised so corners = 1.0
    let centered = uv * 2.0 - 1.0;
    // Aspect-independent radial distance
    let dist = length(centered);
    // Smooth falloff: starts at ~0.5 radius, full at corners
    let falloff = 1.0 - smoothstep(0.3, 1.4, dist);
    let vig = mix(1.0, falloff, intensity);
    return color * vig;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = textureSample(scene_texture, inputSampler, in.uv_coords).rgb;

    // 1. Lift / Gamma / Gain
    color = lift_gamma_gain(color, grade.lift.rgb, grade.gamma.rgb, grade.gain.rgb);

    // 2. Saturation
    color = adjust_saturation(color, grade.saturation);

    // 3. S-curve contrast
    color = scurve_contrast(color, grade.contrast);

    // 4. Vignette
    color = vignette(color, in.uv_coords, grade.vignette);

    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
