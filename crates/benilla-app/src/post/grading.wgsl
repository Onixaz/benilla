// Ported from WarcraftXL (https://github.com/WarcraftXL) by iThorgrim — module wxl-retail-grading, Grading.cpp, shaders/Grading.ps.hlsl.
// Used with the author's permission; attribution required. See THIRD-PARTY.md.
// MONKEY (post): real 32³ textures make the original 1024x32 strip's two bilinear taps plus
// lerp(fract(b*31)) one hardware-trilinear lookup, for both the day and night cubes.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
@group(0) @binding(2) var day_lut: texture_3d<f32>;
@group(0) @binding(3) var night_lut: texture_3d<f32>;
@group(0) @binding(4) var lut_sampler: sampler;

struct Grade {
    control: vec4<f32>,
};
@group(0) @binding(5) var<uniform> grade: Grade;
// MONKEY (polish): the previous zone's pair, crossfaded out over `grade.control.z`.
@group(0) @binding(6) var prev_day_lut: texture_3d<f32>;
@group(0) @binding(7) var prev_night_lut: texture_3d<f32>;

fn grade_one(lut: texture_3d<f32>, color: vec3<f32>) -> vec3<f32> {
    // Half a texel in from each face, with 31 intervals across the 32-texel cube.
    let uvw = color * (31.0 / 32.0) + vec3(0.5 / 32.0);
    return textureSampleLevel(lut, lut_sampler, uvw, 0.0).rgb;
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let source = textureSampleLevel(scene, scene_sampler, in.uv, 0.0);
    let base = clamp(source.rgb, vec3(0.0), vec3(1.0));
    let day = grade_one(day_lut, base);
    let night = grade_one(night_lut, base);
    let authored = mix(day, night, grade.control.y);
    var graded = mix(base, authored, grade.control.x);
    // MONKEY (polish): mid-crossfade, the previous zone's pair through the same day/night blend
    // at its own strength; a missing zone or LUT is the identity cube, so it fades to ungraded.
    if (grade.control.z < 1.0) {
        let prev_day = grade_one(prev_day_lut, base);
        let prev_night = grade_one(prev_night_lut, base);
        let prev_authored = mix(prev_day, prev_night, grade.control.y);
        let prev_graded = mix(base, prev_authored, grade.control.w);
        graded = mix(prev_graded, graded, grade.control.z);
    }
    // Bloom already consumed the HDR signal; preserve its remaining excess for the FFX clamp.
    return vec4(graded + max(source.rgb - vec3(1.0), vec3(0.0)), source.a);
}
