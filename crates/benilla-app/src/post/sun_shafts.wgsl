// MONKEY (post): screen-space radial blur through sky pixels in the scene depth.
// MONKEY (polish): two passes. `SHAFT_MASK` reads the depth once per half-resolution texel into
// a sky mask; the blur then samples that mask 28 times instead of the full-size depth.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

struct Shafts {
    sun: vec4<f32>,
    color: vec4<f32>,
    // MONKEY (leftovers): world-up of the view ray in NDC (xyz) and the sky threshold (w).
    horizon: vec4<f32>,
    ray: vec4<f32>,
};

#ifdef SHAFT_MASK

#ifdef MULTISAMPLED
@group(0) @binding(0) var depth: texture_depth_multisampled_2d;
#else
@group(0) @binding(0) var depth: texture_depth_2d;
#endif
@group(0) @binding(1) var<uniform> shafts: Shafts;

@fragment
fn fs_mask(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(depth));
    let p = min(vec2<i32>(in.position.xy) * 2, dims - vec2<i32>(1));
    // MONKEY (fix-post): one sample, not a max over all of them: the march is a blur.
    // Only the cleared / sky-pinned depth (exactly 0 on infinite reverse-Z) is sky; a ramp on
    // `near / z` would move with the live `nearclip` cvar and pour shafts through far terrain.
    // MONKEY (leftovers): water writes no depth, so depth 0 under the horizon is the sea (or the
    // far-clipped ground), never sky.
    let ndc = (vec2<f32>(p) + 0.5) / vec2<f32>(dims) * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    let up = dot(shafts.horizon.xyz, vec3(ndc, 1.0))
        / length(vec3(ndc * shafts.ray.xy, 1.0));
    let sky = select(0.0, 1.0, textureLoad(depth, p, 0) <= 1.0e-7 && up > shafts.horizon.w);
    return vec4(sky, 0.0, 0.0, 1.0);
}

#else

@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
@group(0) @binding(2) var mask: texture_2d<f32>;
@group(0) @binding(3) var<uniform> shafts: Shafts;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let base = textureSample(scene, scene_sampler, in.uv);
    let count = u32(shafts.sun.w);
    let step_uv = (shafts.sun.xy - in.uv) * (0.88 / shafts.sun.w);
    var uv = in.uv;
    var decay = 1.0;
    var light = 0.0;
    var norm = 0.0;
    for (var i = 0u; i < 32u; i++) {
        if (i >= count) { break; }
        uv += step_uv;
        decay *= 0.965;
        // The clamp-to-edge bilinear tap is the old clamped depth read, pre-thresholded.
        let sky = textureSampleLevel(mask, scene_sampler, uv, 0.0).r;
        light += sky * decay;
        norm += decay;
    }
    // MONKEY (fix-post): the path mean is only the occlusion ratio; the shaft itself falls off
    // radially from the sun (aspect-corrected), else every clear sky pixel got the same lift.
    let dims = vec2<f32>(textureDimensions(scene));
    let to_sun = (in.uv - shafts.sun.xy) * vec2<f32>(dims.x / dims.y, 1.0);
    let radial = 1.0 - clamp(length(to_sun) / 0.9, 0.0, 1.0);
    let rays = light / max(norm, 0.0001) * radial * radial;
    return vec4(base.rgb + shafts.color.rgb * rays * shafts.sun.z * 0.16, base.a);
}

#endif
