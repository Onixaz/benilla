// GFX (volumetric light): shadow-mapped light shafts, three stages (see volumetric_light.rs).
//   VL_MARCH     half resolution: march the directional shadow map along the view ray.
//   VL_BLUR      half resolution: depth-aware 4x4 gather over the jitter pattern.
//   VL_COMPOSITE full resolution: depth-aware upsample, colour, add over the scene.
// Identifiers carry no trailing digit (naga_oil).

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

struct VolLight {
    dir_steps: vec4<f32>, // toward the light body (unit), march steps
    colour: vec4<f32>,    // in-scatter colour, extinction per yard
    range: vec4<f32>,     // march length cap, march start, anisotropy g, height falloff (yd)
    debug: vec4<f32>,     // x = 1: show the in-scatter alone
}

// Distance (yd) along the view ray to the scene at `ndc_xy` with reverse-Z depth `z`; the sky
// (depth 0 on infinite reverse-Z) is `cap` away. Shared by the march and the composite so the two
// bilateral keys are the same quantity.
fn ray_distance(world_from_clip: mat4x4<f32>, eye: vec3<f32>, ndc_xy: vec2<f32>, z: f32, cap: f32) -> f32 {
    if (z <= 0.0) {
        return cap;
    }
    let q = world_from_clip * vec4<f32>(ndc_xy, z, 1.0);
    return min(length(q.xyz / q.w - eye), cap);
}

#ifdef VL_MARCH

#import bevy_render::view::View
#import bevy_pbr::mesh_view_types::Lights

#ifdef MULTISAMPLED
@group(0) @binding(0) var depth: texture_depth_multisampled_2d;
#else
@group(0) @binding(0) var depth: texture_depth_2d;
#endif
@group(0) @binding(1) var<uniform> view: View;
@group(0) @binding(2) var<uniform> lights: Lights;
@group(0) @binding(3) var shadows: texture_depth_2d_array;
@group(0) @binding(4) var shadow_sampler: sampler_comparison;
@group(0) @binding(5) var<uniform> vl: VolLight;

// 1 lit, 0 shadowed, -1 outside every cascade (unknown).
fn visibility(light_index: u32, p: vec3<f32>) -> f32 {
    let light = &lights.directional_lights[light_index];
    let view_distance = -(view.view_from_world * vec4(p, 1.0)).z;
    for (var cascade = 0u; cascade < (*light).num_cascades; cascade += 1u) {
        let shadow_cascade = &(*light).cascades[cascade];
        if (view_distance > (*shadow_cascade).far_bound) { continue; }
        let q = (*shadow_cascade).clip_from_world
            * vec4(p + (*light).direction_to_light * (*light).shadow_depth_bias, 1.0);
        let ndc = q.xyz / q.w;
        let uv = ndc.xy * vec2(0.5, -0.5) + 0.5;
        if (any(uv < vec2(0.0)) || any(uv > vec2(1.0)) || ndc.z < 0.0 || ndc.z > 1.0) { continue; }
        return textureSampleCompareLevel(shadows, shadow_sampler, uv,
            i32((*light).depth_texture_base_index + cascade), ndc.z);
    }
    return -1.0;
}

// Interleaved gradient noise (Jimenez 2014): a per-pixel offset whose 3x3/4x4 neighbourhoods
// cover the unit interval evenly, so a small blur removes it.
fn ign(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

fn phase(cosine: f32) -> f32 {
    let g = vl.range.z;
    let denom = 1.0 + g * g - 2.0 * g * cosine;
    let hg = 0.07957747 * (1.0 - g * g) / (denom * sqrt(denom));
    // A quarter isotropic keeps side-on shafts faintly visible, as real haze does.
    return mix(hg, 0.07957747, 0.25);
}

@fragment
fn fs_march(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(depth));
    let hp = vec2<i32>(in.position.xy);
    let full = min(hp * 2, dims - vec2<i32>(1));
    let z = textureLoad(depth, full, 0);
    let uv = (vec2<f32>(full) + 0.5) / vec2<f32>(dims);
    let ndc = uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    let cap = vl.range.x;
    let far = view.world_from_clip * vec4<f32>(ndc, max(z, 0.000001), 1.0);
    let ray = normalize(far.xyz / far.w - view.world_position);
    let dist = ray_distance(view.world_from_clip, view.world_position, ndc, z, cap);
    let start = vl.range.y;
    let span = dist - start;
    if (span <= 0.0) {
        return vec4<f32>(0.0, dist, 0.0, 1.0);
    }
    let count = u32(vl.dir_steps.w);
    let n = f32(count);
    let sigma = vl.colour.w;
    let jitter = ign(vec2<f32>(hp));
    var scatter = 0.0;
    for (var light_index = 0u; light_index < lights.n_directional_lights; light_index += 1u) {
        // The shadow rig's light is the one with a live map (SHADOWS_ENABLED).
        if ((lights.directional_lights[light_index].flags & 1u) == 0u) { continue; }
        var known = 0.0;
        var known_count = 0.0;
        for (var step = 0u; step < 64u; step += 1u) {
            if (step >= count) { break; }
            // Quadratic spacing: dense near the eye, where the shafts are.
            let u = (f32(step) + jitter) / n;
            let t = start + span * u * u;
            let dt = span * 2.0 * u / n;
            var seen = visibility(light_index, view.world_position + ray * t);
            if (seen >= 0.0) {
                known += seen;
                known_count += 1.0;
            } else {
                // Past the cascades: the mean of what this ray could see, never a hard 0 ring.
                seen = select(0.5, known / known_count, known_count > 0.0);
            }
            // Thinner air above the eye (down to 2 yd under it, roughly the ground).
            let above = max(ray.y * t + 2.0, 0.0);
            let local = sigma * exp(-above / vl.range.w);
            scatter += seen * exp(-sigma * t) * local * dt;
        }
        break;
    }
    scatter *= phase(dot(ray, vl.dir_steps.xyz));
    return vec4<f32>(scatter, dist, 0.0, 1.0);
}

#endif

#ifdef VL_BLUR

@group(0) @binding(0) var march: texture_2d<f32>;

@fragment
fn fs_blur(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(march));
    let p = vec2<i32>(in.position.xy);
    let centre = textureLoad(march, p, 0);
    let key = centre.y;
    let tolerance = 0.08 * key + 0.75;
    var sum = 0.0;
    var weight = 0.0;
    for (var dy = -2; dy <= 1; dy += 1) {
        for (var dx = -2; dx <= 1; dx += 1) {
            let q = clamp(p + vec2<i32>(dx, dy), vec2<i32>(0), dims - vec2<i32>(1));
            let s = textureLoad(march, q, 0);
            let w = exp(-abs(s.y - key) / tolerance);
            sum += s.x * w;
            weight += w;
        }
    }
    return vec4<f32>(sum / max(weight, 1e-5), key, 0.0, 1.0);
}

#endif

#ifdef VL_COMPOSITE

#import bevy_render::view::View

@group(0) @binding(0) var scene: texture_2d<f32>;
#ifdef MULTISAMPLED
@group(0) @binding(1) var depth: texture_depth_multisampled_2d;
#else
@group(0) @binding(1) var depth: texture_depth_2d;
#endif
@group(0) @binding(2) var shafts: texture_2d<f32>;
@group(0) @binding(3) var<uniform> view: View;
@group(0) @binding(4) var<uniform> vl: VolLight;

@fragment
fn fs_composite(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let p = vec2<i32>(in.position.xy);
    let source = textureLoad(scene, p, 0);
    let dims = vec2<i32>(textureDimensions(depth));
    let z = textureLoad(depth, min(p, dims - vec2<i32>(1)), 0);
    let uv = (vec2<f32>(p) + 0.5) / vec2<f32>(dims);
    let ndc = uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    let key = ray_distance(view.world_from_clip, view.world_position, ndc, z, vl.range.x);
    // Bilinear footprint in the half-resolution grid, each tap weighted by depth agreement.
    let half_dims = vec2<i32>(textureDimensions(shafts));
    let h = (vec2<f32>(p) + 0.5) * 0.5 - 0.5;
    let base = vec2<i32>(floor(h));
    let f = h - floor(h);
    let tolerance = 0.08 * key + 0.75;
    var sum = 0.0;
    var weight = 0.0;
    for (var j = 0; j <= 1; j += 1) {
        for (var i = 0; i <= 1; i += 1) {
            let q = clamp(base + vec2<i32>(i, j), vec2<i32>(0), half_dims - vec2<i32>(1));
            let s = textureLoad(shafts, q, 0);
            let bilinear = select(1.0 - f.x, f.x, i == 1) * select(1.0 - f.y, f.y, j == 1);
            let w = bilinear * exp(-abs(s.y - key) / tolerance) + 1e-4;
            sum += s.x * w;
            weight += w;
        }
    }
    let scatter = max(sum / weight, 0.0);
    let light = vl.colour.rgb * scatter;
    if (vl.debug.x > 0.5) {
        return vec4<f32>(vec3<f32>(1.0) - exp(-light), source.a);
    }
    // The fog pass's soft screen curve: bright pixels saturate gently instead of clipping.
    let lifted = source.rgb + max(vec3<f32>(0.0), vec3<f32>(1.0) - source.rgb)
        * (vec3<f32>(1.0) - exp(-light));
    return vec4<f32>(lifted, source.a);
}

#endif
