// MONKEY (ao): depth-only screen-space ambient occlusion for the opaque scene.
// Three fullscreen stages, chosen by shader def:
//   STAGE_AO     half resolution: occlusion from a rotated spiral of depth taps (Alchemy-style
//                cosine term with a range falloff), plus view distance and a bright-pixel mask.
//   STAGE_BLUR   half resolution: 4x4 depth-aware box; the 4x4 Bayer rotation cancels inside it.
//   STAGE_APPLY  full resolution: joint-bilateral upsample, output multiplied into the scene.
// Stored texel: r = occlusion 0..1 (fade applied), g = view distance (SKY_DIST for sky),
// b = bright-pixel protection 0..1, a (MONKEY (followups)) = surface smoothness 0..1 after
// STAGE_AO (0 where both neighbours on an axis jump in depth), 1 - foliage mask after STAGE_BLUR.
#import bevy_render::view::View

struct Ao {
    // Radius in yards, darkening strength, sample count, cosine bias.
    params: vec4<f32>,
    // Distance fade start and end in yards, protection luma ramp start and end.
    fade: vec4<f32>,
    // x: debug view (0 off, 1 factor, 2 protection, 3 distance/50, 4 raw occlusion, 5 smoothness); y: gain.
    debug: vec4<f32>,
}

const SKY_DIST: f32 = 60000.0;

#ifdef STAGE_BLUR
@group(0) @binding(0) var raw: texture_2d<f32>;
#else
#ifdef MULTISAMPLED
@group(0) @binding(0) var depth: texture_depth_multisampled_2d;
#else
@group(0) @binding(0) var depth: texture_depth_2d;
#endif
// STAGE_AO: the resolved opaque scene colour. STAGE_APPLY: the blurred half-resolution AO.
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var<uniform> view: View;
#endif
@group(0) @binding(3) var<uniform> ao: Ao;

#ifndef STAGE_BLUR
// Sample zero, as the water and fog passes read MSAA depth.
fn depth_at(p: vec2<i32>) -> f32 {
    let size = vec2<i32>(textureDimensions(depth));
    return textureLoad(depth, clamp(p, vec2<i32>(0), size - vec2<i32>(1)), 0);
}

fn view_at(p: vec2<i32>, z: f32) -> vec3<f32> {
    let uv = (vec2<f32>(p) + 0.5 - view.viewport.xy) / view.viewport.zw;
    let c = view.view_from_clip * vec4(uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), z, 1.0);
    return c.xyz / c.w;
}

// Reverse-Z with an infinite far plane: the cleared value 0 is the sky.
fn is_sky(z: f32) -> bool {
    return z <= 1e-7;
}
#endif

#ifdef STAGE_AO
fn bayer4(p: vec2<i32>) -> f32 {
    var m = array<u32, 16>(0u, 8u, 2u, 10u, 12u, 4u, 14u, 6u, 3u, 11u, 1u, 9u, 15u, 7u, 13u, 5u);
    let x = u32(p.x) & 3u;
    let y = u32(p.y) & 3u;
    return f32(m[y * 4u + x]);
}

// A neighbour for the normal; the sky counts as infinitely far so it is never chosen.
fn neighbour(p: vec2<i32>, centre: vec3<f32>) -> vec3<f32> {
    let z = depth_at(p);
    if (is_sky(z)) { return centre + vec3(0.0, 0.0, -1e4); }
    return view_at(p, z);
}

@fragment
fn ao_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let h = vec2<i32>(frag.xy);
    let f = h * 2;
    let size = vec2<i32>(textureDimensions(source));
    // Protection from the brightest pixel of the 2x2 footprint: lit windows, flames, sunlit sand.
    var bright = 0.0;
    for (var j = 0; j < 4; j += 1) {
        let q = clamp(f + vec2(j & 1, j >> 1u), vec2<i32>(0), size - vec2<i32>(1));
        let c = textureLoad(source, q, 0).rgb;
        bright = max(bright, max(c.r, max(c.g, c.b)));
    }
    let protect = smoothstep(ao.fade.z, ao.fade.w, bright);
    let z = depth_at(f);
    if (is_sky(z)) { return vec4(0.0, SKY_DIST, protect, 1.0); }
    let p = view_at(f, z);
    let dist = -p.z;
    let fade = 1.0 - smoothstep(ao.fade.x, ao.fade.y, dist);
    let radius = ao.params.x;
    let proj = 0.5 * view.viewport.w * view.clip_from_view[1][1];
    let r_px = min(radius * proj / max(dist, 0.01), 0.12 * view.viewport.w);
    if (fade <= 0.0 || r_px < 2.0) { return vec4(0.0, dist, protect, 1.0); }

    // Normal from the flatter neighbour on each axis, so silhouettes do not bend it.
    let pr = neighbour(f + vec2(1, 0), p);
    let pl = neighbour(f - vec2(1, 0), p);
    let pd = neighbour(f + vec2(0, 1), p);
    let pu = neighbour(f - vec2(0, 1), p);
    let dx = select(p - pl, pr - p, abs(pr.z - p.z) < abs(p.z - pl.z));
    let dy = select(p - pu, pd - p, abs(pd.z - p.z) < abs(p.z - pu.z));
    // MONKEY (reviewfix-a): sky on both sides of an axis makes the two steps parallel (a 1-px pole
    // or rope against the sky); normalize(0) would be NaN and bloom would spread it. No AO there.
    let c = cross(dx, dy);
    let l2 = dot(c, c);
    if (l2 < 1e-12) { return vec4(0.0, dist, protect, 1.0); }
    var n = c * inverseSqrt(l2);
    if (dot(n, p) > 0.0) { n = -n; }
    // MONKEY (followups): cutout leaves scatter depth spikes through a canopy, and there the taps
    // are noise. The spike is the smaller of the two one-sided steps two pixels out, capped by the
    // second difference, which is ~0 on any plane (even at grazing angles) and on a silhouette
    // (one side continuous). Contact creases stay smooth, so contact shadows keep full strength.
    // The blur turns the spike DENSITY of its window into the foliage mask.
    let ax = neighbour(f + vec2(2, 0), p).z - p.z;
    let bx = p.z - neighbour(f - vec2(2, 0), p).z;
    let ay = neighbour(f + vec2(0, 2), p).z - p.z;
    let by = p.z - neighbour(f - vec2(0, 2), p).z;
    let spike = max(
        min(min(abs(ax), abs(bx)), abs(ax - bx)),
        min(min(abs(ay), abs(by)), abs(ay - by)),
    );
    let smooth_surface = 1.0 - smoothstep(0.01, 0.04, spike / dist);

    let count = u32(ao.params.z);
    let rot = (bayer4(h) + 0.5) * (6.2831853 / 16.0);
    let jitter = (bayer4(h + vec2(2, 1)) + 0.5) / 16.0;
    let r2 = radius * radius;
    var occ = 0.0;
    for (var i = 0u; i < count; i += 1u) {
        let t = (f32(i) + jitter) / f32(count);
        let a = f32(i) * 2.3999632 + rot;
        // Linear radius: taps crowd the centre, where contact shadows live.
        let off = vec2(cos(a), sin(a)) * mix(2.0, r_px, t);
        let q = f + vec2<i32>(round(off));
        let zq = depth_at(q);
        if (is_sky(zq)) { continue; }
        let v = view_at(q, zq) - p;
        let vv = dot(v, v);
        let w = saturate(1.0 - vv / r2);
        occ += w * max(0.0, dot(v, n) * inverseSqrt(vv + 1e-6) - ao.params.w);
    }
    occ = saturate(ao.debug.y * occ / f32(count)) * fade;
    return vec4(occ, dist, protect, smooth_surface);
}
#endif

#ifdef STAGE_BLUR
@fragment
fn blur_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let h = vec2<i32>(frag.xy);
    let size = vec2<i32>(textureDimensions(raw));
    let c = textureLoad(raw, h, 0);
    if (c.g >= SKY_DIST) { return c; }
    // MONKEY (followups): foliage = spike density over the window. There the depth test would
    // reject every leaf neighbour and keep the grain, so the tolerance widens and the leftover
    // occlusion is scaled down; a window of smooth surfaces keeps the tight edge and full AO.
    var spikes = 0.0;
    for (var y = -2; y <= 1; y += 1) {
        for (var x = -2; x <= 1; x += 1) {
            spikes += 1.0 - textureLoad(raw, clamp(h + vec2(x, y), vec2<i32>(0), size - vec2<i32>(1)), 0).a;
        }
    }
    let foliage = saturate(spikes / 16.0 * 4.0);
    let tolerance = (0.04 * c.g + 0.05) * (1.0 + 10.0 * foliage);
    var sum = 0.0;
    var weight = 0.0;
    for (var y = -2; y <= 1; y += 1) {
        for (var x = -2; x <= 1; x += 1) {
            let s = textureLoad(raw, clamp(h + vec2(x, y), vec2<i32>(0), size - vec2<i32>(1)), 0);
            let w = saturate(1.0 - abs(s.g - c.g) / tolerance);
            sum += s.r * w;
            weight += w;
        }
    }
    return vec4(sum / max(weight, 1e-4) * (1.0 - 0.4 * foliage), c.g, c.b, 1.0 - foliage);
}
#endif

#ifdef STAGE_APPLY
@fragment
fn apply_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(frag.xy);
    let z = depth_at(p);
    let mode = u32(ao.debug.x);
    if (is_sky(z)) {
        // Sky is magenta in the diagnostic views, so "depth reads as sky" is visible.
        return select(vec4(1.0), vec4(1.0, 0.0, 1.0, 1.0), mode >= 2u);
    }
    let dist = -view_at(p, z).z;
    let size = vec2<i32>(textureDimensions(source));
    let hc = frag.xy * 0.5 - 0.5;
    let base = vec2<i32>(floor(hc));
    let t = hc - floor(hc);
    var occ = 0.0;
    var protect = 0.0;
    var smooth_surface = 0.0;
    var weight = 0.0;
    for (var j = 0; j < 4; j += 1) {
        let o = vec2(j & 1, j >> 1u);
        let s = textureLoad(source, clamp(base + o, vec2<i32>(0), size - vec2<i32>(1)), 0);
        let b = mix(1.0 - t, t, vec2<f32>(o));
        let w = b.x * b.y / (0.02 + abs(s.g - dist) / dist) + 1e-5;
        occ += s.r * w;
        protect += s.b * w;
        smooth_surface += s.a * w;
        weight += w;
    }
    occ /= weight;
    protect /= weight;
    if (mode == 2u) { return vec4(protect, protect, protect, 1.0); }
    // Red = positive view distance, green = negative (a sign error), both / 50 yd.
    if (mode == 3u) { return vec4(saturate(dist / 50.0), saturate(-dist / 50.0), 0.0, 1.0); }
    if (mode == 4u) { return vec4(1.0 - occ, 1.0 - occ, 1.0 - occ, 1.0); }
    // MONKEY (followups): 5 = foliage mask (black = treated as foliage).
    if (mode == 5u) { let m = smooth_surface / weight; return vec4(m, m, m, 1.0); }
    let k = 1.0 - ao.params.y * occ * (1.0 - protect);
    return vec4(k, k, k, 1.0);
}
#endif
