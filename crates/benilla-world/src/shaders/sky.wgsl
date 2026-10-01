// The sky-dome gradient (`SkyExt`): the five Light.dbc SkyColor stops at the reference's ring
// elevations, then the fog colour at and below the horizon (`0x6d0d10`, `0x6d0f50`), interpolated
// per fragment by elevation. Depth is the far pin in `sky_vertex.wgsl`; this writes colour only.
#import bevy_pbr::{
    forward_io::VertexOutput,
    mesh_view_bindings::view,
}
// MONKEY (sky): the Enhanced/High library (smooth gradient, sun glow, night sky), compiled only
// into the `SKY_FX` pipeline so Classic keeps the reference's exact code.
#ifdef SKY_FX
#import benilla_world::sky_fx
#endif
// MONKEY (fog): the Modern fog colour at the horizon.
#import benilla::fog_hook

struct SkyColors {
    sky0: vec4<f32>, // zenith (90°)
    sky1: vec4<f32>, // 16.8°
    sky2: vec4<f32>, // 9.8°
    sky3: vec4<f32>, // 3.7°
    sky4: vec4<f32>, // 1.8°
    fog: vec4<f32>,  // horizon (0°) and below: LightIntBand row 7
    warp: vec4<f32>, // x = dawn/dusk warp strength S (0 = off), y = sun azimuth (rad), zw reserved
    // MONKEY (sky): x = skyQuality (0 Classic), y = reserved, z = night-sky alpha, w = glow
    // strength.
    fx: vec4<f32>,
    // MONKEY (sky): xyz = camera to the visible sun, w unused.
    sun: vec4<f32>,
    // MONKEY (sky): rgb = glow colour (gamma), w unused.
    glow: vec4<f32>,
    // MONKEY (fog): MonkeyFrame rows B-D (all zero = Classic).
    mf_fog_b: vec4<f32>,
    mf_fog_c: vec4<f32>,
    mf_fog_d: vec4<f32>,
};
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> sky: SkyColors;
// MONKEY (polish): x = the sky clock (s, wrapped at a day), written per frame outside the material.
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var<storage, read> sky_clock: vec4<f32>;

// Glow `g` for a sun-relative azimuth phase, sampled linearly like `0x6d0f50` from the reference's
// six-keyframe wrap-around table at `0xce9af8` (written by `0x6ce210`); 0.125 is the sun bearing.
fn azimuth_glow(phase: f32) -> f32 {
    let p = fract(phase);
    if (p < 0.125) { return mix(0.0, 1.0, (p + 0.125) / 0.25); } // wrap 0.875(g0)→1.125(g1)
    else if (p < 0.375) { return mix(1.0, 0.0, (p - 0.125) / 0.25); }
    else if (p < 0.5) { return mix(0.0, -0.5, (p - 0.375) / 0.125); }
    else if (p < 0.625) { return mix(-0.5, -0.7, (p - 0.5) / 0.125); }
    else if (p < 0.75) { return mix(-0.7, -0.5, (p - 0.625) / 0.125); }
    else if (p < 0.875) { return mix(-0.5, 0.0, (p - 0.75) / 0.125); }
    else { return mix(0.0, 1.0, (p - 0.875) / 0.25); } // wrap 0.875(g0)→1.125(g1)
}

// One mid-ring's warped colour for glow `g` (`0x6d0f50`): `S^2` is the prepass S nested in the
// per-segment S, and 0.7 is the constant at `0x7ffd7c`.
fn warp_one(base: vec3<f32>, g: f32, s: f32) -> vec3<f32> {
    let s2 = s * s;
    if (g >= 0.0) {
        return mix(base, sky.sky1.rgb, (1.0 - g) * s2);
    }
    return mix(mix(base, sky.sky1.rgb, s), sky.sky0.rgb, 0.7 * (-g) * s2);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let dir = normalize(in.world_position.xyz - view.world_position.xyz);
    let elev = degrees(asin(clamp(dir.y, -1.0, 1.0))); // −90..90, 0 = horizon

    // Dawn/dusk warp (`0x6d0f50`): only the four mid rings, never the apex or the fog rim, and
    // never brighter. The reference bakes it per vertex at 24 azimuth segments and interpolates,
    // hence the lerp of two segments; `+ 0.125` puts the sun bearing at glow phase 0.125.
    var s1 = sky.sky1.rgb;
    var s2c = sky.sky2.rgb;
    var s3 = sky.sky3.rgb;
    var s4 = sky.sky4.rgb;
    let warp_s = sky.warp.x;
    if (warp_s > 0.0) {
        let az = fract((atan2(dir.z, dir.x) - sky.warp.y) / 6.2831853 + 0.125);
        let seg = az * 24.0; // 24 azimuth segments, like the dome vertices
        let s0 = floor(seg);
        let f = seg - s0;
        let g0 = azimuth_glow(s0 / 24.0);
        let g1 = azimuth_glow((s0 + 1.0) / 24.0);
        s1 = mix(warp_one(sky.sky1.rgb, g0, warp_s), warp_one(sky.sky1.rgb, g1, warp_s), f);
        s2c = mix(warp_one(sky.sky2.rgb, g0, warp_s), warp_one(sky.sky2.rgb, g1, warp_s), f);
        s3 = mix(warp_one(sky.sky3.rgb, g0, warp_s), warp_one(sky.sky3.rgb, g1, warp_s), f);
        s4 = mix(warp_one(sky.sky4.rgb, g0, warp_s), warp_one(sky.sky4.rgb, g1, warp_s), f);
    }

    // Elevation gradient, linear between rings like the reference's Gouraud-shaded dome.
    var col: vec3<f32>;
    if (elev <= 0.0) {
        col = sky.fog.rgb; // horizon and below = fog colour (row 7), unwarped
    } else if (elev < 1.8) {
        col = mix(sky.fog.rgb, s4, elev / 1.8);
    } else if (elev < 3.7) {
        col = mix(s4, s3, (elev - 1.8) / (3.7 - 1.8));
    } else if (elev < 9.8) {
        col = mix(s3, s2c, (elev - 3.7) / (9.8 - 3.7));
    } else if (elev < 16.8) {
        col = mix(s2c, s1, (elev - 9.8) / (16.8 - 9.8));
    } else {
        col = mix(s1, sky.sky0.rgb, (elev - 16.8) / (90.0 - 16.8));
    }

    // MONKEY (sky): Enhanced/High. The same stops through a monotone cubic in linear light, then
    // the sun glow and the night sky added in linear light, back to gamma, dithered.
#ifdef SKY_FX
    if (sky.fx.x >= 0.5) {
        let stops = array<vec3<f32>, 6>(
            sky_fx::srgb_to_linear(sky.fog.rgb),
            sky_fx::srgb_to_linear(s4),
            sky_fx::srgb_to_linear(s3),
            sky_fx::srgb_to_linear(s2c),
            sky_fx::srgb_to_linear(s1),
            sky_fx::srgb_to_linear(sky.sky0.rgb),
        );
        var lin = sky_fx::smooth_gradient(elev, stops);
        // MONKEY (fix-sky): the Modern fog horizon goes into the gradient, in linear light, BEFORE
        // the glow and stars, so the fog wall meets the dome without a seam and the sunset halo is
        // not erased at 0° (the Classic tail below applies it to `col` instead).
        if (fog_hook::fog_is_modern(sky.mf_fog_d)) {
            let far = fog_hook::fog_modern_colour(
                sky.fog.rgb, dir, 1.0e9, sky.mf_fog_b, sky.mf_fog_c, sky.mf_fog_d,
            );
            let w = 1.0 - smoothstep(0.0, 0.26179939, asin(clamp(dir.y, -1.0, 1.0)));
            lin = mix(lin, sky_fx::srgb_to_linear(far), w);
        }
        // Sun glow: a halo tinted by the sun and fog colours, faded at night and under cloud on the
        // CPU; below the horizon it thins out over the fog band.
        if (sky.fx.w > 0.0) {
            let below = smoothstep(-0.12, 0.02, dir.y);
            lin += sky_fx::srgb_to_linear(sky.glow.rgb)
                * (sky.fx.w * below * sky_fx::sun_glow(dir, normalize(sky.sun.xyz)));
        }
        // Night sky: stars and the Milky Way, fixed to world directions, thinned toward the
        // horizon. The cloud dome draws over this, so clouds hide it where they stand.
        let night = sky.fx.z;
        // The pixel's size in star-grid units; derivatives stay in uniform control flow.
        let px = length(fwidth(dir)) * sky_fx::STAR_GRID * 0.5;
        if (night > 0.0 && dir.y > 0.0) {
            let band = sky_fx::galaxy_band(dir);
            let stars = sky_fx::star_field(dir, sky_clock.x, max(px, 1e-4), band)
                * smoothstep(0.0, 0.2, dir.y);
            let milky = sky_fx::milky_way(dir, band) * 0.05 * smoothstep(0.03, 0.35, dir.y);
            lin += (stars * 0.85 + milky) * night;
        }
        let out = sky_fx::linear_to_srgb(lin) + vec3<f32>(sky_fx::dither_tri(in.position.xy));
        // The horizon is already in: skip the Classic-path tail.
        return vec4<f32>(clamp(out, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    }
#endif

    // Raw gamma out: the reference draws the sky as raw DBC bytes, sRGB off (`0x6d4940`).
    // MONKEY (fog): Modern eases the low band into the world's far fog colour after the Classic
    // gradient (Enhanced/High mixes it in above); Classic fog returns `col` unchanged.
    let rgb = fog_hook::fog_sky_horizon(
        col, sky.fog.rgb, dir, sky.mf_fog_b, sky.mf_fog_c, sky.mf_fog_d,
    );
    return vec4<f32>(rgb, 1.0);
}
