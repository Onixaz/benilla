#define_import_path benilla::fog_hook

// MONKEY (p0 fog hook): the ONE place the distance fog is computed.
//
// Every receiver — terrain, wow_model, static_gx, liquid, enhanced_water, wow_effect, wdl — calls
// in here instead of hand-rolling the fog law, so a new fog model touches this file only (the
// `shadow_hook.wgsl` pattern). Pure functions: the receivers bind the shared light buffer at
// different slots (material binding 90, group 2 binding 0, group 1 binding 2, binding 105), so
// every value arrives as an argument and this file owns no bindings.
//
// The classic law is the reference's GL_LINEAR fog on PLANAR eye-Z in gamma space (terrain.wgsl
// has the full note): `factor = clamp((end - eye_z) / max(end - start, 0.001), 0, 1)`, then
// `mix(fog_rgb, rgb, factor)`. The expressions below are the ones the receivers carried, in the
// same order, so the classic result is bit-identical.
//
// Extension point for the FOG lane: `fog_sample` branches on `mf.fog_d.x` (`MonkeyFrame.fog_model`,
// 0 classic / 1 modern). The modern arm currently returns the classic result.
//
// MONKEY (fog): the Modern model (cvar `fogModel` 1; CPU half `lighting/fog_model.rs`). `fog_d.x`
// holds the scene fog end when Modern is on (0 = Classic, every path below is then untouched):
// - survival: radial distance, exponential from the zone's start, fitted so 60 % survives at the
//   middle of the 1.12 span, closed linearly over the last stretch (from max(start, 0.3·end), the
//   modern client's 1/0.7 end fade) so the fog still ends where the zone authored it; optional
//   height fog below `fog_a.y` (Bevy Y); `fog_a.w` blends back toward the 1.12 linear ramp.
// - colour (scene colour only, never the black/white/grey blend policies): shifts to the end-fog
//   colour `fog_c` by (d / end)³, then leans to the sun-fog colour `fog_b` inside a cubic lobe around
//   the sun (cosine threshold `fog_d.y`, direction octahedral in `fog_d.zw`; strength already faded
//   for night on the CPU).
// - only spans ending at the scene fog end take it: an interior WMO fog pair stays classic.
// - the WDL hull's own (0, 1) pair asks for the horizon: fully fogged, in the far colour.
// The sky dome (`fog_sky_horizon`) and the volumetric haze (`fog_modern_colour`) call the same
// colour function, so the fogged world, the far hull and the sky's horizon meet in one colour.

#import benilla::monkey_frame::MonkeyFrame

// The classic survival factor: 1 = no fog, 0 = fully fogged.
fn fog_linear(eye_z: f32, span: vec2<f32>) -> f32 {
    let denom = max(span.y - span.x, 0.001);
    return clamp((span.y - eye_z) / denom, 0.0, 1.0);
}

// MONKEY (fog): exp survival at the span midpoint = 0.6, i.e. k·(end − start) = −2·ln 0.6.
const FOG_EXP_MID: f32 = 1.0216512;

// MONKEY (fog): the unit vector packed by `fog_model::oct_encode`.
fn fog_oct_decode(e: vec2<f32>) -> vec3<f32> {
    var v = vec3<f32>(e.x, 1.0 - abs(e.x) - abs(e.y), e.y);
    let t = max(-v.y, 0.0);
    v.x += select(t, -t, v.x >= 0.0);
    v.z += select(t, -t, v.z >= 0.0);
    return normalize(v);
}

// MONKEY (fog): whether the Modern model is on for this frame.
fn fog_is_modern(sel_row: vec4<f32>) -> bool {
    return sel_row.x > 0.5;
}

// MONKEY (fog): the Modern fog colour seen along unit `dir` at radial distance `d` (yd), from the
// scene fog colour `base`: end-fog shift, then the sun lobe. `d` may be huge (the sky).
fn fog_modern_colour(
    base: vec3<f32>,
    dir: vec3<f32>,
    d: f32,
    sun_row: vec4<f32>,
    end_row: vec4<f32>,
    sel_row: vec4<f32>,
) -> vec3<f32> {
    var c = base;
    if (end_row.w > 0.0) {
        let t = clamp(d / end_row.w, 0.0, 1.0);
        c = mix(c, end_row.rgb, t * t * t);
    }
    if (sun_row.w > 0.0) {
        let sun = fog_oct_decode(sel_row.zw);
        let a = sel_row.y;
        let s = clamp((dot(dir, sun) - a) / max(1.0 - a, 0.001), 0.0, 1.0);
        c = mix(c, sun_row.rgb, sun_row.w * s * s * s);
    }
    return c;
}

// MONKEY (fog): the Modern survival (1 = no fog) at radial distance `d` for the pair `span`.
fn fog_modern_survival(span: vec2<f32>, d: f32, world_y: f32, height_row: vec4<f32>) -> f32 {
    let len = max(span.y - span.x, 0.001);
    let k = FOG_EXP_MID / len;
    let x = max(d - span.x, 0.0);
    var expo = exp(-k * x);
    if (height_row.x > 0.0) {
        // Extra density at and below the plane, fading out `1 / falloff` yd above it.
        let hf = clamp(1.0 - (world_y - height_row.y) * height_row.z, 0.0, 1.0);
        expo = mix(expo, exp(-k * (1.0 + height_row.x) * x), hf);
    }
    let fade_from = max(span.x, 0.3 * span.y);
    let end_fade = clamp((span.y - d) / max(span.y - fade_from, 0.001), 0.0, 1.0);
    let linear = clamp((span.y - d) / len, 0.0, 1.0);
    return mix(min(expo, end_fade), linear, clamp(height_row.w, 0.0, 1.0));
}

// MONKEY (fog): the sky dome's low band under Modern. `col` is the dome's own colour at `dir`,
// `horizon_rgb` the colour it uses at 0° (the scene fog colour); the far fog colour along `dir`
// replaces the horizon and eases into the dome's own gradient by 15° of elevation (a mix, never an
// offset, so an already warm dusk ring is not pushed past itself).
// Classic returns `col` untouched.
fn fog_sky_horizon(
    col: vec3<f32>,
    horizon_rgb: vec3<f32>,
    dir: vec3<f32>,
    sun_row: vec4<f32>,
    end_row: vec4<f32>,
    sel_row: vec4<f32>,
) -> vec3<f32> {
    if (!fog_is_modern(sel_row)) {
        return col;
    }
    let far = fog_modern_colour(horizon_rgb, dir, 1.0e9, sun_row, end_row, sel_row);
    let elev = asin(clamp(dir.y, -1.0, 1.0));
    let w = 1.0 - smoothstep(0.0, 0.26179939, elev);
    return mix(col, far, w);
}

// The fog a surface stands in: rgb = the fog colour, w = how much of the surface survives (1 = no
// fog). `fog_rgb` is the colour the caller's policy picked; `scene_colour` is true when that is the
// scene's own fog colour (false for the forced black/white/grey policies of additive, Mod and Mod2x
// batches, which a new model may attenuate but must not recolour). `span` = (start, end) yd,
// `eye_z` = planar eye depth (yd), `world_pos` / `camera_pos` in Bevy world space.
fn fog_sample(
    fog_rgb: vec3<f32>,
    span: vec2<f32>,
    eye_z: f32,
    world_pos: vec3<f32>,
    camera_pos: vec3<f32>,
    scene_colour: bool,
    mf: MonkeyFrame,
) -> vec4<f32> {
    let classic = vec4<f32>(fog_rgb, fog_linear(eye_z, span));
    if (mf.fog_d.x > 0.5) {
        // MONKEY (fog): the Modern model (header).
        let delta = world_pos - camera_pos;
        let d = length(delta);
        let dir = delta / max(d, 0.0001);
        if (span.x == 0.0 && span.y == 1.0) {
            // The WDL hull's own pair: the horizon, fully fogged in the far colour.
            return vec4<f32>(fog_modern_colour(fog_rgb, dir, d, mf.fog_b, mf.fog_c, mf.fog_d), 0.0);
        }
        if (abs(span.y - mf.fog_d.x) > 0.01) {
            // Not the scene fog (an interior WMO pair): classic.
            return classic;
        }
        var rgb = fog_rgb;
        if (scene_colour) {
            rgb = fog_modern_colour(fog_rgb, dir, d, mf.fog_b, mf.fog_c, mf.fog_d);
        }
        return vec4<f32>(rgb, fog_modern_survival(span, d, world_pos.y, mf.fog_a));
    }
    return classic;
}

// Fog `rgb` in place: `mix(fog colour, rgb, survival)`. Same arguments as `fog_sample`.
fn apply_fog(
    rgb: vec3<f32>,
    fog_rgb: vec3<f32>,
    span: vec2<f32>,
    eye_z: f32,
    world_pos: vec3<f32>,
    camera_pos: vec3<f32>,
    scene_colour: bool,
    mf: MonkeyFrame,
) -> vec3<f32> {
    let f = fog_sample(fog_rgb, span, eye_z, world_pos, camera_pos, scene_colour, mf);
    return mix(f.rgb, rgb, f.w);
}
