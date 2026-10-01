#define_import_path benilla::moonlight_hook

// GFX (moonlight): the MOON as an additive night light. The stock 1.12 night (the `Light.dbc`
// night colours on the never-setting lighting sun) stays the base; this adds a cool, dim
// directional term from the visible white moon, per fragment, which the receivers (terrain,
// wow_model, static_gx) shadow with the moon's own map through `shadow_hook::realtime_shadow_moonlit`
// `.z`. The CPU half is `benilla_world::lighting::moonlight` (MonkeyFrame row 16 `moon`).
//
// `moon.w == 0` (moonLight 0, daylight, the moon below the horizon) returns exact zero, and every
// receiver guards its add on `moon.w > 0`, so the pre-feature render is untouched.
//
// Identifiers carry no trailing digit (naga_oil rejects them in composable modules).

// Cool moonlight in gamma space: blue-white, never saturated blue.
const MOON_TINT: vec3<f32> = vec3<f32>(0.62, 0.74, 1.0);

// A small wrap keeps the terminator soft (moonlight is a large, dim source seen through haze).
const MOON_WRAP: f32 = 0.2;

fn moon_light(normal: vec3<f32>, moon: vec4<f32>) -> vec3<f32> {
    if (moon.w <= 0.0) {
        return vec3<f32>(0.0);
    }
    let len_sq = dot(normal, normal);
    if (len_sq < 1e-12) {
        // The M2 corpus authors zero normals; give them the flat-lit share.
        return MOON_TINT * (moon.w * 0.5);
    }
    let n = normal * inverseSqrt(len_sq);
    let ndl = max((dot(n, moon.xyz) + MOON_WRAP) / (1.0 + MOON_WRAP), 0.0);
    return MOON_TINT * (moon.w * ndl);
}

// The moon's own colour for the volumetric pass (same tint, intensity baked in).
fn moon_colour(moon: vec4<f32>) -> vec3<f32> {
    return MOON_TINT * max(moon.w, 0.0);
}
