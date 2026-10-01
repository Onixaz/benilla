#define_import_path benilla::emissive_hook

// MONKEY (post): the opt-in HDR seam shared by windows, additive models/effects and magma.
// `light_diffuse.w` was the reference pack's unused clamp marker (always 1); the CPU now adds
// the `bloom` tier, so 1 remains a byte-identical Off sentinel without growing the light blob.
const EMISSIVE_WMO_WINDOW: u32 = 0u;
const EMISSIVE_M2_ADD: u32 = 1u;
const EMISSIVE_PARTICLE_ADD: u32 = 2u;
const EMISSIVE_MAGMA: u32 = 3u;

fn tier(packed_lane: f32) -> f32 {
    return clamp(round(packed_lane - 1.0), 0.0, 2.0);
}

fn emissive_boost(
    color: vec3<f32>,
    kind: u32,
    packed_lane: f32,
    authored_weight: f32,
) -> vec3<f32> {
    let t = tier(packed_lane);
    if (t < 0.5) {
        return color;
    }
    let high = step(1.5, t);
    var gain = mix(1.45, 1.85, high);
    if (kind == EMISSIVE_WMO_WINDOW) { gain = mix(1.70, 2.35, high); }
    if (kind == EMISSIVE_MAGMA) { gain = mix(1.80, 2.60, high); }
    // MONKEY (fix-post): boost only above a knee, so the base image keeps its look (below 0.6
    // untouched) and the excess goes to the bloom extract; at 1.0 the result equals the old gain.
    // A flat multiply saturated every additive spell and the lava to white once the combine clamps.
    // A window's gain is applied to its emissive light term alone (static_gx), so it stays flat.
    if (kind == EMISSIVE_WMO_WINDOW) {
        return color * mix(1.0, gain, clamp(authored_weight, 0.0, 1.0));
    }
    let knee = 0.6;
    let over = max(color - vec3<f32>(knee), vec3<f32>(0.0));
    return color + over * ((gain - 1.0) * clamp(authored_weight, 0.0, 1.0) / (1.0 - knee));
}

// Distant magma keeps part of its emissive body instead of converging completely to fog.
fn magma_fog_resist(packed_lane: f32) -> f32 {
    let t = tier(packed_lane);
    return select(0.0, mix(0.32, 0.48, step(1.5, t)), t > 0.5);
}
