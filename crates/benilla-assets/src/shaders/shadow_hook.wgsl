#define_import_path benilla::shadow_hook

// MONKEY (shadow extension hook): the ONE place the realtime directional-shadow term is computed.
//
// benilla's receivers — `terrain.wgsl`, `wow_model.wgsl`, `static_gx.wgsl` — call `realtime_shadow`
// instead of each hand-rolling the fetch + fades. Adding or changing the realtime shadow now touches
// THIS file only; the three receiver shaders (the ones that blanked the terrain when edited by hand)
// stay untouched. The receivers keep their own downstream gating (terrain's MCSH suppression + spec
// gate, the model's interior/rig-skin exclusion + ambient-preserving apply, static_gx's ambient arm)
// — those are lighting composition, not shadow sampling, so they belong to each receiver.
//
// It is a no-op whenever no shadow sun exists: the only Bevy `DirectionalLight` benilla ever spawns
// is the shadow rig's (`benilla_app::shadow_core`), so `lights.n_directional_lights == 0` exactly
// when shadows are off, and the loop below never runs. No shader-def toggle is needed.

#import bevy_pbr::{
    shadows,
    mesh_view_bindings::lights,
}

// The realtime shadow lightens over the last SHADOW_EDGE_BAND yards of the cascade's max distance so
// it fades in rather than popping at the resolve boundary.
const SHADOW_EDGE_BAND: f32 = 14.0;

// MONKEY (moon shadows): the `fog_params.z` lane decodes. ONE SIGNED float carries BOTH directional
// weights — `+w` the sun's, `-w` the moon's — because `global_light::moon_shadow_weight` guarantees
// they are never both non-zero (the rig holds ONE body's depth map at a time; the receiver loop
// below ASSIGNS, last light wins). The CPU half is `global_light::pack_shadow_lane`; KEEP THE TWO
// ENDS IN SYNC.
//
// Every OTHER reader of this lane was already sign-safe, which is what made the sign free: the
// three `ext_night_w` decodes are `clamp(1 - z, 0, 1)`, and a negative `z` gives them the same 1.0
// the 0.0 they used to read gave them, so the exterior torch lane is bit-identical after dark.
fn sun_shadow_w(lane: f32) -> f32 {
    return max(lane, 0.0);
}

fn moon_shadow_w(lane: f32) -> f32 {
    return max(-lane, 0.0);
}

// MONKEY (moon shadows): ONE map fetch, the piece both weights share. Split out of
// `realtime_shadow` so `realtime_shadow_terms` can weight the SAME sample twice instead of
// sampling twice — the fetch is a 9-tap Gaussian PCF and the whole feature's cost budget is "no new
// passes, no new textures, and nothing a second time".
fn shadow_fetch(sample_pos: vec4<f32>, normal: vec3<f32>, view_z: f32) -> f32 {
    var shadow = 1.0;
    if (lights.n_directional_lights > 0u) {
        for (var light_id = 0u; light_id < lights.n_directional_lights; light_id = light_id + 1u) {
            if ((lights.directional_lights[light_id].flags & 1u) != 0u) {
                shadow = shadows::fetch_directional_shadow(light_id, sample_pos, normal, view_z);
                break;
            }
        }
    }
    return shadow;
}

// The realtime directional-shadow factor at `sample_pos`: 1.0 = lit, 0.0 = fully shadowed.
//
// - `sample_pos`   world-space point to sample the shadow map at — a per-fragment position, or a
//                  unit's nudged rig anchor (the caller chooses; that is the rig-skin decision).
// - `normal`       receiver normal, for the slope-aware depth bias.
// - `view_z`       view-space z, for cascade selection.
// - `cam_dist`     distance from the camera to the receiver fragment, for the EDGE fade.
// - `shadow_range` the cascade's max distance (the `shadowDistance` slider) — where the edge fade ends.
// - `night`        the sun-height strength (0 at night .. 1 by day) — the NIGHT fade.
//
// Both fades lighten the result toward 1.0 (no shadow). The caller passes `shadow_range`/`night`
// from its own light buffer (the receivers declare that buffer three incompatible ways, so the hook
// takes the scalars as arguments rather than reading it).
fn realtime_shadow(
    sample_pos: vec4<f32>,
    normal: vec3<f32>,
    view_z: f32,
    cam_dist: f32,
    shadow_range: f32,
    night: f32,
) -> f32 {
    let shadow = shadow_fetch(sample_pos, normal, view_z);
    let edge_fade = smoothstep(shadow_range - SHADOW_EDGE_BAND, shadow_range, cam_dist);
    return 1.0 - (1.0 - shadow) * night * (1.0 - edge_fade);
}

// MONKEY (moon shadows): BOTH directional weights over ONE fetch — `.x` the SUN arm (exactly what
// `realtime_shadow` above returns, character for character), `.y` the MOON arm.
//
// The three receivers call this instead of `realtime_shadow` because they apply the two arms to
// DIFFERENT terms, and must do so from the same sample:
//   · the SUN arm keeps its existing home in each receiver (terrain's `character_shadow_term`, the
//     model's and static_gx's `ambient + (lit − ambient) × shadow_term`), untouched;
//   · the MOON arm scales the exterior NIGHT law — see each receiver's own note for which term and
//     why point lights, torches and spell lights are outside it.
//
// Both arms are written as the ONE original expression rather than factored through a shared
// `occ = (1 − shadow)·(1 − edge_fade)`: float multiply is not associative, and `.x` must be the
// same BITS as before this function existed. At `moon == 0.0` — every daylight frame, every frame
// with `moonShadowStrength 0`, and the ≈20:30-22:17 window with the sun down and the moon not yet
// up — `.y` is exactly `1.0` whatever the map holds, and every receiver's moon branch is guarded on
// exactly that, so it is not entered and the render is the pre-feature one.
fn realtime_shadow_terms(
    sample_pos: vec4<f32>,
    normal: vec3<f32>,
    view_z: f32,
    cam_dist: f32,
    shadow_range: f32,
    night: f32,
    moon: f32,
) -> vec2<f32> {
    // MONKEY (moon shadows): neither body casts during hand-over or feature-off night. Skip
    // the map entirely, including PCF; the old night cost must not grow with an inert moon.
    if (night <= 0.0 && moon <= 0.0) {
        return vec2<f32>(1.0);
    }
    let shadow = shadow_fetch(sample_pos, normal, view_z);
    let edge_fade = smoothstep(shadow_range - SHADOW_EDGE_BAND, shadow_range, cam_dist);
    return vec2<f32>(
        1.0 - (1.0 - shadow) * night * (1.0 - edge_fade),
        1.0 - (1.0 - shadow) * moon * (1.0 - edge_fade),
    );
}

// GFX (moonlight): `realtime_shadow_terms` plus `.z` = the MOONLIGHT visibility — how much of the
// additive moon term (`benilla::moonlight_hook`) reaches this fragment. Unlike `.y` (which darkens
// the stock night SKY by `moonShadowStrength`), `.z` shadows the moon's OWN light fully, faded by
// the rig's hand-over ramp `moon_conf` (MonkeyFrame `misc.w`) and the cascade edge. `.z` is 1.0
// whenever the map does not hold the moon (`moon <= 0` or `moon_conf <= 0`): an unshadowed moon
// term rather than the sun's map applied to the moon. `.x`/`.y` are exactly `realtime_shadow_terms`.
fn realtime_shadow_moonlit(
    sample_pos: vec4<f32>,
    normal: vec3<f32>,
    view_z: f32,
    cam_dist: f32,
    shadow_range: f32,
    night: f32,
    moon: f32,
    moon_conf: f32,
) -> vec3<f32> {
    if (night <= 0.0 && moon <= 0.0) {
        return vec3<f32>(1.0);
    }
    let shadow = shadow_fetch(sample_pos, normal, view_z);
    let edge_fade = smoothstep(shadow_range - SHADOW_EDGE_BAND, shadow_range, cam_dist);
    var lit = 1.0;
    if (moon > 0.0 && moon_conf > 0.0) {
        lit = 1.0 - (1.0 - shadow) * clamp(moon_conf, 0.0, 1.0) * (1.0 - edge_fade);
    }
    return vec3<f32>(
        1.0 - (1.0 - shadow) * night * (1.0 - edge_fade),
        1.0 - (1.0 - shadow) * moon * (1.0 - edge_fade),
        lit,
    );
}

// MONKEY (torch shadows Phase 1): the point/cluster-based torch shadow is DEAD. benilla's world
// camera sets `ClusterConfig::None`, which starves `clusterable_objects` AND the point-shadow prep,
// so `fetch_point_shadow`/the clusterable scan can never fire. The replacement is a from-scratch
// depth map (`benilla_world::static_gx::torch_depth` + `torch_map_shadow` below), sampled by the
// static_gx interior lane through its own group-3 bindings, and (Phase 3A) by `wow_model.wgsl`'s
// entity lane through its material's own bindings 91/92/93 — both call `torch_map_shadow` below.
// These two functions are kept only so `interior_debug_override` (mode 2) compiles; they return
// "unshadowed" and no receiver's lighting path calls them any more.
fn torch_shadow(sample_pos: vec4<f32>, normal: vec3<f32>) -> f32 {
    return 1.0;
}

fn torch_shadow_for(light_pos: vec3<f32>, sample_pos: vec4<f32>, normal: vec3<f32>) -> f32 {
    return 1.0;
}

// MONKEY (torch shadows Phase 1): the old clusterable-caster count diagnostic. The clustering is
// dead (see above), so this is now a constant 0 — kept only so `interior_debug_override` compiles.
fn torch_caster_count() -> u32 {
    return 0u;
}

// MONKEY (pcss): the contact-hardening constants.
//
// `TORCH_DEPTH_B` is the reverse-Z projection's depth OFFSET, mirrored from
// `benilla_app::torch_shadow::{reverse_z_perspective, cube_view_projs}` (near 0.1, far TORCH_RANGE
// 48 — every matrix this function is ever handed is one of those six cube faces). Under reverse-Z
// the stored depth is an AFFINE function of the RECIPROCAL distance along the face axis:
// `z = A/t + B`, `A = near*far/(far-near)`, `B = -near/(far-near)`. So `z - B` is proportional to
// `1/t`, and the one quantity a penumbra needs — the ratio of two distances — is
// `(z_blocker - B) / (z_receiver - B)` with `A` cancelling: no linearisation, no extra uniform, one
// subtraction per term. A constant rather than something decoded out of `view_proj` because the
// w-row of a look-at product carries the eye translation, so `B` cannot be read off a single
// element of that matrix without dividing two of them — and a wrong `B` would silently mis-size
// every penumbra instead of failing.
const TORCH_DEPTH_B: f32 = -0.1 / 47.9;
// How far out (in `soft` texels) the blocker search looks. Two: wide enough to find the caster of
// an edge a texel or two away, narrow enough that a fragment well inside a lit area still reads
// four empty texels and takes the cheap path.
const TORCH_PCSS_SEARCH: f32 = 2.0;
// Penumbra gain — extra `soft` radii per unit of `(d_receiver - d_blocker)/d_blocker`. A receiver
// twice as far from the fixture as its blocker (ratio 1) is filtered three times as wide; the
// Darkmoon tent (post ~2 yd from the torch, grass ~10) saturates the clamp below, and a chair leg
// resting ON the floor (ratio ~0.05) stays within 10 % of the hard kernel — the whole point.
const TORCH_PCSS_K: f32 = 2.0;
// …clamped at four radii. Past that the 8 taps are too sparse for the area they cover and the
// penumbra starts to band, and the shadow there is faint enough that widening it further is
// invisible anyway.
const TORCH_PCSS_MAX: f32 = 4.0;
// Above this filter radius (texels) the 4-tap box is too sparse and earns the second ring.
const TORCH_PCSS_WIDE: f32 = 1.5;

// MONKEY (slope bias): the per-tap RECEIVER-PLANE correction's ceiling, in multiples of the
// constant `bias`. A PCF/PCSS tap `n` texels off centre lands on a piece of the SAME surface whose
// stored depth differs by the receiver plane's own depth gradient times `n`; `torch_map_shadow`
// walks that tap's compare reference along the plane by exactly that much, so a tap on the
// receiver's own plane compares equal-to-equal and can never fail, at any grazing angle. What the
// correction must NOT do is walk so far that a tap on a real OCCLUDER also passes - that is
// peter-panning, a shadow lifting off its contact - so it is clamped at `TORCH_SLOPE_MAX * bias`.
//
// THE ARITHMETIC - and a CORRECTION to round 3's. The receivers' `TORCH_NORMAL_OFFSET` note prices
// all of this in "yards of RAY distance", which is not what `textureSampleCompareLevel` compares:
// the stored value is `ndc.z` on the cube FACE the fragment lands in, and under a perspective
// projection that is a function of the VIEW-SPACE Z - the component along the FACE AXIS - not of
// the ray length. Two things follow that the round-3 table cannot show:
//
//   · a floor directly under a fixture lands in the DOWN face, where its view-space Z is the
//     constant `h`, so its stored depth is CONSTANT across that whole face and there is no
//     gradient to clear at all - acne is impossible there. It begins the moment `r > h` and the
//     fragment crosses into a SIDE face, whose view-space Z is the HORIZONTAL distance and along
//     which the floor runs away from the eye. On a side face the gradient works out to `2A/(f*h)`
//     per unit uv (`A = near*far/(far-near)`, `f = 1/tan(fov/2)`) - i.e. it depends on the
//     fixture's HEIGHT and not at all on `r`: the same at 3 yd out as at 30.
//   · the note's normal-offset term `0.15*(h/t)` is INVERTED. Moving the sample point 0.15 yd up
//     off a grazing floor moves the place where its own shadow ray meets that floor by
//     `0.15*(t/h)` - amplified by the grazing angle, not reduced by it. So the offset is worth
//     about `(t/h)^2` times what round 3 credited it with, which is why a knee-high fixture's pool
//     was in fact clean to ~10 yd rather than to ~2.1.
//
// Re-measured exactly (scratchpad `slopebias_exact.py`: the real reverse-Z cube face, the real
// `torch_face` pick on the OFFSET point, the widened kernel's 8 taps at 3 and 6 texels, each
// SNAPPED to its texel centre, compared against the real floor plane). BREAK RADIUS = how far a
// fixture `h` yd above a flat floor lights before its own worst tap starts to fail:
//
//     h (yd)      before        after
//     4.0         clean         clean        a chandelier, a lamppost head
//     3.0         clean         clean        a WALL TORCH - always was fine, and stays fine
//     2.0        25.8 yd        clean        a brazier
//     1.5        14.1 yd        clean
//     1.0        10.1 yd        clean        the IMP's hand flame - the reported case
//     0.5         7.3 yd        clean        a candle standing on the floor
//     0.25        7.1 yd       11.7 yd       CLAMPED - see below
//
// "clean" = no tap fails anywhere out to the projection's usable 46 yd, i.e. the acne is GONE and
// not merely pushed outward. Where a tap was already passing the margin simply widens, at h = 1 yd:
// +3.5e-4 -> +2.3e-3 at r = 9, +1.7e-3 -> +3.6e-3 at r = 5; and where it was failing, -5.1e-4 ->
// +1.5e-3 at r = 15. The h = 3 wall-torch rows move 1.0x-2.7x in the same direction and never
// change verdict, which is the "forge floor unchanged, still attached at contact" requirement.
//
// The BLOCKER SEARCH matters as much as the taps. At h = 1, r = 15 its own 3-texel taps were
// self-detecting (-5.1e-4). That darkens nothing, but it hands `TORCH_PCSS_K` a blocker ratio made
// entirely of bias error - a pool going soft with nothing in it. Corrected: +1.5e-3.
//
// WHY 4 AND NOT MORE. The correction the outermost tap WANTS is 2.8e-3 at h = 1 and 6.8e-3 at
// h = 0.5, against the `4*bias` = 4e-3 ceiling: free at h = 1, clamped at h = 0.5 - and clamped is
// still enough there (+3.5e-3 at r = 9). It only falls short under a QUARTER-yard fixture, a light
// effectively lying on the ground, where the surface is near-edge-on to it, `max(N.L, 0)` has
// already taken the term toward zero, and refusing to move further is the conservative answer. A
// TILTED receiver wants LESS, not more (2.7e-4 at 45 deg, 3.0e-4 at 75 deg: the face pick swings
// back to the DOWN face and the gradient collapses), so raising the ceiling would buy nothing
// anywhere except the one case where it is deliberately declining to act, while starting to lift
// genuine contact shadows - the single failure mode a shadow bias must never have.
const TORCH_SLOPE_MAX: f32 = 4.0;

// MONKEY (slope bias): ONE tap's compare reference - the centre reference walked along the RECEIVER
// PLANE by `dot(grad, off)` and bounded at +/-`lim`. `grad` is `d(ndc.z)/d(uv)` restricted to that
// plane (built once per call in `torch_map_shadow`), `off` is the tap's uv displacement from the
// centre. Three ALU and a clamp, no texture read. It is a function rather than four inline
// expressions so a tap's OFFSET and its REFERENCE cannot drift apart - they are the same argument.
fn torch_plane_ref(ref_depth: f32, grad: vec2<f32>, off: vec2<f32>, lim: f32) -> f32 {
    return ref_depth + clamp(dot(grad, off), -lim, lim);
}

// MONKEY (pcss): one RAW depth texel of a torch map — the blocker search's read. `textureLoad`
// takes no sampler and imposes no uniformity requirement, which is what makes the search legal
// inside the receivers' per-fixture loop. The clamp is the border rule: a search that walks off the
// face re-reads its edge texel, the same answer the comparison sampler's ClampToEdge address mode
// gives the PCF taps beside it.
fn torch_map_depth(
    depth_tex: texture_depth_2d_array,
    layer: i32,
    uv: vec2<f32>,
    dims: vec2<f32>,
) -> f32 {
    let c = clamp(vec2<i32>(floor(uv * dims)), vec2<i32>(0, 0), vec2<i32>(dims) - vec2<i32>(1, 1));
    return textureLoad(depth_tex, c, layer, 0);
}

// MONKEY (torch shadows Phase 1): the depth-map projector. Because `shadow_hook` is imported by
// receivers WITHOUT the group-3 torch bindings (wow_model, terrain), the texture + comparison
// sampler come in AS PARAMETERS — the caller supplies them only where they are bound (static_gx).
//   - `view_proj`  the fixture's down-looking reverse-Z matrix.
//   - `layer`      which array layer (fixture index) to sample.
//   - `world_pos`  the receiving fragment's world position.
//   - `normal`     MONKEY (slope bias): the receiving surface's world-space normal, used ONLY to
//                  build the receiver plane whose depth gradient each tap's compare reference is
//                  walked along (see `TORCH_SLOPE_MAX`). It need not be unit and may be the zero
//                  vector - the M2 corpus authors those, and a zero normal simply switches the
//                  slope term off, leaving the constant bias exactly as it was. Callers pass the
//                  SAME normal they already offset `world_pos` along (`TORCH_NORMAL_OFFSET`), so
//                  the offset point and the plane through it agree by construction.
//   - `depth_tex`/`comp`  the group-3 depth array + `GreaterEqual` comparison sampler.
//   - `bias`       reverse-Z receiver bias (nudges the compare ref up to kill self-shadow acne).
//                  MONKEY (slope bias): still a CONSTANT floor under every tap - the slope term is
//                  added on top of it per tap, never in place of it.
//   - `soft`       MONKEY (torch caster selection): the BASE PCF tap-radius scale
//                  (`interiorShadowSoft`, 0.5..3, carried in the table's `count.y` LOW half as
//                  `soft x 100`). The four taps sit at ±0.5 texel × this; 1 is the historical
//                  half-texel box. MONKEY (pcss): it is now the radius at a CONTACT (a receiver
//                  touching its blocker) — the contact-hardening search below grows it with the
//                  blocker distance, so `soft` sets how sharp the sharpest edge in the scene is
//                  and every penumbra scales off it.
//   - `fade`       MONKEY (outdoor torch shadows): WHICH far fade ends the shadow.
//                  **Negative = the interior/legacy behaviour, bit-for-bit** — the reverse-Z
//                  far-plane fade computed from `ndc.z` below. Zero-or-positive = the CALLER's own
//                  fade weight, used verbatim (0 = no shadow left, 1 = full strength).
//
//                  Why an override at all: the projection is near/far `0.1/48`
//                  (`torch_shadow::cube_view_projs`), so under reverse-Z `ndc.z` is
//                  `near·(far−t)/((far−near)·t)` = **0.0079 at t = 10 yd**, which
//                  `smoothstep(0, 0.06, ·)` turns into a 5 % shadow. For a candle whose pool is
//                  3-4 yd that is a correct soft end; for a campfire whose pool is 15-25 yd it
//                  erases the shadow exactly where the shadow IS the effect. The fix cannot live
//                  in here — it wants to be a function of WORLD distance from the fixture, and
//                  this function is deliberately a pure projector that never sees the fixture — so
//                  the exterior lane computes `1 − smoothstep(0.8R, R, d)` at its call site.
// Returns 1.0 outside the frustum / behind the light; a PCF factor otherwise (0 = shadowed).
//
// MONKEY (pcss): CONTACT-HARDENING. The old kernel was one fixed ±0.5·soft texel box everywhere,
// which is why a torch beside a Darkmoon tent post threw the canvas onto the grass fifteen yards
// away as a razor-edged cut-out: a real penumbra grows with the receiver's distance FROM ITS
// BLOCKER, and a constant kernel has no way to know that distance. The three steps below recover
// it from the map itself — search for blockers, estimate how far in front of the receiver they
// are, widen the filter by that much — so a chair leg stays crisp where it meets the floor while
// the tent's edge goes soft where the canvas is metres above the grass.
//
// The blocker search reads RAW depth with `textureLoad`, not through the comparison sampler: the
// array is bound as `texture_depth_2d_array` (wgpu sample type Depth), and a Depth binding serves
// BOTH `textureSampleCompare*` (through `comp`) and an unfiltered `textureLoad` that takes no
// sampler at all — so no binding, layout or pipeline changed. That is worth insisting on, because
// the alternative approximation (infer the blocker from the FRACTION of blocked compares at two or
// three radii) only ever answers "how deep inside the shadow am I", never "how far in front of me
// is the caster", and those two differ by exactly the quantity a penumbra is made of.
//
// Cost: 4 unfiltered loads, then 4 comparison taps (the shipped kernel, unchanged) whenever the
// search finds no blocker — which is every fully-lit fragment, i.e. most of a pool — or 8 when the
// widened radius earns them. Worst case 8 compares + 4 loads per fixture per fragment.
// MONKEY (slope bias) adds NO texture read at all: ~90 ALU once per call (two crosses, two
// `mat4x4 * vec4`, a 2x2 solve) to build the receiver-plane gradient, then a dot-clamp-add — about
// 5 ALU — on each of the 4 search taps and 4 or 8 PCF taps. The gradient cannot hoist out of the
// receivers' per-fixture loop (it depends on the fixture's own face matrix), but it is bounded by
// `EXT_SEL_SHADOWED` there exactly as the taps are.
fn torch_map_shadow(
    view_proj: mat4x4<f32>,
    layer: i32,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    depth_tex: texture_depth_2d_array,
    comp: sampler_comparison,
    bias: f32,
    soft: f32,
    fade: f32,
) -> f32 {
    let clip = view_proj * vec4<f32>(world_pos, 1.0);
    if (clip.w <= 0.0) {
        return 1.0;
    }
    let ndc = clip.xyz / clip.w;
    if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    let dims = vec2<f32>(textureDimensions(depth_tex).xy);
    let inv_dims = 1.0 / dims;
    // MONKEY (torch caster selection): the base radius, clamped off zero so a table that never got
    // a scale still samples a real (hard) box rather than four copies of one texel.
    let base = max(soft, 0.05);
    // Reverse-Z: the fragment is lit iff its own depth is at least the stored nearest depth, so the
    // compare ref is `ndc.z + bias` against `GreaterEqual`. UNCHANGED bias semantics: this is the
    // CENTRE reference, and the blocker test and every comparison tap are derived from this one
    // expression, so a surface can no more blocker-detect itself than it could shadow itself.
    // MONKEY (slope bias): "derived from" rather than "is" - each tap now adds its own
    // receiver-plane term on top (`torch_plane_ref` below), which strengthens that property rather
    // than weakening it: the taps that used to self-detect were exactly the off-centre ones.
    let ref_depth = ndc.z + bias;

    // MONKEY (slope bias) — RECEIVER-PLANE DEPTH BIAS. `bias` is a constant and the thing it has to
    // clear is not: a tap `n` texels off centre reads a piece of the SAME surface whose stored depth
    // differs by the receiver plane's own gradient times `n`, and on a floor lit from a low angle
    // that gradient is the whole problem (`TORCH_SLOPE_MAX` has the measured numbers — a knee-high
    // flame's floor starts losing to it past 10 yd, a candle's past 7, and the taps that fail
    // print as faint rings out at the rim of the pool where the term is still visible). So every
    // tap gets its OWN reference, walked along the receiver's plane by what
    // that plane does over that tap's displacement: a tap sitting on the plane then compares
    // equal-to-equal and cannot fail at any grazing angle, while a tap on a real occluder is
    // untouched — the occluder is not on this plane, which is what makes it an occluder.
    //
    // `grad` is `d(ndc.z)/d(uv)` restricted to the plane through `world_pos` with normal `normal`,
    // computed ANALYTICALLY rather than from `dpdx`/`dpdy` of the ndc. Two reasons, both hard:
    //   · every call site is inside a receiver's per-fixture table scan, with data-dependent
    //     `continue`s and an early `return` (`torch_map_at`, `torch_terrain_shadow`,
    //     `torch_entity_shadow_at`) — NON-UNIFORM control flow, where WGSL's derivative-uniformity
    //     rule makes `dpdx` a diagnostic at best and a neighbouring-fixture read at worst;
    //   · screen-space derivatives are a QUAD difference, so they are wrong by construction on a
    //     two-pixel-wide sliver and along every silhouette — exactly the geometry (fence rails,
    //     chair legs, tent ropes) this lane spends its taps on.
    // The analytic form needs no quad and no uniformity, and is EXACT for a plane.
    //
    // Any two independent vectors IN the plane give the same gradient, so the tangents are left
    // un-normalised and `normal` need not be unit: only the plane's direction SPACE matters and the
    // 2x2 solve is invariant to how it is spanned. `a` picks the axis the normal is least aligned
    // with, so the first cross is never near-degenerate. A ZERO normal (the M2 corpus authors them
    // — see `wow_normalize` in the receivers) leaves `grad` at zero, i.e. the old constant-bias
    // behaviour, bit-for-bit.
    var grad = vec2<f32>(0.0);
    if (dot(normal, normal) > 1e-12) {
        let a = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0),
                       abs(normal.x) > abs(normal.z));
        let tangent_first = cross(normal, a);
        let tangent_second = cross(normal, tangent_first);
        // d(clip) along each tangent, then through the perspective divide:
        // `d(ndc) = (d(clip).xyz - ndc * d(clip).w) / clip.w`.
        let clip_first = view_proj * vec4<f32>(tangent_first, 0.0);
        let clip_second = view_proj * vec4<f32>(tangent_second, 0.0);
        let inv_w = 1.0 / clip.w;
        let grad_first = (clip_first.xyz - ndc * clip_first.w) * inv_w;
        let grad_second = (clip_second.xyz - ndc * clip_second.w) * inv_w;
        let uv_first = grad_first.xy * vec2<f32>(0.5, -0.5);   // the same ndc -> uv flip `uv` above uses
        let uv_second = grad_second.xy * vec2<f32>(0.5, -0.5);
        // Solve `dot(grad, u_i) = g_i.z` for i = 1, 2. `det` collapses only as the plane goes
        // EDGE-ON to the fixture, where the plane projects to a line in the map, `max(N·L, 0)` has
        // already taken the lit term to zero, and a zero gradient is the right answer anyway.
        let det = uv_first.x * uv_second.y - uv_first.y * uv_second.x;
        if (abs(det) > 1e-9) {
            let inv = 1.0 / det;
            grad = vec2<f32>((grad_first.z * uv_second.y - uv_first.y * grad_second.z) * inv,
                             (uv_first.x * grad_second.z - grad_first.z * uv_second.x) * inv);
        }
    }
    // …and the ceiling on what any one tap may be moved by. See `TORCH_SLOPE_MAX`.
    let slope_lim = TORCH_SLOPE_MAX * bias;

    // MONKEY (pcss) 1/3 — BLOCKER SEARCH. Four raw reads on the diagonals of a box of half-width
    // `soft * TORCH_PCSS_SEARCH` texels, averaging the depths that lie IN FRONT of the receiver.
    // Four taps at one radius rather than a ring at several: they only have to answer "is there a
    // caster near me, and roughly how far in front", and denser searching costs more than the extra
    // precision buys on a 512² face whose texels are ~10 cm at a candle's range.
    let search = base * TORCH_PCSS_SEARCH * inv_dims;
    // MONKEY (slope bias): the offsets are NAMED so the search's reference and its sample are the
    // same displacement, exactly as in the PCF kernel below.
    let search_offset_a = vec2<f32>(-1.0, -1.0) * search;
    let search_offset_b = vec2<f32>( 1.0, -1.0) * search;
    let search_offset_c = vec2<f32>(-1.0,  1.0) * search;
    let search_offset_d = vec2<f32>( 1.0,  1.0) * search;
    let blocker_a = torch_map_depth(depth_tex, layer, uv + search_offset_a, dims);
    let blocker_b = torch_map_depth(depth_tex, layer, uv + search_offset_b, dims);
    let blocker_c = torch_map_depth(depth_tex, layer, uv + search_offset_c, dims);
    let blocker_d = torch_map_depth(depth_tex, layer, uv + search_offset_d, dims);
    var blocker = 0.0;
    var blockers = 0.0;
    // MONKEY (slope bias): the SEARCH is plane-corrected too, and it has to be. Left on the flat
    // reference, a grazing floor detects ITSELF as its own blocker, and the penumbra width below is
    // then estimated from a depth difference that is pure bias error — a pool that goes soft with
    // nothing in it. Same law, same `grad`, same clamp as the taps.
    if (blocker_a > torch_plane_ref(ref_depth, grad, search_offset_a, slope_lim)) { blocker += blocker_a; blockers += 1.0; }
    if (blocker_b > torch_plane_ref(ref_depth, grad, search_offset_b, slope_lim)) { blocker += blocker_b; blockers += 1.0; }
    if (blocker_c > torch_plane_ref(ref_depth, grad, search_offset_c, slope_lim)) { blocker += blocker_c; blockers += 1.0; }
    if (blocker_d > torch_plane_ref(ref_depth, grad, search_offset_d, slope_lim)) { blocker += blocker_d; blockers += 1.0; }

    // MONKEY (pcss) 2/3 — PENUMBRA WIDTH. `radius` stays at `base` when the search found nothing: a
    // caster thinner than the search box (a chair leg, a tent rope, a candlestick) must NOT be
    // widened away to nothing, and such a fragment is then filtered by exactly the kernel this
    // function has always used.
    var radius = base;
    if (blockers > 0.0) {
        // `(d_receiver - d_blocker) / d_blocker`, exact — see TORCH_DEPTH_B: `z - B` is
        // proportional to `1/distance`, so the quotient of the two `z - B` terms IS the quotient of
        // the two distances, and the projection's scale factor cancels. The receiver term uses the
        // UNBIASED `ndc.z` (the bias exists to keep a surface off its own stored depth, not to move
        // the surface), and `max` only guards f32 noise: inside the frustum a detected blocker is
        // always the nearer of the two.
        let ratio = max((blocker / blockers - TORCH_DEPTH_B) / (ndc.z - TORCH_DEPTH_B) - 1.0, 0.0);
        radius = min(base * (1.0 + TORCH_PCSS_K * ratio), base * TORCH_PCSS_MAX);
    }

    // MONKEY (pcss) 3/3 — PCF at that radius. The 4-tap box below `TORCH_PCSS_WIDE` texels is the
    // shipped kernel bit-for-bit (`radius == base` reproduces the old `texel` expression exactly);
    // past it that box would alias into four separate soft bands, so a second ring of four cardinal
    // taps at the full radius fills the area in.
    let texel = inv_dims * radius;
    // MONKEY (slope bias): every tap carries its OWN plane-corrected reference. With `grad` zero -
    // a receiver square-on to the fixture, or a zero normal - `torch_plane_ref` returns `ref_depth`
    // unchanged and this kernel is the shipped one bit-for-bit.
    let pcf_offset_a = vec2<f32>(-0.5, -0.5) * texel;
    let pcf_offset_b = vec2<f32>( 0.5, -0.5) * texel;
    let pcf_offset_c = vec2<f32>(-0.5,  0.5) * texel;
    let pcf_offset_d = vec2<f32>( 0.5,  0.5) * texel;
    var sum = 0.0;
    sum += textureSampleCompareLevel(depth_tex, comp, uv + pcf_offset_a, layer,
                                     torch_plane_ref(ref_depth, grad, pcf_offset_a, slope_lim));
    sum += textureSampleCompareLevel(depth_tex, comp, uv + pcf_offset_b, layer,
                                     torch_plane_ref(ref_depth, grad, pcf_offset_b, slope_lim));
    sum += textureSampleCompareLevel(depth_tex, comp, uv + pcf_offset_c, layer,
                                     torch_plane_ref(ref_depth, grad, pcf_offset_c, slope_lim));
    sum += textureSampleCompareLevel(depth_tex, comp, uv + pcf_offset_d, layer,
                                     torch_plane_ref(ref_depth, grad, pcf_offset_d, slope_lim));
    var pcf = sum * 0.25;
    if (radius > TORCH_PCSS_WIDE) {
        let pcf_offset_e = vec2<f32>(-1.0,  0.0) * texel;
        let pcf_offset_f = vec2<f32>( 1.0,  0.0) * texel;
        let pcf_offset_g = vec2<f32>( 0.0, -1.0) * texel;
        let pcf_offset_h = vec2<f32>( 0.0,  1.0) * texel;
        sum += textureSampleCompareLevel(depth_tex, comp, uv + pcf_offset_e, layer,
                                         torch_plane_ref(ref_depth, grad, pcf_offset_e, slope_lim));
        sum += textureSampleCompareLevel(depth_tex, comp, uv + pcf_offset_f, layer,
                                         torch_plane_ref(ref_depth, grad, pcf_offset_f, slope_lim));
        sum += textureSampleCompareLevel(depth_tex, comp, uv + pcf_offset_g, layer,
                                         torch_plane_ref(ref_depth, grad, pcf_offset_g, slope_lim));
        sum += textureSampleCompareLevel(depth_tex, comp, uv + pcf_offset_h, layer,
                                         torch_plane_ref(ref_depth, grad, pcf_offset_h, slope_lim));
        pcf = sum * 0.125;
    }
    // Phase 5: the six cube faces tile the whole sphere, so there is no cone edge to soften — a fade
    // there would punch a lit seam along every face border. Only the far plane fades (reverse-Z:
    // ndc.z → 0 at far), so a shadow at the fixture's range limit ends softly.
    // MONKEY (outdoor torch shadows): …unless the caller supplied its own (see `fade`). `select`
    // rather than a branch so both arms cost the same and the interior arm is the identical
    // expression it always was.
    let ndc_fade = smoothstep(0.0, 0.06, ndc.z);
    return mix(1.0, pcf, select(fade, ndc_fade, fade < 0.0));
}

// MONKEY (interior debug): the interior lane's diagnostic overlay. `mode` is decoded from the
// receiver's `wmo_fog_params.w` (`1 + debug`); the receivers call this at the END of their interior
// branch so ONLY interior-lit fragments are recoloured — a wrongly-classified outdoor entity then
// stands out against the normally-shaded world. `mode == 0` returns the real `rgb` unchanged.
//   1 CLASSIFICATION → solid green (this fragment took the interior lane)
//   2 SHADOW         → the torch-shadow factor as greyscale (black shadowed .. white lit)
//   3 CASTER COUNT   → red 0 / green 1 / blue 2 / white 3+ shadow proxies visible to this view
fn interior_debug_override(
    mode: u32,
    rgb: vec3<f32>,
    sample_pos: vec4<f32>,
    normal: vec3<f32>,
) -> vec3<f32> {
    if (mode == 1u) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    if (mode == 2u) {
        return vec3<f32>(torch_shadow(sample_pos, normal));
    }
    if (mode == 3u) {
        let n = torch_caster_count();
        if (n == 0u) { return vec3<f32>(1.0, 0.0, 0.0); }
        if (n == 1u) { return vec3<f32>(0.0, 1.0, 0.0); }
        if (n == 2u) { return vec3<f32>(0.0, 0.0, 1.0); }
        return vec3<f32>(1.0, 1.0, 1.0);
    }
    return rgb;
}
