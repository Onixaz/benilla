// The model pass for M2, WMO and ground-clutter meshes, lit and fogged in gamma space:
//   M2:           color = clamp01(A + D·I·(4/17)(0.375 + 2μ + 1.875μ²)) × tex × tint, μ = N·L
//   clutter, WMO: color = clamp(ambient + diffuse·max(N·L, 0)) × tex × tint
//   fog:          color = mix(fog_color, color, fog_factor); out = color, raw gamma
// The M2 law is the order-2 SH lobe of `Shaders\Vertex\Model2.bls` (cvar `M2UseShaders`, default
// 1); M2's one FFP light site (`70bdf6`) runs only with that cvar off. Clutter and WMO are the FFP
// light (GL_LIGHTING, GL_LIGHT0, GL_COLOR_MATERIAL); clutter's normal is the terrain normal under
// the tuft, which the reference writes onto the clutter vertex.
// Not built: a specular term (the M2 per-material shininess is only inferred) and the WMO
// per-group authored colour.
// Ported from WarcraftXL (https://github.com/WarcraftXL) by iThorgrim — module
// wxl-experimental-wind, grass/GrassWind.cpp (grass vertex displacement seam).

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    pbr_bindings,
    forward_io::VertexOutput,
    mesh_view_bindings::{lights, view},
    shadows,
    mesh_functions,
}
// MONKEY (shadow hook): the realtime directional-shadow term (fetch + edge/night fade) lives here.
#import benilla::shadow_hook
// GFX (moonlight): the additive moon term.
#import benilla::moonlight_hook
// MONKEY (p0 MonkeyFrame): the programme block's struct, mirrored after the point table.
#import benilla::monkey_frame
// MONKEY (wet): rain on surfaces (wet_hook.wgsl).
#import benilla::wet_hook
// MONKEY (p0 fog hook): the one distance-fog law every receiver calls.
#import benilla::fog_hook
// MONKEY (post): shared tier-gated HDR emission; Off is an exact identity.
#import benilla::emissive_hook
// MONKEY (wind): WarcraftXL-derived grass motion and the matching tree fade-twin motion.
#import benilla::wind_hook

// bevy_pbr 0.18.1's `forward_io::FragmentOutput`. No depth output: a fragment depth write costs
// the pipeline early-Z, so the sky lane pins its depth in the vertex stage.
struct WowFragOut {
    @location(0) color: vec4<f32>,
}

// Per-material uniforms at binding 100, in `WowModelExt`'s field order (materials.rs).
//   clutter_fade: x = plateau-end view depth (yd, 0.75·far), y = ramp-zero view depth (the ~70 yd
//     detail-doodad horizon `[0x867958]`), z = the batch marker bits, w = clutter
//   model_flags: x = WMO, y = fade blend twin, z = interior (a WMO interior group, or an interior
//     M2 lit by its SH probe), w = unlit fullbright (M2 UNLIT 0x01 or Mod/Mod2x, or WMO UNLIT on
//     an exterior-group batch; the interior drawer ignores it)
struct ModelParams {
    clutter_fade: vec4<f32>,
    model_flags: vec4<f32>,
    // x = the terrain-shade selector (see the doodad sun), y = the WMO batch order, zw = the
    // UV-scroll seed.
    sun_scale: vec4<f32>,
    // xyz = the animated M2Color tint (identity when static); w = the WMO interior batch class:
    // 0 exterior law, 1 INT, 2 TRANS.
    tint: vec4<f32>,
    // WMO glass, 0 on M2: xyz = the MOMT SIDN (0x10) emissive (gamma /255), w = MOMT WINDOW (0x20).
    sidn: vec4<f32>,
    // Rows of `wow_light.matanim`, 0 = identity: x = UV scroll, y = tint, z = texture-transform
    // affine, w = the UI tile's cell clip.
    anim_slots: vec4<f32>,
    // MONKEY (skybox): stage 1 of a two-texture batch: x = 0 none / 1 Mod / 2 Mod2x, z/w = the
    // matanim rows of its translation and affine.
    stage1: vec4<f32>,
};
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> m: ModelParams;
// MONKEY (skybox): stage 1's texture, the fallback image off `WOW_STAGE1`.
@group(#{MATERIAL_BIND_GROUP}) @binding(94) var stage1_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(95) var stage1_sampler: sampler;

// A fully covered character shadow retains 45% of the authored model lighting.
const SHADOW_SUN_FLOOR: f32 = 0.45;

// The shared global light (lighting::global_light): ONE storage buffer every material reads, updated
// once/frame in place — replaces the per-material light/fog uniforms the old apply_wow_lighting re-pushed
// each frame. The model reads rows 0-2 (ambient/diffuse/sun) + fog, plus rows 6-12: the disassembled
// `Model2.bls` probe of the day/night scene light at intensity 1 — the reference's own exterior M2
// response, and since 0803 the live law on the doodad/entity lane.
// Shared with terrain.wgsl — a tree and the dirt under it fade into the same haze.
// (light_spec is part of the prefix layout but the model path never reads it.)
struct WowLight {
    light_ambient: vec4<f32>, // rgb ambient; w = Mod2x scale
    light_diffuse: vec4<f32>, // rgb sun diffuse; w = clamp-light flag (>0.5 ⇒ saturate)
    light_sun: vec4<f32>,     // xyz sun travel dir (to-light = −xyz); w = directional enable
    light_spec: vec4<f32>,    // terrain's specular (w = its shininess); unread here
    fog_color: vec4<f32>,     // rgb row-7 fog (gamma); w = enable (>0.5)
    fog_params: vec4<f32>,    // x=start y=end z=signed +sun / -moon shadow weight w=farclip wall
    // The global Model2.bls SH rows (6-12): the scene day/night light as an order-2 probe at
    // intensity 1 — DC (ambient, `.w` of c10) + the sun's linear/quad bands in c10.xyz / c13 /
    // c16.xyz, the disassembled closed form (wow-re model2-bls-vertex-sh.md). Every sun band is
    // linear in the committed colour, so a consumer scales them ALL by the per-instance intensity
    // (never I²); the sun's DC redistribution rides `grade.yzw` (× intensity).
    //
    // **LIVE — the exterior doodad/entity response (0803).** History worth knowing before you touch
    // it: 0410 took this lane OFF the curve onto a hard-cutoff FFP matte on the director's look call,
    // and the block then sat packed-but-unread for months. 0796 refuted the fidelity premise behind
    // that retirement (the reference's M2 lane IS this SH shader, not an FFP light slot — the FFP
    // commits in a world frame are terrain's), which reopened the call as a pure look question;
    // 0799 restored the eval behind an A/B so it could be seen; 0803 is the director calling it,
    // and the cutoff branch came out with the flag.
    sh_c10_r: vec4<f32>,
    sh_c10_g: vec4<f32>,
    sh_c10_b: vec4<f32>,
    sh_c13_r: vec4<f32>,
    sh_c13_g: vec4<f32>,
    sh_c13_b: vec4<f32>,
    // xyz = the true c16 quad band (x²−y², per channel), live with the block above. `.w` is the
    // SHARED world-shadow / bake-floor lane, and it is no longer free: its INTEGER part is the
    // `worldShadows` flag (`terrain.wgsl`'s MCSH gate reads `> 0.5`) and MONKEY (bake floor) its
    // FRACTION is `interiorBakeFloor × interiorGain × BAKE_LANE_SCALE`, which the interior entity
    // lane below decodes. (It had carried the 0273 point gain, the 0750/0751 sun dial and 0799's
    // response A/B before those were retired — hence the long-standing "free" note here.)
    sh_c16: vec4<f32>,
    _water: array<vec4<f32>, 4>, // rows 13-16: the liquid swatches, unread here
    // x = the SIDN night fraction (1 overnight, 0 by day); yzw = the sun's SH DC at intensity 1.
    grade: vec4<f32>,
    // Rows 18-19: the interior fog, the 4 s camera-in-WMO MFOG crossfade (the scene fog outdoors).
    wmo_fog_color: vec4<f32>,    // rgb interior fog (gamma); w = enable (mirrors fog_color.w)
    wmo_fog_params: vec4<f32>,   // x = start yd; y = end yd; zw = free lanes (retired A/B dials)
    // The dynamic point-light table (decision 0278), packed by `global_light::build_light_data`:
    // row 20 `.x` = live entry count; then TWO rows per light — `[pos.xyz, range]`, `[rgb, lane]`.
    // Rides this buffer (not bevy's clusterables) because the view layout exposes those to the
    // fragment stage only, and the Gouraud term is evaluated in the VERTEX stage.
    //
    // MONKEY (light lanes): the colour row's `.w` splits the table into two DISJOINT halves.
    // `0` = an EXTERIOR light (nothing claims a room) — read only by `point_light_sum`. Anything
    // `> 0.5` = an INTERIOR fixture — read only by `interior_room_light`, and the value itself is
    // that fixture's REACH IN YARDS (its authored MOLT attenuation end, or the M2 intensity
    // bucket, already scaled by the live `interiorAttenScale`). One float carries both because a
    // packed reach is always ≥ 1 yd and can never be mistaken for the exterior 0.
    point_count: vec4<f32>,
    points: array<vec4<f32>, 512>,
    // MONKEY (p0 MonkeyFrame): the programme block after the point table (monkey_frame.wgsl).
    monkey: monkey_frame::MonkeyFrame,
    // MONKEY (rainshelter): the rain-occlusion grid (`weather/shelter.rs`): `[origin_x, origin_z,
    // 1/cell, active]`, `[base_y, cells per side, 0, 0]`, then one packed word a cell (wet_hook.wgsl).
    shelter_hdr: vec4<f32>,
    shelter_cfg: vec4<f32>,
    shelter: array<u32, 16384>,
    // The interior-prop SH probes, 7 rows per slot (`MAX_PROP_PROBES` = 8192 slots). Only this
    // shader declares this tail; the other shaders bind the same buffer by its prefix.
    prop_probes: array<vec4<f32>, 57344>,
    // Per rig slot (sizes mirrored in rig_palette.rs): the base bone index into `palettes`, whose
    // 3 rows per bone are `rig_from_joint × inverse_bindpose` from the rig's own origin.
    rig_table: array<u32, 2048>,
    // Per rig slot: the CM2 body tint (`model+0x184/188/18c`) packed `0xFFRRGGBB` like the
    // reference's node value (`0x60d840`: `param | 0xff000000`); 0 is identity.
    rig_tint: array<u32, 2048>,
    // Per rig slot: the world origin its palette rows are measured from.
    rig_origin: array<vec4<f32>, 2048>,
    // The mat-anim rows (size mirrored in mat_anim_table.rs), row 0 zero: a UV-scroll delta (xy),
    // a tint delta (xyz), a texture-transform affine `[cos − 1, sin, sx − 1, sy − 1]` or a UI cell.
    matanim: array<vec4<f32>, 2048>,
    // Per rig slot, the straddle waterline (size mirrored in straddle.rs): x = its world height
    // (Bevy Y), y = the side the near copy keeps (+1 above, −1 below, 0 not straddling).
    water_clip: array<vec2<f32>, 2048>,
    palettes: array<vec4<f32>>,
};
@group(#{MATERIAL_BIND_GROUP}) @binding(90) var<storage, read> wow_light: WowLight;

// MONKEY (torch shadows Phase 3A): the ENTITY receiver's torch bindings — the SAME depth array
// static_gx's group 3 samples (one shared `Image`, rendered by `static_gx::torch_depth`) and the
// SAME 6416-byte table, riding the material's own group (`WowModelExt` bindings 91/92/93) because
// a Bevy material draw sets groups 0/1/2 only. ALWAYS bound (the image and buffer exist from
// startup), so no shader-def guards this block; only the fragment stage reads it. The struct is
// std430 here and std140 in static_gx — identical bytes, every member is 16-aligned.
struct TorchTable {
    // MONKEY (static torch cache): byte-identical in BOTH shaders and TorchTableUniform.
    // count@0 (16): x high-water slot count, y soft*100, z dynamic/live-bank mask, w reserved.
    // positions@16 (256), view_projs@272 (6144): total 6416 bytes. A mismatch hides buildings.
    // MONKEY (live bank rank): 96 static layers + 48 live layers; matrices stay slot-addressed.
    count: vec4<u32>,
    positions: array<vec4<f32>, 16>,
    view_projs: array<mat4x4<f32>, 96>,
}
@group(#{MATERIAL_BIND_GROUP}) @binding(91) var torch_depth: texture_depth_2d_array;
@group(#{MATERIAL_BIND_GROUP}) @binding(92) var torch_samp: sampler_comparison;
@group(#{MATERIAL_BIND_GROUP}) @binding(93) var<storage, read> torch_table: TorchTable;
// Reverse-Z receiver bias — KEEP IN SYNC with static_gx.wgsl's TORCH_BIAS (0.001, the known-good
// value; larger detaches every shadow far from the torch).
const TORCH_BIAS: f32 = 0.001;
// NORMAL-OFFSET for entity receivers (yd): sample the map a hand's width OUTSIDE the surface along
// its normal. An entity's caster is the CPU-skinned copy of the same body; the GPU-skinned receiver
// differs by tiny pose/interpolation deltas, so with a plain depth compare every fragment sits a
// hair BEHIND its own caster and the body shadows itself everywhere (a solid-black NPC in
// interiorDebug 2, flickering shadows in the real render). Pushing the sample point out of the skin
// makes self-occlusion impossible while a pillar or another character still casts onto it.
const TORCH_NORMAL_OFFSET: f32 = 0.15;
// MONKEY (torch caster selection, MIRRORED from static_gx.wgsl): the live PCF tap-radius scale
// (count.y LOW half) and MONKEY (shadow floor)'s strength (HIGH half),
// unpacked from the table's `count.y` (stored x100 — the row is `vec4<u32>`).
fn torch_soft() -> f32 {
    return max(f32(torch_table.count.y & 0xffffu) * 0.01, 0.05);
}
// MONKEY (shadow floor): the live SHADOW STRENGTH (`torchShadowStrength`, 0..1, default 0.7),
// unpacked from `count.y`'s HIGH half (`(strength x 100) << 16 | soft x 100` - see
// `TorchTableUniform::pack`; the low half is `torch_soft` above and `count.w`'s flags are
// untouched). It is applied as `mix(1, s, w * strength)` where the cross-fade weight already
// multiplies the shadow factor, which is algebraically the same thing as flooring the factor
// itself (`1 - w*strength*(1-s)` either way) for one extra multiply and no extra tap.
//
// WHY a floor at all: a torch map is the ONLY occlusion in the direct term, so a blocked fragment
// used to drop that term to exactly zero - a pitch-black, razor-edged scar of tent canvas across
// Darkmoon's grass, table legs printed on the Darkshire inn floor. Nothing in this renderer
// bounces, so the 30 % that survives at the default IS the bounce: the fill/ambient arms are
// untouched (they never saw this factor), only the DIRECT arm is floored. 1 restores the shipped
// pitch-black look exactly, 0 disables torch shadows without disturbing the lane behind them.
fn torch_strength() -> f32 {
    return clamp(f32(torch_table.count.y >> 16u) * 0.01, 0.0, 1.0);
}
// MONKEY (outdoor torch shadows, MIRRORED from static_gx.wgsl — keep in sync): the EXTERIOR lane's
// world-distance fade radius (yd) and the CPU's one-bit lane gate in `count.w`. See the static_gx
// copies for the full rationale (the interior `ndc.z` fade is already down to 5 % at 10 yd, which
// erases a campfire's 15-25 yd pool).
const TORCH_EXT_FADE_YD: f32 = 44.0;
const TORCH_EXT_LANE: u32 = 1u;
// MONKEY (torch lane perf): below THIS much unshadowed direct contribution a fragment skips the
// torch table scan and its four comparison taps entirely, because the shadow factor is multiplied
// into that contribution and can therefore only ever subtract less than this from the frame.
//
// Sized off what the eye can see, not off "small": the interior budget goes through the
// `1 - exp(-x * exposure)` rolloff, which for a term this small is just `x * exposure`, so at the
// shipped `interiorExposure 2.5` the worst possible step across the guard's boundary is
// 2.5e-4 linear = well under one 8-bit code. A tenth of this (1e-3) would have been a visible
// ~8/255 contour along the iso-surface where the guard flips. Most of the skips it buys are exact
// zeros anyway -- `window` IS 0 past the fixture's reach and `nl` IS 0 on a surface facing away --
// so the threshold only has to be small enough to be invisible, never large enough to be useful.
const TORCH_SKIP_EPS: f32 = 1e-4;
fn torch_ext_on() -> bool {
    return (torch_table.count.w & TORCH_EXT_LANE) != 0u;
}

// MONKEY (Phase 5, MIRRORED from static_gx.wgsl — keep in sync): the cube face that contains
// direction `d` (fixture → fragment) — the major axis, signed. Face order is the contract with
// `benilla_app::torch_shadow::cube_view_projs`: 0 +X, 1 −X, 2 +Y, 3 −Y, 4 +Z, 5 −Z.
fn torch_face(d: vec3<f32>) -> u32 {
    let a = abs(d);
    if (a.x >= a.y && a.x >= a.z) {
        return select(1u, 0u, d.x > 0.0);
    }
    if (a.y >= a.z) {
        return select(3u, 2u, d.y > 0.0);
    }
    return select(5u, 4u, d.z > 0.0);
}

// MONKEY (torch shadows Phase 3A): this interior fixture's OWN cast shadow on an ENTITY fragment —
// exactly static_gx.wgsl's `torch_surface_shadow` (keep in sync): correlate the `wow_light`
// fixture at `light_pos` to a promoted torch (position match within 1 yd), pick the cube face
// facing the fragment, and sample that layer through the shared projector. 1.0 (unshadowed) when
// no map matches — the lane is off, or the fixture was not among the ≤4 promoted.
//
// MONKEY (outdoor torch shadows): `fade_radius` picks the far fade and is the ONLY lane difference
// (`0` = the interior/legacy `ndc.z` fade, bit-for-bit; `> 0` = the exterior world-distance fade
// measured from the fixture). MIRRORS static_gx.wgsl's `torch_map_at`.
fn torch_entity_shadow_at(light_pos: vec3<f32>, P: vec3<f32>, N: vec3<f32>, fade_radius: f32) -> f32 {
    // MONKEY (torch lane perf): MIRRORS static_gx.wgsl's `torch_map_at` — the empty table is the
    // common case, so answer it before the normal offset and the loop set-up.
    if (torch_table.count.x == 0u) {
        return 1.0;
    }
    // Normal-offset the sample point out of the body (see TORCH_NORMAL_OFFSET).
    let Ps = P + N * TORCH_NORMAL_OFFSET;
    for (var i = 0u; i < torch_table.count.x; i = i + 1u) {
        // MONKEY (static torch cache): holes and pending uploads never sample stale layers.
        if (torch_table.positions[i].w <= 0.0) { continue; }
        let fixture = torch_table.positions[i].xyz;
        if (distance(fixture, light_pos) < 1.0) {
            let face = torch_face(Ps - fixture);
            let layer = i * 6u + face;
            // MONKEY (live bank rank): count.z is CPU-ready-filtered; holes must not consume
            // live cubes. Keep the projection on the static slot while compacting depth only.
            let rank = countOneBits(torch_table.count.z & ((1u << i) - 1u));
            let depth_layer = select(layer, 96u + 6u * rank + face, (torch_table.count.z & (1u << i)) != 0u);
            // MONKEY (outdoor torch shadows): negative = the interior contract (let the projector
            // use its own reverse-Z far fade); otherwise this lane's world-distance weight.
            let fade = select(
                -1.0,
                1.0 - smoothstep(0.8 * fade_radius, fade_radius, distance(fixture, Ps)),
                fade_radius > 0.0,
            );
            // MONKEY (slope bias): the entity's own normal - the one `Ps` was offset along.
            let s = shadow_hook::torch_map_shadow(
                torch_table.view_projs[layer], i32(depth_layer), Ps, N, torch_depth, torch_samp,
                TORCH_BIAS, torch_soft(), fade);
            // MONKEY (torch caster selection, MIRRORED from static_gx.wgsl): `.w` is the slot's
            // FADE WEIGHT, so a promoted fixture's shadow ramps in over ~1/3 s and a demoted one
            // ramps out. An entity and the floor under it MUST use the same weight or the NPC's
            // shadow would pop while the floor's faded.
            // MONKEY (shadow floor): and `torch_strength()` is the DIRECT-term floor folded
            // into that same weight (see the function). `w * strength` rather than a second `mix`
            // because the two are the same expression.
            return mix(1.0, s, torch_table.positions[i].w * torch_strength());
        }
    }
    return 1.0;
}

// The INTERIOR lane's call - unchanged behaviour (`fade_radius 0` => the reverse-Z `ndc.z` fade).
fn torch_entity_shadow(light_pos: vec3<f32>, P: vec3<f32>, N: vec3<f32>) -> f32 {
    return torch_entity_shadow_at(light_pos, P, N, 0.0);
}

// MONKEY (outdoor torch shadows): the EXTERIOR lane's call - a campfire/brazier/lamppost throwing
// the player, an NPC or a doodad's shadow across the ground and the wall behind them.
fn torch_entity_exterior_shadow(light_pos: vec3<f32>, P: vec3<f32>, N: vec3<f32>) -> f32 {
    return torch_entity_shadow_at(light_pos, P, N, TORCH_EXT_FADE_YD);
}

// MONKEY (torch debug, interiorDebug 2 on ENTITIES): the MIN raw cube-map shadow factor over every
// promoted fixture (ignoring the fixture match) — mirrors static_gx's `torch_debug_factor`, so an
// entity's sampling can be SEEN as greyscale: all-white = the table is empty / the projection misses;
// shaped dark = the depth map reaches the entity and any fault is downstream of sampling.
fn torch_entity_debug_factor(P: vec3<f32>, N: vec3<f32>) -> f32 {
    let Ps = P + N * TORCH_NORMAL_OFFSET;
    var s = 1.0;
    for (var i = 0u; i < torch_table.count.x; i = i + 1u) {
        if (torch_table.positions[i].w <= 0.0) { continue; }
        let face = torch_face(Ps - torch_table.positions[i].xyz);
        let layer = i * 6u + face;
        // MONKEY (live bank rank): debug samples the same compact bank as the lit path.
        let rank = countOneBits(torch_table.count.z & ((1u << i) - 1u));
        let depth_layer = select(layer, 96u + 6u * rank + face, (torch_table.count.z & (1u << i)) != 0u);
        // MONKEY (slope bias): same plane as the lit path, so `interiorDebug 2` stays honest.
        let raw = shadow_hook::torch_map_shadow(
            torch_table.view_projs[layer], i32(depth_layer), Ps, N, torch_depth, torch_samp,
            TORCH_BIAS, torch_soft(), -1.0);
        // MONKEY (torch caster selection): the WEIGHTED factor, matching the real render.
        s = min(s, mix(1.0, raw, torch_table.positions[i].w));
    }
    return s;
}

// Vanilla M2 cutout alpha-test reference (224/255 on ≤ WotLK) — kept in sync with
// `debug_panel::VANILLA_ALPHA_KEY_REF`. Used to re-apply the hard cutout on the distance-fade blend
// twin so its silhouette matches the steady cutout exactly.
const VANILLA_ALPHA_KEY: f32 = 0.8784314;

// The detail-doodad pass's stage-0 `D3DSAMP_MIPMAPLODBIAS = +0.25` (`0x6813f4`).
const DETAIL_DOODAD_LOD_BIAS: f32 = 0.25;

// Bevy's `VertexOutput` (same fields, locations and defs; no tangents, morphs or visibility
// ranges) plus our interpolants; the fragment rebuilds a `VertexOutput` from it.
struct WowVsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
#ifdef VERTEX_UVS_A
    @location(2) uv: vec2<f32>,
#endif
#ifdef VERTEX_UVS_B
    @location(3) uv_b: vec2<f32>,
#endif
#ifdef VERTEX_COLORS
    @location(5) color: vec4<f32>,
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    @location(6) @interpolate(flat) instance_index: u32,
#endif
    // The point-light sum `Σ att·sat(N·L)·colour`, per vertex like the reference FFP.
    @location(8) point_lit: vec3<f32>,
#ifdef WOW_MERGED_FADE
    // A merged blob's per-placement fade alpha, constant over the placement.
    @location(9) merged_fade: f32,
#endif
#ifdef WOW_MERGED_SLOT
    @location(10) @interpolate(flat) merged_slot: u32,
#endif
    // MONKEY (outdoor torch shadows; ext light k8): WHICH <=`EXT_SEL_K` exterior table entries
    // `point_lit` was summed from, packed 8 bits each across FOUR u32s (see `EXT_SEL_NONE` /
    // `ext_sel_get`). FLAT, because it is a choice and not a quantity - interpolating packed
    // indices would produce a different, meaningless one. `EXT_SEL_NONE` on every lane that takes
    // no exterior point term (interior WMO surfaces, interior props), which makes the night lane a
    // no-op there by construction. Widened from ONE u32 of three 10-bit ranks to TWO u32s of eight
    // 8-bit ranks and then to FOUR u32s of twelve - three extra interstage components in all (25
    // for this struct, against a 60-component limit) - see `EXT_SEL_K`.
    @location(11) @interpolate(flat) ext_sel: vec4<u32>,
}

// The dynamic point-light term at a world-space point (decisions 0016/0273/0278, selection 0285) —
// the reference commits AT MOST THREE point lights per draw, the nearest to the RECEIVING UNIT'S OWN
// position (byte law: the gather `0x71bf90` keeps the nearest by squared distance from the
// caller-supplied unit position — never the camera, never the vertex — and the commit `0x71c730`
// seats slots 1-3, dropping the 4th; wow-re `wmo-surface-dynamic-light` §4/§6). Summing the whole
// table instead lit every candelabra pole in the abbey from a dozen sideways fixtures the real
// client never commits for it — the director's "stands must not light up with the point gain".
//
// So: pass 1 picks the ≤`EXT_SEL_K` nearest table entries to `anchor` (the unit position, or its
// AABB where it has one — a light's packed
// range bounds candidacy); pass 2 evaluates ONLY those at the vertex — the byte-verified falloff
// `1/(0.7·d + 0.03·d²)`, diffuse-only (committed ambient/specular are zero), on the SUBMITTED normal
// (the FFP never enables GL_LIGHT_MODEL_TWO_SIDE — no per-face flip). A selected light reaches every
// vertex of its unit with no distance cutoff, exactly like a committed GL light — selection pops at
// unit granularity (the authored vanilla behaviour), never mid-surface. Mirrored in terrain.wgsl.
// **Zero-normal-safe normalize — the M2 corpus authors `(0,0,0)` vertex normals and the reference
// draws them lit** (decision 1268). `Creature\QuirajProphet` (the AQ40 Qiraji Brainwasher)
// authors them on 28 of its sleeve batch's 40 vertices; `Creature\TitanFemale` (Uldaman's Ironaya)
// on 82. A plain `normalize()` turns that legal, shipped datum into NaN, and NaN poisons the whole
// lighting chain: every SH dot is NaN, `clamp(NaN, 0, 1)` floors to 0, and the batch renders PURE
// BLACK over its correct texture — the reported symptom, which only "came back" while the unit was
// targeted because the highlight (the scene ambient, added) rides OUTSIDE the poisoned factor.
//
// The reference lands on the zero vector instead: its `Model2.bls` lit permutations normalize with
// the era's `RSQ`/`MUL` pair (wow-re `models/scratch/model2-bls-vertex-sh.md` §2), where the
// `0 × INF` product resolves to 0, so the order-2 SH quadratic form is evaluated at `N = 0` and
// collapses to its **DC term** — the flat, direction-independent ambient the reference's own
// screenshots show on exactly these surfaces. Passing the zero vector through reproduces that
// through every lane here: the SH lobe keeps `c10.w + sun_dc`, `lit_nl` keeps its ambient, and
// `point_light_sum`'s `max(N·L, 0)` is 0.
// Guarded around the SAME `normalize` rather than an open-coded `v · inverseSqrt(l2)`: every
// non-degenerate normal — i.e. the whole corpus outside these few batches — keeps its existing bits,
// so the visual baselines cannot drift by a rounding step on account of this fix.
fn wow_normalize(v: vec3<f32>) -> vec3<f32> {
    let l2 = dot(v, v);
    return select(vec3<f32>(0.0), normalize(v), l2 > 1e-12);
}

// MONKEY (outdoor torch shadows, MIRRORED from static_gx.wgsl — keep in sync): the
// ≤`EXT_SEL_K`-nearest EXTERIOR selection packed into FOUR u32s, twelve 8-bit indices (rank 0 in the
// low byte of `.x`) with `EXT_SEL_EMPTY` for an unfilled rank. It exists so the per-FRAGMENT
// shadowed term below can re-evaluate the VERTEX stage's choice rather than making its own —
// re-ranking per fragment would draw a hard line wherever the ranking flips, which is precisely
// what the Gouraud term never does.
//
// MONKEY (ext light k12) - **WHY K WENT 3 -> 8.** Three is the reference FFP's own commit limit (GL
// slots 1-3; wow-re `wmo-surface-dynamic-light` sections 4/6) and it was the right number for the
// reference's SPARSE, hand-authored light set: with two or three authored lights in a village the
// nearest three IS all of them, and no two draw units can disagree. benilla does not have that set
// - `fire_light.rs` SYNTHESISES a point light for every torch, lantern, brazier and campfire
// GameObject/doodad in range, so a lamp-lit set piece now puts 15+ exterior fixtures inside ONE
// chunk's candidacy box (measured at the Darkmoon Faire: 12 `Free Standing Torch 01` + 3
// `General Lantern 01` within reach of the player's chunk, plus a stall lamp and the fireworks'
// spell lights). Once the candidates outnumber the slots, adjacent draw units keep DIFFERENT
// threes; and because the boundary between draw units is a straight line - an MCNK cell edge, or
// the jump from a cell-anchored chunk to the origin-anchored bench standing on it - the
// disagreement reads as a HARD STRAIGHT EDGE across a torch's pool of light (the director's
// "scars"), and as a bench lit by a firework over ground that is not. More slots make neighbouring
// units AGREE, so a pool ends where the FALLOFF ends instead of where the cell does; the only open
// question is how many is enough, and the next paragraph answers it with the measurement.
//
// 8 bits per rank caps the LIVE table at 255 real entries, `EXT_SEL_EMPTY` = 255 being the
// sentinel; `global_light::MAX_LIVE_POINT_LIGHTS` enforces that CPU-side. The buffer still carries
// 256 slots, so `LightStd430` keeps its 8528 B and no mirror struct moves.
//
// MONKEY (ext light k12) - **AND WHY 8 -> 12, one round later.** Eight was MARGINAL the day it
// landed and the note here said so. The Darkmoon Faire's densest chunk holds THIRTEEN candidates
// and EIGHT of them stand strictly inside its own 33.33 yd cell, so a torch 2.5 yd outside the edge
// - lighting the grass right at the seam - cannot win a slot under any ranking, and that chunk goes
// on disagreeing with its neighbour about it. On that particular seam K=8 did not help, it made
// things WORSE (0.198 -> 0.356 of a falloff unit): raising K let the SPARSE neighbour take in
// lights the dense chunk still had no room for, so the two sets grew further apart, not closer.
// Twelve is the first K at which every chunk around the player keeps everything it can see bar one:
//
//   worst ABSOLUTE seam jump, the four chunks meeting at the corner nearest WoW (-9557, 99)
//   (`k12_table.py` in the scratchpad; falloff units, colour and N.L factored out, the byte-verified
//   `1/(0.7d + 0.03d^2)`, 200 samples along each shared edge)
//     seam                     cands      K=3      K=8     K=12     K=16
//     (-288,2)|(-288,3)         6 | 2   0.0418   0.1117   0.1117   0.1117
//     (-288,2)|(-287,2)        6 | 13   0.6425   0.0882   0.0669   0.1281
//     (-288,3)|(-287,3)         2 | 7   0.2888   0.0862   0.0862   0.0862
//     (-287,2)|(-287,3)        13 | 7   0.1980   0.3556   0.1144   0.1406
//     WORST                            0.6425   0.3556   0.1144   0.1406
//
// Read the last two columns together. At K>=13 every chunk keeps ALL its candidates, so 0.1406 is
// the CANDIDACY BOX's own floor - the residual disagreement two chunks have because their gather
// boxes are centred 33 yd apart - and no K can touch it (widening the box is a separate change, and
// it is the byte-verified one). K=12 measures 0.1144, slightly UNDER that floor, because the dense
// chunk drops exactly one far candidate its neighbour also barely weights. So twelve ENDS this line
// of attack: 16 is not better, only wider.
//
// The own-origin case in the same table - a bench standing on the ground, ranking from its own
// position, against the chunk under it - goes 79 % / 71 % / 52 % apart at K=3, to 31 % / 27 % / 5 %
// at K=8, to 3 % / 1 % / 5 % at K=12. That is the "bench lit by a firework over dark grass" closing.
//
// RAISING K FURTHER: `ext_sel_get` and the pack tail index the selection vector dynamically
// (`sel[s >> 2u]`) and the rank arrays are zero-constructed and filled, so K is not spelled out
// anywhere but here. The vector is `vec4<u32>` now and holds SIXTEEN rank slots, so K = 13..16 is
// this constant ALONE - no type site, no interstage field, no `EXT_SEL_NONE` edit. Past 16 the next
// step is a second `vec4<u32>` (four more interstage components) and `ext_sel_get` learning which
// vector to read, which is a real change rather than a constant.
const EXT_SEL_K: u32 = 12u;
const EXT_SEL_EMPTY: u32 = 255u;
// Twelve empty ranks. MONKEY (ext light k12): FOUR words - the vector holds sixteen 8-bit rank
// slots and only the first `EXT_SEL_K` are ever read, so filling all four with the sentinel costs
// nothing and leaves no word that could be mistaken for a real index if K is raised again. (The
// PACK tail below builds its `vec4<u32>` from ZERO and writes only ranks 0..K-1, so its unused high
// word reads as index 0 rather than the sentinel - harmless, because no loop here ever looks past
// `EXT_SEL_K`.)
const EXT_SEL_NONE: vec4<u32> = vec4<u32>(4294967295u, 4294967295u, 4294967295u, 4294967295u);

// MONKEY (ext light k12): how many of the `EXT_SEL_K` ranks pay for a CUBE-MAP OCCLUSION lookup in
// the night lane. The ranking is by distance, so ranks 0..2 are the three fixtures whose term
// dominates this fragment; ranks 3..11 are the long tail that fixes the SELECTION (the scars) and
// contribute a soft, low-amplitude wash where a hard-edged shadow would not be legible anyway.
// Holding the shadowed count at the OLD K pins the per-fragment cost of the night lane at exactly
// what it was - three table scans and their taps - while the selection itself gets twelve deep.
const EXT_SEL_SHADOWED: u32 = 3u;

// Unpack rank `s` (0..`EXT_SEL_K`-1) from the four-word selection. MIRRORED - keep in sync.
fn ext_sel_get(sel: vec4<u32>, s: u32) -> u32 {
    return (sel[s >> 2u] >> (8u * (s & 3u))) & 255u;
}

// The ranking half of `point_light_sum` (same tests, same order, same ties — strictly-less
// inserts, so an equal distance leaves the earlier table index at the better rank).
//
// MONKEY (ext light k12): `box` is the draw unit's HORIZONTAL half-extent in yards, and it changes
// what "nearest" MEANS - ranking is by the distance from the light to the unit's AABB, not to the
// unit's anchor POINT. A cell-anchored unit (terrain's 33.33 yd MCNK chunk, clutter, an
// exterior-class WMO street) passes `MCNK_CELL_HALF`: a torch standing 2 yd outside a cell's edge
// is 2 yd from the nearest surface that cell draws, yet ~18 yd from its CENTRE, which is how it
// used to lose its slot to three torches clustered near the middle while lighting nothing of the
// ground right under it. Clamping the light into the box (`max(|d| - box, 0)` per horizontal axis)
// ranks it the way the surface actually sees it, and - the point of the exercise - makes two
// ADJACENT cells rank a light on their shared edge almost identically, so their sets agree there.
// The VERTICAL stays unbounded, exactly as the reference's hash sweep is. `box = 0` reproduces the
// old point-anchored ranking BIT-FOR-BIT, and that is what the own-origin units (props, entities)
// pass, so nothing on those lanes moves.
//
// CANDIDACY deliberately stays on the anchor POINT (`dc2` below): it is the byte-verified gather
// test, and widening it is a different question from how the survivors are ordered.
fn point_light_pick(anchor: vec3<f32>, box: f32) -> vec4<u32> {
    let count = u32(wow_light.point_count.x);
    var sel = array<u32, EXT_SEL_K>();
    var sd = array<f32, EXT_SEL_K>();
    for (var s = 0u; s < EXT_SEL_K; s = s + 1u) {
        sd[s] = 1e30;
    }
    for (var i = 0u; i < count; i = i + 1u) {
        // MONKEY (light lanes): skip INTERIOR fixtures (colour row `.w > 0.5`) — they light the
        // room that claims them and nothing else. Before the split, a proximity-admitted inn
        // fixture reached every exterior receiver near the building through its own walls. Skipped
        // BEFORE the ≤`EXT_SEL_K` ranking, so it cannot take a slot an outdoor fire should have
        // had.
        if (wow_light.points[2u * i + 1u].w > 0.5) {
            continue;
        }
        let pos_range = wow_light.points[2u * i];
        let dv = pos_range.xyz - anchor;
        let dc2 = dot(dv, dv);
        if (dc2 > pos_range.w * pos_range.w) {
            continue;
        }
        // MONKEY (ext light k12): rank by the distance to the draw unit's BOX (see the header
        // note). With `box = 0` both `max`es are identities and this is exactly the old `dc2`.
        let e = max(abs(dv.xz) - vec2<f32>(box), vec2<f32>(0.0));
        let d2 = dot(e, e) + dv.y * dv.y;
        // MONKEY (ext light k12): an `EXT_SEL_K`-deep insertion in place of the hand-unrolled
        // 3-deep cascade. Both loops are bounded by a module const, so the compiler unrolls them;
        // the comparison is strictly-less, which keeps the old tie order (first-found wins).
        //
        // The guard on the WORST kept rank first: `sd` is sorted, so a candidate that cannot beat
        // `sd[K-1]` cannot beat anything, and the scan below would walk all eight slots only to
        // decide that. It makes the REJECT path - which is what nearly every table entry takes once
        // the set is full - ONE comparison, i.e. cheaper than the three the old cascade spent, so
        // widening K did not make the common case more expensive. Exactly equivalent to letting the
        // scan run: `d2 >= sd[K-1]` is precisely the condition under which it returns `r = K`.
        if (d2 >= sd[EXT_SEL_K - 1u]) {
            continue;
        }
        var r = EXT_SEL_K;
        for (var s = 0u; s < EXT_SEL_K; s = s + 1u) {
            if (d2 < sd[s]) {
                r = s;
                break;
            }
        }
        if (r < EXT_SEL_K) {
            for (var s = EXT_SEL_K - 1u; s > r; s = s - 1u) {
                sd[s] = sd[s - 1u];
                sel[s] = sel[s - 1u];
            }
            sd[r] = d2;
            sel[r] = i;
        }
    }
    // Pack low-byte-first, `EXT_SEL_EMPTY` for a rank nothing reached. No real index can collide
    // with the sentinel: the live table caps at 255 (`MAX_LIVE_POINT_LIGHTS`).
    var packed = vec4<u32>(0u, 0u, 0u, 0u);
    for (var s = 0u; s < EXT_SEL_K; s = s + 1u) {
        let idx = select(EXT_SEL_EMPTY, sel[s], sd[s] <= 9.9e29);
        packed[s >> 2u] |= idx << (8u * (s & 3u));
    }
    return packed;
}

// The evaluation half — the byte-verified falloff `1/(0.7d + 0.03d²)` × `max(N·L, 0)` × the
// committed colour, in rank order, stopping at the first empty rank exactly as the old
// `sd[s] > 9.9e29` break did. Shared by both pickers (their sum loops were already identical).
// MONKEY (ext light k12): up to `EXT_SEL_K` terms now, still Gouraud (per vertex) and still linear
// in the falloff, so nothing about the day lane's FORM changed — a unit simply stops dropping the
// fixtures its neighbour kept.
fn point_light_eval(sel: vec4<u32>, P: vec3<f32>, N: vec3<f32>) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    for (var s = 0u; s < EXT_SEL_K; s = s + 1u) {
        let idx = ext_sel_get(sel, s);
        if (idx == EXT_SEL_EMPTY) {
            break;
        }
        let to_light = wow_light.points[2u * idx].xyz - P;
        let d = length(to_light);
        let atten = 1.0 / (0.7 * d + 0.03 * d * d);
        let nl = max(dot(N, to_light / max(d, 1e-4)), 0.0);
        sum += wow_light.points[2u * idx + 1u].rgb * (atten * nl);
    }
    return sum;
}

// MONKEY (outdoor torch shadows): the SHADOWED evaluation — the same entries the vertex picked,
// each multiplied by its OWN fixture's cube-map occlusion (normal-offset, like every entity sample
// on this lane), so a fence between the player and campfire A darkens A's term while lamppost B's
// is untouched. Reached only under `torch_ext_on()`, so nothing here runs in daylight or with
// `exteriorShadows 0`.
//
// MONKEY (ext light k12): the SUM runs to `EXT_SEL_K`, but only the first `EXT_SEL_SHADOWED` ranks
// pay for an occlusion lookup, so the per-fragment cost of this lane is pinned at exactly what it
// was before the widening while the selection itself got twelve deep. Ranks 3..11 are the
// distance-ordered tail — the terms that make a torch's pool agree across a draw-unit boundary —
// and they arrive unshadowed, which at their amplitude is not a look the eye can separate from a
// shadowed one.
fn point_light_eval_shadowed(sel: vec4<u32>, P: vec3<f32>, N: vec3<f32>) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    for (var s = 0u; s < EXT_SEL_K; s = s + 1u) {
        let idx = ext_sel_get(sel, s);
        if (idx == EXT_SEL_EMPTY) {
            break;
        }
        let fixture = wow_light.points[2u * idx].xyz;
        let to_light = fixture - P;
        let d = length(to_light);
        let atten = 1.0 / (0.7 * d + 0.03 * d * d);
        let nl = max(dot(N, to_light / max(d, 1e-4)), 0.0);
        // MONKEY (torch lane perf, MIRRORED from static_gx.wgsl): the occlusion is a factor on a
        // term that is already zero on any surface facing away from the fire.
        let ext_w = atten * nl;
        var occ = 1.0;
        if (s < EXT_SEL_SHADOWED && ext_w > TORCH_SKIP_EPS) {
            occ = torch_entity_exterior_shadow(fixture, P, N);
        }
        sum += wow_light.points[2u * idx + 1u].rgb * (ext_w * occ);
    }
    return sum;
}

fn point_light_sum(P: vec3<f32>, N: vec3<f32>, anchor: vec3<f32>) -> vec3<f32> {
    return point_light_eval(point_light_pick(anchor, 0.0), P, N);
}

// MONKEY (dynamic interiors): the per-FRAGMENT room light — MIRRORED from `static_gx.wgsl`
// (`interior_room_light`; keep in sync). MONKEY (light lanes / interior attenuation): the loop is
// over the INTERIOR half of the table only (colour row `.w > 0.5`), and that same `.w` is each
// fixture's REACH in yards — its authored MOLT attenuation end (M2 sources bucket by intensity),
// scaled live by `interiorAttenScale`; it bounds the loop, shapes the fill and drives the direct
// window. `interiorAttenScale 0` packs the legacy 48 yd — under the soft profile below that is the
// widest, flattest pool the lane can make, the nearest thing left to the pre-window flat lane (no
// longer the byte-exact restore it was, because the window's SHAPE moved with it).
// Every in-range light of the room-gated table, no
// nearest-K selection at all: DIRECT (falloff × wrapped Lambert, so a floor-level hearth still lights
// the floor) + FILL (normal-free, half-desaturated bounce) + the base ambient. The knobs are the
// live cvars packed into `point_count.yzw` (`.y` ambient, `.z` fill gain, `.w` exposure — the
// caller's multiplier before the rolloff). Here it lights INDOOR units and GameObjects (gated on the
// camera-independent probe slot) so they match the fixture-lit room around them instead of the
// day/night CGLight, and takes the SAME per-fixture torch shadows (Phase 3A: the shared depth
// array through this material's own group-2 bindings) as the surfaces.
const INTERIOR_WRAP: f32 = 0.5;

// MONKEY (bake floor): the ENTITY half of `static_gx.wgsl`'s interior bake floor. Keep
// `BAKE_LANE_SCALE` byte-identical with that file and with
// `benilla_world::lighting::BAKE_LANE_SCALE`: the value rides the FRACTION of `sh_c16.w` (whose
// integer part is the world-shadow flag), with `interiorGain` already folded in by the packer.
const BAKE_LANE_SCALE: f32 = 0.49;
// The surfaces get `vc.rgb * k` — each batch's OWN authored MOCV. An entity has no MOCV: it is an
// M2 standing in the room, not a piece of the room's mesh, and nothing on this lane knows which
// WMO group it is in (the entity call passes `room_inst`/`room_group` as 0 — the ungated stub —
// and the claim binding is `static_gx`-only, group 3). Computing a per-group mean MOCV luminance
// at spawn and threading it onto the entity's record would be the better answer, but there is no
// entity->group record on this path to put it in; inventing one is a spawn-side plumbing change of
// its own. So the entity lane uses a FLAT MEAN — deliberately, and this is the note that says so.
//
// 0.45 is MEASURED, not guessed: over the Lion's Pride Inn's 70 INT+TRANS batches
// (`benilla-extract wmolights`), the per-batch mean MOCV luminances run 0.183 (the cellar `g10`)
// to 0.692 (the hearth floor under L9), median 0.433, quartiles 0.310/0.522 — and the
// TRIANGLE-WEIGHTED mean, i.e. what an entity is actually likely to be standing among, is
// **0.451**. So an entity is lifted by about what the floor under it is, which is the whole
// requirement: a chair in the inn’s fixture-starved vestibule must not be a black cut-out against
// the wall behind it. At the owner’s cvars and `interiorBakeFloor 0.12` that chair renders
// 0.083 x its texture where the wall beside it renders 0.108 — near enough that it reads as part
// of the room, which is all a flat mean can promise.
//
// ZERO on the exterior lane by construction — this term is only reachable inside the
// `is_interior || matte_indoor` branch below, which is also gated on `wmo_fog_params.w > 0.5`.
const INTERIOR_BAKE_ENTITY_MEAN: f32 = 0.45;
// MONKEY (soft falloff): the interior pool's PROFILE — **keep this whole block byte-identical
// between wow_model.wgsl and static_gx.wgsl.**
//
// The colour row's `.w` packs each fixture's EFFECTIVE RADIUS `R`: its authored MOLT
// `attenuation_end` (an M2 source buckets by intensity instead) already multiplied by the live
// `interiorAttenScale` (default 1.6). MONKEY (pool energy): the EXTENT terms — the direct window
// and the fill span — are still written as fractions of `R`, so the cvar stays a pure zoom on how
// far every pool REACHES; the core is not, so it is no longer also a brightness dial.
//
// The RETIRED profile was the reference's `1/(0.7d + 0.03d²)` windowed by
// `1 − smoothstep(0.65R, R, d)` at `R` = the authored reach. That window closed a still-BRIGHT
// curve — 0.28 of its 1 yd value at 0.65R — over the last 35 % of the radius, i.e. a bright disc
// with a visible rim and black beyond it. That is the abbey candelabra ring. Three changes replace
// it, and the ring cannot come back because the curve is already dim wherever the window bites:
//  · CORE — `1/(1 + (d/r0)²)`: inverse square with a soft core. MONKEY (pool energy): `r0` is a
//    CONSTANT `INTERIOR_CORE_YD` **yards**, not a fraction of `R` any more. It was `0.26·R`, and
//    that tied a fixture's BRIGHTNESS to its REACH — the pool's total ENERGY grew as `R²`. The
//    Goldshire inn's `R ≈ 11-15` fixtures therefore ran a 2.1× (at 3 yd) to 2.9× (at 5 yd) hotter
//    curve than Northshire abbey's `R ≈ 7`; ten of them overlap in one room, the
//    `1 − exp(−x·exposure)` rolloff saturated, and the inn read as one flat yellow while the
//    abbey — whose pools the owner signed off — sat exactly where it should. Reach must BOUND a
//    pool, not fuel it. So the energy is now one curve for every fixture and only the WINDOW and
//    the fill span still scale with `R`: two fixtures of different reach read IDENTICALLY until
//    the shorter one's window closes. `1.75 yd` is what the abbey already had (`0.26 × 6.7` — its
//    median authored end 4.2 at the default `interiorAttenScale` 1.6), so the under-candle level
//    the exposure was tuned against is preserved to within ±10 % across the abbey's own reach
//    spread (−5 % at its median fixture) and nothing about the approved look moves.
//  · GAIN — `INTERIOR_CORE_GAIN` normalises the core against the retired hyperbolic's 1 yd value,
//    `1/(0.7 + 0.03) = 1.3699`. With a CONSTANT `r0` that normalisation is the same for every
//    fixture instead of holding only at the median reach, which is the whole point; it lands at
//    `1.5/(1 + (1/1.75)²) = 1.1308`, i.e. 0.83 × the hyperbolic — the abbey's own current level.
//    (Gain 1.817 would restore 1.3699 exactly; that is a 21 % brighter room than the approved
//    one, so it is deliberately NOT taken. Change the GAIN, never the core radius, to re-level.)
//  · WINDOW — `(clamp01(1 − (d/R)^p))²`, the UE4-style windowing: value AND derivative both vanish
//    at `d = R` (C1, so no rim), ≈1 over most of the radius, and it only ever removes energy the
//    curve has already lost.
// MONKEY (pool energy): the direct term (before the wrapped Lambert) at d = 1 / 3 / 5 / 8 / 12 yd,
// abbey-class `R = 7` beside inn-class `R = 13` —
//   was   R=7  1.152 / 0.403 / 0.164 / 0     / 0       R=13  1.379 / 0.839 / 0.470 / 0.224 / 0.033
//   now   R=7  1.131 / 0.381 / 0.153 / 0     / 0       R=13  1.131 / 0.381 / 0.164 / 0.067 / 0.009
// The two "now" rows are the SAME curve until R=7's window bites (where it is already down to
// 0.15); R=13 only keeps a longer, dimmer TAIL, because its window is wider. That is the fix.
//
// FILL keeps its EXACT form (`(1 − d/r)²`, which is `interior_window(d, r, 1.0)`) and only widens:
// its radius is `INTERIOR_FILL_SPAN·R`, so the floor BETWEEN two pools takes a gentle wash instead
// of the bare ambient floor. Its value at the fixture is still 1, so `interiorFill` keeps the
// meaning it was tuned with.
const INTERIOR_CORE_YD: f32 = 1.75;
const INTERIOR_CORE_GAIN: f32 = 1.5;
const INTERIOR_DIRECT_POW: f32 = 10.0;
const INTERIOR_FILL_SPAN: f32 = 1.5;
const INTERIOR_FILL_POW: f32 = 1.0;
// `(clamp01(1 − (d/r)^p))²`. The inner clamp keeps `pow` off a negative base; `max(r, …)` keeps it
// off a zero divisor (the packer floors R at 1.0, but a shader must not lean on a producer's
// invariant). `clamp(…, 0, 1)` rather than `saturate` — nothing else in this shader set uses
// `saturate`, and the two are the same instruction.
fn interior_window(d: f32, r: f32, p: f32) -> f32 {
    let w = clamp(1.0 - pow(clamp(d / max(r, 1e-4), 0.0, 1.0), p), 0.0, 1.0);
    return w * w;
}
// MONKEY (room gate): the ENTITY/prop copy has no claim-table binding and no per-fragment room
// key (`static_gx.wgsl` owns both — the claim table hangs off its own group 2), so every fixture is
// admitted here, exactly as before this change. Follow-up: an entity already knows the room it
// STANDS in (the interior classifier's down-ray, `carried_light`'s `WmoGroupVis::single`), so the
// same gate could ride a per-instance key pushed through the model material.
fn interior_room_admits(i: u32, room_inst: u32, room_group: u32) -> bool {
    return true;
}
fn interior_room_light(P: vec3<f32>, N: vec3<f32>, room_inst: u32, room_group: u32) -> vec3<f32> {
    let count = u32(wow_light.point_count.x);
    let k_fill = wow_light.point_count.z;
    var direct = vec3<f32>(0.0);
    var fill = vec3<f32>(0.0);
    for (var i = 0u; i < count; i = i + 1u) {
        let color_lane = wow_light.points[2u * i + 1u];
        // MONKEY (light lanes): only a fixture that CLAIMS a room lights this room (keep in sync
        // with static_gx.wgsl). `.w` is 0 on every exterior source, so a campfire burning outside
        // the door stops reaching the floor inside it.
        if (color_lane.w < 0.5) {
            continue;
        }
        // MONKEY (room gate): and only a fixture that claims THIS ROOM lights this room's surfaces.
        // Before this, the whole INT half of the table lit every interior fragment of every
        // building in range and the only occlusion was the <=6 promoted cube-shadow casters — so an
        // inn's ground-floor candles lit its basement THROUGH the floor, and an upstairs corridor
        // wall glowed from the fixture in the room behind it. Tested here, before the distance
        // test, because it is the term that throws away the most: a room claims one or two of the
        // table's fixtures, not all of them.
        if (!interior_room_admits(i, room_inst, room_group)) {
            continue;
        }
        // MONKEY (soft falloff): `.w` is ALSO this fixture's EFFECTIVE RADIUS `R` in yards — the
        // authored MOLT `attenuation_end` (or the M2 intensity bucket) already multiplied by the
        // live `interiorAttenScale` at pack time. It replaces the flat 48 yd candidacy radius in
        // `pos_range.w` on this lane, which is why a 10-candle inn read as one uniform wash.
        let reach_yd = color_lane.w;
        // Candidacy is the FILL radius, not R: the wash reaches further than the direct pool (see
        // the profile block), and rejecting at R would cut it off exactly at the pool's own edge —
        // putting the rim back one term down.
        let fill_yd = INTERIOR_FILL_SPAN * reach_yd;
        let pos_range = wow_light.points[2u * i];
        let to_light = pos_range.xyz - P;
        let d2 = dot(to_light, to_light);
        if (d2 > fill_yd * fill_yd) {
            continue;
        }
        let d = sqrt(d2);
        let c = color_lane.rgb;
        // Normalised for BOTH terms (keep in sync with static_gx.wgsl): the table commits RAW
        // over-gamut colour × intensity; a hot forge would wash the direct term to white.
        let c_norm = c / max(1.0, max(c.r, max(c.g, c.b)));
        // MONKEY (soft falloff): inverse square with the authored-start soft core, normalised so
        // the 1 yd value is the retired hyperbolic's (profile block above).
        // MONKEY (pool energy): a CONSTANT core radius in yards (profile block above) — the reach
        // no longer scales the pool's brightness, only its window. `max` keeps the divide honest
        // if the constant is ever tuned toward 0.
        let r0 = max(INTERIOR_CORE_YD, 1e-3);
        let atten = INTERIOR_CORE_GAIN / (1.0 + (d / r0) * (d / r0));
        let window = interior_window(d, reach_yd, INTERIOR_DIRECT_POW);
        let nl = max(
            (dot(N, to_light / max(d, 1e-4)) + INTERIOR_WRAP) / (1.0 + INTERIOR_WRAP),
            0.0,
        );
        // Phase 3A per-fixture cast shadow (keep in sync with static_gx.wgsl's
        // `torch_surface_shadow` call): this fixture's OWN cube map, sampled through the
        // material's group-2 torch bindings — a pillar between an NPC and torch A darkens A's
        // term without touching torch B's, exactly as on the surfaces around it.
        // MONKEY (torch lane perf, MIRRORED from static_gx.wgsl): skip the table scan and its four
        // comparison taps where the direct term this occlusion multiplies is already nothing —
        // past the fixture's reach (`window` is exactly 0 there) or on a surface facing away from
        // it. An entity and the floor under it must agree about the shadow they show, so the
        // threshold and the shape of the guard are the same on both sides.
        let direct_w = atten * nl * window;
        var s = 1.0;
        if (direct_w > TORCH_SKIP_EPS) {
            s = torch_entity_shadow(pos_range.xyz, P, N);
        }
        direct += c_norm * direct_w * s;
        // MONKEY (soft falloff): the fill's profile is UNCHANGED in FORM — `(1 − d/r)²`, which is
        // exactly `interior_window(d, r, 1.0)` — and only its radius moved, from R to
        // `INTERIOR_FILL_SPAN·R`. That is the whole "gentle wash between the pools": at the fixture
        // it is still 1 (so `interiorFill` keeps its tuned meaning) and it decays to 0 at 2R with a
        // vanishing derivative, so the floor half-way between two candles is dim, not black.
        let c_fill = mix(c_norm, vec3<f32>(dot(c_norm, vec3<f32>(0.299, 0.587, 0.114))), 0.5);
        // DOMINANT fixture, not the sum (keep in sync with static_gx.wgsl): fill must not scale
        // with candle count or a dense room saturates flat. Fill is indirect, so not shadowed.
        fill = max(fill, c_fill * (k_fill * interior_window(d, fill_yd, INTERIOR_FILL_POW)));
    }
    return direct + fill + vec3<f32>(wow_light.point_count.y);
}

// The MCNK chunk cell center under a world point — the light-selection anchor for geometry merged in
// WORLD space (clutter; terrain.wgsl mirrors this): the reference draws terrain per 33.33-yd MCNK
// chunk and gathers that unit's lights like any other. Grid constants per the ADT format (chunk =
// 533.3333/16 yd, world half-extent 32 tiles); WoW x/y are Bevy −z/−x, and the grid is symmetric, so
// snapping Bevy x/z directly lands on the same cells. Height keeps the vertex's own y (the ref anchors
// at the chunk record's position; lights sit near the surface, so the horizontal snap is what matters).
fn mcnk_cell_anchor(P: vec3<f32>) -> vec3<f32> {
    let cell = 533.33333 / 16.0;
    let half = 32.0 * 533.33333;
    let ix = floor((half + P.x) / cell);
    let iz = floor((half + P.z) / cell);
    return vec3<f32>((ix + 0.5) * cell - half, P.y, (iz + 0.5) * cell - half);
}

// MONKEY (ext light k12): the MCNK cell's HORIZONTAL half-extent (yd) - the `box` a cell-anchored
// draw unit ranks by. The same grid constant as above, halved: 533.33333/16/2 = 16.666666. An
// own-origin unit passes 0 instead. MIRRORED in terrain.wgsl - keep in sync.
const MCNK_CELL_HALF: f32 = 533.33333 / 32.0;

// MONKEY (wmo exterior points): the EXTERIOR-lane point term for a WMO's OUTDOOR-class surfaces —
// MIRRORED VERBATIM from `static_gx.wgsl` (`wmo_exterior_point_sum`; that file owns the full
// rationale — keep the two bodies identical). This copy exists because the retained WMO collector
// declines a small minority of batches (env-mapped, depth-flag oddities) and they fall through to
// this entity pipeline: a Trade District street lit on one pipeline and black on the other is
// worse than either. It reuses this file's OWN `mcnk_cell_anchor` (identical constants to
// static_gx's private copy) — a WMO batch's anchor is its PLACEMENT origin, one point for the whole
// of Stormwind, so ranking from it would commit the same three lights to every street in the city;
// the 33.33 yd MCNK cell is the unit terrain ranks by, so a street and the road it runs into rank
// the SAME candidates and agree at the seam. Candidacy is terrain's Chebyshev box, not this file's
// 48 yd sphere, for that same agreement reason.
const WMO_EXT_REACH: f32 = 33.570166;
// MONKEY (outdoor torch shadows): the ranking half, split out for the same reason as
// `point_light_pick` — the fragment stage re-evaluates the VERTEX's choice, never its own.
fn wmo_exterior_pick(P: vec3<f32>) -> vec4<u32> {
    let anchor = mcnk_cell_anchor(P);
    let count = u32(wow_light.point_count.x);
    var sel = array<u32, EXT_SEL_K>();
    var sd = array<f32, EXT_SEL_K>();
    for (var s = 0u; s < EXT_SEL_K; s = s + 1u) {
        sd[s] = 1e30;
    }
    for (var i = 0u; i < count; i = i + 1u) {
        // Exterior lane only: an interior fixture's `.w` is its reach (≥ 1), an exterior source's
        // is 0. A building's own candles must not pool on the street outside its wall.
        if (wow_light.points[2u * i + 1u].w > 0.5) {
            continue;
        }
        let dv = wow_light.points[2u * i].xyz - anchor;
        // Terrain's hash sweep is horizontal and unbounded vertically; the ranking below is the
        // full 3-D distance, as `0x71bf90` does.
        if (max(abs(dv.x), abs(dv.z)) > WMO_EXT_REACH) {
            continue;
        }
        // MONKEY (ext light k12): rank by the distance to the MCNK cell's BOX, not to its centre -
        // see `point_light_pick`. This is the term that makes a street and the road it runs into
        // agree about a torch standing on the seam between them.
        let e = max(abs(dv.xz) - vec2<f32>(MCNK_CELL_HALF), vec2<f32>(0.0));
        let d2 = dot(e, e) + dv.y * dv.y;
        // MONKEY (ext light k12): an `EXT_SEL_K`-deep insertion in place of the hand-unrolled
        // 3-deep cascade. Both loops are bounded by a module const, so the compiler unrolls them;
        // the comparison is strictly-less, which keeps the old tie order (first-found wins).
        //
        // The guard on the WORST kept rank first: `sd` is sorted, so a candidate that cannot beat
        // `sd[K-1]` cannot beat anything, and the scan below would walk all eight slots only to
        // decide that. It makes the REJECT path - which is what nearly every table entry takes once
        // the set is full - ONE comparison, i.e. cheaper than the three the old cascade spent, so
        // widening K did not make the common case more expensive. Exactly equivalent to letting the
        // scan run: `d2 >= sd[K-1]` is precisely the condition under which it returns `r = K`.
        if (d2 >= sd[EXT_SEL_K - 1u]) {
            continue;
        }
        var r = EXT_SEL_K;
        for (var s = 0u; s < EXT_SEL_K; s = s + 1u) {
            if (d2 < sd[s]) {
                r = s;
                break;
            }
        }
        if (r < EXT_SEL_K) {
            for (var s = EXT_SEL_K - 1u; s > r; s = s - 1u) {
                sd[s] = sd[s - 1u];
                sel[s] = sel[s - 1u];
            }
            sd[r] = d2;
            sel[r] = i;
        }
    }
    // Pack low-byte-first, `EXT_SEL_EMPTY` for a rank nothing reached. No real index can collide
    // with the sentinel: the live table caps at 255 (`MAX_LIVE_POINT_LIGHTS`).
    var packed = vec4<u32>(0u, 0u, 0u, 0u);
    for (var s = 0u; s < EXT_SEL_K; s = s + 1u) {
        let idx = select(EXT_SEL_EMPTY, sel[s], sd[s] <= 9.9e29);
        packed[s >> 2u] |= idx << (8u * (s & 3u));
    }
    return packed;
}
fn wmo_exterior_point_sum(P: vec3<f32>, N: vec3<f32>) -> vec3<f32> {
    return point_light_eval(wmo_exterior_pick(P), P, N);
}

// The vertex input — bevy 0.18's `forward_io::Vertex` fields at bevy's shader locations (the
// VERTEX_* defs come from the base mesh-pipeline specialize, driven by what the mesh authors;
// tangents / morphs never — no model mesh has them), PLUS the owned-palette joint attributes at
// locations 10/11 under WOW_RIG_SKIN (`WowModelExt::specialize` sets the def and appends the
// attributes to the buffer layout when the mesh carries `ATTRIBUTE_WOW_JOINT_INDEX` — decision
// 0720; Bevy's `forward_io::Vertex` only declares joints under its own SKINNED path, which no
// benilla mesh triggers anymore).
#ifdef WOW_MERGED_FADE
// The doodad fade curve (`0x683f80`, in sync with `model_fade::doodad_fade_alpha`): alpha =
// 1 − (d − start)/range, d = horizontal distance − radius, over a size-bucketed band.
fn merged_fade_alpha(radius: f32, horiz_dist: f32) -> f32 {
    if (radius > 7.0) {
        return 1.0;
    }
    var start = 150.0;
    var range = 50.0;
    if (radius <= 0.5) {
        start = 40.0;
        range = 10.0;
    } else if (radius <= 2.5) {
        start = 100.0;
        range = 25.0;
    }
    let d = horiz_dist - radius;
    return clamp(1.0 - (d - start) / range, 0.0, 1.0);
}
#endif

// Bevy 0.18's `forward_io::Vertex` at Bevy's locations, plus the palette joints at 10/11 (appended
// by `WowModelExt::specialize` for a mesh with `ATTRIBUTE_WOW_JOINT_INDEX`) and the merged-blob
// attributes at 12/13.
struct WowVertex {
    @builtin(instance_index) instance_index: u32,
#ifdef VERTEX_POSITIONS
    @location(0) position: vec3<f32>,
#endif
#ifdef VERTEX_NORMALS
    @location(1) normal: vec3<f32>,
#endif
#ifdef VERTEX_UVS_A
    @location(2) uv: vec2<f32>,
#endif
#ifdef VERTEX_UVS_B
    @location(3) uv_b: vec2<f32>,
#endif
#ifdef VERTEX_COLORS
    @location(5) color: vec4<f32>,
#endif
#ifdef WOW_RIG_SKIN
    @location(10) joint_indices: vec4<u32>,
    @location(11) joint_weights: vec4<f32>,
#endif
#ifdef WOW_MERGED_FADE
    // The placement fade sphere: xyz = world centre, w = fade radius.
    @location(12) fade_sphere: vec4<f32>,
#endif
#ifdef WOW_MERGED_SLOT
    // The interior-prop SH-probe slot, in place of the per-entity MeshTag payload.
    @location(13) merged_slot: u32,
#endif
}

#ifdef WOW_RIG_SKIN
fn wow_rig_slot(instance_index: u32) -> u32 {
    return (mesh_functions::get_tag(instance_index) >> 19u) & 0x7ffu;
}

// Blends the four weighted bones' palette rows into `rig_from_local`, which replaces the mesh's
// world matrix like Bevy's `skin_model`; its translation is relative to `rig_origin[slot]`.
fn wow_skin_model(instance_index: u32, indices: vec4<u32>, weights: vec4<f32>) -> mat4x4<f32> {
    let base = wow_light.rig_table[wow_rig_slot(instance_index)];
    let b0 = 3u * (base + indices.x);
    let b1 = 3u * (base + indices.y);
    let b2 = 3u * (base + indices.z);
    let b3 = 3u * (base + indices.w);
    let r0 = weights.x * wow_light.palettes[b0]
        + weights.y * wow_light.palettes[b1]
        + weights.z * wow_light.palettes[b2]
        + weights.w * wow_light.palettes[b3];
    let r1 = weights.x * wow_light.palettes[b0 + 1u]
        + weights.y * wow_light.palettes[b1 + 1u]
        + weights.z * wow_light.palettes[b2 + 1u]
        + weights.w * wow_light.palettes[b3 + 1u];
    let r2 = weights.x * wow_light.palettes[b0 + 2u]
        + weights.y * wow_light.palettes[b1 + 2u]
        + weights.z * wow_light.palettes[b2 + 2u]
        + weights.w * wow_light.palettes[b3 + 2u];
    // r0/r1/r2 are the affine's rows; a WGSL matrix is column-major.
    return mat4x4<f32>(
        vec4<f32>(r0.x, r1.x, r2.x, 0.0),
        vec4<f32>(r0.y, r1.y, r2.y, 0.0),
        vec4<f32>(r0.z, r1.z, r2.z, 0.0),
        vec4<f32>(r0.w, r1.w, r2.w, 1.0),
    );
}

// bevy_pbr::skinning's inverse-transpose via the adjugate, verbatim.
fn inverse_transpose_3x3m(in: mat3x3<f32>) -> mat3x3<f32> {
    let x = cross(in[1], in[2]);
    let y = cross(in[2], in[0]);
    let z = cross(in[0], in[1]);
    let det = dot(in[2], z);
    return mat3x3<f32>(x / det, y / det, z / det);
}

fn wow_skin_normals(frame_from_local: mat4x4<f32>, normal: vec3<f32>) -> vec3<f32> {
    return wow_normalize(
        inverse_transpose_3x3m(mat3x3<f32>(
            frame_from_local[0].xyz,
            frame_from_local[1].xyz,
            frame_from_local[2].xyz
        )) * normal
    );
}
#endif

// Bevy 0.18's `mesh.wgsl` vertex stage plus our skinning and point lights. A `MaterialExtension`
// replaces the whole stage, so this must track Bevy's on upgrades.
@vertex
fn vertex(vertex: WowVertex) -> WowVsOut {
    var out: WowVsOut;

    let mesh_world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    // Precision: `frame_from_local` holds the orientation and a small translation, `frame_origin`
    // the ~9 k yd world position, so no f32 product mixes the two (an f32 ULP at 9 k is ~1 mm).
#ifdef WOW_RIG_SKIN
    var frame_from_local = wow_skin_model(
        vertex.instance_index,
        vertex.joint_indices,
        vertex.joint_weights
    );
    let frame_origin = wow_light.rig_origin[wow_rig_slot(vertex.instance_index)].xyz;
#else
    var frame_from_local = mesh_world_from_local;
    let frame_origin = mesh_world_from_local[3].xyz;
    frame_from_local[3] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
#endif

#ifdef VERTEX_NORMALS
#ifdef WOW_RIG_SKIN
    out.world_normal = wow_skin_normals(frame_from_local, vertex.normal);
#else
    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        vertex.normal,
        vertex.instance_index
    );
#endif
#endif

#ifdef VERTEX_POSITIONS
    // Precision: camera-relative to clip space; `clip_from_world × p_world` cancels
    // catastrophically with camera and geometry near 9 k yd. `world_position` is absolute again:
    // lighting and fog need no such precision.
    var p_cam = (frame_from_local * vec4<f32>(vertex.position, 1.0)).xyz
        + (frame_origin - view.world_position);
    // MONKEY (wind): clutter UV_B is `(height-from-base, per-tuft phase)` authored by clutter.rs;
    // UV_0 and vertex colour alpha retain their texture/cutout meanings.
#ifdef VERTEX_UVS_B
    if (m.clutter_fade.w > 0.5) {
        let world = p_cam + view.world_position;
        p_cam += wind_hook::grass_offset(
            world, vertex.uv_b.x, vertex.uv_b.y, view.world_position, wow_light.monkey
        );
    }
#endif
    // MONKEY (wind): a retained tree inside its fade band renders here with the same anchor/field
    // maths, to avoid a pop. MONKEY (wind): the leaf-batch marker is material bit 15
    // (`model_render::FOLIAGE_WIND_MARKER`), never a MeshTag bit (bit 18 aliases the lane weight
    // and the interior probe slot, so doorway units and interior props swayed).
    if ((u32(m.clutter_fade.z) & 32768u) != 0u && m.clutter_fade.w <= 0.5) {
        let world = p_cam + view.world_position;
        p_cam += wind_hook::tree_offset(
            world, frame_origin, view.world_position, wow_light.monkey
        );
    }
    out.world_position = vec4<f32>(p_cam + view.world_position, 1.0);
    let view_rot = mat3x3<f32>(
        view.view_from_world[0].xyz,
        view.view_from_world[1].xyz,
        view.view_from_world[2].xyz,
    );
    out.position = view.clip_from_view * vec4<f32>(view_rot * p_cam, 1.0);
    // WMO batch order (`sun_scale.y`, 0 off WMO): the reference layers coplanar batches by MOBA
    // draw order under depth-write + LEQUAL; Bevy reorders draws, so a later batch must win the
    // reverse-Z GreaterEqual test. Scaling clip z by (1 + n·2⁻²³) raises z/w by n ULPs. Uniform
    // data, not a `DepthBiasState`, which would make every batch index its own pipeline.
    out.position.z *= 1.0 + m.sun_scale.y * 1.1920929e-7;
#ifdef WOW_SKY_DEPTH
    // The WMO-skybox lane (`clutter_fade.z` bit 13): clip z = 0 is reverse-Z infinitely far, so
    // the world always draws over the sky shell (`benilla_world::sky_order`).
    out.position.z = 0.0;
#endif
#ifdef WOW_MERGED_FADE
    // A fully faded placement leaves the clip volume, so its triangles never rasterize.
    let fade_d = distance(view.world_position.xz, vertex.fade_sphere.xz);
    out.merged_fade = merged_fade_alpha(vertex.fade_sphere.w, fade_d);
    if (out.merged_fade <= 0.0) {
        out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    }
#ifdef WOW_MERGED_SLOT
    out.merged_slot = vertex.merged_slot;
#endif
#endif
#endif

#ifdef VERTEX_UVS_A
    out.uv = vertex.uv;
    // Env-mapped batches (`clutter_fade.z` bit 12; `texture_unit_lookup[texCoordSet] > 2` at
    // `0x70b8bd`) generate their texcoord per vertex as `Model2.bls` does: in view space
    // R = normalize(P − 2(P·N)N), uv = R.xy·0.5 + 0.5 (`0x70b8d0`). The reference view basis
    // (`0x5c3e70`) and Bevy's −Z-forward one differ only in z, so R.xy needs no fixup.
#ifdef VERTEX_POSITIONS
#ifdef VERTEX_NORMALS
    if ((u32(m.clutter_fade.z) & 4096u) != 0u) {
        let p_view = view_rot * p_cam;
        let n_view = normalize(view_rot * out.world_normal);
        let refl = normalize(p_view - 2.0 * dot(p_view, n_view) * n_view);
        out.uv = refl.xy * 0.5 + vec2<f32>(0.5, 0.5);
    }
#endif
#endif
#endif
#ifdef VERTEX_UVS_B
    out.uv_b = vertex.uv_b;
#endif

#ifdef VERTEX_COLORS
    out.color = vertex.color;
#endif

#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex.instance_index;
#endif

    // Dynamic point lights, by receiver class (wow-re trace-forensics-abbey-interior-d3d §2/§4):
    // INTERIOR WMO surfaces take NONE (zero point lights on every observed WMO surface batch —
    // the earlier "point-lit abbey wall" was a mis-identified unit draw; they light per FRAGMENT
    // instead, `interior_room_light`), and interior M2 props take none HERE (their group-MOLR
    // point lobes are folded into the per-instance SH probe). Everything else —
    // exterior doodads, entities, clutter — keeps the nearest-`EXT_SEL_K` selection: clutter (merged in
    // world space) anchors at its MCNK chunk cell (the terrain draw unit it belongs to), every M2
    // at its INSTANCE origin (wow-re wmo-surface-dynamic-light §6 — the receiving unit's own
    // position, deliberately not the skinned per-vertex matrix).
    if (m.model_flags.x > 0.5 && m.model_flags.z < 0.5) {
        // MONKEY (wmo exterior points): an EXTERIOR-class WMO group (`model_flags.z` clear = MOGP
        // `& 0x48` set) is a street, a courtyard, a porch — drawn by the exterior law, the same law
        // the terrain beside it is drawn by, so it takes the exterior point term terrain takes. The
        // §2 "zero on every WMO surface" finding was measured on abbey INTERIOR rooms and reading
        // it as a blanket zero is what left Stormwind's Trade District torches lighting nothing but
        // the NPC beside them. Mirrors `static_gx.wgsl`'s branch (which draws the vast majority of
        // these batches) — keep the two in step.
        // MONKEY (outdoor torch shadows): the pick is PUBLISHED so the fragment stage can shadow
        // this same selection at night without re-ranking. One table walk still, not two -
        // the sum was always `eval(pick(..))`, it is just no longer inlined.
        let sel = wmo_exterior_pick(out.world_position.xyz);
        out.ext_sel = sel;
        out.point_lit = point_light_eval(sel, out.world_position.xyz, out.world_normal);
    } else if (m.model_flags.x > 0.5 || m.model_flags.z > 0.5) {
        out.point_lit = vec3<f32>(0.0);
        out.ext_sel = EXT_SEL_NONE;
    } else {
        var anchor = mesh_world_from_local[3].xyz;
        // MONKEY (ext light k12): CLUTTER is merged in world space and anchors at its MCNK cell, so
        // it ranks by that cell's BOX like the terrain under it; a standalone M2 keeps its own
        // origin, i.e. `box = 0` — the old point ranking, unchanged.
        var box = 0.0;
        if (m.clutter_fade.w > 0.5) {
            anchor = mcnk_cell_anchor(out.world_position.xyz);
            box = MCNK_CELL_HALF;
        }
        let sel = point_light_pick(anchor, box);
        out.ext_sel = sel;
        out.point_lit = point_light_eval(sel, out.world_position.xyz, out.world_normal);
    }
    return out;
}

@fragment
fn fragment(in: WowVsOut, @builtin(front_facing) is_front: bool) -> WowFragOut {
    // The UI model tile's cell clip: the reference gives each `<Model>` pane its widget rect as the
    // viewport; our panes share an atlas, so the tile passes its cell as a mat-anim row
    // (`anim_slots.w`, `[min.x, min.y, max.x, max.y]` in atlas texels) and this is the scissor.
    if (m.anim_slots.w > 0.5) {
        let r = wow_light.matanim[u32(m.anim_slots.w)];
        if (in.position.x < r.x || in.position.y < r.y
            || in.position.x > r.z || in.position.y > r.w) {
            discard;
        }
    }
    // The far-clip wall: discard past `farclip` (`fog_params.w`, 0 = off) by planar eye depth, as
    // terrain.wgsl does.
    if (wow_light.fog_params.w > 0.0) {
        let clip_z = -(view.view_from_world * vec4<f32>(in.world_position.xyz, 1.0)).z;
        if (clip_z > wow_light.fog_params.w) {
            discard;
        }
    }
#ifdef WOW_WATER_CLIP
    // The straddle split: a translucent model crossing its water plane draws on each side of the
    // water pass (the far copy has `clutter_fade.z` bit 11) and each copy keeps its half, the
    // reference's `M2UseClipPlanes` plane at the waterline. In sync with `straddle::keeps`.
    let water_clip = wow_light.water_clip[(mesh_functions::get_tag(in.instance_index) >> 19u) & 0x7ffu];
    if (water_clip.y != 0.0) {
        let far_copy = (u32(m.clutter_fade.z) & 2048u) != 0u;
        let keep_side = select(water_clip.y, -water_clip.y, far_copy);
        if (keep_side * (in.world_position.y - water_clip.x) < 0.0) {
            discard;
        }
    }
#endif
    // Rebuild Bevy's `VertexOutput` with the M2 texture transform (in sync with
    // `tex_anim::uv_transform`): uv' = R((uv + t − p) ⊙ s) + p, p = (½, ½), t = `sun_scale.zw`
    // plus its mat-anim delta, R and s from the affine row `[cos − 1, sin, sx − 1, sy − 1]`.
    var vo: VertexOutput;
    vo.position = in.position;
    vo.world_position = in.world_position;
    vo.world_normal = in.world_normal;
#ifdef VERTEX_UVS_A
    // An env-mapped coordinate takes no UV animation: the reference excludes an env stage from
    // `textureTransform`.
    if ((u32(m.clutter_fade.z) & 4096u) != 0u) {
        vo.uv = in.uv;
    } else {
        let uv_t = in.uv + m.sun_scale.zw + wow_light.matanim[u32(m.anim_slots.x)].xy;
        let affine = wow_light.matanim[u32(m.anim_slots.z)];
        let d = (uv_t - vec2<f32>(0.5, 0.5)) * vec2<f32>(1.0 + affine.z, 1.0 + affine.w);
        let c = 1.0 + affine.x;
        vo.uv = vec2<f32>(0.5 + d.x * c - d.y * affine.y, 0.5 + d.x * affine.y + d.y * c);
    }
#endif
#ifdef VERTEX_UVS_B
    vo.uv_b = in.uv_b;
#endif
#ifdef VERTEX_COLORS
    vo.color = in.color;
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    vo.instance_index = in.instance_index;
#endif
    var pbr_input = pbr_input_from_standard_material(vo, is_front);
    var base_color = pbr_input.material.base_color;
    // WMO MOCV alpha is lighting data (the TRANS lerp, the INT glow mask), and the reference's WMO
    // output alpha is tex.a alone. Bevy folds the vertex alpha into base_color, so re-sample the
    // texel alpha: dividing the fold back out loses everything where MOCV.a ≈ 0.
#ifdef VERTEX_COLORS
    if (m.model_flags.x > 0.5) {
#ifdef VERTEX_UVS_A
        base_color.a = textureSampleBias(
            pbr_bindings::base_color_texture,
            pbr_bindings::base_color_sampler,
            vo.uv,
            // The colour's LOD: `pbr_input_from_standard_material` applies `view.mip_bias`, and
            // coverage from another mip erodes out of step with the art.
            view.mip_bias,
        ).a;
#else
        base_color.a = 1.0;
#endif
    }
#endif
    // The detail-doodad LOD bias (`0x6813f4`): on the atlases whose alpha pyramid is binary below
    // mip 0 it keeps every fragment below full alpha, so the fade erodes per pixel, not per leaf.
    // wgpu has no sampler LOD bias and the sampler is shared with unbiased batches, so it goes on
    // this sample. Clutter's tint is white and its vertex colour the MCSH grey, so `* vo.color`
    // reproduces Bevy's fold.
    if (m.clutter_fade.w > 0.5) {
#ifdef VERTEX_UVS_A
        var biased = textureSampleBias(
            pbr_bindings::base_color_texture,
            pbr_bindings::base_color_sampler,
            vo.uv,
            view.mip_bias + DETAIL_DOODAD_LOD_BIAS,
        );
#ifdef VERTEX_COLORS
        biased = biased * vo.color;
#endif
        base_color = biased;
#endif
    }
    // The clutter distance fade, the reference's stage-1 ramp: u = (z_eye − 52.5)/17.5 by a
    // camera-space texgen (`0x6b2b80`), so the boundary is a view plane, not a sphere; the ramp is
    // a bilinear read of the 64-texel table `4·(63 − col)` (`0x6b2320`). It multiplies the alpha
    // the cutout tests (ALPHAREF `detailDoodadAlpha` = 128).
    if (m.clutter_fade.w > 0.5) {
        let z_eye = -(view.view_from_world * vec4<f32>(in.world_position.xyz, 1.0)).z;
        let u = (z_eye - m.clutter_fade.x) / max(m.clutter_fade.y - m.clutter_fade.x, 0.001);
        let ramp = clamp((254.0 - 256.0 * u) / 255.0, 0.0, 252.0 / 255.0);
        base_color.a = base_color.a * ramp;
    }

    // MONKEY (skybox): stage 1 multiplies in on UV set B, moved by its own texture transform
    // (the stage-0 law above): colour ×1 (Mod) or ×2 (Mod2x), alpha ×1.
#ifdef WOW_STAGE1
#ifdef VERTEX_UVS_B
    {
        let t1 = wow_light.matanim[u32(m.stage1.z)];
        let a1 = wow_light.matanim[u32(m.stage1.w)];
        let d1 = (in.uv_b + t1.xy - vec2<f32>(0.5, 0.5)) * vec2<f32>(1.0 + a1.z, 1.0 + a1.w);
        let c1 = 1.0 + a1.x;
        let uv1 = vec2<f32>(0.5 + d1.x * c1 - d1.y * a1.y, 0.5 + d1.x * a1.y + d1.y * c1);
        let s1 = textureSampleBias(stage1_texture, stage1_sampler, uv1, view.mip_bias);
        let k1 = select(1.0, 2.0, m.stage1.x > 1.5);
        base_color = vec4<f32>(base_color.rgb * s1.rgb * k1, base_color.a * s1.a);
    }
#endif
#endif
    // The MeshTag (mesh_tag.rs): bit 31 = highlight, bit 30 = interior fog; payload bits 0-5 = the
    // fade alpha (a zero payload is untagged, opaque), 19-29 = the rig slot, 6-13 = the ground
    // shade (0 lit, 255 MCSH-shadowed) or, on an interior-prop material, 6-18 = the SH-probe slot.
    let interior_prop = m.model_flags.z > 0.5 && m.model_flags.x < 0.5;
    let raw_tag = mesh_functions::get_tag(in.instance_index);
    let highlighted = (raw_tag & 0x80000000u) != 0u;
    let interior_fogged = (raw_tag & 0x40000000u) != 0u;
    let fade_tag = raw_tag & 0x3fffffffu;
    let alpha6 = f32(fade_tag & 0x3fu) / 63.0;
    var obj_fade = select(alpha6, 1.0, fade_tag == 0u);
#ifdef WOW_MERGED_FADE
    // A merged blob's per-vertex fade composes where the tag fade does, so it feathers the same.
    obj_fade = obj_fade * in.merged_fade;
#endif
    // The body tint (an aura colouring the whole model), by rig slot, 0 = identity: the material
    // ambient+diffuse colour (gx SetState(1), GL_COLOR_MATERIAL), so it multiplies the light sum
    // inside the clamp, never the emission.
    let tint_word = wow_light.rig_tint[(fade_tag >> 19u) & 0x7ffu];
    let inst_tint = select(
        vec3<f32>(
            f32((tint_word >> 16u) & 0xffu),
            f32((tint_word >> 8u) & 0xffu),
            f32(tint_word & 0xffu),
        ) * (1.0 / 255.0),
        vec3<f32>(1.0),
        tint_word == 0u,
    );
    // The blend twin re-applies its source's cutout: the reference scales ALPHAREF with the fade,
    // so the cutoff stays tex.a < 224/255 on the unfaded alpha. Only for a source batch that
    // alpha-tests (bit 10): ALPHAREF keys on the stored blend mode.
    if ((u32(m.clutter_fade.z) & 1024u) != 0u && base_color.a < VANILLA_ALPHA_KEY) {
        discard;
    }
    // The depth-prime twin (`M2UseZFill`) masks colour writes, so only the discards above shape its
    // depth. No early return: naga's MSL backend miscompiles the dead tail ("redefinition of
    // '_tmp'").
    let faded_alpha = base_color.a * obj_fade;
    let base = alpha_discard(pbr_input.material, base_color);

    // --- Lighting --------------------------------------------------------------------------------
    let is_clutter = m.clutter_fade.w > 0.5;
    let L = -normalize(wow_light.light_sun.xyz);
    // `wow_normalize`: an authored zero normal reaches here on the unskinned lane.
    let n_m2 = wow_normalize(pbr_input.world_normal);
    // Bevy negates `world_normal` on back faces of double-sided materials (foliage, every WMO
    // face); the reference never enables GL_LIGHT_MODEL_TWO_SIDE (`0x59ce30`), so undo it.
    let n_lit = select(-n_m2, n_m2, is_front);
    let ndotl = max(dot(n_lit, L), 0.0);
    let lit_nl = clamp(wow_light.light_ambient.rgb + wow_light.light_diffuse.rgb * ndotl, vec3<f32>(0.0), vec3<f32>(1.0));
    // The order-2 SH basis, shared by the exterior lobe and the interior probes.
    let quad = vec4<f32>(n_lit.x * n_lit.y, n_lit.y * n_lit.z, n_lit.z * n_lit.z, n_lit.x * n_lit.z);
    let n1 = vec4<f32>(n_lit, 1.0);
    let x2y2 = n_lit.x * n_lit.x - n_lit.y * n_lit.y;
    // Exterior doodads and entities: the header's `Model2.bls` lobe at the per-instance intensity
    // I, `[node+0xa4]` (animator targets 2.5 lit and 0.5 shadowed, `0x69e4ad`/`0x69e280`; 1.0
    // indoors, `0x69e36b`). `sun_scale.x` picks the family:
    //   ≥ 0.85: an entity M2, 2.5 mixed toward 0.5 by the tag's shade byte (ramped like `0x69e770`)
    //   0.5..0.85: a doodad (ADT MDDF or WMO MODD, both `CMapDoodadDef`): 1.0, never 2.5
    //   < 0.5: a doodad on MCSH-shadowed ground: 0.5
    // Deviation: `min(I, 1)`. The reference bakes I into the SH unclamped (`0x71c4e0`), so a lit
    // entity commits ×2.5 there and ×1.0 here; lifting the cap takes sun-facing surfaces past 1.0.
    // `entity_shade::LIT_T` aims the lit ramp at 1.0 because of it: lift the cap and set `LIT_T`
    // back to 0.0 together.
    let inst_shade = select(f32((fade_tag >> 6u) & 0xffu) / 255.0, 0.0, interior_prop);
    let mat_shade = select(0.0, 1.0, m.sun_scale.x < 0.5);
    let shade_t = max(mat_shade, inst_shade);
    let mid_band = m.sun_scale.x >= 0.5 && m.sun_scale.x < 0.85;
    let intensity = min(select(mix(2.5, 0.5, shade_t), 1.0, mid_band), 1.0);
    // One `intensity` multiply covers every sun band (never I²); the c10 `.w` ambient does not
    // scale. `pack_model_core_rows` packs the same closed form.
    let sun_dc = wow_light.grade.yzw * intensity;
    let sun_lobe = vec3<f32>(
        wow_light.sh_c10_r.w + sun_dc.x
            + intensity
                * (dot(wow_light.sh_c10_r.xyz, n_lit) + dot(wow_light.sh_c13_r, quad)
                    + wow_light.sh_c16.x * x2y2),
        wow_light.sh_c10_g.w + sun_dc.y
            + intensity
                * (dot(wow_light.sh_c10_g.xyz, n_lit) + dot(wow_light.sh_c13_g, quad)
                    + wow_light.sh_c16.y * x2y2),
        wow_light.sh_c10_b.w + sun_dc.z
            + intensity
                * (dot(wow_light.sh_c10_b.xyz, n_lit) + dot(wow_light.sh_c13_b, quad)
                    + wow_light.sh_c16.z * x2y2),
    );
    // Clamp the sum, never a term: the lobe's own ringing dips to −0.037·C near μ ≈ −0.53, and a
    // per-term clamp would erase it.
    let lit_doodad = clamp(sun_lobe, vec3<f32>(0.0), vec3<f32>(1.0));
    // WMO and clutter take the FFP N·L light with no terrain shade; the lobe needs the directional
    // light on (`light_sun.w`).
    let is_wmo = m.model_flags.x > 0.5;
    let is_interior = m.model_flags.z > 0.5;
    let use_doodad_shade = (wow_light.light_sun.w > 0.5) && !is_clutter && !is_wmo;
    let lit_exterior = select(lit_nl, lit_doodad, use_doodad_shade);
    // WMO interior groups (`groupFlags & 0x48 == 0`) by batch class (`tint.w`): INT (1) is unlit
    // `tex × MOCV`; TRANS (2) is the reference's two passes, lit × MOCV.a + unlit × (1 − MOCV.a),
    // as one lerp; EXT (0) is `lit_nl`. A WINDOW batch (MOMT 0x20) on the interior drawer lights
    // with ambient = diffuse = the Direct/Ambient midpoint, ambient +16/255 (`0x6d37e0`).
    var trans_a = 1.0;
#ifdef VERTEX_COLORS
    trans_a = in.color.a;
#endif
    let window_mid = 0.5 * (wow_light.light_ambient.rgb + wow_light.light_diffuse.rgb);
    let lit_window = clamp(
        window_mid + vec3<f32>(16.0 / 255.0) + window_mid * ndotl,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    let lit_int_base = select(lit_nl, lit_window, m.sidn.w > 0.5);
    var lit_wmo_interior = vec3<f32>(1.0);
    if (m.tint.w > 1.5) {
        lit_wmo_interior = mix(vec3<f32>(1.0), lit_int_base, trans_a);
    } else if (m.tint.w < 0.5) {
        lit_wmo_interior = lit_int_base;
    }
    // Interior M2 props (WMO MODD doodads): the reference commits the prop's MODD colour through a
    // fixed-axis diffuse lobe plus its group's MOLR lights as an order-2 SH probe, folded at spawn
    // by `lighting::prop_probe_coeffs` and evaluated here per fragment (the reference: per vertex).
    // Its soft wrap (≈ 0.088·C side-on) is the reference's response, not a max(N·L, 0).
#ifdef WOW_MERGED_SLOT
    // A merged blob bakes the slot per vertex; its tag carries only fog and alpha.
    let probe = 7u * in.merged_slot;
#else
    let probe = 7u * ((fade_tag >> 6u) & 0x1fffu);
#endif
    let lit_m2_interior = clamp(
        vec3<f32>(
            dot(wow_light.prop_probes[probe + 0u], n1)
                + dot(wow_light.prop_probes[probe + 3u], quad)
                + wow_light.prop_probes[probe + 6u].x * x2y2,
            dot(wow_light.prop_probes[probe + 1u], n1)
                + dot(wow_light.prop_probes[probe + 4u], quad)
                + wow_light.prop_probes[probe + 6u].y * x2y2,
            dot(wow_light.prop_probes[probe + 2u], n1)
                + dot(wow_light.prop_probes[probe + 5u], quad)
                + wow_light.prop_probes[probe + 6u].z * x2y2,
        ),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    let lit_interior = select(lit_m2_interior, lit_wmo_interior, is_wmo);
    // The authored-rig lane (`ShadeSel::Rig`, glue scenes): the scene's lights are folded into
    // probe slot 0 of the material's own buffer, which `lit_m2_interior` already evaluated (tag 0).
    let is_rig = m.sun_scale.x >= 1.5;
    let lit = select(select(lit_exterior, lit_interior, is_interior), lit_m2_interior, is_rig);
    // MONKEY (shadow hook): the realtime shadow (fetch + edge/night fade) is computed by
    // `benilla::shadow_hook`. This lane keeps only the rig-skin SAMPLE-POINT choice + the
    // interior/rig exclusion + ambient-preserving apply below.
    // MONKEY (moon shadows): the NIGHT arm of the same one fetch — 1.0 (inert) by day, by feature-off
    // and in the sun-down/moon-not-yet-up window. See `shadow_hook::realtime_shadow_terms`.
    var player_shadow = 1.0;
    var player_moon = 1.0;
    // GFX (moonlight): the moonlight's own visibility (`realtime_shadow_moonlit` `.z`).
    var player_moon_lit = 1.0;
    let view_z = (view.view_from_world * in.world_position).z;
    let shadow_cam_dist = distance(in.world_position.xyz, view.world_position.xyz);
    let sun_lane_w = shadow_hook::sun_shadow_w(wow_light.fog_params.z);
    let moon_lane_w = shadow_hook::moon_shadow_w(wow_light.fog_params.z);
#ifdef WOW_RIG_SKIN
    // A skinned UNIT samples the map ONCE at its rig origin (between the feet), nudged 2.5 units
    // TOWARD the sun, and dims uniformly — the reference's per-unit response. Per-fragment sampling
    // is wrong here: the caster proxy holds a CPU-skinned copy of this very body, so a body fragment
    // reads its own limbs/weapon as occluders (self-shadow speckle). Gated on the directional-enable
    // flag so an interior-lit unit never takes an exterior dim (player_shadow stays 1.0).
    if (wow_light.light_sun.w > 0.5) {
        let sun_ray = normalize(wow_light.light_sun.xyz);
        let anchor = vec4<f32>(
            wow_light.rig_origin[(fade_tag >> 19u) & 0x7ffu].xyz - 2.5 * sun_ray,
            1.0,
        );
        let terms_rig = shadow_hook::realtime_shadow_moonlit(
            anchor,
            vec3<f32>(0.0, 1.0, 0.0),
            view_z,
            shadow_cam_dist,
            wow_light.wmo_fog_params.z,
            sun_lane_w,
            moon_lane_w,
            wow_light.monkey.misc.w,
        );
        player_shadow = terms_rig.x;
        player_moon = terms_rig.y;
        player_moon_lit = terms_rig.z;
    }
#else
    let terms_frag = shadow_hook::realtime_shadow_moonlit(
        in.world_position,
        wow_normalize(in.world_normal),
        view_z,
        shadow_cam_dist,
        wow_light.wmo_fog_params.z,
        sun_lane_w,
        moon_lane_w,
        wow_light.monkey.misc.w,
    );
    player_shadow = terms_frag.x;
    player_moon = terms_frag.y;
    player_moon_lit = terms_frag.z;
#endif
    // The realtime map blocks only the directional sun. Preserve the authored ambient/probe
    // contribution instead of multiplying the whole lighting result; the latter makes interiors,
    // point-lit props, and shadow-side characters globally too dark.
    //
    // `is_rig` (the glue create-booth lane) is excluded DELIBERATELY: a booth scene's light is
    // entirely its authored rig — there is no world sun to block. World units are `ShadeSel::Lit`
    // and receive through the WOW_RIG_SKIN ground sample above. The `player_shadow < 0.999`
    // arm keeps `worldShadows 0` (and a fully sunlit fragment) byte-identical: `ambient +
    // (lit − ambient) × 1.0` is not guaranteed to round back to `lit`.
    let shadow_term = mix(SHADOW_SUN_FLOOR, 1.0, player_shadow);
    var lit_with_shadow = select(
        lit,
        clamp(
            wow_light.light_ambient.rgb + (lit - wow_light.light_ambient.rgb) * shadow_term,
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        ),
        !is_interior && !is_rig && player_shadow < 0.999,
    );
    // MONKEY (moon shadows): the night sky includes ambient AND the directional/SH lobe.
    // Scale its UNCLAMPED value; points join below, then the combine saturates. A saturated
    // torch must never lose energy to the moon. Interiors/authored rigs and the exact off path
    // retain their old expressions; no multiply-by-one round trip on daylight or strength zero.
    if (player_moon < 1.0 && !is_interior && !is_rig) {
        let sky = select(wow_light.light_ambient.rgb + wow_light.light_diffuse.rgb * ndotl,
            sun_lobe, use_doodad_shade);
        lit_with_shadow = sky * player_moon;
    }
    // GFX (moonlight): ADD the moon onto the exterior night (never interiors or the glue booth).
    // `moon.w` is exactly 0 by day and with `moonLight 0`: the branch is not entered.
    if (wow_light.monkey.moon.w > 0.0 && !is_interior && !is_rig) {
        lit_with_shadow += moonlight_hook::moon_light(n_lit, wow_light.monkey.moon) * player_moon_lit;
    }

    // Gamma-space albedo: lighting runs on the authored byte values. `m.tint` plus its mat-anim
    // delta is the animated M2Color tint.
    let anim_tint = m.tint.rgb + wow_light.matanim[u32(m.anim_slots.y)].xyz;
    let albedo = base.rgb * anim_tint;
    // An unlit batch replaces the lit path: fullbright, with only the highlight added (below).
    let is_emissive = m.model_flags.w > 0.5;
    // SIDN night glow: the emissive × the night fraction (ramping 20:30→21:30, 06:00→07:00), a
    // GL_EMISSION term inside the clamped lit sum, on lit lanes only.
    var sidn_w = 1.0;
    if (is_interior && is_wmo) {
        if (m.tint.w > 1.5) {
            sidn_w = trans_a; // TRANS: weighted by its lit pass
        } else if (m.tint.w > 0.5) {
            sidn_w = 0.0; // INT: unlit, so no emission
        }
    }
    let sidn_e = m.sidn.rgb * (wow_light.grade.x * sidn_w);
    // WoW dynamic point lights (decisions 0016/0273/0278, selection 0285) — exterior doodads,
    // entities, clutter, and terrain receive their unit's committed lights: the ≤`EXT_SEL_K` NEAREST to the
    // receiving unit's own position or AABB, never the whole scene (`point_light_sum`). INTERIOR WMO
    // surfaces take ZERO (observed on every WMO surface batch in the abbey capture — an interior
    // capture, hence the MONKEY split above) and interior props fold their group-MOLR lobes into
    // the SH probe instead — both zeroed in the VERTEX stage; an EXTERIOR-class WMO group takes
    // the exterior-lane term (`wmo_exterior_point_sum`) there instead, ranked from its MCNK cell
    // like the terrain it adjoins, which is what puts a street torch on the cobbles. The term
    // arrives GOURAUD-INTERPOLATED — per-vertex like the reference FFP, whose tessellation-scale
    // smoothing is the authored look. Diffuse-only (committed ambient/specular are zero).
    // MONKEY (outdoor torch shadows): ...and after dark that Gouraud term is CAST-SHADOWED.
    //
    // The same entries the vertex stage picked (`in.ext_sel`) are re-evaluated PER FRAGMENT
    // with each fixture's own cube-map occlusion folded in, and the two results are blended by
    // `night_w = 1 - sun_shadow_strength`. Written as a blend rather than a swap for two reasons:
    //   - `fog_params.z` is EXACTLY 1.0 whenever the sun is above the daylight threshold
    //     (`global_light::sun_shadow_strength` is a smoothstep that saturates), so `night_w` is
    //     exactly 0 and the branch is not entered - DAYLIGHT IS THE SAME BITS, not "a mix that
    //     ought to round back to b".
    //   - at dusk the shadow, and the Gouraud->per-fragment change of the term itself, arrive on
    //     the same clock the sun shadows leave on instead of snapping at some threshold.
    // `torch_ext_on()` is the CPU's one-bit verdict (`exteriorShadows` AND night AND at least one
    // promoted exterior fixture), so with the cvar off this is dead too. `n_lit` is the same
    // normal every other lit term here uses, so the normal-offset sample leaves the skin the way
    // the interior entity lane's does.
    var point_diffuse = in.point_lit;
    let ext_night_w = select(0.0, clamp(1.0 - wow_light.fog_params.z, 0.0, 1.0), torch_ext_on());
    if (ext_night_w > 0.0) {
        point_diffuse = mix(
            in.point_lit,
            point_light_eval_shadowed(in.ext_sel, in.world_position.xyz, n_lit),
            ext_night_w,
        );
    }

    // The hover/target highlight (tag bit 31): the scene's committed ambient
    // (`0x614576`-`0x6145bd`), added to the batch colour inside the final clamp, lit or unlit
    // (`c29`). Sampled live, where the reference holds the value sampled when the highlight
    // began: a unit's tag has no slot to hold a colour, and the ambient moves slowly.
    let highlight = select(vec3<f32>(0.0), wow_light.light_ambient.rgb, highlighted);
    // The FFP combine: the light sum (lit, point lights, emission) clamps first and the texture
    // modulates it, `tex × clamp(C·sum + emission)`, with C the GL_COLOR_MATERIAL colour (MOCV on
    // WMO, the body tint on M2). The WMO branch divides Bevy's MOCV fold back out, guarded; a dim
    // channel's product is ~0 either way.
    var lit_rgb: vec3<f32>;
#ifdef VERTEX_COLORS
    if (is_wmo) {
        let vc = in.color.rgb;
        let primary = clamp(
            vc * (lit_with_shadow + point_diffuse) + sidn_e + highlight,
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
        let tex_rgb = base.rgb / max(vc, vec3<f32>(1.0 / 255.0));
        lit_rgb = tex_rgb * m.tint.rgb * primary;
        if (is_interior && m.tint.w > 0.5 && m.tint.w < 1.5) {
            // INT: the reference's interior pixel shader, `tex·MOCV.rgb·(1 + 4·MOCV.a)` with only
            // the framebuffer's final clamp.
            lit_rgb = clamp(
                tex_rgb * m.tint.rgb * vc * (1.0 + 4.0 * trans_a),
                vec3<f32>(0.0),
                vec3<f32>(1.0),
            );
        }
    } else {
        // An M2: the body tint multiplies the light terms as MOCV does above. A WMO surface has
        // no tint slot, so the WMO branch leaves it out.
        let primary = clamp(
            inst_tint * (lit_with_shadow + point_diffuse) + sidn_e + highlight,
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
        lit_rgb = albedo * primary;
    }
#else
    // A WMO batch without MOCV lands here too; its tint slot is the identity slot 0.
    let primary = clamp(
        inst_tint * (lit_with_shadow + point_diffuse) + sidn_e + highlight,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    lit_rgb = albedo * primary;
#endif
    // MONKEY (dynamic interiors, `interiorLight` on): an INDOOR unit / GameObject / WMO prop — the
    // entity's OWN light-node classification (tag bit 30, `interior_fogged`), never a WMO surface,
    // never the create booth's authored rig — takes the fixture-lit room light
    // (`interior_room_light`, mirrored from `static_gx.wgsl`) instead of the day/night CGLight or
    // the baked probe, so it matches the surfaces around it (the tavern's chairs and NPCs stayed
    // bake-bright in a torch-lit room). The instance tint still modulates; the same soft rolloff
    // as the surfaces; and (Phase 3A) the same per-fixture torch shadows, from the shared map.
    // CAMERA-INDEPENDENT indoor test: `is_interior` (`model_flags.z`) — the SAME signal the WMO
    // surface lane and the original interior-prop lane trust to route an entity to interior lighting,
    // set from the anchor's down-ray classification, not the camera. (The earlier gates were both
    // wrong: `interior_fogged` is the MFOG bit the camera rewrites every frame — an indoor unit went
    // dark when the camera left; `probe != 0` is non-zero for EVERY entity, indoor or out, since an
    // exterior entity also carries a day/night SH probe slot — so every outdoor GO/unit lit as
    // interior, the debug-1 green everywhere.)
    // PLUS the classifier's MATTE law (tag bit 14, `mesh_tag::MATTE_INDOOR_BIT`): the anchor
    // stands indoors but its material stayed in exterior mode (no bake to fold) — the same room,
    // the same lane. Without it a moving unit whose verdict flickers Bake↔Matte at one spot
    // (the down-ray marginally hitting the baked floor) switched its lane off and on. Read only
    // in exterior material mode: in interior mode those bits are the probe slot.
    let matte_indoor = !is_interior && (fade_tag & 0x4000u) != 0u;
    if ((is_interior || matte_indoor) && !is_wmo && !is_rig && wow_light.wmo_fog_params.w > 0.5) {
        let room = interior_room_light(in.world_position.xyz, n_lit, 0u, 0u);
        // MONKEY (indoor highlight): the hover/target emissive (`highlight`, tag bit 31) rides THIS
        // lane's light sum too. It is the same GL_EMISSION placement as the exterior branches above
        // — added to the material's ambient+diffuse product INSIDE the [0,1] saturate, with the
        // texture (`albedo`) modulating the clamped result — so an indoor chair lifts by exactly the
        // the scene-ambient lift an outdoor one does. It has to be folded HERE, not left in `lit_rgb`, because this
        // lane REPLACES the exterior result a few lines down (`mix(lit_rgb, room_rgb, lane_w)`):
        // with `lane_w` at 1 (a settled indoor unit) the exterior sum that carried the lift was
        // discarded wholesale, which is why hovering a chair inside a Stormwind house brightened
        // nothing at all while the same chair on the street did.
        // The clamp is a no-op when nothing is hovered (`highlight` 0, and both factors are already
        // ≤1), so the un-hovered indoor look is bit-identical to before.
        // Placed BEFORE the debug branch on purpose: modes 1/2/3 all discard `room_rgb` and return
        // their own diagnostic colour, so `interiorDebug` is untouched by this.
        // MONKEY (bake floor): the room's own bake floor, so an entity in a fixture-starved room is
        // not a black silhouette against a dimly lit wall. A flat room mean rather than the
        // surfaces' per-batch `vc.rgb` — see `INTERIOR_BAKE_ENTITY_MEAN` for why the per-group
        // number is not reachable on this lane. Inside the same rolloff as the surfaces, so a
        // torch-lit NPC is moved by single-digit percent and an unlit one is carried.
        let bake = INTERIOR_BAKE_ENTITY_MEAN * (fract(wow_light.sh_c16.w) / BAKE_LANE_SCALE);
        var room_rgb = albedo * clamp(
            inst_tint * (vec3<f32>(1.0) - exp(-(room + vec3<f32>(bake)) * wow_light.point_count.w))
                + highlight,
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
        let idbg = u32(max(wow_light.wmo_fog_params.w - 1.0, 0.0) + 0.5);
        if (idbg == 2u) {
            // The entity's OWN cube-map sampling as greyscale (the shared stub would show white).
            room_rgb = vec3<f32>(torch_entity_debug_factor(in.world_position.xyz, n_lit));
        } else {
            room_rgb = shadow_hook::interior_debug_override(idbg, room_rgb, in.world_position, n_lit);
        }
        // MONKEY (portal lane fade): CROSSFADE, not a switch. `lit_rgb` still holds this part's
        // EXTERIOR result (day/night × its ground-shade byte) — the "outside" colour — and the
        // classifier ramps tag bits 15..=18 from 0 to 15 over ~half a second as the entity's
        // anchor crosses the portal, so the two lanes blend the way the surfaces at the seam
        // already do. Without it a character stepping through a doorway went from night-dark to
        // torch-warm in ONE frame while the wall beside it faded.
        //
        // Forced to 1.0 in INTERIOR material mode: there those bits are the middle of the SH probe
        // slot (`mesh_tag`'s two payload modes), so reading them as a weight would fade an indoor
        // baked unit by its probe INDEX. The classifier only ever commits the interior material at
        // full weight anyway — every frame of an actual blend is written in exterior mode.
        let lane_w = select(f32((fade_tag >> 15u) & 0xfu) / 15.0, 1.0, is_interior);
        lit_rgb = mix(lit_rgb, room_rgb, lane_w);
    }
    // The unlit path: an M2's unlit program outputs `c28 + c29` (`0x70c663`-`0x70c693` fold the
    // tint·M2Color term into `c29` beside the highlight), so the texel modulates
    // `clamp(C·tint + highlight)`, C the M2Color. A WMO keeps the plain modulate.
    var unlit_rgb = albedo * inst_tint;
    if (!is_wmo) {
#ifdef VERTEX_COLORS
        // The constant M2Color rides the vertex colour, which Bevy folded into `base`.
        let unlit_c = in.color.rgb * anim_tint;
        let unlit_tex = base.rgb / max(in.color.rgb, vec3<f32>(1.0 / 255.0));
#else
        let unlit_c = anim_tint;
        let unlit_tex = base.rgb;
#endif
        let unlit_sum = unlit_c * inst_tint + highlight;
        unlit_rgb = unlit_tex * clamp(unlit_sum, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    var rgb = select(lit_rgb, unlit_rgb, is_emissive);
    // An M2 Mod or Mod2x batch draws the bare texel: the reference zeroes its tint·M2Color term and
    // forces the primary colour to the blend identity (`0x70c507`/`0x70c5b8`), so neither the
    // animated M2Color nor the body tint reaches it. Its alpha still does, through the lerp below.
    let is_mod = (u32(m.clutter_fade.z) & 128u) != 0u;
    let is_mod2x = (u32(m.clutter_fade.z) & 256u) != 0u;
    if ((is_mod || is_mod2x) && !is_wmo) {
        rgb = base.rgb;
    }

    // MONKEY (wet): rain on sky-exposed WMO/M2 surfaces (wet_hook.wgsl); interior, glue booth, unlit,
    // Mod/Mod2x, additive and dry = untouched.
    let wet_dry = is_interior || interior_fogged || is_rig || is_emissive || is_mod || is_mod2x
        || (u32(m.clutter_fade.z) & 4u) != 0u;
    // MONKEY (rainshelter): nothing gets wet under a roof, porch or bridge (the shelter grid).
    var wet_open = 1.0;
    if (wow_light.monkey.wet_a.y > 0.0) {
        let wet_st = wet_hook::shelter_taps(in.world_position.xyz, wow_light.shelter_hdr,
            wow_light.shelter_cfg);
        if (wet_st.live > 0.0) {
            wet_open = 1.0 - wet_hook::shelter_amount(wet_st, wow_light.shelter[wet_st.idx.x],
                wow_light.shelter[wet_st.idx.y], wow_light.shelter[wet_st.idx.z],
                wow_light.shelter[wet_st.idx.w], in.world_position.y, wow_light.shelter_cfg);
        }
    }
    let wet = wet_hook::wet_surface(rgb, n_lit, in.world_position.xyz,
        select(1.0, 0.0, wet_dry) * wet_open, wow_light.monkey);
    rgb = wet.albedo;
    if (wet.boost > 0.0) {
        rgb = min(rgb + wet_hook::wet_sheen(wet.boost, n_lit, normalize(view.world_position - in.world_position.xyz),
            L, wow_light.light_diffuse.rgb, wow_light.fog_color.rgb, player_shadow), vec3<f32>(1.0));
    }

    // Linear fog by planar eye depth, as in terrain.wgsl. Per-batch colour policy (`clutter_fade.z`
    // bits 4-6, the M2 state setter `0x70baf0`): 0 scene, 1 black (additive), 2 white (Mod),
    // 3 grey (Mod2x), 4 unfogged (render flag 0x02). Tag bit 30, not the static `model_flags.z`,
    // selects the interior triple: WMO content by the per-group `[0xca7f00]` gate on the pushes
    // `0x6b5190`/`0x6b62e0`, an entity M2 by its node's classification (`0x71c110`, `[node+0xc]`).
    var fog_color = wow_light.fog_color;
    var fog_span = wow_light.fog_params.xy;
    if (interior_fogged) {
        fog_color = wow_light.wmo_fog_color;
        fog_span = wow_light.wmo_fog_params.xy;
    }
    let fog_policy = (u32(m.clutter_fade.z) >> 4u) & 7u;
    if (fog_color.w > 0.5 && fog_policy != 4u) {
        let eye_z = -(view.view_from_world * vec4<f32>(in.world_position.xyz, 1.0)).z;
        var fog_rgb = fog_color.xyz;
        if (fog_policy == 1u) { fog_rgb = vec3<f32>(0.0); }
        else if (fog_policy == 2u) { fog_rgb = vec3<f32>(1.0); }
        else if (fog_policy == 3u) { fog_rgb = vec3<f32>(0.50196078); }
        // MONKEY (p0 fog hook): the shared fog law (fog_hook.wgsl); classic is bit-identical.
        rgb = fog_hook::apply_fog(rgb, fog_rgb, fog_span, eye_z, in.world_position.xyz,
            view.world_position, fog_policy == 0u, wow_light.monkey);
    }


    var out: WowFragOut;
    // Output alpha is tex × fade. Opaque intent (`clutter_fade.z` bit 3) pins it to 1: a no-op
    // under correct pipeline state, and a guard for macOS/Metal with an extra camera, where an
    // opaque draw can bind a blending pipeline and show the BLP's garbage alpha.
    let opaque_intent = (u32(m.clutter_fade.z) & 8u) != 0u;
    // Additive (`clutter_fade.z` bit 2, the bit `specialize` keys the (ONE, ONE) blend on): the
    // alpha weight folds into the colour here, in gamma, as the reference weights its source.
    let is_additive = (u32(m.clutter_fade.z) & 4u) != 0u;
    var out_rgb = rgb;
    if (is_additive) {
        out_rgb = out_rgb * faded_alpha;
        // MONKEY (post): additive M2 cards become HDR before their framebuffer blend stacks them.
        out_rgb = emissive_hook::emissive_boost(
            out_rgb, emissive_hook::EMISSIVE_M2_ADD, wow_light.light_diffuse.w, 1.0);
    }
    // Mod (bit 7) and Mod2x (bit 8) read no source alpha, so the fade rides the colour as in the
    // reference: texenv preset 5, `mix(prev.rgb, tex.rgb, prev.a)`, with the primary colour forced
    // to the blend identity (white, or 0.5 grey for Mod2x) and prev.a the instance alpha. The fog
    // above commutes with this because its white and grey are that identity.
    if (is_mod || is_mod2x) {
        let identity = select(vec3<f32>(1.0), vec3<f32>(0.5), is_mod2x);
        out_rgb = mix(identity, out_rgb, obj_fade);
    }
    // Raw gamma out: blending happens in gamma like the reference's byte framebuffer; the frame
    // decodes once, in the FFXGlow combine.
    out.color = vec4<f32>(out_rgb, select(faded_alpha, 1.0, opaque_intent));
    return out;
}
