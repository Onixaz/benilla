// The retained static-world pass: `wow_model.wgsl`'s shading for statics that never fade or
// animate (ADT doodads, WMO groups, interior M2 props), with per-vertex flag words and a per-item
// record in place of materials. `StaticGx::divert` admits opaque and alpha-tested batches with no
// env map and no depth flags. The light prefix, helpers and lighting lanes copy `wow_model.wgsl`'s
// and must stay in sync: naga_oil cannot import functions that use another module's bindings.

#import bevy_pbr::{
    mesh_view_bindings::{lights, view},
    shadows,
}
// MONKEY (shadow hook): the realtime directional-shadow term (fetch + edge/night fade) lives here.
#import benilla::shadow_hook
// GFX (moonlight): the additive moon term.
#import benilla::moonlight_hook
// MONKEY (p0 MonkeyFrame): the programme block's struct, mirrored after the point table.
#import benilla::monkey_frame
// MONKEY (p0 fog hook): the one distance-fog law every receiver calls.
#import benilla::fog_hook
// MONKEY (post): shared tier-gated HDR emission; Off is an exact identity.
#import benilla::emissive_hook
// MONKEY (wind): shared foliage vertex displacement; tree shadows intentionally stay static.
#import benilla::wind_hook
// MONKEY (wet): rain on surfaces (wet_hook.wgsl).
#import benilla::wet_hook

// Group 0 is Bevy's standard mesh-view bind group (view matrices, directional-light records and
// the shadow textures the retained pass reads).
const SHADOW_SUN_FLOOR: f32 = 0.45;

// Prefix of lighting::global_light's buffer; field order in sync with it and wow_model.wgsl.
struct WowLight {
    light_ambient: vec4<f32>,
    light_diffuse: vec4<f32>,
    light_sun: vec4<f32>,
    light_spec: vec4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
    sh_c10_r: vec4<f32>,
    sh_c10_g: vec4<f32>,
    sh_c10_b: vec4<f32>,
    sh_c13_r: vec4<f32>,
    sh_c13_g: vec4<f32>,
    sh_c13_b: vec4<f32>,
    sh_c16: vec4<f32>,
    _water: array<vec4<f32>, 4>,
    grade: vec4<f32>,
    wmo_fog_color: vec4<f32>,
    wmo_fog_params: vec4<f32>,
    // MONKEY (light lanes): each light is TWO rows — `[pos.xyz, range]`, `[rgb, lane]`. `lane` is
    // `0` for an EXTERIOR light (read only by `point_light_sum`) and `> 0.5` for an INTERIOR
    // fixture (read only by `interior_room_light`), where the value IS that fixture's reach in
    // yards. wow_model.wgsl owns the full note.
    point_count: vec4<f32>,
    points: array<vec4<f32>, 512>,
    // MONKEY (p0 MonkeyFrame): the programme block after the point table (monkey_frame.wgsl).
    monkey: monkey_frame::MonkeyFrame,
    // MONKEY (rainshelter): the rain-occlusion grid (`weather/shelter.rs`): `[origin_x, origin_z,
    // 1/cell, active]`, `[base_y, cells per side, 0, 0]`, then one packed word a cell (wet_hook.wgsl).
    shelter_hdr: vec4<f32>,
    shelter_cfg: vec4<f32>,
    shelter: array<u32, 16384>,
    // lighting::prop_probes: 8192 slots of 7 rows; the buffer's later regions are not mirrored.
    prop_probes: array<vec4<f32>, 57344>,
}
@group(2) @binding(0) var<storage, read> wow_light: WowLight;
// MONKEY (room gate): the per-fixture ROOM CLAIM table — which WMO groups each packed INTERIOR
// fixture is allowed to light. Packed by `lighting::global_light`'s `build_light_data` into its own
// storage buffer (NEVER into `WowLight`: that struct is mirrored three ways and must not resize),
// and bound here beside it. `ROOM_CLAIM_STRIDE` u32 per packed light, index-parallel with the
// point table:
//   [0] the claiming WMO placement's ENTITY INDEX   [1] claim count (**0 = UNGATED**)
//   [2 .. 2+ROOM_CLAIM_MAX) the claimed group ids, stored as `group + 1` so 0 stays "empty"
// MONKEY (soft portal claims): then one 4-word FADE record per claim slot, at `ROOM_CLAIM_FADE`:
//   [+0..+3] the doorway's WORLD-space centre, raw f32 bits (a world coordinate reaches ~17000 yd,
//            so nothing narrower carries it; the CPU side converts out of WMO model space once, at
//            spawn, because this side has only a fragment position and no placement matrix)
//   [+3]     `radius | slack << 16`, both u16 yards x CLAIM_FADE_SCALE. **radius 0 = a HARD claim**
//            (containment/MOLR: weight 1 everywhere), which is also what the zero padding reads as
// Every claim of one fixture shares one instance, because a fixture belongs to exactly one
// building (`LightRooms`/`WmoGroupVis` carries one instance and a group list) — which is what lets
// the instance sit in the head word instead of once per claim.
@group(2) @binding(1) var<storage, read> room_claims: array<u32>;

// Per-cell state (static_gx/render.rs `cell_layout`).
struct GxCell {
    // xyz = the bake's recentring origin (0974): world = vertex + origin.
    // MONKEY (room gate): `.w` is this REGION's identity — the WMO placement instance's entity
    // index, `bitcast` out of the float lane (the uniform's declared size is 16 bytes and must not
    // move; the lane was a hard 0.0 pad). 0 on terrain cells, which name no building.
    origin: vec4<f32>,
}
@group(1) @binding(0) var<uniform> cell: GxCell;
// The per-item records, indexed by the vertex word's low 16 bits: x the texture-array layer, y
// the WMO batch order (0 on cells), z the MOMT SIDN colour r|g<<8|b<<16 in gamma bytes, w flags:
// bit 0 exile kill, bits 1..=13 the interior-prop probe slot, bit 14 interior fog (`[0xca7f00]`).
@group(1) @binding(1) var<storage, read> recs: array<vec4<u32>>;
@group(1) @binding(2) var tex_array: texture_2d_array<f32>;
// Repeat and clamp model-albedo samplers: trilinear, aniso 8, the same as the entity path's.
@group(1) @binding(3) var samp_repeat: sampler;
@group(1) @binding(4) var samp_clamp: sampler;

// Vertex word bits; keep in sync with static_gx/mod.rs WORD_*.
const WORD_WRAP_X: u32 = 65536u;    // 1 << 16
const WORD_WRAP_Y: u32 = 131072u;   // 1 << 17
const WORD_UNLIT: u32 = 262144u;    // 1 << 18
const WORD_FOG_OFF: u32 = 524288u;  // 1 << 19
const WORD_SHADE_LIT: u32 = 1048576u; // 1 << 20: ShadeSel::Lit, which no static carries
const WORD_TEXTURED: u32 = 2097152u;  // 1 << 21
// The WMO lane: the entity path's per-material facts as bits.
const WORD_WMO: u32 = 4194304u;        // 1 << 22: model_flags.x, a WMO surface
const WORD_INTERIOR: u32 = 8388608u;   // 1 << 23: model_flags.z, an interior group
const WORD_CLASS_INT: u32 = 16777216u; // 1 << 24: tint.w == 1, an INT batch
const WORD_CLASS_TRANS: u32 = 33554432u; // 1 << 25: tint.w == 2, a TRANS batch
const WORD_WINDOW: u32 = 67108864u;    // 1 << 26: sidn.w, the WINDOW midpoint light
const WORD_HAS_VC: u32 = 134217728u;   // 1 << 27: the batch authors vertex colours
// INTERIOR without WMO is an interior M2 prop, the entity shader's `interior_prop =
// flags.z && !flags.x`: probe lighting, interior fog, no point lights.
const WORD_MATTE: u32 = 268435456u;    // 1 << 28: ShadeSel::Matte, fixed intensity 1.0
// MONKEY (wind): alpha-tested leaf batch of a classified static tree/bush model.
const WORD_FOLIAGE_WIND: u32 = 536870912u; // 1 << 29

// MONKEY (room gate): record column `w` bits 15..=26 — this item's ROOM KEY, packed as
// `group + 1`, so **0 means the item names no room** and takes every fixture (a terrain-cell item,
// or a WMO prop whose referrer set is not exactly one group). Bit 0 is the exile kill bit, bits
// 1..=13 the interior-prop probe slot and bit 14 the interior fog lane, so 15 is the first free
// bit; 12 bits covers every shipped WMO's group count (the largest, Stratholme, has 92).
const RECORD_ROOM_SHIFT: u32 = 15u;
const RECORD_ROOM_MASK: u32 = 4095u;

// MONKEY (ext-class night law): record column `w`, bit 27 — this batch's group is EXTERIOR-class
// (MOGP `& 0x48`) but at BUILDING scale, `benilla_formats::room_claim::ext_building_scale`. The
// room key ends at bit 26, so 27 is the first free one. Keep in sync with `static_gx/render.rs`'s
// `RECORD_EXT_NIGHT_BIT`.
const RECORD_EXT_NIGHT: u32 = 134217728u; // 1 << 27

// MONKEY (enclosed day floor): record column `w`, bit 28 — this batch's group is an INTERIOR room
// whose box centre sits inside a BUILDING-SCALE exterior shell of the same WMO
// (`benilla_formats::room_claim::enclosed_by_building_shell`). Bit 27 is the night law, so 28 is
// the first free one; 29..=31 stay free. Keep in sync with `static_gx/render.rs`'s
// `RECORD_ENCLOSED_BIT`.
const RECORD_ENCLOSED: u32 = 268435456u; // 1 << 28

// MONKEY (enclosed day floor): the fraction of `wmo_fog_params.w` that carries `interiorDaylight`
// (`w = 1 + interiorDebug + interiorDaylight * this`). Keep in sync with
// `benilla_world::lighting::DAYLIGHT_LANE_SCALE`, which carries the whole argument for why the
// value rides a lane's spare RANGE rather than a spare slot, and why every existing decode of this
// lane (`> 0.5`, `u32(max(w - 1, 0) + 0.5)`) is insensitive to a fraction below 0.5.
const DAYLIGHT_LANE_SCALE: f32 = 0.49;

// MONKEY (bake floor): the fraction of `sh_c16.w` that carries `interiorBakeFloor`, with
// `interiorGain` ALREADY FOLDED IN by the packer (`w = worldShadows + bakeFloor * gain * this`).
// Keep in sync with `benilla_world::lighting::BAKE_LANE_SCALE` and with `wow_model.wgsl`'s copy.
// `sh_c16.w` has exactly ONE other decode in the whole shader set — `terrain.wgsl`'s
// `sh_c16.w > 0.5` world-shadow gate — and a fraction strictly below 0.5 cannot move it from
// either side; the packer clamps the product to 1 so the fraction can never reach the cliff.
const BAKE_LANE_SCALE: f32 = 0.49;

// MONKEY (bake floor): the SHARE OF ITS OWN MOCV BAKE that an interior-lane fragment keeps even
// when no fixture reaches it.
//
// The room lane's premise is that the bake's LEVEL is wrong — authored for a different global
// exposure, and the live fixtures decide instead. True, and it leaves a hole: a room the fixture
// table cannot reach has NO budget at all beyond `interiorAmbient`, so it renders black. The
// Lion's Pride Inn's east vestibule `g0` (the door band, box x 14.1..20.5, between the ext-class
// porch `g11` and the INT room `g1`) is the measured case — MOLR 0, ZERO fixture claims (the
// nearest fixtures are L2 at 16 yd against R 11.2 and L3 at 17.6 against R 14.7), one faded portal
// hop worth ~0.0003, so the whole budget is the bare ambient floor and the band reads 0.0194 x tex
// at the owner's cvars. The reference client has no such hole: every interior batch draws at its
// authored MOCV whether or not a light is registered, so a fixture-starved vestibule is DIM there,
// never black.
//
// What survives the lane's premise is the bake's RELATIVE statement about the room, so the floor
// is a FRACTION of it (`vc.rgb * k`), not the bake. It is added INSIDE the rolloff next to
// `enclosed_day_floor`, for the same reason that one is: it saturates with the fixtures rather
// than stacking on a lit room, so a candle-lit surface barely moves while an unlit one comes up
// off the floor. The previous attempt put the UNCAPPED bake share on the TRANS batches alone and
// produced the flat grey band this file's portal-bleed comment records (15-18 x its INT
// neighbour); a small fraction of the bake on EVERY interior batch is the other end of that trade.
//
// SCOPE — zero on an EXT-class batch of an interior group (`!class_int && !class_trans`, the
// cellar stair). The loader forces that population's MOCV alpha to 1.0 and its `vc.rgb` is the
// flat white the EXT batches carry (the extract prints `mocv rgb (255,255,255)` for every one of
// them), so a bake term there would be a full-strength WHITE lift on exactly the geometry the
// verified green-stair fix rebuilt — and there is no authored bake to restore anyway. That
// predicate is written as the two class bits rather than as `trans_a >= 1.0` because the class
// bits are what the loader actually sets; `trans_a` being 1 is a consequence. Zero as well on a
// batch with NO authored MOCV at all (`has_vc`), where `vc` is the synthetic `vec4(1.0)` default:
// there is no bake to keep a share of, and a full white `k` there would be an invention.
//
// NOT scaled by `fireLightGain` and NOT flickered: this is not a fire, it is the room's own
// authored light. The `interiorGain` scale IS applied, folded in at pack time exactly as the
// lane's other two inputs (base ambient, per-fixture fill) are — so the Dim preset dims it too and
// the shader carries no second knob.
//
// MEASURED at the owner's live cvars (`interiorGain 0.5`, `interiorAmbient 0.015`,
// `interiorFill 0.08`, `interiorExposure 2.5`, `interiorAttenScale 1.6`, `interiorDaylight 0`) at
// `interiorBakeFloor 0.12` (so `k` = 0.06), at NIGHT:
//   * `g0` one yard inside p1 (TRANS, MOCV (165,160,146) => bake 0.0377):
//       room law 0.0194 -> **0.1075** x tex; the existing 1.5x TRANS floor then displays it at
//       **0.1613** (the floor follows the raised room — it was 0.0291 before).
//   * `g1` one yard the other side (INT, MOCV (119,106,77) => bake 0.0251):
//       0.0215 -> **0.0810** x tex. Room-law ratio across the doorway 1.33x (was 1.11x); the
//       DISPLAYED ratio is 1.99x because the pre-existing TRANS floor still lifts `g0` by 1.5x.
//   * `g3`'s entry floor directly under L9 (1.52 yd, MOCV (250,164,83)): 0.5668 -> 0.6078, **+7.2 %**.
//   * a `g5` hall wall 3 yd from L0 (MOCV (146,113,78)): 0.4192 -> 0.4578, **+9.2 %**.
// i.e. both fixture-lit references move by well under the 15 % bar, and the black band does not.
// (The floor is NOT sun-gated, unlike `enclosed_day_floor`: a bake floor that vanished by day
// would black the same vestibule out at noon. Daylight therefore lifts by the same ~7-9 % on
// fixture-lit interiors, and the `g0` band by ~+20 % through the TRANS blend.)
fn interior_bake_floor(
    vc_rgb: vec3<f32>,
    has_vc: bool,
    class_int: bool,
    class_trans: bool,
) -> vec3<f32> {
    // `has_vc` too: with no MOCV authored, `vc` is the synthetic `vec4(1.0)` default, and lifting
    // by a full white `k` would be inventing a bake rather than restoring a share of one.
    if (!has_vc || (!class_int && !class_trans)) {
        return vec3<f32>(0.0);
    }
    return vc_rgb * (fract(wow_light.sh_c16.w) / BAKE_LANE_SCALE);
}

// MONKEY (enclosed day floor): the SUN'S OWN AMBIENT FLOOR for a room inside a building — the
// daylight that comes through the doorways this renderer cannot locate.
//
// Three seeds now stand real fixtures in a building's authored openings (`lighting::daylight`:
// interior<->exterior portals, EXT-class batches of interior groups, stitched group seams), and
// where an opening IS in the data that is the better answer — a pool at the door falling off
// inward, not a uniform lift. But the shipped corpus keeps rooms whose doorway is in no table at
// all: the Goldshire inn's entry group authors no portal to its shell, no EXT-class batch, no
// vertex within half a yard of the shell that owns its threshold planks, and no localized hot spot
// in its own MOCV bake. There is nowhere to stand a light. What IS known is that the group is a
// room in a building and the sun is up, and the honest rendering of that is "not pitch dark".
//
// The colour is the sky's own AMBIENT band renormalised to unit luminance, so the scalar cvar is
// the floor's luminance outright and the tint is the day's. `sun_w` is `fog_params.z` —
// `sun_shadow_strength(celestial_dir.y)`, exactly 0 at and below the horizon — so the NIGHT LOOK IS
// BIT-IDENTICAL and the ramp rides the same dusk clock as `day_w`, the realtime shadows and the
// daylight fixtures.
//
// It is added at the CALL SITE of `interior_room_light` rather than inside it, deliberately: that
// function is also the exterior lane's `ext_room` term, which would otherwise lift every street
// and outer wall in the world by this constant. And it composes with `interiorGain` by NOT being
// scaled by it — the gain dims candle light, and this is sunlight.
// `record_w` is the RECORD TABLE's `w` column (`recs[word & 0xffff].w`) — where every record bit
// lives — and NOT the vertex word, whose own flag bits are a different space entirely.
fn enclosed_day_floor(record_w: u32) -> vec3<f32> {
    if ((record_w & RECORD_ENCLOSED) == 0u) {
        return vec3<f32>(0.0);
    }
    let k = fract(wow_light.wmo_fog_params.w) / DAYLIGHT_LANE_SCALE;
    let sun_w = clamp(wow_light.fog_params.z, 0.0, 1.0);
    if (k * sun_w <= 0.0) {
        return vec3<f32>(0.0);
    }
    let amb = wow_light.light_ambient.rgb;
    let lum = dot(amb, vec3<f32>(0.2126, 0.7152, 0.0722));
    // A degenerate ambient band (pitch-black zone) has no hue to carry, so fall back to white
    // rather than dividing by nothing.
    let tint = select(vec3<f32>(1.0), amb / max(lum, 1e-4), lum > 1e-4);
    return tint * (k * sun_w);
}

// Vanilla cutout ref (224/255) — wow_model.wgsl's VANILLA_ALPHA_KEY.
const VANILLA_ALPHA_KEY: f32 = 0.8784314;

// MONKEY (torch shadows Phase 1): group 3 — the interior torch depth map + the ≤16-entry fixture
// table. Declared (and sampled) ONLY under TORCH_SHADOWS, which the static_gx pipeline always sets;
// the wow_model.wgsl copy of interior_room_light has no group 3 and must not see this block.
#ifdef TORCH_SHADOWS
struct TorchTable {
    // MONKEY (static torch cache): byte-identical in BOTH shaders and TorchTableUniform.
    // count@0 (16): x high-water slot count, y soft*100, z dynamic/live-bank mask, w reserved.
    // positions@16 (256), view_projs@272 (6144): total 6416 bytes. A mismatch hides buildings.
    // MONKEY (live bank rank): 96 static layers + 48 live layers; matrices stay slot-addressed.
    count: vec4<u32>,
    positions: array<vec4<f32>, 16>,
    view_projs: array<mat4x4<f32>, 96>,
}
@group(3) @binding(0) var torch_depth: texture_depth_2d_array;
@group(3) @binding(1) var torch_samp: sampler_comparison;
@group(3) @binding(2) var<uniform> torch_table: TorchTable;
// Reverse-Z depth bias: nudges the receiver's own depth up so a surface never shadows ITSELF (the
// striped acne). KEEP SMALL — reverse-Z compresses depth far from the torch, so a large constant
// bias there detaches every shadow (0.004 made the whole room read "lit"). 0.001 is the known-good
// value that gave the clean forge-cast floor shadows; the residual acne only showed in the
// worst-case interiorDebug 2 (min over all 4 maps), not the real per-fixture render.
const TORCH_BIAS: f32 = 0.001;
// MONKEY (surface normal offset): sample the map a hand's width OFF the receiving surface, along
// its own normal. **MIRRORED from `terrain.wgsl` and `wow_model.wgsl` (same name, same 0.15) - keep
// the three in sync**; this file was the one receiver still projecting the raw `P`, which is the
// whole of the imp's "boxy" light pool.
//
// WHY A CONSTANT BIAS IS NOT ENOUGH, in numbers. A cube face is 512^2 at `TORCH_FACE_FOV` (pi/2 +
// 0.02), so a texel covers `2*t*tan(45.57 deg)/512 = t/251` yd at ray distance `t`. On a floor `h`
// below the fixture, `r` out from under it (`t = sqrt(r^2+h^2)`), the STORED depth changes across a
// texel by that footprint times the grazing slope `r/h`, and the PCSS kernel's outermost tap sits
// `soft * TORCH_PCSS_MAX = 1.5 * 4 = 6` texels out:
//     kernel depth spread  D = 6 * (t/251) * (r/h)
// while `TORCH_BIAS` in reverse-Z (near 0.1, `cube_view_projs`) is worth `0.001 * t^2 / 0.1 =
// 0.01*t^2` YARDS of separation, and this offset adds `0.15 * (h/t)` (the normal's component along
// the ray - which is exactly why it is taken along the NORMAL and not straight up: on a floor lit
// from a low angle the two agree, on a wall the vertical offset would buy nothing).
//
//   WALL TORCH, h = 3 yd        r=1: D 0.025 vs bias 0.100     r=3: D 0.101 vs 0.180
//                               r=6: D 0.321 vs 0.450          r=9: D 0.680 vs 0.900
//     -> the bias alone clears the kernel by 1.3-1.8x at every radius. Torches were always fine.
//   IMP HAND FLAME, h = 1 yd    r=1: D 0.034 vs bias 0.020     r=2: D 0.107 vs 0.050
//                               r=3: D 0.227 vs 0.100
//     -> the bias LOSES by 1.7-2.3x over the whole pool: every tap past the centre lands on floor
//        texels nearer the fixture than the biased reference, so a fraction of the 4 (or 8) taps
//        fails and the floor self-shadows. The fraction changes with the grazing angle and steps at
//        the CUBE FACE boundaries, which is what turns a round pool into a square one (and, before
//        PCSS widened the kernel, into concentric rings).
//   WITH THIS OFFSET, h = 1 yd  r=1: 0.020+0.106 = 0.126 vs D 0.034  (3.7x margin)
//                               r=2: 0.050+0.067 = 0.117 vs D 0.107  (1.1x)
//                               r=3: 0.100+0.047 = 0.147 vs D 0.227  (still 1.5x short)
//     -> clean out to r ~ 2.1 yd, which is where a 1 yd fixture's own direct term has already
//        fallen to ~36 % of its peak (`INTERIOR_CORE_GAIN/(1+(d/1.75)^2)` at d = 2.3). RESIDUAL,
//        named rather than hidden: a knee-high fixture's OUTER pool can still acne, and the honest
//        cure for that is a slope-scaled bias or a grazing-angle clamp on the PCF radius, not a
//        bigger constant (which detaches every real shadow - see TORCH_BIAS).
//
// MONKEY (slope bias): that RESIDUAL is now fixed - and TWO numbers above are wrong, so do not
// re-derive from them. The table prices depth in yards along the RAY, but the compare reads
// `ndc.z` on a cube FACE, which is a function of the VIEW-SPACE Z (the component along the face
// AXIS). So (a) a floor is on the DOWN face while `r < h`, where its depth is the constant `h` and
// acne is impossible, and crosses to a SIDE face after that, where the gradient is `2A/(f*h)` per
// unit uv and does NOT grow with `r`; and (b) `0.15*(h/t)` is inverted - on a grazing floor this
// offset moves the ray's own floor hit by `0.15*(t/h)`. Re-measured exactly, the offset holds a
// h = 1 yd fixture clean to r = 10.1 yd, not 2.1. `shadow_hook::TORCH_SLOPE_MAX` carries the
// corrected table and the receiver-plane bias that closes the rest.
const TORCH_NORMAL_OFFSET: f32 = 0.15;
// MONKEY (torch caster selection): the live PCF tap-radius scale, unpacked from the table's
// `count.y`'s LOW 16 bits (stored x100 because the row is `vec4<u32>`; the high half is the
// MONKEY (shadow floor) strength). Floored so a zero table cannot collapse
// the kernel to a single texel.
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
#endif

// MONKEY (outdoor torch shadows): the EXTERIOR lane's world-distance fade radius (yd). See
// `shadow_hook::torch_map_shadow`'s `fade` note for why the interior's reverse-Z `ndc.z` fade is
// useless out here (it is already down to 5 % at 10 yd, and a campfire's pool is 15-25). Chosen
// just inside the cube projection's own 48 yd far plane, so a shadow is at full strength across
// the whole pool and has finished fading by the time the fragment leaves the frustum — which is
// what stops the frustum edge from being a visible hard cut.
const TORCH_EXT_FADE_YD: f32 = 44.0;
// MONKEY (outdoor torch shadows): `count.w` bit 0 — the EXTERIOR receiver lane's live gate,
// published by `benilla_app::torch_shadow` (the `exteriorShadows` cvar AND night AND at least one
// promoted exterior fixture). Everything the exterior lane costs hangs off this ONE uniform read,
// so by day / with the cvar off the receivers below are exactly the code they were before.
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
#ifdef TORCH_SHADOWS
    return (torch_table.count.w & TORCH_EXT_LANE) != 0u;
#else
    return false;
#endif
}

// MONKEY (Phase 5): the cube face that contains direction `d` (fixture → fragment) — the major
// axis, signed. The face order is the contract with `benilla_app::torch_shadow::cube_view_projs`:
// 0 +X, 1 −X, 2 +Y, 3 −Y, 4 +Z, 5 −Z. Each face is a 90°(+ε) frustum down that axis, so the face
// holding the major axis always contains the direction.
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

// This fixture's OWN cast shadow: correlate the `wow_light` fixture at `light_pos` to a
// promoted torch (position match within 1 yd), pick the cube face facing the fragment, and sample
// that layer. 1.0 (unshadowed) when no map matches, or when TORCH_SHADOWS is off (wow_model's copy
// of the caller never sets it).
//
// MONKEY (outdoor torch shadows): `fade_radius` selects the FAR FADE, and it is the ONLY difference
// between the interior and exterior lanes here — the correlation, the face pick, the live-bank
// compaction and the slot cross-fade are one body because a shadow map does not know which lane
// promoted it. `0` = the interior/legacy `ndc.z` fade, unchanged bit-for-bit; `> 0` = the exterior
// world-distance fade `1 − smoothstep(0.8R, R, d)` measured from the FIXTURE (not from the camera:
// this is "how far does this fire's shadow carry", a property of the fire).
// MONKEY (slope bias): `N` rides along purely so the projector can bias against the RECEIVER'S
// PLANE. `P` is the already-offset point, `N` the normal it was offset along, and this function
// makes no other use of it - the correlation, the face pick and the fades are unchanged.
fn torch_map_at(light_pos: vec3<f32>, P: vec3<f32>, N: vec3<f32>, fade_radius: f32) -> f32 {
#ifdef TORCH_SHADOWS
    // MONKEY (torch lane perf): the whole table is empty far more often than not (no promoted
    // fixture in range, or the lane switched off while the shader-def is still compiled in). Say so
    // once, up front, so the caller pays one uniform read rather than a loop set-up per fixture.
    if (torch_table.count.x == 0u) {
        return 1.0;
    }
    for (var i = 0u; i < torch_table.count.x; i = i + 1u) {
        // MONKEY (static torch cache): holes and pending uploads never sample stale layers.
        if (torch_table.positions[i].w <= 0.0) { continue; }
        let fixture = torch_table.positions[i].xyz;
        if (distance(fixture, light_pos) < 1.0) {
            let face = torch_face(P - fixture);
            let layer = i * 6u + face;
            // MONKEY (live bank rank): count.z is CPU-ready-filtered; holes must not consume
            // live cubes. Keep the projection on the static slot while compacting depth only.
            let rank = countOneBits(torch_table.count.z & ((1u << i) - 1u));
            let depth_layer = select(layer, 96u + 6u * rank + face, (torch_table.count.z & (1u << i)) != 0u);
            // MONKEY (outdoor torch shadows): negative = "use your own ndc fade" (the interior
            // contract). The exterior weight is computed HERE because only this scope knows both
            // the fixture and the fragment.
            let fade = select(
                -1.0,
                1.0 - smoothstep(0.8 * fade_radius, fade_radius, distance(fixture, P)),
                fade_radius > 0.0,
            );
            let s = shadow_hook::torch_map_shadow(
                torch_table.view_projs[layer], i32(depth_layer), P, N, torch_depth, torch_samp,
                TORCH_BIAS, torch_soft(), fade);
            // MONKEY (torch caster selection): `.w` is the slot's FADE WEIGHT. A slot ramps 0 -> 1
            // over ~1/3 s when it is promoted and 1 -> 0 before it is reused, and `mix` turns that
            // into the shadow appearing/dissolving instead of switching. Without it the selection
            // churn in a candle-dense room (Northshire's 42 candelabra, 5 yd apart) reads as
            // shadows popping on and off as you walk - the bug this lane exists to fix.
            // MONKEY (shadow floor): and `torch_strength()` is the DIRECT-term floor folded
            // into that same weight (see the function). `w * strength` rather than a second `mix`
            // because the two are the same expression.
            return mix(1.0, s, torch_table.positions[i].w * torch_strength());
        }
    }
#endif
    return 1.0;
}

// The INTERIOR lane's call — unchanged behaviour (`fade_radius 0` ⇒ the reverse-Z `ndc.z` fade).
// MONKEY (surface normal offset): `N` is the LIT normal of the receiving fragment, and the offset
// point is what gets projected, face-picked and compared — exactly as `terrain.wgsl`'s
// `torch_terrain_shadow` and `wow_model.wgsl`'s `torch_entity_shadow_at` already do (both offset
// BEFORE `torch_face`, so a fragment right on a cube-face boundary picks the same face its
// neighbour does). See TORCH_NORMAL_OFFSET for the texel-vs-bias arithmetic this is the answer to.
fn torch_surface_shadow(light_pos: vec3<f32>, P: vec3<f32>, N: vec3<f32>) -> f32 {
    return torch_map_at(light_pos, P + N * TORCH_NORMAL_OFFSET, N, 0.0);
}

// MONKEY (outdoor torch shadows): the EXTERIOR lane's call — the campfire/brazier/lamppost pool on
// a WMO's outdoor-class surfaces and on exterior doodads.
// MONKEY (surface normal offset): the same 0.15 yd normal offset as the interior lane and the
// terrain/entity receivers — an outdoor WMO floor under a low fire (a campfire on a porch) has the
// same grazing-angle PCF acne as the imp's floor; see TORCH_NORMAL_OFFSET for the arithmetic.
fn torch_exterior_shadow(light_pos: vec3<f32>, P: vec3<f32>, N: vec3<f32>) -> f32 {
    return torch_map_at(light_pos, P + N * TORCH_NORMAL_OFFSET, N, TORCH_EXT_FADE_YD);
}

// MONKEY (torch debug, interiorDebug 2): the MIN raw depth-map shadow factor over EVERY promoted map
// (ignores the fixture-position match), so the map's actual content is visible as greyscale on the
// floor: all-WHITE = maps empty / projection misses the fragment; uniform GREY = self-shadow/bias;
// SHAPED dark regions = real occlusion is being captured (then the fix is correlation/placement).
// MONKEY (surface normal offset): the overlay projects the SAME offset point the lit path does, or
// it would show acne the render no longer has (and hide acne it does).
fn torch_debug_factor(P: vec3<f32>, N: vec3<f32>) -> f32 {
#ifdef TORCH_SHADOWS
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
        // MONKEY (torch caster selection): the WEIGHTED factor, so the debug view shows what the
        // real render shows - a fading slot greys out here too, and a slot stuck at w = 0 (never
        // ramping in) is visible as a map that contributes nothing.
        s = min(s, mix(1.0, raw, torch_table.positions[i].w));
    }
    return s;
#else
    return 1.0;
#endif
}

// ---- mirrored law (wow_model.wgsl) ----

fn wow_normalize(v: vec3<f32>) -> vec3<f32> {
    let l2 = dot(v, v);
    return select(vec3<f32>(0.0), normalize(v), l2 > 1e-12);
}

// MONKEY (outdoor torch shadows): the ≤`EXT_SEL_K`-nearest EXTERIOR selection, packed into FOUR
// u32s as twelve 8-bit indices (rank 0 in the low byte of `.x`), `EXT_SEL_EMPTY` for an unfilled
// rank.
//
// WHY the selection is packed and TRAVELS from the vertex stage: the exterior point term is chosen
// per VERTEX (the reference FFP's own granularity, and — on a WMO's outdoor-class surfaces —
// anchored at the MCNK CELL so a street and the road it runs into rank the same lights and agree
// at the seam, see `wmo_exterior_point_sum`). The per-FRAGMENT shadowed term must use exactly
// THAT selection: re-ranking per fragment would put a hard line across the cobbles at every cell
// boundary, where the interpolated vertex term has none. So the vertex stage publishes its choice
// and the fragment stage only re-EVALUATES it (and shadows it).
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

// The ranking half of the old `point_light_sum` — the EXTERIOR doodad/MODD-prop family, anchored
// at the receiving unit's own origin. Same tests, same order, same tie handling (strictly-less
// inserts, so an equal distance leaves the earlier table index at the better rank) so the sum can
// be re-evaluated later against the identical choice.
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
        // MONKEY (light lanes): skip INTERIOR fixtures (colour row `.w > 0.5`) — mirrored from
        // wow_model.wgsl. This is the EXTERIOR doodad/MODD-prop family (a WMO surface takes
        // `wmo_exterior_point_sum` or nothing), so a building's own fixtures must not reach it — the
        // warm pool on the grass at the foot of the inn's wall. Skipped before the ≤`EXT_SEL_K`
        // ranking, so an interior fixture cannot take a slot an outdoor fire should have had.
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
// each multiplied by its OWN fixture's cube-map occlusion, so a crate between the fragment and
// campfire A darkens A's term while lamppost B's is untouched. Per fixture, exactly as the interior
// lane is (this is the same argument one lane over: a single scene-wide shadow factor cannot
// express "shadowed from one fire, lit by another", which is what a village square at night
// actually looks like). Reached only under `torch_ext_on()`, so nothing here executes in daylight
// or with `exteriorShadows 0`.
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
        // MONKEY (torch lane perf): same argument as the interior lane's guard one screen up — the
        // occlusion is a factor on a term that is already zero on any surface facing away from the
        // fire, and the scan + four taps are the expensive half of this loop body.
        let ext_w = atten * nl;
        var occ = 1.0;
        if (s < EXT_SEL_SHADOWED && ext_w > TORCH_SKIP_EPS) {
            occ = torch_exterior_shadow(fixture, P, N);
        }
        sum += wow_light.points[2u * idx + 1u].rgb * (ext_w * occ);
    }
    return sum;
}

fn point_light_sum(P: vec3<f32>, N: vec3<f32>, anchor: vec3<f32>) -> vec3<f32> {
    return point_light_eval(point_light_pick(anchor, 0.0), P, N);
}

// MONKEY (wmo exterior points): the EXTERIOR-lane point term for a WMO's OUTDOOR-class surfaces —
// Stormwind's cobbles, its city walls, an inn's porch: every group whose MOGP flags carry `0x48`
// (`WORD_INTERIOR` clear). Until now those took NO point term at all: the vertex stage zeroed
// `point_lit` for every `WORD_WMO` batch, because the reference really does zero it on the surfaces
// it lights with the MOCV bake — but that finding is about INTERIOR groups (wow-re
// `trace-forensics-abbey-interior-d3d` §2 measured abbey rooms). An exterior-class group is drawn
// by the exterior law, the same law terrain is drawn by, and terrain takes the point term. The
// symptom of the over-broad zero: a Trade District wall torch at night lit the NPC beside it (an
// M2, on `wow_model.wgsl`'s lane) and the dirt road at the gate (terrain), but not one cobble of
// the street it hangs over.
//
// It is `terrain.wgsl`'s `point_light_sum` semantics DELIBERATELY, not this file's own:
//  · the anchor is the **MCNK cell centre** ([`mcnk_cell_anchor`]), not the batch's baked
//    placement anchor. A WMO's baked anchor is its PLACEMENT origin — one point for the whole of
//    Stormwind — so a nearest-K ranked from it would commit the same few lights to every street
//    in the city. The 33.33 yd cell is the unit terrain uses, so the road and the street it runs
//    into rank the SAME candidates and agree at the seam, which is the whole requirement.
//  · candidacy is terrain's Chebyshev box `TERRAIN_REACH`, not this file's 48 yd sphere, for the
//    same agreement reason (and because it is the byte-verified gather — see `terrain.wgsl`).
// Cost: one bounded pass over the ≤256-entry table per VERTEX (never per fragment), the same walk
// the doodad lane beside it already pays. Gated only on the table's contents — the exterior lane
// is independent of `interiorLight`, and an empty/daytime table makes it exactly zero. Additive
// into the clamped `primary` sum, so a fully sunlit street cannot brighten past white: daytime is
// materially unchanged, exactly as on terrain.
//
// MIRRORED verbatim in `wow_model.wgsl` (which keeps its own `mcnk_cell_anchor`) — the small
// minority of WMO batches the retained collector declines (env-mapped, depth-flag oddities) falls
// through to the entity path, and a street lit on one pipeline and black on the other is worse
// than either. Keep the two bodies identical.
const WMO_EXT_REACH: f32 = 33.570166;
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
// MONKEY (outdoor torch shadows): the ranking half, split out for the same reason as
// `point_light_pick` — the fragment stage must re-evaluate the VERTEX's choice, not make its own.
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

// MONKEY (dynamic interiors): the per-FRAGMENT room light for interior WMO surfaces AND interior
// props — EVERY in-range light of the room-gated table, no nearest-3 selection (a per-fragment
// selection swap draws a seam where the 4th light drops out). Two terms per light:
//  · DIRECT — the reference's falloff × a WRAPPED Lambert, (N·L + wrap)/(1 + wrap): a hearth
//    sitting at floor level is horizontal to the floor around it, and a hard max(N·L, 0) lights
//    nothing there (the smithy only worked because its forges are raised) — times the fixture's
//    AUTHORED ATTENUATION WINDOW (MONKEY, below).
//  · FILL — a normal-free bounce, colour × K_FILL × (1 − d/reach)²: the "lamps everywhere" glow
//    Blizzard's bake carried, sourced from the room's own fixtures instead of a constant. The
//    colour is normalised for this term (the table commits RAW over-gamut colour × intensity),
//    so one hot forge doesn't wash a whole room.
// MONKEY (light lanes / interior attenuation): the loop is now over the INTERIOR half of the table
// only (colour row `.w > 0.5`), and that same `.w` is each fixture's REACH in yards — its authored
// MOLT attenuation end (M2 sources bucket by intensity), scaled live by `interiorAttenScale`. The
// reach bounds the loop, shapes the fill, and drives the direct window. `interiorAttenScale 0`
// packs the legacy 48 yd instead — under the soft profile below that is the widest, flattest pool
// the lane can make, the nearest thing left to the pre-window flat lane (it is no longer the
// byte-exact restore it was, because the window's SHAPE moved with it).
// `torch_shadow` (the #2 lane, 1.0 when no torch is promoted) shades the direct term only —
// bounce is indirect. A base ambient keeps a fixture-less nook from going black. The knobs are
// LIVE cvars packed into `point_count.yzw` — `.y` base ambient, `.z` fill gain, `.w` exposure
// (the callers' multiplier on the whole budget before the rolloff) — so
// `/script SetCVar("interiorExposure", 2)` retunes a room without a rebuild. The fill is
// half-desaturated: bounce off wood and stone is not candle-orange.
const INTERIOR_WRAP: f32 = 0.5;
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
// MONKEY (portal claims): the exterior-lane room term's distance gate (yd) — see its call site.
const EXT_ROOM_FADE_START: f32 = 60.0;
const EXT_ROOM_FADE_END: f32 = 90.0;
// MONKEY (soft portal claims): 32, not 8 — six 4-word fade records follow the six ids. Keep in
// sync with `benilla_world::lighting::{ROOM_CLAIM_STRIDE, ROOM_CLAIM_FADE, CLAIM_FADE_SCALE}`: a
// stride that disagrees reads every fixture's claims out of a neighbour's record, and every
// building in the frame goes dark (or floods) at once.
const ROOM_CLAIM_STRIDE: u32 = 32u;
const ROOM_CLAIM_MAX: u32 = 6u;
const ROOM_CLAIM_FADE: u32 = 8u;
const CLAIM_FADE_SCALE: f32 = 256.0;
// MONKEY (portal claims): bit 16 of a packed claim word — "the EXTERIOR batch lane may honour this
// claim". Keep in sync with `benilla_world::lighting::CLAIM_EXT_OK`; the group id (`group + 1`, 12
// bits) lives below it, hence the mask.
const CLAIM_EXT_OK: u32 = 65536u;
// MONKEY (soft portal claims): bits 17..=24 of a claim word — the ENTRY weight as a byte. Keep in
// sync with `benilla_world::lighting::CLAIM_ENTRY_SHIFT`.
const CLAIM_ENTRY_SHIFT: u32 = 17u;
const CLAIM_GROUP_MASK: u32 = 0xffffu;
// MONKEY (room gate): may fixture `i` light a surface of room `(room_inst, room_group)`? Three
// fail-OPEN arms, deliberately, because a wrong "no" is a black room while a wrong "yes" is only
// the leak we already had: a fragment with no room key (`room_group == 0`), a fixture that claims
// nothing (count 0 — an unclaimed MOLT fixture, or the whole gate switched off from `cvars`, which
// simply packs every light that way), and any fixture whose claim list overflowed the table. Only
// a fixture that positively claims SOME room of THIS building can be refused.
//
// MONKEY (portal claims): `strict` inverts all three of those arms, and it is what the EXTERIOR
// batch lane passes. That lane is an ADDITION — an exterior-class group inside a building (the
// inn's shell, its basement stairwell) is drawn by the night-sky law and gains its room's candle
// light — so its wrong answers are the other way round: a wrong "yes" would light a whole city
// street from a tavern candle through a wall, while a wrong "no" only leaves a surface exactly as
// it renders today. So it fails CLOSED: no room key, no claims, another building's claim, or a
// claim not marked [`CLAIM_EXT_OK`] (a district-scale shell) all mean "not this fixture".
//
// MONKEY (soft portal claims): the answer is a WEIGHT in [0, 1], not a yes/no — 1 admits the
// fixture in full, 0 refuses it, and a PORTAL claim lands in between. The binary form is what put
// the reported scar on the Lion's Pride Inn's upstairs floor: two groups meet mid-plank at the
// doorway, the fixture was admitted at full strength on one side and refused outright on the
// other, so the boards changed brightness along a straight line. Every fail-open/fail-closed arm
// keeps its old meaning (1.0 / 0.0), so `interiorRoomGate 0` — which packs every fixture with
// count 0 — is still exactly "ungated".
//
// The weight of a portal claim is `entry * (1 − smoothstep(0, radius, max(|P − center| − slack,
// 0)))`: `entry` at the doorway itself, gone where the fixture's remaining reach runs out.
// `slack` is the doorway's own bounding-sphere radius, and it is what makes the threshold
// continuous: on the near side the fixture is admitted at 1 by CONTAINMENT, and on the far side
// every fragment within the door frame measures distance 0 and is admitted at `entry` — which the
// claim rule sets to exactly the weight the near side has there (1 off a base claim, the first
// hop's decayed value off a first hop). Away from the frame the far side falls off smoothly, but
// there the two sides are separated by a WALL rather than by continuous floor, so nothing reads as
// a step.
fn interior_room_admits(i: u32, room_inst: u32, room_group: u32, strict: bool, P: vec3<f32>) -> f32 {
    if (room_group == 0u) {
        return select(1.0, 0.0, strict);
    }
    let base = ROOM_CLAIM_STRIDE * i;
    let n = room_claims[base + 1u];
    if (n == 0u) {
        return select(1.0, 0.0, strict);
    }
    // A different building's fixture never lights this one — one compare rejects a neighbouring
    // inn's whole table, which is why the instance sits in the head word.
    if (room_claims[base] != room_inst) {
        return 0.0;
    }
    for (var k = 0u; k < min(n, ROOM_CLAIM_MAX); k = k + 1u) {
        let claim = room_claims[base + 2u + k];
        if ((claim & CLAIM_GROUP_MASK) == room_group && (!strict || (claim & CLAIM_EXT_OK) != 0u)) {
            // A group appears at most once in a claim list (the rule dedupes), so the first match
            // is the only one — no need to keep looking for a better weight.
            let f = base + ROOM_CLAIM_FADE + 4u * k;
            let packed = room_claims[f + 3u];
            let radius = f32(packed & 0xffffu) / CLAIM_FADE_SCALE;
            if (radius <= 0.0) {
                return 1.0; // a HARD claim: the fixture stands in this room, or MOLR names it
            }
            let center = vec3<f32>(
                bitcast<f32>(room_claims[f]),
                bitcast<f32>(room_claims[f + 1u]),
                bitcast<f32>(room_claims[f + 2u]),
            );
            let slack = f32(packed >> 16u) / CLAIM_FADE_SCALE;
            let entry = f32((claim >> CLAIM_ENTRY_SHIFT) & 0xffu) / 255.0;
            let d = max(distance(P, center) - slack, 0.0);
            return entry * (1.0 - smoothstep(0.0, radius, d));
        }
    }
    return 0.0;
}
// This region's building identity (the cell uniform's `.w` lane; 0 on a terrain cell). Carried as
// a NUMBER, not as bits: an entity index is a small integer, and `f32::from_bits(12345)` is a
// DENORMAL float that a driver flushing denormals to zero would silently collapse to 0 — every
// building would then share identity 0. An f32 represents every integer below 2^24 exactly, which
// is far above any live entity index, so the round trip is lossless and FTZ-proof; the packer
// refuses to gate anything at or above that bound (see `assemble_region`).
fn gx_room_inst() -> u32 {
    return u32(cell.origin.w);
}
// This item's room key — `group + 1`, 0 when the item names no room (see RECORD_ROOM_SHIFT).
fn gx_room_group(word: u32) -> u32 {
    return (recs[word & 0xffffu].w >> RECORD_ROOM_SHIFT) & RECORD_ROOM_MASK;
}
// MONKEY (room gate): `room_inst`/`room_group` are the RECEIVING surface's room (see
// `interior_room_admits`). `wow_model.wgsl`'s copy takes the same room arguments and stubs the
// predicate to `true` — it has neither the claim binding nor a per-fragment room key, which is
// also why it needs no `strict` (MONKEY, portal claims: only this file has an exterior WMO lane).
// **The PROFILE block above stays byte-identical between the two files; the signature does not.**
fn interior_room_light(
    P: vec3<f32>,
    N: vec3<f32>,
    room_inst: u32,
    room_group: u32,
    strict: bool,
) -> vec3<f32> {
    let count = u32(wow_light.point_count.x);
    let k_fill = wow_light.point_count.z;
    var direct = vec3<f32>(0.0);
    var fill = vec3<f32>(0.0);
    for (var i = 0u; i < count; i = i + 1u) {
        let color_lane = wow_light.points[2u * i + 1u];
        // MONKEY (light lanes): only a fixture that CLAIMS a room lights this room (keep in sync
        // with wow_model.wgsl). `.w` is 0 on every exterior source, so a campfire burning just
        // outside the door stops reaching the floor inside it — the leak's other direction.
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
        // MONKEY (soft portal claims): `w` is the claim's WEIGHT, and it multiplies BOTH terms
        // below. `wow_model.wgsl`'s copy of this loop keeps the old boolean stub (entities are
        // ungated — it has neither the claim binding nor a per-fragment room key), which is the
        // one place the two files' loop bodies deliberately differ; the PROFILE constants above
        // stay byte-identical.
        let w = interior_room_admits(i, room_inst, room_group, strict, P);
        if (w <= 0.0) {
            continue;
        }
        // MONKEY (soft falloff): `.w` is ALSO this fixture's EFFECTIVE RADIUS `R` in yards — the
        // authored MOLT `attenuation_end` (or the M2 intensity bucket) already multiplied by the
        // live `interiorAttenScale` at pack time. It replaces the flat 48 yd candidacy radius in
        // `pos_range.w` on this lane, which is why the inn's 10 candles read as one uniform wash
        // while the smithy's 3 forges read fine.
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
        // Normalised for BOTH terms: the table commits RAW over-gamut colour × intensity, and the
        // smithy's three forges (1.4, 0.87, 0.4) drove the direct term to a washed-out white while
        // the inn's unit candles sat where they should. The hue is what survives.
        let c_norm = c / max(1.0, max(c.r, max(c.g, c.b)));
        // Phase 1: this fixture's OWN cast shadow, sampled from its down-looking depth map
        // (`torch_surface_shadow` correlates the fixture to its promoted map by position; 1.0 when
        // it was not promoted). Per-fixture, so a pillar between the fragment and torch A darkens A's
        // term without touching torch B's — the occlusion that makes an interior read
        // lit-and-shadowed instead of flat. (`static_gx.wgsl` only; the wow_model copy of this
        // function keeps the 1.0 `torch_shadow_for` stub — it has no group 3.)
        // MONKEY (torch lane perf): the shadow sample is MULTIPLIED into the direct term, so
        // where that term is already nothing the table scan and its four comparison taps buy
        // nothing. And it very often is: `window` is exactly 0 past the fixture's reach, `nl` is 0
        // on a surface facing away even under the wrap, and `w` is the claim weight a portal fade
        // has taken to 0. The scan is by POSITION over up to sixteen slots, so this is the
        // difference between "every interior fixture in range pays a scan on every fragment it
        // touches" and "only the ones actually lighting it do". A branch rather than a `select`
        // deliberately: the whole point is to NOT execute the taps.
        let direct_w = atten * nl * window;
        var s = 1.0;
        if (direct_w * w > TORCH_SKIP_EPS) {
            s = torch_surface_shadow(pos_range.xyz, P, N);
        }
        direct += c_norm * direct_w * s * w;
        // MONKEY (soft falloff): the fill's profile is UNCHANGED in FORM — `(1 − d/r)²`, which is
        // exactly `interior_window(d, r, 1.0)` — and only its radius moved, from R to
        // `INTERIOR_FILL_SPAN·R`. That is the whole "gentle wash between the pools": at the fixture
        // it is still 1 (so `interiorFill` keeps its tuned meaning) and it decays to 0 at 2R with a
        // vanishing derivative, so the floor half-way between two candles is dim, not black.
        let c_fill = mix(c_norm, vec3<f32>(dot(c_norm, vec3<f32>(0.299, 0.587, 0.114))), 0.5);
        // The DOMINANT fixture's fill, not the SUM: an ambient room glow must not scale with the
        // candle count, or a dense room (the inn's ~10 fixtures vs the smithy's 3) piles fill up
        // until the rolloff saturates every surface to a flat white. `max` keeps the nearest/
        // brightest fixture's glow and leaves the direct term to carry the per-fixture relief.
        fill = max(fill, c_fill * (k_fill * interior_window(d, fill_yd, INTERIOR_FILL_POW) * w));
    }
    // Fill is INDIRECT bounce, so it is not shadowed; direct already carries each fixture's shadow.
    // MONKEY (shell candle add): STRICT callers want only the claimed DIRECT + FILL. The shell
    // already has sky ambient; returning the raw budget here also avoids adding then subtracting
    // that floor (cancellation would lose weak fixtures). Non-strict room lighting is unchanged.
    if (strict) {
        return direct + fill;
    }
    return direct + fill + vec3<f32>(wow_light.point_count.y);
}

// MONKEY (trans blend continuity): the day blend's "never darker than the room law" guard, as a
// C1-CONTINUOUS maximum instead of a per-channel `max`.
//
// THE BUG IT FIXES, measured on the Lion's Pride Inn's vestibule floor (`GoldshireInn` g0 batch b0 -
// 8 TRANS triangles - rendered off `benilla-extract wmolights --verts 0` at 0.05 yd with this file's
// exact interior-lane arithmetic at the owner's live cvars):
//
//   `max(room_rgb, rgb)` is evaluated PER CHANNEL, so the room law and the sunlit reference cross at
//   a DIFFERENT place in R, in G and in B. Each crossing is continuous in value and BROKEN in its
//   first derivative, and each is therefore a Mach line - three of them, at three different places,
//   with the colour stepping warmer across each (the fragments between two loci have one channel on
//   the reference and two on the room law, which is a hue no law in the shader ever asks for).
//   The BLUE locus is the reported scar: a line from the door's south jamb running diagonally into
//   the floor and dying near the middle -
//       08:20  (20.30, -1.23) -> (15.16, -4.33)   jump 0.1595/yd = 32.5 %/yd at a level of 0.491
//       noon   (19.96, -0.83) -> (15.68, -4.68)   jump 0.1594/yd = 28.6 %/yd at a level of 0.557
//   with GREEN a second, shorter line nearer the door and RED crossing only at 08:20 (which is why
//   the scar is worse in the morning than at noon). The wedge between the blue and green loci reads
//   lighter and less warm than the floor beyond it (mean 0.621 lum / R:B 1.206 against 0.415 / 1.280
//   at 08:20) - the "wedge on one side" of the report.
//   That these are TRUE C1 breaks and not the light pool's own curvature is settled by halving the
//   sample step: a break's |d2| scales as 1/h (2.25 -> 4.49 -> 8.93 /yd^2 at 0.05 / 0.025 / 0.0125 yd)
//   while a smooth arc's does not.
//
// WHY THIS SHAPE. Three laws were measured against the same floor:
//   (1) pick by LUMINANCE (one achromatic locus, all three channels switch together) - still a hard
//       C1 break, and the colour step across it gets WORSE, not better (hue step 16.7 / 20.0 against
//       the shipped 14.5 / 12.1 at 08:20 / noon): concentrating three small hue jumps into one big
//       one is the wrong direction.
//   (2) THIS - a quadratic smooth-max of half-width `TRANS_SMAX_W`. The selector disappears outright:
//       the function is C1 everywhere, so there is no locus left to draw a line at. Costs at most
//       0.0087 (08:20) / 0.0085 (noon) of display luminance against the shipped result - under 1 %,
//       and under a single 8-bit code over most of the floor.
//   (3) drop the `max` and `mix` straight to the reference - the same clean derivative, but it
//       re-opens the subtraction the `max` was added against: it renders this floor up to 0.0811 x tex
//       (11.0 %) DARKER at 08:20, 4.7 % at noon and 3.0 % at 20:00, because the room law is above the
//       reference over the whole area the p0 daylight fixture actually reaches.
//
// It OVERSHOOTS rather than undershoots, deliberately and by at most `w/4` (0.0125): a C1 function
// that equals `max` outside a band must do one or the other inside it, and undershooting would dip
// BELOW `room_rgb` - which is precisely the "the blend may only ever brighten" invariant the outer
// `max` and the night floor exist to hold. `smax >= max(a, b) >= room_rgb` still holds here.
//
// NIGHT IS BIT-IDENTICAL and not by argument: the whole selector is behind `day_w`, and
// `mix(a, b, 0.0)` is `a * 1 + b * 0` = `a` exactly, whatever `b` is. Verified numerically against the
// shipped law at `sun_w = 0` - max |difference| 0.000e+00 on every fragment of the floor.
const TRANS_SMAX_W: f32 = 0.05;
fn trans_smax(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let h = clamp(0.5 + 0.5 * (b - a) / TRANS_SMAX_W, vec3<f32>(0.0), vec3<f32>(1.0));
    return a + (b - a) * h + TRANS_SMAX_W * h * (vec3<f32>(1.0) - h);
}

// MONKEY (shell candle add): candle irradiance lights the TEXTURE, not the authored MOCV shadow
// bake. Give it the room lane's exposure rolloff, then fade the RESULT over 60..90 yd: fading the
// raw budget before exp would change the pool's brightness profile with camera distance. The sky,
// exterior points and SIDN keep their original law; only their remaining display headroom bounds
// the addition. A hard zero guard keeps unclaimed/distant facades bit-identical, even by daylight.
// `texture_rgb` is the sample BEFORE the MOCV fold: dividing folded black by 1/255 cannot recover
// a texture on a zero-bake vertex. The reference path keeps its fold/divide arithmetic untouched.
fn shell_candle_add(
    sky_rgb: vec3<f32>,
    texture_rgb: vec3<f32>,
    room_raw: vec3<f32>,
    far: f32,
) -> vec3<f32> {
    if (far <= 0.0 || all(room_raw <= vec3<f32>(0.0))) {
        return sky_rgb;
    }
    let candle = vec3<f32>(1.0) - exp(-room_raw * wow_light.point_count.w);
    return clamp(sky_rgb + texture_rgb * candle * far, vec3<f32>(0.0), vec3<f32>(1.0));
}

// ---- the pass ----

struct GxVertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) word: u32,
    @location(4) anchor: vec3<f32>,
    // MOCV or the baked constant tint; white where the batch authors none (WORD_HAS_VC).
    @location(5) color: vec4<f32>,
}

struct GxVsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) word: u32,
    @location(4) point_lit: vec3<f32>,
    @location(5) color: vec4<f32>,
    // MONKEY (outdoor torch shadows; ext light k8): WHICH ≤`EXT_SEL_K` exterior table entries
    // `point_lit` was summed from, packed 8 bits each across FOUR u32s (see `EXT_SEL_NONE` /
    // `ext_sel_get`). FLAT, because it is a choice, not a quantity — interpolating packed indices
    // would produce a different, meaningless one. `EXT_SEL_NONE` on every lane that takes no
    // exterior point term (interior WMO surfaces, interior props, the collapsed exile vertex),
    // which makes the fragment lane a no-op there by construction. Widened from ONE u32 of three
    // 10-bit ranks to TWO u32s of eight 8-bit ranks and then to FOUR u32s of twelve — three extra
    // interstage components in all (21 for this struct, against a 60-component limit) — see
    // `EXT_SEL_K`.
    @location(6) @interpolate(flat) ext_sel: vec4<u32>,
}

@vertex
fn vertex(v: GxVertex) -> GxVsOut {
    var out: GxVsOut;
    // Exile kill bit (record w bit 0): every vertex of the item collapses to one point, so its
    // triangles are zero-area and rasterize nothing; triangles never span items.
    if ((recs[v.word & 0xffffu].w & 1u) != 0u) {
        out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        out.world_position = vec4<f32>(0.0);
        out.world_normal = vec3<f32>(0.0, 1.0, 0.0);
        out.uv = vec2<f32>(0.0);
        out.word = v.word;
        out.point_lit = vec3<f32>(0.0);
        out.color = vec4<f32>(1.0);
        out.ext_sel = EXT_SEL_NONE;
        return out;
    }
    // Camera-relative for f32 precision: the recentred vertex plus (cell origin - camera).
    var p_cam = v.position + (cell.origin.xyz - view.world_position);
    var world = v.position + cell.origin.xyz;
    // MONKEY (wind): placement anchor is already baked per vertex. Only classified leaf-card
    // batches carry the bit; animated doodads were rejected before this retained path.
    if ((v.word & WORD_FOLIAGE_WIND) != 0u) {
        let offset = wind_hook::tree_offset(world, v.anchor, view.world_position, wow_light.monkey);
        world += offset;
        p_cam += offset;
    }
    out.world_position = vec4<f32>(world, 1.0);
    let view_rot = mat3x3<f32>(
        view.view_from_world[0].xyz,
        view.view_from_world[1].xyz,
        view.view_from_world[2].xyz,
    );
    out.position = view.clip_from_view * vec4<f32>(view_rot * p_cam, 1.0);
    // Authored MOBA batch order (record y, 0 on cells): a later coplanar batch wins reverse-Z
    // GreaterEqual in any draw order, because the draw sort does not keep authored order.
    out.position.z *= 1.0 + f32(recs[v.word & 0xffffu].y) * 1.1920929e-7;
    out.world_normal = v.normal;
    out.uv = v.uv;
    out.word = v.word;
    out.color = v.color;
    // The ≤`EXT_SEL_K`-nearest selection, anchored at the PLACEMENT origin exactly like the entity
    // path (the baked per-vertex anchor — 1429's parity note; the blob path coarsened this).
    // WMO surfaces take ZERO point lights — the entity path zeroes them in the vertex stage
    // (wow-re trace-forensics-abbey-interior-d3d §2: zero on every observed WMO surface) —
    // and so do interior M2 props (B4): their group-MOLR point lobes are folded into the
    // per-item SH probe at spawn, the entity path's own vertex-stage zeroing.
    if ((v.word & WORD_WMO) != 0u && (v.word & WORD_INTERIOR) != 0u) {
        // INTERIOR WMO surfaces stay zero here, the reference's own vertex-stage zeroing. MONKEY
        // (dynamic interiors): they light from the live torches PER-FRAGMENT in the fragment
        // stage (`interior_room_light`) — a per-vertex sum on a WMO's huge floor triangles is
        // Gouraud: straight-edged wedges, and a torch mid-triangle lights nothing.
        out.point_lit = vec3<f32>(0.0);
        out.ext_sel = EXT_SEL_NONE;
    } else if ((v.word & WORD_WMO) != 0u) {
        // MONKEY (wmo exterior points): an EXTERIOR-class group (MOGP `& 0x48`) is a street, a
        // courtyard, a porch — drawn by the exterior law, so it takes the exterior point term the
        // terrain beside it takes. Its own anchor is the placement origin (one point for all of
        // Stormwind), so `wmo_exterior_point_sum` re-anchors on the MCNK cell; see its comment.
        // Per-vertex like terrain's, not per-fragment: WMO surfaces are MOCV-baked per vertex, so
        // they carry the tessellation a Gouraud term needs, and this stays one bounded walk.
        // MONKEY (outdoor torch shadows): the pick is published so the fragment stage can shadow
        // this same selection at night without re-ranking (see `EXT_SEL_NONE`). One table walk
        // still, not two — the sum was always `eval(pick(...))`, it is just no longer inlined.
        let sel = wmo_exterior_pick(world);
        out.ext_sel = sel;
        out.point_lit = point_light_eval(sel, world, v.normal);
    } else {
        // MONKEY (ext light k12): `box = 0` — an exterior doodad/MODD prop is its own draw unit and
        // ranks from its baked PLACEMENT origin, exactly as it did before the widening.
        let sel = point_light_pick(v.anchor, 0.0);
        out.ext_sel = sel;
        out.point_lit = point_light_eval(sel, world, v.normal);
    }
    return out;
}

@fragment
fn fragment(in: GxVsOut) -> @location(0) vec4<f32> {
    // The hard farclip wall: per-pixel planar eye-Z, the same plane as the entity path.
    let eye_z = -(view.view_from_world * vec4<f32>(in.world_position.xyz, 1.0)).z;
    if (wow_light.fog_params.w > 0.0 && eye_z > wow_light.fog_params.w) {
        discard;
    }
    // The sampler follows the wrap flags; a mixed batch clamps its clamped axis a half texel in.
    // Every sample runs unconditionally: implicit derivatives need uniform control flow.
    var base = vec4<f32>(1.0);
    let wrap_x = (in.word & WORD_WRAP_X) != 0u;
    let wrap_y = (in.word & WORD_WRAP_Y) != 0u;
    if ((in.word & WORD_TEXTURED) != 0u) {
        let layer = i32(recs[in.word & 0xffffu].x);
        let dims = vec2<f32>(textureDimensions(tex_array).xy);
        let inset = 0.5 / dims;
        var uv_mixed = in.uv;
        if (!wrap_x) {
            uv_mixed.x = clamp(uv_mixed.x, inset.x, 1.0 - inset.x);
        }
        if (!wrap_y) {
            uv_mixed.y = clamp(uv_mixed.y, inset.y, 1.0 - inset.y);
        }
        // The render-scale LOD bias: bevy's material path applies it for free, this lane by hand.
        let c_repeat = textureSampleBias(tex_array, samp_repeat, in.uv, layer, view.mip_bias);
        let c_clamp = textureSampleBias(tex_array, samp_clamp, in.uv, layer, view.mip_bias);
        let c_mixed = textureSampleBias(tex_array, samp_repeat, uv_mixed, layer, view.mip_bias);
        if (wrap_x && wrap_y) {
            base = c_repeat;
        } else if (!wrap_x && !wrap_y) {
            base = c_clamp;
        } else {
            base = c_mixed;
        }
    }
#ifdef GX_CUTOUT
    if (base.a < VANILLA_ALPHA_KEY) {
        discard;
    }
#endif
    // Both faces light from the submitted normal (no GL_LIGHT_MODEL_TWO_SIDE in the reference);
    // this pipeline never negates back faces, so it has no front-face select like wow_model.wgsl.
    let n_lit = wow_normalize(in.world_normal);
    // MONKEY (outdoor torch shadows): the EXTERIOR point term, cast-shadowed at night.
    //
    // `in.point_lit` is the Gouraud (per-vertex) exterior sum, and it stays the ONLY thing this
    // lane reads by day. After dark the same entries are re-evaluated PER FRAGMENT with each
    // one's own cube-map occlusion folded in (`point_light_eval_shadowed`), and the two are blended
    // by `night_w = 1 − sun_shadow_strength`. Two things fall out of writing it as a blend rather
    // than a swap:
    //   · `fog_params.z` is EXACTLY 1.0 whenever the sun is above the daylight threshold
    //     (`global_light::sun_shadow_strength`, a smoothstep that saturates), so `night_w` is
    //     exactly 0 and the `if` is not entered at all — daylight is the same instructions and the
    //     same bits it was, not "a mix that ought to round back".
    //   · at dusk the shadow arrives on the same clock the sun shadows leave on, and the
    //     Gouraud→per-fragment change of the term itself arrives with it instead of snapping.
    // `torch_ext_on()` is the CPU's one-bit verdict (`exteriorShadows` && night && ≥1 promoted
    // exterior fixture), so with the cvar off the branch is dead too.
    var point_lit = in.point_lit;
    let ext_night_w = select(0.0, clamp(1.0 - wow_light.fog_params.z, 0.0, 1.0), torch_ext_on());
    if (ext_night_w > 0.0) {
        point_lit = mix(
            in.point_lit,
            point_light_eval_shadowed(in.ext_sel, in.world_position.xyz, n_lit),
            ext_night_w,
        );
    }
    let L = -normalize(wow_light.light_sun.xyz);
    let ndotl = max(dot(n_lit, L), 0.0);
    let lit_nl = clamp(
        wow_light.light_ambient.rgb + wow_light.light_diffuse.rgb * ndotl,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    // The retained path uses the same Bevy directional shadow map as terrain and entity models.
    // `static_gx` used to have no mesh-view shadow bindings, which made every Stormwind WMO act
    // as if shadows were disabled even though the caster map was populated.
    // MONKEY (shadow hook): EXTERIOR surfaces take the directional sun shadow (fetch + edge/night
    // fade, via `benilla::shadow_hook`) — now with the same fades as terrain + models. INTERIOR
    // batches (`WORD_INTERIOR`) take the point-light (torch) shadow instead (#2): no sun reaches a
    // sealed room, but a promoted torch does. Both are no-ops when their light source is absent.
    // MONKEY (moon shadows): the NIGHT arm rides the SAME fetch and the SAME `WORD_INTERIOR` gate —
    // 1.0 (inert) on every interior batch, all day, and with the feature off. Interiors stay
    // untouched at both ends of the clock: a sealed room has no sky, so it has no moon either.
    var world_shadow = 1.0;
    var world_moon = 1.0;
    // GFX (moonlight): the moon term for EXTERIOR surfaces (zero indoors, by day and when off).
    var moon_add = vec3<f32>(0.0);
    if ((in.word & WORD_INTERIOR) == 0u) {
        let view_z = (view.view_from_world * in.world_position).z;
        let cam_dist = distance(in.world_position.xyz, view.world_position.xyz);
        let terms = shadow_hook::realtime_shadow_moonlit(
            in.world_position,
            n_lit,
            view_z,
            cam_dist,
            wow_light.wmo_fog_params.z,
            shadow_hook::sun_shadow_w(wow_light.fog_params.z),
            shadow_hook::moon_shadow_w(wow_light.fog_params.z),
            wow_light.monkey.misc.w,
        );
        world_shadow = terms.x;
        world_moon = terms.y;
        if (wow_light.monkey.moon.w > 0.0) {
            moon_add = moonlight_hook::moon_light(n_lit, wow_light.monkey.moon) * terms.z;
        }
    } else {
        world_shadow = shadow_hook::torch_shadow(in.world_position, n_lit);
    }
    let shadow_term = mix(SHADOW_SUN_FLOOR, 1.0, world_shadow);
    // Keep ambient energy when the realtime map blocks the sun. The retained pass also carries
    // authored fixture/probe lighting, which must not be darkened by a directional shadow. The
    // `world_shadow < 0.999` arm keeps `worldShadows 0` (no shadow-mapped light) and a fully
    // sunlit fragment byte-identical: `ambient + (lit − ambient) × 1.0` is not guaranteed to
    // round back to `lit`.
    var lit_nl_shadowed = select(
        lit_nl,
        clamp(
            wow_light.light_ambient.rgb + (lit_nl - wow_light.light_ambient.rgb) * shadow_term,
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        ),
        world_shadow < 0.999,
    );
    // MONKEY (moon shadows): shadow the unclamped exterior SKY, ambient AND directional.
    // Points join later, before the final saturation, so a hot torch stays hot beneath a canopy.
    // world_moon is exactly one indoors and when off; leave those arithmetic paths untouched.
    if (world_moon < 1.0) {
        lit_nl_shadowed = (wow_light.light_ambient.rgb + wow_light.light_diffuse.rgb * ndotl)
            * world_moon;
    }
    // GFX (moonlight): the moon adds onto the exterior night; `moon_add` is exactly zero otherwise.
    if (wow_light.monkey.moon.w > 0.0) {
        lit_nl_shadowed += moon_add;
    }

    // The order-2 SH basis products over the fragment normal — shared by the exterior
    // doodad lobe and the interior-prop probe lane (wow_model.wgsl computes them once too).
    let quad = vec4<f32>(n_lit.x * n_lit.y, n_lit.y * n_lit.z, n_lit.z * n_lit.z, n_lit.x * n_lit.z);
    let x2y2 = n_lit.x * n_lit.x - n_lit.y * n_lit.y;
    // A colour-less batch takes a constant 1.0, not interpolated white: interpolating a constant
    // attribute gives 1.0±ε, and base × 0.99999994 rounds about half the pixels one byte down.
    let has_vc = (in.word & WORD_HAS_VC) != 0u;
    let vc = select(vec4<f32>(1.0), in.color, has_vc);
    let folded = select(base.rgb, base.rgb * vc.rgb, has_vc);
    var rgb: vec3<f32>;
    if ((in.word & WORD_WMO) != 0u) {
        // ---- WMO surfaces: wow_model.wgsl's is_wmo branch ----
        let interior = (in.word & WORD_INTERIOR) != 0u;
        let class_int = (in.word & WORD_CLASS_INT) != 0u;
        let class_trans = (in.word & WORD_CLASS_TRANS) != 0u;
        let trans_a = vc.a; // 1.0 where no MOCV is authored
        // WINDOW (MOMT 0x20), interior drawer only: GL_LIGHT0 becomes the Direct/Ambient
        // midpoint pair, ambient +16/255 saturating (0x6d37e0).
        let window_mid = 0.5 * (wow_light.light_ambient.rgb + wow_light.light_diffuse.rgb);
        // The window's directional half rides the sun, so it takes the shadow term (×1.0 exact
        // when the caster is off — no identity guard needed on a pure multiply).
        let lit_window_shadowed = clamp(
            window_mid + vec3<f32>(16.0 / 255.0) + window_mid * ndotl * shadow_term,
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
        let lit_int_base = select(lit_nl_shadowed, lit_window_shadowed, (in.word & WORD_WINDOW) != 0u);
        // The interior BATCH-CLASS lanes (trace-forensics-abbey-interior-d3d §2): INT = unlit
        // (the bake IS the room's light), TRANS = the per-vertex MOCV-alpha lit↔bake lerp,
        // EXT = plain lit_nl. Exterior groups take lit_nl at sun-scale 1 (prog 198/VS 151 —
        // no terrain shade, no SH lobe).
        var lit_wmo_interior = vec3<f32>(1.0);
        if (class_trans) {
            lit_wmo_interior = mix(vec3<f32>(1.0), lit_int_base, trans_a);
        } else if (!class_int) {
            lit_wmo_interior = lit_int_base;
        }
        // EXTERIOR surfaces are the ones the realtime map can shadow (the fetch above is
        // gated on !interior, so `lit_nl_shadowed` collapses to `lit_nl` for interior
        // fragments anyway). Taking plain `lit_nl` here left every Stormwind street and
        // wall unshadowed while the caster map was fully populated.
        let lit_wmo = select(lit_nl_shadowed, lit_wmo_interior, interior);
        // SIDN night glow (MOMT 0x10): the authored emissive × the live night fraction, an
        // EMISSION term — inside the clamped sum, never MOCV-multiplied; dead on the unlit
        // INT lane, TRANS-weighted by the lit-pass alpha (wmo-interior-night-light §4).
        let rec = recs[in.word & 0xffffu];
        let sidn_rgb = vec3<f32>(
            f32(rec.z & 0xffu),
            f32((rec.z >> 8u) & 0xffu),
            f32((rec.z >> 16u) & 0xffu),
        ) / 255.0;
        var sidn_w = 1.0;
        if (interior) {
            if (class_trans) {
                sidn_w = trans_a;
            } else if (class_int) {
                sidn_w = 0.0;
            }
        }
        var sidn_e = sidn_rgb * (wow_light.grade.x * sidn_w);
        // MONKEY (fix-post): the bloom window gain lifts only this light term, before fog (it used
        // to multiply the whole fogged batch, so fogged buildings glowed as fog x 2.35 at night).
        if ((in.word & WORD_WINDOW) != 0u) {
            sidn_e = emissive_hook::emissive_boost(
                sidn_e, emissive_hook::EMISSIVE_WMO_WINDOW, wow_light.light_diffuse.w, 1.0);
        }
        // MONKEY (ext-class night law): plenty of geometry INSIDE a building is authored
        // EXTERIOR-class — the Goldshire inn's whole 57.7 yd shell is one group (`upstairs`,
        // MOGP 0x0a09, and it holds the stair down to the cellar), its east stairwell annex is
        // another — and the exterior law is the DAY/NIGHT SKY law: after dark that is `light_ambient`
        // + `light_diffuse`, which the night sky makes GREEN. A fragment of one of those groups
        // therefore renders sky-green one plank away from a candle-lit interior group, which is the
        // reported hard break at the cellar stair and (with the shell's outer face) on the floors
        // above it. The claim-gated `ext_room` term below already reaches those surfaces, but it
        // ADDS to the sky rather than replacing it, so the green survived underneath the candles.
        //
        // The fix is a per-batch LAW BLEND rather than another additive term: a building-scale
        // exterior group is one the room gate is willing to light (the same
        // `ext_building_scale`/`CLAIM_EXT_OK` eligibility, so no group can land between the two
        // rules), so after dark it should be lit the way the room next to it is — the interior
        // lane's ambient floor, its claim-gated fixtures, its exposure rolloff, and NO sky term.
        // By day nothing moves: `night_w` is 0 whenever the sun is up, so the authored sunlit look
        // of every porch, courtyard and city street is bit-for-bit what it was.
        //
        // `fog_params.z` is `sun_shadow_strength(celestial_dir.y)` — 1 in daylight, ramping to 0
        // as the sun reaches the horizon (see `global_light.rs`), i.e. exactly the "how much of the
        // sky law is real right now" number the sun shadows already fade on. Reusing it keeps the
        // law swap on the same dusk/dawn clock as everything else that fades with the sun.
        let interiors_on = wow_light.wmo_fog_params.w > 0.5;
        let ext_night = !interior && interiors_on && (rec.w & RECORD_EXT_NIGHT) != 0u;
        // MONKEY (ext-class night law, REVERTED for GROUPS 2026-09-09): swapping a whole
        // exterior-class SHELL group onto the room law after dark painted every building FACADE
        // black at dusk — the inn's and the smithy's outer walls are that shell, no candle claims
        // them, and `interiorAmbient` (0.02) is the whole of what the room law had for them while
        // the ground beside them was still sunset-lit (user screenshots, 20:40 server time). The
        // shell keeps the reference sky law day and night; its claimed fixtures still arrive
        // through the strict `ext_room` term below. The BATCH-level half of the fix (an EXT-class
        // batch inside an INTERIOR group — the cellar stair) lives in `day_w` and is untouched.
        // MONKEY (shell candle add): `ext_night` is the building eligibility gate as well as the
        // `interiorDebug 4` overlay key; despite its historical name it applies by DAY too.
        // MONKEY (portal claims): the EXTERIOR-class group's own room light. Bug B's second
        // cause: plenty of groups INSIDE a building are authored EXTERIOR-class (MOGP `& 0x48`) —
        // the Goldshire inn's whole shell is one, and its basement stairwell reads night-sky GREEN
        // a step away from candle-lit rooms — and the interior lane never runs on them, so no
        // fixture could reach them at all. It does now, through the SAME soft profile and the SAME
        // claim table, in STRICT mode: only a fixture that positively claims THIS group of THIS
        // building, and only when that claim is exterior-lane eligible (`CLAIM_EXT_OK` — a
        // building's shell is, a city's district shell is not; see
        // `benilla_world::lighting::LIT_ROOM_EXT_DENY`). Stormwind's streets are unaffected twice
        // over: their own torches are EXTERIOR-lane sources, which this loop skips, and the
        // district shells they belong to carry no eligible claim.
        //
        // MONKEY (shell candle add): STRICT now returns DIRECT + FILL without the room ambient.
        // Carry it separately from `primary`: multiplying this live light through MOCV suppresses
        // it on baked-dark reveals, and omitting the exposure rolloff dims even an unbaked shell
        // beside its room. Goldshire g4 actually has NO MOCV (vc = 1); its measured doorway also
        // changes normal across g3/g4, so matching the law does not promise identical irradiance.
        //
        // COST: this is a per-FRAGMENT walk of the point table on surfaces that never had one —
        // and exterior WMO geometry is most of a city's screen. So it is distance-gated, with a
        // fade rather than a cliff: past `EXT_ROOM_FADE_END` the loop is not entered at all. The
        // widest pool the lane can make is `1.5 x INTERIOR_LEGACY_REACH`, but every shipped fixture
        // sits at `R = 7..15` yd (fill 10..22), so the gate costs nothing visible and is the
        // difference between paying for the effect where it can be seen and paying for it across
        // the whole skyline.
        var ext_room_raw = vec3<f32>(0.0);
        var ext_room_far = 0.0;
        let ext_room_d = distance(in.world_position.xyz, view.world_position.xyz);
        if (ext_night && ext_room_d < EXT_ROOM_FADE_END) {
            ext_room_raw = interior_room_light(
                in.world_position.xyz,
                n_lit,
                gx_room_inst(),
                gx_room_group(in.word),
                true, // fail CLOSED: an unclaimed exterior surface renders exactly as before
            );
            ext_room_far = 1.0 - smoothstep(EXT_ROOM_FADE_START, EXT_ROOM_FADE_END, ext_room_d);
        }
        // GL_COLOR_MATERIAL: MOCV multiplies the lit terms INSIDE the clamp, emission adds
        // beside. MONKEY (wmo exterior points): `in.point_lit` is zero on INTERIOR groups (the
        // reference's own vertex-stage zeroing — they light per fragment below) and carries the
        // exterior nearest-3 point term on EXTERIOR-class ones, which is what makes a street
        // torch reach the cobbles. The clamp is why it costs nothing in daylight.
        let primary = clamp(
            vc.rgb * (lit_wmo + point_lit) + sidn_e,
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
        // The entity path's algebra: (tex×vc)/max(vc, 1/255) is not tex in floats; keep it.
        let tex_rgb = folded / max(vc.rgb, vec3<f32>(1.0 / 255.0));
        rgb = tex_rgb * primary;
        rgb = shell_candle_add(rgb, base.rgb, ext_room_raw, ext_room_far);
        // MONKEY (shell candle add): only actual interior groups enter the room-law blend. Shells
        // keep the sky plus the strictly claimed texture addition above, without a room ambient
        // floor or an enclosed day floor spilling across their whole building-sized extent.
        if (interior && interiors_on) {
            // MONKEY (dynamic interiors #A, `interiorLight` on): the WHOLE interior group — INT, TRANS and EXT batch
            // classes alike — lights from the LIVE room torches instead of the MOCV bake. Gating
            // this on `class_int` left every TRANS/EXT batch (most of a floor near a doorway) on
            // the baked `vc.rgb × lit` path above: the "old baked light/shadow still there".
            //
            // Per-FRAGMENT (`interior_room_light`, every room light): a per-vertex sum on a WMO's
            // huge floor triangles is Gouraud — straight-edged wedges, a torch mid-triangle lights
            // nothing — the same artefact shape as Blizzard's per-vertex bake. The torch shadow
            // (#2 lane, 1.0 when no torch is promoted) shades the torch term only.
            //
            // Daylight keeps the AUTHORED batch-class weight, so a doorway still takes the sun:
            // INT = none, TRANS = the per-vertex MOCV alpha lerp, EXT = full `lit_int_base`.
            // MOCV.rgb (the baked light + shadow) is what gets dropped; its alpha is the exterior
            // blend weight, not a shadow, and stays.
            //
            // Soft rolloff instead of a hard clamp: `1 − exp(−x)` maps the ILLUMINATION into [0,1)
            // with derivative 1 at 0 (dim nooks untouched) and a smooth asymptote — a bright forge
            // never draws the iso-brightness contour a hard `clamp` did. `tex × illum` can't clip.
            // SIDN rides inside as the emission it is. Ambient 0.15 fills the dark (cvar/slider
            // once the look is confirmed).
            let room = interior_room_light(
                in.world_position.xyz,
                n_lit,
                gx_room_inst(),
                gx_room_group(in.word),
                // The interior lane's gate fails OPEN (see `interior_room_admits`).
                // MONKEY (ext-class night law): the EXTERIOR-class lane keeps the STRICT gate it
                // has always used here. Failing open on this lane would hand every un-keyed
                // exterior group in the frame the whole fixture table the moment the sun set — a
                // barn lit by a city's candles. Failing closed leaves an unclaimed one at exactly
                // the ambient floor, which is what an unlit interior nook reads as, and is the
                // "no sky" half of the fix on its own.
                ext_night,
            );
            var day_w = 0.0;
            if (class_trans) {
                day_w = trans_a;
            } else if (!class_int) {
                day_w = 1.0;
            }
            // MONKEY (ext-class night law): …and the authored weight is scaled by the SUN. This is
            // the second half of the same bug, one lane over, and it is what actually paints the
            // Lion's Pride Inn's cellar stair sky-green: the stair is NOT an exterior group (its
            // 700 vertices are group 10, `room03`, MOGP 0x2805 — interior), but its MOBA batches
            // are EXT-CLASS, and an EXT-class batch of an interior group took `day_w = 1`
            // unconditionally, i.e. it rendered on the reference path with `lit_int_base` = the
            // sun/sky ambient+diffuse. That is the right law at noon (a doorway takes the sun) and
            // it is the NIGHT SKY after dark — green, next to the INT-class wall beside it on the
            // candle lane, with the break exactly where the batch class changes.
            //
            // The weight's own rationale is what fixes it: it exists so the interior side of a
            // threshold matches the EXTERIOR group a step away. That group is now on the room law
            // after dark too (`ext_night` above), so matching it at night means the room law here
            // as well — the same argument, evaluated at the right time of day. `fog_params.z` is
            // exactly 1.0 while the sun is up, so `day_w` and therefore every daylight fragment of
            // every interior group is bit-for-bit unchanged.
            let sun_w = clamp(wow_light.fog_params.z, 0.0, 1.0);
            day_w = day_w * sun_w;
            // MONKEY (ext-class night law): `in.point_lit` joins the rolloff. It is a HARD ZERO on
            // every interior batch (the vertex stage zeroes it for `WORD_INTERIOR`), so this is an
            // exact no-op for the lane that already ran here — and on an ext-class batch it is the
            // exterior point term, i.e. the street torch hanging on the wall being lit. Without it
            // the blend would take that torch away from every building facade the moment it swapped
            // laws, which is a new seam where the wall meets the lit cobbles.
            // MONKEY (enclosed day floor): the day's ambient floor joins the room's own budget,
            // INSIDE the rolloff — so it saturates with everything else instead of stacking on top
            // of a lit room, and a fixture-lit corner by day reads brighter than an unlit one by
            // exactly as much as the rolloff still has room for. Zero on every batch that is not a
            // room in a building, and zero at night, so both of those are bit-identical.
            // MONKEY (bake floor): the batch's own MOCV bake, at `interiorBakeFloor x interiorGain`,
            // joins the same budget — see `interior_bake_floor` for the whole argument and the
            // measured before/after at the owner's cvars. Inside the rolloff for the same reason
            // the day floor is: a lit room absorbs it, an unlit one is carried by it.
            let illum = vec3<f32>(1.0)
                - exp(
                    -(room
                        + enclosed_day_floor(rec.w)
                        + interior_bake_floor(vc.rgb, has_vc, class_int, class_trans)
                        + point_lit
                        + sidn_e)
                        * wow_light.point_count.w,
                );
            // BLEND toward the reference's own lit result (`rgb` above — MOCV × the exterior law)
            // by the authored batch-class weight; never ADD daylight on top of the room light,
            // which left the interior side of every threshold brighter than the exterior floor a
            // step away (the hard line at the smithy door). At the portal (weight 1) this IS the
            // exterior group's law, so the seam closes by construction, day or night; deep inside
            // (weight 0) it is pure room light. EXT-class batches (1) stay on the reference path.
            //
            // MONKEY (trans night floor): …and the blend may only ever BRIGHTEN, and it carries a
            // FLOOR. Both halves are one rule — **the room lane must never render a threshold batch
            // darker than the reference law's own guaranteed minimum for that fragment** — and both
            // are needed, because the Lion's Pride Inn's front door is dark in two different ways.
            //
            // MEASURED (`benilla-extract wmolights`, GoldshireInn, with the new MOBA batch-class
            // table): the front-door vestibule is group **g0** — the unnamed group the MOGN table
            // prints as `-`, portalled to the ext* porch `room04` (p0) on one side and to the room
            // beyond (p1) on the other, and the ONLY group in the whole WMO with TRANS batches
            // (3/3, MOCV alpha 13..255, mean 111..131 ⇒ `trans_a ≈ 0.44..0.51`). It is NOT the group
            // named `entry` (g3 — 13/13 INT, and portalled only to the cellar and the kitchen, so
            // nothing enters the building through it). That misidentification is why this was
            // measured as "13/13 INT, no TRANS" once before. g0 is what `interiorDebug 4` paints
            // YELLOW-GREEN in the doorway, framed by the BLUE ext* porch and backed by GREEN INT.
            //
            // (1) At NIGHT the blend is already off (`day_w = trans_a · sun_w`, and
            //     `sun_shadow_strength` is exactly 0 at and below the horizon — in Elwynn that is
            //     from ~20:30 on, so it was 0 in BOTH of the user's 20:40 and after-dark
            //     screenshots). What is left is the room lane, and no fixture reaches g0: all ten
            //     MOLT fixtures are ≥ 13 yd away with their windows closed, and its only claim is a
            //     faded portal hop, so the budget collapses to the bare `interiorAmbient` floor —
            //     `1 − exp(−0.0075 × 4) = 0.0296 × tex`, i.e. black, against `0.73 × tex` for the
            //     candle-lit floor a step away. A 25× step, which is the reported hard rectangle.
            //     The FLOOR fixes that: a TRANS batch's reference value is `vc × ((1 − trans_a) +
            //     trans_a · sky)`, so `vc × (1 − trans_a)` is what the reference renders it at under
            //     a COMPLETELY BLACK SKY — its guaranteed minimum, the artist's own "the doorway is
            //     the light" bake share, and the one term the live-fixture lane threw away with the
            //     rest of the bake. Restoring it puts the band at `0.324 × tex`: dimmer than the lit
            //     room (0.44×), brighter than an unlit corner, no rectangle.
            // (2) In the dusk WINDOW where `sun_w` is between 0 and 1 (Elwynn ~19:19→20:30) the
            //     blend runs toward a reference that is ALREADY DARKER than the room law — the
            //     night sky under `nightGain` — so it actively subtracts light from exactly this
            //     band: at `sun_w = 0.2` it drags `0.19` down to `0.14`. `max(room_rgb, rgb)`
            //     forbids that: the mix now runs between the room result and the BRIGHTER of the
            //     two, so it can add the sky's contribution and never take the room's away. This
            //     also makes "the enclosed day floor must be on both sides" automatic — the floor
            //     lives in `room_rgb`, and the result is now `>= room_rgb` by construction.
            //
            // SCOPE, by construction rather than by a new flag:
            //   · INT batches — `day_w` is 0 and `class_trans` is false, so BOTH halves are exact
            //     no-ops. That is the overwhelming majority of interior geometry, and dynamic
            //     interiors keeps every bit of it.
            //   · EXT-class batches of an interior group (the cellar stair the sun-scaled `day_w`
            //     was introduced for) — the loader forces their MOCV alpha to 1.0, so their bake
            //     share `1 − trans_a` is exactly 0; they keep the sun-gated blend and CANNOT get the
            //     night sky back through the floor. The verified green-stair fix is untouched.
            //   · DAYLIGHT — with the sun up the reference is the sun law and the room lane carries
            //     `enclosed_day_floor`, so both are far above the bake share and `max` is a no-op:
            //     the same fragment at noon computes 0.4476 before and 0.4476 after. At the very top
            //     of the dusk ramp (sun_w still 1, the sky already warm-dim) the `max` can bite by
            //     ~0.0003 — a 0.07 % brightening of a band that the next minute falls off a cliff,
            //     which is the trade this whole comment is about.
            let room_rgb = tex_rgb * illum;
            // MONKEY (portal bleed): …and the floor is a RELATIVE lift now, not an absolute display
            // value. The bake share below is `vc x (1 - trans_a)` — what the REFERENCE renders this
            // fragment at under a black sky — and the room lane's whole premise is that the bake's
            // LEVEL is wrong (it was authored for a different global exposure; the live fixtures
            // decide). What survives that premise is the bake's RELATIVE statement: "this threshold
            // is brighter than the room around it". So the floor is capped at
            // `TRANS_NIGHT_FLOOR_LIFT x the room result` and carries the ROOM's own hue, which fixes
            // the reported band in both of the ways it was wrong.
            //
            // MEASURED at the owner's live cvars (`interiorGain 0.5`, `interiorAmbient 0.015`,
            // `interiorFill 0.08`, `interiorExposure 2.5`, `interiorAttenScale 1.6`,
            // `interiorDaylight 0`) on the Lion's Pride Inn's vestibule `g0`, one yard inside p1,
            // at night, with the portal bleed in:
            //   room law here 0.0194 x tex; its INT neighbour g1 one yard the other side 0.0215.
            //   UNCAPPED bake floor 0.324..0.357 (the three TRANS batches' mean MOCV) = **15-18x the
            //   neighbour**, and neutral grey where the neighbour is candle-warm. That is the flat
            //   grey band in the 20:50 and 23:59 screenshots, exactly: the floor had become the
            //   brightest surface in the entrance and the only unwarm one.
            //   CAPPED at 1.5x: 0.0291, i.e. 1.35x the neighbour, in the neighbour's own colour.
            // The 25x black-hole step the floor was introduced against is gone for a different
            // reason — the "candle-lit floor a step away" it was measured against (0.73 x tex) was
            // an `interiorExposure 4` / `interiorGain 0.7` number, and at 2.5/0.5 that same floor
            // renders 0.019. The floor's own justification evaporated with the cvars; what remains
            // of its job is done by the bleed (absolute level) and by this cap (the doorway is a
            // little brighter than its room), and neither can produce a 15x step at any cvar setting
            // because one is bounded by the neighbour and the other by 1.5x the fragment's own room.
            //
            // 1.5 rather than 2: a threshold plainly reads as the brightest thing in an unlit
            // vestibule at half again, and at low absolute levels a 2x step across a batch-class
            // boundary starts to read as an edge rather than as a doorway. Where the bake share is
            // small the `min` keeps taking it, so a nearly-opaque TRANS batch is unchanged, and an
            // EXT-class batch of an interior group (`trans_a` forced to 1, bake share exactly 0)
            // still gets nothing at all — the verified green-cellar-stair fix is untouched.
            //
            // BY DAY it is a no-op twice over, as before: at noon this fragment's room law is 0.276,
            // the blend toward the sunlit reference takes it to 0.562, and the floor is
            // `min(0.324, 1.5 x 0.276) = 0.324` — under the blend, so `max` keeps the blend.
            let trans_floor_lift = 1.5;
            let trans_night_floor = select(
                vec3<f32>(0.0),
                min(
                    // `primary` with `lit_int_base := 0` — literally the reference's own expression
                    // evaluated against a black sky, so the bake share can never exceed what the
                    // reference renders, at any hour, and needs no tuning constant of its own.
                    tex_rgb
                        * clamp(
                            vc.rgb * (1.0 - trans_a) + sidn_e,
                            vec3<f32>(0.0),
                            vec3<f32>(1.0),
                        ),
                    room_rgb * trans_floor_lift,
                ),
                class_trans,
            // MONKEY (trans floor sun gate): x (1 - sun_w). The floor was meant as a NIGHT term and
            // its comment above calls it a day no-op, but the evaluator that found the diagonal
            // (MONKEY trans blend continuity) showed it is not: at noon `max(blend, floor)` flips to
            // the floor from ~5.2 yd inside the vestibule door inward (0.3395 -> 0.3475, a
            // non-monotone step), and the selector locus is a near-VERTICAL C1 break of ~0.12/yd
            // running the full width of the floor about 1.5 yd inside p1 - the same class of line
            // as the diagonal, parallel to the threshold. Gating it on `1 - sun_w` (the exact mirror
            // of `day_w = trans_a * sun_w`) makes the day no-op true by construction: the floor is
            // exactly 0 while the sun is up, ramps in over the same dusk clock the blend ramps out
            // on, and is bit-identical at night (`sun_w = 0`). By day the bake floor
            // (`interior_bake_floor`, inside the rolloff, not sun-gated) already carries a
            // fixture-starved TRANS room, so nothing goes dark: the inner end of the vestibule at
            // noon moves from 0.3475 to the room law (~0.33), a 5 % drop over its last yard.
            ) * (1.0 - sun_w);
            // MONKEY (trans blend continuity): `trans_smax` replaces the per-channel `max` here, and
            // nothing else about this line moves - see the function for the three-locus scar it
            // removes, the two laws it was measured against, and the night bit-identity.
            rgb = max(mix(room_rgb, trans_smax(room_rgb, rgb), day_w), trans_night_floor);
            let idbg = u32(max(wow_light.wmo_fog_params.w - 1.0, 0.0) + 0.5);
            if (idbg == 2u) {
                rgb = vec3<f32>(torch_debug_factor(in.world_position.xyz, n_lit));
            } else {
                rgb = shadow_hook::interior_debug_override(idbg, rgb, in.world_position, n_lit);
            }
        }
        // MONKEY (ext-class night law): `interiorDebug 4` — the LANE MAP. Every WMO fragment is
        // painted by which lighting law its GROUP falls under, and nothing else is touched
        // (terrain, M2 props and entities keep their real shading), so one screenshot of any
        // building answers "which of these three populations is this surface in?" — the question
        // every seam in this system reduces to, and the one the offline `wmolights` group table can
        // only answer by bbox arithmetic.
        //   GREEN family  interior-class group (MOGP & 0x48 == 0) — always on the room lane, shaded
        //                   by MOBA BATCH class because that is its own seam: pure green = INT
        //                   (never took the sky), yellow-green = TRANS (the MOCV-alpha lerp),
        //                   MINT = an EXT-class batch of an interior group — the population that
        //                   used to render by the night SKY inside a lit room (the cellar stair).
        //   BLUE          exterior-class at BUILDING scale — the sky lane by day, the room lane
        //                   after dark (this fix).
        //   RED           exterior-class shell/district — the sky lane, always; the room gate
        //                   refuses its claims (`LIT_ROOM_EXT_DENY`).
        // A seam between two like-coloured fragments is a CLAIM problem (which fixtures each side
        // admits); a seam between two different colours is a LAW problem. Read off the record bit
        // rather than `ext_night`, so the map is the same with the sun up and with `interiorLight`
        // toggling only whether the overlay runs at all.
        let lane_dbg = u32(max(wow_light.wmo_fog_params.w - 1.0, 0.0) + 0.5);
        if (lane_dbg == 4u) {
            if (interior) {
                if (class_int) {
                    rgb = vec3<f32>(0.0, 1.0, 0.0);
                } else if (class_trans) {
                    rgb = vec3<f32>(0.55, 1.0, 0.0);
                } else {
                    rgb = vec3<f32>(0.0, 1.0, 0.75);
                }
            } else if ((rec.w & RECORD_EXT_NIGHT) != 0u) {
                rgb = vec3<f32>(0.1, 0.35, 1.0);
            } else {
                rgb = vec3<f32>(1.0, 0.0, 0.0);
            }
        }
    } else if ((in.word & WORD_INTERIOR) != 0u) {
        // ---- interior M2 props: wow_model.wgsl's interior-prop branch ----
        // The spawn-folded SH probe (MODD ambient, fixed-axis diffuse, the group's MOLR lobes)
        // over the fragment normal; its soft wrap is the reference's, not a hard max(N·L, 0).
        let probe = 7u * ((recs[in.word & 0xffffu].w >> 1u) & 0x1fffu);
        let n1 = vec4<f32>(n_lit, 1.0);
        let lit_prop = clamp(
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
        let primary = clamp(lit_prop + point_lit, vec3<f32>(0.0), vec3<f32>(1.0));
        rgb = folded * primary;
        // MONKEY (dynamic interiors #A): interior props take the SAME live room light as the
        // surfaces around them (`interior_room_light`) instead of the baked probe — on the probe a
        // tavern's tables and barrels stayed bake-bright inside a torch-lit room ("the .m2s ignore
        // the light"). Same soft rolloff as the surface lane. UNCONDITIONAL for the prove-out; the
        // probe path above is the faithful baked result the cvar will restore.
        if (wow_light.wmo_fog_params.w > 0.5) {
            // MONKEY (room gate): a WMO prop takes its OWNER GROUP's key too, where the bake
            // could resolve one (a referrer set naming exactly one group); an ADT map doodad and a
            // multi-room prop pass 0 and stay ungated.
            let room = interior_room_light(
                in.world_position.xyz,
                n_lit,
                gx_room_inst(),
                gx_room_group(in.word),
                false, // the interior lane's gate fails OPEN (see `interior_room_admits`)
            );
            rgb = folded * (vec3<f32>(1.0) - exp(-room * wow_light.point_count.w));
            let idbg = u32(max(wow_light.wmo_fog_params.w - 1.0, 0.0) + 0.5);
            if (idbg == 2u) {
                rgb = vec3<f32>(torch_debug_factor(in.world_position.xyz, n_lit));
            } else {
                rgb = shadow_hook::interior_debug_override(idbg, rgb, in.world_position, n_lit);
            }
        }
    } else {
        // ---- exterior ADT doodads and exterior MODD props ----
        // Model2.bls order-2 SH sun lobe at intensity 0.5 Shaded (MCSH), 1.0 Matte, 2.5 Lit.
        // Deviation: the `min(I, 1)` cap, as in wow_model.wgsl, since lifting it takes sun-facing
        // surfaces past 1.0; Matte keeps its own bit so it stays 1.0 if the cap goes.
        let shade_t = select(1.0, 0.0, (in.word & WORD_SHADE_LIT) != 0u);
        let intensity = min(
            select(mix(2.5, 0.5, shade_t), 1.0, (in.word & WORD_MATTE) != 0u),
            1.0,
        );
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
        // The SH lobe's own ambient rows (the .w column) are the doodad lane's ambient share:
        // subtract-and-restore around the shadow term so only the directional part of the lobe
        // dims. Guarded like `lit_nl_shadowed` so the no-shadow state stays byte-identical.
        let sun_ambient = vec3<f32>(
            wow_light.sh_c10_r.w,
            wow_light.sh_c10_g.w,
            wow_light.sh_c10_b.w,
        );
        let lit_doodad_plain = clamp(sun_lobe, vec3<f32>(0.0), vec3<f32>(1.0));
        var lit_doodad = select(
            lit_doodad_plain,
            clamp(
                sun_ambient + (sun_lobe - sun_ambient) * shadow_term,
                vec3<f32>(0.0),
                vec3<f32>(1.0),
            ),
            world_shadow < 0.999,
        );
        // MONKEY (moon shadows): retained ADT doodads and exterior MODD props select this SH
        // branch, not lit_nl_shadowed. Shadow the whole sky lobe before points/clamp here too.
        if (world_moon < 1.0) {
            lit_doodad = sun_lobe * world_moon;
        }
        // GFX (moonlight): and the retained doodads take the moon term too.
        if (wow_light.monkey.moon.w > 0.0) {
            lit_doodad += moon_add;
        }
        // Sun disabled (light_sun.w) falls back to the FFP matte, like the entity path.
        let lit = select(lit_nl_shadowed, lit_doodad, wow_light.light_sun.w > 0.5);
        // FFP combine: the light sum saturates FIRST, the texture (× the baked constant
        // tint, when authored) modulates the clamped result. Statics: inst_tint identity,
        // no highlight, no SIDN.
        let primary = clamp(lit + point_lit, vec3<f32>(0.0), vec3<f32>(1.0));
        rgb = folded * primary;
    }
    // Unlit (M2 UNLIT 0x01, WMO UNLIT on an exterior group): texture × vertex colour, no light.
    if ((in.word & WORD_UNLIT) != 0u) {
        rgb = folded;
    }
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
    // MONKEY (wet): rain on sky-exposed surfaces (wet_hook.wgsl); interior, unlit and dry = untouched.
    let wet = wet_hook::wet_surface(rgb, n_lit, in.world_position.xyz,
        // MONKEY (fix-wet): lit WMO windows (WORD_WINDOW) are not darkened by rain either.
        select(1.0, 0.0, (in.word & (WORD_INTERIOR | WORD_UNLIT | WORD_WINDOW)) != 0u) * wet_open,
        wow_light.monkey);
    rgb = wet.albedo;
    if (wet.boost > 0.0) {
        rgb = min(rgb + wet_hook::wet_sheen(wet.boost, n_lit, normalize(view.world_position - in.world_position.xyz),
            L, wow_light.light_diffuse.rgb, wow_light.fog_color.rgb, world_shadow), vec3<f32>(1.0));
    }
    // Planar eye-Z fog; other fog modes belong to blends never admitted here. The interior triple
    // keys on the per-frame record bit, not `WORD_INTERIOR`: the client sets it per group under
    // `[0xca7f00]` (`0x6b5190` for surfaces, `0x6b62e0` for the group's doodads).
    var fog_color = wow_light.fog_color;
    var fog_span = wow_light.fog_params.xy;
    if ((recs[in.word & 0xffffu].w & 16384u) != 0u) {
        fog_color = wow_light.wmo_fog_color;
        fog_span = wow_light.wmo_fog_params.xy;
    }
    if (fog_color.w > 0.5 && (in.word & WORD_FOG_OFF) == 0u) {
        // MONKEY (p0 fog hook): the shared fog law (fog_hook.wgsl); classic is bit-identical.
        rgb = fog_hook::apply_fog(rgb, fog_color.xyz, fog_span, eye_z, in.world_position.xyz,
            view.world_position, true, wow_light.monkey);
    }
    // MONKEY (post): SIDN/window light crosses 1.0 only at night and when bloom is armed; the
    // gain is applied to `sidn_e` above (MONKEY fix-post).
    // Gamma-space output; alpha pinned 1.0, every draw here is opaque.
    return vec4<f32>(rgb, 1.0);
}
