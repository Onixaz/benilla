#define_import_path benilla::enhanced_water

// Ported from WarcraftXL (https://github.com/WarcraftXL) by iThorgrim — module wxl-experimental-water, shaders/Surface.ps.hlsl, sea/Spectrum.*, sea/Ocean.*.
// ENHANCED WATER - the optional water module (Video options -> Water Quality; see WATER.md).
//
// Credits: original project WarcraftXL (`wxl-experimental-water`, https://github.com/WarcraftXL),
// author iThorgrim. Its author permits reuse of that code in this module provided the author and
// the original project are named; this notice is that attribution and must stay with the module.
// A block that ports WarcraftXL code says so where it stands and is listed in WATER.md.
// Ported from WarcraftXL (https://github.com/WarcraftXL) by iThorgrim — module wxl-experimental-water, shaders/Surface.ps.hlsl, render/Refraction.cpp, render/Noise.cpp, sea/Shore.hpp.
// MONKEY (p0 credits): every WarcraftXL-derived file is listed in THIRD-PARTY.md (benilla root).
//
// This file is the whole module on the shader side. `liquid.wgsl` (upstream's liquid renderer)
// carries three hooks only: the import, `water_swell` in its vertex stage, and a branch at the top
// of its fragment stage that returns `enhanced_water` when `water_active()` (already fogged).
// Classic never enters this file, and must stay byte-identical to upstream.
//
// It owns its own bindings in the liquid material's group, next to upstream's (90, 100-102):
//   103  the opaque scene depth, resolved after the main opaque pass (`liquid/scene_depth.rs`)
//   104  `WaterParams`, the module's own uniform (`benilla_assets::WaterUniform`)
//   105  the shared global-light buffer again, as the rows this module reads (the same buffer
//        upstream binds at 90 - read-only storage may be bound twice)
//   106  the opaque scene COLOUR, copied beside the depth: what the refraction looks through

#import bevy_pbr::mesh_view_bindings::{view, globals}
// MONKEY (p0 MonkeyFrame): the programme block's struct, mirrored after the point table.
#import benilla::monkey_frame
// MONKEY (rainshelter): the shelter-grid lookup the surface receivers share.
#import benilla::wet_hook
// MONKEY (p0 fog hook): the one distance-fog law every receiver calls.
#import benilla::fog_hook

@group(#{MATERIAL_BIND_GROUP}) @binding(103) var scene_depth: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(106) var scene_colour: texture_2d<f32>;

struct WaterParams {
    // x = quality (0 Classic, 1 Enhanced, 2 High); y = wave energy (ocean 1.0, ADT inland 0.18,
    // WMO exterior 0.26, WMO interior 0.08); z = pinned capture time; w = clock enable.
    mode: vec4<f32>,
    // x = which renderer (0 ADT MCLQ, 1 WMO exterior, 2 WMO interior); y = ocean; z = fullbright;
    // w = this surface may take the WMO INTERIOR fog block (upstream's kind.z).
    lane: vec4<f32>,
    sky_zenith: vec4<f32>,  // linear RGB, LightIntBand 2
    sky_horizon: vec4<f32>, // linear RGB, LightIntBand 6
    celestial: vec4<f32>,   // xyz toward the visible body; w = 0 sun, 1 white moon
};
@group(#{MATERIAL_BIND_GROUP}) @binding(104) var<uniform> water: WaterParams;

// MONKEY (water): the shared global light (`lighting::global_light`), rows 0-20 + the point-light
// table, is deliberately a row view of the exact terrain/model ABI. Bevy's composable-module
// writer rejects a second partial struct view because its unused members require substitution.
@group(#{MATERIAL_BIND_GROUP}) @binding(105) var<storage, read> water_light: array<vec4<f32>>;

// MONKEY (integration): the p0 MonkeyFrame block as seen through the water lane's row view — it
// starts right after the 21 fixed rows + the 512-row point table (row 533; buffer 8528 -> 8784 B).
const MONKEY_ROW: u32 = 533u;
fn water_monkey() -> monkey_frame::MonkeyFrame {
    var m: monkey_frame::MonkeyFrame;
    m.fog_a = water_light[MONKEY_ROW + 0u];
    m.fog_b = water_light[MONKEY_ROW + 1u];
    m.fog_c = water_light[MONKEY_ROW + 2u];
    m.fog_d = water_light[MONKEY_ROW + 3u];
    m.wind_a = water_light[MONKEY_ROW + 4u];
    m.wind_b = water_light[MONKEY_ROW + 5u];
    m.wet_a = water_light[MONKEY_ROW + 6u];
    m.misc = water_light[MONKEY_ROW + 7u];
    for (var i = 0u; i < 8u; i++) { m.benders[i] = water_light[MONKEY_ROW + 8u + i]; }
    // GFX (moonlight): row 16.
    m.moon = water_light[MONKEY_ROW + 16u];
    return m;
}

// MONKEY (rainshelter): the rain-occlusion grid right after the MonkeyFrame block
// (`weather/shelter.rs`): two header rows, then the packed cell words, four to a row.
const SHELTER_ROW: u32 = 550u;
fn water_shelter_word(i: u32) -> u32 {
    return bitcast<u32>(water_light[SHELTER_ROW + 2u + i / 4u][i % 4u]);
}
// 1 open .. 0 under a bridge or a roof.
fn water_rain_open(world_pos: vec3<f32>) -> f32 {
    let hdr = water_light[SHELTER_ROW];
    let cfg = water_light[SHELTER_ROW + 1u];
    let st = wet_hook::shelter_taps(world_pos, hdr, cfg);
    if (st.live <= 0.0) {
        return 1.0;
    }
    return 1.0 - wet_hook::shelter_amount(st, water_shelter_word(st.idx.x),
        water_shelter_word(st.idx.y), water_shelter_word(st.idx.z), water_shelter_word(st.idx.w),
        world_pos.y, cfg);
}

// What the fragment stage hands over: the three varyings this module reads.
struct WaterFragment {
    clip_position: vec4<f32>,
    world_position: vec4<f32>,
    // MONKEY (reviewfix): world XZ before `water_swell`, so fragment waves share its source domain.
    source_xz: vec2<f32>,
    // The authored per-vertex swatch depth (UV1.x): ocean byte/255, about 148 yd at 1.0.
    depth: f32,
    // MONKEY (water): an interior WMO pool's authored MOMT diffuse colour; white on other lanes.
    colour: vec4<f32>,
    // Upstream's per-surface room-fog lane (MeshTag bit 30), to pick the same fog block it would.
    room_fog: u32,
};

// The fog this surface stands in: rgb = the fog colour, w = how much of the surface survives it
// (1 = no fog). The same law as upstream's `apply_fog`, returned as a factor instead of applied,
// because the module must fog ONLY its own surface terms: the refracted bed and the reflected
// scenery come out of the scene copy, which the terrain and model shaders have already fogged
// (fogging them again made shallow water foggier than the dry sand beside it).
fn water_fog(world_pos: vec3<f32>, room_fog: u32) -> vec4<f32> {
    var colour = water_light[4];
    var span = water_light[5].xy;
    if water.lane.w > 0.5 && room_fog != 0u {
        colour = water_light[18];
        span = water_light[19].xy;
    }
    if colour.w <= 0.5 { return vec4<f32>(colour.rgb, 1.0); }
    let eye_z = -(view.view_from_world * vec4<f32>(world_pos, 1.0)).z;
    // MONKEY (p0 fog hook): the shared fog law (fog_hook.wgsl), as a factor; classic is bit-identical.
    return fog_hook::fog_sample(colour.rgb, span, eye_z, world_pos, view.world_position, true,
        water_monkey());
}

// True where this surface takes the Enhanced path: water (not magma/slime), a tier above
// Classic, and a live scene-depth image (the 1x1 placeholder means "not resolved this frame").
fn water_active() -> bool {
    return water.lane.z < 0.5 && water.mode.x > 0.5 && textureDimensions(scene_depth).x > 1u;
}

// The module's clock: the shader time, or the pinned capture phase on a deterministic run.
fn water_time() -> f32 {
    if water.mode.w == 0.0 { return water.mode.z; }
    return globals.time;
}

// Vertex Gerstner displacement for the ocean's long swell (zero everywhere else).
// `liquid/waves.rs` mirrors it on the CPU for the swimmer bob; the two are one contract.
fn water_swell(xz: vec2<f32>, authored_depth: f32, stitch: f32) -> vec3<f32> {
    if water.lane.z < 0.5 && water.lane.y > 0.5 && water.lane.x < 0.5 && water.mode.x > 0.5 {
        let now = water_time();
        let shore = swell_shore_fade(authored_depth);
        // MONKEY (reviewfix): a tagged fine boundary vertex follows the straight edge between the
        // two displaced coarse endpoints. That closes the T-junction against a coarse neighbour.
        if stitch >= 1.0 {
            let lane = u32(floor(stitch));
            let along = fract(stitch);
            var direction = vec2<f32>(1.0, 0.0);
            if lane == 2u { direction = vec2<f32>(-1.0, 0.0); }
            if lane == 3u { direction = vec2<f32>(0.0, 1.0); }
            if lane == 4u { direction = vec2<f32>(0.0, -1.0); }
            let lo = xz - direction * along * 4.1666665;
            let hi = lo + direction * 4.1666665;
            return mix(water_gerstner(lo, now, shore).xyz,
                water_gerstner(hi, now, shore).xyz, along);
        }
        return water_gerstner(xz, now, shore).xyz;
    }
    return vec3<f32>(0.0);
}

// Enhanced is entirely analytic: height and its exact x/z derivatives, in yards.
// Direction is radians from world +X toward +Z; phase speed follows deep-water dispersion.
// Two mesh-resolvable long waves sum to at most 0.34 yd before energy/shore attenuation.
// MONKEY (water): one temporary wind bearing until the weather lane supplies shared wind.
// Every wind-aligned water term derives from this constant rather than owning a second heading.
const WATER_WIND_DIR: f32 = 0.35;
const WATER_GERSTNER_CHOP: f32 = 2.4;
const WATER_WAVES: array<vec4<f32>, 8> = array<vec4<f32>, 8>(
    // direction, wavelength, amplitude, phase offset
    vec4<f32>(WATER_WIND_DIR, 18.0, 0.200, 0.0),
    vec4<f32>(0.80, 12.8, 0.140, 1.7),
    vec4<f32>(-0.18, 9.5, 0.090, 3.1),
    vec4<f32>(0.52, 4.8, 0.055, 0.8),
    vec4<f32>(1.10, 3.3, 0.033, 2.4),
    vec4<f32>(-0.45, 2.6, 0.018, 4.6),
    vec4<f32>(0.15, 1.3, 0.006, 1.2),
    vec4<f32>(0.95, 0.85, 0.003, 3.8),
);

// ── SHORE BREAK (item 1) + FOAM LEVELS: the tuned constants, all in one place ──
//
// The shore break is a SEPARATE band from `WATER_WAVES` and touches neither the long swell that
// moves ocean vertices nor the inland colour profile. It exists only in the shoaling zone —
// roughly 0.35 to 3.5 yd of water over the bed — and it is normals + foam, never displacement.
//
// **Its phase coordinate is the vertical depth itself**, not a world ruler. That is the whole
// trick: an iso-depth line IS the shoreline's own contour, so crests run exactly parallel to the
// beach on a curved coast, a headland and a cove alike, with no per-chunk seam and no phase jump
// when the depth gradient rotates. The world direction only ever enters through the NORMAL, where
// it arrives as the analytic derivative `d(phase)/d(depth) · grad(depth)` — so the picture and the
// lighting agree by construction. Where the bed is flat enough that the gradient is unusable the
// direction falls back to the wind (`SHORE_WIND_DIR`, the primary swell's bearing), which is also
// the only place the fallback can show, because a flat bed has no shoaling band to draw.
//
// `shore_g(d) = (d + A·(1 − e^(−d/B))) / L` is a smooth, strictly increasing crest count. Its
// derivative `(1 + (A/B)·e^(−d/B)) / L` is the local wavenumber in DEPTH space, so with A/B = 1.3
// the crest spacing runs 0.5 yd of depth at the waterline out to ~1.07 yd offshore: the wave
// shortens by better than 2× as it shoals, which is the shoaling law's visible half. Because the
// coordinate is depth, that spacing turns into a WORLD wavelength through the bed slope — so a
// gentle beach gets long rollers and a steep one short ones, and the number of visible lines
// approaching the sand stays about four either way.
const SHORE_LAMBDA_D: f32 = 1.15;      // yards of depth per crest, offshore
const SHORE_COMPRESS: f32 = 1.56;      // A — shoaling compression (A/B = 1.3 ⇒ 2.3× at the edge)
const SHORE_COMPRESS_D: f32 = 1.2;     // B — yards of depth the compression decays over
const SHORE_PERIOD: f32 = 3.6;         // seconds between arrivals; the swash runs at 2×  this
const SHORE_TILT: f32 = 0.09;          // peak crest steepness (tan of the tilt), slope-independent
const SHORE_WARP_A: f32 = 0.22;        // yards of depth — coarse crest wander (never ruler-straight)
const SHORE_WARP_B: f32 = 0.10;        // yards of depth — finer segmentation of the same crests
const SHORE_WIND_DIR: f32 = WATER_WIND_DIR;

// Foam alphas. The owner rejected BOTH a thick icing sheet and straight stripes before this, so
// every one of these is gated behind a noise breakup and a depth window; the numbers are the
// ceiling a fully-lit, fully-broken crest can reach, not what a typical pixel gets.
const FOAM_WET_EDGE: f32 = 0.35;       // the faint wet line where water meets anything solid
const FOAM_SWASH: f32 = 0.62;          // the sheet running up the sand and fading
const FOAM_CREST: f32 = 0.88;          // the white front of the last wave or two, and its lace
const FOAM_WHITECAP: f32 = 0.68;       // High-only open-sea crest fold
const FOAM_MAX: f32 = 0.90;            // hard ceiling on the sum

fn swell_shore_fade(depth: f32) -> f32 {
    // Ocean V = byte/255, about 148 yd at 1.0: fade in over ~0.15..3.7 yd.
    return smoothstep(0.001, 0.025, depth);
}

fn water_waves(p: vec2<f32>, t: f32, distance: f32, footprint: f32,
    shore: f32, long_only: bool) -> vec3<f32> {
    let energy = clamp(water.mode.y, 0.0, 1.0);
    let tempo = mix(0.4, 1.0, sqrt(energy));
    let inland = water.lane.y < 0.5 || water.lane.x > 0.5;
    // MONKEY (water): inland ADT and exterior WMO water drift gently (a constant vector: a rigid
    // translation, never a shear). True interior pools stay still.
    var ripple_p = p;
    if inland && water.lane.x < 1.5 {
        ripple_p -= t * vec2<f32>(0.06, 0.025);
    }
    var result = vec3<f32>(0.0);
    for (var i = 0u; i < 8u; i += 1u) {
        if long_only && i >= 2u { break; }
        if inland && i < 3u { continue; }
        let wave = WATER_WAVES[i];
        let direction = vec2<f32>(cos(wave.x), sin(wave.x));
        let k = 6.2831853 / wave.y;
        let speed = sqrt(10.72 / k); // gravity in yd/s^2; phase speed in yd/s
        let phase = k * (dot(direction, ripple_p) - speed * tempo * t) + wave.w;
        // Suppress unresolved waves before Nyquist, and remove fine ripples beyond 35 yd.
        var fade = 1.0 - smoothstep(0.10, 0.45, footprint / wave.y);
        if i >= 6u { fade *= 1.0 - smoothstep(10.0, 35.0, distance); }
        if i < 2u { fade *= shore; }
        let amplitude = wave.z * energy * fade;
        result += vec3<f32>(amplitude * sin(phase), amplitude * k * cos(phase) * direction);
    }
    return result;
}

// MONKEY (water): Ported from WarcraftXL's Gerstner displacement and fold-driven gFoam path.
// xyz is horizontal/vertical/horizontal displacement; w is 1 - det(J), the amount the horizontal
// map has compressed toward a fold. Only the two mesh-resolvable long bands move geometry.
fn water_gerstner(p: vec2<f32>, t: f32, shore: f32) -> vec4<f32> {
    let energy = clamp(water.mode.y, 0.0, 1.0);
    let tempo = mix(0.4, 1.0, sqrt(energy));
    var displacement = vec3<f32>(0.0);
    var jacobian_xx = 1.0;
    var jacobian_xz = 0.0;
    var jacobian_zx = 0.0;
    var jacobian_zz = 1.0;
    for (var i = 0u; i < 2u; i += 1u) {
        let wave = WATER_WAVES[i];
        let direction = vec2<f32>(cos(wave.x), sin(wave.x));
        let k = 6.2831853 / wave.y;
        let speed = sqrt(10.72 / k);
        let phase = k * (dot(direction, p) - speed * tempo * t) + wave.w;
        let amplitude = wave.z * energy * shore;
        let horizontal = WATER_GERSTNER_CHOP * amplitude;
        displacement += vec3<f32>(direction.x * horizontal * cos(phase),
            amplitude * sin(phase), direction.y * horizontal * cos(phase));
        let compression = horizontal * k * sin(phase);
        jacobian_xx -= compression * direction.x * direction.x;
        jacobian_xz -= compression * direction.x * direction.y;
        jacobian_zx -= compression * direction.y * direction.x;
        jacobian_zz -= compression * direction.y * direction.y;
    }
    let fold = max(1.0 - (jacobian_xx * jacobian_zz - jacobian_xz * jacobian_zx), 0.0);
    return vec4<f32>(displacement, fold);
}

// MONKEY (visualfix): the whitecap fold. The two long bands alone cross in a regular diamond
// lattice, so every crest broke at the same spots; the next two bands (non-integer length and
// direction ratios to the first two) add their compression at a reduced weight. Mask only: the
// displaced geometry still moves on the two long bands.
fn whitecap_fold(p: vec2<f32>, t: f32, shore: f32) -> f32 {
    var fold = water_gerstner(p, t, shore).w;
    let energy = clamp(water.mode.y, 0.0, 1.0);
    let tempo = mix(0.4, 1.0, sqrt(energy));
    for (var i = 2u; i < 4u; i += 1u) {
        let wave = WATER_WAVES[i];
        let direction = vec2<f32>(cos(wave.x), sin(wave.x));
        let k = 6.2831853 / wave.y;
        let phase = k * (dot(direction, p) - sqrt(10.72 / k) * tempo * t) + wave.w;
        fold += 0.55 * WATER_GERSTNER_CHOP * wave.z * energy * shore * k * sin(phase);
    }
    return max(fold, 0.0);
}

// MONKEY (water): seam for the later wet-weather lane. Its result is a height-gradient
// perturbation in world XZ; zero preserves today's image until that lane supplies rain data.
// MONKEY (wet): rain rings. Each layer is a grid of cells, one drop per cell at a hashed spot and
// phase; a drop lands when its cell hash is under the rain rate, and its ring expands and fades
// over one period. The ring's slope profile is a Gaussian-windowed cosine across the front. The
// clock is MonkeyFrame `wet_a.z` (wraps at 1000 s; the rates are multiples of 1/1000, so the wrap is
// seamless). Dry (`wet_a.x == 0`), WMO interior pools, interior-fog rooms, and pixels too far or too
// coarse to hold a ring return zero.
fn ripple_hash(p: vec2<f32>) -> vec2<f32> {
    var q = fract(vec3<f32>(p.xyx) * vec3<f32>(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yzx + 33.33);
    return fract((q.xx + q.yz) * q.zy);
}
fn rain_ripple_normal(world_pos: vec3<f32>, footprint: f32, distance: f32, room: bool) -> vec2<f32> {
    let wet_row = water_monkey().wet_a;
    let rain = clamp(wet_row.x, 0.0, 1.0);
    if rain <= 0.0 || water.lane.x > 1.5 || room {
        return vec2<f32>(0.0);
    }
    let world_xz = world_pos.xz;
    // MONKEY (rainshelter): no rings under a bridge or a roof.
    let reach = (1.0 - smoothstep(35.0, 60.0, distance)) * (1.0 - smoothstep(0.05, 0.14, footprint))
        * water_rain_open(world_pos);
    if reach <= 0.0 {
        return vec2<f32>(0.0);
    }
    var grad = vec2<f32>(0.0);
    for (var layer = 0; layer < 2; layer += 1) {
        let c = select(0.9, 1.45, layer == 1);
        let rate = select(0.8, 0.65, layer == 1);
        let base = floor(world_xz / c);
        for (var j = -1; j <= 1; j += 1) {
            for (var i = -1; i <= 1; i += 1) {
                let cell = base + vec2<f32>(f32(i), f32(j));
                let seed = cell + vec2<f32>(f32(layer) * 57.0, f32(layer) * 113.0);
                let h = ripple_hash(seed);
                let h_alt = ripple_hash(seed + 19.19);
                // Each period re-rolls whether this cell rains, so the pattern does not repeat.
                let cycle = wet_row.z * rate + h.x;
                // MONKEY (fix-wet): `% 50` keeps the roll sequence continuous across the 1000 s
                // clock wrap (cycle drops by 800 / 650, both multiples of 50).
                let roll = ripple_hash(seed + (floor(cycle) % 50.0) * 7.31).x;
                if roll > rain * 0.85 {
                    continue;
                }
                let phase = fract(cycle);
                let centre = (cell + 0.5 + (h_alt - 0.5) * 0.6) * c;
                let to_p = world_xz - centre;
                let d = length(to_p);
                let radius = phase * 0.6 * c;
                let x = d - radius;
                let width = 0.05 + 0.07 * phase;
                let fade = (1.0 - phase) * (1.0 - phase) * smoothstep(0.0, 0.08, phase);
                let slope = fade * exp(-(x * x) / (width * width)) * cos(x * 3.1415927 / width);
                grad += to_p / max(d, 1e-4) * slope;
            }
        }
    }
    return grad * (0.7 + 0.5 * rain) * reach;
}

fn foam_hash(p_in: vec2<f32>) -> f32 {
    // MONKEY (foam noise): world coordinates here are ~1e4 yd and are scaled further before they
    // arrive, so `p * 0.1031` sat near 2e3 where an f32 keeps only ~13 fraction bits - the hash
    // degenerated into a visible regular tile grid inside the foam (capture it11-water-beach-top).
    // Wrap the LATTICE CELL to a 512 period first: exact for integers, invisible at 512 cells, and it
    // gives the hash its full precision back.
    let p = p_in - 512.0 * floor(p_in / 512.0);
    let q = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    let r = q + dot(q, q.yzx + 33.33);
    return fract((r.x + r.y) * r.z);
}

fn foam_value_noise(p: vec2<f32>) -> f32 {
    let cell = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    return mix(mix(foam_hash(cell), foam_hash(cell + vec2<f32>(1.0, 0.0)), u.x),
        mix(foam_hash(cell + vec2<f32>(0.0, 1.0)),
            foam_hash(cell + vec2<f32>(1.0, 1.0)), u.x), u.y);
}

// MONKEY (foam noise): thresholded VALUE noise shows its lattice - the foam and crest masks came out
// as boxy slabs with stair-stepped edges aligned to the world axes (capture it9/it10-water-beach-top).
// Two samples on differently ROTATED and scaled lattices, averaged, have no shared axis, so the
// thresholded shapes turn into irregular blobs. Same range (0..1), same call sites, twice the taps.
fn foam_noise(p: vec2<f32>) -> f32 {
    let a = vec2<f32>(0.7986 * p.x - 0.6018 * p.y, 0.6018 * p.x + 0.7986 * p.y);
    let q = p * 1.37 + vec2<f32>(17.3, -9.1);
    let b = vec2<f32>(0.3584 * q.x + 0.9336 * q.y, -0.9336 * q.x + 0.3584 * q.y);
    // Averaging narrows the distribution; re-expand it so the existing thresholds keep their cut.
    return clamp((0.5 * (foam_value_noise(a) + foam_value_noise(b)) - 0.5) * 1.4 + 0.5, 0.0, 1.0);
}
// ── DEPTH LOOK: extinction, refraction, caustics ─────────────────────────────────────────────
// Portions derived from WarcraftXL wxl-experimental-water by iThorgrim, used with permission
// (`shaders/Surface.ps.hlsl`: the view path through the column, per-channel Beer-Lambert
// extinction, the body as a lerp from the scattered colour to what is behind by the transmittance,
// the normal-bent scene copy scaled by the water there is, and the two-layer multiplied caustic
// web; `render/Noise.cpp`: the caustic web as ridged noise raised to a power). Ours differs where
// we have more to go on: the column is the MEASURED scene depth, so an object in the water is seen
// as one, and a bent sample that lands on something in front of the water is refused outright
// instead of kept short.
//
// Per-yard attenuation, R G B. Red goes first, so a deepening sea turns teal, then blue; inland
// water loses blue faster than green and reads green-brown, as a lake over a muddy bed does.
const EXTINCTION_OCEAN: vec3<f32> = vec3<f32>(0.55, 0.19, 0.15);
const EXTINCTION_INLAND: vec3<f32> = vec3<f32>(0.75, 0.42, 0.58);
const EXTINCTION_MAX_PATH: f32 = 40.0;   // yards; past this nothing comes back anyway
const REFRACT_STRENGTH: f32 = 0.30;      // screen bend per unit of normal tilt, at 10 yd
const REFRACT_MAX: f32 = 0.018;          // cap, in viewport heights (WXL's clamp)
const CAUSTIC_GAIN: f32 = 2.6;
const CAUSTIC_DEPTH: f32 = 6.0;          // yards of water the floor web survives
const CAUSTIC_SCALE: f32 = 0.65;         // web cells per yard (fine layer)

// The web itself: animated cell EDGES (the gap between the nearest and second-nearest of a set of
// drifting points), which are thin, connected and curved - the network a wavy surface focuses
// sunlight into. Smooth value noise thresholded into "creases" showed its lattice as blocky crosses
// on the sand (capture r3-caustics), so the pattern is built from points, not a grid of values.
fn caustic_point(cell: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(foam_hash(cell), foam_hash(cell + vec2<f32>(37.0, 17.0)));
}

fn caustic_edges(p: vec2<f32>, t: f32) -> f32 {
    let cell = floor(p);
    let f = fract(p);
    var nearest_distance = 8.0;
    var next_distance = 8.0;
    for (var j = -1; j <= 1; j += 1) {
        for (var i = -1; i <= 1; i += 1) {
            let g = vec2<f32>(f32(i), f32(j));
            let h = caustic_point(cell + g);
            // Each point orbits its cell on its own phase: the web swims instead of sliding.
            let o = 0.5 + 0.38 * sin(vec2<f32>(t * 0.83, t * 0.71) + 6.2831853 * h);
            let r = g + o - f;
            let d = dot(r, r);
            if d < nearest_distance {
                next_distance = nearest_distance;
                nearest_distance = d;
            } else if d < next_distance {
                next_distance = d;
            }
        }
    }
    return sqrt(next_distance) - sqrt(nearest_distance);
}

// Two webs at incommensurate scales and rates: the fine one draws the lines, the coarse one sets
// where they burn bright, so the pattern never repeats on a visible period (WXL's two layers).
fn caustic_web(xz: vec2<f32>, t: f32) -> f32 {
    let r = vec2<f32>(0.7501 * xz.x - 0.6613 * xz.y, 0.6613 * xz.x + 0.7501 * xz.y) * CAUSTIC_SCALE;
    // A low-frequency warp bends the straight cell edges into curves.
    let warp = vec2<f32>(foam_noise(r * 0.62 + vec2<f32>(t * 0.05, 3.1)),
        foam_noise(r * 0.62 + vec2<f32>(11.7, -t * 0.04))) - vec2<f32>(0.5);
    let q = r + warp * 1.7;
    // A thin bright core and a soft glow around it, as a focused line has.
    let e = caustic_edges(q, t);
    let fine = (1.0 - smoothstep(0.0, 0.06, e)) + 0.35 * (1.0 - smoothstep(0.0, 0.22, e));
    // The coarse web decides where the fine one burns and where it fades out altogether, so the
    // network breaks up instead of tiling the whole bed evenly.
    let coarse = 1.0 - smoothstep(0.0, 0.45, caustic_edges(q * 0.41 + vec2<f32>(5.3, 1.9), t * 0.77 + 2.0));
    let patchy = smoothstep(0.25, 0.75, foam_noise(r * 0.18 + vec2<f32>(t * 0.02, -7.0)));
    return fine * (0.28 + 0.72 * coarse) * mix(0.45, 1.0, patchy) * 1.25;
}

// ── HIGH: screen-space reflection of the scenery ─────────────────────────────────────────────
// Portions derived from WarcraftXL wxl-experimental-water by iThorgrim, used with permission
// (`shaders/Surface.ps.hlsl`: reflect the scene copy along an almost-planar normal carrying only a
// fraction of the waves, masked by the screen edge and by rays that turn back toward the eye).
// WXL takes one probe a fixed distance down the ray; with the scene depth already in hand, High
// marches the ray against it instead, so a bank reflects at the right place, not at a guess.
const SSR_STEPS: i32 = 32;
const SSR_FIRST_YD: f32 = 0.35;     // first step; each next one is SSR_GROWTH times longer
const SSR_GROWTH: f32 = 1.20;       // 32 steps reach ~600 yd; bisection restores the precision
const SSR_REFINE: i32 = 5;          // bisections between the last miss and the hit
const SSR_RIPPLE: f32 = 0.35;       // share of the wave normal the reflected ray sees (WXL ssrRipple)

// The scene's view distance under the point `pos`, and that point's own; x = scene, y = ray,
// z = 1 when `pos` is on screen. Sky pixels report no surface.
fn ssr_probe(pos: vec3<f32>, dims: vec2<f32>) -> vec4<f32> {
    let clip = view.clip_from_world * vec4<f32>(pos, 1.0);
    if clip.w <= 0.0 { return vec4<f32>(0.0); }
    let ndc = clip.xyz / clip.w;
    if abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0 { return vec4<f32>(0.0); }
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    let px = clamp(vec2<i32>(uv * dims), vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1));
    let scene = textureLoad(scene_depth, px, 0).r;
    if scene <= 0.0 { return vec4<f32>(1e9, water_view_distance(ndc.z), 1.0, 0.0); }
    return vec4<f32>(water_view_distance(scene), water_view_distance(ndc.z), 1.0, 0.0);
}

// x/y/z = the reflected scene colour (gamma lane), w = how much of it to trust (0 = use the sky).
fn water_ssr(origin: vec3<f32>, dir: vec3<f32>) -> vec4<f32> {
    let dims = vec2<f32>(textureDimensions(scene_depth));
    var step_len = SSR_FIRST_YD;
    var along = 0.0;
    var last_miss = 0.0;
    for (var i = 0; i < SSR_STEPS; i += 1) {
        along += step_len;
        let probe = ssr_probe(origin + dir * along, dims);
        if probe.z < 0.5 { break; }                       // left the screen: nothing to show
        let behind = probe.y - probe.x;
        if behind > 0.0 && behind < 2.0 + 0.1 * probe.x + step_len {
            // Bisect back toward the last miss, so the reflection sits on the surface it hit.
            var lo = last_miss;
            var hi = along;
            for (var k = 0; k < SSR_REFINE; k += 1) {
                let mid = 0.5 * (lo + hi);
                let m = ssr_probe(origin + dir * mid, dims);
                if m.y > m.x { hi = mid; } else { lo = mid; }
            }
            let clip = view.clip_from_world * vec4<f32>(origin + dir * hi, 1.0);
            let uv = (clip.xy / clip.w) * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
            let px = clamp(vec2<i32>(uv * dims), vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1));
            // Fade toward the screen edges (the top one widest: reflections run off it) and with
            // distance, so a missing reflection hands over to the sky instead of cutting off.
            let edge = smoothstep(0.0, 0.06, min(uv.x, 1.0 - uv.x))
                * smoothstep(0.0, 0.22, uv.y) * smoothstep(0.0, 0.06, 1.0 - uv.y);
            let trust = edge * (1.0 - smoothstep(450.0, 600.0, hi));
            return vec4<f32>(textureLoad(scene_colour, px, 0).rgb, trust);
        }
        last_miss = along;
        step_len *= SSR_GROWTH;
    }
    return vec4<f32>(0.0);
}

// Reverse-Z perspective: valid for both finite and infinite far planes. For z_view = -distance,
// depth = -P22 + P32 / distance. Do not use a forward-Z near/far approximation.
fn water_view_distance(depth: f32) -> f32 {
    return view.clip_from_view[3][2] / max(depth + view.clip_from_view[2][2], 1e-7);
}

fn enhanced_water(in: WaterFragment, shallow: vec4<f32>, deep: vec4<f32>) -> vec4<f32> {
    let pixel = clamp(vec2<i32>(in.clip_position.xy), vec2<i32>(0),
        vec2<i32>(textureDimensions(scene_depth)) - vec2<i32>(1));
    let own_depth = textureLoad(scene_depth, pixel, 0).r;
    // MONKEY (bed clutter): grass blades and reeds standing in a stream are in the opaque depth, so
    // every blade read as "something solid at the surface" - each one grew a white contact-foam
    // outline and a paler, less-absorbed tint than the bed around it (owner screenshot, Stonefield
    // Farm). Anything THIN is not a shore: take the FARTHEST of nine taps (reverse-Z: the smallest)
    // over a ~0.7 % of the view height ring, so a blade a few pixels wide resolves to the bed behind
    // it, while a bank, a rock or a hull - wider than the ring - is untouched bar a few pixels.
    let ring = max(i32(view.viewport.w * 0.0065), 2);
    let top = vec2<i32>(textureDimensions(scene_depth)) - vec2<i32>(1);
    var far_depth = own_depth;
    var offs = array<vec2<i32>, 8>(
        vec2<i32>(1, 1), vec2<i32>(1, 0), vec2<i32>(-1, 1), vec2<i32>(0, 1),
        vec2<i32>(-1, -1), vec2<i32>(-1, 0), vec2<i32>(1, -1), vec2<i32>(0, -1));
    for (var k = 0; k < 8; k += 1) {
        let reach = select(ring, 2 * ring, (k & 1) == 1);
        let tap = textureLoad(scene_depth,
            clamp(pixel + offs[k] * reach, vec2<i32>(0), top), 0).r;
        // Reverse-Z: farther is SMALLER; 0 is the sky, which is not a bed.
        if tap < far_depth && tap > 0.0 { far_depth = tap; }
    }
    // The open sea keeps its own pixel: its shore train's phase IS this depth, and sand has no reeds.
    let sea = water.lane.y > 0.5 && water.lane.x < 0.5;
    let bed_depth = select(far_depth, own_depth, sea);
    let to_view = normalize(view.world_position.xyz - in.world_position.xyz);
    let eye_pos = (view.view_from_world * vec4<f32>(in.world_position.xyz, 1.0)).xyz;
    // Convert the two eye-Z distances to a distance ALONG the pixel's view ray.
    let ray_cos = max(abs(normalize(eye_pos).z), 0.001);
    let thickness = min(max(water_view_distance(bed_depth)
        - water_view_distance(in.clip_position.z), 0.0) / ray_cos, 1000.0);
    // Reconstruct the opaque scene point, then measure height below the surface.
    // Ray thickness is only the contact measure: opacity must not follow eye-Z.
    let ndc_xy = ((in.clip_position.xy - view.viewport.xy) / view.viewport.zw)
        * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    let scene_h = view.world_from_clip * vec4<f32>(ndc_xy, bed_depth, 1.0);
    var vertical_depth = 1000.0;
    if bed_depth > 0.0 && abs(scene_h.w) > 1e-7 {
        vertical_depth = clamp(in.world_position.y - scene_h.y / scene_h.w, 0.0, 1000.0);
    }
    let t = water_time(); // WOW_CAPTURE_WATER_T pins the frozen water phase.
    let fog = water_fog(in.world_position.xyz, in.room_fog);
    let p = in.source_xz;
    let energy = clamp(water.mode.y, 0.0, 1.0);
    let ocean_mesh = water.lane.y > 0.5 && water.lane.x < 0.5;

    // ── EVERY screen derivative in this shader is taken HERE ────────────────────────────────
    // Top level of the function, before any branch, loop or early return: WGSL's uniformity rule
    // makes a derivative inside non-uniform control flow invalid, and this function is long enough
    // that the only safe discipline is to have exactly one place they can live. Nothing below may
    // reintroduce a `dpdx`/`dpdy` inside a conditional.
    let ddx_p = dpdx(p);
    let ddy_p = dpdy(p);
    let ddx_depth = dpdx(vertical_depth);
    let ddy_depth = dpdy(vertical_depth);
    let footprint = max(length(ddx_p), length(ddy_p));

    // The BED's slope in world yards, recovered by inverting the pixel→world Jacobian: solve
    // `g·(dp/dx) = dd/dx` and `g·(dp/dy) = dd/dy` for the world-space gradient of the vertical
    // depth field. `offshore` is the unit direction of INCREASING depth, so `−offshore` points at
    // the beach and the shore train travels that way.
    let jac_det = ddx_p.x * ddy_p.y - ddx_p.y * ddy_p.x;
    var depth_grad = vec2<f32>(0.0, 0.0);
    if abs(jac_det) > 1e-12 {
        depth_grad = vec2<f32>(
            (ddx_depth * ddy_p.y - ddy_depth * ddx_p.y) / jac_det,
            (ddy_depth * ddx_p.x - ddx_depth * ddy_p.x) / jac_det,
        );
    }
    let bed_slope = length(depth_grad);
    // Flat bed, or the depth field broke across a silhouette (the ratio blows up there): fall back
    // to the wind. A flat bed has no shoaling band to draw, so the fallback is mostly a guard.
    var offshore = vec2<f32>(cos(SHORE_WIND_DIR), sin(SHORE_WIND_DIR));
    if bed_slope > 0.02 && bed_slope < 12.0 {
        offshore = depth_grad / bed_slope;
    }

    // ── The shore break: shoaling trains riding the bathymetry (see the constants block) ────
    // Domain-warped in DEPTH units, so the crests wander and segment along the beach instead of
    // running as ruler-straight bands.
    let shore_warp =
        SHORE_WARP_A * (2.0 * foam_noise(p * 0.075 + t * vec2<f32>(0.010, -0.007)) - 1.0)
        + SHORE_WARP_B * (2.0 * foam_noise(p * 0.21 - t * vec2<f32>(0.006, 0.009)) - 1.0);
    let dq = max(vertical_depth + shore_warp, 0.0);
    let shore_g = (dq + SHORE_COMPRESS * (1.0 - exp(-dq / SHORE_COMPRESS_D))) / SHORE_LAMBDA_D;
    // `+ t/period` with a crest-count that RISES with depth ⇒ a crest of fixed phase slides to
    // shallower water as time runs: the train travels shoreward.
    let shore_phase = 6.2831853 * (shore_g + t / SHORE_PERIOD);
    let shore_phase_secondary = 6.2831853 * (1.9 * shore_g + t / (SHORE_PERIOD * 0.62)) + 2.1;
    // Steepness rises as it shoals (3.5 → 0.75 yd), collapses into the break below ~0.35 yd, and
    // is gone past the band's offshore edge.
    let shoal = smoothstep(3.5, 0.75, vertical_depth);
    let shore_collapse = smoothstep(0.12, 0.38, vertical_depth);
    let shore_offshore_fade = 1.0 - smoothstep(2.4, 3.6, vertical_depth);
    // MONKEY (surf on wrecks): the shore train reads the SCENE depth, and a sunken boat, a pier
    // foot or a rock shelf is shallow scene depth in the middle of deep water - so the whole surf
    // (crests, lace, swash) was painted over the hull of a wreck off Longshore. A shore is where
    // the SEA BED is shallow. The mesh carries the authored bed depth (`in.depth`, byte/255 of about
    // 148 yd): where the bed lies well below what the pixel sees, the pixel is an OBJECT, and an
    // object gets the thin wet-edge line only. The byte is a FLOOR (1.72 per yd), so on a real beach the
    // authored bed is never deeper than the scene by more than interpolation error: allow 0.5 yd.
    // MONKEY (water): MLIQ carries an opacity-ramp byte, not ADT's authored bed-depth byte. For a
    // WMO pool the only actual column measurement is the reconstructed scene depth, so never turn
    // its opacity into 148 yards of fictitious water.
    let bed_yd = select(vertical_depth, clamp(in.depth, 0.0, 1.0) * 148.0,
        water.lane.x < 0.5);
    let on_bed = 1.0 - smoothstep(0.5, 1.0, bed_yd - vertical_depth);
    let shore_gain = select(0.0, shoal * shore_collapse * shore_offshore_fade * on_bed, ocean_mesh)
        * energy;
    // Steepness is set DIRECTLY — the tangent of the crest tilt — rather than through an amplitude
    // and a wavenumber, so a gentle beach and a steep one roll with the same visible strength
    // instead of one washing out and the other exploding.
    let shore_grad = SHORE_TILT * shore_gain
        * (cos(shore_phase) + 0.45 * cos(shore_phase_secondary)) * offshore;

    let shore = select(1.0, swell_shore_fade(in.depth), ocean_mesh);
    let wave = water_waves(p, t, length(eye_pos), footprint, shore, false);
    var open_sea_fold = 0.0;
    if water.mode.x > 1.5 && ocean_mesh {
        open_sea_fold = whitecap_fold(p, t, shore);
    }
    // One surface gradient: the procedural bands and the shore break.
    let surf_grad = wave.yz + shore_grad + rain_ripple_normal(in.world_position.xyz, footprint,
        length(eye_pos), in.room_fog != 0u && water.lane.w > 0.5);
    var n = normalize(vec3<f32>(-surf_grad.x, 1.0, -surf_grad.y));
    if dot(n, to_view) < 0.0 { n = -n; }

    // Far water settles into a calm sky sheet.
    n = normalize(mix(n, vec3<f32>(0.0, sign(n.y), 0.0),
        0.6 * smoothstep(60.0, 120.0, length(eye_pos))));
    // Taken here, where `n` is final and control flow has reconverged (the derivative rule above).
    let normal_variance = dot(dpdx(n), dpdx(n)) + dot(dpdy(n), dpdy(n));
    // Clean translucent teal at the edge; zone tint remains in the deeper body.
    let teal_shallow = mix(vec3<f32>(0.12, 0.42, 0.39), shallow.rgb, 0.18);
    // MONKEY (water body): the zone deep row alone renders the open sea and lake middles a muddy
    // grey-brown (Westfall ocean, Loch Modan) - it was authored to sit UNDER the reference ripple
    // sheet, not to be a lit body colour. Pull it 70 % toward the zone zenith sky (deep water takes
    // its colour from the sky it scatters) and keep it blue-dominant, so the body reads as water by
    // day, follows dusk and night through the same sky row, and still differs zone to zone.
    let zenith_gamma = pow(max(water.sky_zenith.rgb, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
    let deep_sky = mix(deep.rgb, zenith_gamma * 1.1, 0.7);
    let deep_body = vec3<f32>(min(deep_sky.r, deep_sky.g * 0.85), deep_sky.g,
        max(deep_sky.b, deep_sky.g * 1.12));
    var body = mix(teal_shallow, deep_body, smoothstep(0.0, 3.5, vertical_depth));
    if !ocean_mesh {
        let clear_row = mix(shallow.rgb, vec3<f32>(0.20, 0.40, 0.36), 0.35);
        let clear_tint = vec3<f32>(min(clear_row.r, clear_row.g * 0.72), clear_row.g,
            clamp(clear_row.b, clear_row.g * 0.82, clear_row.g * 1.15));
        let inland_row = mix(deep.rgb, zenith_gamma * 1.1, 0.25);
        let inland_deep = vec3<f32>(min(inland_row.r, inland_row.g * 0.72), inland_row.g,
            clamp(inland_row.b, inland_row.g * 0.82, inland_row.g * 1.15));
        body = mix(clear_tint, inland_deep, smoothstep(0.0, 5.0, vertical_depth));
    }
    // MONKEY (water): the reference interior renderer carries MOMT.diffColor on the pool vertex.
    // Under a roof that authored colour replaces the outdoor Light.dbc/sky-derived body palette.
    if water.lane.x > 1.5 {
        body = in.colour.rgb;
    }
    let to_light = -normalize(water_light[2].xyz);
    // MONKEY (water body): water is lit by the light it SCATTERS, not only by N.L on its skin, so
    // the lit body keeps a high floor (ambient x 1.35, N.L floor 0.5). With the old 0.25 floor a
    // noon lake rendered near-black navy where the owner reference is a bright teal-blue; night
    // still darkens because both rows do.
    let lighting = clamp(water_light[0].rgb * 1.35 + water_light[1].rgb
        * max(dot(n, to_light), 0.5), vec3<f32>(0.0), vec3<f32>(1.0));
    var rgb = body * lighting;
    if water.lane.x > 1.5 {
        rgb = body;
    }
    // MONKEY (shore waves): the shore train is shown mainly as a CONTINUOUS crest brightening, not as
    // a normal tilt. Its direction comes from the screen-space gradient of the bed depth, and the
    // terrain is flat triangles, so that direction jumps at every triangle edge: at SHORE_TILT 0.30
    // the highlights broke into blocky rectangular shards with staircase edges (capture
    // it9-water-beach-top). The PHASE is a function of the depth itself and is continuous, so a term
    // driven by the phase alone cannot facet. The tilt stays at 0.09 for a little specular life.
    let shore_crest = pow(max(sin(shore_phase), 0.0), 2.0)
        + 0.45 * pow(max(sin(shore_phase_secondary), 0.0), 2.0);
    rgb += shore_gain * shore_crest * 0.13 * lighting * vec3<f32>(0.72, 0.95, 0.90);
    if !ocean_mesh {
        // Keep the zone's green-blue absorption under warm dusk illumination.
        rgb = body * dot(lighting, vec3<f32>(0.2126, 0.7152, 0.0722));
    }
    let celestial_dir = normalize(water.celestial.xyz);
    let crest = smoothstep(0.35, 0.95, 0.5 + 0.5 * wave.x / max(0.545 * energy, 0.001));
    let transmission = crest * pow(max(dot(to_view, -celestial_dir), 0.0), 3.0) * energy
        * smoothstep(-0.02, 0.12, celestial_dir.y);
    rgb += vec3<f32>(0.06, 0.30, 0.19) * transmission * lighting;
    let reflection_n = normalize(mix(vec3<f32>(0.0, sign(n.y), 0.0), n, 0.45));
    let reflected = reflect(-to_view, reflection_n);
    let sky_linear = mix(water.sky_horizon.rgb, water.sky_zenith.rgb, clamp(reflected.y * 1.4, 0.0, 1.0));
    // Linear reflection interpolation, then conversion to the world's gamma blend/fog lane.
    var sky = select(12.92 * sky_linear,
        1.055 * pow(max(sky_linear, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055,
        sky_linear > vec3<f32>(0.0031308));
    // MONKEY (water): an interior pool has no sky probe. High may still replace this fallback with
    // a real SSR hit, while Enhanced reflects the room's own fog colour and keeps sun glints off.
    if water.lane.x > 1.5 {
        sky = fog.rgb;
    }
    var reflectivity = mix(0.06, 0.88, pow(1.0 - max(dot(n, to_view), 0.0), 4.0));
    if !ocean_mesh { reflectivity = min(reflectivity, 0.55); }

    // ── What is behind the surface, and how much of it survives the trip ────────────────────
    // The path is the measured vertical column opened out by the view angle, capped at grazing
    // (WXL: ten times the column) and in absolute length.
    let ndv_flat = max(abs(to_view.y), 0.10);
    let path = min(vertical_depth / ndv_flat, EXTINCTION_MAX_PATH);
    let transmit = exp(-select(EXTINCTION_INLAND, EXTINCTION_OCEAN, ocean_mesh) * path);
    // The frame behind, bent by the surface normal. The bend is world-anchored (it shrinks with
    // distance), scaled by how much water stands there (a film over wet sand bends nothing, or
    // the shoreline crawls) and capped. A bent sample that lands on something IN FRONT of the
    // water - a hull, a leg, the bank - is refused: the scene depth there says so.
    let view_n = (view.view_from_world * vec4<f32>(n, 0.0)).xy;
    let bend_px = clamp(view_n * vec2<f32>(1.0, -1.0) * view.viewport.w
            * REFRACT_STRENGTH * (10.0 / max(length(eye_pos), 6.0)) * saturate(path * 0.5),
        vec2<f32>(-REFRACT_MAX * view.viewport.w), vec2<f32>(REFRACT_MAX * view.viewport.w));
    let colour_top = vec2<i32>(textureDimensions(scene_colour)) - vec2<i32>(1);
    let bent = clamp(pixel + vec2<i32>(round(bend_px)), vec2<i32>(0), colour_top);
    let bent_depth = textureLoad(scene_depth, bent, 0).r;
    let use_bent = bent_depth < in.clip_position.z;  // reverse-Z: smaller = farther than the water
    var behind = textureLoad(scene_colour, select(clamp(pixel, vec2<i32>(0), colour_top), bent,
        use_bent), 0).rgb;
    // Caustics on the bed: shallow, sunlit water only; the moon focuses too, faintly.
    let bed_world = scene_h.xyz / scene_h.w;
    let sun_focus = smoothstep(0.05, 0.45, celestial_dir.y) * select(1.0, 0.12, water.celestial.w > 0.5);
    let caustic_fade = saturate(1.0 - vertical_depth / CAUSTIC_DEPTH) * smoothstep(0.02, 0.25, vertical_depth);
    if sun_focus * caustic_fade > 0.001 {
        // Faded out where a web cell would span only a few pixels, which would alias into shimmer.
        let web = caustic_web(bed_world.xz + n.xz * 0.35, t)
            * (1.0 - smoothstep(0.12, 0.40, footprint * CAUSTIC_SCALE * 4.0));
        behind *= 1.0 + web * CAUSTIC_GAIN * sun_focus * caustic_fade
            * dot(water_light[1].rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    }
    // Light that reaches the eye through water has been scattered on the way, so it carries the
    // water's own tint as well as the colour extinction leaves: without this, yellow sand under a
    // clear sea reads olive rather than turquoise. Half the hue is handed to the shallow tint.
    let tint_norm = teal_shallow / max(max(teal_shallow.r, teal_shallow.g), teal_shallow.b);
    let behind_luma = dot(behind, vec3<f32>(0.2126, 0.7152, 0.0722));
    behind = mix(behind, behind_luma * tint_norm * 1.25, select(0.30, 0.45, ocean_mesh));
    // The body: from the scattered colour to what is behind, by what survives the path.
    // The scattered body is a surface term and takes the fog; `behind` is already fogged.
    rgb = mix(mix(fog.rgb, rgb, fog.w), behind, transmit);
    // High: the scenery reflects, not only the sky. The ray leaves along an almost-planar normal
    // so the reflection holds together and the waves only ripple it; rays climbing toward the eye
    // (a steep look down) have nothing on screen to reflect and keep the sky.
    var reflection = mix(fog.rgb, sky, fog.w);   // the sky reflection is a surface term
    if water.mode.x > 1.5 && reflectivity > 0.1 {
        let ssr_n = normalize(mix(vec3<f32>(0.0, 1.0, 0.0), n, SSR_RIPPLE));
        let ssr_dir = reflect(-to_view, ssr_n);
        if ssr_dir.y > 0.0 {
            let hit = water_ssr(in.world_position.xyz, ssr_dir);
            // Rays heading back toward the eye find nothing on screen that faces them.
            let facing = 1.0 - smoothstep(0.55, 0.9, dot(ssr_dir, to_view));
            reflection = mix(sky, hit.rgb, hit.w * facing);
            // Scenery reads a little stronger than Fresnel alone gives at a low angle - the eye
            // expects a lake to mirror its banks - but only where a reflection was actually found.
            reflectivity = mix(reflectivity, max(reflectivity, 0.42), hit.w * facing);
        }
    }
    rgb = mix(rgb, reflection, reflectivity);
    // Everything added from here on (glints, light speculars) is surface light: fog attenuates it.
    let fogged_base = rgb;
    // Smooth crest glints; widen and conserve lobe energy as the footprint grows.
    let half_v = normalize(celestial_dir + to_view);
    let ndoth = max(dot(n, half_v), 0.0);
    let spread = 1.0 + 400.0 * normal_variance + 2.0 * smoothstep(30.0, 160.0, length(eye_pos));
    if water.lane.x < 1.5 {
        if water.celestial.w > 0.5 {
            // Own peak intensity, never nightGain or the signed moon-shadow weight.
            // sin(8 degrees): the reflection is exactly zero below the horizon.
            rgb += vec3<f32>(0.80, 0.88, 1.0) * 0.45
                * (pow(ndoth, 400.0 / spread) / sqrt(spread)
                    + 0.25 * pow(ndoth, 60.0 / sqrt(spread)) / sqrt(spread))
                * smoothstep(0.0, 0.1391731, celestial_dir.y);
        } else {
            rgb += water_light[1].rgb
                * (pow(ndoth, 400.0 / spread) / sqrt(spread)
                    + 0.25 * pow(ndoth, 60.0 / sqrt(spread)) / sqrt(spread))
                * smoothstep(-0.02, 0.08, celestial_dir.y);
        }
    }

    // Rank candidates by actual fragment distance, not table order. Only four lights are shaded.
    var nearest = array<u32, 4>(256u, 256u, 256u, 256u);
    var distances = array<f32, 4>(900.0, 900.0, 900.0, 900.0);
    for (var i = 0u; i < min(u32(water_light[20].x), 256u); i += 1u) {
        let colour = water_light[21u + 2u * i + 1u];
        // MONKEY (water): outside takes only exterior lights; a true interior pool takes only its
        // room fixtures. This keeps street torches on canals and out of pools behind closed walls.
        let interior_fixture = colour.w >= 0.5;
        if interior_fixture != (water.lane.x > 1.5) { continue; }
        let delta = water_light[21u + 2u * i].xyz - in.world_position.xyz;
        let squared_distance = dot(delta, delta);
        if squared_distance >= distances[3] { continue; }
        var slot = 3u;
        loop {
            if slot == 0u { break; }
            if squared_distance >= distances[slot - 1u] { break; }
            distances[slot] = distances[slot - 1u];
            nearest[slot] = nearest[slot - 1u];
            slot -= 1u;
        }
        distances[slot] = squared_distance;
        nearest[slot] = i;
    }
    for (var j = 0u; j < 4u; j += 1u) {
        if nearest[j] == 256u { continue; }
        let pos = water_light[21u + nearest[j] * 2u];
        let colour = water_light[21u + nearest[j] * 2u + 1u];
        let delta = pos.xyz - in.world_position.xyz;
        let distance = sqrt(max(distances[j], 0.0001));
        let light_dir = delta / distance;
        let reach = min(select(pos.w, 2.0 * colour.w, colour.w >= 0.5), 30.0);
        let attenuation = pow(1.0 - smoothstep(0.0, max(reach, 0.01), distance), 2.0);
        rgb += colour.rgb * pow(max(dot(n, normalize(light_dir + to_view)), 0.0), 64.0)
            * max(dot(n, light_dir), 0.0) * attenuation * 0.7;
    }

    // Patchy, low-frequency coverage, never a texture-derived white outline.
    let noise = 0.7 * foam_noise(p * 0.6 + t * vec2<f32>(0.025, -0.018))
        + 0.3 * foam_noise(p * 1.7 - t * vec2<f32>(0.014, 0.021));
    let breakup = smoothstep(0.40, 0.72, noise);
    // MONKEY (water): WarcraftXL's white belongs to a connected horizontal fold, not to every
    // steep normal. Threshold jitter stops identical crests drawing identical white contours.
    let fold_jitter = (noise - 0.5) * 0.04;
    // MONKEY (visualfix): a slow, domain-warped gust field (tens of yards, unrelated to the wave
    // lengths) moves the break threshold and calms whole patches, so the caps do not repeat.
    let gust_warp = vec2<f32>(foam_noise(p * 0.013 + vec2<f32>(t * 0.004, 7.3)),
        foam_noise(p * 0.013 + vec2<f32>(-4.1, t * 0.003))) - vec2<f32>(0.5);
    let gust = foam_noise(p * 0.037 + gust_warp * 2.3 + t * vec2<f32>(0.006, -0.005));
    let cap_lo = 0.13 + (0.5 - gust) * 0.16;
    // A crest breaks in short runs, not along its whole length: ~6 yd warped segments.
    let segment = smoothstep(0.42, 0.62,
        foam_noise(p * 0.16 + gust_warp * 1.7 + t * vec2<f32>(0.011, 0.007)));
    let breaking = smoothstep(cap_lo, cap_lo + 0.14, open_sea_fold + fold_jitter)
        * smoothstep(0.25, 0.6, gust) * segment;
    let whitecap_alpha = FOAM_WHITECAP * breaking * mix(0.55, 1.0, breakup)
        * smoothstep(0.55, 1.0, energy);
    // The cached derivatives from the top of the function — same expression as before, one tap.
    let depth_gradient = length(vec2<f32>(ddx_depth, ddy_depth))
        / max(length(vec2<f32>(length(ddx_p), length(ddy_p))), 0.001);
    let wall_suppression = mix(1.0, 0.3, smoothstep(0.8, 3.0, depth_gradient));
    let contact = smoothstep(0.0, 0.025, thickness)
        * (1.0 - smoothstep(0.055, 0.15, thickness));
    let contact_alpha = FOAM_WET_EDGE * contact * wall_suppression * breakup;
    // MONKEY (beach foam, third pass). The owner called the previous surf ugly: it was the crest
    // line chopped into DASHES by a breakup noise, plus swash "crescents" stamped from an 8 yd patch
    // grid - rows of white blobs. Surf is not dashes. A breaking wave is (1) a thin, nearly
    // continuous bright FRONT, (2) a LACE of foam left behind it that thins out and dissolves, and
    // (3) a sheet that runs up the sand after each arrival and fizzles. All three are driven by the
    // shore train's own phase, so the foam sits on the wave the normals show.
    //
    // `surf_u` is the fraction of a wave spacing BEHIND the front (the phase rises with depth, so
    // behind = seaward): 0 at the front, which sits just ahead of the crest the normals draw.
    let surf_u = fract(shore_g + t / SHORE_PERIOD - 0.20);
    // Web-like lace: ridged noise (bright along the zero set of two noise fields), two octaves.
    let lace_a = 1.0 - abs(2.0 * foam_noise(p * 1.25 + t * vec2<f32>(0.030, -0.020)) - 1.0);
    let lace_b = 1.0 - abs(2.0 * foam_noise(p * 3.30 - t * vec2<f32>(0.020, 0.035)) - 1.0);
    let lace = 0.62 * lace_a + 0.38 * lace_b;
    // The lace dissolves with age: the threshold climbs from "most of it" to "only the ridges".
    let dissolve = mix(0.42, 0.93, smoothstep(0.02, 0.60, surf_u));
    let lace_mask = smoothstep(dissolve, dissolve + 0.16, lace);
    // The front itself: crisp on its shoreward side, solid for a few percent of a spacing.
    let front = smoothstep(0.0, 0.018, surf_u) * (1.0 - smoothstep(0.035, 0.11, surf_u));
    let trail = smoothstep(0.0, 0.03, surf_u) * (1.0 - smoothstep(0.25, 0.70, surf_u));
    // Strength wanders along the beach, but never to nothing - no gaps, no dashes.
    let along = mix(0.55, 1.0, foam_noise(p * 0.055 + t * vec2<f32>(0.008, 0.005)));
    let break_zone = smoothstep(0.10, 0.30, vertical_depth)
        * (1.0 - smoothstep(1.3, 2.6, vertical_depth)) * on_bed;
    let crest_alpha = select(0.0,
        FOAM_CREST * break_zone * along * max(front, 0.85 * trail * lace_mask), ocean_mesh)
        * wall_suppression * energy;

    // The swash: what is left of each wave runs up the last hand of water and fizzles. Its clock
    // is the front's arrival at the break (0.22 yd of water): 0 when it lands, 1 as the next does.
    let land_g = (0.22 + SHORE_COMPRESS * (1.0 - exp(-0.22 / SHORE_COMPRESS_D))) / SHORE_LAMBDA_D;
    let swash_age = fract(land_g + t / SHORE_PERIOD - 0.20 + 0.10 * (along - 0.75));
    let swash_life = smoothstep(0.0, 0.06, swash_age) * (1.0 - smoothstep(0.30, 0.95, swash_age));
    let swash_band = (1.0 - smoothstep(0.10, 0.34, vertical_depth + 0.5 * shore_warp)) * on_bed;
    let swash_thin = mix(0.30, 0.90, smoothstep(0.05, 0.85, swash_age));
    let swash_lace = smoothstep(swash_thin, swash_thin + 0.18, lace);
    // The very lip of the water keeps a thin bright line while the sheet is alive.
    let lip = 1.0 - smoothstep(0.015, 0.07, vertical_depth);
    let arcs_alpha = select(0.0,
        FOAM_SWASH * swash_life * swash_band * max(swash_lace, 0.8 * lip), ocean_mesh)
        * wall_suppression * energy;

    let foam = min(FOAM_MAX, contact_alpha + arcs_alpha + crest_alpha + whitecap_alpha);
    let illumination = water_light[0].rgb + water_light[1].rgb;
    let foam_luma = clamp(dot(illumination, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.22, 1.0);
    // Matte foam composites on top of reflection with its own coverage.
    // The body now carries its own transmission (above), so the surface is opaque bar the very
    // edge, where it hands over to the real frame to keep the waterline free of a seam.
    let body_alpha = smoothstep(0.0, 0.18, vertical_depth);
    let foam_alpha = smoothstep(0.0, 0.045, vertical_depth) * foam;
    let alpha = foam_alpha + body_alpha * (1.0 - foam_alpha);
    // Foam is white UNDER the light it stands in: half the way to the light's own colour, so a
    // dusk surf is warm and a moonlit one blue-grey, not a neutral paste.
    rgb = fogged_base + (rgb - fogged_base) * fog.w;
    let foam_tint = mix(fog.rgb, mix(vec3<f32>(foam_luma),
        clamp(illumination, vec3<f32>(0.22), vec3<f32>(1.0)), 0.5), fog.w);
    rgb = (foam_tint * foam_alpha + rgb * body_alpha * (1.0 - foam_alpha))
        / max(alpha, 0.0001);
    // Already fogged, term by term (see `water_fog`): the caller must not fog it again.
    return vec4<f32>(rgb, alpha);
}
