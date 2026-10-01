// MONKEY (sky): the Enhanced/High sky library (`skyQuality` ≥ 1), imported by `sky.wgsl` and
// `cloud.wgsl`. Classic never calls into it. Colour math is in linear light; the callers convert
// the gamma stops in and the result back out.
//
// The cloud detail (`cloud_billow`, `cloud_detail`):
// Ported from WarcraftXL (https://github.com/WarcraftXL) by iThorgrim — module wxl-retail-clouds, Clouds.cpp, Clouds.hpp.
#define_import_path benilla_world::sky_fx

const TAU: f32 = 6.2831853;
// MONKEY (fix-sky): the sky clock's wrap (`sky_fx::SKY_CLOCK_WRAP_S`, a day). Twinkle rates are
// whole cycles per wrap, so the stars never jump at the wrap.
const SKY_WRAP: f32 = 86400.0;

// The phase of a `rate` rad/s oscillator, rounded to whole cycles per wrap.
fn wrap_phase(t: f32, rate: f32) -> f32 {
    let cycles = round(rate * SKY_WRAP / TAU);
    return fract(cycles * (t / SKY_WRAP)) * TAU;
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((max(c, vec3<f32>(0.0)) + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

// ---- hashes and noise -------------------------------------------------------------------------

// pcg3d (Jarzynski & Olano 2020): three well-mixed u32 from three.
fn pcg_mix(v_in: vec3<u32>) -> vec3<u32> {
    var v = v_in * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v = v ^ (v >> vec3<u32>(16u));
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

fn hash_three(p: vec3<i32>) -> vec3<f32> {
    return vec3<f32>(pcg_mix(bitcast<vec3<u32>>(p)) >> vec3<u32>(8u)) * (1.0 / 16777216.0);
}

fn hash_cell(p: vec2<i32>, salt: u32) -> f32 {
    return f32(pcg_mix(vec3<u32>(bitcast<vec2<u32>>(p), salt)).x >> 8u) * (1.0 / 16777216.0);
}

// Value noise with the quintic fade: the cubic's second-derivative seam shows as square plateaus
// on a slowly drifting sky (the WarcraftXL note).
// MONKEY (polish): the lattice repeats every `period` cells, so a shift by `period` is no change.
fn value_noise_plane(p: vec2<f32>, salt: u32, period: i32) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let per = vec2<i32>(period);
    let lo = ((i % per) + per) % per;
    let hi = (lo + vec2<i32>(1)) % per;
    let a = hash_cell(lo, salt);
    let b = hash_cell(vec2<i32>(hi.x, lo.y), salt);
    let c = hash_cell(vec2<i32>(lo.x, hi.y), salt);
    let d = hash_cell(hi, salt);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn value_noise_space(p: vec3<f32>) -> f32 {
    let i = vec3<i32>(floor(p));
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let noise_aaa = hash_three(i).x;
    let noise_baa = hash_three(i + vec3<i32>(1, 0, 0)).x;
    let noise_aba = hash_three(i + vec3<i32>(0, 1, 0)).x;
    let noise_bba = hash_three(i + vec3<i32>(1, 1, 0)).x;
    let noise_aab = hash_three(i + vec3<i32>(0, 0, 1)).x;
    let noise_bab = hash_three(i + vec3<i32>(1, 0, 1)).x;
    let noise_abb = hash_three(i + vec3<i32>(0, 1, 1)).x;
    let noise_bbb = hash_three(i + vec3<i32>(1, 1, 1)).x;
    let interp_aa = mix(noise_aaa, noise_baa, u.x);
    let interp_ba = mix(noise_aba, noise_bba, u.x);
    let interp_ab = mix(noise_aab, noise_bab, u.x);
    let interp_bb = mix(noise_abb, noise_bbb, u.x);
    return mix(mix(interp_aa, interp_ba, u.y), mix(interp_ab, interp_bb, u.y), u.z);
}

fn fbm_space(p: vec3<f32>) -> f32 {
    var s = 0.0;
    var a = 0.5;
    var q = p;
    for (var o = 0; o < 3; o++) {
        s += a * value_noise_space(q);
        q = q * 2.03 + vec3<f32>(1.7, 9.2, 3.4);
        a *= 0.5;
    }
    return s / 0.875;
}

// ---- S1: the smooth gradient ------------------------------------------------------------------

// The reference's ring elevations (degrees), horizon to zenith.
const STOP_ELEV = array<f32, 6>(0.0, 1.8, 3.7, 9.8, 16.8, 90.0);

// Fritsch–Carlson tangent at an interior stop: 0 at a local extremum, else the weighted harmonic
// mean of the two secants, so the curve never overshoots a stop (no halos, no new colours).
fn pchip_tangent(first_delta: vec3<f32>, second_delta: vec3<f32>, first_span: f32, second_span: f32) -> vec3<f32> {
    let left_weight = 2.0 * second_span + first_span;
    let right_weight = second_span + 2.0 * first_span;
    let left_safe = select(first_delta, vec3<f32>(1.0), first_delta == vec3<f32>(0.0));
    let right_safe = select(second_delta, vec3<f32>(1.0), second_delta == vec3<f32>(0.0));
    let m = (left_weight + right_weight) / (left_weight / left_safe + right_weight / right_safe);
    return select(vec3<f32>(0.0), m, first_delta * second_delta > vec3<f32>(0.0));
}

// Monotone cubic through the six stops (`y[0]` = fog at 0°, `y[5]` = zenith), linear light. The
// end tangents are 0: flat into the fog band below the horizon and flat over the zenith, so the
// curve is C1 everywhere and the rings leave no Mach bands.
fn smooth_gradient(elev: f32, y: array<vec3<f32>, 6>) -> vec3<f32> {
    if (elev <= 0.0) {
        return y[0];
    }
    if (elev >= 90.0) {
        return y[5];
    }
    var yy = y;
    var se = STOP_ELEV;
    var d: array<vec3<f32>, 5>;
    var h: array<f32, 5>;
    for (var i = 0; i < 5; i++) {
        h[i] = se[i + 1] - se[i];
        d[i] = (yy[i + 1] - yy[i]) / h[i];
    }
    var k = 0;
    for (var i = 1; i < 5; i++) {
        if (elev >= se[i]) {
            k = i;
        }
    }
    var tangent_start = vec3<f32>(0.0);
    var tangent_end = vec3<f32>(0.0);
    if (k > 0) {
        tangent_start = pchip_tangent(d[k - 1], d[k], h[k - 1], h[k]);
    }
    if (k < 4) {
        tangent_end = pchip_tangent(d[k], d[k + 1], h[k], h[k + 1]);
    }
    let hk = h[k];
    let t = (elev - se[k]) / hk;
    let t_sq = t * t;
    let t_cube = t_sq * t;
    let blend_start = 2.0 * t_cube - 3.0 * t_sq + 1.0;
    let blend_start_tangent = t_cube - 2.0 * t_sq + t;
    let blend_end = -2.0 * t_cube + 3.0 * t_sq;
    let blend_end_tangent = t_cube - t_sq;
    return blend_start * yy[k] + blend_start_tangent * hk * tangent_start
        + blend_end * yy[k + 1] + blend_end_tangent * hk * tangent_end;
}

// Triangular dither of ±1 LSB at 8 bits, fixed per pixel (no temporal shimmer).
fn dither_tri(frag: vec2<f32>) -> f32 {
    let h = hash_three(vec3<i32>(vec2<i32>(frag), 7));
    return (h.x + h.y - 1.0) / 255.0;
}

// ---- S4: the sun glow -------------------------------------------------------------------------

// Henyey–Greenstein phase normalised to 1 toward the sun.
fn hg_norm(c: f32, g: f32) -> f32 {
    let g_sq = g * g;
    let a = 1.0 + g_sq - 2.0 * g * c;
    let a_min = (1.0 - g) * (1.0 - g);
    return pow(a_min / a, 1.5);
}

// A two-lobe Mie-like halo: a broad aureole plus a tighter core round the disc.
fn sun_glow(dir: vec3<f32>, sun: vec3<f32>) -> f32 {
    let c = dot(dir, sun);
    return 0.55 * hg_norm(c, 0.70) + 0.45 * hg_norm(c, 0.93);
}

// ---- S5: the night sky ------------------------------------------------------------------------

// Grid cells per unit radius: one cell spans ~0.3°, so a candidate star every few pixels.
const STAR_GRID: f32 = 180.0;

// A procedural star field fixed to world directions. Each grid cell holds at most one star (a
// density draw), jittered inside the cell; the eight cells round the sample are tested so no star
// is cut at a cell face. Brightness is a steep power law (few bright, many faint), the footprint
// is at least a pixel wide so a star does not vanish between pixels, and a slow twinkle is
// stronger low in the sky. `px` is the pixel's size in grid units.
fn star_field(dir: vec3<f32>, t: f32, px: f32, band: f32) -> vec3<f32> {
    let p = dir * STAR_GRID;
    let base = floor(p);
    let f = p - base;
    let o = select(vec3<f32>(-1.0), vec3<f32>(1.0), f > vec3<f32>(0.5));
    let density = 0.022 * (1.0 + 2.5 * band);
    var acc = vec3<f32>(0.0);
    for (var i = 0u; i < 8u; i++) {
        let off = vec3<f32>(f32(i & 1u), f32((i >> 1u) & 1u), f32((i >> 2u) & 1u)) * o;
        let cell = vec3<i32>(base + off);
        let h = hash_three(cell);
        if (h.x > density) {
            continue;
        }
        let j = hash_three(cell + vec3<i32>(7919, 104729, 1299709));
        let centre = normalize(vec3<f32>(cell) + 0.5 + (j - 0.5) * 0.9) * STAR_GRID;
        let d = length(p - centre);
        let mag = pow(h.y, 22.0);
        let b = 0.035 + 0.965 * mag;
        let sigma = px * (0.55 + 0.7 * mag);
        let spot = exp(-d * d / (2.0 * sigma * sigma));
        let tw_amp = 0.18 + 0.3 * (1.0 - clamp(dir.y, 0.0, 1.0));
        let tw = 1.0 + tw_amp * sin(wrap_phase(t, 1.3 + 3.1 * j.x) + TAU * j.y)
            * sin(wrap_phase(t, 0.6 + 1.7 * j.z) + TAU * h.z);
        // Star colour: mostly white, a few warm and a few blue-white.
        let tint = mix(vec3<f32>(0.78, 0.86, 1.0), vec3<f32>(1.0, 0.88, 0.72), h.z);
        acc += tint * (b * spot * tw);
    }
    return acc;
}

// The Milky Way's plane in Bevy space (y up): tilted so the band arcs across the dome.
const GALAXY_N: vec3<f32> = vec3<f32>(0.40, 0.52, -0.754);
const GALAXY_CORE: vec3<f32> = vec3<f32>(-0.62, 0.36, -0.08);

// Band weight in [0, 1] toward `dir`.
fn galaxy_band(dir: vec3<f32>) -> f32 {
    let b = dot(dir, normalize(GALAXY_N));
    return exp(-b * b / (2.0 * 0.15 * 0.15));
}

// A faint, mottled band with a dark dust lane down its middle and a brighter core.
fn milky_way(dir: vec3<f32>, band: f32) -> vec3<f32> {
    if (band < 0.01) {
        return vec3<f32>(0.0);
    }
    let b = dot(dir, normalize(GALAXY_N));
    let cloud = fbm_space(dir * 5.0);
    let lane = exp(-(b - 0.02) * (b - 0.02) / (2.0 * 0.03 * 0.03)) * smoothstep(0.35, 0.7, fbm_space(dir * 9.0 + 11.0));
    let core = 1.0 + 1.2 * pow(max(dot(dir, normalize(GALAXY_CORE)), 0.0), 4.0);
    let v = band * (0.3 + 0.7 * cloud) * (1.0 - 0.75 * lane) * core;
    return vec3<f32>(0.72, 0.8, 1.0) * v;
}

// ---- S3: cloud detail -------------------------------------------------------------------------

// MONKEY (polish): the detail field loops over the sky clock's wrap. It tiles every `CLOUD_TILE`
// cells of the detail plane (octave o every `CLOUD_TILE << o` of its own lattice; the half-rate
// warp every `CLOUD_TILE / 2`), and the drift covers whole tiles per wrap, so the field at
// `SKY_WRAP` is the field at 0. One tile is 64 / 36 of the sheet, wider than the dome's UV.
const CLOUD_TILE: i32 = 64;
// Tiles of drift per wrap on u and v: 0.01111 and 0.00444 cells/s, WarcraftXL's 0.011 and 0.0043
// rounded to whole tiles.
const CLOUD_DRIFT_TILES: vec2<f32> = vec2<f32>(15.0, 6.0);

// WarcraftXL's billow blend: fold each octave toward rounded puffs by `cotton` (0.6).
fn billow(v: f32) -> f32 {
    return v + (1.0 - abs(2.0 * v - 1.0) - v) * 0.6;
}

// Billowed fbm, amplitude 0.55 per octave as in WarcraftXL.
fn cloud_billow(p: vec2<f32>, octaves: i32) -> f32 {
    var s = 0.0;
    var a = 1.0;
    var tot = 0.0;
    var q = p;
    for (var o = 0; o < octaves; o++) {
        s += a * billow(value_noise_plane(q, u32(o) * 57u, CLOUD_TILE << u32(o)));
        tot += a;
        a *= 0.55;
        q = q * 2.0 + vec2<f32>(3.7, 1.9);
    }
    return s / tot;
}

// The detail field in [0, 1] on the sheet: a broad warp twists the domain before the billowed fbm
// is read, which turns thresholded blobs into cauliflower edges.
fn cloud_detail(uv: vec2<f32>, t: f32, octaves: i32) -> f32 {
    // The drift modulo one tile: seamless at the wrap, and small enough to keep f32 precision.
    let drift = fract(CLOUD_DRIFT_TILES * (t / SKY_WRAP)) * f32(CLOUD_TILE);
    let p = uv * 36.0 + drift;
    let half_tile = CLOUD_TILE / 2;
    let w = vec2<f32>(
        value_noise_plane(p * 0.5, 173u, half_tile),
        value_noise_plane(p * 0.5 + 5.3, 219u, half_tile),
    ) - 0.5;
    return cloud_billow(p + w * 2.6, octaves);
}

// MONKEY (visualfix): the mean of `cloud_detail` (4 octaves; 0.573 measured over 1e6 samples of
// the quintic value noise through `billow`). `cloud_erode` is centred on 0.5.
const CLOUD_DETAIL_MEAN: f32 = 0.573;

// Apply the detail to a base coverage: the edges (mid coverage) are pushed in and out by the
// detail, solid cores and clear sky stay put, so the CPU tile still says where cloud is.
fn cloud_erode(a: f32, n: f32) -> f32 {
    // MONKEY (visualfix): the weight peaks at a = 1/3 (thin edges) and is 0.43 at a = 0.7, where
    // 4a(1-a) gave 0.84: one low base-octave cell (~5 degrees) punched a round hole in a dense
    // mass, which read as a ring round the glow.
    let edge = 6.75 * a * (1.0 - a) * (1.0 - a);
    return clamp(a + (n - 0.5) * 1.2 * edge, 0.0, 1.0);
}
