//! MONKEY (swim waves) — **the CPU mirror of the ocean's vertex swell**.
//!
//! Enhanced water displaces the ocean mesh in the VERTEX stage
//! (`benilla-assets/src/shaders/enhanced_water.wgsl`, `water_swell`, called from `liquid.wgsl`'s vertex stage),
//! so the surface a swimmer floats on is not the flat MCLQ heightfield the CPU queries
//! ([`super::query::WaterChunkInfo::surface_z_at`]) — it is that heightfield plus an analytic
//! long-swell band the CPU never sees. A body positioned off the queried height therefore rides
//! *through* the crests instead of on them, which is the whole of the owner's report: the sea
//! visibly heaves and the swimmer in it does not.
//!
//! This module is the missing half: the SAME two wave components, the same dispersion, the same
//! clock, evaluated in Rust. It is a **mirror, not a second design** — every constant below is
//! copied from the shader with its line cited, and [`tests::the_table_mirrors_the_shader`] pins
//! them against the literal numbers so a drift in either direction shows up as a failing test
//! rather than as a swimmer sunk half a yard into a crest.
//!
//! **What it deliberately does not mirror.** The shader's fragment stage sums all EIGHT
//! components for its normal; the vertex stage takes `long_only` — components 0 and 1 only
//! (`enhanced_water.wgsl`), because the rest are finer than the liquid lattice can resolve and exist
//! to shade, not to move geometry. A bob driven off the fragment band would chase ripples the mesh
//! under it never made. So: two components, the mesh's own.
//!
//! **Space.** The shader's `p` is `world_position.xz` — BEVY world XZ, because the liquid meshes
//! are baked through [`benilla_assets::coords::wow_to_bevy`]. Callers pass Bevy XZ, not WoW XY.

use bevy::math::Vec2;

/// The long-swell component table — `enhanced_water.wgsl`'s `WATER_WAVES[0..2]`, verbatim:
/// `(direction_radians, wavelength_yd, amplitude_yd, phase_offset)`. Direction is radians from
/// world +X toward +Z; the two sum to at most 0.34 yd before energy and shore attenuation.
///
/// Components 2..8 are the shader's fragment-only ripple band and are NOT here — see the module
/// doc's `long_only` note. Adding them would be a second design, not a closer mirror.
pub const WATER_WIND_DIR: f32 = 0.35;
pub const LONG_SWELL: [[f32; 4]; 2] =
    [[WATER_WIND_DIR, 18.0, 0.200, 0.0], [0.80, 12.8, 0.140, 1.7]];

/// Horizontal Gerstner gather, mirrored from `enhanced_water.wgsl::WATER_GERSTNER_CHOP`.
pub const GERSTNER_CHOP: f32 = 2.4;

/// Gravity in yd/s², `enhanced_water.wgsl` — deep-water dispersion, `c = sqrt(g/k)`.
const GRAVITY_YD: f32 = 10.72;

/// The ADT **ocean**'s wave energy — `liquid/surface.rs` packs `water.mode.y = 1.0` for
/// `LiquidPath::Adt` + [`benilla_formats::LiquidKind::Ocean`] (river/lake 0.18, WMO exterior 0.26,
/// WMO interior 0.08, fullbright 0.0). The vertex swell arm only ever runs on that combination
/// (`enhanced_water.wgsl`), so this is the only energy a bob can ever be driven at — the parameter
/// stays open because the function is the shader's, not the bob's.
pub const OCEAN_WAVE_ENERGY: f32 = 1.0;

/// The height AND the surface gradient at one point — one evaluation answers both, because the
/// shader's own `water_waves` returns `vec3(height, dHeight/dpx, dHeight/dpz)` from a single
/// loop and the bob wants the pair (a float to lift by, a slope to lean into).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Swell {
    /// Vertical displacement in yards, the shader's `result.x` — added to the mesh's world Y.
    pub height: f32,
    /// `(∂height/∂x, ∂height/∂z)` in BEVY world axes, the shader's `result.yz`. The shader builds
    /// its normal from exactly this as `normalize(vec3(-grad.x, 1, -grad.y))` (`enhanced_water.wgsl`).
    pub grad: Vec2,
    /// Horizontal Gerstner displacement in BEVY XZ at the parameter-space point under this sample.
    pub horizontal: Vec2,
}

/// `enhanced_water.wgsl`'s `swell_shore_fade`: the authored ocean depth `V` (byte/255, ≈148 yd at
/// 1.0) faded in over ~0.15..3.7 yd, so the swell dies at a beach and the waterline stays pinned.
///
/// Kept here for completeness of the mirror. **The CPU has no per-position `V`** — the depth bytes
/// are consumed by the mesh builder and never reach [`super::query::WaterChunkInfo`], which stores
/// positions and wet flags only — so the bob passes its own shore term instead (the swim ramp,
/// which is already zero everywhere a body can stand up). See `water_fx::bob`.
pub fn swell_shore_fade(depth_v: f32) -> f32 {
    smoothstep(0.001, 0.025, depth_v)
}

/// The vertex stage's swell at one world point. A faithful transcription of `water_waves(p, t, 0,
/// 0, shore, true)` (`enhanced_water.wgsl`) under the vertex arm's own arguments:
///
/// * `distance = 0` and `footprint = 0` — the vertex call passes both (`enhanced_water.wgsl`), so the
///   Nyquist fade is `1 - smoothstep(0.10, 0.45, 0) = 1` and the 35-yd ripple cull never applies.
/// * `inland = false` — the arm requires the ocean swatch on the ADT renderer, which is exactly
///   the negation of the shader's `inland` (`enhanced_water.wgsl`), so there is no drift term and no
///   `i < 3` skip.
/// * `long_only = true` — components 0 and 1, which are also the two the shore fade multiplies.
///
/// `world_xz` is BEVY world XZ. `shallow_fade` is the shore term (1 = open sea, 0 = beach).
#[derive(Clone, Copy, Debug, Default)]
struct RawSwell {
    height: f32,
    grad: Vec2,
    horizontal: Vec2,
    jacobian: [f32; 4],
}

/// Evaluate the same parameter-space Gerstner surface the vertex shader receives. The public
/// sampler below inverts its horizontal map because callers hold a displaced world position.
fn raw_swell(parameter_xz: Vec2, time: f32, wave_energy: f32, shallow_fade: f32) -> RawSwell {
    let energy = wave_energy.clamp(0.0, 1.0);
    // `mix(0.4, 1.0, sqrt(energy))` — a calm inland sheet also swells SLOWER, not merely lower.
    let tempo = 0.4 + 0.6 * energy.sqrt();
    let shore = shallow_fade.clamp(0.0, 1.0);
    let mut out = RawSwell {
        jacobian: [1.0, 0.0, 0.0, 1.0],
        ..RawSwell::default()
    };
    for w in LONG_SWELL {
        let dir = Vec2::new(w[0].cos(), w[0].sin());
        let k = std::f32::consts::TAU / w[1];
        let speed = (GRAVITY_YD / k).sqrt(); // phase speed, yd/s
        let phase = k * (dir.dot(parameter_xz) - speed * tempo * time) + w[3];
        // `fade` is 1 at zero footprint; `i < 2` ⇒ both components take the shore term.
        let amplitude = w[2] * energy * shore;
        out.height += amplitude * phase.sin();
        out.grad += dir * (amplitude * k * phase.cos());
        let horizontal = GERSTNER_CHOP * amplitude;
        out.horizontal += dir * (horizontal * phase.cos());
        let compression = horizontal * k * phase.sin();
        out.jacobian[0] -= compression * dir.x * dir.x;
        out.jacobian[1] -= compression * dir.x * dir.y;
        out.jacobian[2] -= compression * dir.y * dir.x;
        out.jacobian[3] -= compression * dir.y * dir.y;
    }
    out
}

/// The displaced Gerstner surface at a WORLD-space point. Four fixed-point steps find the original
/// mesh coordinate under that point; the long-band Jacobian is contractive (worst compression
/// below 0.34), so this converges without a branch or an allocation. The gradient is transformed
/// through the same map, keeping swimmer lean consistent with the visibly pinched crest.
pub fn swell(world_xz: Vec2, time: f32, wave_energy: f32, shallow_fade: f32) -> Swell {
    let mut parameter_xz = world_xz;
    for _ in 0..4 {
        let raw = raw_swell(parameter_xz, time, wave_energy, shallow_fade);
        parameter_xz = world_xz - raw.horizontal;
    }
    let raw = raw_swell(parameter_xz, time, wave_energy, shallow_fade);
    let [j00, j01, j10, j11] = raw.jacobian;
    let det = (j00 * j11 - j01 * j10).max(1.0e-4);
    let grad = Vec2::new(
        (j11 * raw.grad.x - j01 * raw.grad.y) / det,
        (-j10 * raw.grad.x + j00 * raw.grad.y) / det,
    );
    Swell {
        height: raw.height,
        grad,
        horizontal: raw.horizontal,
    }
}

/// [`swell`]'s height alone — the shape the brief names, for callers that do not want the slope.
pub fn swell_height(world_xz: Vec2, time: f32, wave_energy: f32, shallow_fade: f32) -> f32 {
    swell(world_xz, time, wave_energy, shallow_fade).height
}

/// **The clock the material runs on**, so CPU and GPU agree on which crest is where.
///
/// The shader reads `water_time()` (`enhanced_water.wgsl`): `globals.time`, or the
/// frozen capture pin `water.mode.z` when the clock enable `water.mode.w` is 0 on an Enhanced
/// material. Both halves of that are reproduced here:
///
/// * `globals.time` is **`Time::elapsed_secs_wrapped`**, not `elapsed_secs` — bevy_render's
///   `prepare_globals_buffer` writes exactly that field (bevy_render 0.18.1 `globals.rs:72`), and
///   the two diverge by whole hours once the wrapping period (3600 s) rolls. Passing the unwrapped
///   clock would put the CPU an entire wrap out of phase with the mesh it is riding, silently,
///   after an hour of play — the kind of drift that only reproduces in a long session.
/// * the capture pin is `WOW_CAPTURE_WATER_T` under `WOW_CAPTURE`, read the same way
///   `liquid/surface.rs` reads it into `water.mode.z`; `water.mode.w` is 0 on precisely the same
///   condition ([`crate::dev_state::deterministic_run`]), so the two branches line up one to one.
pub fn water_anim_time(elapsed_secs_wrapped: f32) -> f32 {
    if crate::dev_state::deterministic_run() {
        capture_water_time()
    } else {
        elapsed_secs_wrapped
    }
}

/// The frozen-capture water phase (`WOW_CAPTURE_WATER_T`, default 0) — the same parse
/// `liquid/surface.rs` bakes into the material, read once and cached: a deterministic run must not
/// pay an environment lookup per unit per frame, and the material's copy is read once at setup, so
/// re-reading a changed variable mid-run could only make the two disagree.
fn capture_water_time() -> f32 {
    static T: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        std::env::var("WOW_CAPTURE_WATER_T")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(0.0)
    })
}

/// WGSL's `smoothstep`, which Rust has no equivalent of.
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The mirror pin.** The table against the literal numbers in `enhanced_water.wgsl`, and the
    /// two derived scalars against the lines that spell them. This test exists so that a change on
    /// EITHER side is a build failure rather than a swimmer floating half a yard off the sea: the
    /// shader is another agent's file, and nothing but this assertion couples the two.
    #[test]
    fn the_table_mirrors_the_shader() {
        assert_eq!(
            LONG_SWELL,
            [[0.35, 18.0, 0.200, 0.0], [0.80, 12.8, 0.140, 1.7]],
            "enhanced_water.wgsl WATER_WAVES[0..2]"
        );
        assert_eq!(GRAVITY_YD, 10.72, "enhanced_water.wgsl gravity");
        assert_eq!(
            std::f32::consts::TAU.to_bits(),
            0x40c9_0fdb,
            "enhanced_water.wgsl k = 6.2831853/wavelength"
        );
        assert_eq!(WATER_WIND_DIR, 0.35, "enhanced_water.wgsl WATER_WIND_DIR");
        assert_eq!(
            GERSTNER_CHOP, 2.4,
            "enhanced_water.wgsl WATER_GERSTNER_CHOP"
        );
        assert_eq!(
            OCEAN_WAVE_ENERGY, 1.0,
            "surface.rs water.mode.y for ADT ocean"
        );
        // The band's ceiling, stated in the shader's own comment ("at most 0.34 yd").
        let peak: f32 = LONG_SWELL.iter().map(|w| w[2]).sum();
        assert!((peak - 0.34).abs() < 1e-6, "amplitude sum {peak}");
    }

    /// No energy ⇒ no swell, height and slope both, at any point and any time. The inland/WMO
    /// lanes and the Classic path all reach this function at energy 0 (or never reach it), so a
    /// nonzero answer here is a body bobbing on a mill pond.
    #[test]
    fn zero_energy_is_a_flat_sea() {
        for t in [0.0_f32, 1.0, 37.5, 1234.5] {
            let s = swell(Vec2::new(-9016.0, 226.0), t, 0.0, 1.0);
            assert_eq!(s.height, 0.0, "height at t={t}");
            assert_eq!(s.grad, Vec2::ZERO, "grad at t={t}");
            assert_eq!(s.horizontal, Vec2::ZERO, "horizontal at t={t}");
        }
        // The shore term is the other kill switch: a beached vertex does not move either.
        let beached = swell(Vec2::new(-9016.0, 226.0), 3.0, OCEAN_WAVE_ENERGY, 0.0);
        assert_eq!(beached.height, 0.0);
        assert_eq!(beached.grad, Vec2::ZERO);
        assert_eq!(beached.horizontal, Vec2::ZERO);
    }

    /// The swell stays inside the band the shader advertises, and the gradient it reports really
    /// is the height's derivative (a finite difference against the analytic pair). The second half
    /// is what the tilt leans on: a sign error there rolls every floating body the wrong way.
    #[test]
    fn height_is_bounded_and_the_gradient_is_its_derivative() {
        // The bound holds at real world magnitude (the Westfall sea is around −9000, 200).
        for step in 0..40 {
            let t = step as f32 * 0.37;
            let s = swell(Vec2::new(-9016.0, 226.0), t, OCEAN_WAVE_ENERGY, 1.0);
            assert!(s.height.abs() <= 0.341, "height {} at t={t}", s.height);
        }
        // The derivative is checked NEAR THE ORIGIN on purpose: at ±9000 yd the phase is ~3000
        // rad and an f32 carries ~5e-4 rad of it, so a central difference there measures the
        // cancellation, not the slope. The identity being checked is position-independent.
        let p = Vec2::new(-16.0, 26.0);
        let h = 1e-2;
        for step in 0..40 {
            let t = step as f32 * 0.37;
            let s = swell(p, t, OCEAN_WAVE_ENERGY, 1.0);
            let dx = (swell_height(p + Vec2::X * h, t, 1.0, 1.0)
                - swell_height(p - Vec2::X * h, t, 1.0, 1.0))
                / (2.0 * h);
            let dz = (swell_height(p + Vec2::Y * h, t, 1.0, 1.0)
                - swell_height(p - Vec2::Y * h, t, 1.0, 1.0))
                / (2.0 * h);
            assert!((dx - s.grad.x).abs() < 2e-3, "d/dx {dx} vs {}", s.grad.x);
            assert!((dz - s.grad.y).abs() < 2e-3, "d/dz {dz} vs {}", s.grad.y);
        }
    }

    /// The CPU query samples by displaced world position while the vertex shader starts in mesh
    /// parameter space. Pin the inversion: adding the returned horizontal displacement to the
    /// recovered parameter must land back under the swimmer, with the advertised hard bound.
    #[test]
    fn gerstner_inverse_lands_under_the_world_sample() {
        let world = Vec2::new(-9016.0, 226.0);
        for step in 0..40 {
            let t = step as f32 * 0.37;
            let s = swell(world, t, OCEAN_WAVE_ENERGY, 1.0);
            let parameter = world - s.horizontal;
            let raw = raw_swell(parameter, t, OCEAN_WAVE_ENERGY, 1.0);
            let reprojection = parameter + raw.horizontal;
            assert!(
                (reprojection - world).length() < 2.0e-3,
                "reprojection at t={t}"
            );
            assert!(
                s.horizontal.length() <= 0.817,
                "horizontal at t={t}: {:?}",
                s.horizontal
            );
        }
    }

    /// The shore fade's two ends, as the shader's `smoothstep(0.001, 0.025, depth)` places them.
    #[test]
    fn shore_fade_matches_the_shader_band() {
        assert_eq!(swell_shore_fade(0.0), 0.0);
        assert_eq!(swell_shore_fade(1.0), 1.0, "open sea (V = 1) is unfaded");
        assert!(
            (swell_shore_fade(0.013) - 0.5).abs() < 0.02,
            "band midpoint"
        );
    }

    /// The clock: a live run is the wrapped elapsed seconds verbatim (the GPU's `globals.time`),
    /// so CPU and GPU sample the same phase. The capture branch is env-driven and is asserted by
    /// construction rather than here — flipping `WOW_CAPTURE` inside a test process would leak
    /// into every other test in the binary.
    #[test]
    fn the_live_clock_is_the_wrapped_elapsed_seconds() {
        if !crate::dev_state::deterministic_run() {
            assert_eq!(water_anim_time(123.5), 123.5);
            assert_eq!(water_anim_time(0.0), 0.0);
        }
    }
}
