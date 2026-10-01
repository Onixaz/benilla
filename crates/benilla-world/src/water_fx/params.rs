//! The `CWater0Ripple` driver math as pure functions: `0x5fa760`'s per-emission parameters, the
//! record lifecycle and the render texgen.

/// Ring pulse interval in seconds: `400 + U[0, 50)` ms (`0x5fac41`..`0x5fac53`, the roll
/// `(50·rng) >> 32` at `0x455c70`, millisecond ticks from `0x42b790`).
pub(super) const RING_INTERVAL: (f32, f32) = (0.4, 0.45);

/// Wake cooldown in seconds at `speed` yd/s: `k·625/min(speed, 20)` ms, one decal per 0.625 yd of
/// travel. `k` is uniform in [0.9, 1.1); the reference's spread is about ±15%, its distribution
/// untraced.
pub(super) fn wake_cooldown(speed: f32, rng: &mut u32) -> f32 {
    let k = 0.9 + 0.2 * rand01(rng);
    k * 0.625 / speed.clamp(0.1, 20.0)
}

/// MONKEY (swim waves) — **how swimming this body is**, `0` (wading, feet down) to `1` (off the
/// floor), from the feet-referenced depth and the unit's own collision height.
///
/// The latch itself is byte-verified and binary: `0x6030c0` sets `MOVEFLAG_SWIMMING` at
/// `depth > 0.75 · collisionHeight` (`player::swim::SWIM_DEPTH_FRAC`) with a 1/36-yd hysteresis.
/// Here it is deliberately a **band** rather than that edge, for two reasons. First, this crate
/// cannot see the flag at all: movement flags are the game's, and a streamed murloc's swim state
/// reaches `benilla-world` only as the depth under it — so this is the same law, evaluated where
/// the answer is needed. Second, the consumers are continuous: a bob and a foam gain that switch
/// on a hard edge pop at the exact depth a player spends the most time hovering around (the
/// shoreline), and a body crossing the line repeatedly would strobe.
///
/// The band's top is just above the latch so a genuinely swimming body is fully weighted; the
/// bottom is where the water is deep enough that a wader is mostly under it anyway.
pub(super) fn swim_ramp(depth: f32, h: f32) -> f32 {
    let (lo, hi) = (0.60 * h, 0.78 * h);
    let t = ((depth - lo) / (hi - lo).max(1.0e-4)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// MONKEY (swim waves) — **how loudly this record should read.** Classic is the reference and must
/// stay byte-identical, so every Enhanced multiplier in this file is reached through this one
/// struct and every one of them is inert at [`FoamLook::CLASSIC`].
///
/// The reference's foam is deliberately dim: its fragment is a full MODULATE against a stencil
/// whose RGB is near-black, and recolouring that stencil to white is what made the 20× "blown-out
/// white wedge" (see the module doc). So Enhanced does NOT touch the stencil either — the gain
/// rides the vertex colour, where 3.5 × a ~0.05 stencil is still ~0.18 and cannot blow out, and
/// the draw takes `EffectLighting::Scene` so the wake is tinted by the world's own light instead
/// of glowing at midnight.
#[derive(Clone, Copy, Default)]
pub(super) struct FoamLook {
    /// [`benilla_assets::WaterQuality`] ≥ 1 — the one gate. `false` ⇒ every field below is unused.
    pub(super) enhanced: bool,
    /// Is this body SWIMMING (as opposed to wading)? Swimming wakes are the ones the owner is
    /// looking at, so they get the length and the size; a wader's stay as they are.
    pub(super) swimming: bool,
}

impl FoamLook {
    /// The reference, unchanged — what every Classic frame and every test baseline takes.
    pub(super) const CLASSIC: FoamLook = FoamLook {
        enhanced: false,
        swimming: false,
    };
}

/// Enhanced readability multipliers, all inert in Classic. Sized to stay far under the reference's
/// own blow-out (the stencil's near-black RGB gives ~19× of headroom before white).
const ENHANCED_ALPHA_GAIN: f32 = 3.5;
/// A swimming body's wake lives ~1.4× longer and starts ~1.3× larger — a longer, wider V behind a
/// swimmer, where the reference's wade-tuned record dissolves almost before it is seen at swim
/// speed (a stroking swimmer moves at 4.7 yd/s, so the 0.625-yd distance law is re-arming every
/// ~130 ms and each record had 0.65 s to live).
const SWIM_LIFETIME_GAIN: f32 = 1.4;
const SWIM_SIZE_GAIN: f32 = 1.3;
/// The step-in/out splash gets a visibly bigger ring in Enhanced — it is the one record the eye is
/// guaranteed to be looking at, because the player just caused it.
const SPLASH_SIZE_GAIN: f32 = 1.8;
const SPLASH_GROWTH_GAIN: f32 = 1.3;

/// MONKEY (swim waves) — the **treading ring**: a soft, wide, slow ring a unit hanging in deep
/// water pushes out every [`TREAD_INTERVAL`] seconds, on its own cadence cell, in Enhanced only.
///
/// It is not the reference's standing ring with different numbers; it is a second, additive record
/// that exists because a swimmer treading water is the one state the reference's driver makes
/// almost invisible (the standing branch multiplies size by 0.6, growth by 0.25 and alpha by 0.8,
/// and then the depth attenuation halves all three again past half the gate — which surface
/// swimming always is). The ring below starts ~2.2× a standing one, grows slowly, and fades over
/// a long life: a slow expanding circle, not a pulse.
pub(super) const TREAD_INTERVAL: f32 = 0.9;
const TREAD_SIZE: f32 = 2.2;
const TREAD_GROWTH: f32 = 0.55;
const TREAD_LIFETIME: f32 = 1.6;
const TREAD_ALPHA: f32 = 0.5;

/// The treading ring's params for a body of this `scale`, or `None` when the look is Classic or
/// the body is not swimming. Independent of [`foam_params`] and of the shared cooldown cell — it
/// has its own timer, so it neither delays nor is delayed by the reference's cadence.
pub(super) fn tread_params(look: FoamLook, scale: f32, rng: &mut u32) -> Option<FoamParams> {
    if !look.enhanced || !look.swimming {
        return None;
    }
    let mut uni = |a: f32, b: f32| a + (b - a) * rand01(rng);
    let size0 = (scale * (1.0 / 3.0) * uni(0.9, 1.1)).clamp(1.0 / 3.0, 5.0 / 3.0) * TREAD_SIZE;
    Some(FoamParams {
        size0,
        growth: TREAD_GROWTH * uni(0.9, 1.1),
        lifetime: TREAD_LIFETIME * uni(0.9, 1.1),
        // Soft on its OWN terms: the Enhanced gain rides the vertex colour for every record
        // alike ([`FoamLook`]), so folding it in here too would double-count it and make the
        // "soft" ring the loudest thing in the bay.
        peak: TREAD_ALPHA,
        ring: true,
    })
}

/// What a unit is doing in the water this frame — the reference's two selection bits
/// (`MOVEMENTFLAGS & 0xf` / `& 0x30`), resolved.
#[derive(Clone, Copy)]
pub(super) enum WadeState {
    Translating { speed: f32, heading: f32 },
    Turning,
    Standing,
}

/// One record's emission parameters, as the driver computes them (`0x5fa760`).
pub(super) struct FoamParams {
    pub(crate) size0: f32,
    /// yd/s.
    pub(crate) growth: f32,
    /// s.
    pub(crate) lifetime: f32,
    /// Peak vertex alpha, `min(6 × driver alpha, 1)` (`0x68be62`).
    pub(crate) peak: f32,
    /// Render category: ring (`splash.blp`) vs wake (`wake.blp`).
    pub(crate) ring: bool,
}

/// The driver's parameter computation. `scale` = `OBJECT_FIELD_SCALE_X`; `gate` = the depth
/// gate `max(2 × collisionHeight, 1.0)` (`[unit+0x297]` = CMovement+0xb4, ≈4.06 yd for a human
/// — decision 0489); `depth` = surface − feet (yd, > 0 in water). `None` when the depth gate
/// rejects: not in water, or dived deeper than ~2 body heights — surface swimming (rest depth
/// ~0.75·h) stays well inside and emits.
/// `look` is MONKEY (swim waves)'s Enhanced readability pass — [`FoamLook::CLASSIC`] reproduces
/// the byte-verified reference exactly, and is what every Classic frame and every baseline test
/// passes.
pub(super) fn foam_params(
    state: WadeState,
    oneshot: bool,
    scale: f32,
    gate: f32,
    depth: f32,
    look: FoamLook,
    rng: &mut u32,
) -> Option<FoamParams> {
    if depth <= 0.0 || depth >= gate {
        return None;
    }
    let mut uni = |a: f32, b: f32| a + (b - a) * rand01(rng);
    let mut size0 = (scale * (1.0 / 3.0) * uni(0.9, 1.1)).clamp(1.0 / 3.0, 5.0 / 3.0);
    let mut lifetime = uni(0.6, 0.7);
    let mut growth = uni(1.0, 1.5);
    let mut alpha = 1.0 / 6.0;
    let ring = oneshot || !matches!(state, WadeState::Translating { .. });
    match (oneshot, state) {
        (false, WadeState::Translating { speed, .. }) => {
            growth *= speed.min(20.0) / 2.5;
        }
        (false, WadeState::Standing) => {
            alpha *= 0.8;
            growth *= 0.25;
            size0 *= 0.6;
        }
        // Turning and the one-shot take the unreduced ring params: the driver's standing branch
        // skips them, and `flagtable[1|3] = 0` keeps them ring-textured.
        _ => {}
    }
    // Past half the gate depth, alpha, lifetime and size0 (never growth) ramp down toward ×0.5.
    let half = gate * 0.5;
    if depth > half {
        let k = 0.5 + 0.5 * (gate - depth) / half;
        alpha *= k;
        lifetime *= k;
        size0 *= k;
    }
    // MONKEY (swim waves) — the Enhanced readability pass, LAST so it scales the reference's own
    // answer (including that depth attenuation, which surface swimming always takes: a swimmer
    // sits past half the gate by construction, so without this a swim wake is the dimmest,
    // shortest-lived record the driver can emit). Inert at `FoamLook::CLASSIC`.
    if look.enhanced && look.swimming {
        lifetime *= SWIM_LIFETIME_GAIN;
        size0 *= SWIM_SIZE_GAIN;
    }
    if look.enhanced && oneshot {
        size0 *= SPLASH_SIZE_GAIN;
        growth *= SPLASH_GROWTH_GAIN;
    }
    Some(FoamParams {
        size0,
        growth,
        lifetime,
        peak: (6.0 * alpha).min(1.0),
        ring,
    })
}

/// MONKEY (swim waves) — the **vertex-colour gain** an Enhanced foam draw carries, `1.0` in
/// Classic. The one brightness knob, and it is on the vertex colour rather than on `peak` for a
/// structural reason: a wake's `peak` is already `min(6 × 1/6, 1)` = **1.0**, so there is no
/// headroom left in the alpha at all — the reference's dimness lives entirely in the stencil's
/// near-black RGB, which the fragment MODULATEs against this. 3.5 × ~0.05 is ~0.18: brighter by a
/// factor the eye reads immediately, and still an order of magnitude short of the white wedge.
pub(super) fn foam_gain(enhanced: bool) -> f32 {
    if enhanced {
        ENHANCED_ALPHA_GAIN
    } else {
        1.0
    }
}

/// A live record's size at `now`: `size0 + growth · age`. (The reference render integrates one
/// `dt·growth` step *before* a record's first draw — visible at a capture's ~20 fps, negligible
/// at ours; we draw `size0` on the first frame.)
pub(super) fn record_size(size0: f32, growth: f32, born: f32, now: f32) -> f32 {
    size0 + growth * (now - born)
}

/// A record's alpha at `now`: up to `peak` over the first 0.4 of its life and down to 0 over the
/// rest (rate table `0x810348`, both categories).
pub(super) fn record_alpha(peak: f32, lifetime: f32, born: f32, now: f32) -> f32 {
    let age = (now - born) / lifetime;
    if age <= 0.4 {
        peak * (age / 0.4).max(0.0)
    } else {
        peak * (1.0 - (age - 0.4) / 0.6).max(0.0)
    }
}

/// The texgen `uv = Rz(heading − π/2)·(p − center)/(2s) + 0.5`, with the −π/2 folded into the
/// heading: `u` runs across the track and `v` against the heading, so the wake chevron's apex
/// (low `v` in `wake.blp`) lands ahead of the unit.
pub(super) fn foam_uv(center: [f32; 2], heading: f32, size: f32, p: [f32; 2]) -> [f32; 2] {
    let (dx, dy) = (p[0] - center[0], p[1] - center[1]);
    let (s, c) = heading.sin_cos();
    let inv = 1.0 / (2.0 * size);
    [
        (-s * dx + c * dy) * inv + 0.5,
        (-c * dx - s * dy) * inv + 0.5,
    ]
}

/// xorshift32: the reference's RNG differs, so this matches its distribution, not its stream.
fn next_u32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

pub(super) fn rand01(state: &mut u32) -> f32 {
    (next_u32(state) >> 8) as f32 / (1u32 << 24) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The signs of the affine texgen fitted to the reference's frames: `u` across the track, `v`
    /// against the heading, and the box `[c−s, c+s]` one UV unit.
    #[test]
    fn texgen_matches_the_reference_fit() {
        let c = [-9016.0_f32, -226.0];
        let h = 0.4637_f32; // about 26.6°, a heading from the reference capture
        let s = 1.25_f32;
        let (sh, ch) = h.sin_cos();
        let uv_a = foam_uv(c, h, s, [c[0] + ch, c[1] + sh]); // 1 yd ahead
        let uv_b = foam_uv(c, h, s, [c[0] - ch, c[1] - sh]); // 1 yd behind
        let uv_l = foam_uv(c, h, s, [c[0] - sh, c[1] + ch]); // 1 yd across
        assert!((uv_a[0] - 0.5).abs() < 1e-4 && uv_a[1] < 0.5, "{uv_a:?}");
        assert!(uv_b[1] > 0.5, "{uv_b:?}");
        assert!(
            (uv_l[1] - 0.5).abs() < 1e-4 && (uv_l[0] - 0.5).abs() > 0.1,
            "{uv_l:?}"
        );
        let edge = foam_uv(c, h, s, [c[0] + ch * s, c[1] + sh * s]);
        assert!(edge[1].abs() < 1e-4, "box edge → v = 0: {edge:?}");
    }

    #[test]
    fn params_match_the_verified_formulas() {
        let mut rng = 1u32;
        for _ in 0..64 {
            let translating = WadeState::Translating {
                speed: 7.0,
                heading: 0.0,
            };
            let p = foam_params(
                translating,
                false,
                1.0,
                1.0,
                0.3,
                FoamLook::CLASSIC,
                &mut rng,
            )
            .unwrap();
            assert!(!p.ring);
            assert!((0.3..=0.37).contains(&p.size0), "wake size0 {}", p.size0);
            assert!((2.8..=4.2).contains(&p.growth), "wake growth {}", p.growth);
            assert!((0.6..0.7).contains(&p.lifetime));
            assert!((p.peak - 1.0).abs() < 1e-6);

            let p = foam_params(
                WadeState::Standing,
                false,
                1.0,
                1.0,
                0.3,
                FoamLook::CLASSIC,
                &mut rng,
            )
            .unwrap();
            assert!(p.ring);
            assert!((0.16..=0.23).contains(&p.size0), "ring size0 {}", p.size0);
            assert!(
                (0.25..=0.375).contains(&p.growth),
                "ring growth {}",
                p.growth
            );
            assert!((0.6..0.7).contains(&p.lifetime), "ring life {}", p.lifetime);
            assert!((p.peak - 0.8).abs() < 1e-6);

            let p = foam_params(
                translating,
                true,
                1.0,
                1.0,
                0.3,
                FoamLook::CLASSIC,
                &mut rng,
            )
            .unwrap();
            assert!(p.ring, "a one-shot is ring-category");
            assert!(
                (0.3..=0.377).contains(&p.size0),
                "one-shot size0 {}",
                p.size0
            );
            assert!(
                (1.0..1.5).contains(&p.growth),
                "one-shot growth unscaled {}",
                p.growth
            );
            assert!((p.peak - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn cadence_laws() {
        let mut rng = 3u32;
        for _ in 0..32 {
            let w = wake_cooldown(7.0, &mut rng);
            assert!((0.080..=0.103).contains(&w), "wake@7 {w}");
            let w = wake_cooldown(50.0, &mut rng);
            assert!((0.028..=0.036).contains(&w), "wake@cap {w}");
        }
        assert!(RING_INTERVAL.0 >= 0.4 && RING_INTERVAL.1 <= 0.45);
    }

    #[test]
    fn depth_gate_and_attenuation() {
        let mut rng = 7u32;
        assert!(foam_params(
            WadeState::Standing,
            false,
            1.0,
            1.0,
            1.05,
            FoamLook::CLASSIC,
            &mut rng
        )
        .is_none());
        assert!(foam_params(
            WadeState::Standing,
            false,
            1.0,
            1.0,
            -0.1,
            FoamLook::CLASSIC,
            &mut rng
        )
        .is_none());
        // Near the gate depth, k → 0.5: a standing ring's peak → 6 × (0.8/6 × ~0.5) ≈ 0.4.
        let deep = foam_params(
            WadeState::Standing,
            false,
            1.0,
            1.0,
            0.99,
            FoamLook::CLASSIC,
            &mut rng,
        )
        .unwrap();
        assert!((deep.peak - 0.8 * 0.505).abs() < 0.02, "peak {}", deep.peak);
    }

    /// MONKEY (swim waves) — **the Classic path is untouched.** Every Enhanced knob, evaluated at
    /// [`FoamLook::CLASSIC`], reproduces the reference's answer bit for bit: same seed, same
    /// stream, same numbers. This is the gate the whole readability pass hangs on, so it is
    /// asserted against the reference's own function rather than against remembered constants.
    #[test]
    fn classic_is_byte_identical_and_enhanced_only_scales() {
        let call = |look, oneshot, state, seed| {
            let mut rng = seed;
            foam_params(state, oneshot, 1.0, 4.06, 1.52, look, &mut rng).expect("in the gate")
        };
        let swim = WadeState::Translating {
            speed: 4.72,
            heading: 0.0,
        };
        for seed in [1u32, 7, 99, 12345] {
            // The Classic look draws the same RNG in the same order as the pre-`look` function.
            let base = call(FoamLook::CLASSIC, false, swim, seed);
            let also_classic = call(
                FoamLook {
                    enhanced: false,
                    swimming: true, // swimming, but NOT Enhanced ⇒ still the reference
                },
                false,
                swim,
                seed,
            );
            assert_eq!(base.size0, also_classic.size0);
            assert_eq!(base.lifetime, also_classic.lifetime);
            assert_eq!(base.growth, also_classic.growth);
            assert_eq!(base.peak, also_classic.peak);

            let enhanced = call(
                FoamLook {
                    enhanced: true,
                    swimming: true,
                },
                false,
                swim,
                seed,
            );
            assert!((enhanced.size0 / base.size0 - 1.3).abs() < 1e-5);
            assert!((enhanced.lifetime / base.lifetime - 1.4).abs() < 1e-5);
            assert_eq!(enhanced.growth, base.growth, "growth is never attenuated");
            assert_eq!(enhanced.peak, base.peak, "the gain is not in the alpha");

            // The step-in splash: bigger ring, faster growth, Enhanced only.
            let splash_c = call(FoamLook::CLASSIC, true, swim, seed);
            let splash_e = call(
                FoamLook {
                    enhanced: true,
                    swimming: false,
                },
                true,
                swim,
                seed,
            );
            assert!((splash_e.size0 / splash_c.size0 - 1.8).abs() < 1e-5);
            assert!((splash_e.growth / splash_c.growth - 1.3).abs() < 1e-5);
        }
        assert_eq!(
            foam_gain(false),
            1.0,
            "Classic draws at the authored colour"
        );
        assert!((foam_gain(true) - 3.5).abs() < 1e-6);
    }

    /// The treading ring exists only in Enhanced, only for a swimmer, and stays soft — a wide slow
    /// circle whose peak is well under a wake's.
    #[test]
    fn the_treading_ring_is_enhanced_and_swimming_only() {
        let mut rng = 11u32;
        assert!(tread_params(FoamLook::CLASSIC, 1.0, &mut rng).is_none());
        assert!(tread_params(
            FoamLook {
                enhanced: true,
                swimming: false
            },
            1.0,
            &mut rng
        )
        .is_none());
        assert!(tread_params(
            FoamLook {
                enhanced: false,
                swimming: true
            },
            1.0,
            &mut rng
        )
        .is_none());
        let p = tread_params(
            FoamLook {
                enhanced: true,
                swimming: true,
            },
            1.0,
            &mut rng,
        )
        .expect("an Enhanced swimmer treads");
        assert!(p.ring);
        assert!((0.6..=0.9).contains(&p.size0), "wide: {}", p.size0);
        assert!(p.growth < 0.7, "slow: {}", p.growth);
        assert!(p.lifetime > 1.3, "long-lived: {}", p.lifetime);
        assert!(p.peak <= 0.5, "soft: {}", p.peak);
    }

    /// The swim ramp: dry and wading are zero, a body at the verified `0.75·h` rest line is most
    /// of the way up, and deeper saturates. Monotone throughout — a ramp that dipped would make a
    /// body bob down as it sank.
    #[test]
    fn swim_ramp_spans_the_verified_latch() {
        let h = 2.031_f32;
        assert_eq!(swim_ramp(-1.0, h), 0.0, "dry");
        assert_eq!(swim_ramp(0.0, h), 0.0);
        assert_eq!(swim_ramp(0.5 * h, h), 0.0, "knee-deep is not swimming");
        assert!(
            swim_ramp(0.75 * h, h) > 0.7,
            "the verified latch is mostly in"
        );
        assert_eq!(swim_ramp(0.9 * h, h), 1.0, "past the band, saturated");
        assert_eq!(swim_ramp(10.0, h), 1.0);
        let mut prev = 0.0;
        for i in 0..=200 {
            let r = swim_ramp(i as f32 * 0.02 * h, h);
            assert!(r >= prev - 1e-6, "monotone at {i}: {r} < {prev}");
            prev = r;
        }
    }

    /// The lifecycle: 0.4/0.6 rise/decay to the peak, zero at the lifetime; linear size growth.
    #[test]
    fn alpha_ramp_and_size_growth() {
        assert!((record_alpha(0.8, 1.0, 0.0, 0.2) - 0.4).abs() < 1e-6);
        assert!((record_alpha(0.8, 1.0, 0.0, 0.4) - 0.8).abs() < 1e-6);
        assert!((record_alpha(0.8, 1.0, 0.0, 0.7) - 0.4).abs() < 1e-6);
        assert!(record_alpha(0.8, 1.0, 0.0, 1.0) < 1e-6);
        assert!((record_size(0.5, 1.0, 0.0, 0.5) - 1.0).abs() < 1e-6);
    }
}
