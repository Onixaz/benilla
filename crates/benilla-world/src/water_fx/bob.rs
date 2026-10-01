//! MONKEY (swim waves) — **bodies ride the swell**: the visual-only vertical bob (and the small
//! lean into the wave slope) every swimming or floating unit takes on Enhanced water.
//!
//! Ported from WarcraftXL (https://github.com/WarcraftXL) by iThorgrim — module wxl-experimental-water, world/Ride.cpp.
//! (The principle — the authoritative position never moves — implemented independently; listed in
//! `THIRD-PARTY.md`.)
//!
//! The defect this closes is a mismatch, not a missing effect. Enhanced water heaves the ocean
//! MESH in the vertex stage ([`crate::liquid::waves`] mirrors the exact band), while every body in
//! it is placed at the height the CPU queries — the flat MCLQ heightfield, which knows nothing
//! about the swell. So the sea visibly rises and falls and the swimmer in it holds still, cutting
//! through crests. Here the two are put back on one surface.
//!
//! **Visual only, and structurally so.** The bob is written onto the unit's `Transform` in
//! `PostUpdate` *immediately before* [`TransformSystems::Propagate`] and taken off again
//! *immediately after* it — [`apply_swim_bob`] / [`clear_swim_bob`], with the pre-bob values
//! stored verbatim so the restore is exact rather than an inverse-multiply that would drift a
//! quaternion over a long float. The offset therefore exists only inside that one `PostUpdate`
//! window — nothing in `Update`, and nothing from the clear onward, can see it.
//!
//! Within the window it is visible to any *unordered* `PostUpdate` system that reads a unit's
//! `Transform` (the shadow, nameplate and billboard placements are the population), and that is
//! deliberate rather than merely tolerated: those consumers are the body's own presentation, so a
//! nameplate that rides the bob is the right answer and one that does not is the wrong one. What
//! matters is the set that must NEVER see it, and none of it is here:
//!
//! * movement, collision and the capsule run in `Update` off `player::Player::pos` and the wire —
//!   none of them read a `Transform` inside this window, and all of them overwrite it next frame;
//! * the network position is `bevy_to_wow(player.pos)`, never the body transform;
//! * the camera seat rides the pivot channel off `Player`, so the eye is NOT dragged along — the
//!   body heaves under a steady camera, which is the readable way round. (The brief allows the
//!   camera to follow at 50 %; that lives in `player/camera.rs` and is deliberately left out of
//!   this diff — nothing here can seasick-lag a view it never touches.)
//!
//! **Gates**, all four of which must hold or the offset is exactly zero:
//! 1. [`benilla_assets::WaterQuality`] ≥ 1 — Classic renders a flat sea, so a bobbing body there
//!    would be floating over nothing. The Classic path is byte-identical with this module armed.
//! 2. The surface under the body actually carries the vertex swell
//!    ([`WaterChunkInfo::has_vertex_swell`]: ADT + ocean). Rivers, WMO pools and lava are flat.
//! 3. The body is deep enough to be off the floor — the swim ramp
//!    ([`super::params::swim_ramp`], the byte-verified `0.75 · collisionHeight` line, softened
//!    into a band so a wader standing up does not pop).
//! 4. It is a body that displaces water at all ([`WorldUnit::wades`]) — which is precisely the
//!    predicate that already excludes corpses, loot and ground anchors, so "corpses and floating
//!    loot: skip" needs no second rule.

use bevy::ecs::entity::EntityHashMap;
use bevy::math::Vec2;
use bevy::prelude::*;
use bevy::transform::TransformSystems;

use benilla_assets::coords::bevy_to_wow;

use crate::liquid::{waves, WaterChunkInfo, WaterIndex};
use crate::world_unit::WorldUnit;

use super::params::swim_ramp;

/// How much of the swell a body takes — the brief's 0.85. Not 1.0 on purpose: a body sitting at
/// exactly the surface height reads as *welded* to the mesh, and the slight under-follow is what
/// makes it read as floating IN water rather than as part of it. (The low-pass below shaves
/// another ~10 % off the ~3 s swell period on top, and lags it ~0.23 s — that phase offset is a
/// feature at these amplitudes: the body answers the wave, it does not anticipate it.)
const BOB_GAIN: f32 = 0.85;

/// The low-pass time constant (s) — the brief's ~0.25 s. Its job is not the swell (which is
/// already smooth, 2.7–3.3 s per component) but everything that can step the INPUT: a unit
/// teleporting, a streamed body's extrapolation snapping back, the swim ramp latching, a surface
/// streaming in under a swimmer. Any of those is a one-frame jump of up to the full 0.34 yd band,
/// and unfiltered that is a visible twitch on every other body in the bay.
const BOB_TAU: f32 = 0.25;

/// The slope lean's ceiling — the brief's ≤ 4°, held as the tangent of the angle so the clamp is
/// one `min` on the gradient's length instead of a trigonometric round trip per body per frame.
/// `tan(4°)` = 0.06993.
const MAX_TILT_TAN: f32 = 0.069_927;

/// The lean is for a body **floating at rest**: above this planar speed (yd/s) it fades out. A
/// stroking swimmer is driven by its own swim pitch (`creature_anim::swim_body_rotation`) and a
/// second attitude term fighting it reads as a wobble, not as buoyancy.
const REST_SPEED: (f32, f32) = (0.8, 2.5);

/// One body's filtered ride. Kept per unit rather than recomputed, because the whole point of the
/// filter is that this frame's answer depends on the last one's.
#[derive(Default)]
struct BobState {
    /// The low-passed vertical offset actually applied (yd).
    lift: f32,
    /// The low-passed surface gradient the lean is built from, already weighted by the rest term.
    grad: Vec2,
    /// Last frame's position, for the planar-speed rest test (the same velocity proxy the foam
    /// emitter uses for streamed units — a body's own velocity is not published to this crate).
    last_pos: Option<Vec3>,
    /// Fed this frame; unfed entries are dropped (the unit despawned).
    active: bool,
}

/// Every body's filtered ride, plus the exact pre-bob transforms this frame's apply pass must put
/// back after propagation.
#[derive(Resource, Default)]
struct SwimBob {
    units: EntityHashMap<BobState>,
    /// `(entity, translation.y, rotation)` as they were before the offset. A stored ORIGINAL, not
    /// a delta to subtract: restoring by arithmetic would leave a float residue on a body that is
    /// never rewritten (an idle streamed unit at rest), and that residue accumulates.
    applied: Vec<(Entity, f32, Quat)>,
}

/// The visual lift + lean for one body, given its filtered state. Split out so the filter and the
/// clamp are testable without an ECS world (see [`tests`]).
fn ride(state: &BobState) -> (f32, Quat) {
    let g = state.grad;
    // Clamp the SLOPE, not the resulting quaternion: the wave normal is `(-∂x, 1, -∂z)`
    // (`enhanced_water.wgsl`), so a gradient of length `tan(θ)` is a lean of exactly θ.
    let len = g.length();
    let g = if len > MAX_TILT_TAN {
        g * (MAX_TILT_TAN / len)
    } else {
        g
    };
    let normal = Vec3::new(-g.x, 1.0, -g.y).normalize_or(Vec3::Y);
    (state.lift, Quat::from_rotation_arc(Vec3::Y, normal))
}

/// Exponential low-pass toward `target` over [`BOB_TAU`], frame-rate independent.
fn low_pass(current: f32, target: f32, dt: f32) -> f32 {
    let a = 1.0 - (-dt / BOB_TAU).exp();
    current + (target - current) * a
}

/// Write this frame's bob onto every eligible body, remembering what it overwrote.
///
/// Ordered before [`TransformSystems::Propagate`] so the rendered pose carries it, and paired with
/// [`clear_swim_bob`] after it so nothing else ever does. The water lookup is the foam emitter's
/// own shape — one [`WaterIndex`] hash per body, a dry body costing one miss — not a walk of the
/// ~2 k loaded surfaces per unit.
fn apply_swim_bob(
    time: Res<Time>,
    quality: Res<benilla_assets::WaterQuality>,
    mut bob: ResMut<SwimBob>,
    index: Res<WaterIndex>,
    water: Query<&WaterChunkInfo>,
    mut units: Query<(Entity, &mut Transform, &WorldUnit)>,
) {
    bob.applied.clear();
    for s in bob.units.values_mut() {
        s.active = false;
    }
    // Gate 1: Classic water renders a flat sea. Everything below is skipped wholesale, so the
    // Classic frame is the frame it was before this module existed.
    let enhanced = quality.0 >= 1;
    let dt = time.delta_secs().max(1.0e-4);
    let t = waves::water_anim_time(time.elapsed_secs_wrapped());

    for (entity, mut transform, unit) in &mut units {
        if !unit.wades {
            continue; // gate 4 — and the whole of "corpses and floating loot: skip"
        }
        let pos = transform.translation;
        let state = bob.units.entry(entity).or_default();
        state.active = true;
        let prev = state.last_pos.replace(pos);
        // The planar speed proxy, in Bevy's XZ (the horizontal plane) — only the rest test uses it.
        let speed = prev.map_or(0.0, |p| (pos.xz() - p.xz()).length() / dt);

        // Gates 2 + 3: a swelling surface over this body, and deep enough to be off the floor.
        let wow = bevy_to_wow(pos);
        let ramp = enhanced
            .then(|| {
                // First SWELLING surface over this XY — `has_vertex_swell` before `surface_z_at`,
                // so a flat river lying over an ocean chunk cannot claim the body and then report
                // a height nothing is heaving at.
                let surface = index.over(wow[0], wow[1]).iter().find_map(|&e| {
                    let info = water.get(e).ok()?;
                    info.has_vertex_swell()
                        .then(|| info.surface_z_at(wow[0], wow[1]))
                        .flatten()
                })?;
                let ramp = swim_ramp(surface - wow[2], unit.height);
                (ramp > 0.0).then_some(ramp)
            })
            .flatten();

        let (lift_target, grad_target) = match ramp {
            Some(ramp) => {
                // The shore term is the swim ramp itself, not `waves::swell_shore_fade` — the
                // authored depth `V` the shader fades by never reaches the CPU (see that
                // function's doc). It is the better proxy anyway for this consumer: the band
                // where the sea stops heaving is the band where a body can stand up, and the
                // ramp is already zero there, so a swimmer never fades out mid-stroke.
                let s = waves::swell(pos.xz(), t, waves::OCEAN_WAVE_ENERGY, ramp);
                // The lean is for a body at rest; a stroking swimmer already has a swim pitch.
                let rest =
                    1.0 - ((speed - REST_SPEED.0) / (REST_SPEED.1 - REST_SPEED.0)).clamp(0.0, 1.0);
                (s.height * BOB_GAIN * ramp, s.grad * rest * ramp)
            }
            // Out of the water (or Classic, or over a flat liquid): the target is FLAT, and the
            // filter walks down to it over ~τ rather than snapping — a swimmer wading out or
            // jumping clear settles instead of dropping a quarter yard in one frame.
            None => (0.0, Vec2::ZERO),
        };
        state.lift = low_pass(state.lift, lift_target, dt);
        state.grad = Vec2::new(
            low_pass(state.grad.x, grad_target.x, dt),
            low_pass(state.grad.y, grad_target.y, dt),
        );

        let (lift, tilt) = ride(state);
        // Below a tenth of a millimetre there is nothing to see and nothing to restore; skipping
        // keeps every dry body's `Transform` genuinely untouched (and unchanged, so bevy's
        // propagation has no reason to revisit it).
        if lift.abs() < 1.0e-4 && state.grad.length_squared() < 1.0e-8 {
            continue;
        }
        bob.applied
            .push((entity, transform.translation.y, transform.rotation));
        transform.translation.y += lift;
        // Pre-multiply: the lean is a WORLD-space tilt toward the wave normal, so it must not be
        // re-framed by the body's own yaw (post-multiplying would roll a north-facing swimmer
        // where it should pitch one facing east).
        transform.rotation = tilt * transform.rotation;
    }
    bob.units.retain(|_, s| s.active);
}

/// Put every bobbed `Transform` back exactly as it was, after propagation has read it.
///
/// This is what makes the offset visual: from here to the end of the frame, and through all of the
/// next one's `Update`, the ECS holds the unbobbed pose that movement, collision and the wire were
/// always written against.
fn clear_swim_bob(mut bob: ResMut<SwimBob>, mut units: Query<&mut Transform>) {
    // `take` so a frame in which `apply` did not run (a run condition, a disabled schedule) cannot
    // replay last frame's restore onto this frame's pose.
    for (entity, y, rot) in std::mem::take(&mut bob.applied) {
        if let Ok(mut t) = units.get_mut(entity) {
            t.translation.y = y;
            t.rotation = rot;
        }
    }
}

/// Register the bob pass — called from [`super::WaterFxPlugin`].
pub(super) fn register(app: &mut App) {
    app.init_resource::<SwimBob>().add_systems(
        PostUpdate,
        (
            apply_swim_bob.before(TransformSystems::Propagate),
            clear_swim_bob.after(TransformSystems::Propagate),
        ),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The filter: a STEP input never appears as a step on the output, it approaches smoothly and
    /// stays bounded by the step. This is the property the whole `BOB_TAU` term exists for — the
    /// swell itself is smooth; what is not is a teleport, a snap-back or a ramp latch.
    #[test]
    fn the_filter_smooths_a_step_and_stays_bounded() {
        let (dt, step) = (1.0 / 60.0, 0.34_f32);
        let mut x = 0.0_f32;
        let mut prev = 0.0_f32;
        for frame in 0..240 {
            x = low_pass(x, step, dt);
            assert!((0.0..=step).contains(&x), "bounded at frame {frame}: {x}");
            assert!(
                x - prev < step * 0.1,
                "no step at frame {frame}: {} yd in one frame",
                x - prev
            );
            prev = x;
        }
        assert!((x - step).abs() < 1e-3, "converged: {x}");
        // One τ reaches ~63 % of the step — the pinned time constant, not merely "some smoothing".
        // (`round`, not `as i32`: 0.25/(1/60) is 14.999… in f32 and truncates to 14 frames.)
        let mut y = 0.0_f32;
        for _ in 0..(BOB_TAU / dt).round() as i32 {
            y = low_pass(y, step, dt);
        }
        assert!((y / step - 0.632).abs() < 0.02, "one τ: {}", y / step);
    }

    /// The lean's ceiling holds however steep the sampled slope gets, and a flat sea leans not at
    /// all. 4° is small enough to read as buoyancy; the clamp is what keeps a degenerate gradient
    /// (a snapped surface, a filter transient) from cartwheeling a body.
    #[test]
    fn the_lean_is_capped_at_four_degrees() {
        let flat = ride(&BobState::default());
        assert_eq!(flat.0, 0.0);
        assert!(flat.1.angle_between(Quat::IDENTITY) < 1e-6, "flat ⇒ level");
        for g in [0.01_f32, 0.07, 0.5, 5.0, 100.0] {
            let state = BobState {
                grad: Vec2::new(g, -g),
                ..default()
            };
            let angle = ride(&state).1.angle_between(Quat::IDENTITY);
            assert!(
                angle <= 4.0_f32.to_radians() + 1e-4,
                "grad {g} leaned {}°",
                angle.to_degrees()
            );
        }
    }

    /// **The Classic gate, end to end.** With [`benilla_assets::WaterQuality`] at 0 a body sitting
    /// well inside an ocean grid is never touched at all: the pose the propagation window sees is
    /// the pose it was given, bit for bit, and no filter state accumulates behind it.
    #[test]
    fn classic_water_never_moves_a_body() {
        let (mut app, body, placed) = swimming_world(0);
        for _ in 0..90 {
            app.update();
            assert_eq!(
                seen(&app),
                placed.translation.y,
                "Classic ⇒ the rendered pose is the placed pose"
            );
            assert_eq!(*app.world().get::<Transform>(body).expect("body"), placed);
        }
        assert_eq!(app.world().resource::<SwimBob>().units[&body].lift, 0.0);
    }

    /// Enhanced, the same world: the body is lifted inside the propagation window and handed back
    /// its original pose afterwards. Both halves matter — the first is the feature, the second is
    /// the promise that movement, collision and the wire never see it.
    #[test]
    fn enhanced_water_bobs_the_body_and_hands_the_pose_back() {
        let (mut app, body, placed) = swimming_world(1);
        // Several frames: the low-pass starts at zero, so one frame proves nothing about the ride.
        let mut lifted = false;
        for _ in 0..200 {
            app.update();
            lifted |= (seen(&app) - placed.translation.y).abs() > 1.0e-4;
            assert_eq!(
                *app.world().get::<Transform>(body).expect("body"),
                placed,
                "the ECS pose is restored exactly, every frame"
            );
        }
        assert!(lifted, "an Enhanced swimmer takes a visual offset");
    }

    /// A body on DRY land inside the same world takes nothing — the depth gate, not merely the
    /// quality one, and the reason a bob can never lift someone walking along a beach.
    #[test]
    fn a_dry_body_is_never_bobbed() {
        let (mut app, swimmer, _) = swimming_world(1);
        let dry = app
            .world_mut()
            .spawn((
                Transform::from_translation(benilla_assets::coords::wow_to_bevy([
                    500.0, 500.0, 40.0,
                ])),
                WorldUnit {
                    wades: true,
                    scale: 1.0,
                    height: 2.0,
                    bound: None,
                },
            ))
            .id();
        for _ in 0..60 {
            app.update();
        }
        let bob = app.world().resource::<SwimBob>();
        assert_eq!(bob.units[&dry].lift, 0.0, "dry land ⇒ no lift");
        assert_eq!(bob.units[&dry].grad, Vec2::ZERO, "dry land ⇒ no lean");
        assert!(bob.units[&swimmer].lift != 0.0, "the control IS riding");
    }

    /// What the propagation window saw this frame — the probe system's capture (see
    /// [`swimming_world`]), i.e. the body's rendered height.
    fn seen(app: &App) -> f32 {
        app.world().resource::<SeenY>().0
    }

    /// The probe's capture slot: the bobbed `translation.y`, read between apply and clear.
    #[derive(Resource, Default)]
    struct SeenY(f32);

    /// An app with one all-wet ADT OCEAN grid and one body swimming in it, at the given water
    /// quality.
    ///
    /// The three schedule slots stand in for the real bracket: `Update` places and applies,
    /// `PostUpdate` is where `TransformSystems::Propagate` would read the pose (the probe reads it
    /// instead), `Last` clears. Placing the body every frame is not test scaffolding — it is what
    /// a real frame does, and it is what proves the clear is not merely hiding behind a body that
    /// nobody rewrites.
    fn swimming_world(quality: u8) -> (App, Entity, Transform) {
        // Feet at the verified swim rest line (`surface − 0.75·h`) — fully swimming.
        let placed = Transform::from_translation(benilla_assets::coords::wow_to_bevy([
            50.0,
            50.0,
            50.0 - 1.52,
        ]));
        let mut app = App::new();
        app.init_resource::<Time>()
            .init_resource::<WaterIndex>()
            .init_resource::<SwimBob>()
            .init_resource::<SeenY>()
            .insert_resource(benilla_assets::WaterQuality(quality))
            .add_systems(
                Update,
                (
                    crate::liquid::maintain_water_index,
                    move |mut q: Query<&mut Transform, With<WorldUnit>>| {
                        for mut t in &mut q {
                            if t.translation.x == placed.translation.x {
                                *t = placed;
                            }
                        }
                    },
                    apply_swim_bob,
                )
                    .chain(),
            )
            .add_systems(
                PostUpdate,
                |mut seen: ResMut<SeenY>, q: Query<(&Transform, &WorldUnit)>| {
                    for (t, _) in &q {
                        if (t.translation.x - -50.0).abs() < 1.0 {
                            seen.0 = t.translation.y;
                        }
                    }
                },
            )
            .add_systems(Last, clear_swim_bob);
        // A 3×3-vertex all-wet ocean grid over [0,100]², surface z = 50.
        let mut positions = Vec::new();
        for j in 0..3 {
            for i in 0..3 {
                positions.push([i as f32 * 50.0, j as f32 * 50.0, 50.0]);
            }
        }
        app.world_mut().spawn(WaterChunkInfo::new(
            crate::liquid::LiquidSource::AdtChunk,
            benilla_formats::LiquidKind::Ocean,
            [3, 3],
            positions,
            vec![true; 4],
        ));
        let body = app
            .world_mut()
            .spawn((
                placed,
                WorldUnit {
                    wades: true,
                    scale: 1.0,
                    height: 2.031,
                    bound: None,
                },
            ))
            .id();
        (app, body, placed)
    }
}
