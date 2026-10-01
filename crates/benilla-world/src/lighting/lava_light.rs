//! MONKEY (lava light): warm, slowly breathing fixtures over streamed magma surfaces.
//!
//! Read the same world-space wet grid used by swimming: MCLQ and placed/rotated MLIQ already
//! publish it on their rendered entity. No bounds-only lights over a shore or a masked hole.
//! Sampling and nearest-first budgeting run on surface changes (or the gain's on/off edge),
//! not camera motion. Only the <=[`LAVA_LIGHTS_MAX`] slow intensity writes run each frame. Slime
//! stays dark; whether Felwood's green pools count as fel fire is deliberately left to the owner.
//!
//! **Two things in here are MEASURED against the "lava not really glowing" report** (Searing Gorge
//! at night, `_fx/lava_findings.md`) rather than chosen: the lane stamp at the spawn (an
//! open-world fixture the shared down-ray classifier called INTERIOR lights no terrain at all),
//! and the geometry constants below. See each for the numbers.

use benilla_assets::coords::wow_to_bevy;
use bevy::prelude::*;
use std::collections::HashMap;

use super::{DaylightFixture, DaylightHow, LightReach, WorldPointLight};
use crate::liquid::{LiquidSoundSource, LiquidSurface, WaterChunkInfo};

/// MONKEY (lava light: read): 6 was a GORGE-KILLER. A magma channel's footprint is one MCNK liquid
/// layer per 33.3 yd chunk, so a pool that fits in one chunk got six fixtures TOTAL — measured on
/// the Searing Gorge channel at `(-7200, -930, 139)`: six local fixtures and then a 289 yd gap to
/// the next one, i.e. 42 of the 48-fixture budget spent a quarter of a mile away where they lit
/// nothing. Twelve per surface lets a single-chunk pool actually use the budget it is next to.
pub const LAVA_LIGHTS_PER_SURFACE: usize = 12;
pub const LAVA_LIGHTS_MAX: usize = 64;
/// The sample lattice over a magma footprint, and the minimum spacing the budgeter keeps.
///
/// MEASURED against the terrain shader's own falloff, which is the thing that has to be satisfied:
/// `terrain.wgsl::point_light_eval` commits `rgb / (0.7 d + 0.03 d²) × max(N·L, 0)`. Ground beside
/// a river reads the SUM of the fixtures in its 12-deep per-chunk selection, so spacing is not
/// cosmetic — it is most of the amplitude. At the old 12 yd / +1 yd / 2.0, a ground patch 6 yd off
/// the nearest fixture collects `0.186 × 0.164 × 2.0 ≈ 0.06` red and its neighbours add ~0.013
/// each: **0.08**, which is exactly the "faint orange tint on some ground" the report describes.
/// At 8 yd / +2.5 yd / 3.0 the same patch collects ~0.4 from the nearest and ~0.2 from the ring
/// behind it. Cheap, too: the cost of a fixture is one packed row, and the per-chunk selection
/// keeps only the nearest twelve whatever we spawn.
const GRID_YD: f32 = 8.0;
const MERGE_YD: f32 = 7.0;
/// How far ABOVE the magma surface a fixture floats. The old +1 yd was the geometric reason the
/// pool could not light its own banks: for flat ground at the liquid's own level `N·L` degenerates
/// to `lift / d`, so a 1 yd lift throws away ~85 % of the term at 6 yd. +2.5 yd is still inside the
/// heat haze over the surface and nearly triples it.
const LIFT_YD: f32 = 2.5;
const COLOR: [f32; 3] = [1.0, 0.42, 0.10];
/// MONKEY (lava light): the authored rung on `fire_light.rs::fire_intensity`'s ladder. A river of
/// molten rock is a BONFIRE (3.0), not the brazier (2.0) it shipped as — the brazier rung is a
/// standing fire bowl, and a 30 yd channel of it does not read as one. `m2_light_reach(3.0)` is
/// 24 yd, so [`REACH_YD`] is that rung's own reach and not a second free parameter.
const AUTHORED: f32 = 3.0;
const REACH_YD: f32 = 24.0;
// `WorldPointLight` stores 4*pi times authored units (`spawn_point_light`'s premultiply).
const INTENSITY: f32 = AUTHORED * 4.0 * std::f32::consts::PI;

/// MONKEY (lava light): ownership is explicit rather than ChildOf: the shared down-ray lane
/// classifier deliberately skips children. Reconciliation removes fixtures with their surface
/// before that frame's light packing, and leaves retained entities (and their lane cache) alone.
#[derive(Component)]
pub struct LavaLight {
    surface: Entity,
    position: Vec3,
    phase: f32,
}

#[derive(Resource, Default)]
struct LavaSurfaces {
    candidates: HashMap<Entity, Vec<Vec3>>,
    dirty: bool,
    enabled: bool,
}

/// MONKEY (lava light): live settings dial; zero removes all magma fixtures.
#[derive(Resource, Clone, Copy)]
pub struct LavaLightGain(pub f32);

impl Default for LavaLightGain {
    fn default() -> Self {
        Self(1.0)
    }
}

pub(super) fn register(app: &mut App) {
    app.init_resource::<LavaLightGain>()
        .init_resource::<LavaSurfaces>()
        .add_systems(
            PostUpdate,
            (reconcile, breathe)
                .chain()
                .after(bevy::transform::TransformSystems::Propagate)
                // Doorway bleed must read this frame's magma gain/breath, including removal.
                .before(super::daylight::update_bleed_fixtures)
                .before(super::global_light::classify_light_lanes),
        )
        // MONKEY (lava light): the LANE census, AFTER the classifier has answered this frame —
        // "the module made 48 fixtures" and "48 fixtures light the rock" are different claims, and
        // the interior lane is where the second one is lost. Rides `WOW_POINTS_DUMP` (the same
        // knob the packer's table prints under) so a run log carries both halves.
        .add_systems(
            PostUpdate,
            lane_census.after(super::global_light::classify_light_lanes),
        );
}

// MONKEY (lava light): use the wet vertices for bounds, then the grid's own inverse/bilinear
// sampler for coverage and height. In particular, a rotated MLIQ's AABB is NOT its footprint.
// Partial edge cells use their own centre, allowing a pool smaller than 12 yd to contribute.
fn sample_grid(info: &WaterChunkInfo) -> Vec<Vec3> {
    let mut lo = Vec2::splat(f32::INFINITY);
    let mut hi = Vec2::splat(f32::NEG_INFINITY);
    info.for_each_wet_cell(|corners| {
        for [x, y, _] in corners {
            lo = lo.min(Vec2::new(x, y));
            hi = hi.max(Vec2::new(x, y));
        }
    });
    if !lo.is_finite() || !hi.is_finite() {
        return Vec::new();
    }
    let mut points = Vec::new();
    for j in 0..((hi.y - lo.y) / GRID_YD).ceil() as usize {
        for i in 0..((hi.x - lo.x) / GRID_YD).ceil() as usize {
            let cell_lo = lo + Vec2::new(i as f32, j as f32) * GRID_YD;
            let centre = (cell_lo + (cell_lo + Vec2::splat(GRID_YD)).min(hi)) * 0.5;
            if let Some(z) = info.surface_z_at(centre.x, centre.y) {
                points.push(wow_to_bevy([centre.x, centre.y, z + LIFT_YD]));
            }
        }
    }
    points
}

fn select(candidates: &HashMap<Entity, Vec<Vec3>>, camera: Vec3) -> Vec<(Entity, Vec3)> {
    let mut ranked: Vec<_> = candidates
        .iter()
        .flat_map(|(&surface, points)| points.iter().map(move |&p| (surface, p)))
        .collect();
    ranked.sort_by(|(a, p), (b, q)| {
        p.distance_squared(camera)
            .total_cmp(&q.distance_squared(camera))
            .then_with(|| a.to_bits().cmp(&b.to_bits()))
            .then_with(|| p.x.total_cmp(&q.x))
            .then_with(|| p.y.total_cmp(&q.y))
            .then_with(|| p.z.total_cmp(&q.z))
    });
    let mut selected: Vec<(Entity, Vec3)> = Vec::new();
    let mut counts = HashMap::<Entity, usize>::new();
    for (surface, p) in ranked {
        if counts.get(&surface).copied().unwrap_or(0) >= LAVA_LIGHTS_PER_SURFACE
            || selected
                .iter()
                .any(|(_, q)| p.distance_squared(*q) < MERGE_YD * MERGE_YD)
        {
            continue;
        }
        // Keep the nearer representative in place: averaging a cluster could put it on dry land.
        selected.push((surface, p));
        *counts.entry(surface).or_default() += 1;
        if selected.len() == LAVA_LIGHTS_MAX {
            break;
        }
    }
    selected
}

fn reconcile(
    mut commands: Commands,
    gain: Res<LavaLightGain>,
    mut cache: ResMut<LavaSurfaces>,
    surfaces: Query<(Entity, Ref<WaterChunkInfo>, Ref<LiquidSoundSource>), With<LiquidSurface>>,
    camera: Query<&GlobalTransform, With<crate::view::WorldCamera>>,
    lights: Query<(Entity, &LavaLight)>,
) {
    let old_len = cache.candidates.len();
    cache.candidates.retain(|e, _| surfaces.contains(*e));
    cache.dirty |= old_len != cache.candidates.len();
    // MONKEY (lava light): the census the "lava not really glowing" report had no line for. One
    // `info!` per REBUILD (a rare event — surfaces streaming or the gain's on/off edge), never per
    // frame, so a run log answers "were there magma surfaces, how many candidates, how many kept"
    // without a debugger. `nibbles` is the raw liquid sound class of every live surface: that is
    // the number the `& 3 == 2` predicate is decided on.
    let mut nibbles: HashMap<u8, usize> = HashMap::new();
    for (_, _, sound) in &surfaces {
        *nibbles.entry(sound.nibble).or_default() += 1;
    }
    let surfaces_seen = surfaces.iter().count();
    for (surface, info, sound) in &surfaces {
        // Both spawn paths publish the same sound class: nibble & 3 == 2 is magma.
        // Water (0/1) and slime (3) NEVER enter the emitting set, including fullbright slime.
        if sound.nibble & 3 != 2 {
            cache.dirty |= cache.candidates.remove(&surface).is_some();
        } else if info.is_changed()
            || sound.is_changed()
            || !cache.candidates.contains_key(&surface)
        {
            cache.candidates.insert(surface, sample_grid(&info));
            cache.dirty = true;
        }
    }
    let enabled = gain.0.is_finite() && gain.0 > 0.0;
    cache.dirty |= enabled != cache.enabled;
    cache.enabled = enabled;
    if !cache.dirty {
        return;
    }
    let mut selected = if enabled {
        let Ok(camera) = camera.single() else {
            // Retire orphaned fixtures even while the world camera is being replaced.
            for (e, light) in &lights {
                if !cache.candidates.contains_key(&light.surface) {
                    commands.entity(e).despawn();
                }
            }
            return; // keep dirty until nearest-to-camera has a real camera
        };
        select(&cache.candidates, camera.translation())
    } else {
        Vec::new()
    };
    cache.dirty = false;
    // MONKEY (lava light): the rebuild census — see the note by `nibbles` above.
    {
        let cam = camera.single().map(|t| t.translation()).ok();
        let magma = cache.candidates.len();
        let points: usize = cache.candidates.values().map(Vec::len).sum();
        let mut census: Vec<_> = nibbles.iter().map(|(n, c)| (*n, *c)).collect();
        census.sort_unstable();
        let first: Vec<[f32; 3]> = selected
            .iter()
            .take(4)
            .map(|(_, p)| benilla_assets::coords::bevy_to_wow(*p))
            .collect();
        info!(
            "lava light: surfaces {surfaces_seen} (nibbles {census:?}) magma {magma} \
             candidates {points} kept {} gain {} cam {:?} first {first:?}",
            selected.len(),
            gain.0,
            cam.map(benilla_assets::coords::bevy_to_wow),
        );
    }
    for (e, light) in &lights {
        if let Some(i) = selected
            .iter()
            .position(|&(s, p)| s == light.surface && p == light.position)
        {
            selected.swap_remove(i);
        } else {
            commands.entity(e).despawn();
        }
    }
    for (surface, position) in selected {
        let hash = position.x.to_bits().wrapping_mul(0x9e3779b9)
            ^ position.y.to_bits().rotate_left(11)
            ^ position.z.to_bits().rotate_left(21);
        let fixture = commands.spawn((
            LavaLight {
                surface,
                position,
                phase: hash as f32 / u32::MAX as f32 * std::f32::consts::TAU,
            },
            WorldPointLight {
                color: COLOR,
                intensity: INTENSITY * gain.0,
                range: 48.0,
            },
            LightReach(REACH_YD),
            Transform::from_translation(position),
            GlobalTransform::from_translation(position),
            // MONKEY (lava light): torch_shadow already excludes DaylightFixture. Borrow that
            // marker without inventing a room claim; daylight's updater explicitly excludes us,
            // and the packer/bleed source retain the normal interiorGain law for How::Lava.
            //
            // The "lava not really glowing" round CHECKED this borrow rather than assuming it, and
            // it is sound — all four consumers carve `How::Lava` out correctly
            // (`daylight.rs:~1021` `Without<LavaLight>`, `:~1835` the bleed gain,
            // `global_light.rs:~1598` the `interiorGain` exemption, `torch_shadow.rs:~752` the
            // caster refusal). So this is NOT a dedicated-marker job and nothing outside this file
            // needed touching; the defect was the LANE, stamped below.
            DaylightFixture {
                instance: surface,
                group: 0,
                portal: None,
                how: DaylightHow::Lava,
                reach: REACH_YD,
                cal_d: 0.0,
                cal_ndl: 0.0,
            },
        ));
        // MONKEY (lava light: lane). A HYPOTHESIS THIS ROUND TESTED AND **REFUTED** — left here so
        // the next reader does not spend the session re-deriving it.
        //
        // The shared classifier ([`super::global_light::classify_light_lanes`]) files 9 of 48
        // Searing Gorge fixtures INTERIOR, and an interior row is skipped by `terrain.wgsl` before
        // its 12-deep per-chunk ranking (colour `.w > 0.5`) and dimmed by `interiorGain` on top —
        // which LOOKS exactly like the cause of "lava not really glowing". It is not. Those pools
        // are under real cave roofs, their neighbours are interior-class WMO walls, and INTERIOR is
        // the lane that lights them. Pinning ADT-sourced fixtures to the exterior lane was built,
        // captured and MEASURED: the cave walls went from `(26,4,2)` to `(17,2,1)` — the fix made
        // the rock DARKER, and the exterior tuning below moved those pixels by exactly zero
        // because nothing in that frame reads the exterior lane at all
        // (`_fx/lava-river-{probe,fixA,after}.png`). Reverted. The classifier is right; do not
        // override it from here, in either direction.
        let _ = fixture;
    }
}

/// MONKEY (lava light): `WOW_POINTS_DUMP` — the fixtures this module owns, with the LANE the
/// shared classifier put each on and its distance from the camera. The packer's own table caps its
/// printout at the nearest eight, which in a zone full of braziers prints no lava row at all; this
/// one is lava-only, so "are there fixtures" and "are they on the lane terrain reads" stay separable.
fn lane_census(
    time: Res<Time>,
    gain: Res<LavaLightGain>,
    lights: Query<(&LavaLight, &WorldPointLight, Option<&super::LightLane>)>,
    camera: Query<&GlobalTransform, With<crate::view::WorldCamera>>,
    mut last: Local<f64>,
) {
    static DUMP: std::sync::OnceLock<Option<std::ffi::OsString>> = std::sync::OnceLock::new();
    let Some(mode) = DUMP.get_or_init(|| std::env::var_os("WOW_POINTS_DUMP")) else {
        return;
    };
    let every = if mode.as_os_str() == "frame" {
        0.0
    } else {
        1.0
    };
    let now = time.elapsed_secs_f64();
    if now - *last < every {
        return;
    }
    *last = now;
    let Ok(camera) = camera.single() else { return };
    let cam = camera.translation();
    let mut rows: Vec<(f32, bool, f32, Vec3)> = lights
        .iter()
        .map(|(lava, light, lane)| {
            (
                lava.position.distance(cam),
                // No lane yet == the packer's own fallback, which for a room-less fixture is
                // EXTERIOR. Printed as the classifier's answer would be, not as "unknown".
                lane.is_some_and(|l| l.interior),
                light.intensity / (4.0 * std::f32::consts::PI),
                lava.position,
            )
        })
        .collect();
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    let interior = rows.iter().filter(|r| r.1).count();
    eprintln!(
        "[lava] {} fixture(s) ({interior} INT / {} EXT), gain {:.2}, cam {:?}",
        rows.len(),
        rows.len() - interior,
        gain.0,
        benilla_assets::coords::bevy_to_wow(cam),
    );
    for (d, interior, i, p) in rows.iter().take(8) {
        eprintln!(
            "  d {d:6.2}  at {:?}  I {i:.2}  {}",
            benilla_assets::coords::bevy_to_wow(*p),
            if *interior { "INT" } else { "EXT" },
        );
    }
}

fn breathe(
    time: Res<Time>,
    gain: Res<LavaLightGain>,
    mut lights: Query<(&LavaLight, &mut WorldPointLight)>,
) {
    for (lava, mut light) in &mut lights {
        let wave = (time.elapsed_secs() * std::f32::consts::TAU * 0.4 + lava.phase).sin();
        let intensity = INTENSITY * gain.0.max(0.0) * (1.0 + 0.1 * wave);
        // MONKEY (lava light): stationary/paused clocks and sub-epsilon breathing do not churn
        // change detection. No FlameFlicker or SyntheticFireLight: only LavaLightGain owns it.
        if (light.intensity - intensity).abs() > 1e-4 {
            light.intensity = intensity;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liquid::LiquidSource;
    use benilla_formats::LiquidKind;

    fn grid(cols: usize, rows: usize, wet: Vec<bool>, kind: LiquidKind) -> WaterChunkInfo {
        let positions = (0..rows)
            .flat_map(|j| {
                (0..cols)
                    .map(move |i| [i as f32 * 12.0, j as f32 * 12.0, i as f32 * 2.0 + j as f32])
            })
            .collect();
        WaterChunkInfo::new(LiquidSource::AdtChunk, kind, [cols, rows], positions, wet)
    }

    // Denominated in the CONSTANTS rather than in the numbers they happened to have: the tuning
    // pass that fixed the report moved `GRID_YD`/`LIFT_YD`, and a test pinned to "6.0, 6.0, 2.5"
    // fails on a value change while saying nothing about the rule (mask + sampled height) it was
    // written to protect. One exact case is still checked below, built from the constants.
    #[test]
    fn grid_sampling_respects_mask_and_surface_height() {
        let info = grid(3, 3, vec![true, false, false, true], LiquidKind::Magma);
        let points = sample_grid(&info);
        assert!(!points.is_empty());
        for point in &points {
            let p = benilla_assets::coords::bevy_to_wow(*point);
            // Every emitted point is over a WET cell, and stands `LIFT_YD` over the SAMPLED
            // surface — never the chunk maximum, and never a masked hole.
            assert_eq!(
                info.surface_z_at(p[0], p[1]).map(|z| z + LIFT_YD),
                Some(p[2]),
            );
        }
        // The lattice is the one the constant names: the first cell's centre is half a cell in.
        assert_eq!(
            points[0],
            wow_to_bevy([
                GRID_YD * 0.5,
                GRID_YD * 0.5,
                // `z = i·2 + j` over a 12 yd vertex spacing ⇒ `x/6 + y/12` at the sample.
                GRID_YD * 0.5 / 6.0 + GRID_YD * 0.5 / 12.0 + LIFT_YD,
            ]),
        );
        assert!(sample_grid(&grid(2, 2, vec![false], LiquidKind::Magma)).is_empty());
    }

    #[test]
    fn rotated_grid_does_not_emit_over_aabb_corners_or_holes() {
        let rot = Quat::from_rotation_z(0.37);
        let positions = (0..4)
            .flat_map(|j| {
                (0..4).map(move |i| {
                    (rot * Vec3::new(i as f32 * 12.0, j as f32 * 12.0, 3.0)).to_array()
                })
            })
            .collect();
        let info = WaterChunkInfo::new(
            LiquidSource::AdtChunk,
            LiquidKind::Magma,
            [4, 4],
            positions,
            vec![true, true, true, true, false, true, true, true, true],
        );
        let points = sample_grid(&info);
        assert!(!points.is_empty());
        for point in points {
            let p = benilla_assets::coords::bevy_to_wow(point);
            assert_eq!(
                info.surface_z_at(p[0], p[1]).map(|z| z + LIFT_YD),
                Some(p[2])
            );
            let local = rot.inverse() * Vec3::from_array(p);
            assert!(!(local.x > 12.0 && local.x < 24.0 && local.y > 12.0 && local.y < 24.0));
        }
    }

    #[test]
    fn merge_across_surfaces_keeps_nearest_and_accepts_the_spacing_exactly() {
        let mut world = World::new();
        let a = world.spawn_empty().id();
        let b = world.spawn_empty().id();
        // Inside the merge radius ⇒ dropped; exactly AT it ⇒ kept (the test is `< MERGE_YD²`).
        let points = HashMap::from([
            (a, vec![Vec3::ZERO]),
            (b, vec![Vec3::X * (MERGE_YD - 1.0), Vec3::X * MERGE_YD]),
        ]);
        assert_eq!(
            select(&points, Vec3::ZERO),
            vec![(a, Vec3::ZERO), (b, Vec3::X * MERGE_YD)]
        );
    }

    #[test]
    fn caps_are_global_and_per_surface_with_nearest_winning() {
        let mut world = World::new();
        // Enough per surface that the PER-SURFACE cap bites before the global one, and enough
        // surfaces that the global cap bites too — both, or the test only proves one of them.
        let per_surface = LAVA_LIGHTS_PER_SURFACE + 3;
        let points: HashMap<_, _> = (0..10)
            .map(|s| {
                let e = world.spawn_empty().id();
                (
                    e,
                    (0..per_surface)
                        .map(|i| Vec3::new(s as f32 * 200.0, 0.0, i as f32 * (MERGE_YD + 1.0)))
                        .collect(),
                )
            })
            .collect();
        let selected = select(&points, Vec3::ZERO);
        assert_eq!(selected.len(), LAVA_LIGHTS_MAX);
        let mut capped = 0;
        for &e in points.keys() {
            let n = selected.iter().filter(|(s, _)| *s == e).count();
            assert!(n <= LAVA_LIGHTS_PER_SURFACE);
            capped += usize::from(n == LAVA_LIGHTS_PER_SURFACE);
        }
        assert!(capped > 0, "the per-surface cap must actually bind here");
        // Nearest-first: the budget is spent on the near surfaces, never scattered over the far
        // ones. This is the property the Searing Gorge run failed — 42 of 48 fixtures 289 yd out.
        let far = 200.0 * (LAVA_LIGHTS_MAX / LAVA_LIGHTS_PER_SURFACE + 1) as f32;
        assert!(selected.iter().all(|(_, p)| p.x <= far));
        assert_eq!(selected[0].1, Vec3::ZERO);
    }

    fn app() -> App {
        let mut app = App::new();
        app.init_resource::<Time>();
        register(&mut app);
        app.world_mut()
            .spawn((crate::view::WorldCamera, GlobalTransform::IDENTITY));
        app
    }

    fn surface(app: &mut App, kind: LiquidKind, nibble: u8) -> Entity {
        app.world_mut()
            .spawn((
                LiquidSurface,
                LiquidSoundSource { nibble },
                grid(3, 3, vec![true; 4], kind),
            ))
            .id()
    }

    fn lights(app: &mut App) -> usize {
        app.world_mut()
            .query_filtered::<Entity, With<LavaLight>>()
            .iter(app.world())
            .count()
    }

    /// How many fixtures one full test surface is worth, derived rather than written down: the
    /// lattice constants move with the tuning and the count is not the subject of any test below.
    fn per_surface() -> usize {
        sample_grid(&grid(3, 3, vec![true; 4], LiquidKind::Magma)).len()
    }

    /// MONKEY (lava light: read) — THE REGRESSION TEST FOR THE DEFECT THE REPORT FOUND.
    ///
    /// The budget must be spent on the magma the camera is STANDING BY, and a pool that fits in
    /// one MCNK liquid layer must be able to take enough of it to read. The shipped
    /// `LAVA_LIGHTS_PER_SURFACE = 6` could not: MEASURED in Searing Gorge, the pool under the
    /// camera got six fixtures at ~12 yd spacing and the seventh-nearest fixture in the whole
    /// world was **289 yd away** — 42 of the 48-fixture budget spent where nothing could see it.
    ///
    /// The near surface here is worth more candidates than the cap, and the far one is a whole
    /// zone away; every slot must go to the near one.
    #[test]
    fn the_budget_goes_to_the_pool_the_camera_is_standing_by() {
        let mut world = World::new();
        let near = world.spawn_empty().id();
        let far = world.spawn_empty().id();
        let line = |x: f32| {
            (0..LAVA_LIGHTS_PER_SURFACE * 2)
                .map(|i| Vec3::new(x, 0.0, i as f32 * (MERGE_YD + 1.0)))
                .collect::<Vec<_>>()
        };
        let points = HashMap::from([(near, line(0.0)), (far, line(289.0))]);
        let selected = select(&points, Vec3::ZERO);
        assert_eq!(
            selected.iter().filter(|(s, _)| *s == near).count(),
            LAVA_LIGHTS_PER_SURFACE,
            "the near pool must be allowed to fill its whole per-surface allowance",
        );
        // One MCNK chunk is 33.3 yd across, so a pool that fits in one must get fixtures close
        // enough together to sum in the shader — `GRID_YD` apart, not a third of the chunk apart.
        assert!(
            LAVA_LIGHTS_PER_SURFACE as f32 * GRID_YD * GRID_YD >= 533.333 / 16.0 * 20.0,
            "a single-chunk pool must be able to cover itself at the sample spacing",
        );
    }

    #[test]
    fn gain_zero_removes_all_and_positive_gain_restores_without_surface_churn() {
        let mut app = app();
        surface(&mut app, LiquidKind::Magma, 6);
        app.update();
        assert_eq!(lights(&mut app), per_surface());
        app.world_mut().resource_mut::<LavaLightGain>().0 = 0.0;
        app.update();
        assert_eq!(lights(&mut app), 0);
        app.world_mut().resource_mut::<LavaLightGain>().0 = 2.0;
        app.update();
        assert_eq!(lights(&mut app), per_surface());
        for (light, reach) in app
            .world_mut()
            .query::<(&WorldPointLight, &LightReach)>()
            .iter(app.world())
        {
            assert!((INTENSITY * 1.8..=INTENSITY * 2.2).contains(&light.intensity));
            // The reach is the AUTHORED rung's own, not an independent number.
            assert_eq!(reach.0, super::super::m2_light_reach(AUTHORED));
            assert_eq!(reach.0, REACH_YD);
        }
    }

    #[test]
    fn streaming_removes_orphans_and_slime_never_emits() {
        let mut app = app();
        let magma = surface(&mut app, LiquidKind::Magma, 2);
        surface(&mut app, LiquidKind::Slime, 3);
        app.update();
        assert_eq!(lights(&mut app), per_surface());
        for (e, marker) in app
            .world_mut()
            .query::<(Entity, &LavaLight)>()
            .iter(app.world())
        {
            assert_eq!(marker.surface, magma);
            assert!(app.world().get::<DaylightFixture>(e).is_some());
            assert!(app.world().get::<super::super::FlameFlicker>(e).is_none());
            assert!(app
                .world()
                .get::<super::super::SyntheticFireLight>(e)
                .is_none());
            assert!(app.world().get::<ChildOf>(e).is_none());
        }
        app.world_mut().despawn(magma);
        app.update();
        assert_eq!(lights(&mut app), 0);
    }
}
