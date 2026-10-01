//! The shared **shadow rig** — the framework both shadow lanes plug into.
//!
//! A realtime shadow needs three things: something to CAST (a lane's caster geometry), a LIGHT +
//! shadow map to cast INTO, and RECEIVERS that read the map and darken. This module owns the middle
//! part — one directional "sun" on a private render layer, that layer's membership on the world
//! camera, the shadow-map resource, the invisible proxy caster material, and the sun aiming — plus
//! the geometry-collection helpers the lanes share. The receivers live in benilla's own shaders.
//!
//! The two lanes ([`super::character_shadow`], [`super::world_shadow`]) are EQUAL and independent:
//! each is its own plugin that depends only on this core, never on the other. They coordinate with
//! the rig through two resources, with a deliberate one-frame handshake:
//! - [`ShadowDemand`] — each active lane ORs `true` in (in the [`ShadowSet::Lanes`] set); the rig
//!   reads it next frame to decide whether the sun should exist. A lane that is removed simply stops
//!   contributing demand — the core needs no edit.
//! - [`ShadowFrame`] — the rig publishes the shared per-frame facts (is it active, the caster
//!   material, the camera/light position, the sun direction, the two collection reaches) for the
//!   lanes to read when they build their casters.
//!
//! Why ONE shared light and not one per lane: benilla's custom receiver shaders sample a SINGLE
//! directional light (they assign, not accumulate, over the light loop). Two lights would double the
//! shadow-pass cost and the last would win the receiver. So the lanes share this rig and each merely
//! contributes its own caster geometry into the one map.
//!
//! The rig deliberately does NOT put a directional light on the normal world layer: that would make
//! every existing WoW material enter Bevy's shadow-prepass path, the source of the pipeline
//! corruption seen during the first experiment. Only the private-layer proxy casts.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::{NoFrustumCulling, RenderLayers};
use bevy::ecs::entity::EntityHashSet;
use bevy::light::{
    CascadeShadowConfigBuilder, DirectionalLight, DirectionalLightShadowMap, NotShadowReceiver,
    ShadowFilteringMethod,
};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::pbr::{Material, MaterialPipeline, MaterialPlugin, MeshMaterial3d};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use benilla_assets::coords::wow_to_bevy;
use benilla_assets::materials::WowModelMaterial;
use benilla_formats::ModelBlend;
use benilla_world::billboard::BillboardCard;
use benilla_world::interact::PickMesh;
// MONKEY (moon shadows): `sun_shadow_strength` is the rig's half of the hand-over law — WHICH body
// this one directional light is aimed at. Imported rather than mirrored so the aim and the packed
// weight can never disagree about which map the receivers are reading.
use benilla_world::lighting::{
    moon_shadow_weight, sun_shadow_strength, MoonShadowStrength, ShadowBody, ShadowDistance,
    ShadowFilterGaussian, ShadowHandover, WowLighting,
};
use benilla_world::model_render::{ModelKind, ModelPart, ShadowOccluder};
use benilla_world::rig_palette::{RigPalettes, RigPart, RigSkin};
use benilla_world::view::WorldCamera;

use crate::char_select::ClientState;
use crate::video::VideoConfig;

/// Layer 31 is deliberately private to this feature. Bevy supports 32 render layers; the normal
/// world is layer 0 and the UI/portrait layers occupy the low numbered slots.
pub(crate) const PLAYER_SHADOW_LAYER: usize = 31;

/// Shipped realtime-shadow render distance in yards — the `shadowDistance` default. How far from the
/// camera the shadow map RESOLVES (the single cascade's `maximum_distance`). Collection reaches
/// farther than this (a caster's body may stand outside the cascade while its shadow lands inside).
pub(crate) const DEFAULT_SHADOW_DISTANCE: f32 = 80.0;

/// The `shadowDistance` slider range (yd). Min keeps a useful near shadow; max is bounded because
/// the ONE shadow map spreads thinner (softer) over a larger area and the caster-collection cost
/// grows with distance. (The map's edge is no longer the fixed 4096 this note was written against —
/// it is `shadowMapSize`, default 2048 — which makes the softening at long range twice as quick.)
pub(crate) const SHADOW_DISTANCE_RANGE: std::ops::RangeInclusive<f32> = 40.0..=200.0;

/// How far the camera may drift before a cached caster-collection pass must refresh. The collection
/// reach carries this as margin so coverage holds everywhere between refreshes.
pub(crate) const STATIC_REBUILD_STEP: f32 = 16.0;

// -------------------------------------------------------------------------------------------
// MONKEY (sun shadow perf): the five live dials over this rig's cost. Measured on an RTX 3070 at
// 1080p in the Lion's Pride Inn: 45-47 fps with both lanes on, 60-62 with `characterShadows 0`,
// 68-73 with `worldShadows 0` too — i.e. ~5 ms/frame in the sun lanes, the renderer's single
// biggest line item. The cost splits three ways and each dial takes one of them:
//   * the shadow PASS's fill + its depth texture .... `shadowMapSize` (quadratic in the edge)
//   * the RECEIVERS' PCF fetches ..................... `shadowFilter`  (9 samples vs 1)
//   * the CPU caster rebuild + GPU re-upload ......... `characterShadowRate` / `worldShadowRate`
// and `shadowCasterReach` trims the caster POPULATION those rebuilds walk. All live: nothing here
// is latched at boot, so the user A/Bs the whole set from one chat line.
// -------------------------------------------------------------------------------------------

/// The `shadowMapSize` ladder. Powers of two only — Bevy's `validate_shadow_map_size` rounds a
/// non-power-of-two up with a warning, so an off-ladder value would silently become a different
/// (and larger) map than the one the user typed.
pub(crate) const SHADOW_MAP_SIZES: [u32; 3] = [1024, 2048, 4096];

/// The shipped shadow-map edge. **2048**, down from the rig's original 4096 literal: the pass is
/// quadratic in this, and over ONE 80 yd cascade 2048 is ~26 texels/yd — finer than the Gaussian
/// receiver kernel resolves. The softer edge is the accepted trade; `shadowMapSize 4096` restores
/// the old crispness at the old price.
pub(crate) const DEFAULT_SHADOW_MAP_SIZE: u32 = 2048;

/// `shadowFilter` default: **1 = Gaussian**, the look the rig has always had. The cheap arm (0 =
/// Hardware2x2) is one comparison sample instead of nine and is where the receiver-side win is,
/// but it stair-steps edges, so it ships OFF and the user judges it.
pub(crate) const DEFAULT_SHADOW_FILTER: u32 = 1;

/// The `shadowFilter` ladder's top. Bevy also has `Temporal`, which is deliberately NOT offered:
/// it is a randomized filter that only resolves under `TemporalAntiAliasing`, which benilla's
/// world camera does not run — it would read as noise.
pub(crate) const MAX_SHADOW_FILTER: u32 = 1;

/// The shipped Hz cap on both lanes' per-frame caster rebuild. **30** — half of a 60 Hz frame's
/// rebuilds for a silhouette that lags at most 33 ms, which is under the reaction threshold for a
/// shadow you are not looking directly at.
pub(crate) const DEFAULT_SHADOW_RATE: u32 = 30;

/// The rate ladder's top. Above the frame rate the cap is inert, so 120 is "off" with headroom for
/// a high-refresh panel; `0` is the explicit "every frame" (the pre-cvar behaviour).
pub(crate) const MAX_SHADOW_RATE: u32 = 120;

/// The `shadowCasterReach` multiplier range. `1` is the untouched reach law. The floor is 0.25 and
/// not 0 because a 0 reach admits nothing and would read as "shadows broke", which is what
/// `worldShadows 0` / `characterShadows 0` are for.
pub(crate) const CASTER_REACH_RANGE: std::ops::RangeInclusive<f32> = 0.25..=2.0;

/// Snap a requested `shadowMapSize` onto [`SHADOW_MAP_SIZES`] — nearest in LOG space, so 1500 lands
/// on 1024 and 3000 on 4096 (halfway in ratio, not in texels, is what "one step" means here).
pub(crate) fn clamp_shadow_map_size(asked: u32) -> u32 {
    let asked = asked.clamp(
        SHADOW_MAP_SIZES[0],
        SHADOW_MAP_SIZES[SHADOW_MAP_SIZES.len() - 1],
    );
    *SHADOW_MAP_SIZES
        .iter()
        .min_by(|a, b| {
            let d = |v: u32| ((v as f32).ln() - (asked as f32).ln()).abs();
            d(**a).total_cmp(&d(**b))
        })
        .unwrap_or(&DEFAULT_SHADOW_MAP_SIZE)
}

/// A lane's rebuild cadence gate. Holds the timestamp of the last rebuild and answers "may I
/// rebuild now?" against a live Hz cap.
///
/// Deliberately time-based rather than frame-counted: the point is to bound the rebuild WORK per
/// second, and a frame counter would tighten the real cadence exactly when the frame rate is
/// already high (where the work is affordable) and loosen it when it drops (where it is not).
///
/// `rate == 0` means "every frame" — the pre-cvar behaviour, kept reachable so the cap can be
/// ruled out as the cause of any artefact in one keystroke.
#[derive(Default)]
pub(crate) struct RebuildRate {
    last: Option<f32>,
}

impl RebuildRate {
    /// True when a rebuild is due at `now` (seconds since app start) for a cap of `rate` Hz; the
    /// timestamp is taken on the way out, so a `true` consumes the slot. The FIRST call is always
    /// due — a lane that has never built has nothing to show.
    pub(crate) fn due(&mut self, now: f32, rate: u32) -> bool {
        if rate == 0 {
            self.last = Some(now);
            return true;
        }
        let interval = 1.0 / rate as f32;
        // `now < last` (a time reset) is treated as due rather than as a very long wait.
        let due = self
            .last
            .is_none_or(|last| now - last >= interval || now < last);
        if due {
            self.last = Some(now);
        }
        due
    }

    /// Forget the last rebuild, so the lane's next frame rebuilds immediately. Called when a lane
    /// tears its caster down — the mesh it was pacing no longer exists.
    pub(crate) fn reset(&mut self) {
        self.last = None;
    }
}

/// The tallest COMMON caster the reach law budgets for (the big Elwynn/Duskwood tree class, in world
/// units). Not a clamp on what casts — only on how far past the resolve range collection hunts.
const MAX_CASTER_HEIGHT: f32 = 35.0;

/// Ceiling on the shadow-reach extension: a degenerate near-horizontal sun must not explode the
/// collection radius (and with it the rebuild cost) unbounded.
const SHADOW_REACH_CAP: f32 = 120.0;

/// How far the LIVE sun may drift from the last-written direction before the shadow light snaps to
/// it — about one game minute of travel. The basis is quantised because a light that rotates EVERY
/// frame defeats the cascade fit's texel snap and crawls every shadow edge at WoW map coordinates
/// (~9,300 units out). Held between snaps the basis is bit-stable; only the shadow basis steps.
const SUN_SNAP_RADIANS: f32 = 8.0e-4;

/// The daytime elevation band (radians above the horizon) the shadow sun is clamped to. The visible
/// celestial sun sweeps roughly −10°..+85° and dips below the horizon at night; unclamped that would
/// INVERT the shadow (a set sun's travel points upward) or DEGENERATE the `looking_to` basis at high
/// noon (travel nearly ∥ up). Pinning to ~18°..65° keeps shadows visible, right-way-up and bounded
/// while the AZIMUTH still tracks the real sun — so shadows sweep through the day ("moving sun").
const MIN_SHADOW_SUN_ELEVATION: f32 = 0.314; // ~18°
const MAX_SHADOW_SUN_ELEVATION: f32 = 1.134; // ~65°

fn shadow_reach_extension(sun_travel: Vec3) -> f32 {
    // `sun_travel` is the direction the light TRAVELS (downward): -y is the vertical drop.
    let down = -sun_travel.y;
    if down <= 1e-3 {
        return SHADOW_REACH_CAP;
    }
    let horizontal = Vec3::new(sun_travel.x, 0.0, sun_travel.z).length();
    (MAX_CASTER_HEIGHT * horizontal / down).min(SHADOW_REACH_CAP)
}

/// The collection law for a lane that can contribute a TALL caster (a doodad/WMO exile, or the whole
/// static world): resolve range + rebuild margin + the sun-dependent shadow reach.
pub(crate) fn static_collection_reach(sun_travel: Vec3, distance: f32) -> f32 {
    distance + STATIC_REBUILD_STEP + shadow_reach_extension(sun_travel)
}

/// `WOW_SHADOW_TRACE=1` — log the caster population when it CHANGES, to attribute a vanished shadow
/// to a lane rather than the map/receiver.
pub(crate) fn shadow_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("WOW_SHADOW_TRACE").is_ok_and(|v| v != "0"))
}

fn sun_snap_due(held: Option<Vec3>, live: Vec3) -> bool {
    match held {
        None => true,
        Some(held) => held.angle_between(live) > SUN_SNAP_RADIANS,
    }
}

/// The shadow sun's TRAVEL direction (light propagation — points AWAY from the sun) from the visible
/// celestial TO-sun direction, elevation clamped to [`MIN_SHADOW_SUN_ELEVATION`,
/// `MAX_SHADOW_SUN_ELEVATION`]. Azimuth is preserved so shadows sweep with the real sun; only the
/// height is tamed so a set/zenith sun can't invert or degenerate the basis.
///
/// MONKEY (moon shadows): the MOON is aimed through this same function, unchanged. The clamp is
/// exactly what a moonrise needs — the white moon climbs from −10° through 0° to +55°, and an
/// unclamped basis at 1° of elevation stretches the cascade's footprint toward the horizon until
/// every shadow in it is a smear. The moon also has no separate azimuth problem to solve: its
/// bearing is the SAME constant 45° the celestial sun uses (`daynight::moon_direction`), so a moon
/// shadow falls along a midday sun shadow's compass line, only longer.
fn shadow_sun_travel(to_sun: Vec3) -> Vec3 {
    let s = to_sun.normalize_or_zero();
    if s == Vec3::ZERO {
        return Vec3::NEG_Z;
    }
    let horizontal = Vec3::new(s.x, 0.0, s.z);
    let bearing = if horizontal.length() > 1e-4 {
        horizontal.normalize()
    } else {
        Vec3::NEG_Z
    };
    let elevation =
        s.y.clamp(-1.0, 1.0)
            .asin()
            .clamp(MIN_SHADOW_SUN_ELEVATION, MAX_SHADOW_SUN_ELEVATION);
    let to_sun_clamped = bearing * elevation.cos() + Vec3::Y * elevation.sin();
    -to_sun_clamped
}

/// Marks a layer-31 caster mesh (either lane's). Kept for teardown/inspection symmetry.
#[derive(Component)]
pub(crate) struct ShadowCaster;

#[derive(Component)]
// MONKEY (volumetric fog): identify the celestial map without touching other lights.
pub(crate) struct ShadowSun;

#[derive(Component)]
struct ShadowCameraLayer {
    had_layers: bool,
}

/// MONKEY (moon shadows): the rig runs in PostUpdate BEFORE Propagate (spawns/aims the
/// sun, publishes [`ShadowFrame`]), then lanes run in Last in [`ShadowSet::Lanes`] — reading the frame
/// to build its casters and ORing its demand back in for the rig to read next frame.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum ShadowSet {
    Rig,
    Lanes,
}

/// Each active lane ORs `true` here (in [`ShadowSet::Lanes`]); the rig reads it the next frame to
/// decide whether the sun should exist, then resets it. A removed lane just stops contributing —
/// the core needs no edit to notice a lane is gone.
#[derive(Resource, Default)]
pub(crate) struct ShadowDemand(pub bool);

/// The rig's per-frame publication for the lanes: whether the rig is live, the shared proxy caster
/// material, and the shared collection inputs (camera/light position, sun direction, and the two
/// reaches). A lane builds nothing until `active` and only with the `material` here.
#[derive(Resource, Default)]
pub(crate) struct ShadowFrame {
    pub active: bool,
    /// MONKEY (moon shadows): keep cached geometry, but spend no rebuild work on an off night.
    pub suspended: bool,
    pub light_position: Vec3,
    pub sun_direction: Vec3,
    /// Reach for short casters (creatures/gameobjects — at most a body tall).
    pub entity_reach: f32,
    /// Reach for tall casters (doodads/WMO/static world) — carries the sun-dependent shadow reach.
    pub tall_reach: f32,
    pub material: Option<Handle<ShadowCasterMaterial>>,
}

/// The rig's own retained state — private to the core.
#[derive(Resource, Default)]
struct ShadowRigState {
    sun: Option<Entity>,
    /// The direction the sun was LAST GIVEN — the held shadow basis (see [`SUN_SNAP_RADIANS`]).
    sun_written: Option<Vec3>,
    /// The invisible proxy material shared by every lane's SOLID caster.
    material: Option<Handle<ShadowCasterMaterial>>,
    /// The cascade's currently-applied `maximum_distance` (the `shadowDistance` slider). The
    /// `CascadeShadowConfig` is only re-inserted when this changes — rebuilding it every frame would
    /// re-fit the cascade and defeat the texel snap (crawling every shadow edge).
    cascade_distance: Option<f32>,
    /// MONKEY (sun shadow perf): the last (map size, gaussian, character Hz, world Hz, reach)
    /// [`shadow_trace`] reported. Purely the trace's change detector — the values themselves are
    /// applied straight from [`VideoConfig`], never cached here.
    traced_quality: Option<(u32, bool, u32, u32, f32)>,
}

pub(crate) struct ShadowCorePlugin;

/// Opaque to Bevy's shadow pass, discarded by the forward pass — the private-layer proxy that lets a
/// caster participate in the real shadow map without drawing a second copy of the world. Shared by
/// every lane's SOLID caster.
#[derive(Asset, AsBindGroup, TypePath, Debug, Clone, Default)]
pub(crate) struct ShadowCasterMaterial {}

impl Material for ShadowCasterMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://benilla_app/shaders/shadow_caster.wgsl".into()
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &bevy::mesh::MeshVertexBufferLayoutRef,
        _key: bevy::pbr::MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // WoW batches are frequently two-sided; keep both faces so a mixed-winding batch's
        // projected silhouette has no holes.
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

impl Plugin for ShadowCorePlugin {
    fn build(&self, app: &mut App) {
        // MONKEY (sun shadow perf): the shipped edge is now 2048, and it is a LIVE cvar
        // (`apply_shadow_quality` writes this resource) rather than the 4096 boot literal this
        // line used to be. Bevy re-extracts the resource on change and `prepare_lights` re-keys
        // the texture cache with the new descriptor, so the map is genuinely re-created.
        app.insert_resource(DirectionalLightShadowMap {
            size: DEFAULT_SHADOW_MAP_SIZE as usize,
        })
        .add_plugins(MaterialPlugin::<ShadowCasterMaterial>::default())
        .init_resource::<ShadowRigState>()
        .init_resource::<ShadowDemand>()
        .init_resource::<ShadowFrame>()
        // The rig runs before the lanes every frame; the lanes read the frame it publishes.
        // MONKEY (moon shadows): after ALL Update work (lighting resolve + strength bridge),
        // before this frame's transform propagation and light-buffer packing. Last was too
        // late: a time jump had already packed the new body's weight over the old basis.
        .configure_sets(
            PostUpdate,
            ShadowSet::Rig.before(bevy::transform::TransformSystems::Propagate),
        )
        // MONKEY (sun shadow perf): the quality dials land BEFORE the rig each frame, and
        // unconditionally — a `shadowMapSize` write must take even while the lanes are off, so
        // turning shadows back on doesn't render one frame at the previous size.
        .add_systems(
            PostUpdate,
            (apply_shadow_quality, manage_rig)
                .chain()
                .in_set(ShadowSet::Rig),
        );
    }
}

/// MONKEY (sun shadow perf): push the "how expensive is a shadow" QUALITY dials from [`VideoConfig`]
/// to the places that own them, every frame, whether or not the lanes are up.
///
/// Three destinations, because Bevy spreads the shadow cost over three unrelated mechanisms:
/// - **`shadowMapSize`** → the [`DirectionalLightShadowMap`] resource. Bevy's `extract_lights`
///   re-publishes it into the render world on change and `prepare_lights` asks the texture cache
///   for a descriptor carrying the new edge — a different descriptor is a different cache entry, so
///   the depth texture is really re-created. Written through `ResMut` only on a genuine difference:
///   touching it every frame would re-extract (and re-validate) it every frame for nothing.
/// - **`shadowFilter`** → the world camera's [`ShadowFilteringMethod`], which keys the mesh
///   pipeline for every Bevy-material receiver (terrain, `wow_model`). Bevy has no "default"
///   component here — absent means Gaussian — but the component is inserted either way so the
///   value on the camera always names what is actually running.
/// - the same **`shadowFilter`** → [`ShadowFilterGaussian`], which `static_gx`'s hand-specialized
///   retained pipeline reads in the render world. Both must move together or the ground and the
///   buildings on it filter differently.
///
/// The rate dials need no push at all: each lane reads `VideoConfig` directly.
fn apply_shadow_quality(
    video: Res<VideoConfig>,
    mut rig: ResMut<ShadowRigState>,
    mut shadow_map: ResMut<DirectionalLightShadowMap>,
    mut filter_out: ResMut<ShadowFilterGaussian>,
    mut commands: Commands,
    cameras: Query<(Entity, Option<&ShadowFilteringMethod>), With<WorldCamera>>,
) {
    let size = clamp_shadow_map_size(video.shadow_map_size) as usize;
    if shadow_map.size != size {
        shadow_map.size = size;
    }

    let gaussian = video.shadow_filter != 0;
    let method = if gaussian {
        ShadowFilteringMethod::Gaussian
    } else {
        ShadowFilteringMethod::Hardware2x2
    };
    for (entity, current) in &cameras {
        if current != Some(&method) {
            commands.entity(entity).insert(method);
        }
    }
    if filter_out.0 != gaussian {
        filter_out.0 = gaussian;
    }

    if shadow_trace() {
        let now = (
            size as u32,
            gaussian,
            video.character_shadow_rate,
            video.world_shadow_rate,
            video.shadow_caster_reach,
        );
        if rig.traced_quality != Some(now) {
            info!(
                "shadow-trace: quality — map {}² | filter {} | rebuild caps character {} / world {} Hz | caster reach ×{:.2}",
                now.0,
                if gaussian { "gaussian(9)" } else { "hardware2x2(1)" },
                if now.2 == 0 { "every-frame".to_string() } else { now.2.to_string() },
                if now.3 == 0 { "every-frame".to_string() } else { now.3.to_string() },
                now.4,
            );
            rig.traced_quality = Some(now);
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn manage_rig(
    state: Res<State<ClientState>>,
    video: Res<VideoConfig>,
    lighting: Res<WowLighting>,
    time: Res<Time>,
    mut handover: ResMut<ShadowHandover>,
    mut demand: ResMut<ShadowDemand>,
    mut frame: ResMut<ShadowFrame>,
    mut rig: ResMut<ShadowRigState>,
    // MONKEY (distance slider): bridge the app-side `shadowDistance` to benilla-world so
    // `global_light` can pack it for the receivers' edge fade (which must fade at THIS distance).
    mut shadow_distance_out: ResMut<ShadowDistance>,
    // MONKEY (moon shadows): read-only here — the settings registry owns the write. `0` keeps the
    // rig aimed at the celestial sun all night, i.e. the pre-feature behaviour exactly.
    moon_strength: Res<MoonShadowStrength>,
    mut commands: Commands,
    mut materials: ResMut<Assets<ShadowCasterMaterial>>,
    mut cameras: Query<
        (
            Entity,
            Option<&mut RenderLayers>,
            Option<&ShadowCameraLayer>,
        ),
        With<WorldCamera>,
    >,
    mut suns: Query<(&mut Transform, &mut DirectionalLight), With<ShadowSun>>,
    cameras_for_position: Query<&GlobalTransform, With<WorldCamera>>,
) {
    let in_world = *state.get() == ClientState::InWorld;
    // The `shadowDistance` slider (already clamped by the cvar). Publish it to benilla-world every
    // frame so the receiver edge fade tracks it.
    let distance = video.shadow_distance;
    if shadow_distance_out.0 != distance {
        shadow_distance_out.0 = distance;
    }
    // Read the demand the lanes accumulated LAST frame, then reset so they re-accumulate this frame
    // (they run after us, in ShadowSet::Lanes). One-frame lag on rig spawn — imperceptible.
    let wanted = demand.0 && in_world;
    demand.0 = false;
    // MONKEY (moon shadows): publish zero immediately on a body change. Only the actual write
    // below acknowledges the basis; fresh spawns may need a deferred-command frame to get here.
    let sun_w = sun_shadow_strength(lighting.celestial_dir().y);
    handover.request(
        sun_w,
        moon_shadow_weight(lighting.celestial_dir().y, lighting.moon_dir().y),
        moon_strength.0,
        wanted,
        time.delta_secs(),
    );
    frame.suspended = sun_w <= 0.0 && moon_strength.0 <= 0.0;

    if !wanted {
        if let Some(entity) = rig.sun.take() {
            commands.entity(entity).despawn();
        }
        // Forget the held basis with the light, so a re-enable aims the fresh sun immediately.
        rig.sun_written = None;
        if let Some(handle) = rig.material.take() {
            materials.remove(handle.id());
        }
        for (entity, layers, marker) in &mut cameras {
            let Some(marker) = marker else { continue };
            if marker.had_layers {
                if let Some(mut layers) = layers {
                    *layers = layers.clone().without(PLAYER_SHADOW_LAYER);
                }
            } else {
                commands.entity(entity).remove::<RenderLayers>();
            }
            commands.entity(entity).remove::<ShadowCameraLayer>();
        }
        frame.active = false;
        frame.material = None;
        return;
    }

    // MONKEY (moon shadows): an externally removed light is not an acknowledged live rig.
    // Deferred first spawns have no written basis yet, so they are allowed their creation frame.
    if rig.sun_written.is_some() && rig.sun.is_some_and(|sun| !suns.contains(sun)) {
        rig.sun = None;
        rig.sun_written = None;
        *handover = ShadowHandover::default();
        handover.request(
            sun_w,
            moon_shadow_weight(lighting.celestial_dir().y, lighting.moon_dir().y),
            moon_strength.0,
            wanted,
            0.0,
        );
    }

    if rig.material.is_none() {
        rig.material = Some(materials.add(ShadowCasterMaterial {}));
    }
    if rig.sun.is_none() {
        let sun = commands
            .spawn((
                DirectionalLight {
                    // The light exists to populate a shadow map; WoW's shader lighting stays
                    // authoritative, so it must not add a second diffuse term.
                    illuminance: 0.0,
                    shadows_enabled: !frame.suspended,
                    shadow_depth_bias: 0.02,
                    shadow_normal_bias: 0.8,
                    ..default()
                },
                CascadeShadowConfigBuilder {
                    num_cascades: 1,
                    minimum_distance: 0.1,
                    maximum_distance: distance,
                    ..default()
                }
                .build(),
                Transform::IDENTITY,
                Visibility::Visible,
                RenderLayers::layer(PLAYER_SHADOW_LAYER),
                ShadowSun,
            ))
            .id();
        rig.sun = Some(sun);
        rig.sun_written = None;
        rig.cascade_distance = Some(distance);
    }
    // Re-fit the cascade only when the slider changed (not every frame — that would defeat the
    // texel snap and crawl every shadow edge).
    if rig.cascade_distance != Some(distance) {
        if let Some(sun) = rig.sun {
            commands.entity(sun).insert(
                CascadeShadowConfigBuilder {
                    num_cascades: 1,
                    minimum_distance: 0.1,
                    maximum_distance: distance,
                    ..default()
                }
                .build(),
            );
        }
        rig.cascade_distance = Some(distance);
    }

    for (entity, layers, marker) in &mut cameras {
        if marker.is_some() {
            if let Some(mut layers) = layers {
                *layers = layers.clone().with(PLAYER_SHADOW_LAYER);
            }
        } else if let Some(mut layers) = layers {
            *layers = layers.clone().with(PLAYER_SHADOW_LAYER);
            commands
                .entity(entity)
                .insert(ShadowCameraLayer { had_layers: true });
        } else {
            commands.entity(entity).insert((
                RenderLayers::default().with(PLAYER_SHADOW_LAYER),
                ShadowCameraLayer { had_layers: false },
            ));
        }
    }

    let light_position = cameras_for_position
        .iter()
        .next()
        .map(GlobalTransform::translation)
        .unwrap_or(Vec3::ZERO);

    // MONKEY (moon shadows): request/acknowledge rather than assuming the clock is continuous.
    // The published weight stays zero across the write and its propagation frame, including a
    // noon-to-midnight jump, login and a strength-zero-to-enabled transition at midnight.
    let aim_at_moon = handover.wanted == Some(ShadowBody::Moon);
    let sun_direction = shadow_sun_travel(if aim_at_moon {
        lighting.moon_dir()
    } else {
        lighting.celestial_dir()
    });
    // MONKEY (sun shadow perf): `shadowCasterReach` scales BOTH reaches by the same factor, after
    // the reach law rather than inside it — the law's terms (resolve range, rebuild margin, the
    // sun-elevation extension) each mean something, and a dial that rewrote one of them would
    // change what the others compensate for. A flat multiplier is honestly just "collect less".
    let reach_scale = video
        .shadow_caster_reach
        .clamp(*CASTER_REACH_RANGE.start(), *CASTER_REACH_RANGE.end());
    let entity_reach = (distance + STATIC_REBUILD_STEP) * reach_scale;
    let tall_reach = static_collection_reach(sun_direction, distance) * reach_scale;

    if let Some(sun) = rig.sun {
        // Only the rotation matters; QUANTISED to hold the cascade texel snap (see SUN_SNAP_RADIANS).
        if let Ok((mut current, mut light)) = suns.get_mut(sun) {
            if light.shadows_enabled == frame.suspended {
                light.shadows_enabled = !frame.suspended;
            }
            if handover.aimed != handover.wanted || sun_snap_due(rig.sun_written, sun_direction) {
                *current = Transform::IDENTITY.looking_to(sun_direction, Vec3::Y);
                rig.sun_written = Some(sun_direction);
                if let Some(body) = handover.wanted {
                    handover.aim_written(body);
                }
            }
        }
    }

    frame.active = rig.sun.is_some_and(|sun| suns.contains(sun));
    frame.light_position = light_position;
    frame.sun_direction = sun_direction;
    frame.entity_reach = entity_reach;
    frame.tall_reach = tall_reach;
    frame.material = rig.material.clone();
}

// ---------------------------------------------------------------------------------------------
// Shared caster-building helpers, used by both lanes.
// ---------------------------------------------------------------------------------------------

/// Spawn a SOLID caster entity on the private shadow layer with the shared proxy material.
pub(crate) fn spawn_solid_caster(
    commands: &mut Commands,
    mesh: Handle<Mesh>,
    material: Handle<ShadowCasterMaterial>,
) -> Entity {
    commands
        .spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            Transform::IDENTITY,
            Visibility::Visible,
            RenderLayers::layer(PLAYER_SHADOW_LAYER),
            NoFrustumCulling,
            NotShadowReceiver,
            ShadowCaster,
        ))
        .id()
}

/// Append a triangle list, dropping WHOLE triangles whose indices exceed the vertex range (filtering
/// single indices out of a `TriangleList` rewires the rest of the submesh into garbage).
pub(crate) fn append_triangles(
    source: &[u32],
    vertex_count: u32,
    base: u32,
    indices: &mut Vec<u32>,
) {
    for triangle in source.as_chunks::<3>().0 {
        if triangle.iter().all(|index| *index < vertex_count) {
            indices.extend(triangle.iter().map(|index| base + *index));
        }
    }
}

/// Take a caster mesh's own buffers so a rebuild reuses their allocations. The mesh is left
/// attribute-less until [`restore_mesh_buffers`] puts them back.
pub(crate) fn take_mesh_buffers(mesh: &mut Mesh) -> (Vec<[f32; 3]>, Vec<u32>) {
    let mut positions = match mesh.remove_attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(bevy::mesh::VertexAttributeValues::Float32x3(values)) => values,
        _ => Vec::new(),
    };
    let mut indices = match mesh.remove_indices() {
        Some(Indices::U32(values)) => values,
        _ => Vec::new(),
    };
    positions.clear();
    indices.clear();
    (positions, indices)
}

pub(crate) fn restore_mesh_buffers(mesh: &mut Mesh, positions: Vec<[f32; 3]>, indices: Vec<u32>) {
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_indices(Indices::U32(indices));
}

pub(crate) fn empty_shadow_mesh() -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new());
    mesh.insert_indices(Indices::U32(Vec::new()));
    mesh
}

/// A cutout caster mesh: like [`empty_shadow_mesh`] but carrying a `UV_0` lane, because the
/// alpha-tested prepass fragment samples the leaf sheet at these UVs. The mesh MUST declare UV_0
/// (even empty) so Bevy's prepass specializer sets `VERTEX_UVS_A` and the fragment's `in.uv` exists.
pub(crate) fn empty_cutout_mesh() -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new());
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, Vec::<[f32; 2]>::new());
    mesh.insert_indices(Indices::U32(Vec::new()));
    mesh
}

/// Collect the per-frame ENTITY caster geometry a lane wants into its buffers. `want_creatures` /
/// `want_environment` select which kinds this caster admits — the CHARACTER lane wants creatures,
/// the WORLD lane wants entity-resident environment (gameobjects, faded doodads, WMO props). Returns
/// `(admitted, occluder-rejected)` for the trace. Range-gates before any vertex work. The `parts`
/// query is written inline (not aliased) so a lane can pass its own identically-typed query.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn collect_entity_geometry(
    parts: &Query<
        (
            Entity,
            &PickMesh,
            &ModelPart,
            Option<&GlobalTransform>,
            Option<&RigPart>,
            &ShadowOccluder,
            Option<&MeshMaterial3d<WowModelMaterial>>,
        ),
        Without<BillboardCard>,
    >,
    rigs: &Query<&RigSkin>,
    palettes: &RigPalettes,
    want_creatures: bool,
    want_environment: bool,
    light_position: Vec3,
    creature_reach: f32,
    doodad_reach: f32,
    positions: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
    mut built_parts: Option<&mut EntityHashSet>,
) -> (u32, u32) {
    let (mut admitted, mut rejected) = (0u32, 0u32);
    for (entity, pick, part, global, rig_part, occluder, _material) in parts.iter() {
        if !casts_realtime_shadow(part.kind, part.blend, want_creatures, want_environment) {
            continue;
        }
        // The content half of the visibility law (toggles, far clip, distance fade, material
        // alpha) — NOT `InheritedVisibility`, which folds camera-orientation verdicts a shadow
        // caster must ignore (a tree behind the camera still blocks the sun).
        if !occluder.0 {
            rejected += 1;
            continue;
        }
        let reach = match part.kind {
            ModelKind::Creature | ModelKind::GameObject => creature_reach,
            ModelKind::Doodad | ModelKind::Wmo => doodad_reach,
        };
        let anchor = global.map(GlobalTransform::translation).or_else(|| {
            rig_part
                .and_then(|rig_part| rigs.get(rig_part.0).ok())
                .and_then(|rig| palettes.slot_origin(rig.slot))
        });
        let Some(anchor) = anchor else { continue };
        if anchor.distance_squared(light_position) > reach * reach {
            continue;
        }
        admitted += 1;
        let base = positions.len() as u32;
        if let Some(rig_part) = rig_part {
            append_skinned(pick, rig_part, rigs, palettes, positions);
        } else if let Some(global) = global {
            positions.extend(
                pick.0
                    .positions
                    .iter()
                    .map(|p| global.transform_point(wow_to_bevy(*p)).to_array()),
            );
        }
        let added = positions.len() as u32 - base;
        if added != 0 {
            let before = indices.len();
            append_triangles(&pick.0.indices, added, base, indices);
            // MONKEY (moon shadows): readiness is a successful triangle append, not admission
            // or a material handle. Missing palettes / pending mesh data must keep the oval.
            if indices.len() > before {
                if let Some(built) = built_parts.as_deref_mut() {
                    built.insert(entity);
                }
            }
        }
    }
    (admitted, rejected)
}

/// The per-frame entity caster's admit law. CHARACTERS (creatures/players/mounts) are opaque +
/// alpha-test (hair/fringe cards are small, so casting them solid doesn't box out). Entity-resident
/// ENVIRONMENT (gameobjects, faded doodads, WMO props) is OPAQUE only — a foliage leaf-card cast
/// solid becomes a box; retained-world cutout is cast leaf-shaped by the world lane's alpha-tested
/// material instead. Additive/transparent geometry never casts a solid shadow.
fn casts_realtime_shadow(
    kind: ModelKind,
    blend: ModelBlend,
    want_creatures: bool,
    want_environment: bool,
) -> bool {
    match kind {
        ModelKind::Creature => {
            want_creatures && matches!(blend, ModelBlend::Opaque | ModelBlend::AlphaTest)
        }
        _ => want_environment && matches!(blend, ModelBlend::Opaque),
    }
}

fn append_skinned(
    pick: &PickMesh,
    rig_part: &RigPart,
    rigs: &Query<&RigSkin>,
    palettes: &RigPalettes,
    positions: &mut Vec<[f32; 3]>,
) {
    let Ok(rig) = rigs.get(rig_part.0) else {
        return;
    };
    let Some(palette) = palettes.world_palette(rig.slot, rig.bones() as usize) else {
        return;
    };
    for (index, position) in pick.0.positions.iter().enumerate() {
        // A vertex without skinning data still gets an entry, or `positions` shortens against the
        // index buffer and rewires later triangles. Seat it on the root matrix.
        let (Some(joints), Some(weights)) = (pick.0.joints.get(index), pick.0.weights.get(index))
        else {
            let fallback = palette
                .first()
                .map(|matrix| matrix.transform_point3(wow_to_bevy(*position)))
                .unwrap_or(Vec3::ZERO);
            positions.push(fallback.to_array());
            continue;
        };
        let mut skinned = Vec3::ZERO;
        for lane in 0..4 {
            if weights[lane] > 0.0 {
                if let Some(matrix) = palette.get(joints[lane] as usize) {
                    skinned += matrix.transform_point3(wow_to_bevy(*position)) * weights[lane];
                }
            }
        }
        positions.push(skinned.to_array());
    }
}

#[cfg(test)]
mod tests {
    use super::{
        append_triangles, casts_realtime_shadow, clamp_shadow_map_size, shadow_reach_extension,
        shadow_sun_travel, static_collection_reach, sun_snap_due, RebuildRate,
        DEFAULT_SHADOW_DISTANCE, DEFAULT_SHADOW_MAP_SIZE, MAX_CASTER_HEIGHT,
        MAX_SHADOW_SUN_ELEVATION, MIN_SHADOW_SUN_ELEVATION, SHADOW_MAP_SIZES, SHADOW_REACH_CAP,
        STATIC_REBUILD_STEP, SUN_SNAP_RADIANS,
    };
    use benilla_formats::ModelBlend;
    use benilla_world::model_render::ModelKind;
    use bevy::prelude::{Quat, Vec3};

    #[test]
    fn each_lane_admits_only_its_own_kinds() {
        // The character lane wants creatures (opaque + alpha-test), not environment.
        assert!(casts_realtime_shadow(
            ModelKind::Creature,
            ModelBlend::Opaque,
            true,
            false
        ));
        assert!(casts_realtime_shadow(
            ModelKind::Creature,
            ModelBlend::AlphaTest,
            true,
            false
        ));
        assert!(!casts_realtime_shadow(
            ModelKind::Creature,
            ModelBlend::Blend,
            true,
            false
        ));
        assert!(!casts_realtime_shadow(
            ModelKind::Doodad,
            ModelBlend::Opaque,
            true,
            false
        ));
        // The world lane wants environment (opaque only), not creatures.
        for kind in [ModelKind::GameObject, ModelKind::Doodad, ModelKind::Wmo] {
            assert!(casts_realtime_shadow(kind, ModelBlend::Opaque, false, true));
            assert!(!casts_realtime_shadow(
                kind,
                ModelBlend::AlphaTest,
                false,
                true
            ));
        }
        assert!(!casts_realtime_shadow(
            ModelKind::Creature,
            ModelBlend::Opaque,
            false,
            true
        ));
    }

    #[test]
    fn the_moving_sun_is_clamped_in_elevation_but_keeps_its_bearing() {
        // A SET sun (to-sun below the horizon) is lifted to the floor; travel still points DOWN.
        let set = Vec3::new(0.6, -0.5, 0.6).normalize();
        let t = shadow_sun_travel(set);
        assert!(t.y < 0.0, "travel points down even for a set sun");
        assert!(
            ((-t.y).asin() - MIN_SHADOW_SUN_ELEVATION).abs() < 1e-3,
            "a set sun is pinned to the floor elevation"
        );
        // A near-ZENITH sun is capped so `looking_to(travel, Y)` never degenerates.
        let high = Vec3::new(0.02, 0.999, 0.02).normalize();
        assert!(
            ((-shadow_sun_travel(high).y).asin() - MAX_SHADOW_SUN_ELEVATION).abs() < 1e-2,
            "a near-zenith sun is capped at the ceiling elevation"
        );
        // AZIMUTH preserved: a sun bearing due +X throws its travel toward -X.
        let east = Vec3::new(1.0, 0.4, 0.0).normalize();
        let te = shadow_sun_travel(east);
        assert!(
            te.x < 0.0 && te.z.abs() < 1e-3,
            "travel points away from the sun's horizontal bearing"
        );
        assert_eq!(shadow_sun_travel(Vec3::ZERO), Vec3::NEG_Z);
    }

    /// MONKEY (moon shadows): the RIG's half of the hand-over — WHICH body the one directional
    /// light is aimed at, and that the swap lands where nothing is being cast.
    ///
    /// `manage_rig`'s predicate is `moonShadowStrength > 0 && sun_shadow_strength(sun.y) <= 0`.
    /// What has to hold for that to be a swap and not a POP is that on the frame it flips, BOTH
    /// weights are zero — the sun's because the predicate says so, the moon's because
    /// `moon_shadow_weight` has its own elevation ramp and the moon is still under the horizon
    /// there. (The weights' own continuity over the whole game day is
    /// `benilla_world`'s `the_two_shadow_weights_never_overlap_and_neither_jumps`; this is the
    /// aiming side of the same contract, and the reason the two are not one test is that the aim
    /// lives in this crate and the weights in that one.)
    ///
    /// The third assertion is the one a reviewer should look at hardest: a moon at or under the
    /// horizon must still produce a DOWNWARD, elevation-clamped basis. Unclamped, a moonrise aim
    /// either inverts (a below-horizon body's travel points up, so every shadow falls the wrong
    /// way) or stretches the cascade's footprint toward the horizon until the map resolves nothing.
    #[test]
    fn the_rig_hands_over_to_the_moon_only_where_neither_body_casts() {
        use benilla_world::lighting::{moon_shadow_weight, sun_shadow_strength};

        // Dusk, the frame the aim flips: the sun has just reached the horizon.
        let sun_down = -0.02_f32;
        assert_eq!(
            sun_shadow_strength(sun_down),
            0.0,
            "the sun casts nothing at the horizon"
        );
        // …and the moon is still under it (the shipped tables put moonrise ~1h45m later).
        let moon_under = -0.17_f32;
        assert_eq!(
            moon_shadow_weight(sun_down, moon_under),
            0.0,
            "the moon must not start casting the instant the sun stops — the swap has to happen \
             where BOTH weights are zero or the map's contents change under a live weight"
        );
        // While the sun is still up the moon is refused outright, whatever its elevation.
        assert!(sun_shadow_strength(0.5) > 0.0);
        assert_eq!(
            moon_shadow_weight(0.5, 0.9),
            0.0,
            "one body casts at a time"
        );
        // A high midnight moon does cast, and at full weight (the strength dial scales it later).
        assert_eq!(moon_shadow_weight(sun_down, 0.82), 1.0);
        // The aim itself: a below-horizon moon still travels DOWN, pinned at the floor elevation —
        // the same treatment `shadow_sun_travel` gives a set sun.
        let t = shadow_sun_travel(Vec3::new(0.7, moon_under, 0.7).normalize());
        assert!(t.y < 0.0, "a moonrise basis still points down");
        assert!(
            ((-t.y).asin() - MIN_SHADOW_SUN_ELEVATION).abs() < 1e-3,
            "a low moon is pinned to the floor elevation, so its shadows cannot smear to the horizon"
        );
    }

    #[test]
    fn the_shadow_sun_snaps_on_first_write_holds_under_drift_and_rearms() {
        let held = Vec3::new(0.3, -0.8, 0.52).normalize();
        assert!(sun_snap_due(None, held), "first write always fires");
        let axis = Vec3::Y;
        let small = Quat::from_axis_angle(axis, SUN_SNAP_RADIANS * 0.5) * held;
        assert!(
            !sun_snap_due(Some(held), small),
            "sub-threshold drift holds"
        );
        let large = Quat::from_axis_angle(axis, SUN_SNAP_RADIANS * 2.0) * held;
        assert!(sun_snap_due(Some(held), large), "over-threshold snaps");
        assert!(
            !sun_snap_due(Some(large), large),
            "identical direction never rewrites"
        );
    }

    #[test]
    fn shadow_reach_follows_the_sun_and_is_capped() {
        let diagonal = Vec3::new(1.0, -1.0, 0.0).normalize();
        assert!((shadow_reach_extension(diagonal) - MAX_CASTER_HEIGHT).abs() < 1e-3);
        assert_eq!(shadow_reach_extension(Vec3::NEG_Y), 0.0);
        assert_eq!(shadow_reach_extension(Vec3::NEG_Z), SHADOW_REACH_CAP);
        assert!(
            static_collection_reach(Vec3::NEG_Y, DEFAULT_SHADOW_DISTANCE)
                >= DEFAULT_SHADOW_DISTANCE + STATIC_REBUILD_STEP
        );
    }

    #[test]
    fn an_out_of_range_index_drops_its_whole_triangle_and_no_others() {
        let mut out = Vec::new();
        append_triangles(&[0, 1, 2, 1, 2, 9, 2, 0, 1], 3, 10, &mut out);
        assert_eq!(out, vec![10, 11, 12, 12, 10, 11]);
    }

    /// MONKEY (sun shadow perf): the rate gate's arithmetic, which is the whole of the two lanes'
    /// new behaviour. Three properties, each one a way the cap could go wrong in a way you would
    /// only notice as a stuttering or frozen shadow:
    /// 1. the FIRST call is always due (a lane with no mesh yet must not wait a frame-interval),
    /// 2. a due call CONSUMES the slot (otherwise a 30 Hz cap rebuilds every frame forever),
    /// 3. the interval is measured from the last GRANT, not from the last ask.
    #[test]
    fn the_rebuild_gate_paces_from_the_last_grant_and_first_call_is_always_due() {
        let mut gate = RebuildRate::default();
        assert!(gate.due(10.0, 30), "first call has nothing to show yet");
        assert!(
            !gate.due(10.01, 30),
            "10 ms later is inside a 33 ms interval"
        );
        assert!(
            !gate.due(10.03, 30),
            "still inside, measured from the grant"
        );
        assert!(gate.due(10.04, 30), "past 1/30 s — due again");
        assert!(!gate.due(10.05, 30), "the grant reset the clock");
        // A reset (the lane tore its caster down) re-arms immediately.
        gate.reset();
        assert!(gate.due(10.051, 30));
    }

    /// `rate == 0` is the documented "every frame" escape hatch, and a backwards clock (a time
    /// reset) must read as due rather than as a very long wait — a lane frozen until the clock
    /// catches up would look exactly like a broken shadow.
    #[test]
    fn the_rebuild_gate_has_an_every_frame_arm_and_survives_a_clock_reset() {
        let mut gate = RebuildRate::default();
        for t in 0..5 {
            assert!(gate.due(100.0 + t as f32 * 0.001, 0), "rate 0 never gates");
        }
        let mut gate = RebuildRate::default();
        assert!(gate.due(100.0, 15));
        assert!(!gate.due(100.01, 15));
        assert!(
            gate.due(0.0, 15),
            "a clock that went backwards is due, not stuck"
        );
    }

    /// The map-size ladder snaps in LOG space, so "halfway" means halfway in ratio (1448 = √2·1024)
    /// rather than in texels — and every answer is a power of two, because Bevy silently rounds a
    /// non-power-of-two UP and the user would get a bigger, slower map than the one they typed.
    #[test]
    fn the_shadow_map_size_snaps_onto_the_power_of_two_ladder() {
        assert_eq!(clamp_shadow_map_size(2048), 2048);
        assert_eq!(clamp_shadow_map_size(1024), 1024);
        assert_eq!(clamp_shadow_map_size(4096), 4096);
        assert_eq!(clamp_shadow_map_size(0), 1024, "clamped to the floor");
        assert_eq!(
            clamp_shadow_map_size(99_999),
            4096,
            "clamped to the ceiling"
        );
        assert_eq!(
            clamp_shadow_map_size(1500),
            2048,
            "1500 is above the 1448 log midpoint"
        );
        assert_eq!(clamp_shadow_map_size(1400), 1024, "1400 < 1448");
        assert!(SHADOW_MAP_SIZES.iter().all(|s| s.is_power_of_two()));
        assert!(SHADOW_MAP_SIZES.contains(&DEFAULT_SHADOW_MAP_SIZE));
    }
}
