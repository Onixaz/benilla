//! The WORLD shadow lane — a thin plug-in on top of [`super::shadow_core`], equal to and independent
//! of [`super::character_shadow`].
//!
//! It owns the shadows cast by the WORLD: the retained `static_gx` geometry (trees + buildings, cast
//! solid), the alpha-tested foliage (leaf canopies, one caster per leaf texture), and the per-frame
//! ENTITY-resident environment (gameobjects, distance-faded doodads, WMO props). When `worldShadows`
//! is on it declares demand to the shared rig and, while the rig is live, builds those casters using
//! the rig's shared material + per-frame facts ([`ShadowFrame`]). It also drives the terrain
//! MCSH-suppression flag ([`WorldShadowActive`]) — baked terrain shadows switch off only for THIS
//! lane. Remove this module + its plugin registration + the `worldShadows` cvar and the character
//! lane is untouched.
//!
//! Cadence: the static world doesn't move, so its solid + cutout casters rebuild only on camera
//! drift ([`STATIC_REBUILD_STEP`]); the environment-entity caster (doodads fade in/out, gameobjects
//! spawn) is collected fresh every frame like the character lane.

use bevy::asset::AssetId;
use bevy::camera::visibility::{NoFrustumCulling, RenderLayers};
use bevy::image::Image;
use bevy::light::NotShadowReceiver;
use bevy::mesh::Indices;
use bevy::pbr::{Material, MaterialPipeline, MaterialPlugin, MeshMaterial3d};
use bevy::platform::collections::HashSet;
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use benilla_assets::materials::WowModelMaterial;
use benilla_world::billboard::BillboardCard;
use benilla_world::interact::PickMesh;
use benilla_world::lighting::WorldShadowActive;
use benilla_world::model_render::{ModelPart, ShadowOccluder};
use benilla_world::rig_palette::{RigPalettes, RigPart, RigSkin};
use benilla_world::static_gx::StaticGx;

use crate::shadow_core::{
    collect_entity_geometry, empty_cutout_mesh, empty_shadow_mesh, restore_mesh_buffers,
    shadow_trace, spawn_solid_caster, take_mesh_buffers, RebuildRate, ShadowCaster, ShadowDemand,
    ShadowFrame, ShadowSet, PLAYER_SHADOW_LAYER, STATIC_REBUILD_STEP,
};
use crate::video::VideoConfig;

pub(crate) struct WorldShadowPlugin;

impl Plugin for WorldShadowPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<CutoutShadowCasterMaterial>::default())
            .init_resource::<WorldLane>()
            // MONKEY (followups): refresh the static caster cache when streamed geometry arrives.
            // This used to live in the volumetric-fog plugin and so only ran with that cvar on.
            .add_systems(Last, refresh_streamed_shadows.before(ShadowSet::Lanes))
            .add_systems(Last, update_world_shadows.in_set(ShadowSet::Lanes));
    }
}

/// MONKEY (followups): the static caster rebuilds on 16 yd of camera drift, so a standing camera
/// never picks up trees and doodads that stream in later. Invalidate it whenever the retained
/// scene's residency changes, while world shadows are on (Classic has them off, so it is inert).
fn refresh_streamed_shadows(
    video: Res<VideoConfig>,
    gx: Option<Res<StaticGx>>,
    mut lane: ResMut<WorldLane>,
    mut seen: Local<Option<u64>>,
) {
    if !video.world_shadows {
        *seen = None;
        return;
    }
    let Some(gx) = gx else { return };
    let generation = gx.torch_residency_generation();
    if *seen != Some(generation) {
        lane.invalidate_static();
        *seen = Some(generation);
    }
}

/// The alpha-tested twin of [`ShadowCasterMaterial`]: same invisible-in-forward proxy, but its
/// prepass fragment samples a leaf sheet and discards transparent texels so cutout FOLIAGE casts a
/// leaf-shaped silhouette instead of the solid box a positions-only proxy throws. One instance per
/// distinct leaf texture. `AlphaMode::Mask` sets Bevy's `MAY_DISCARD` key and the EXPLICIT
/// `prepass_fragment_shader` makes the otherwise depth-only directional-shadow pass run a fragment —
/// the exact pair Bevy's prepass specializer requires to alpha-test in the shadow map.
#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
// MONKEY (leftovers): crate-visible for the pipe_warm menagerie.
pub(crate) struct CutoutShadowCasterMaterial {
    /// The leaf sheet, bound at material binding 0/1. Reusing the SAME image handle the forward pass
    /// draws makes the sampler inherit that image's clamp/repeat address mode, so the shadow
    /// silhouette matches the visible foliage (a cutout card's out-of-range UVs must clamp).
    #[texture(0)]
    #[sampler(1)]
    pub(crate) leaf: Handle<Image>,
}

impl Material for CutoutShadowCasterMaterial {
    fn fragment_shader() -> ShaderRef {
        // Forward pass: still fully invisible — the shared proxy fragment discards every fragment.
        "embedded://benilla_app/shaders/shadow_caster.wgsl".into()
    }

    fn prepass_fragment_shader() -> ShaderRef {
        // EXPLICIT → the shadow pass runs this fragment for the MAY_DISCARD material and alpha-tests.
        "embedded://benilla_app/shaders/shadow_caster_cutout_prepass.wgsl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Mask(benilla_assets::materials::VANILLA_ALPHA_KEY_REF)
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &bevy::mesh::MeshVertexBufferLayoutRef,
        _key: bevy::pbr::MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Two-sided like the solid caster; applies to the shadow (prepass) pipeline too.
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

/// One live alpha-tested foliage caster: its layer-31 entity, its per-texture mesh, and the material
/// carrying that one leaf sheet. Keyed by the leaf texture's `AssetId` so it persists across rebuilds.
struct CutoutCaster {
    entity: Entity,
    mesh: Handle<Mesh>,
    material: Handle<CutoutShadowCasterMaterial>,
}

/// The world lane's retained casters.
#[derive(Resource, Default)]
// MONKEY (volumetric fog, followups): the streamed-residency refresh invalidates the cache.
pub(crate) struct WorldLane {
    /// The cached static caster — the retained `static_gx` world (trees + buildings), rebuilt on drift.
    static_caster: Option<Entity>,
    static_mesh: Option<Handle<Mesh>>,
    static_rebuilt_at: Option<Vec3>,
    /// The alpha-tested foliage casters — one per distinct leaf texture.
    cutout: bevy::platform::collections::HashMap<AssetId<Image>, CutoutCaster>,
    /// The per-frame ENTITY-resident environment caster (gameobjects, faded doodads, WMO props).
    env_caster: Option<Entity>,
    env_mesh: Option<Handle<Mesh>>,
    env_traced: (u32, u32, u32),
    /// MONKEY (sun shadow perf): the `worldShadowRate` cadence gate on the ENVIRONMENT caster only.
    /// The static solid + cutout casters keep their own, much coarser cadence (16 yd camera drift).
    env_rate: RebuildRate,
}

// MONKEY (followups): `refresh_streamed_shadows` decides when streamed geometry needs a fresh map.
impl WorldLane {
    pub(crate) fn invalidate_static(&mut self) {
        self.static_rebuilt_at = None;
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn update_world_shadows(
    video: Res<VideoConfig>,
    time: Res<Time>,
    mut demand: ResMut<ShadowDemand>,
    frame: Res<ShadowFrame>,
    mut lane: ResMut<WorldLane>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut cutout_materials: ResMut<Assets<CutoutShadowCasterMaterial>>,
    // The retained static world — `Option` because it exists only when the retained pass is armed.
    gx: Option<Res<StaticGx>>,
    // The terrain MCSH-suppression flag (this lane owns it).
    mut world_active: ResMut<WorldShadowActive>,
    parts: Query<
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
    rigs: Query<&RigSkin>,
    palettes: Res<RigPalettes>,
) {
    // Declare demand so the shared rig stays up while this lane is on (rig reads it next frame).
    demand.0 = demand.0 || video.world_shadows;
    // The lane is casting when the rig is live AND its cvar is on. The MCSH terrain-shadow
    // suppression keys on exactly this (a character-only shadow sun must leave baked MCSH alone).
    let world_on = frame.active && video.world_shadows;
    if world_active.0 != world_on {
        world_active.0 = world_on;
    }

    if !world_on {
        teardown(&mut lane, &mut commands, &mut meshes, &mut cutout_materials);
        return;
    }
    // MONKEY (moon shadows): no receiver can use the off-night map. Preserve cached geometry,
    // but suspend BOTH entity and retained-world rebuilds until a body can cast again.
    if frame.suspended {
        lane.env_rate.reset();
        return;
    }
    let Some(material) = frame.material.clone() else {
        return;
    };

    // (1) The per-frame ENTITY-resident environment caster (opaque doodads/gameobjects/WMO props).
    if lane.env_caster.is_none() {
        let mesh = meshes.add(empty_shadow_mesh());
        let entity = spawn_solid_caster(&mut commands, mesh.clone(), material.clone());
        lane.env_caster = Some(entity);
        lane.env_mesh = Some(mesh);
        // A brand-new caster is an EMPTY mesh; make sure this frame fills it.
        lane.env_rate.reset();
    }
    // MONKEY (sun shadow perf): the `worldShadowRate` gate — the world lane's twin of the character
    // lane's. Only THIS caster is paced: the static solid + cutout casters below already rebuild on
    // 16 yd of camera drift and cost nothing on a standing frame. Skipping leaves the previous
    // environment proxy in place (the map is still rendered from it every frame), so the only
    // visible effect is a swinging lamp's or a fading doodad's shadow lagging by up to 1/rate s.
    //
    // Its own cvar rather than sharing `characterShadowRate`: this population barely moves, so it
    // tolerates a far lower rate than an animated crowd does, and a dial named for characters that
    // silently also governs the world is a trap for whoever reads this next.
    let env_due = lane
        .env_rate
        .due(time.elapsed_secs(), video.world_shadow_rate);
    if let Some(handle) = lane.env_mesh.clone().filter(|_| env_due) {
        if let Some(mesh) = meshes.get_mut(&handle) {
            let (mut positions, mut indices) = take_mesh_buffers(mesh);
            let (admitted, rejected) = collect_entity_geometry(
                &parts,
                &rigs,
                &palettes,
                false, // want_creatures (the character lane's job)
                true,  // want_environment
                frame.light_position,
                frame.entity_reach,
                frame.tall_reach,
                &mut positions,
                &mut indices,
                None,
            );
            let tris = (indices.len() / 3) as u32;
            restore_mesh_buffers(mesh, positions, indices);
            if shadow_trace() {
                let now = (tris, admitted, rejected);
                if lane.env_traced != now {
                    info!(
                        "shadow-trace: world-entity {} tris | env admitted {} / rejected {} (was {:?})",
                        tris, admitted, rejected, lane.env_traced
                    );
                    lane.env_traced = now;
                }
            }
        }
    }

    // (2) The cached static world (trees + buildings) + alpha-tested foliage — rebuilt on drift.
    let Some(gx) = gx.as_ref() else {
        return;
    };
    // Ensure the static SOLID caster.
    if lane.static_caster.is_none() {
        let mesh = meshes.add(empty_shadow_mesh());
        let entity = spawn_solid_caster(&mut commands, mesh.clone(), material);
        lane.static_caster = Some(entity);
        lane.static_mesh = Some(mesh);
        lane.static_rebuilt_at = None;
    }
    let due = lane
        .static_rebuilt_at
        .is_none_or(|at| at.distance(frame.light_position) > STATIC_REBUILD_STEP);
    if !due {
        return;
    }

    // Solid pass — the retained world's opaque geometry (minus the cutout foliage it skips).
    if let Some(handle) = lane.static_mesh.clone() {
        if let Some(mesh) = meshes.get_mut(&handle) {
            let (mut positions, mut indices) = take_mesh_buffers(mesh);
            gx.append_shadow_triangles(
                frame.light_position,
                frame.tall_reach,
                &mut positions,
                &mut indices,
            );
            let tris = (indices.len() / 3) as u32;
            restore_mesh_buffers(mesh, positions, indices);
            lane.static_rebuilt_at = Some(frame.light_position);
            if shadow_trace() {
                info!(
                    "shadow-trace: static rebuild — {} tris within {:.0}yd",
                    tris, frame.tall_reach
                );
            }
        }
    }

    // Alpha-tested foliage pass — the cutout leaf cards, grouped one caster per leaf texture.
    let buckets = gx.collect_cutout_shadow_triangles(frame.light_position, frame.tall_reach);
    let mut seen: HashSet<AssetId<Image>> = HashSet::new();
    let mut cutout_tris = 0u32;
    for bucket in buckets {
        seen.insert(bucket.texture_id);
        cutout_tris += (bucket.indices.len() / 3) as u32;
        if !lane.cutout.contains_key(&bucket.texture_id) {
            let mesh = meshes.add(empty_cutout_mesh());
            let material = cutout_materials.add(CutoutShadowCasterMaterial {
                leaf: bucket.texture.clone(),
            });
            let entity = commands
                .spawn((
                    Mesh3d(mesh.clone()),
                    MeshMaterial3d(material.clone()),
                    Transform::IDENTITY,
                    Visibility::Visible,
                    RenderLayers::layer(PLAYER_SHADOW_LAYER),
                    NoFrustumCulling,
                    NotShadowReceiver,
                    ShadowCaster,
                ))
                .id();
            lane.cutout.insert(
                bucket.texture_id,
                CutoutCaster {
                    entity,
                    mesh,
                    material,
                },
            );
        }
        let mesh_handle = lane.cutout[&bucket.texture_id].mesh.clone();
        if let Some(mesh) = meshes.get_mut(&mesh_handle) {
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, bucket.positions);
            mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, bucket.uvs);
            mesh.insert_indices(Indices::U32(bucket.indices));
        }
    }
    // Despawn casters whose leaf texture left the reach, so the set stays bounded across zones.
    let stale: Vec<AssetId<Image>> = lane
        .cutout
        .keys()
        .filter(|id| !seen.contains(*id))
        .copied()
        .collect();
    for id in stale {
        if let Some(caster) = lane.cutout.remove(&id) {
            commands.entity(caster.entity).despawn();
            meshes.remove(caster.mesh.id());
            cutout_materials.remove(caster.material.id());
        }
    }
    if shadow_trace() {
        info!(
            "shadow-trace: cutout rebuild — {} tris across {} leaf textures",
            cutout_tris,
            lane.cutout.len()
        );
    }
}

/// Tear down every world-lane caster (static solid, all foliage casters, the environment caster).
fn teardown(
    lane: &mut WorldLane,
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    cutout_materials: &mut Assets<CutoutShadowCasterMaterial>,
) {
    if let Some(entity) = lane.static_caster.take() {
        commands.entity(entity).despawn();
    }
    if let Some(handle) = lane.static_mesh.take() {
        meshes.remove(handle.id());
    }
    lane.static_rebuilt_at = None;
    for (_texture, caster) in lane.cutout.drain() {
        commands.entity(caster.entity).despawn();
        meshes.remove(caster.mesh.id());
        cutout_materials.remove(caster.material.id());
    }
    if let Some(entity) = lane.env_caster.take() {
        commands.entity(entity).despawn();
    }
    if let Some(handle) = lane.env_mesh.take() {
        meshes.remove(handle.id());
    }
    // The mesh the gate was pacing is gone — re-arm so a re-enable builds on its first frame.
    lane.env_rate.reset();
}

// MONKEY (followups): the refresh must not depend on volumetric fog.
#[cfg(test)]
mod streamed_refresh_tests {
    use super::*;

    #[test]
    fn shadow_cache_refresh_runs_without_volumetric_fog_and_is_quiet_when_unchanged() {
        let mut app = App::new();
        let mut video = VideoConfig::default();
        video.volumetric_fog = 0;
        video.world_shadows = false;
        app.insert_resource(video)
            .init_resource::<StaticGx>()
            .init_resource::<WorldLane>()
            .add_systems(Update, refresh_streamed_shadows);
        app.world_mut().clear_trackers();
        app.world_mut().run_schedule(Update);
        assert!(!app.world().resource_ref::<WorldLane>().is_changed());
        app.world_mut().resource_mut::<VideoConfig>().world_shadows = true;
        app.world_mut().run_schedule(Update);
        assert!(app.world().resource_ref::<WorldLane>().is_changed());
        app.world_mut().clear_trackers();
        app.world_mut().run_schedule(Update);
        assert!(!app.world().resource_ref::<WorldLane>().is_changed());
    }
}
