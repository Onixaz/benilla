//! The CHARACTER shadow lane — a thin plug-in on top of [`super::shadow_core`].
//!
//! It owns nothing but its own per-frame caster (players, NPCs, creatures, mounts). When
//! `characterShadows` is on it declares demand to the shared rig ([`ShadowDemand`]); while the rig
//! is live it collects the CREATURE entity parts into its caster using the rig's shared material and
//! the per-frame facts the rig publishes ([`ShadowFrame`]). It does not know [`super::world_shadow`]
//! exists — remove either lane and the other is untouched.

use bevy::ecs::entity::EntityHashSet;
use bevy::pbr::MeshMaterial3d;
use bevy::prelude::*;

use benilla_assets::materials::WowModelMaterial;
use benilla_world::billboard::BillboardCard;
use benilla_world::interact::PickMesh;
use benilla_world::model_render::{ModelPart, ShadowOccluder};
use benilla_world::rig_palette::{RigPalettes, RigPart, RigSkin};

use crate::shadow_core::{
    collect_entity_geometry, empty_shadow_mesh, restore_mesh_buffers, shadow_trace,
    spawn_solid_caster, take_mesh_buffers, RebuildRate, ShadowDemand, ShadowFrame, ShadowSet,
};
use crate::video::VideoConfig;

pub(crate) struct CharacterShadowPlugin;

impl Plugin for CharacterShadowPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CharacterLane>()
            .init_resource::<CharacterShadowReady>()
            .add_systems(Last, collect_character_shadows.in_set(ShadowSet::Lanes));
    }
}

/// MONKEY (moon shadows): roots/ancestors with triangles in the LAST completed caster build.
/// Blobs consume this in PostUpdate, before this frame's Last rebuild: a newly uploaded proxy
/// gets a render frame before its oval yields. Cadence skips retain the same geometry/verdict.
#[derive(Resource, Default)]
pub(crate) struct CharacterShadowReady(pub EntityHashSet);

/// The character lane's retained per-frame caster.
#[derive(Resource, Default)]
struct CharacterLane {
    caster: Option<Entity>,
    mesh: Option<Handle<Mesh>>,
    /// [`shadow_trace`]'s change detector: (tris, admitted, rejected).
    traced: (u32, u32, u32),
    /// MONKEY (sun shadow perf): the `characterShadowRate` cadence gate — see the rebuild comment
    /// in [`collect_character_shadows`].
    rate: RebuildRate,
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn collect_character_shadows(
    video: Res<VideoConfig>,
    time: Res<Time>,
    mut demand: ResMut<ShadowDemand>,
    frame: Res<ShadowFrame>,
    mut lane: ResMut<CharacterLane>,
    mut ready: ResMut<CharacterShadowReady>,
    parents: Query<&ChildOf>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
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
    let on = video.character_shadows;
    // Declare demand so the shared rig stays up while this lane is on (the rig reads it next frame).
    demand.0 = demand.0 || on;

    // Rig down or lane off → drop our caster so it stops casting.
    if !(frame.active && on) {
        ready.0.clear();
        if let Some(entity) = lane.caster.take() {
            commands.entity(entity).despawn();
        }
        if let Some(handle) = lane.mesh.take() {
            meshes.remove(handle.id());
        }
        // The mesh this gate was pacing is gone — re-arm so a re-enable builds on its first frame.
        lane.rate.reset();
        return;
    }
    // MONKEY (moon shadows): keep the cache but suspend CPU skin/re-upload on feature-off nights.
    if frame.suspended {
        ready.0.clear();
        lane.rate.reset();
        return;
    }
    let Some(material) = frame.material.clone() else {
        return;
    };

    // Animated rigs re-skin every frame and creatures spawn/despawn under lifecycles this lane
    // can't cheaply observe — so the caster is collected fresh, at a CAPPED cadence.
    if lane.caster.is_none() {
        let mesh = meshes.add(empty_shadow_mesh());
        let entity = spawn_solid_caster(&mut commands, mesh.clone(), material);
        lane.caster = Some(entity);
        lane.mesh = Some(mesh);
        // A brand-new caster is an EMPTY mesh; make sure this frame fills it.
        lane.rate.reset();
    }
    // MONKEY (sun shadow perf): the `characterShadowRate` gate. This rebuild is the expensive half
    // of the ~3 ms the character lane costs — it CPU-skins every admitted unit and then mutates the
    // `Mesh` asset, and a mutated mesh is re-extracted and re-uploaded (vertices AND indices) to the
    // GPU. Skipping it simply leaves the previous proxy in place: the shadow MAP is still rendered
    // from that proxy every frame, so nothing flickers or disappears — a moving unit's silhouette
    // merely lags by up to 1/rate second.
    //
    // Why a rate cap and not change detection: the obvious `Changed<GlobalTransform>` gate does not
    // work here. A unit's shadow geometry comes from the RIG PALETTE (`append_skinned` reads
    // `RigPalettes`, the per-frame skinning pose), not from the part's transform — so a standing NPC
    // whose idle animation is breathing has an unchanged `GlobalTransform` and changed VERTICES.
    // Gating on the transform would freeze exactly the poses this lane exists to draw. The rate cap
    // is the honest saving, and it is bounded work rather than a guess about content.
    if !lane
        .rate
        .due(time.elapsed_secs(), video.character_shadow_rate)
    {
        return;
    }
    if let Some(handle) = lane.mesh.clone() {
        if let Some(mesh) = meshes.get_mut(&handle) {
            let (mut positions, mut indices) = take_mesh_buffers(mesh);
            let mut built_parts = EntityHashSet::default();
            let (admitted, rejected) = collect_entity_geometry(
                &parts,
                &rigs,
                &palettes,
                true,  // want_creatures
                false, // want_environment (the world lane's job)
                frame.light_position,
                frame.entity_reach,
                frame.tall_reach,
                &mut positions,
                &mut indices,
                Some(&mut built_parts),
            );
            ready.0.clear();
            for part in built_parts {
                ready.0.insert(part);
                // The unit root owns its blob; mounts and equipment can be nested below it.
                ready.0.extend(parents.iter_ancestors(part));
            }
            let tris = (indices.len() / 3) as u32;
            restore_mesh_buffers(mesh, positions, indices);
            if shadow_trace() {
                let now = (tris, admitted, rejected);
                if lane.traced != now {
                    info!(
                        "shadow-trace: character {} tris | creatures admitted {} / rejected {} (was {:?})",
                        tris, admitted, rejected, lane.traced
                    );
                    lane.traced = now;
                }
            }
        }
    }
}
