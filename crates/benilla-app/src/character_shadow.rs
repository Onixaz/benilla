//! The CHARACTER shadow lane — a thin plug-in on top of [`super::shadow_core`].
//!
//! It owns nothing but its own casters (players, NPCs, creatures, mounts). When `characterShadows`
//! is on it declares demand to the shared rig ([`ShadowDemand`]); while the rig is live it keeps
//! one PROXY per admitted creature part on the private shadow layer, using the per-frame facts the
//! rig publishes ([`ShadowFrame`]). It does not know [`super::world_shadow`] exists — remove either
//! lane and the other is untouched.
//!
//! A proxy shares its part's own render mesh and is skinned on the GPU from the same palette rows
//! the part draws with ([`SkinnedShadowCasterMaterial`]), so the caster is the visible pose every
//! frame with no CPU skin and no mesh upload. The part itself never casts: `WowModelExt::specialize`
//! rewrites the vertex layout to forward-pass locations, which Bevy also applies to the prepass
//! descriptor, so a `WowModelMaterial` in the shadow pass gets a layout its prepass shader does not
//! read (the corruption `shadow_core` names). The proxy's own material owns both layouts instead.

use bevy::camera::visibility::{NoFrustumCulling, RenderLayers};
use bevy::ecs::entity::{EntityHashMap, EntityHashSet};
use bevy::light::NotShadowReceiver;
use bevy::mesh::MeshTag;
use bevy::pbr::{Material, MaterialPipeline, MaterialPlugin, MeshMaterial3d};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, Buffer, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::{ShaderDefVal, ShaderRef};

use benilla_assets::{ATTRIBUTE_WOW_JOINT_INDEX, ATTRIBUTE_WOW_JOINT_WEIGHT};
use benilla_world::billboard::BillboardCard;
use benilla_world::interact::PickMesh;
use benilla_world::lighting::SharedLightBuffer;
use benilla_world::mesh_tag::{rig_bits, rig_of};
use benilla_world::model_render::{ModelPart, ShadowOccluder};
use benilla_world::rig_palette::{
    palette_region_offset, rig_origin_region_offset, rig_table_region_offset, RigPalettes, RigPart,
    RigSkin,
};

use crate::shadow_core::{
    casts_realtime_shadow, shadow_trace, ShadowCaster, ShadowDemand, ShadowFrame, ShadowSet,
    PLAYER_SHADOW_LAYER,
};
use crate::video::VideoConfig;

pub(crate) struct CharacterShadowPlugin;

impl Plugin for CharacterShadowPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<SkinnedShadowCasterMaterial>::default())
            .init_resource::<CharacterLane>()
            .init_resource::<CharacterShadowReady>()
            .add_systems(Last, collect_character_shadows.in_set(ShadowSet::Lanes));
    }
}

/// MONKEY (moon shadows): roots/ancestors whose parts cast in the LAST lane pass. Blobs consume
/// this in PostUpdate, before this frame's Last pass: a newly shown proxy gets a render frame
/// before its oval yields.
#[derive(Resource, Default)]
pub(crate) struct CharacterShadowReady(pub EntityHashSet);

/// The character proxy's material: invisible in the forward pass, and in the shadow pass a
/// positions-only vertex stage that skins a unit part's own mesh from the shared light buffer's
/// palette rows, exactly as `wow_model.wgsl` does.
#[derive(Asset, AsBindGroup, TypePath, Clone)]
// MONKEY (gpu character shadows): crate-visible for the pipe_warm menagerie.
pub(crate) struct SkinnedShadowCasterMaterial {
    /// Word offsets into `light`: x = the rig slot table, y = the rig origins, z = the palette rows.
    #[uniform(0)]
    pub(crate) regions: UVec4,
    /// The shared global light buffer (`SharedLightBuffer`), the one the units draw from.
    #[storage(1, read_only, buffer, visibility(vertex))]
    pub(crate) light: Buffer,
}

impl SkinnedShadowCasterMaterial {
    /// The material over `light`, addressed at `rig_palette`'s published region offsets.
    pub(crate) fn new(light: Buffer) -> Self {
        let words = |bytes: u64| (bytes / 4) as u32;
        Self {
            regions: UVec4::new(
                words(rig_table_region_offset()),
                words(rig_origin_region_offset()),
                words(palette_region_offset()),
                0,
            ),
            light,
        }
    }
}

impl Material for SkinnedShadowCasterMaterial {
    fn vertex_shader() -> ShaderRef {
        "embedded://benilla_app/shaders/shadow_caster_skinned.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://benilla_app/shaders/shadow_caster_skinned.wgsl".into()
    }

    fn prepass_vertex_shader() -> ShaderRef {
        "embedded://benilla_app/shaders/shadow_caster_skinned_prepass.wgsl".into()
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &bevy::mesh::MeshVertexBufferLayoutRef,
        _key: bevy::pbr::MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Two-sided like the solid caster: WoW batches are frequently mixed-winding.
        descriptor.primitive.cull_mode = None;
        let prepass =
            descriptor.vertex.shader_defs.iter().any(
                |def| matches!(def, ShaderDefVal::Bool(name, true) if name == "PREPASS_PIPELINE"),
            );
        if !prepass {
            // The forward vertex stage reads no attribute; Bevy's layout is left as built.
            return Ok(());
        }
        // The shadow pass reads positions (and the palette joints) only. The UV and colour defs
        // Bevy derived from the mesh would declare outputs this vertex stage does not write.
        let mut attrs = vec![Mesh::ATTRIBUTE_POSITION.at_shader_location(0)];
        let skinned = layout.0.contains(ATTRIBUTE_WOW_JOINT_INDEX);
        if skinned {
            attrs.push(ATTRIBUTE_WOW_JOINT_INDEX.at_shader_location(10));
            attrs.push(ATTRIBUTE_WOW_JOINT_WEIGHT.at_shader_location(11));
        }
        descriptor.vertex.buffers = vec![layout.0.get_layout(&attrs)?];
        let unread = |def: &ShaderDefVal| {
            matches!(def, ShaderDefVal::Bool(name, _)
                if matches!(name.as_str(), "VERTEX_UVS" | "VERTEX_UVS_A" | "VERTEX_UVS_B" | "VERTEX_COLORS"))
        };
        descriptor.vertex.shader_defs.retain(|def| !unread(def));
        if let Some(fragment) = descriptor.fragment.as_mut() {
            fragment.shader_defs.retain(|def| !unread(def));
        }
        if skinned {
            descriptor.vertex.shader_defs.push("WOW_RIG_SKIN".into());
        }
        Ok(())
    }
}

/// Marks a character-lane proxy, so its writes never alias a unit part's components.
#[derive(Component)]
struct CharacterShadowProxy;

/// One part's proxy and the state last written to it, so a frame writes only what changed.
struct Proxy {
    entity: Entity,
    /// The lane pass that last saw the part; an older stamp means the part is gone.
    seen: u32,
    /// The part transform last written. Compared here, not against the proxy's own
    /// `GlobalTransform`, which propagation rewrites from the decomposed `Transform`.
    global: GlobalTransform,
}

/// The character lane's retained casters.
#[derive(Resource, Default)]
struct CharacterLane {
    material: Option<Handle<SkinnedShadowCasterMaterial>>,
    /// Creature part → its proxy.
    proxies: EntityHashMap<Proxy>,
    pass: u32,
    /// [`shadow_trace`]'s change detector: (proxies shown, admitted, rejected).
    traced: (u32, u32, u32),
}

impl CharacterLane {
    fn despawn_all(&mut self, commands: &mut Commands) {
        for (_, proxy) in self.proxies.drain() {
            commands.entity(proxy.entity).despawn();
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn collect_character_shadows(
    video: Res<VideoConfig>,
    mut demand: ResMut<ShadowDemand>,
    frame: Res<ShadowFrame>,
    mut lane: ResMut<CharacterLane>,
    mut ready: ResMut<CharacterShadowReady>,
    parents: Query<&ChildOf>,
    mut commands: Commands,
    mut materials: ResMut<Assets<SkinnedShadowCasterMaterial>>,
    light: Option<Res<SharedLightBuffer>>,
    parts: Query<
        (
            Entity,
            &ModelPart,
            &ShadowOccluder,
            &Mesh3d,
            Option<&MeshTag>,
            Option<&GlobalTransform>,
            Option<&RigPart>,
        ),
        (With<PickMesh>, Without<BillboardCard>),
    >,
    mut proxies: Query<
        (
            &mut Mesh3d,
            &mut MeshTag,
            &mut Visibility,
            &mut Transform,
            &mut GlobalTransform,
        ),
        (With<CharacterShadowProxy>, Without<PickMesh>),
    >,
    rigs: Query<&RigSkin>,
    palettes: Res<RigPalettes>,
) {
    let on = video.character_shadows;
    // Declare demand so the shared rig stays up while this lane is on (the rig reads it next frame).
    demand.0 = demand.0 || on;

    // Rig down or lane off → drop every proxy so nothing casts.
    if !(frame.active && on) {
        ready.0.clear();
        lane.despawn_all(&mut commands);
        if let Some(handle) = lane.material.take() {
            materials.remove(handle.id());
        }
        return;
    }
    // MONKEY (moon shadows): keep the proxies but spend no sync work on feature-off nights.
    if frame.suspended {
        ready.0.clear();
        return;
    }
    if lane.material.is_none() {
        let Some(light) = light else { return };
        lane.material = Some(materials.add(SkinnedShadowCasterMaterial::new(light.0.clone())));
    }
    let Some(material) = lane.material.clone() else {
        return;
    };

    // Every frame, but bounded: the skin runs on the GPU, so this walk is a reach test and a few
    // compares a part; components are written only when they differ.
    lane.pass = lane.pass.wrapping_add(1);
    let pass = lane.pass;
    let reach_sq = frame.entity_reach * frame.entity_reach;
    let (mut shown, mut admitted, mut rejected) = (0u32, 0u32, 0u32);
    ready.0.clear();
    for (entity, part, occluder, mesh, tag, global, rig_part) in &parts {
        if !casts_realtime_shadow(part.kind, part.blend, true, false) {
            continue;
        }
        // The content half of the visibility law, not `InheritedVisibility` (see `ShadowOccluder`).
        let anchor = global.map(GlobalTransform::translation).or_else(|| {
            rig_part
                .and_then(|rig_part| rigs.get(rig_part.0).ok())
                .and_then(|rig| palettes.slot_origin(rig.slot))
        });
        let in_reach = anchor.is_some_and(|a| a.distance_squared(frame.light_position) <= reach_sq);
        if !occluder.0 {
            rejected += 1;
        } else if in_reach {
            admitted += 1;
        }
        // A skinned part whose palette is gone (rig torn down) would read zeroed rows: hold it.
        let posed = rig_part.is_none_or(|rig_part| rigs.contains(rig_part.0));
        let show = occluder.0 && in_reach && posed;
        // The rig field only: the shadow pass reads the slot, not the part's fade or lighting bits.
        let want_tag = MeshTag(rig_bits(tag.map_or(0, |t| rig_of(t.0))));
        let want_vis = if show {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        // A skinned part's world comes from the palette, so its proxy never needs a transform
        // write as the unit moves; a rigid one's comes from its own transform.
        let want_global = match rig_part {
            Some(_) => GlobalTransform::IDENTITY,
            None => global.copied().unwrap_or_default(),
        };

        match lane.proxies.get_mut(&entity) {
            Some(proxy) => {
                proxy.seen = pass;
                if let Ok((mut p_mesh, mut p_tag, mut p_vis, mut p_tf, mut p_global)) =
                    proxies.get_mut(proxy.entity)
                {
                    if p_mesh.0 != mesh.0 {
                        p_mesh.0 = mesh.0.clone();
                    }
                    p_tag.set_if_neq(want_tag);
                    p_vis.set_if_neq(want_vis);
                    // Both halves: this runs after propagation, and the draw extracts the global.
                    if proxy.global != want_global {
                        proxy.global = want_global;
                        *p_global = want_global;
                        *p_tf = want_global.compute_transform();
                    }
                }
            }
            None => {
                let proxy = commands
                    .spawn((
                        Mesh3d(mesh.0.clone()),
                        MeshMaterial3d(material.clone()),
                        want_tag,
                        want_global.compute_transform(),
                        want_global,
                        want_vis,
                        RenderLayers::layer(PLAYER_SHADOW_LAYER),
                        // The rig's palette pose, not the bind-pose box, is what casts.
                        NoFrustumCulling,
                        NotShadowReceiver,
                        ShadowCaster,
                        CharacterShadowProxy,
                    ))
                    .id();
                lane.proxies.insert(
                    entity,
                    Proxy {
                        entity: proxy,
                        seen: pass,
                        global: want_global,
                    },
                );
            }
        }
        if show {
            shown += 1;
            ready.0.insert(entity);
            // The unit root owns its blob; mounts and equipment can be nested below it.
            ready.0.extend(parents.iter_ancestors(entity));
        }
    }
    // A part this pass did not see has despawned: its proxy goes with it.
    lane.proxies.retain(|_, proxy| {
        let live = proxy.seen == pass;
        if !live {
            commands.entity(proxy.entity).despawn();
        }
        live
    });

    if shadow_trace() {
        let now = (shown, admitted, rejected);
        if lane.traced != now {
            info!(
                "shadow-trace: character {} proxies | creatures admitted {} / rejected {} (was {:?})",
                shown, admitted, rejected, lane.traced
            );
            lane.traced = now;
        }
    }
}

#[cfg(test)]
mod tests {
    use benilla_world::rig_palette::{
        palette_region_offset, rig_origin_region_offset, rig_table_region_offset,
    };

    /// The proxy shader reads the light buffer as words at these offsets, and reads the origin and
    /// palette regions as `vec4`s: each must be 16-byte aligned, as `wow_model.wgsl`'s struct lays
    /// them out, or the byte → word division truncates onto another region's data.
    #[test]
    fn the_proxy_addresses_the_palette_regions_on_vec4_boundaries() {
        for bytes in [
            rig_table_region_offset(),
            rig_origin_region_offset(),
            palette_region_offset(),
        ] {
            assert_eq!(bytes % 16, 0, "region at byte {bytes} is not vec4-aligned");
        }
        assert!(rig_table_region_offset() < rig_origin_region_offset());
        assert!(rig_origin_region_offset() < palette_region_offset());
    }
}
