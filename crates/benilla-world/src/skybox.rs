//! The skybox: an authored sky M2 standing in for the `Light.dbc` gradient dome, drawn
//! camera-anchored at the far depth ([`crate::sky_order`]) through the ordinary M2 material lane
//! ([`M2BatchMaterials::skybox`]).
//!
//! Two slots feed it. A WMO root names a model in MOSB, and the reference draws it when any group
//! the portal flood reaches carries flag `0x40000` (`0x6b42e0` inside the flood `0x6b41c0`,
//! published to `[0xca8080]`, read at `0x681282`). The ghost sky (`LightSkybox.dbc`) fills the DBC
//! slot, whose weight 1.0 skips the WMO slot outright (`0x6d4ac1`–`0x6d4acc`), so it is taken
//! first.
//!
//! The WMO slot's weight is the camera-in-WMO interior crossfade `[0xce9bdc]`
//! (`0x6d4810(0, [0xca8080], [0xce9bdc])`), the number the MFOG fog lerp rides. Above
//! `[0x808aac]` = 0.99 a slot replaces the whole celestial pass (`0x6d4a3b`): stars, sun disc,
//! both moons, gradient band and cloud dome; below it they draw under the sky. The glare quads
//! draw on their own path (`0x483740` → `0x6d48c0` → `0x7e57e0`), and the fog, ambient and
//! diffuse are untouched.
//!
//! MONKEY (skybox): a third slot, the living player's zone skybox, behind the cvar
//! `zoneSkyboxes` ([`ZoneSkyboxes`]). The 1.12 engine never draws a `LightParams` skybox for the
//! living; the modern client does, weighted by the Light sphere falloff
//! ([`benilla_formats::LightCatalog::zone_skyboxes`]). The backdrop is a weighted list
//! ([`CameraSkybox`]): ghost > WMO > zone, the WMO sky crossfading the zone list by the modern
//! collector's rule. `LightSkybox.dbc` flags from the extended table: `0x1` plays sequence 0 over
//! the game day, `0x2` keeps the celestial pass, `0x4` draws a fog-colour cone over the horizon;
//! a celestial model draws as a layer under its main model. Every batch animates
//! ([`crate::skybox_anim`]): bones, texture transforms, colour and alpha tracks.
//!
//! MONKEY (reviewfix): those M2-fidelity repairs deliberately also apply when `zoneSkyboxes` is
//! off. The reference WMO and ghost slots animate their authored colour/alpha and texture tracks,
//! use non-white M2 colours, and draw batches in authored order; deterministic captures pose those
//! same tracks at their pinned time rather than substituting the bind pose. The batch-order layout
//! leaves room for a celestial under-layer and the final fog cone. The cvar gates the added
//! zone slot, not fixes to the two 1.12 slots.

use std::collections::HashMap;
use std::sync::Arc;

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::math::Affine3A;
use bevy::mesh::{Indices, MeshTag, PrimitiveTopology};
use bevy::prelude::*;

use crate::model_render::M2BatchMaterials;
use crate::skybox_anim::{SkyMatLane, SkyRig};
use crate::view::WorldCamera;
use benilla_assets::coords::{bevy_to_wow, wow_to_bevy};
use benilla_assets::materials::WowModelMaterial;
use benilla_assets::WmoModel;
use benilla_assets::{LockRecover, WorldAssets};
use benilla_formats::{SKYBOX_FOG_BLEND, SKYBOX_FULL_DAY, SKYBOX_KEEP_CELESTIAL};

/// The MOGP/MOGI group flag asking for the root's MOSB sky; the loader keeps the MOGP copy in
/// `WmoGroupNav::flags`.
const SHOW_SKYBOX: u32 = 0x40000;

/// MONKEY (reviewfix): the fog cone owns the skybox band's final distinct rung. Orders above this
/// clamp onto it in [`crate::model_render::skybox_sort_bias`], so model batches stop one rung below.
const FOG_CONE_ORDER: u16 = 58;

/// MONKEY (skybox): cvar `zoneSkyboxes` (0/1, default 0): draw the living player's zone skybox
/// from `LightParams`. Off leaves the reference's two slots only.
#[derive(Resource, Default, Clone, Copy, PartialEq, Eq, Debug)]
pub struct ZoneSkyboxes(pub bool);

/// One skybox model the frame asks for, at its weight.
#[derive(Clone, Debug, PartialEq)]
pub struct SkyboxLayer {
    /// The model path, as the slot named it.
    pub path: String,
    /// The slot weight, `[0, 1]`; 0 fills the slot without drawing (`0x6d4afe`).
    pub weight: f32,
    /// `LightSkybox.dbc` flags, 0 without a row or in the 1.12 table.
    pub flags: u32,
    /// The celestial layer of another layer's row, drawn under it.
    pub celestial: bool,
}

/// The skybox models this frame asks for; empty leaves the [`crate::sky`] gradient dome.
#[derive(Resource, Default, PartialEq)]
pub struct CameraSkybox(pub Vec<SkyboxLayer>);

impl CameraSkybox {
    /// The heaviest main layer, the one a readout names.
    pub fn primary(&self) -> Option<&SkyboxLayer> {
        self.0.iter().filter(|l| !l.celestial).fold(
            None,
            |best: Option<&SkyboxLayer>, l| match best {
                Some(b) if b.weight >= l.weight => Some(b),
                _ => Some(l),
            },
        )
    }

    /// MONKEY (leftovers): the heaviest main layer's weight among those carrying `flag`, 0 if none.
    pub fn flag_weight(&self, flag: u32) -> f32 {
        self.0
            .iter()
            .filter(|l| !l.celestial && l.flags & flag != 0)
            .map(|l| l.weight)
            .fold(0.0, f32::max)
    }

    fn layer(&self, path: &str) -> Option<&SkyboxLayer> {
        self.0.iter().find(|l| l.path == path)
    }
}

/// The skybox slot's weight this frame: the interior crossfade [`crate::lighting::WmoCrossfade`]
/// (±0.25/s, `[0x8115b0]`) for a WMO sky, 1.0 for the ghost sky (`0x6d2260`), and 0 when none
/// resolves. MONKEY (skybox): the heaviest layer that does not keep the celestial pass (flag
/// `0x2`), so a combining zone sky never stands the procedural sky down.
#[derive(Resource, Default, PartialEq)]
pub struct SkyboxWeight(pub f32);

impl SkyboxWeight {
    /// Whether the skybox replaces the whole celestial pass: weight above `[0x808aac]` = 0.99
    /// (`0x6d49e8`/`0x6d4a1f`); below it the sky blends over the six elements.
    pub fn replaces_celestial(&self) -> bool {
        self.0 > 0.99
    }
}

/// One batch of a built skybox model.
#[derive(Component)]
struct SkyboxPart {
    /// The skybox model path this batch belongs to.
    path: String,
    /// The batch's authored-blend material, drawn at weight 1.0.
    steady: Handle<WowModelMaterial>,
    /// The blend-promotion twin for `0 < weight < 1` (`0x811fe0`); `steady` itself when the batch
    /// already blends.
    fade_blend: Handle<WowModelMaterial>,
    /// MONKEY (skybox): the batch's colour-alpha × transparency loops (`0x707680`).
    alpha: Option<benilla_formats::AlphaAnim>,
    /// MONKEY (skybox): the last sampled alpha factor.
    anim_alpha: f32,
    /// MONKEY (skybox): the batch's texture-transform and colour rows.
    lane: SkyMatLane,
    /// MONKEY (skybox): how the rig moves the batch.
    pose: PartPose,
    /// MONKEY (visualfix): the model-clock pose generation this skinned batch was last uploaded
    /// at; an unchanged pose is not re-skinned or re-uploaded.
    skinned_at: Option<u64>,
}

/// MONKEY (skybox): how a batch follows its model's rig.
enum PartPose {
    /// No moving bone under the batch.
    Static,
    /// Wholly weighted to one bone: the bone's matrix is the batch's transform.
    Rigid(u16),
    /// Spread over moving bones: skinned on the CPU into its mesh.
    Skinned {
        mesh: Handle<Mesh>,
        base: Vec<Vec3>,
        joints: Vec<[u16; 4]>,
        weights: Vec<[f32; 4]>,
    },
}

/// MONKEY (skybox): the batch's model-space pose this frame, composed with the eye in
/// [`follow_camera`].
#[derive(Component, Default)]
struct SkyboxLocal(Affine3A);

/// Skybox paths built this session в†’ their batch counts. Failed loads use zero, so a failure is not
/// retried per frame; successful celestial counts set the main model's dynamic order base.
#[derive(Resource, Default)]
struct BuiltSkyboxes(HashMap<String, u16>);

/// MONKEY (skybox): each built model's rig, by path.
#[derive(Resource, Default)]
struct SkyRigs(HashMap<String, Arc<SkyRig>>);

/// MONKEY (skybox): the flag `0x4` fog cone: its entity and its tint row.
#[derive(Resource, Default)]
struct FogCone {
    entity: Option<Entity>,
    slot: Option<u16>,
}

/// MONKEY (skybox): marks the fog cone's entity.
#[derive(Component)]
struct FogConePart;

/// [`CameraSkybox`] is settled after this set; the dome's gate runs after it, so the two backdrops
/// agree within a frame.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SkyboxResolve;

/// Resolves the wanted skybox, builds it on first need, shows it and pins it to the camera.
pub(crate) struct SkyboxPlugin;

impl Plugin for SkyboxPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CameraSkybox>()
            .init_resource::<SkyboxWeight>()
            .init_resource::<BuiltSkyboxes>()
            .init_resource::<ZoneSkyboxes>()
            .init_resource::<SkyRigs>()
            .init_resource::<FogCone>()
            .add_systems(
                Update,
                (
                    resolve_camera_skybox,
                    build_skybox,
                    animate_skyboxes,
                    apply_skybox_visibility,
                )
                    .chain()
                    // After the PVS pass, whose flood this reads the same frame.
                    .after(crate::wmo_portal::WmoPvsSet)
                    // After the lighting resolve, so the weight is this frame's crossfade, as the
                    // fog's is.
                    .after(crate::lighting::LightingResolveSet)
                    // MONKEY (integration): a `WowLighting` reader joins the consume set.
                    .in_set(crate::lighting::LightingConsumeSet)
                    .in_set(SkyboxResolve),
            )
            // Camera-anchored placement runs after propagation, off this frame's camera pose.
            .add_systems(
                PostUpdate,
                follow_camera.in_set(crate::billboard::BillboardPlace),
            );
    }
}

/// MONKEY (skybox): a path as `model_path` spells it, for comparing a MOSB name with a DBC one.
fn norm(path: &str) -> String {
    let lower = path.to_ascii_lowercase();
    match lower
        .strip_suffix(".mdx")
        .or_else(|| lower.strip_suffix(".mdl"))
    {
        Some(stem) => format!("{stem}.m2"),
        None => lower,
    }
}

/// MONKEY (visualfix): `norm(a) == norm(b)` without allocating (the per-frame resolver's test).
fn norm_eq(a: &str, b: &str) -> bool {
    fn split(p: &str) -> (&str, bool) {
        for ext in [".mdx", ".mdl", ".m2"] {
            let cut = p.len().saturating_sub(ext.len());
            if let (Some(stem), Some(tail)) = (p.get(..cut), p.get(cut..)) {
                if tail.eq_ignore_ascii_case(ext) {
                    return (stem, true);
                }
            }
        }
        (p, false)
    }
    let ((sa, ma), (sb, mb)) = (split(a), split(b));
    ma == mb && sa.eq_ignore_ascii_case(sb)
}

/// MONKEY (reviewfix): celestial batches occupy the low rungs and a main model starts immediately
/// after the largest active celestial model. Oversized art compresses onto the last model rung but
/// can never collide with the fog cone.
fn skybox_batch_order(celestial: bool, celestial_batches: u16, batch: usize) -> u16 {
    let base = if celestial { 0 } else { celestial_batches };
    base.saturating_add(u16::try_from(batch.saturating_add(1)).unwrap_or(u16::MAX))
        .min(FOG_CONE_ORDER - 1)
}

/// MONKEY (skybox): the modern collector's step on layers (`addSkyBox`): the same model keeps the
/// larger weight and every other layer is scaled by `1 − weight`.
fn collect_layer(layers: &mut Vec<SkyboxLayer>, path: &str, weight: f32, flags: u32) {
    let at = match layers.iter().position(|l| norm_eq(&l.path, path)) {
        Some(i) => {
            layers[i].weight = layers[i].weight.max(weight);
            i
        }
        None => {
            layers.push(SkyboxLayer {
                path: path.to_owned(),
                weight,
                flags,
                celestial: false,
            });
            layers.len() - 1
        }
    };
    let w = layers[at].weight;
    for (i, l) in layers.iter_mut().enumerate() {
        if i != at {
            l.weight *= 1.0 - w;
        }
    }
}

/// Resolve the wanted skybox and its weight. A WMO sky needs a group of the placement's flood PVS,
/// not the camera's own group, to carry [`SHOW_SKYBOX`] (`0x6b42e0`, `ebx` the group visited) and
/// its root to name a MOSB; the weight rides the down-ray claim's crossfade, a separate resolver.
#[allow(clippy::too_many_arguments)]
fn resolve_camera_skybox(
    instances: Query<&crate::wmo_portal::WmoPortalInstance>,
    wmos: Res<Assets<WmoModel>>,
    sampler: Option<Res<crate::lighting::LightSampler>>,
    viewer: Res<crate::view::Viewer>,
    current_map: Option<Res<crate::world_map::CurrentMap>>,
    cam: Query<&GlobalTransform, With<WorldCamera>>,
    crossfade: Res<crate::lighting::WmoCrossfade>,
    zone: Res<ZoneSkyboxes>,
    weather: Option<Res<crate::weather::WeatherState>>,
    eye_liquid: crate::liquid::EyeLiquid,
    mut want: ResMut<CameraSkybox>,
    mut weight: ResMut<SkyboxWeight>,
) {
    let pos = cam.single().ok().map(|t| bevy_to_wow(t.translation()));
    let map = current_map.as_ref().map_or(0, |m| m.0);
    let catalog = sampler.as_ref().map(|s| &s.0);
    let flags_of = |path: &str| {
        catalog
            .and_then(|c| c.skybox_def_by_path(&norm(path)))
            .map_or(0, |d| d.flags)
    };
    let mut layers: Vec<SkyboxLayer> = Vec::new();

    // The ghost sky first, as its slot skips the WMO one. Resolved from the atmosphere's map and
    // camera position, off the same ghost flag (`PLAYER_FLAGS` 0x10), so the two switch together.
    let ghost_sky = viewer
        .ghost
        .then(|| catalog.zip(pos).and_then(|(c, p)| c.ghost_skybox(map, p)))
        .flatten();
    if let Some(sky) = ghost_sky {
        // The DBC slot's weight is 1.0 whenever filled (`0x6d26cb`/`0x6d26d0`): it pops in.
        layers.push(SkyboxLayer {
            path: sky.to_owned(),
            weight: 1.0,
            flags: flags_of(sky),
            celestial: false,
        });
    } else {
        // MONKEY (skybox): the zone list, under the WMO slot. Hidden submerged, as the celestial
        // pass is; a storm lerps the storm slot's list over the clear one, as the atmosphere does.
        if zone.0 && !eye_liquid.submersion().any() {
            if let (Some(c), Some(p)) = (catalog, pos) {
                let storm = weather
                    .as_ref()
                    .map_or(0.0, |w| crate::weather::storm_blend(w.sky_density));
                let mut acc: Vec<(u32, f32)> = c
                    .zone_skyboxes(map, p, false, benilla_formats::Submersion::Dry)
                    .into_iter()
                    .map(|e| (e.id, e.weight * (1.0 - storm)))
                    .collect();
                if storm > 0.0 {
                    for e in c.zone_skyboxes(map, p, true, benilla_formats::Submersion::Dry) {
                        match acc.iter_mut().find(|a| a.0 == e.id) {
                            Some(a) => a.1 += e.weight * storm,
                            None => acc.push((e.id, e.weight * storm)),
                        }
                    }
                }
                for (id, w) in acc {
                    let Some(def) = c.skybox_def(id) else {
                        continue;
                    };
                    if w <= 0.0 {
                        continue;
                    }
                    match layers.iter_mut().find(|l| norm_eq(&l.path, &def.path)) {
                        Some(l) => l.weight = l.weight.max(w),
                        None => layers.push(SkyboxLayer {
                            path: def.path.clone(),
                            weight: w,
                            flags: def.flags,
                            celestial: false,
                        }),
                    }
                }
            }
        }
        // `min()`, not the first match: query order is unstable across frames, and two
        // overlapping Caverns of Time shells both qualify.
        let resolved = instances
            .iter()
            .filter_map(|inst| {
                let model = wmos.get(&inst.handle)?;
                // The MOSB test first: 810 of the game's 815 WMO roots name no skybox.
                let sky = model.skybox.as_deref()?;
                model
                    .group_nav
                    .iter()
                    .enumerate()
                    .any(|(i, nav)| {
                        // Fail closed, unlike the cull: a lookup miss here would paint a sky over
                        // the whole world.
                        nav.flags & SHOW_SKYBOX != 0
                            && inst.visible.get(i).copied().unwrap_or(false)
                    })
                    .then_some(sky)
            })
            .min()
            // MONKEY (visualfix): one owned copy for the winner, not one per candidate.
            .map(str::to_owned);
        // The WMO slot's weight is the interior crossfade `[0xce9bdc]`; a name seen through a
        // doorway at weight 0 fills the slot, which `0x6d4afe` declines to draw.
        if let Some(sky) = resolved {
            let t = crossfade.t();
            let flags = flags_of(&sky);
            if t > 0.0 {
                collect_layer(&mut layers, &sky, t, flags);
            } else if !layers.iter().any(|l| norm_eq(&l.path, &sky)) {
                layers.push(SkyboxLayer {
                    path: sky,
                    weight: 0.0,
                    flags,
                    celestial: false,
                });
            }
        }
    }
    // MONKEY (skybox): each row's celestial model, a layer at its main model's weight and flags.
    let celestial: Vec<SkyboxLayer> = layers
        .iter()
        .filter_map(|l| {
            let path = catalog?
                .skybox_def_by_path(&norm(&l.path))?
                .celestial
                .clone()?;
            Some(SkyboxLayer {
                path,
                weight: l.weight,
                flags: l.flags,
                celestial: true,
            })
        })
        .collect();
    layers.extend(celestial);
    if want.0 != layers {
        want.0 = layers;
    }
    let stand_down = want
        .0
        .iter()
        .filter(|l| l.flags & SKYBOX_KEEP_CELESTIAL == 0)
        .map(|l| l.weight)
        .fold(0.0, f32::max);
    weight.set_if_neq(SkyboxWeight(stand_down));
}

/// Build the wanted skyboxes on first request; the models are small and few, so they stay built.
#[allow(clippy::too_many_arguments)]
fn build_skybox(
    mut commands: Commands,
    want: Res<CameraSkybox>,
    mut built: ResMut<BuiltSkyboxes>,
    mut rigs: ResMut<SkyRigs>,
    mut cone: ResMut<FogCone>,
    world_assets: Option<ResMut<WorldAssets>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut table: ResMut<crate::mat_anim_table::MatAnimTable>,
    mut mats: M2BatchMaterials,
) {
    if want.0.is_empty() {
        return;
    }
    // Before the shared light buffer exists: retry, without latching `built`.
    if !mats.ready() {
        return;
    }
    let Some(mut world_assets) = world_assets else {
        return; // assetless run: the gradient dome stays the backdrop
    };
    if cone.entity.is_none() && want.0.iter().any(|l| l.flags & SKYBOX_FOG_BLEND != 0) {
        build_fog_cone(&mut commands, &mut cone, &mut meshes, &mut table, &mut mats);
    }
    // MONKEY (reviewfix): build celestial models first so main bases derive from their real batch
    // counts instead of a fixed split that aliases sufficiently large models.
    let mut pending: Vec<_> = want.0.iter().collect();
    pending.sort_by_key(|layer| !layer.celestial);
    for layer in pending {
        let path = layer.path.as_str();
        if built.0.contains_key(path) {
            continue;
        }
        // MONKEY (skybox): the rig, off the same bytes the batches come from.
        let bytes = world_assets
            .chain
            .lock_recover()
            .read_file(&norm(path))
            .ok();
        let rig = Arc::new(bytes.as_deref().map(SkyRig::from_bytes).unwrap_or_default());
        let subs = benilla_formats::load_m2_mesh(&mut world_assets.chain.lock_recover(), path);
        let subs = match subs {
            Ok(subs) if !subs.is_empty() => subs,
            Ok(_) => {
                warn!("skybox '{path}' has no render batches — keeping the gradient dome");
                built.0.insert(path.to_string(), 0);
                continue;
            }
            Err(e) => {
                warn!("skybox '{path}' failed to load, keeping the gradient dome: {e:#}");
                built.0.insert(path.to_string(), 0);
                continue;
            }
        };
        let celestial_batches = want
            .0
            .iter()
            .filter(|layer| layer.celestial)
            .filter_map(|layer| built.0.get(&layer.path).copied())
            .max()
            .unwrap_or(0);
        for (i, sub) in subs.iter().enumerate() {
            let mut mesh = Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            );
            let positions: Vec<Vec3> = sub.positions.iter().map(|p| wow_to_bevy(*p)).collect();
            mesh.insert_attribute(
                Mesh::ATTRIBUTE_POSITION,
                positions.iter().map(|p| p.to_array()).collect::<Vec<_>>(),
            );
            // Unread by the unlit sky, but the shared shader's vertex layout needs it.
            mesh.insert_attribute(
                Mesh::ATTRIBUTE_NORMAL,
                sub.normals
                    .iter()
                    .map(|n| wow_to_bevy(*n).to_array())
                    .collect::<Vec<_>>(),
            );
            mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, sub.uvs.clone());
            // MONKEY (skybox): the constant M2Color, when the batch has one that is not white.
            if sub.vertex_colors.len() == sub.positions.len()
                && sub
                    .vertex_colors
                    .iter()
                    .any(|c| c.iter().any(|v| (v - 1.0).abs() > 1e-3))
            {
                mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, sub.vertex_colors.clone());
            }
            // MONKEY (skybox): stage 1 reads UV set B.
            if let Some(st) = sub
                .stage1
                .as_ref()
                .filter(|s| s.uvs.len() == sub.positions.len())
            {
                mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, st.uvs.clone());
            }
            mesh.insert_indices(Indices::U32(sub.indices.clone()));
            // The batch's authored address mode: Caverns of Time's belts wrap their UVs.
            let texture = sub
                .texture
                .as_deref()
                .and_then(|t| world_assets.texture(t, (sub.wrap_x, sub.wrap_y), &mut images));
            // MONKEY (skybox): the texture transform and colour loops as material identities.
            let uv = sub.uv_anim.clone().map(Arc::new);
            let tint = sub.rgb_anim.clone().map(Arc::new);
            // The authored batch order after the layer's base (0 is unordered): every batch shares
            // one sort distance.
            let order = skybox_batch_order(layer.celestial, celestial_batches, i);
            let Some(mut pair) = mats.skybox(sub, texture, order, uv.as_ref(), tint.as_ref())
            else {
                return; // light buffer vanished mid-build; `built` is unlatched, so we retry
            };
            // MONKEY (skybox): a two-texture batch takes its own copies with stage 1 bound, so
            // the deduped one-texture materials stay as they were.
            if let Some(st) = sub
                .stage1
                .as_ref()
                .filter(|s| s.uvs.len() == sub.positions.len())
            {
                let tex1 = st
                    .texture
                    .as_deref()
                    .and_then(|t| world_assets.texture(t, (st.wrap_x, st.wrap_y), &mut images));
                let mode = if st.mod2x { 2.0 } else { 1.0 };
                let mut with_stage1 = |h: &Handle<WowModelMaterial>| {
                    let mut m = crate::model_render::lazy::with_material_mut(
                        mats.materials(),
                        h.id(),
                        |m| m.clone(),
                    )?;
                    m.extension.stage1 = Vec4::new(mode, 0.0, 0.0, 0.0);
                    m.extension.stage1_texture = tex1.clone();
                    Some(mats.materials().add(m))
                };
                let shared = pair.fade_blend == pair.steady;
                if let Some(steady) = with_stage1(&pair.steady) {
                    let fade = if shared {
                        Some(steady.clone())
                    } else {
                        with_stage1(&pair.fade_blend)
                    };
                    if let Some(fade) = fade {
                        pair.steady = steady;
                        pair.fade_blend = fade;
                    }
                }
            }
            let mut lane_materials = [pair.steady.clone(), pair.fade_blend.clone()];
            let lane = SkyMatLane::register(
                sub,
                uv,
                tint,
                &mut table,
                mats.materials(),
                &mut lane_materials,
            );
            [pair.steady, pair.fade_blend] = lane_materials;
            let mesh = meshes.add(mesh);
            let pose = match sole_bone(sub) {
                Some(b) if rig.bone_moves(b) => PartPose::Rigid(b),
                Some(_) => PartPose::Static,
                None if rig.animates() && !sub.joints.is_empty() => PartPose::Skinned {
                    mesh: mesh.clone(),
                    base: positions,
                    joints: sub.joints.clone(),
                    weights: sub.weights.clone(),
                },
                None => PartPose::Static,
            };
            let skinned = matches!(pose, PartPose::Skinned { .. });
            let part = commands
                .spawn((
                    Mesh3d(mesh),
                    MeshMaterial3d(pair.steady.clone()),
                    Transform::default(),
                    Visibility::Hidden, // `apply_skybox_visibility` turns on exactly the wanted one
                    // The crossfade's alpha (bits 0..=5), written only by `apply_skybox_visibility`.
                    MeshTag(crate::mesh_tag::spawn_tag(0, 1.0)),
                    SkyboxLocal::default(),
                    SkyboxPart {
                        path: path.to_string(),
                        steady: pair.steady,
                        fade_blend: pair.fade_blend,
                        alpha: sub.alpha_anim.clone(),
                        anim_alpha: 1.0,
                        lane,
                        pose,
                        skinned_at: None,
                    },
                ))
                .id();
            if skinned {
                // The mesh moves under its bounds; the shell surrounds the eye anyway.
                commands.entity(part).insert(NoFrustumCulling);
            }
        }
        rigs.0.insert(path.to_string(), rig);
        built.0.insert(
            path.to_string(),
            u16::try_from(subs.len()).unwrap_or(u16::MAX),
        );
    }
}

/// MONKEY (skybox): the flag `0x4` cone, the modern client's sky-cone draw in the final fog colour
/// (`skyMesh0x4Sky`): a band around the eye, opaque at and below the horizon, fading out by 24°
/// of elevation. White vertices carry the alpha; the fog colour rides its tint row.
fn build_fog_cone(
    commands: &mut Commands,
    cone: &mut FogCone,
    meshes: &mut Assets<Mesh>,
    table: &mut crate::mat_anim_table::MatAnimTable,
    mats: &mut M2BatchMaterials,
) {
    const SEGMENTS: usize = 32;
    const RADIUS: f32 = 40.0;
    // (elevation degrees, alpha), bottom to top.
    const RINGS: [(f32, f32); 7] = [
        (-60.0, 1.0),
        (0.0, 1.0),
        (4.0, 0.85),
        (8.0, 0.6),
        (12.0, 0.35),
        (18.0, 0.12),
        (24.0, 0.0),
    ];
    let mut positions = Vec::new();
    let mut colors = Vec::new();
    for (elev, alpha) in RINGS {
        let (se, ce) = elev.to_radians().sin_cos();
        for s in 0..=SEGMENTS {
            let az = s as f32 / SEGMENTS as f32 * std::f32::consts::TAU;
            positions.push([RADIUS * ce * az.cos(), RADIUS * se, RADIUS * ce * az.sin()]);
            colors.push([1.0, 1.0, 1.0, alpha]);
        }
    }
    let row = SEGMENTS as u32 + 1;
    let mut indices = Vec::new();
    for r in 0..(RINGS.len() as u32 - 1) {
        for s in 0..SEGMENTS as u32 {
            let a = r * row + s;
            let b = a + row;
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    let n = positions.len();
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 1.0, 0.0]; n]);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.0]; n]);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
    let Some(material) = mats.sky_fog_cone(FOG_CONE_ORDER) else {
        return;
    };
    let slot = table.alloc();
    if let Some(slot) = slot {
        crate::model_render::lazy::with_material_mut(mats.materials(), material.id(), |m| {
            m.extension.anim_slots.y = f32::from(slot);
        });
    }
    let entity = commands
        .spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(material),
            Transform::default(),
            Visibility::Hidden,
            MeshTag(crate::mesh_tag::spawn_tag(0, 1.0)),
            SkyboxLocal::default(),
            FogConePart,
            NoFrustumCulling,
        ))
        .id();
    cone.entity = Some(entity);
    cone.slot = slot;
}

/// The bone every vertex of this batch is wholly weighted to, if any: a rigid batch takes the
/// bone's matrix whole, with no skinning.
fn sole_bone(sub: &benilla_formats::RenderSubmesh) -> Option<u16> {
    let bone = sub.joints.first()?[0];
    (sub.weights.len() == sub.joints.len()
        && sub
            .joints
            .iter()
            .zip(&sub.weights)
            // Not `== 1.0`: the weights are bytes normalised by their sum.
            .all(|(j, w)| j[0] == bone && w[0] > 0.999))
    .then_some(bone)
}

/// MONKEY (skybox): one model's clock and pose this frame.
#[derive(Default)]
struct ModelClock {
    band_t: f32,
    gseq: f64,
    live: bool,
    pose: Vec<Affine3A>,
    /// MONKEY (visualfix): bumped (from a system-wide counter) whenever `pose` changes.
    generation: u64,
    /// MONKEY (visualfix): the next pose, sampled here and swapped in when it differs.
    next: Vec<Affine3A>,
    /// MONKEY (visualfix): the resolve memo, reused.
    scratch: Vec<Option<Affine3A>>,
}

/// MONKEY (skybox): `WOW_SKYBOX_T=<secs>` holds a capture's skybox clock there instead of 0, to
/// photograph an animated sky mid-loop.
fn capture_t() -> f32 {
    static T: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        std::env::var("WOW_SKYBOX_T")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|t: &f32| t.is_finite())
            .unwrap_or(0.0)
    })
}

/// MONKEY (skybox): pose every shown skybox and run its material loops on the model's clock:
/// sequence 0 at `duration × day fraction` under flag `0x1`, else the scene clock. A capture holds
/// `t = 0` unless the day drives the clock.
#[allow(clippy::too_many_arguments)]
fn animate_skyboxes(
    time: Res<Time>,
    clock: Res<crate::lighting::GameClock>,
    want: Res<CameraSkybox>,
    rigs: Res<SkyRigs>,
    mut table: ResMut<crate::mat_anim_table::MatAnimTable>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut parts: Query<(&mut SkyboxPart, &mut SkyboxLocal)>,
    // MONKEY (visualfix): kept across frames (keys cloned only when a model first shows), with
    // a pose generation counter so an unchanged skinned pose is not re-uploaded.
    mut clocks: Local<HashMap<String, ModelClock>>,
    mut generation: Local<u64>,
) {
    if want.0.is_empty() {
        clocks.clear();
        return;
    }
    let deterministic = crate::dev_state::deterministic_run();
    // MONKEY (reviewfix): retain sub-frame precision for skybox loops across long sessions; only
    // narrow the wrapped sequence clock that the f32 track sampler consumes.
    let now = time.elapsed_secs_f64();
    let day = clock.minute.min(1439) as f32 / 1440.0;
    clocks.retain(|path, _| want.0.iter().any(|l| l.weight > 0.0 && l.path == *path));
    for layer in want.0.iter().filter(|l| l.weight > 0.0) {
        let Some(rig) = rigs.0.get(&layer.path) else {
            continue;
        };
        if !clocks.contains_key(&layer.path) {
            clocks.insert(layer.path.clone(), ModelClock::default());
        }
        let Some(c) = clocks.get_mut(&layer.path) else {
            continue;
        };
        // A capture poses at `t = 0`, not the bind pose: a converted skybox places its layers with
        // bone keys, and every row at `t = 0` is its seed.
        let (band_t, gseq, live) = if layer.flags & SKYBOX_FULL_DAY != 0 {
            let g = if deterministic { 0.0 } else { now };
            (rig.duration * day, g, true)
        } else if deterministic {
            (capture_t(), f64::from(capture_t()), true)
        } else {
            let band_t = if rig.duration > 0.0 {
                (now % f64::from(rig.duration)) as f32
            } else {
                0.0
            };
            (band_t, now, true)
        };
        if live && rig.animates() {
            let (next, scratch) = (&mut c.next, &mut c.scratch);
            rig.pose_into(rig.band_time(band_t), gseq, scratch, next);
        } else {
            c.next.clear();
        }
        if c.next != c.pose {
            std::mem::swap(&mut c.pose, &mut c.next);
            *generation += 1;
            c.generation = *generation;
        }
        c.band_t = band_t;
        c.gseq = gseq;
        c.live = live;
    }
    for (mut part, mut local) in &mut parts {
        let Some(c) = clocks.get(&part.path) else {
            continue;
        };
        let seq = rigs.0.get(&part.path).and_then(|r| r.seq_slot);
        let a = part
            .alpha
            .as_ref()
            .map_or(1.0, |a| a.sample(seq, c.band_t, c.gseq).clamp(0.0, 1.0));
        if part.anim_alpha != a {
            part.anim_alpha = a;
        }
        if c.live && part.lane.any() {
            part.lane.tick(c.band_t, c.gseq, &mut table);
        }
        let bone = |b: u16| {
            c.pose
                .get(usize::from(b))
                .copied()
                .unwrap_or(Affine3A::IDENTITY)
        };
        let skinned_at = part.skinned_at;
        let mut uploaded = false;
        match &part.pose {
            PartPose::Static => {}
            PartPose::Rigid(b) => {
                let m = bone(*b);
                if local.0 != m {
                    local.0 = m;
                }
            }
            PartPose::Skinned {
                mesh,
                base,
                joints,
                weights,
            } => {
                if c.pose.is_empty() || skinned_at == Some(c.generation) {
                    continue;
                }
                let skinned: Vec<[f32; 3]> = base
                    .iter()
                    .zip(joints.iter().zip(weights))
                    .map(|(p, (j, w))| {
                        let mut out = Vec3::ZERO;
                        for k in 0..4 {
                            if w[k] > 0.0 {
                                out += bone(j[k]).transform_point3(*p) * w[k];
                            }
                        }
                        out.to_array()
                    })
                    .collect();
                if let Some(m) = meshes.get_mut(mesh) {
                    m.insert_attribute(Mesh::ATTRIBUTE_POSITION, skinned);
                }
                uploaded = true;
            }
        }
        if uploaded {
            part.skinned_at = Some(c.generation);
        }
    }
}

/// Show the wanted skyboxes at their layer weights and hide every other, the sole `Visibility`,
/// material and `MeshTag` writer for these entities. As the reference: hidden at weight 0
/// (`0x6d4afe`), the weight in every batch's alpha (`0x710cb0` → `[CM2Model+0x180]`), and below 1
/// the batch promoted to SRC_ALPHA blending whatever its mode (`0x811fe0`). MONKEY (skybox): the
/// weight carries the batch's alpha tracks too, and a layer that keeps the celestial pass draws
/// its batches blended, so they sort over the procedural sky as the modern client orders them.
#[allow(clippy::type_complexity)]
fn apply_skybox_visibility(
    want: Res<CameraSkybox>,
    lighting: Res<crate::lighting::WowLighting>,
    cone: Res<FogCone>,
    mut table: ResMut<crate::mat_anim_table::MatAnimTable>,
    mut parts: Query<(
        &SkyboxPart,
        &mut Visibility,
        &mut MeshMaterial3d<WowModelMaterial>,
        &mut MeshTag,
    )>,
    mut cones: Query<(&mut Visibility, &mut MeshTag), (With<FogConePart>, Without<SkyboxPart>)>,
) {
    for (part, mut vis, mut mat, mut tag) in &mut parts {
        let layer = want.layer(&part.path);
        let w = layer.map_or(0.0, |l| l.weight) * part.anim_alpha;
        let show = w > 0.0;
        let target = if show {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        if *vis != target {
            *vis = target;
        }
        if !show {
            continue;
        }
        let keep = layer.is_some_and(|l| l.flags & SKYBOX_KEEP_CELESTIAL != 0);
        let handle = if w < 1.0 || keep {
            &part.fade_blend
        } else {
            &part.steady
        };
        if mat.0 != *handle {
            mat.0 = handle.clone();
        }
        let bits = crate::mesh_tag::with_alpha(tag.0, w);
        if tag.0 != bits {
            tag.0 = bits;
        }
    }
    // MONKEY (skybox): the fog cone at the heaviest fog-blending layer's weight, in the fog colour.
    let Some(entity) = cone.entity else {
        return;
    };
    let w = want
        .0
        .iter()
        .filter(|l| l.flags & SKYBOX_FOG_BLEND != 0 && !l.celestial)
        .map(|l| l.weight)
        .fold(0.0, f32::max);
    if let Ok((mut vis, mut tag)) = cones.get_mut(entity) {
        let target = if w > 0.0 {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        if *vis != target {
            *vis = target;
        }
        let bits = crate::mesh_tag::with_alpha(tag.0, w);
        if tag.0 != bits {
            tag.0 = bits;
        }
    }
    if let (Some(slot), true) = (cone.slot, w > 0.0) {
        let c = benilla_assets::quant255(lighting.fog_color);
        table.set(slot, [c[0] - 1.0, c[1] - 1.0, c[2] - 1.0, 0.0]);
    }
}

/// Pin the box to the camera, world-aligned and at authored scale, the model's origin at the eye as
/// the reference places it (`0x707680` given a zeroed recentre vector, `0x6d4b3a`–`0x6d4b48`):
/// `StratholmeSkybox` is authored off-centre, so recentring it would be wrong. MONKEY (skybox): a
/// posed batch rides its bone's matrix inside the anchor, `T(eye) · M(bone)`.
#[allow(clippy::type_complexity)]
fn follow_camera(
    cam: Query<&GlobalTransform, With<WorldCamera>>,
    mut parts: Query<(&mut Transform, &mut GlobalTransform, &SkyboxLocal), Without<WorldCamera>>,
) {
    let Some(cam_gt) = cam.iter().next() else {
        return;
    };
    let eye = cam_gt.translation();
    for (mut tf, mut gt, local) in &mut parts {
        let m = Affine3A::from_translation(eye) * local.0;
        *tf = Transform::from_matrix(m.into());
        // Propagation already ran this frame: the direct global write is what renders.
        *gt = GlobalTransform::from(m);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On the real art: the seventeen other batches are wholly on bone 0, which has no track.
    #[test]
    fn only_the_belt_batches_of_the_caverns_sky_resolve_to_a_moving_bone() {
        let data = benilla_formats::wow_data_or_skip!();
        let mut chain = benilla_formats::Chain::open(&data).expect("open vanilla patch chain");
        const SKY: &str = "Environments\\Stars\\CavernsOfTimeSky.m2";
        let subs = benilla_formats::load_m2_mesh(&mut chain, SKY).expect("load the sky");
        let bytes = chain.read_file(SKY).expect("read the sky");
        let rig = SkyRig::from_bytes(&bytes);

        let bones: Vec<Option<u16>> = subs.iter().map(sole_bone).collect();
        assert_eq!(bones.len(), 21, "21 authored batches");
        // Batches 5..=8 are the belts (`benilla-extract m2batch`); every other rides bone 0.
        assert_eq!(
            &bones[5..=8],
            &[Some(1), Some(2), Some(3), Some(3)],
            "the belt batches and the bones they ride"
        );
        assert!(
            bones
                .iter()
                .enumerate()
                .filter(|(i, _)| !(5..=8).contains(i))
                .all(|(_, b)| *b == Some(0)),
            "every non-belt batch is wholly on bone 0: {bones:?}"
        );
        let moving = bones
            .iter()
            .filter(|b| b.is_some_and(|b| rig.bone_moves(b)))
            .count();
        assert_eq!(moving, 4, "exactly the four belt batches turn");
    }

    fn layer(path: &str, weight: f32) -> SkyboxLayer {
        SkyboxLayer {
            path: path.into(),
            weight,
            flags: 0,
            celestial: false,
        }
    }

    /// MONKEY (visualfix): the allocation-free comparison agrees with comparing `norm` strings.
    #[test]
    fn norm_eq_matches_norm() {
        let paths = [
            r"Environments\Stars\StratholmeSkybox.mdx",
            r"environments\stars\stratholmeskybox.m2",
            r"ENVIRONMENTS\STARS\STRATHOLMESKYBOX.MDL",
            r"Environments\Stars\StratholmeSkybox.wmo",
            r"Environments\Stars\Other.m2",
            r"m2",
            r"",
        ];
        for a in paths {
            for b in paths {
                assert_eq!(norm_eq(a, b), norm(a) == norm(b), "{a} vs {b}");
            }
        }
    }

    #[test]
    fn skybox_order_bands_follow_batch_counts_and_reserve_the_cone() {
        assert_eq!(skybox_batch_order(true, 0, 23), 24);
        assert_eq!(skybox_batch_order(false, 24, 0), 25);
        assert_eq!(skybox_batch_order(false, 0, 31), 32);
        assert_eq!(
            skybox_batch_order(false, 40, usize::MAX),
            FOG_CONE_ORDER - 1
        );
        assert_eq!(FOG_CONE_ORDER, 58);
    }

    #[test]
    fn a_wmo_sky_crossfades_the_zone_sky() {
        let mut layers = vec![layer("a.m2", 1.0)];
        collect_layer(&mut layers, "B.MDX", 0.25, 0);
        assert!((layers[0].weight - 0.75).abs() < 1e-6);
        assert!((layers[1].weight - 0.25).abs() < 1e-6);
        // The same model under another spelling merges.
        collect_layer(&mut layers, "b.m2", 1.0, 0);
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].weight, 0.0);
    }

    #[test]
    fn the_primary_layer_is_the_heaviest_main_one() {
        let mut sky = CameraSkybox(vec![layer("a", 0.4), layer("b", 0.6)]);
        sky.0.push(SkyboxLayer {
            celestial: true,
            ..layer("c", 1.0)
        });
        assert_eq!(sky.primary().map(|l| l.path.as_str()), Some("b"));
    }

    /// MONKEY (leftovers): a flag's weight is its heaviest main carrier; celestial layers and
    /// layers without the flag do not count.
    #[test]
    fn flag_weight_is_the_heaviest_main_carrier() {
        let with = |path, w, flags| SkyboxLayer {
            flags,
            ..layer(path, w)
        };
        let mut sky = CameraSkybox(vec![
            with("a", 0.3, 0x8),
            with("b", 0.7, 0x1),
            with("c", 0.5, 0x18),
        ]);
        sky.0.push(SkyboxLayer {
            celestial: true,
            ..with("d", 1.0, 0x18)
        });
        assert_eq!(
            sky.flag_weight(benilla_formats::SKYBOX_FORCE_SUN_SHAFTS),
            0.5
        );
        assert_eq!(sky.flag_weight(benilla_formats::SKYBOX_NO_SUN_FOG), 0.5);
        assert_eq!(sky.flag_weight(benilla_formats::SKYBOX_FOG_BLEND), 0.0);
        assert_eq!(CameraSkybox::default().flag_weight(0x8), 0.0);
    }
}
