//! MONKEY (static torch cache): persistent six-face static maps for up to sixteen fixtures.
//! Bevy point-light clustering is disabled on the world view, so this dedicated depth lane
//! supplies both receivers. Static geometry renders only on promotion/residency/fixture changes;
//! nearest-N fixtures copy their six cached faces into a live bank and overlay moving entities.
//! With the defaults (12 resident, 4 dynamic), a steady frame records 24 copies, 24 entity draws
//! and zero static draws. At most two dirty fixtures rebuild per frame; physical slots survive
//! contribution ranking and cross-fades. See benilla_app::torch_shadow for eligibility and keys.
//!
//! **Phase 3A: the depth array is a shared `Image` ASSET**, not a render-world-owned texture. The
//! entity receiver (`wow_model.wgsl`, every unit/NPC/GameObject/interior prop) has no spare bind
//! group — a Bevy material draw sets groups 0/1/2 only — so its torch bindings ride the material's
//! own group as `Handle<Image>` + raw `Buffer` fields (`WowModelExt::torch_depth`/`torch_buf`),
//! and `AsBindGroup` resolves a `Handle<Image>` through `RenderAssets<GpuImage>`. So the image
//! ([`TorchDepthImage`]) is created ONCE at app startup in the main world (always — an absent image
//! would stall every model material's bind group and blank every model), extracted here, and BOTH
//! receivers read the same `GpuImage`: the depth node renders into per-layer views of its texture,
//! static_gx binds its D2Array view, the material binds the same view + the image's comparison
//! sampler. The 16-fixture table gets the same treatment ([`SharedTorchBuffer`]: one raw buffer,
//! rewritten in place every frame from [`TorchShadowViews`] with the SAME [`TorchTableUniform`]
//! bytes static_gx's group-3 uniform carries).
//!
//! Reverse-Z throughout (the whole engine is — `capture::depth_probe`): the depth pass clears to 0.0
//! (= far), keeps the nearest surface with `GreaterEqual`, and the comparison sampler is `GreaterEqual`
//! too. The per-fixture `view_proj` is a reverse-Z perspective built app-side in f64 (fixtures sit at
//! absolute Bevy world coordinates, ~9,300 out) and downcast to f32.

use std::hash::{Hash, Hasher};
use std::sync::Mutex;

use bevy::asset::RenderAssetUsages;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::ecs::system::SystemParam;
use bevy::image::{ImageCompareFunction, ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
// Bevy's OWNED vertex-buffer layout (the pipeline descriptor wants this, not wgpu's borrowed one).
use bevy::mesh::VertexBufferLayout;
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::mesh::allocator::MeshAllocator;
use bevy::render::mesh::{RenderMesh, RenderMeshBufferInfo};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{
    NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::binding_types::uniform_buffer_sized;
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use bytemuck::Zeroable;

use super::render::StaticGxView;
// MONKEY (torch owner exclusion): the placement identity both caster lanes carry, and the
// subsystem tag that keeps a building's own batches apart from the props standing in it.
use crate::interact::WorldObject;
use crate::model_render::ModelKind;
use benilla_assets::materials::TorchBinds;
use bevy::math::Affine3A;

/// MONKEY (live bank rank): each 512-square Depth32 face is 1 MiB. Sixteen static cubes plus
/// eight compact live overlays cost 144 MiB; 384-square faces would cost 81 MiB.
const TORCH_MAP_EDGE: u32 = 512;
pub(crate) const MAX_TORCH_MAPS: usize = 16;
const MAX_TORCH_DYNAMIC: usize = 8;
/// MONKEY (moving fixture): how many slots may be FULLY DYNAMIC in one frame - re-rendered from
/// scratch (static geometry AND entities) because their fixture is physically moving. Two, because
/// each one costs a whole cached slot's work every frame: the CPU gather that the static cache
/// exists to amortise, plus the render-world depth rebuild. The app lane picks the two by score and
/// fast-fades the rest ([`benilla_app::torch_shadow`]); the same number bounds the rebuild budget
/// in [`torch_cache_plan`], so a moving fixture can never starve a resident room's static maps of
/// their own (separate) two.
pub(crate) const TORCH_MOVING_MAX: usize = 2;
pub(crate) const CUBE_FACES: usize = 6;
/// MONKEY (live bank rank): matrix count stays 96, independent of texture capacity. Static
/// layers 0..96 stay slot-addressed; live layers 96..144 follow ascending dynamic-mask rank.
pub(crate) const MAX_TORCH_LAYERS: usize = MAX_TORCH_MAPS * CUBE_FACES;
const TORCH_TEXTURE_LAYERS: usize = MAX_TORCH_LAYERS + MAX_TORCH_DYNAMIC * CUBE_FACES;
/// MONKEY (static torch cache): count@0 (16), positions@16 (256), view_projs@272 (6144).
/// Total 6416 bytes, identical under WGSL uniform/storage alignment. count.z is the live-bank
/// bit mask, count.w reserved. Both WGSL TorchTable copies and render.rs MUST agree with this.
pub(crate) const TORCH_TABLE_BYTES: u64 = 6416;
/// MONKEY (outdoor torch shadows): `count.w` bit 0 — the exterior receiver lane's live gate. Keep
/// in sync with `TORCH_EXT_LANE` in BOTH `static_gx.wgsl` and `wow_model.wgsl`.
const TORCH_EXT_LANE: u32 = 1;

/// MONKEY (torch owner exclusion): WHO a light belongs to — the identity a caster gather uses to
/// drop a fixture's OWN body out of the fixture's OWN shadow map.
///
/// The bug this answers. A synthetic fire light is placed AT THE FLAME, and a flame is *inside* the
/// thing that burns it: inside a lantern's glass housing, inside a campfire's ring of logs, at the
/// head of a wall torch. Nothing in this lane knew the two belonged together — the light is an ECS
/// entity, the mesh is either a retained [`super::GxItem`] or a model-part entity, and the
/// placement that spawned both is not carried on either side — so the fixture was the nearest
/// occluder on all six of its cube faces and blacked out its own pool. That is the moving bright
/// WEDGE on the ground under the inn's swinging lantern (the housing shadowing every direction but
/// one gap), the BLACK SQUARE under the wall torch by the crate, and the dark blotches under the
/// faire torches.
///
/// The first attempt approximated ownership by PROXIMITY (the retired `TORCH_EXT_SELF_EXCLUDE`:
/// drop an item whose whole bounding sphere lies within 2.5 yd of the flame). That covers a
/// campfire — a prop that is a ball around its own flame — and essentially nothing else: a lantern
/// hanging off a 4-yd post, or a torch on a tall bracket, has bounds that run all the way to the
/// ground and is never "wholly within 2.5 yd" of anything. So ownership is carried EXPLICITLY here,
/// and proximity is demoted to a bounded fallback ([`torch_bound_contains`]).
///
/// Two shapes, because lights reach the world by two routes:
///  * [`Self::Placement`] — a placed ADT doodad's or WMO prop's own M2 light
///    (`terrain_stream::spawn`'s `spawn_lights_for`). The key is the placement's
///    [`WorldObject`] identity, because that is the one identity BOTH caster lanes already carry:
///    the retained `GxItem` holds it as an `Arc`, and the entity path's parts hold a CLONE of it.
///    Compared by CONTENT (kind + placement uniqueId + label hash) for exactly that reason — an
///    `Arc` pointer would match the retained half and never the entity half.
///  * [`Self::Instance`] — an entity-hosted light (`entities::carried_light`): a GameObject
///    brazier, a placed campfire, an NPC's torch. Those hang under the host model's FRAME entity,
///    which no `WorldObject` is available at, so the frame IS the identity and a caster part is
///    "mine" iff it is a descendant of it.
///
/// Deliberately NOT written onto a WMO's authored MOLT fixtures. A MOLT light has no model of its
/// own, and its placement identity is the BUILDING's — tagging it would exclude every wall, floor
/// and pillar of the building from its own torch's map, i.e. delete the interior shadow lane.
#[derive(Component, Clone, Copy, PartialEq, Eq, Debug)]
pub enum LightOwner {
    Placement {
        kind: ModelKind,
        /// The placement uniqueId (`WorldObject::id`).
        id: u32,
        /// A hash of the model path, so two identical lanterns in one building are told apart from
        /// its barrels and its chairs (WMO props all share their BUILDING's uniqueId).
        label: u64,
    },
    Instance(Entity),
}

impl LightOwner {
    /// The placement key of `object` — the value both caster lanes are compared against.
    pub fn placement(object: &WorldObject) -> Self {
        Self::Placement {
            kind: object.kind,
            id: object.id,
            label: label_hash(&object.label),
        }
    }

    /// Does this owner name `object`? The two SCALAR lanes are tested first and the label hash only
    /// if they both match: this runs per candidate item inside a 48-yd gather, and hashing a model
    /// path for every barrel in a city would be real cost for an answer that is almost always "no".
    pub fn owns(&self, object: &WorldObject) -> bool {
        match *self {
            Self::Placement { kind, id, label } => {
                id == object.id && kind == object.kind && label == label_hash(&object.label)
            }
            Self::Instance(_) => false,
        }
    }

    /// The host model frame, for the entity lane's ancestry test — `None` for a placed light.
    pub fn instance(&self) -> Option<Entity> {
        match *self {
            Self::Instance(e) => Some(e),
            Self::Placement { .. } => None,
        }
    }
}

fn label_hash(label: &str) -> u64 {
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    label.hash(&mut hash);
    hash.finish()
}

/// MONKEY (torch owner exclusion): how far OUTSIDE its own bounds a flame may sit and still count
/// as being inside the fixture (yd). A synthesised light is positioned from the model's flame
/// EMITTER, which routinely sits a finger's width proud of the housing it burns in; 0.3 yd closes
/// that gap without reaching anything the fixture merely stands next to.
const TORCH_SELF_PAD: f32 = 0.3;
/// MONKEY (torch owner exclusion): the containment fallback applies only to FIXTURE-SIZED items —
/// this is the cap on the transformed bounding-sphere radius (yd).
///
/// Unbounded containment would be a disaster, and specifically on the lane that works today: a WMO
/// group's wall batch is a ROOM-sized box, it trivially "contains" every candle in the room, and
/// excluding it would delete that room's shadows and leak the candle straight through its walls.
/// 6 yd covers a tall lamp post, a gallows-arm lantern bracket and a bonfire whole, and is an order
/// of magnitude under any wall, floor, building or terrain batch.
const TORCH_SELF_MAX_EXTENT: f32 = 6.0;
/// MONKEY (torch owner exclusion): the last resort, for items whose bounds are DEGENERATE (a candle
/// flame card, a zero-extent helper) where containment cannot decide anything. An item whose WHOLE
/// bounding sphere lies inside this of the flame is the flame's own body by construction. This is
/// the surviving half of the retired `TORCH_EXT_SELF_EXCLUDE`, shrunk from 2.5 yd to 0.6 so that it
/// can no longer swallow the crate, fence post or barrel standing beside the fire.
const TORCH_SELF_TINY: f32 = 0.6;

/// **The app→render publication** (`benilla_app::torch_shadow` writes it each frame, extracted here):
/// the promoted fixtures' cube-face matrices, world positions and static mesh revisions.
/// `count` is the sparse slot high-water mark (0 when the lane is off).
#[derive(Resource, Clone, ExtractResource)]
pub struct TorchShadowViews {
    pub count: u32,
    // MONKEY (static torch cache): sparse physical slots; count is the high-water mark, not the
    // population. Holes have weight zero. ready_mask is assigned only in render preparation.
    pub dynamic_mask: u32,
    pub ready_mask: u32,
    /// MONKEY (torch caster selection): the live PCF tap-radius scale (`interiorShadowSoft`,
    /// 0.5..3). Carried here rather than in `DynamicInteriors` because it belongs to the shadow
    /// table's own bytes — [`TorchTableUniform::pack`] puts it in the otherwise-padding `count.y`.
    pub soft: f32,
    /// MONKEY (shadow floor): the live torch SHADOW STRENGTH (`torchShadowStrength`, 0..1, default
    /// 0.7) - how much of the direct term a fully shadowed fragment loses. It rides the same
    /// `count.y` word as [`Self::soft`] (high half; see [`TorchTableUniform::pack`]) and the
    /// receivers fold it into the slot's cross-fade weight, so a shadow darkens the direct arm to
    /// 30 % instead of to nothing. The fill/ambient arms never saw this factor and are untouched.
    pub strength: f32,
    /// MONKEY (moving fixture): which slots are MOVING this frame - a carried/unit-hosted fixture
    /// or one that has drifted off the position its cached map was baked at. They get their own
    /// rebuild budget in [`torch_cache_plan`] ([`TORCH_MOVING_MAX`]) ahead of the resident slots,
    /// because a moving fixture's map is WRONG (not merely stale) the moment it is deferred: the
    /// projections republish from the live position every frame while the depth still holds the
    /// old one, which is the smeared, lagging pool the imp dragged across the inn floor.
    pub moving_mask: u32,
    /// MONKEY (outdoor torch shadows): does the EXTERIOR receiver lane run this frame? The app lane
    /// sets it when `exteriorShadows` is on, the sun is below the daylight threshold, AND at least
    /// one promoted slot is an exterior fixture. It rides to the receivers in the table's
    /// `count.w` bit 0 (the word that has been reserved padding since the table was written), and
    /// it is the ONE uniform read every exterior-lane cost in `static_gx.wgsl`/`wow_model.wgsl`
    /// hangs off — by day, or with the cvar off, the receivers are the code they always were.
    pub exterior: bool,
    /// `positions[i].xyz` = fixture world position (absolute Bevy), `.w` = the slot's FADE WEIGHT
    /// `0..1` (MONKEY, torch caster selection). The `.w` lane used to carry the fixture's range,
    /// which no shader ever read; it now carries the cross-fade, and both receivers return
    /// `mix(1.0, shadow, w)` so a promoted fixture's shadow ramps in over ~1/3 s and a demoted one
    /// ramps out before its slot is reused — the fix for the pop this whole change is about.
    pub positions: [Vec4; MAX_TORCH_MAPS],
    /// The reverse-Z `view_proj` of cube face `f` of fixture `i` at `[i * CUBE_FACES + f]`
    /// (identity in unused slots).
    pub view_projs: [Mat4; MAX_TORCH_LAYERS],
    /// MONKEY (static torch cache): fresh asset identity per geometry revision/slot owner;
    /// None while the app's two-rebuild budget defers this slot. Static uploads must be resident
    /// before depth can be marked valid. Empty resident meshes still clear all six faces.
    pub caster_meshes: [Option<AssetId<Mesh>>; MAX_TORCH_MAPS],
    /// MONKEY (static torch cache): moving/skinned entities only, regenerated each frame.
    pub entity_mesh: Option<AssetId<Mesh>>,
}

impl TorchShadowViews {
    /// MONKEY (live bank rank): expose the allocation cap through the existing app resource.
    pub const MAX_TORCH_DYNAMIC: usize = MAX_TORCH_DYNAMIC;
    /// MONKEY (moving fixture): ONE definition of the moving budget, shared by the app lane that
    /// picks the slots and the render plan that rebuilds them.
    pub const MAX_TORCH_MOVING: usize = TORCH_MOVING_MAX;

    /// MONKEY (live bank rank): WGSL countOneBits(mask & ((1u << slot) - 1u)). Use the
    /// FINAL ready-filtered mask for rendering; app traces describe only requested ranks.
    pub fn live_rank(dynamic_mask: u32, slot: usize) -> usize {
        debug_assert!(slot < MAX_TORCH_MAPS);
        (dynamic_mask & ((1u32 << slot) - 1)).count_ones() as usize
    }
}

/// Hand-written because `[Mat4; 96]` has no `Default` — std only derives array `Default` up to
/// `N = 32`, so the derive that worked at 24 layers stops compiling at 96. (It also lets `soft`
/// default to a sane `1.0` rather than a zero tap radius.)
impl Default for TorchShadowViews {
    fn default() -> Self {
        Self {
            count: 0,
            dynamic_mask: 0,
            ready_mask: 0,
            soft: 1.0,
            // MONKEY (shadow floor): 1.0 = the pre-feature pitch-black shadow, so a views value
            // that never went through the cvar cannot silently lighten the world.
            strength: 1.0,
            moving_mask: 0,
            exterior: false,
            positions: [Vec4::ZERO; MAX_TORCH_MAPS],
            view_projs: [Mat4::ZERO; MAX_TORCH_LAYERS],
            caster_meshes: [None; MAX_TORCH_MAPS],
            entity_mesh: None,
        }
    }
}

/// MONKEY (torch shadows Phase 3A): the ONE shared torch depth array — a `Depth32Float`
/// MONKEY (live bank rank): 512×512×144-layer `Image` asset (render, sample, copy-src/dst),
/// no CPU data) whose sampler descriptor carries `compare: Some(GreaterEqual)`, so its `GpuImage`
/// sampler is a real comparison sampler and its default view is a `D2Array` over all 144 layers
/// (wgpu's default view dimension for a multi-layer 2D texture). Created ONCE at startup by
/// [`new_torch_shared`] (always — regardless of the interior-shadow cvars — because every model
/// material binds it; a missing image would stall every material's bind group), inserted in the
/// main world so material construction can clone the handle, and cloned into the render world so
/// the depth node and static_gx's group-3 prepare can look its `GpuImage` up.
#[derive(Resource, Clone, ExtractResource)]
pub struct TorchDepthImage(pub Handle<Image>);

/// MONKEY (torch shadows Phase 3A): the shared 16-fixture torch TABLE buffer — 6416 bytes of
/// [`TorchTableUniform`] (`STORAGE | COPY_DST`), the entity receiver's `torch_table` at material
/// binding 93. Created once beside [`TorchDepthImage`], cloned into every model material, and
/// rewritten in place every frame by [`upload_torch_table`] (render world, `PrepareResources`) —
/// the exact pattern of the shared light buffer (`lighting::global_light::upload_light`).
#[derive(Resource, Clone, ExtractResource)]
pub struct SharedTorchBuffer(pub Buffer);

/// The main-world pair of torch resources as the ONE system param a material-building site adds
/// beside its `SharedLightBuffer` (several of those systems sit at Bevy's 16-param ceiling).
/// [`Self::binds`] is `None` until [`new_torch_shared`] has run — the same "retry, don't bake
/// against nothing" contract the light buffer gives those sites.
#[derive(SystemParam)]
pub struct TorchShared<'w> {
    image: Option<Res<'w, TorchDepthImage>>,
    table: Option<Res<'w, SharedTorchBuffer>>,
}

impl TorchShared<'_> {
    /// The receiver bindings to clone into a `WowModelExt`, or `None` before startup created them.
    pub fn binds(&self) -> Option<TorchBinds> {
        Some(TorchBinds {
            depth: self.image.as_ref()?.0.clone(),
            table: self.table.as_ref()?.0.clone(),
        })
    }
}

/// Create the shared torch depth image + table buffer (main world, `Startup` — `RenderDevice` and
/// `Assets<Image>` are both live there). Called by the asset foundation next to the shared light
/// buffer so the pair exists before any material is built, whatever the cvars say.
pub fn new_torch_shared(
    device: &RenderDevice,
    images: &mut Assets<Image>,
) -> (TorchDepthImage, SharedTorchBuffer) {
    let image = Image {
        // A pure GPU render target: no CPU pixels, ever.
        data: None,
        texture_descriptor: TextureDescriptor {
            label: Some("torch_depth"),
            size: Extent3d {
                width: TORCH_MAP_EDGE,
                height: TORCH_MAP_EDGE,
                depth_or_array_layers: TORCH_TEXTURE_LAYERS as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            // MONKEY (static torch cache): wgpu 27 transfer.rs permits partial ARRAY layers
            // for depth, provided x/y cover the entire mip. Banks are disjoint subresources.
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST,
            view_formats: &[],
        },
        // The receivers' `GreaterEqual` comparison sampler (reverse-Z: lit iff the fragment's own
        // depth ≥ the stored nearest depth). `ImageSamplerDescriptor::as_wgpu` carries `compare`
        // through, so `GpuImage::sampler` IS a comparison sampler — which the `AsBindGroup`
        // derive's `sampler_type = "comparison"` binding hands to the material.
        sampler: ImageSampler::Descriptor(ImageSamplerDescriptor {
            label: Some("torch_depth_cmp".into()),
            mag_filter: ImageFilterMode::Linear,
            min_filter: ImageFilterMode::Linear,
            compare: Some(ImageCompareFunction::GreaterEqual),
            ..Default::default()
        }),
        // `None` ⇒ wgpu's default view: the texture's format, all mips/layers, and — for a D2
        // texture with >1 layers — `D2Array`, exactly the `texture_depth_2d_array` both receivers
        // declare. (Spelled out rather than relied on silently: it is the one fact the whole
        // sharing scheme rests on.)
        texture_view_descriptor: None,
        asset_usage: RenderAssetUsages::RENDER_WORLD,
        ..Default::default()
    };
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("wow_shared_torch_table"),
        size: TORCH_TABLE_BYTES,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    (
        TorchDepthImage(images.add(image)),
        SharedTorchBuffer(buffer),
    )
}

/// MONKEY (static torch cache): byte-identical std140/std430 table in BOTH receiver shaders.
/// count@0: live high-water mark, (strength*100 << 16 | soft*100), dynamic mask, lane flags (16 B).
/// positions[16]@16: world xyz + fade weight (256 bytes).
/// view_projs[96]@272: six static-bank face matrices per physical slot (6144 bytes).
/// Total 6416. Unready/hole slots have zero weight; no stale owner's depth can be sampled.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct TorchTableUniform {
    count: u32,
    /// MONKEY (torch caster selection): `soft x 100` — the PCF tap-radius scale, riding the first
    /// of the three padding words the `vec4<u32>` alignment already forced us to carry. The
    /// alternative was a whole new 16-byte row for one dial.
    ///
    /// MONKEY (shadow floor): and now `strength x 100` in its HIGH 16 bits, for the same reason
    /// one level down — the word was carrying one dial in a range (1..300) that needs nine bits,
    /// and growing the table by a row costs 16 bytes in a struct whose SIZE is the contract with
    /// three shaders (a mismatch hides every building). Low half `soft`, high half `strength`;
    /// both receivers mask/shift rather than reading the word whole.
    soft_strength: u32,
    dynamic_mask: u32,
    /// MONKEY (outdoor torch shadows): lane flags, the word that was reserved padding.
    /// Bit 0 ([`TORCH_EXT_LANE`]) = the EXTERIOR receiver lane is live this frame. Both WGSL
    /// copies read it as `count.w`; the table's SIZE is untouched, which is what keeps the
    /// 6416-byte contract (and therefore every building and model on screen) intact.
    flags: u32,
    positions: [[f32; 4]; MAX_TORCH_MAPS],
    view_projs: [[[f32; 4]; 4]; MAX_TORCH_LAYERS],
}

const _: () = assert!(std::mem::size_of::<TorchTableUniform>() as u64 == TORCH_TABLE_BYTES);

impl TorchTableUniform {
    /// Pack this frame's table from the extracted views — a zero table (count 0) when the lane is
    /// off or the resource is absent, so both receivers read "no torches" rather than stale data.
    pub(crate) fn pack(views: Option<&TorchShadowViews>) -> Self {
        let mut table = Self::zeroed();
        let Some(views) = views else {
            return table;
        };
        let count = (views.count as usize).min(MAX_TORCH_MAPS);
        table.count = count as u32;
        table.dynamic_mask = views.dynamic_mask & views.ready_mask;
        // MONKEY (torch caster selection): 0 would mean a zero tap radius (four identical taps =
        // a hard edge), so a table published without a scale reads as the neutral 1.0.
        let soft = if views.soft > 0.01 { views.soft } else { 1.0 };
        // MONKEY (shadow floor): the cvar is clamped 0..1 on the way in, so the pack only has to
        // keep the two halves from colliding - `soft` is capped at 3.0 (300, nine bits) by its own
        // cvar clamp and belt-and-braces here, and `strength` cannot exceed 100. A strength of 0 is
        // MEANINGFUL (shadows off) and must survive the pack, which is why it gets no `max(1)`.
        let soft_pct = ((soft * 100.0).round() as i64).clamp(1, 0xffff) as u32;
        let strength_pct =
            ((views.strength.clamp(0.0, 1.0) * 100.0).round() as i64).clamp(0, 100) as u32;
        table.soft_strength = (strength_pct << 16) | soft_pct;
        // MONKEY (outdoor torch shadows): the exterior receiver lane's live gate. Packed even when
        // `count` is 0 costs nothing and is simpler to reason about than a conditional flag.
        table.flags = if views.exterior { TORCH_EXT_LANE } else { 0 };
        for i in 0..count {
            table.positions[i] = views.positions[i].to_array();
            if views.ready_mask & (1 << i) == 0 {
                table.positions[i][3] = 0.0;
            }
            // Six cube faces per fixture: layer = fixture * 6 + face (the depth node renders and
            // the shaders' `torch_face` pick by the same index).
            for f in 0..CUBE_FACES {
                table.view_projs[i * CUBE_FACES + f] =
                    views.view_projs[i * CUBE_FACES + f].to_cols_array_2d();
            }
        }
        table
    }
}

/// Render-world: rewrite the shared torch table in place every frame, before any draw reads it
/// (`RenderSystems::PrepareResources` — the same stage `upload_light` writes the light blob). One
/// MONKEY (static torch cache): 6416-byte upload per frame, independent of material count. static_gx's group-3 uniform is packed
/// from the same [`TorchTableUniform::pack`] in `render.rs`, so the two receivers can never
/// disagree on a fixture.
fn upload_torch_table(
    queue: Res<RenderQueue>,
    buffer: Option<Res<SharedTorchBuffer>>,
    views: Option<Res<TorchShadowViews>>,
    // MONKEY (torch lane perf): the last bytes actually uploaded, keyed by the buffer they went to
    // (so a recreated buffer can never inherit another one's "already current" verdict), plus the
    // trace counters for the second in progress.
    mut last: Local<Option<(BufferId, TorchTableUniform)>>,
    mut trace: Local<TorchUploadTrace>,
) {
    let Some(buffer) = buffer else {
        return;
    };
    let table = TorchTableUniform::pack(views.as_deref());
    // MONKEY (torch lane perf): skip the upload when the packed bytes are identical to the ones
    // already in the buffer. A standing player in a lit room republishes the SAME 6416 bytes every
    // frame — the fixtures have not moved, the cross-fades have saturated at 1 and the projections
    // are rebuilt from unchanged positions — so this is a staging-belt allocation and a copy with
    // nothing to say. A `memcmp` of 6 KB is far cheaper than the copy it replaces, and the compare
    // is against what we WROTE, so it can never disagree with the buffer's real contents.
    let id = buffer.0.id();
    let current = last
        .as_ref()
        .is_some_and(|(b, t)| *b == id && bytemuck::bytes_of(t) == bytemuck::bytes_of(&table));
    if !current {
        queue.write_buffer(&buffer.0, 0, bytemuck::bytes_of(&table));
        *last = Some((id, table));
        trace.uploads += 1;
    }
    trace.frames += 1;
    trace.report();
}

/// MONKEY (torch lane perf): `WOW_TORCH_TRACE` accounting for [`upload_torch_table`] — the render
/// world's half of the lane's per-second perf line (the app's half is `torch-perf:` in
/// `benilla_app::torch_shadow`). `uploads` well below `frames` is the dedup working.
struct TorchUploadTrace {
    uploads: u32,
    frames: u32,
    since: std::time::Instant,
}

impl Default for TorchUploadTrace {
    fn default() -> Self {
        Self {
            uploads: 0,
            frames: 0,
            since: std::time::Instant::now(),
        }
    }
}

impl TorchUploadTrace {
    fn report(&mut self) {
        if self.since.elapsed().as_secs_f32() < 1.0 {
            return;
        }
        if std::env::var_os("WOW_TORCH_TRACE").is_some() {
            info!(
                "torch-perf: table uploads {}/{} frames",
                self.uploads, self.frames
            );
        }
        *self = Self::default();
    }
}

/// The persistent render-world state that does NOT depend on the image: the comparison sampler
/// static_gx's group 3 binds (kept as its own `GreaterEqual` sampler — unchanged behaviour; the
/// material binds the image's, which is built from the same descriptor), the depth-only pipeline
/// and its per-fixture view-uniform layout. Built once in [`RenderStartup`].
#[derive(Resource)]
pub(crate) struct TorchDepthTargets {
    sampler: Sampler,
    pipeline: CachedRenderPipelineId,
    view_layout: BindGroupLayoutDescriptor,
}

impl TorchDepthTargets {
    /// The `GreaterEqual` comparison sampler for the static_gx group-3 shadow lookup.
    pub(crate) fn sampler(&self) -> &Sampler {
        &self.sampler
    }
}

/// The per-layer D2 render-attachment views over the shared image's texture, cached by that
/// texture's id: rebuilt only if the `GpuImage` is ever re-prepared (it is created once and never
/// modified, so in practice exactly once), never per frame. Empty until the image is resident.
#[derive(Resource, Default)]
struct TorchLayerViews {
    texture: Option<TextureId>,
    /// MONKEY (live bank rank): static slot `i / 6`; live rank `(i - 96) / 6`; face `i % 6`.
    views: Vec<TextureView>,
}

/// MONKEY (static torch cache): only a completed six-face command recording certifies a cache
/// entry. Preparation may run while there is no world view or the pipeline is still compiling;
/// such a frame must not mark a never-rendered map cached. Texture recreation resets all keys.
#[derive(Resource, Default)]
struct TorchStaticCache(Mutex<[Option<AssetId<Mesh>>; MAX_TORCH_MAPS]>);

// MONKEY (static torch cache): the app logs requests; this records actual GPU cache work.
#[derive(Resource, Default)]
struct TorchCacheTrace(Mutex<Option<std::time::Instant>>);

/// This frame's per-fixture view-uniform bind groups (one per live map), rebuilt in
/// MONKEY (static torch cache): retained across frames; refreshed in PrepareResources.
#[derive(Resource, Default)]
struct TorchDepthDraw {
    view_projs: Vec<Mat4>,
    count: usize,
    rebuild_mask: u32,
    /// `bind_groups[i]` binds layer `i`'s `view_proj` at group 0 of the depth pipeline.
    bind_groups: Vec<BindGroup>,
}

fn init_torch_depth(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
) {
    let sampler = render_device.create_sampler(&SamplerDescriptor {
        label: Some("torch_depth_cmp"),
        min_filter: FilterMode::Linear,
        mag_filter: FilterMode::Linear,
        compare: Some(CompareFunction::GreaterEqual),
        ..Default::default()
    });
    // Group 0: the single per-fixture `view_proj` (a mat4, 64 bytes), vertex-only.
    let view_layout = BindGroupLayoutDescriptor::new(
        "torch_depth_view_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX,
            (uniform_buffer_sized(
                false,
                Some(std::num::NonZero::new(64).unwrap()),
            ),),
        ),
    );
    let shader: Handle<Shader> =
        asset_server.load("embedded://benilla_world/shaders/torch_depth.wgsl");
    let pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("torch_depth".into()),
        layout: vec![view_layout.clone()],
        vertex: VertexState {
            shader,
            shader_defs: vec![],
            entry_point: Some("vertex".into()),
            // The caster mesh (`shadow_core::empty_shadow_mesh`) carries ONLY position — stride 12.
            buffers: vec![VertexBufferLayout {
                array_stride: 12,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![VertexAttribute {
                    format: VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                }],
            }],
        },
        // Depth only — no fragment, no color targets.
        fragment: None,
        primitive: PrimitiveState {
            // Cull none is safest for the mixed-winding WMO/static caster geometry.
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: CompareFunction::GreaterEqual,
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        multisample: MultisampleState {
            count: 1,
            ..Default::default()
        },
        ..default()
    });
    commands.insert_resource(TorchDepthTargets {
        sampler,
        pipeline,
        view_layout,
    });
}

/// Build one view-uniform bind group per live layer from the extracted [`TorchShadowViews`], and
/// (re)build the per-layer attachment views whenever the shared image's texture changes identity.
fn prepare_torch_depth(
    mut draw: ResMut<TorchDepthDraw>,
    targets: Option<Res<TorchDepthTargets>>,
    mut views: Option<ResMut<TorchShadowViews>>,
    cache: Res<TorchStaticCache>,
    meshes: Res<RenderAssets<RenderMesh>>,
    allocator: Res<MeshAllocator>,
    image: Option<Res<TorchDepthImage>>,
    images: Res<RenderAssets<GpuImage>>,
    mut layers: ResMut<TorchLayerViews>,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
) {
    draw.count = 0;
    draw.rebuild_mask = 0;
    // MONKEY (static torch cache): fail closed until every prerequisite for this frame exists.
    if let Some(views) = views.as_mut() {
        views.ready_mask = 0;
    }
    // MONKEY (static torch cache): attachment views are allocated once per texture identity.
    let gpu_image = image.as_ref().and_then(|h| images.get(&h.0));
    match gpu_image {
        Some(gpu) if layers.texture != Some(gpu.texture.id()) => {
            *cache.0.lock().unwrap() = [None; MAX_TORCH_MAPS];
            layers.views = (0..TORCH_TEXTURE_LAYERS as u32)
                .map(|i| {
                    gpu.texture.create_view(&TextureViewDescriptor {
                        label: Some("torch_depth_layer"),
                        format: Some(TextureFormat::Depth32Float),
                        dimension: Some(TextureViewDimension::D2),
                        aspect: TextureAspect::DepthOnly,
                        base_mip_level: 0,
                        mip_level_count: Some(1),
                        base_array_layer: i,
                        array_layer_count: Some(1),
                        ..default()
                    })
                })
                .collect();
            layers.texture = Some(gpu.texture.id());
        }
        Some(_) => {}
        None => {
            // Not resident (yet): nothing to render into this frame.
            layers.views.clear();
            layers.texture = None;
        }
    }
    let Some(targets) = targets else {
        return;
    };
    let count = views
        .as_ref()
        .map_or(0, |v| (v.count as usize).min(MAX_TORCH_MAPS));
    if count == 0 {
        return;
    }
    let mut views = views.expect("count > 0 implies views present");
    if layers.texture.is_none()
        || pipeline_cache
            .get_render_pipeline(targets.pipeline)
            .is_none()
    {
        return;
    }
    let cached = cache.0.lock().unwrap();
    let (ready_mask, rebuild_mask, dynamic_mask) = torch_cache_plan(
        &*cached,
        &views.caster_meshes[..count],
        views.dynamic_mask,
        views.moving_mask,
        |id| torch_mesh_ready(id, &meshes, &allocator),
    );
    views.ready_mask = ready_mask;
    views.dynamic_mask = dynamic_mask;
    // If entity upload is late, bind the static bank for this frame, never last frame's overlay.
    if !views
        .entity_mesh
        .is_some_and(|id| torch_mesh_ready(id, &meshes, &allocator))
    {
        views.dynamic_mask = 0;
    }
    let layout = pipeline_cache.get_bind_group_layout(&targets.view_layout);
    // Six cube faces per live fixture: layer = fixture * CUBE_FACES + face.
    let layer_count = count * CUBE_FACES;
    for i in 0..layer_count {
        // MONKEY (static torch cache): fixture projections do not move with the camera.
        // Keep their buffers/bind groups too: zero resource allocation on a steady frame.
        if draw.view_projs.get(i) == Some(&views.view_projs[i]) {
            continue;
        }
        let vp = views.view_projs[i].to_cols_array();
        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("torch_depth_view"),
            contents: bytemuck::cast_slice(&vp),
            usage: BufferUsages::UNIFORM,
        });
        // The wgpu bind group keeps the buffer alive; the local handle can drop.
        let bind_group = render_device.create_bind_group(
            "torch_depth_view",
            &layout,
            &BindGroupEntries::sequential((buffer.as_entire_binding(),)),
        );
        if i < draw.bind_groups.len() {
            draw.bind_groups[i] = bind_group;
            draw.view_projs[i] = views.view_projs[i];
        } else {
            draw.bind_groups.push(bind_group);
            draw.view_projs.push(views.view_projs[i]);
        }
    }
    draw.count = layer_count;
    draw.rebuild_mask = rebuild_mask;
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct TorchDepthLabel;

#[derive(Default)]
struct TorchDepthNode;

impl ViewNode for TorchDepthNode {
    // Run only for the world view — the same marker the retained pass gates on, so a portrait-booth
    // bake never renders torch depth with the booth's matrices.
    type ViewQuery = &'static StaticGxView;

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        _view: QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let Some(views) = world.get_resource::<TorchShadowViews>() else {
            return Ok(());
        };
        let Some(draw) = world.get_resource::<TorchDepthDraw>() else {
            return Ok(());
        };
        let Some(targets) = world.get_resource::<TorchDepthTargets>() else {
            return Ok(());
        };
        // The shared image's per-layer attachment views — empty while the GpuImage is not resident.
        let Some(layers) = world.get_resource::<TorchLayerViews>() else {
            return Ok(());
        };
        // `count` is LAYERS here: six cube faces per live fixture.
        let count = draw
            .count
            .min(views.count as usize * CUBE_FACES)
            .min(MAX_TORCH_LAYERS)
            .min(layers.views.len());
        if count == 0 {
            return Ok(());
        }
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_render_pipeline(targets.pipeline) else {
            return Ok(());
        };
        let meshes = world.resource::<RenderAssets<RenderMesh>>();
        let allocator = world.resource::<MeshAllocator>();
        let image = world.resource::<TorchDepthImage>();
        let images = world.resource::<RenderAssets<GpuImage>>();
        let Some(gpu) = images.get(&image.0) else {
            return Ok(());
        };
        let cache = world.resource::<TorchStaticCache>();
        let mut cached = cache.0.lock().unwrap();
        let trace = world.resource::<TorchCacheTrace>();
        let trace_on = std::env::var_os("WOW_TORCH_TRACE").is_some();
        let mut last_trace = trace.0.lock().unwrap();
        let trace_due = trace_on && last_trace.is_none_or(|t| t.elapsed().as_secs_f32() >= 1.0);
        if trace_due {
            *last_trace = Some(std::time::Instant::now());
        }
        for slot in 0..count / CUBE_FACES {
            if views.ready_mask & (1 << slot) == 0 {
                continue;
            }
            // MONKEY (moving fixture): same reasoning as `torch_cache_plan` — a live slot's mesh
            // id is stable by design, so the id comparison would veto every rebuild the plan just
            // granted it and freeze the map at the fixture's first position.
            let moving = views.moving_mask & (1 << slot) != 0;
            let rebuild = draw.rebuild_mask & (1 << slot) != 0
                && (moving || cached[slot] != views.caster_meshes[slot]);
            let dynamic = views.dynamic_mask & (1 << slot) != 0;
            let live_rank = dynamic.then(|| TorchShadowViews::live_rank(views.dynamic_mask, slot));
            // MONKEY (static torch cache): render static once, then copy to the disjoint live
            // bank BEFORE each entity pass. GreaterEqual + Load preserves the nearest surface.
            for face in 0..CUBE_FACES {
                let layer = slot * CUBE_FACES + face;
                if rebuild {
                    let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                        label: Some("torch_static_rebuild"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                            view: &layers.views[layer],
                            depth_ops: Some(Operations {
                                load: LoadOp::Clear(0.0),
                                store: StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    pass.set_render_pipeline(pipeline);
                    pass.set_bind_group(0, &draw.bind_groups[layer], &[]);
                    draw_torch_mesh(&mut pass, views.caster_meshes[slot], meshes, allocator);
                }
                if let Some(rank) = live_rank {
                    // MONKEY (live bank rank): matrices remain slot-addressed; only the copy
                    // destination/overlay attachment follows count.z's compact ready-set rank.
                    let live = MAX_TORCH_LAYERS + rank * CUBE_FACES + face;
                    render_context.command_encoder().copy_texture_to_texture(
                        TexelCopyTextureInfo {
                            texture: &gpu.texture,
                            mip_level: 0,
                            origin: Origin3d {
                                x: 0,
                                y: 0,
                                z: layer as u32,
                            },
                            aspect: TextureAspect::DepthOnly,
                        },
                        TexelCopyTextureInfo {
                            texture: &gpu.texture,
                            mip_level: 0,
                            origin: Origin3d {
                                x: 0,
                                y: 0,
                                z: live as u32,
                            },
                            aspect: TextureAspect::DepthOnly,
                        },
                        Extent3d {
                            width: TORCH_MAP_EDGE,
                            height: TORCH_MAP_EDGE,
                            depth_or_array_layers: 1,
                        },
                    );
                    let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                        label: Some("torch_dynamic_overlay"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                            view: &layers.views[live],
                            depth_ops: Some(Operations {
                                load: LoadOp::Load,
                                store: StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    pass.set_render_pipeline(pipeline);
                    pass.set_bind_group(0, &draw.bind_groups[layer], &[]);
                    draw_torch_mesh(&mut pass, views.entity_mesh, meshes, allocator);
                }
            }
            if rebuild {
                cached[slot] = views.caster_meshes[slot];
            }
            if trace_on && (trace_due || rebuild) {
                info!(
                    "torch-cache: slot {slot} static {} dynamic {} live_rank {:?} live_base {:?}",
                    if rebuild { "rebuilt" } else { "cached" },
                    if dynamic { "yes" } else { "no" },
                    live_rank,
                    live_rank.map(|rank| MAX_TORCH_LAYERS + rank * CUBE_FACES)
                );
            }
        }
        Ok(())
    }
}

/// Wire the ALWAYS-ON half (MONKEY, Phase 3A): the extract plugins for the shared image + table
/// (so the render world can look up the `GpuImage` and write the buffer), and the per-frame table
/// upload. Registered by the asset foundation next to the shared light buffer — NOT gated on the
/// retained pass (`static_gx::enabled`), because every model material binds these whether or not
/// static_gx draws. With the depth node off the table reads count 0 and the image stays at its
/// zero-initialised (= reverse-Z far ⇒ unshadowed) contents, so entities simply receive no shadow.
pub(crate) fn register_shared(app: &mut App) {
    app.add_plugins((
        ExtractResourcePlugin::<TorchDepthImage>::default(),
        ExtractResourcePlugin::<SharedTorchBuffer>::default(),
    ));
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.add_systems(
            Render,
            upload_torch_table.in_set(RenderSystems::PrepareResources),
        );
    }
}

/// Wire the render half: the persistent targets (RenderStartup), the per-frame view bind groups, and
/// the depth node between the prepass and the main opaque pass.
pub(super) fn build(app: &mut App) {
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    render_app
        .init_resource::<TorchLayerViews>()
        .init_resource::<TorchStaticCache>()
        .init_resource::<TorchDepthDraw>()
        .init_resource::<TorchCacheTrace>()
        .add_systems(RenderStartup, init_torch_depth)
        .add_systems(
            Render,
            // MONKEY (static torch cache): ready/dynamic masks must be final before BOTH
            // the shared storage upload and static_gx's PrepareBindGroups uniform pack.
            prepare_torch_depth
                .in_set(RenderSystems::PrepareResources)
                .before(upload_torch_table),
        )
        .add_render_graph_node::<ViewNodeRunner<TorchDepthNode>>(Core3d, TorchDepthLabel)
        .add_render_graph_edges(
            Core3d,
            // After all prepasses complete, before the main opaque pass — so the depth maps are
            // ready when `StaticGxNode` (between opaque and transparent) samples them.
            (
                Node3d::EndPrepasses,
                TorchDepthLabel,
                Node3d::MainOpaquePass,
            ),
        );
}

// MONKEY (static torch cache): planning does not commit residency. A missing upload cannot
// certify an empty map, and the two-rebuild budget also covers GPU texture recreation.
//
// MONKEY (moving fixture): `moving_mask`'s slots draw from a SEPARATE budget of
// [`TORCH_MOVING_MAX`], and they draw from it first. Two reasons it cannot be one shared budget:
// a moving slot asks for a rebuild EVERY frame (its mesh id changes every frame by construction),
// so sharing would let two moving fixtures own the budget forever and freeze every resident room's
// static map at whatever it last held; and a deferred moving rebuild is not a stale map but a
// wrong one — the table republishes its six projections from the live fixture position each frame
// while the depth still holds the old one, i.e. a shadow drawn from where the light no longer is.
fn torch_cache_plan<T: Copy + Eq>(
    cached: &[Option<T>],
    requested: &[Option<T>],
    dynamic_mask: u32,
    moving_mask: u32,
    mut mesh_ready: impl FnMut(T) -> bool,
) -> (u32, u32, u32) {
    let (mut ready, mut rebuild) = (0u32, 0u32);
    let mut budget = 2usize;
    let mut moving_budget = TORCH_MOVING_MAX;
    for (i, requested) in requested.iter().take(MAX_TORCH_MAPS).enumerate() {
        let Some(id) = *requested else { continue };
        let moving = moving_mask & (1 << i) != 0;
        // MONKEY (moving fixture): a live slot MUTATES its caster mesh in place and keeps its asset
        // id (an asset added this frame would not be certified resident until the next one, and the
        // slot would lose its ready bit — a zero weight — on every frame its fixture moved). So the
        // usual "same id ⇒ the cached depth is still good" shortcut is exactly wrong here: the id
        // is the same and the CONTENTS are a frame old. A live slot always takes the rebuild path.
        if !moving && cached[i] == Some(id) {
            ready |= 1 << i;
            continue;
        }
        let lane = if moving {
            &mut moving_budget
        } else {
            &mut budget
        };
        if *lane > 0 && mesh_ready(id) {
            *lane -= 1;
            rebuild |= 1 << i;
            ready |= 1 << i;
        }
    }
    // MONKEY (live bank rank): remove unready holes BEFORE ranking, exactly as table.count.z
    // does. Bound even malformed resource masks to eight cubes so no copy can escape the bank.
    let eligible = dynamic_mask & ready;
    let dynamic = (0..MAX_TORCH_MAPS).fold(0, |mask, slot| {
        if eligible & (1 << slot) != 0
            && TorchShadowViews::live_rank(eligible, slot) < MAX_TORCH_DYNAMIC
        {
            mask | (1 << slot)
        } else {
            mask
        }
    });
    (ready, rebuild, dynamic)
}

// MONKEY (static torch cache): an empty mesh is a valid all-clear cube. A nonempty mesh without
// allocator slices is still uploading, and MUST NOT become a permanently cached empty shadow.
fn torch_mesh_ready(
    id: AssetId<Mesh>,
    meshes: &RenderAssets<RenderMesh>,
    allocator: &MeshAllocator,
) -> bool {
    let Some(mesh) = meshes.get(id) else {
        return false;
    };
    match mesh.buffer_info {
        RenderMeshBufferInfo::Indexed { count, .. } => {
            count == 0
                || (allocator.mesh_vertex_slice(&id).is_some()
                    && allocator.mesh_index_slice(&id).is_some())
        }
        _ => false,
    }
}

fn draw_torch_mesh<'a>(
    pass: &mut bevy::render::render_phase::TrackedRenderPass<'a>,
    id: Option<AssetId<Mesh>>,
    meshes: &'a RenderAssets<RenderMesh>,
    allocator: &'a MeshAllocator,
) {
    let Some(id) = id else { return };
    let Some(mesh) = meshes.get(id) else { return };
    let RenderMeshBufferInfo::Indexed {
        index_format,
        count,
    } = mesh.buffer_info
    else {
        return;
    };
    if count == 0 {
        return;
    }
    let (Some(v), Some(i)) = (
        allocator.mesh_vertex_slice(&id),
        allocator.mesh_index_slice(&id),
    ) else {
        return;
    };
    pass.set_vertex_buffer(0, v.buffer.slice(..));
    pass.set_index_buffer(i.buffer.slice(..), index_format);
    pass.draw_indexed(
        i.range.clone(),
        i32::try_from(v.range.start).unwrap_or(0),
        0..1,
    );
}

impl super::StaticGx {
    /// MONKEY (static torch cache): fixture-local static collection uses batch bounds, not the
    /// building's placement origin. A distant Abbey room is still part of a WMO anchored more
    /// than 48 yards away. The key and the mesh MUST use this identical admission predicate.
    ///
    /// MONKEY (torch owner exclusion): returns how many in-range items were dropped as the
    /// FIXTURE'S OWN BODY — the `WOW_TORCH_TRACE` counter, and the one number that says whether
    /// this fix is doing anything for a given slot.
    pub fn append_torch_triangles(
        &self,
        center: Vec3,
        reach: f32,
        owner: Option<LightOwner>,
        positions: &mut Vec<[f32; 3]>,
        indices: &mut Vec<u32>,
    ) -> u32 {
        let mut excluded = 0;
        for cell in self
            .cells
            .values()
            .chain(self.wmos.values())
            .chain(self.props.values())
        {
            for item in &cell.items {
                if !torch_item_in_range(item, center, reach) {
                    continue;
                }
                if torch_item_is_emitter(item, center, owner) {
                    excluded += 1;
                    continue;
                }
                let base = positions.len() as u32;
                positions.extend(item.geometry.positions.iter().map(|p| {
                    item.transform
                        .transform_point(benilla_assets::coords::wow_to_bevy(*p))
                        .to_array()
                }));
                let added = positions.len() as u32 - base;
                for tri in item.geometry.indices.as_chunks::<3>().0 {
                    if tri.iter().all(|i| *i < added) {
                        indices.extend(tri.iter().map(|i| base + *i));
                    }
                }
            }
        }
        excluded
    }

    /// MONKEY (torch lane perf): a CHEAP stamp of "has the retained scene changed at all" —
    /// O(regions), where [`Self::torch_geometry_key`] is O(items) and is run per fixture.
    ///
    /// It folds each region's own `last_change` frame together with its population, which between
    /// them cover every way the source set of ANY fixture can move: an item is only ever pushed
    /// (`push_item` stamps `last_change`) or dropped (`release_owner` stamps it and changes the
    /// count), and a whole region arriving or being culled changes the fold because the region set
    /// itself is folded. Order-independent, so `HashMap` iteration order is invisible. A baker's
    /// in-place SORT of a region's items is deliberately not covered and does not need to be: the
    /// geometry key it guards is an order-independent sum too.
    ///
    /// This is a GATE, not a key: the app lane skips a fixture's expensive walk while this and the
    /// entity-part census are both unchanged, and still computes the real per-fixture key whenever
    /// either of them moves.
    pub fn torch_residency_generation(&self) -> u64 {
        let mut sum = 0u64;
        for cell in self
            .cells
            .values()
            .chain(self.wmos.values())
            .chain(self.props.values())
        {
            let mut hash = std::collections::hash_map::DefaultHasher::new();
            cell.last_change.hash(&mut hash);
            cell.items.len().hash(&mut hash);
            sum = sum.wrapping_add(hash.finish());
        }
        sum
    }

    /// MONKEY (static torch cache): residency fingerprint of the exact opaque source set used
    /// by append_torch_triangles. Order-independent per-item hashing ignores HashMap reordering
    /// and camera/portal/fader bookkeeping. Arrival, unload, replacement or transform edits in
    /// THIS fixture's sphere invalidate it even when neither camera nor fixture moves.
    pub fn torch_geometry_key(&self, center: Vec3, reach: f32, owner: Option<LightOwner>) -> u64 {
        let mut sum = 0u64;
        for cell in self
            .cells
            .values()
            .chain(self.wmos.values())
            .chain(self.props.values())
        {
            for item in &cell.items {
                if !torch_item_in_range(item, center, reach) {
                    continue;
                }
                if torch_item_is_emitter(item, center, owner) {
                    continue;
                }
                let mut hash = std::collections::hash_map::DefaultHasher::new();
                // MONKEY (static torch cache): source residency stamp prevents allocator address
                // reuse after unload/reload from impersonating the previous geometry Arc.
                cell.last_change.hash(&mut hash);
                (std::sync::Arc::as_ptr(&item.geometry) as usize).hash(&mut hash);
                item.transform
                    .to_matrix()
                    .to_cols_array()
                    .map(f32::to_bits)
                    .hash(&mut hash);
                sum = sum.wrapping_add(hash.finish());
            }
        }
        sum
    }
}

/// MONKEY (torch owner exclusion): EMITTER-OWNER EXCLUSION — is this item the fixture's own body?
///
/// Three tests, cheapest and most exact first. Only the first is IDENTITY; the other two are the
/// fallback for a light whose owner could not be plumbed (a server-placed GameObject fire, a light
/// whose placement tag has not landed yet) and for one whose own model is split across placements.
///
///  1. OWNERSHIP. `owner` names a placement and this item belongs to it — see [`LightOwner`]. This
///     is the case the lantern-on-a-post and the tall wall torch need: their bounds reach the
///     ground, so no proximity rule of any radius could have found them.
///  2. CONTAINMENT of the FLAME. The item's own bounds hold the light position (padded by
///     [`TORCH_SELF_PAD`]) and the item is FIXTURE-SIZED ([`TORCH_SELF_MAX_EXTENT`]). A lamp
///     housing contains its own flame; a crate a yard and a half away does not. The size cap is
///     load-bearing, not tidiness — see the constant.
///  3. The DEGENERATE case: the item's whole bounding sphere lies within [`TORCH_SELF_TINY`] of the
///     flame, i.e. it is a flame card or a helper sitting on top of the light and can cast nothing
///     but self-shadow. This is all that survives of the old 2.5-yd proximity rule.
///
/// An item with no bounds is never excluded by 2 or 3: the failure direction of a heuristic here
/// must be "keeps a caster", not "loses a wall".
fn torch_item_is_emitter(item: &super::GxItem, center: Vec3, owner: Option<LightOwner>) -> bool {
    if owner.is_some_and(|o| o.owns(&item.object)) {
        return true;
    }
    item.local_aabb.is_some_and(|aabb| {
        let (c, h) = (Vec3::from(aabb.center), Vec3::from(aabb.half_extents));
        torch_bound_contains(&item.transform, c, h, center)
            || torch_bound_contained(&item.transform, c, h, center, TORCH_SELF_TINY)
    })
}

/// MONKEY (torch owner exclusion): does this bound CONTAIN the flame? Tested in MODEL space (the
/// inverse transform), so a rotated lamp post is measured against its own box rather than against
/// the circumsphere a world-space test would have to use — the difference between "the lantern
/// housing" and "a 3-yd ball centred on the lantern" that would swallow the porch it hangs from.
/// The size cap is applied FIRST and on the cheap sphere radius, so the inverse is only ever paid
/// by the handful of fixture-sized items standing at a flame.
///
/// `pub` and taking an `Affine3A` because BOTH caster lanes need the identical rule and they hold
/// their pose in different shapes: the retained lane has a `Transform`, the entity lane a
/// `GlobalTransform`. Two copies of a heuristic that decides "does this cast" would drift.
pub fn torch_flame_inside_bounds(
    world_from_local: Affine3A,
    local_center: Vec3,
    half_extents: Vec3,
    light: Vec3,
) -> bool {
    let m = world_from_local.matrix3;
    // The basis columns' lengths ARE the scale, whatever rotation is folded in with them.
    let scale = Vec3::new(m.x_axis.length(), m.y_axis.length(), m.z_axis.length());
    if (half_extents * scale).length() > TORCH_SELF_MAX_EXTENT {
        return false;
    }
    let local = world_from_local.inverse().transform_point3(light);
    // A world-space pad has to be expressed in model units, hence the divide; the floor keeps a
    // degenerate (zero-scale) placement from producing an infinite pad that contains the world.
    let pad = Vec3::splat(TORCH_SELF_PAD) / scale.max(Vec3::splat(1e-4));
    (local - local_center).abs().cmple(half_extents + pad).all()
}

/// The retained lane's shape of [`torch_flame_inside_bounds`].
fn torch_bound_contains(
    transform: &Transform,
    local_center: Vec3,
    half_extents: Vec3,
    light: Vec3,
) -> bool {
    torch_flame_inside_bounds(
        transform.compute_affine(),
        local_center,
        half_extents,
        light,
    )
}

/// The proximity half of [`torch_item_is_emitter`], split out exactly as `torch_bound_in_range`
/// is: the transformed bounding sphere lies WHOLLY inside `radius` of the fixture. `radius <= 0`
/// contains nothing.
fn torch_bound_contained(
    transform: &Transform,
    local_center: Vec3,
    half_extents: Vec3,
    center: Vec3,
    radius: f32,
) -> bool {
    if radius <= 0.0 {
        return false;
    }
    let origin = transform.transform_point(local_center);
    let r = (half_extents * transform.scale.abs()).length();
    origin.distance(center) + r <= radius
}

// MONKEY (static torch cache): a conservative transformed bounding sphere, independent of
// portal visibility. Missing bounds admit the source rather than silently losing a wall.
fn torch_item_in_range(item: &super::GxItem, center: Vec3, reach: f32) -> bool {
    !item.cutout
        && item.local_aabb.is_none_or(|aabb| {
            torch_bound_in_range(
                &item.transform,
                aabb.center.into(),
                aabb.half_extents.into(),
                center,
                reach,
            )
        })
}

fn torch_bound_in_range(
    transform: &Transform,
    local_center: Vec3,
    half_extents: Vec3,
    center: Vec3,
    reach: f32,
) -> bool {
    let origin = transform.transform_point(local_center);
    let radius = (half_extents * transform.scale.abs()).length();
    origin.distance_squared(center) <= (reach + radius) * (reach + radius)
}

#[cfg(test)]
mod tests {
    use super::*;

    // MONKEY (static torch cache): unchanged owners do no work; replacement, streaming,
    // absent uploads, sparse holes and texture recreation cannot certify stale depth.
    #[test]
    fn cache_plan_reuses_and_limits_rebuilds() {
        let cached = [Some(10), Some(11), None, Some(13)];
        assert_eq!(
            torch_cache_plan(&cached, &cached, 0xf, 0, |_| false),
            (0b1011, 0, 0b1011)
        );
        let requested = [Some(20), Some(11), Some(12), Some(23)];
        assert_eq!(
            torch_cache_plan(&cached, &requested, 0xf, 0, |_| true),
            (0b0111, 0b0101, 0b0111)
        );
        assert_eq!(
            torch_cache_plan(&cached, &requested, 0xf, 0, |id| id == 23),
            (0b1010, 0b1000, 0b1010)
        );
        assert_eq!(
            torch_cache_plan(&cached, &[None, Some(11)], 0xf, 0, |_| true),
            (0b10, 0, 0b10)
        );
        assert_eq!(
            torch_cache_plan(&[None; 4], &requested, 0, 0, |_| true),
            (0b11, 0b11, 0)
        );
    }

    // MONKEY (moving fixture): a moving slot rebuilds out of its OWN budget, so two moving
    // fixtures cannot consume the resident slots' two - the failure that would freeze a room's
    // static shadows for as long as a pet walked around in it.
    #[test]
    fn moving_slots_rebuild_on_a_separate_budget() {
        let cached = [None, None, None, None];
        let requested = [Some(1), Some(2), Some(3), Some(4)];
        // No moving slots: the historical two-rebuild ceiling, untouched.
        assert_eq!(
            torch_cache_plan(&cached, &requested, 0, 0, |_| true).1,
            0b0011
        );
        // Slots 0 and 1 moving: they take the moving budget and slots 2/3 still get the static one.
        assert_eq!(
            torch_cache_plan(&cached, &requested, 0, 0b0011, |_| true).1,
            0b1111
        );
        // Three moving: the third waits (it fast-fades app-side rather than showing a wrong map).
        assert_eq!(
            torch_cache_plan(&cached, &requested, 0, 0b0111, |_| true).1,
            0b1011
        );
    }

    // MONKEY (shadow floor): the two dials share one word; neither may corrupt the other, and a
    // strength of 0 (shadows off) must survive the pack as a real 0 rather than a floored 1.
    #[test]
    fn count_y_packs_soft_low_and_strength_high() {
        for (soft, strength, soft_pct, strength_pct) in [
            (1.5f32, 0.7f32, 150u32, 70u32),
            (3.0, 1.0, 300, 100),
            (0.5, 0.0, 50, 0),
            // Out-of-range inputs are clamped, never wrapped into the other half.
            (0.0, 2.0, 100, 100),
            (0.5, -1.0, 50, 0),
        ] {
            let views = TorchShadowViews {
                count: 1,
                soft,
                strength,
                ..Default::default()
            };
            let word = TorchTableUniform::pack(Some(&views)).soft_strength;
            assert_eq!(word & 0xffff, soft_pct, "soft {soft}");
            assert_eq!(word >> 16, strength_pct, "strength {strength}");
        }
    }

    #[test]
    fn live_bank_rank_matches_count_one_bits() {
        // MONKEY (live bank rank): independently enumerate lower set bits, including holes,
        // slot 15 and eight sparse overlays; every face must fit the compact allocation.
        for mask in [0u32, 1, 1 << 15, 0x8085, 0xaaaa, 0xff] {
            for slot in 0..MAX_TORCH_MAPS {
                let expected = (0..slot).filter(|bit| mask & (1 << bit) != 0).count();
                let rank = TorchShadowViews::live_rank(mask, slot);
                assert_eq!(rank, expected, "mask {mask:#x}, slot {slot}");
                if mask & (1 << slot) != 0 {
                    assert!(MAX_TORCH_LAYERS + rank * CUBE_FACES + 5 < TORCH_TEXTURE_LAYERS);
                }
            }
        }
        let cached = [Some(1); MAX_TORCH_MAPS];
        let (_, _, capped) = torch_cache_plan(&cached, &cached, 0xffff, 0, |_| false);
        assert_eq!(capped, 0xff);
        let mut requested = cached;
        requested[0] = None;
        let (ready, _, dynamic) = torch_cache_plan(&cached, &requested, 0x8001, 0, |_| false);
        let views = TorchShadowViews {
            count: 16,
            ready_mask: ready,
            dynamic_mask: dynamic,
            ..Default::default()
        };
        let table = TorchTableUniform::pack(Some(&views));
        assert_eq!(dynamic, 0x8000);
        assert_eq!(table.dynamic_mask, dynamic);
        assert_eq!(TorchShadowViews::live_rank(table.dynamic_mask, 15), 0);
    }

    #[test]
    fn torch_bounds_include_remote_building_batches() {
        let transform = Transform::from_xyz(200.0, 0.0, 0.0).with_scale(Vec3::splat(2.0));
        assert!(torch_bound_in_range(
            &transform,
            Vec3::new(-100.0, 0.0, 0.0),
            Vec3::splat(10.0),
            Vec3::ZERO,
            48.0
        ));
        assert!(!torch_bound_in_range(
            &transform,
            Vec3::ZERO,
            Vec3::splat(10.0),
            Vec3::ZERO,
            48.0
        ));
    }

    // MONKEY (static torch cache): protect uniform/storage byte agreement and sparse-slot bank
    // selection, including the highest slot and an upload that has not reached the GPU yet.
    #[test]
    fn torch_table_layout_and_readiness() {
        assert_eq!(std::mem::size_of::<TorchTableUniform>(), 6416);
        assert_eq!(std::mem::offset_of!(TorchTableUniform, positions), 16);
        assert_eq!(std::mem::offset_of!(TorchTableUniform, view_projs), 272);
        let mut views = TorchShadowViews {
            count: 16,
            dynamic_mask: (1 << 15) | 1,
            ready_mask: 1 << 15,
            ..Default::default()
        };
        views.positions[0] = Vec4::ONE;
        views.positions[15] = Vec4::ONE;
        let table = TorchTableUniform::pack(Some(&views));
        assert_eq!(table.dynamic_mask, 1 << 15);
        assert_eq!(table.positions[0][3], 0.0);
        assert_eq!(table.positions[15][3], 1.0);
        assert_eq!(MAX_TORCH_LAYERS, 96);
        assert_eq!(MAX_TORCH_DYNAMIC, 8);
        assert_eq!(TORCH_TEXTURE_LAYERS, 144);
        // MONKEY (outdoor torch shadows): the exterior lane rides `count.w`, the word that was
        // reserved padding. The SIZE must not move with it — a table that disagrees with either
        // WGSL copy blanks every building and model on screen.
        assert_eq!(std::mem::offset_of!(TorchTableUniform, flags), 12);
        assert_eq!(table.flags, 0, "not exterior by default");
        views.exterior = true;
        assert_eq!(TorchTableUniform::pack(Some(&views)).flags, TORCH_EXT_LANE);
        assert_eq!(
            TorchTableUniform::pack(None).flags,
            0,
            "no resource = no lane"
        );
    }

    // MONKEY (outdoor torch shadows): EMITTER-OWNER EXCLUSION. A campfire's own logs must leave
    // its caster gather (they are the nearest occluder on all six faces and would black out the
    // fire's own pool); a crate beside it must not. The test is CONTAINMENT of the whole bounding
    // sphere, so "near the fire" is not enough to be excluded — which is what keeps a long wall or
    // a fence that merely passes the fire casting normally.
    #[test]
    fn only_the_fire_s_own_body_leaves_its_caster_gather() {
        let excluded = |x: f32, half: f32, radius: f32| {
            torch_bound_contained(
                &Transform::from_xyz(x, 0.0, 0.0),
                Vec3::ZERO,
                Vec3::splat(half),
                Vec3::ZERO,
                radius,
            )
        };
        // The campfire prop itself: centred on the flame, ~1 yd across.
        assert!(excluded(0.2, 0.5, 2.5));
        // A crate two yards off: its centre is inside 2.5, its sphere is not.
        assert!(!excluded(2.0, 0.6, 2.5));
        // A wall passing right by the fire: a huge sphere, never contained.
        assert!(!excluded(0.0, 30.0, 2.5));
        // The INTERIOR lane passes 0 and excludes nothing at all - its gather is untouched.
        assert!(!excluded(0.0, 0.1, 0.0));
        // Scale is applied to the extents, exactly as `torch_bound_in_range` applies it.
        let scaled = Transform::from_xyz(0.0, 0.0, 0.0).with_scale(Vec3::splat(4.0));
        assert!(!torch_bound_contained(
            &scaled,
            Vec3::ZERO,
            Vec3::splat(0.5),
            Vec3::ZERO,
            2.5
        ));
    }

    // MONKEY (torch owner exclusion): the IDENTITY key. Both caster lanes have to agree on it while
    // holding the placement identity in two different shapes — the retained `GxItem` owns an `Arc`,
    // the entity path's parts own a CLONE — so the comparison must be by CONTENT, never by pointer.
    // And WMO props all share their building's uniqueId, which is why the model path is in the key:
    // without it a candelabra's light would drop every barrel and chair in the building.
    #[test]
    fn a_light_owner_names_one_placement_by_content() {
        let lantern = WorldObject {
            kind: ModelKind::Doodad,
            label: "world/generic/lantern01.m2".into(),
            id: 4242,
            detail: "emitters: 1".into(),
        };
        let owner = LightOwner::placement(&lantern);
        // The entity lane's CLONE of the same identity still matches (a pointer key would not).
        assert!(owner.owns(&lantern.clone()));
        // Same building, different prop model — a WMO prop shares its building's uniqueId.
        let barrel = WorldObject {
            label: "world/generic/barrel01.m2".into(),
            ..lantern.clone()
        };
        assert!(!owner.owns(&barrel));
        // Same model, different placement: the lantern down the street still casts.
        let other = WorldObject {
            id: 4243,
            ..lantern.clone()
        };
        assert!(!owner.owns(&other));
        // The BUILDING itself is a different subsystem, so a prop light can never delete its shell.
        let building = WorldObject {
            kind: ModelKind::Wmo,
            ..lantern.clone()
        };
        assert!(!owner.owns(&building));
        // An entity-hosted light names a frame, and never claims a placement by accident.
        assert!(!LightOwner::Instance(Entity::from_raw_u32(7).unwrap()).owns(&lantern));
        assert_eq!(owner.instance(), None);
    }

    // MONKEY (torch owner exclusion): the CONTAINMENT fallback, for a light whose owner could not
    // be plumbed. The housing that holds the flame is excluded; the crate beside it is not; and —
    // the rule that keeps the shipped interior lane intact — a room-sized batch is never excluded
    // however deep inside it the candle sits.
    #[test]
    fn only_a_fixture_sized_bound_may_contain_its_own_flame() {
        let at = |x: f32, y: f32| Transform::from_xyz(x, y, 0.0);
        // A lantern housing 3 yd up a post: the placement's box runs from the ground to the lamp,
        // so its CENTRE is 1.5 yd below the flame and no proximity rule of any radius finds it —
        // the exact case `TORCH_EXT_SELF_EXCLUDE = 2.5` could not exclude.
        let post = Vec3::new(0.0, 1.5, 0.0);
        let post_half = Vec3::new(0.4, 1.6, 0.4);
        let flame = Vec3::new(0.0, 3.0, 0.0);
        assert!(torch_bound_contains(&at(0.0, 0.0), post, post_half, flame));
        assert!(
            !torch_bound_contained(&at(0.0, 0.0), post, post_half, flame, TORCH_SELF_TINY),
            "the retired proximity rule never saw it"
        );
        // The crate 1.5 yd away keeps casting — that shadow is the whole point of the feature.
        assert!(!torch_bound_contains(
            &at(1.5, 0.3),
            Vec3::ZERO,
            Vec3::splat(0.5),
            flame
        ));
        // A room-sized wall batch holds the candle but is NOT fixture-sized: excluding it would
        // delete the room's shadows and leak the candle through its own walls.
        assert!(!torch_bound_contains(
            &at(0.0, 0.0),
            Vec3::ZERO,
            Vec3::splat(20.0),
            flame
        ));
        assert!(
            !torch_bound_contains(
                &at(0.0, 0.0),
                Vec3::ZERO,
                Vec3::splat(TORCH_SELF_MAX_EXTENT),
                flame
            ),
            "at the cap, still a building"
        );
        // The pad reaches a flame sitting just proud of its housing, and stops well short of the
        // next prop along.
        let head = Vec3::new(0.0, 0.0, 0.0);
        let head_half = Vec3::splat(0.25);
        assert!(torch_bound_contains(
            &at(0.0, 0.0),
            head,
            head_half,
            Vec3::new(0.0, 0.5, 0.0)
        ));
        assert!(!torch_bound_contains(
            &at(0.0, 0.0),
            head,
            head_half,
            Vec3::new(0.0, 0.9, 0.0)
        ));
        // Scale applies to the box AND to the pad (a world-space pad in model units).
        let big = Transform::from_xyz(0.0, 0.0, 0.0).with_scale(Vec3::splat(4.0));
        assert!(torch_bound_contains(
            &big,
            Vec3::ZERO,
            Vec3::splat(0.3),
            Vec3::new(0.0, 1.2, 0.0)
        ));
        assert!(
            !torch_bound_contains(&big, Vec3::ZERO, Vec3::splat(2.0), Vec3::ZERO),
            "scaled past the fixture cap"
        );
        // A rotated post is tested against its own box, not its circumsphere.
        let spun = Transform::from_xyz(0.0, 0.0, 0.0)
            .with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2));
        assert!(torch_bound_contains(
            &spun,
            Vec3::new(0.0, 1.5, 2.0),
            post_half,
            Vec3::new(2.0, 3.0, 0.0)
        ));
        assert!(!torch_bound_contains(
            &spun,
            Vec3::new(0.0, 1.5, 2.0),
            post_half,
            flame
        ));
    }
}
