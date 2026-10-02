//! The dressing-room booth, the reference's `DressUpFrame`/`DressUpModel`: the player's own
//! character wearing items they do not own. Nobody in the world wears the preview, so it is built
//! from a spec by the shared assembly ([`crate::entities::attach`]), not mirrored from a live
//! entity, and it dresses by the select-screen law (weapons in hand, `0x47a0c0`). `DressUpModel`
//! (`0x495c00`) shares the `CharacterModelBase` ctor (`0x505680`) with the character window's
//! `<PlayerModel>`, so it lights through [`BoothLight::pane`] with no glow, as the paper doll does.

use benilla_protocol::CharEnumItem;
use bevy::camera::visibility::RenderLayers;
use bevy::camera::PerspectiveProjection;
use bevy::prelude::*;

use crate::entities::Creatures;
use benilla_assets::materials::WowModelMaterial;

use super::light::BoothLight;
use super::{
    aim, body_frame, booth_anchors, new_target_image, spawn_booth_effects, spawn_booth_model,
    wake_booth, Booth, BoothBillboardSpec, BoothCam, BoothEffects, BoothInstance, BoothMotion,
    BoothPart, BoothRider, BoothTwins, Booths, PortraitImages, PortraitSource, PreviewBillboard,
    PreviewEffects, PreviewPart, PreviewRider, BOOTH_SETTLE_FRAMES, DRESSUP_LAYER,
    DRESSUP_POOL_LAYER_BASE, PAPERDOLL_SIZE,
};

/// The dressing-room booth's key in [`PortraitImages`] and [`Booths`].
pub(crate) const DRESSUP_SLOT: &str = "dressup";

/// The player's own body and appearance wearing [`crate::ui_dressup`]'s equipment array (their
/// visible items with the tried-on ones substituted); a change in it re-assembles the booth.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct DressUpLook {
    /// The player's own body display id (the reference's `SetUnit("player")`).
    pub(crate) display_id: u32,
    pub(crate) race: u8,
    pub(crate) sex: u8,
    pub(crate) skin: u8,
    pub(crate) face: u8,
    pub(crate) hair_style: u8,
    pub(crate) hair_color: u8,
    pub(crate) facial_hair: u8,
    /// `ItemDisplayInfo` ids by equipment slot, in the `SMSG_CHAR_ENUM` shape (helm 0, main hand
    /// 15, off hand 16, ranged 17, tabard 18).
    pub(crate) equipment: [CharEnumItem; 19],
    /// The player's own guild crest, so a tried-on Guild Tabard shows it, not the blank default.
    pub(crate) emblem: Option<benilla_formats::GuildEmblem>,
}

/// The dressing room's input: the look (`None` empties the booth) and the yaw in radians, the
/// reference's `Model:SetRotation` driven by the rotate buttons.
#[derive(Resource)]
pub(crate) struct DressUpPreview {
    pub(crate) look: Option<DressUpLook>,
    pub(crate) yaw: f32,
}

impl Default for DressUpPreview {
    fn default() -> Self {
        Self {
            look: None,
            // `Model_OnLoad`'s default facing (`UIParent.lua:1422`).
            yaw: 0.61,
        }
    }
}

/// The assembled dressing-room parts. `revision` bumps on a fresh assembly or a clear; the booth
/// re-bakes only when it moves, never on a bare yaw change.
#[derive(Resource, Default)]
pub(crate) struct DressUpBake {
    pub(crate) look: Option<DressUpLook>,
    pub(crate) display_id: u32,
    pub(crate) parts: Vec<PreviewPart>,
    pub(crate) riders: Vec<PreviewRider>,
    pub(crate) effects: Vec<PreviewEffects>,
    pub(crate) billboards: Vec<PreviewBillboard>,
    pub(crate) grip: [bool; 2],
    pub(crate) revision: u64,
}

/// How many unclaimed `<DressUpModel>` panes draw at once: Turtle's transmog page shows its doll
/// and fifteen item tiles. A pane past the pool draws nothing.
pub(crate) const DRESSUP_POOL: usize = 16;

/// The key of pool booth `i` in [`PortraitImages`], [`Booths`] and [`super::BoothPanes`].
pub(crate) fn pool_slot(i: usize) -> String {
    format!("{DRESSUP_SLOT}{i}")
}

/// A pane's view in `SetPosition` units: where the body stands now and the root camera 1 was
/// frozen through ([`benilla_ui::widget::ModelState::camera_root`]).
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct PaneView {
    pub(crate) position: Vec3,
    pub(crate) scale: f32,
    pub(crate) camera_position: Vec3,
    pub(crate) camera_scale: f32,
}

impl PaneView {
    /// The camera offset and the body root, in camera-1 model units. The widget's root is
    /// `T(pos·L)·R(facing)·S(G·modelScale·L)` (`0x76d1a0`, `G` the
    /// [`super::framing::pane_model_scale`], `L` the effective scale) and camera 1 was published
    /// through it with the facing zeroed (`0x505890`); dividing both by the frozen scale leaves
    /// `L` out.
    pub(super) fn rig(&self, g: f32, yaw: f32) -> (Vec3, Transform) {
        let s0 = (g * self.camera_scale).max(1e-4);
        let to_bevy = |p: Vec3| benilla_assets::coords::wow_to_bevy((p / s0).to_array());
        let root = Transform {
            translation: to_bevy(self.position),
            rotation: Quat::from_rotation_y(yaw),
            scale: Vec3::splat(self.scale / self.camera_scale.max(1e-4)),
        };
        (to_bevy(self.camera_position), root)
    }
}

/// One pool booth's pane and its look, assembly and bake memory.
#[derive(Default)]
pub(crate) struct PaneDress {
    /// The `<DressUpModel>` this booth draws; `None` while free.
    pub(crate) pane: Option<benilla_ui::widget::FrameHandle>,
    pub(crate) preview: DressUpPreview,
    pub(crate) view: Option<PaneView>,
    pub(crate) bake: DressUpBake,
    /// The assembly's memory ([`crate::entities`]): the look last built, and whether it finished.
    pub(crate) built_look: Option<DressUpLook>,
    pub(crate) built: bool,
    /// What [`sync_dressup_booth`] last staged, and whether a bake stands.
    synced: Option<Synced>,
    staged: bool,
}

/// The pane dressing rooms, one booth each ([`DRESSUP_POOL`]).
#[derive(Resource)]
pub(crate) struct PaneDressUps(pub(crate) Vec<PaneDress>);

impl Default for PaneDressUps {
    fn default() -> Self {
        Self((0..DRESSUP_POOL).map(|_| PaneDress::default()).collect())
    }
}

impl PaneDressUps {
    /// The pool booth drawing `pane`.
    pub(crate) fn slot_of(&self, pane: benilla_ui::widget::FrameHandle) -> Option<usize> {
        self.0.iter().position(|d| d.pane == Some(pane))
    }

    /// The booth of `pane`, or a free one; `None` when the pool is full.
    pub(crate) fn claim(&mut self, pane: benilla_ui::widget::FrameHandle) -> Option<usize> {
        if let Some(i) = self.slot_of(pane) {
            return Some(i);
        }
        let i = self.0.iter().position(|d| d.pane.is_none())?;
        self.0[i].pane = Some(pane);
        Some(i)
    }

    /// Free the booth of `pane` and empty its stage.
    pub(crate) fn release(&mut self, pane: benilla_ui::widget::FrameHandle) {
        if let Some(i) = self.slot_of(pane) {
            let d = &mut self.0[i];
            d.pane = None;
            d.preview.look = None;
            d.view = None;
        }
    }
}

/// Stand the dressing-room booths up, the stock room and every pool booth: a
/// [`PAPERDOLL_SIZE`]² target with no glow on its own layer, framed per bake, and transparent.
pub(super) fn spawn_dressup_booths(
    commands: &mut Commands,
    images: &mut Assets<Image>,
    portraits: &mut PortraitImages,
    booths: &mut Booths,
) {
    let pool = (0..DRESSUP_POOL).map(|i| (pool_slot(i), DRESSUP_POOL_LAYER_BASE + i));
    for (slot, layer) in std::iter::once((DRESSUP_SLOT.to_string(), DRESSUP_LAYER)).chain(pool) {
        spawn_dressup_booth(commands, images, portraits, booths, slot, layer);
    }
}

fn spawn_dressup_booth(
    commands: &mut Commands,
    images: &mut Assets<Image>,
    portraits: &mut PortraitImages,
    booths: &mut Booths,
    slot: String,
    layer_index: usize,
) {
    let image = images.add(new_target_image(PAPERDOLL_SIZE));
    portraits
        .0
        .insert(slot.clone(), PortraitSource::Live(image.clone()));
    let layer = RenderLayers::layer(layer_index);
    let root = commands
        .spawn((Transform::IDENTITY, Visibility::Visible, layer.clone()))
        .id();
    commands.spawn((
        super::booth_view_shape(),
        Camera {
            order: -100 + layer_index as isize,
            // Transparent: `<DressUpModel>` draws only its model, and the room behind it is the
            // window's `DressUpBackground-<Race>` art at a lower frame level, which only
            // compositing can show through.
            clear_color: ClearColorConfig::Custom(Color::NONE),
            ..default()
        },
        bevy::camera::RenderTarget::Image(image.clone().into()),
        benilla_world::ffx_glow::FfxGlow::UI_PANE,
        // Placeholder: `sync_dressup_booth` frames it from the body's bounds on the first bake.
        Projection::from(PerspectiveProjection {
            fov: super::PORTRAIT_FOV,
            near: 0.02,
            far: 100.0,
            ..default()
        }),
        layer.clone(),
        BoothCam(slot.clone()),
    ));
    booths.0.insert(
        slot,
        Booth {
            layer,
            root,
            target: image,
            baked: None,
            baked_guid: None,
            snap: None,
            shown: false,
            show_rev: 0,
            wake: 0,
            live: false,
            pending: Vec::new(),
            pending_since: None,
            pipes_settling: false,
            pipes_since: None,
            aspect: 1.0,
            rigged: false,
            parked: false,
            turn: super::Turn::default(),
        },
    );
}

/// What a dressing-room booth last staged: the bake revision, the yaw and the pane view.
#[derive(Clone, Copy, PartialEq)]
pub(super) struct Synced {
    revision: u64,
    yaw: f32,
    view: Option<PaneView>,
}

/// The shared inputs of the dressing rooms [`sync_dressup_booth`] stages.
#[derive(bevy::ecs::system::SystemParam)]
pub(super) struct RoomStage<'w, 's> {
    commands: Commands<'w, 's>,
    booths: ResMut<'w, Booths>,
    framing: super::BoothFraming<'w>,
    booth_light: ResMut<'w, BoothLight>,
    materials: ResMut<'w, Assets<WowModelMaterial>>,
    creatures: Option<Res<'w, Creatures>>,
    anim_data: Option<Res<'w, crate::creature_anim::AnimData>>,
    cams: Query<
        'w,
        's,
        (
            &'static BoothCam,
            &'static mut Transform,
            &'static mut Projection,
        ),
    >,
    palettes: ResMut<'w, benilla_world::rig_palette::RigPalettes>,
}

/// Bake each dressing room's assembled look as [`super::sync_body_booth`] does, and spin it to its
/// pane's yaw: the stock room, then every pool booth. The assembly holds the weapons, so the hands
/// close on them (`CloseHand` `0x479660`).
pub(super) fn sync_dressup_booth(
    mut stage: RoomStage,
    preview: Res<DressUpPreview>,
    bake: Res<DressUpBake>,
    mut pool: ResMut<PaneDressUps>,
    mut env_cache: Local<Option<bool>>,
    mut last: Local<Option<Synced>>,
    // Whether a bake stands on the stage; `Booth::baked` keys the mirrored booths, not this one.
    mut staged: Local<bool>,
) {
    if super::test_mode(&mut env_cache) {
        return; // the test bake owns the booths
    }
    sync_room(
        &mut stage,
        DRESSUP_SLOT,
        preview.yaw,
        &bake,
        None,
        &mut last,
        &mut staged,
    );
    for (i, d) in pool.0.iter_mut().enumerate() {
        let PaneDress {
            preview,
            view,
            bake,
            synced,
            staged,
            ..
        } = d;
        sync_room(
            &mut stage,
            &pool_slot(i),
            preview.yaw,
            bake,
            *view,
            synced,
            staged,
        );
    }
}

fn sync_room(
    stage: &mut RoomStage,
    slot: &str,
    yaw: f32,
    bake: &DressUpBake,
    view: Option<PaneView>,
    last: &mut Option<Synced>,
    staged: &mut bool,
) {
    let RoomStage {
        commands,
        booths,
        framing,
        booth_light,
        materials,
        creatures,
        anim_data,
        cams,
        palettes,
    } = stage;
    let Some(booth) = booths.0.get_mut(slot) else {
        return;
    };
    // The pane's aspect, latched while on screen: the dressing room's is 316×351, not square.
    let aspect = framing.panes.0.get(slot).copied().unwrap_or(booth.aspect);
    let now = Synced {
        revision: bake.revision,
        yaw,
        view,
    };
    let rebake = last.is_none_or(|l| l.revision != bake.revision) || booth.aspect != aspect;
    if !rebake && *last == Some(now) {
        return;
    }
    booth.aspect = aspect;

    if bake.parts.is_empty() {
        // Nothing to show: empty the stage, and wake only if something stood there, so an
        // already-empty stage at startup costs no booth passes.
        if *staged {
            commands.entity(booth.root).despawn_related::<Children>();
            booth.baked = None;
            booth.wake = BOOTH_SETTLE_FRAMES;
            booth.live = false;
            booth.pending.clear();
            // The despawn reaped meshes and anchors; the rig state on the root needs its own.
            super::clear_booth_rig(commands, booth.root);
            booth.rigged = false;
            booth.parked = false;
            *staged = false;
        }
        *last = Some(now);
        return;
    }
    // Rig and framing come from the display cache the assembly gated on; if not ready, leave
    // `last` alone and retry next frame.
    let Some(creatures) = creatures.as_deref() else {
        return;
    };
    let (Some(rig), Some(anchors)) = (
        creatures.display_rig(bake.display_id),
        booth_anchors(Some(creatures), Some(bake.display_id)),
    ) else {
        booth.wake = booth.wake.max(BOOTH_SETTLE_FRAMES);
        return;
    };

    if rebake || !*staged {
        let mut relight = |m: &Handle<WowModelMaterial>| booth_light.pane.variant(m, materials);
        let booth_parts: Vec<BoothPart> = bake
            .parts
            .iter()
            .map(|p| BoothPart {
                skinned: p.skinned_mesh.clone(),
                static_mesh: p.static_mesh.clone(),
                material: relight(&p.material),
                // Not built, as in the glue preview.
                alpha_anim: None,
                twins: BoothTwins::default(),
                mat_anim: false,
            })
            .collect();
        let booth_riders: Vec<BoothRider> = bake
            .riders
            .iter()
            .map(|r| BoothRider {
                mesh: r.mesh.clone(),
                material: relight(&r.material),
                bone: r.bone,
                offset: r.offset,
                twins: BoothTwins::default(),
            })
            .collect();
        let booth_billboards: Vec<BoothBillboardSpec> = bake
            .billboards
            .iter()
            .map(|b| BoothBillboardSpec {
                mesh: b.mesh.clone(),
                material: relight(&b.material),
                bone: b.bone,
                offset: b.offset,
                kind: b.kind,
                twins: BoothTwins::default(),
            })
            .collect();
        // Never latch a world-lane material into a pane ([`super::light`]): retry instead.
        if booth_light.pane.take_unready() {
            booth.wake = booth.wake.max(BOOTH_SETTLE_FRAMES);
            return;
        }
        commands.entity(booth.root).despawn_related::<Children>();
        let mut booth_rig = spawn_booth_model(
            commands,
            palettes,
            booth.root,
            booth.layer.clone(),
            &booth_parts,
            &booth_riders,
            rig.inverse_bindposes
                .as_ref()
                .map(|ibp| (rig.skeleton, ibp, rig.animations)),
            anim_data.as_deref().map(|a| &a.0),
            // Stand looping, like the paper doll: `<DressUpModel>` renders live.
            BoothMotion::Loop,
            bake.grip,
            &booth_billboards,
            BoothInstance::default(),
        );
        let (fx_emitters, _) = spawn_booth_effects(
            commands,
            &mut booth_rig,
            &booth.layer,
            booth_light.pane.buffer.as_ref(),
            &bake
                .effects
                .iter()
                .map(|fx| BoothEffects {
                    bone: fx.bone,
                    offset: fx.offset,
                    emitters: fx.emitters.clone(),
                })
                .collect::<Vec<_>>(),
            BoothInstance::default(),
        );
        // The bake animates, so `gate_booth_cameras` runs its camera every frame the pane draws.
        booth.turn.rebaked();
        booth.live = true;
        // A fresh bake is animated; the park state is the new rig's.
        booth.rigged = booth_rig.rigged();
        booth_rig.finish(commands);
        booth.parked = false;
        *staged = true;
        // `WOW_BOOTH_LOG=1`: one line per committed bake, as `super::log_bake` for the mirrored
        // booths.
        if super::booth_log() {
            eprintln!(
                "[booth] {slot} bake parts={} riders={} billboards={} fx={} rev={} \
                 aspect={aspect:.3} view={view:?}",
                booth_parts.len(),
                booth_riders.len(),
                booth_billboards.len(),
                fx_emitters,
                bake.revision,
            );
        }
        wake_booth(
            booth,
            materials,
            booth_parts
                .iter()
                .map(|p| &p.material)
                .chain(booth_riders.iter().map(|r| &r.material))
                .chain(booth_billboards.iter().map(|b| &b.material)),
        );
    }

    // Camera 1 as the pane's root stood when it froze; the stock rooms stand at the origin.
    let mut cam = body_frame(&anchors, aspect);
    let root = match view {
        Some(view) => {
            let g = super::framing::pane_model_scale(framing.gx.0);
            let (offset, root) = view.rig(g, yaw);
            cam.0.translation += offset;
            // A pool pane turns by `SetFacing`, which plays no shuffle.
            booth.turn.faced = Some(yaw);
            root
        }
        None => {
            // The yaw, `Model:SetRotation`, applied on a fresh bake and on every spin. A spin also
            // steps the feet ([`super::booth::drive_booth_turn`]), as the stock `Model_OnUpdate`
            // held-arrow turn does; keyed on the yaw alone, since a re-bake is a `RefreshUnit`
            // and does not turn.
            if booth.turn.faced != Some(yaw) {
                if let Some(prev) = booth.turn.faced {
                    booth.turn.spun = Some(super::booth::turn_shuffle(prev, yaw));
                }
                booth.turn.faced = Some(yaw);
            }
            Transform::from_rotation(Quat::from_rotation_y(yaw))
        }
    };
    aim(cams, slot, &cam);
    commands.entity(booth.root).insert(root);
    booth.wake = booth.wake.max(BOOTH_SETTLE_FRAMES);
    *last = Some(now);
}
