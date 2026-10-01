//! MONKEY (torch shadows, Phase 1) — the interior point-light shadow lane, a plug-in on
//! [`super::shadow_core`], independent of [`super::world_shadow`]/[`super::character_shadow`].
//!
//! **Why this is not Bevy's point shadows.** benilla's world camera sets
//! `bevy::light::cluster::ClusterConfig::None`, which disables Bevy's point/spot clustering AND
//! starves the point-light shadow-map prep — so `fetch_point_shadow` / `clusterable_objects` are
//! DEAD here. The old design (promote the nearest fixtures to shadow-casting `PointLight` proxies and
//! sample their cube maps) could never fire, so it is gone. In its place: this lane picks the
//! interior fixtures that actually LIGHT the player, computes a down-looking reverse-Z `view_proj` for each, builds the
//! retained interior caster geometry, and publishes it all as [`TorchShadowViews`] (an
//! [`ExtractResource`]). The render world then renders a depth map per fixture
//! (`benilla_world::static_gx::torch_depth`) and `static_gx.wgsl`'s interior surface lane samples
//! it so a pillar throws a radial shadow.
//!
//! **MONKEY (torch caster selection).** Which fixtures get maps used to be "the four nearest the
//! camera", recomputed and hard-switched every frame. That is the shadow POP: in Northshire Abbey
//! (42 candelabra, ~5 yd apart) the nearest-four set churns every few steps, and every fixture
//! outside it lit with no shadow at all, so a fixture's shadow appeared as you walked up to it and
//! vanished as you moved on. The selection is now (a) ranked by the fixture's own DIRECT
//! CONTRIBUTION at the player using the shaders' profile ([`fixture_score`]), (b) hysteretic
//! (a challenger must beat an incumbent by [`TORCH_SWAP_RATIO`], one swap per
//! [`TORCH_SWAP_COOLDOWN`]), and (c) CROSS-FADED — every slot carries a weight the receivers
//! `mix` with, so a promotion ramps in and a demotion ramps out before the slot is reused. The
//! budget is [`MAX_TORCH_CASTERS`] slots, of which `interiorShadowCasters` (default 12) are filled.
//!
//! Gated on `interiorLight` + `interiorShadows` ([`VideoConfig`]); declares demand so the shared rig
//! stays up. Remove this module + its plugin + the `interiorShadows` cvar and the other lanes are
//! untouched. MONKEY (static torch cache): static_gx and wow_model share the 6416-byte table.
//! Camera distance must be <= ShadowDistance, AND player distance <= 4R+3. Contribution ranking,
//! 1.5x hysteresis and fades choose up to twelve of sixteen resident slots; this is still a finite
//! budget, not a promise that a room with forty eligible candles can give all forty cube maps.
//! Rigid furniture joins the cached static geometry. Only the nearest interiorShadowDynamic
//! promoted fixtures (default four, maximum eight, measured from the player) overlay creatures/animated parts.
//!
//! MONKEY (torch lane perf): what the lane costs when nothing is happening. Three fixed per-frame
//! charges were paid regardless of every count dial — which is why `interiorShadowCasters 8` and
//! `interiorShadowDynamic 2` measured identically to the defaults — and each now has a gate:
//! the static-cache FINGERPRINT walks one slot a frame instead of four AND skips even that while
//! the scene census is unchanged ([`TORCH_SCAN_PER_FRAME`], [`PartCensus`]); the moving-caster
//! gather runs at `interiorShadowEntityRate` Hz (default 30, `0` = every frame) over each dynamic
//! fixture's OWN reach rather than the cube's 48 yd ([`entity_gather_due`],
//! [`entity_gather_radius`]); and the receivers skip the table scan where the fixture's direct term
//! is already nothing (`TORCH_SKIP_EPS`, both shaders). None of it changes what is drawn: the
//! overlay passes still run every frame from the last mesh, so a lower rate ages a POSE and never
//! removes a shadow. `WOW_TORCH_TRACE=1` prints the three counters as `torch-perf:` lines.

use std::hash::{Hash, Hasher};

use bevy::math::{DMat4, DVec3, DVec4};
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResourcePlugin;

use bevy::pbr::MeshMaterial3d;

use benilla_assets::coords::wow_to_bevy;
use benilla_assets::materials::WowModelMaterial;
use benilla_formats::ModelBlend;
use benilla_world::billboard::BillboardCard;
// MONKEY (torch owner exclusion): the placement identity an entity-lane caster part carries, and
// the model-local bound the containment fallback is decided on.
use benilla_world::interact::{PickMesh, WorldObject};
use benilla_world::lighting::{
    interior_reach, m2_light_reach, DaylightFixture, DynamicInteriors, FireLightGain, LightLane,
    LightLitRooms, LightReach, LightRooms, ShadowDistance, ShadowProxyLight, SyntheticFireLight,
    WorldPointLight, WowLighting,
};
use bevy::camera::primitives::Aabb;
// MONKEY (carried light stability): the settle verdict a carried light earns by standing still.
// MONKEY (outdoor torch shadows): …and the marker that says a light is CARRIED BY A BODY.
use crate::entities::{CarriedLightMotion, HeldLight};
use benilla_world::model_render::{ModelKind, ModelPart, ShadowOccluder};
use benilla_world::rig_palette::{RigPalettes, RigPart, RigSkin};
use benilla_world::static_gx::{torch_flame_inside_bounds, LightOwner, StaticGx, TorchShadowViews};
use benilla_world::view::WorldCamera;

use crate::char_select::ClientState;
use crate::net::SelfPlayer;
use crate::shadow_core::{
    empty_shadow_mesh, restore_mesh_buffers, take_mesh_buffers, ShadowDemand, ShadowFrame,
    ShadowSet,
};
use crate::video::VideoConfig;

/// How many interior fixtures can hold a depth map at once — the ALLOCATION. Matches
/// `torch_depth::MAX_TORCH_MAPS`; the number actually filled is the live `interiorShadowCasters`
/// cvar (default 12), so raising the cap does not by itself cost a frame.
const MAX_TORCH_CASTERS: usize = 16;
/// MONKEY (live bank rank): the cvar and resource consumer share the render allocation's cap.
pub(crate) const MAX_TORCH_DYNAMIC: usize = TorchShadowViews::MAX_TORCH_DYNAMIC;
/// MONKEY (torch caster reach): promote out to FOUR times the fixture's effective reach. Its
/// pool still needs shadows when viewed across the room, long before it lights the player's body.
const TORCH_ELIGIBLE_REACH_MULT: f32 = 4.0;
/// MONKEY (static torch cache): minimum coarse CAMERA search; the live search is
/// max(TORCH_SEARCH_RADIUS, ShadowDistance), so it never truncates the shadow-distance setting.
const TORCH_SEARCH_RADIUS: f32 = TORCH_ELIGIBLE_REACH_MULT * TORCH_RANGE + TORCH_ELIGIBLE_SLACK;
/// MONKEY (torch caster reach): extra eligibility slack beyond the expanded 4R window (yd).
const TORCH_ELIGIBLE_SLACK: f32 = 3.0;
/// MONKEY (torch caster selection): a challenger must beat the weakest incumbent by this FACTOR
/// before it may take its slot. Pure "best N" thrashes: two candles a step apart trade the lead
/// every few frames as you walk between them, and each trade used to be an instant on/off. 1.5 is
/// wide enough that a real change of room wins immediately and a step sideways never does.
const TORCH_SWAP_RATIO: f32 = 1.5;
/// MONKEY (torch caster selection): and at most ONE contested swap this often (s). The ratio alone
/// still permits a cascade — six slots could each lose their contest in the same frame, which is
/// six shadows changing at once no matter how smooth each fade is.
const TORCH_SWAP_COOLDOWN: f64 = 0.25;
/// MONKEY (torch caster selection): the cross-fade rate (weight per second) — ~1/3 s to ramp a
/// shadow fully in or out. Slow enough to read as "the light reaches you now", fast enough that
/// the geometry is not visibly shadow-less while you stand in front of it.
const TORCH_FADE_RATE: f32 = 3.0;
/// MONKEY (torch caster selection): the interior direct-light profile, MIRRORED from the
/// `interior_room_light` block that both `static_gx.wgsl` and `wow_model.wgsl` carry ("MONKEY (soft
/// falloff)" — keep the three in sync). Ranking by the shader's OWN contribution is the whole point
/// of this change: distance ranked a bright forge behind you below a dead candle at your feet.
const INTERIOR_CORE_FRAC: f32 = 0.26;
const INTERIOR_CORE_GAIN: f32 = 1.5;
const INTERIOR_DIRECT_POW: f32 = 10.0;
/// MONKEY (torch caster reach): unchanged fixture-centred far plane. `interior_reach` caps R at
/// 48 yd; seeing that pool from 4R away needs earlier promotion, not a longer projection.
const TORCH_RANGE: f32 = 48.0;
/// MONKEY (Phase 5): six cube faces per fixture. Keep in sync with `torch_depth::CUBE_FACES`.
const CUBE_FACES: usize = 6;
/// MONKEY (static torch cache): how many resident slots have their source FINGERPRINT recomputed
/// per frame. Hashing one fixture's 48 yd source set walks every resident `GxItem` and every model
/// part; doing that for all twelve residents every frame hands the CPU back the cost the GPU cache
/// just saved (~12 full scene scans per frame, and it is a per-frame cost even in a room where
/// nothing ever changes).
///
/// MONKEY (torch lane perf): ONE, not four. Four was already the "don't scan them all" compromise,
/// but it is still four full walks of the resident static scene AND of every model part, every
/// frame, forever — measured as the bulk of the ~3 ms the whole lane costs at the Lion's Pride Inn,
/// which is exactly why neither `interiorShadowCasters` nor `interiorShadowDynamic` moved the frame
/// time: the cost was never per map. One slot a frame is a 12-frame (~0.26 s) sweep of a full
/// resident set, which is the same order as the cross-fade a newly promoted slot ramps in over, so
/// nothing the eye can catch waits on it. Two further things keep that honest: a slot that has
/// never been fingerprinted JUMPS the queue (see `scan_start`), so a promotion still builds at
/// once, and the walk itself is skipped whenever neither the retained scene nor the entity-part
/// census has changed since that slot last looked ([`PartCensus`]).
const TORCH_SCAN_PER_FRAME: usize = 1;
/// MONKEY (torch lane perf): slack (yd) added to a fixture's own reach when gathering MOVING
/// casters for it. The receivers' direct term is `interior_window(d, reach, ..)`, which is exactly
/// zero at `d >= reach`, so a caster further than `reach` from the fixture can only shadow
/// fragments the fixture does not light — it is invisible by construction. The margin covers the
/// part's own extent, because the admission test is on the part's ORIGIN and a body is a couple of
/// yards tall: without it a unit standing just inside the pool with its origin just outside would
/// lose its shadow. (The STATIC gather is untouched at the full [`TORCH_RANGE`]: it is cached, so
/// its radius is not a per-frame cost, and shrinking it would invalidate every cached map.)
const TORCH_ENTITY_REACH_MARGIN: f32 = 4.0;
/// MONKEY (static torch cache): how far (yd²) a fixture may drift from the position its cached map
/// was rendered at before that map is WITHDRAWN. A cached map that is merely STALE (the room
/// streamed a chair in since it was baked) keeps being published — the shadow is a few frames out
/// of date, which is invisible. A map whose fixture MOVED is not stale but WRONG: every projection
/// in it aims from somewhere else, and the published `view_projs` (rebuilt from the live position
/// each frame) no longer address it. Withdrawing a merely-stale map instead was a hard on/off
/// blink: an unready slot's `positions[i].w` is forced to 0 with no fade — precisely the pop the
/// whole cross-fade exists to prevent — for however many frames the rebuild budget takes to reach it.
const TORCH_STALE_DRIFT_SQ: f32 = 0.01;
/// MONKEY (outdoor torch shadows): how far from the PLAYER an exterior fire may still be promoted
/// (yd). The interior lane's `4R + slack` window has no meaning out here — an exterior entry packs
/// no reach at all (`lane == 0`, the colour row's reach lane is the interior half's), and the
/// receivers apply the un-windowed `1/(0.7d + 0.03d²)` to it. So the window is the CUBE MAP's own
/// range: past [`TORCH_RANGE`] a fixture's whole shadow volume is outside the projection anyway,
/// and the world fade in the shaders has already ended by then.
const TORCH_EXT_ELIGIBLE_YD: f32 = TORCH_RANGE;
/// MONKEY (moving fixture): how many slots may be FULLY DYNAMIC in one frame — rebuilt from
/// scratch (static geometry AND entities, into their live layers) because their fixture is
/// physically moving. Shared with the render plan, which keeps a matching separate rebuild budget.
const TORCH_MOVING_MAX: usize = TorchShadowViews::MAX_TORCH_MOVING;
/// MONKEY (moving fixture): how long a fixture must hold still before it is treated as SETTLED and
/// handed back to the cached path. Long enough to cover a walk cycle's pauses and a pet's
/// stop-turn-start, short enough that a brazier dropped on the ground is cached within a second.
const TORCH_MOVING_SETTLE: f64 = 0.75;
/// MONKEY (moving fixture): how far (yd²) a fixture must shift between two frames to count as
/// having MOVED at all. A hand-held flame on an idling NPC jitters by millimetres with the breathe
/// animation; at 60 fps this threshold is ~1.2 yd/s, so an idle never registers and a walk always
/// does. It is deliberately NOT [`TORCH_STALE_DRIFT_SQ`]: that one asks "is the cached map still
/// valid" (a question about the map), this one asks "is the fixture in motion" (a question about
/// the fixture), and a fixture creeping 0.09 yd a frame would answer the first "yes, forever".
const TORCH_MOVING_EPS_SQ: f32 = 0.0004;
/// MONKEY (moving fixture): the fade rate (weight per second) for a moving slot that could NOT get
/// the moving budget — 0.15 s to nothing, five times the normal [`TORCH_FADE_RATE`]. The normal
/// rate exists to keep a *correct* shadow on screen while it dissolves; this one is for a shadow
/// that is already wrong (its map is frozen at a position the light has left), so the only good
/// answer is to get rid of it faster than the eye tracks it, and the fade is there purely so it
/// dissolves instead of blinking.
const TORCH_MOVING_FADE_RATE: f32 = 1.0 / 0.15;
/// MONKEY (torch owner exclusion): how far up the `ChildOf` chain a caster part is followed while
/// asking "is this the host model of the light that is lighting me?" — the entity-lane half of the
/// owner test ([`benilla_world::static_gx::LightOwner::Instance`]). A part's chain to its model
/// frame is `part → (joint → …) → frame`, and a rig's joint chain is the deep part; 64 clears every
/// shipped skeleton (it is `carried_light`'s own `ANCHOR_WALK_DEPTH`, for the same hierarchy) and
/// the cap is there so a malformed cycle can never hang the frame.
const OWNER_WALK_DEPTH: usize = 64;
/// Each cube face's FOV: 90° plus a hair, so a direction exactly on a face border still lands
/// inside the face the shader picks for it (the projector returns "lit" outside its frustum).
const TORCH_FACE_FOV: f32 = std::f32::consts::FRAC_PI_2 + 0.02;

pub(crate) struct TorchShadowPlugin;

impl Plugin for TorchShadowPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TorchLane>()
            .init_resource::<TorchShadowViews>()
            // The render world reads the published fixtures/caster through this.
            .add_plugins(ExtractResourcePlugin::<TorchShadowViews>::default())
            .add_systems(Last, update_torch_shadows.in_set(ShadowSet::Lanes));
    }
}

/// MONKEY (torch caster selection): one OCCUPIED caster slot — a fixture that holds (or is
/// releasing) six depth-map layers. Slots are retained across frames, which is what makes
/// hysteresis and cross-fading possible at all: the old lane recomputed "the nearest four" from
/// scratch every frame and so had no notion of an incumbent to keep.
struct TorchSlot {
    // MONKEY (static torch cache): physical residency survives Vec compaction and rank changes.
    cache_slot: usize,
    mesh: Option<Handle<Mesh>>,
    geometry_key: Option<u64>,
    /// MONKEY (static torch cache): the fixture position `mesh` was actually gathered AT, which is
    /// not necessarily `pos` — the cursor scans only [`TORCH_SCAN_PER_FRAME`] slots a frame, so a
    /// slot can be published several frames after it last rebuilt. `None` until the first build.
    /// See [`TORCH_STALE_DRIFT_SQ`] for why staleness publishes and drift does not.
    built_at: Option<Vec3>,
    /// The fixture entity — the identity the slot is held BY. Matching on entity rather than on
    /// position is what lets a CARRIED torch (which moves every frame) keep its slot.
    fixture: Entity,
    /// Its world position this frame (absolute Bevy), the shader's correlation key.
    pos: Vec3,
    /// Its contribution score at the player this frame (see [`fixture_score`]).
    score: f32,
    /// MONKEY (torch lane perf): the fixture's EFFECTIVE reach in yards — the same `R` the light
    /// packer writes into the receivers' colour row and the same one [`fixture_score`] ranks with.
    /// Carried on the slot so the moving-caster gather can be sized off the light that will
    /// actually be sampled, instead of the cube projection's flat 48 yd far plane. Exterior slots
    /// hold [`TORCH_RANGE`]: an exterior entry packs no reach, and its receivers apply an
    /// un-windowed falloff, so there is no smaller honest number out there.
    reach: f32,
    /// MONKEY (torch lane perf): the `(retained-scene generation, static-part census, position)`
    /// this slot last FINGERPRINTED at. While all three are unchanged the source set cannot have
    /// changed either, so the expensive walk (every resident `GxItem` in 48 yd, then every model
    /// part) is skipped outright. `None` = never fingerprinted, which also jumps the scan queue.
    checked: Option<(u64, u64, Vec3)>,
    /// MONKEY (outdoor torch shadows): which LANE promoted it. Drives the exterior half of the
    /// budget split, the emitter-owner exclusion radius its caster gather uses, and the published
    /// `exterior` bit the receivers gate on. A slot never changes lane: a fixture is interior or
    /// exterior by where it physically stands ([`LightLane`]), and if that verdict ever flipped the
    /// fixture would leave `cands` and be cross-faded out like any other loss.
    exterior: bool,
    /// MONKEY (torch owner exclusion): WHO this fixture is part of — the identity its caster
    /// gathers drop out of its own map, so the lamp never shadows itself. `None` for a fixture
    /// that carries no owner tag (an authored WMO MOLT fixture, which has no model at all, and any
    /// light source predating the tag); those fall back to the containment rule alone.
    owner: Option<LightOwner>,
    /// MONKEY (torch owner exclusion): how many items the LAST rebuild of this slot dropped as its
    /// own body — `WOW_TORCH_TRACE` only. Held on the slot rather than in a per-frame counter
    /// because the gather is cached: the answer is "at the last rebuild", not "this frame".
    trace_excluded: (u32, u32),
    /// MONKEY (moving fixture): where the fixture stood LAST frame, and the last time it was seen
    /// to move further than [`TORCH_MOVING_EPS_SQ`]. `still_since` starts "long ago" (0.0) so a
    /// freshly promoted slot takes the ordinary cached path on its first frame and only becomes a
    /// moving slot once it has actually been observed moving — a placed brazier or a standing NPC's
    /// torch must never spend its first three quarters of a second on the moving budget.
    last_pos: Vec3,
    still_since: f64,
    /// MONKEY (moving fixture): this slot holds one of the [`TORCH_MOVING_MAX`] live rebuild
    /// budgets this frame — its caster mesh is regathered and its six faces re-rendered EVERY
    /// frame, from the live fixture position, with the moving entities overlaid.
    moving_live: bool,
    /// MONKEY (moving fixture): …and this one is moving but did NOT get a budget, so it is being
    /// dropped at [`TORCH_MOVING_FADE_RATE`] rather than kept as a stale map.
    fast_fade: bool,
    /// The cross-fade weight, published as `positions[i].w` and applied as `mix(1, shadow, w)` by
    /// both receivers.
    w: f32,
    /// Set once the slot has lost its contest (or its fixture went out of reach, or the budget
    /// shrank). It keeps rendering while `w` ramps down, and the slot is freed for a challenger
    /// only when `w` reaches 0 — "fade, not switch".
    evicting: bool,
}

impl TorchSlot {
    /// MONKEY (moving fixture): has the fixture left the position its cached map was baked from?
    /// This is [`map_publishable`]'s complement with one extra condition — a slot that has NEVER
    /// built is not "drifted", it is simply new, and belongs to the promotion path rather than to
    /// the moving budget.
    fn drifted(&self) -> bool {
        self.built_at
            .is_some_and(|p| p.distance_squared(self.pos) > TORCH_STALE_DRIFT_SQ)
    }

    /// MONKEY (moving fixture): is this fixture MOVING right now, i.e. does it need its whole cube
    /// re-rendered this frame instead of its cached one republished?
    ///
    /// Two halves, and both are needed. WHO it is: an entity-hosted light ([`LightOwner::Instance`]
    /// — `entities::carried_light`'s pet flames, NPC torches, GameObject braziers) is the only kind
    /// that CAN move, and a fixture that has already drifted off its baked map is proof of motion
    /// whoever owns it. WHETHER it is moving NOW: a placed campfire is `Instance`-owned and never
    /// moves, so the owner alone would put every brazier in a village on the moving budget forever.
    /// `still_since` is stamped by the per-frame motion test and the fixture is handed back to the
    /// cache [`TORCH_MOVING_SETTLE`] after it stops.
    fn moving(&self, now: f64) -> bool {
        (matches!(self.owner, Some(LightOwner::Instance(_))) || self.drifted())
            && now - self.still_since < TORCH_MOVING_SETTLE
    }
}

/// MONKEY (static torch cache): stable physical slots and per-fixture source meshes.
/// Camera drift never rebuilds static geometry; only a source-set/fixture change does.
#[derive(Resource, Default)]
struct TorchLane {
    // MONKEY (static torch cache): only moving casters are rebuilt every frame. Each slot owns
    // its own static geometry handle; a replacement gets a NEW asset id, so the render world
    // cannot certify an old GPU mesh as the newly streamed geometry while uploads catch up.
    entity_mesh: Option<Handle<Mesh>>,
    rebuild_cursor: usize,
    /// MONKEY (torch caster selection): the occupied slots, at most `interiorShadowCasters` of them
    /// (plus any still fading out after the budget shrank). Published in this order.
    slots: Vec<TorchSlot>,
    /// When the last CONTESTED swap was started (seconds of `Time::elapsed`), for the cooldown.
    last_swap: f64,
    /// MONKEY (torch lane perf): when the moving-caster mesh was last REGATHERED, and for which
    /// dynamic set. The mesh is retained between regathers — the six overlay passes still run every
    /// frame from it — so this is a staleness dial on the POSE, never an on/off for the shadow.
    /// The mask rides along because a gather is only valid for the fixtures it was gathered
    /// around: when the dynamic set changes the cadence is overridden and the mesh rebuilt at once.
    entity_at: f64,
    entity_mask: u32,
    /// MONKEY (torch lane perf): `WOW_TORCH_TRACE` accounting for the second in progress — how many
    /// fingerprint walks actually ran, how many regathers, and the last gather's admitted-part
    /// count and radius. Counters, not timings: they are what tells the user whether a cadence or
    /// a gate is doing anything at all.
    trace_scans: u32,
    trace_gathers: u32,
    trace_parts: u32,
    trace_reach: f32,
}

/// MONKEY (torch lane perf): the per-frame "has anything a fingerprint would hash MOVED?" summary,
/// computed ONCE for the whole lane instead of once per scanned slot.
///
/// The fingerprint it guards is order-independent and covers (a) the retained static scene and (b)
/// every rigid opaque model part. This pair reproduces both halves cheaply: `gx` folds the retained
/// regions' own change stamps and populations (O(regions), not O(items) — see
/// `StaticGx::torch_residency_generation`), and `parts` folds one branch-free mix per part over its
/// entity id, its geometry `Arc` identity, its translation AND one basis column, so an arrival, a
/// despawn, a geometry swap, a move and a rotation each change it. It is deliberately NOT a
/// substitute for the fingerprint — it is scene-wide where the fingerprint is fixture-local, so it
/// can only ever say "nothing changed anywhere, don't bother looking".
#[derive(Clone, Copy, PartialEq, Eq, Default)]
struct PartCensus {
    gx: u64,
    parts: u64,
}

/// MONKEY (torch caster selection): the interior direct term this fixture delivers at distance `d`
/// — the shaders' own profile (core inverse-square with a soft core, times the C1 window), times
/// the fixture's luminous intensity. This is the ranking key, and it is the answer to the bug: at
/// Northshire's candelabra spacing "nearest 4" and "the 4 that light you" are different sets, and
/// only the second one is stable as you walk, because a fixture's contribution changes smoothly
/// with distance while its RANK by distance changes in steps.
///
/// MONKEY (torch caster selection): one scored candidate — `(fixture, position, score, exterior,
/// reach, owner)`. Named because MONKEY (torch lane perf) added the trailing REACH lane, and a
/// `(.., x)` pattern over a bare tuple would silently have re-bound to it. MONKEY (torch owner
/// exclusion) added the OWNER lane for the same reason it rides the slot.
type Candidate = (Entity, Vec3, f32, bool, f32, Option<LightOwner>);

/// MONKEY (torch owner exclusion): one fixture in a caster gather — `(slot, position, admission
/// radius, owner)`. The owner rides the gather because the exclusion is per FIXTURE, not per lane:
/// the static gather runs for one slot at a time and is therefore exact, and the shared moving
/// gather (one mesh for the whole dynamic set) excludes a part owned by ANY of its fixtures, which
/// is the only thing one mesh can express.
type GatherFixture = (usize, Vec3, f32, Option<LightOwner>);

/// MONKEY (torch caster reach): retain the intensity and soft inverse-square core at the player's
/// distance, but stretch the selection window to 4R (including eligibility slack). This is a
/// ranking extension, not a lighting change: at 3R the score is small, positive and still falling.
/// No window floor flattens distant candidates; the window reaches zero at the eligibility edge.
fn fixture_score(intensity: f32, d: f32, r: f32) -> f32 {
    let r0 = (INTERIOR_CORE_FRAC * r).max(1e-3);
    let atten = INTERIOR_CORE_GAIN / (1.0 + (d / r0) * (d / r0));
    let extent = TORCH_ELIGIBLE_REACH_MULT * r + TORCH_ELIGIBLE_SLACK;
    let w = (1.0
        - (d / extent.max(1e-4))
            .clamp(0.0, 1.0)
            .powf(INTERIOR_DIRECT_POW))
    .clamp(0.0, 1.0);
    intensity * atten * w * w
}

/// MONKEY (review fixes): rank only light the interior receivers actually sample. Match the
/// packer's unresolved-lane fallback AND its synthetic-only gain; a disabled flame or a resolved
/// exterior fixture must score zero before it can evict a useful caster. Reach still uses the
/// authored intensity, so dimming a flame changes its contribution without moving its pool.
fn candidate_score(
    intensity: f32,
    d: f32,
    r: f32,
    lane: Option<&LightLane>,
    has_rooms: bool,
    synthetic: bool,
    fire_gain: f32,
) -> f32 {
    if !lane.map_or(has_rooms, |l| l.interior) {
        return 0.0;
    }
    fixture_score(intensity, d, r) * if synthetic { fire_gain.max(0.0) } else { 1.0 }
}

/// MONKEY (outdoor torch shadows): the EXTERIOR direct term this fixture delivers at distance `d`
/// — the exterior receivers' OWN profile (`point_light_sum` / `wmo_exterior_point_sum`:
/// `1/(0.7d + 0.03d²)`, no soft core, no authored window), times the fixture's luminous intensity.
/// A separate function rather than a flag on [`fixture_score`] because the two lanes genuinely
/// light differently, and the whole point of scoring by CONTRIBUTION is that the ranking key is the
/// number the receiver will actually compute.
///
/// `d` is floored at 1 yd for the ATTENUATION only: the exterior falloff has a pole at 0 and the
/// receivers live with it (no surface is ever at the light), but a ranking key that goes to
/// infinity would let a fixture the player is standing on outrank the entire village for one frame
/// and then hand the slot back. The window is unfloored and reaches zero at the eligibility edge,
/// so a candidate's score falls off smoothly instead of stepping to 0 the frame it drops out.
fn exterior_fixture_score(intensity: f32, d: f32) -> f32 {
    let dd = d.max(1.0);
    let atten = 1.0 / (0.7 * dd + 0.03 * dd * dd);
    // `INTERIOR_DIRECT_POW` only for the WINDOW's shape — the C1 shoulder that takes a candidate's
    // score to exactly zero at the edge instead of stepping it there. It is the interior lane's
    // constant because it is the same shoulder, not because this is an interior term; the
    // attenuation above is the exterior receivers' own and shares nothing with that lane.
    let w = (1.0
        - (d / TORCH_EXT_ELIGIBLE_YD)
            .clamp(0.0, 1.0)
            .powf(INTERIOR_DIRECT_POW))
    .clamp(0.0, 1.0);
    intensity * atten * w * w
}

/// MONKEY (outdoor torch shadows): may an EXTERIOR fire be promoted at all? Same two camera tests
/// as [`fixture_eligible`] (it is the same cube-map budget and the same `shadowDistance` dial);
/// the player-distance window is the flat [`TORCH_EXT_ELIGIBLE_YD`] instead of `4R + slack`,
/// because an exterior entry has no packed reach to take four of.
fn exterior_eligible(player_d: f32, camera_d: f32, distance: f32) -> bool {
    camera_d <= TORCH_SEARCH_RADIUS.max(distance)
        && camera_d <= distance
        && player_d <= TORCH_EXT_ELIGIBLE_YD
}

/// MONKEY (outdoor torch shadows): how many of the `interiorShadowCasters` resident slots an
/// exterior fire may take. HALF, rounded up: a village square at night can easily author more
/// campfires than the whole budget, and without a cap the first night in Goldshire would evict
/// every indoor candle in the inn a step away — the interior lane's look is shipped and tuned, and
/// this feature must not be able to take it away. Rounded UP so the smallest budgets
/// (`interiorShadowCasters 1`) still get one outdoor shadow rather than none.
fn exterior_budget(want: usize) -> usize {
    want.div_ceil(2)
}

/// MONKEY (outdoor torch shadows): apply the budget split, by TRUNCATING the sorted candidate list
/// rather than by teaching the slot machinery about lanes. Everything downstream — the incumbent
/// refresh, the fade-out, the promotion, the contested swap — reads `cands`, so dropping the
/// exterior tail here means an exterior incumbent that falls past the cap cross-fades out exactly
/// like a fixture that walked away ("fade, not switch" holds for this eviction too), and no
/// exterior fixture can ever be promoted into slot `cap + 1`. Interior candidates are untouched:
/// the cap is a ceiling on the outdoor lane, never a floor under it, so a night indoors still fills
/// all twelve slots with candles.
///
/// `cands` must already be sorted best-first — the survivors are then the BEST `cap` exterior
/// fires, which is the same ranking rule the whole lane runs on.
fn cap_exterior_candidates(cands: &mut Vec<Candidate>, cap: usize) {
    let mut seen = 0usize;
    cands.retain(|(_, _, _, exterior, ..)| {
        if !*exterior {
            return true;
        }
        seen += 1;
        seen <= cap
    });
}

/// MONKEY (outdoor torch shadows): the realtime-shadow day strength — a three-line mirror of
/// `benilla_world::lighting::global_light`'s private `sun_shadow_strength` (`blob_shadow.rs` keeps
/// the same mirror, for the same reason: it is three lines and a `pub` on it would be a wider API
/// than the number deserves). **Exactly 1.0 above ~12° of sun elevation**, which is what makes
/// `night_w = 1 - this` exactly 0 by day in the receivers — the daylight-is-untouched guarantee.
fn sun_shadow_strength(sun_height: f32) -> f32 {
    let t = (sun_height / 0.208).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// MONKEY (static torch cache): camera distance uses the world-shadow resource's Bevy-yard
/// yardstick. The player-centred 4R+3 window still limits/ranks useful pools; BOTH tests apply.
/// The coarse camera search grows with shadowDistance so it can never silently shorten that dial.
fn fixture_eligible(player_d: f32, camera_d: f32, r: f32, distance: f32) -> bool {
    camera_d <= TORCH_SEARCH_RADIUS.max(distance)
        && camera_d <= distance
        && player_d <= TORCH_ELIGIBLE_REACH_MULT * r + TORCH_ELIGIBLE_SLACK
}

/// MONKEY (static torch cache): may a slot publish its cached map this frame? Yes while the
/// fixture still stands where the map was baked from (see [`TORCH_STALE_DRIFT_SQ`]) — staleness in
/// the SURROUNDINGS is invisible for the few frames a rebuild takes, whereas withdrawing the map
/// zeroes the slot's weight with no fade. No when the fixture has moved: the six projections are
/// republished from the live position every frame and would no longer address that depth.
fn map_publishable(built_at: Option<Vec3>, pos: Vec3) -> bool {
    built_at.is_some_and(|p| p.distance_squared(pos) <= TORCH_STALE_DRIFT_SQ)
}

/// MONKEY (moving fixture): split this frame's scored movers into the [`TORCH_MOVING_MAX`] that get
/// a live rebuild and the rest, which are dropped (fast-faded) rather than kept as frozen maps.
///
/// Ranked by the SAME contribution score the whole lane ranks by, so the budget goes to the mover
/// whose shadow the player can actually see, and tied scores fall back to slot order so the answer
/// is deterministic frame to frame — a budget that alternated between two equally-scored pets would
/// fast-fade each of them in turn, which is the churn the whole selection machinery exists to avoid.
fn moving_budget(mut movers: Vec<(usize, f32)>) -> (Vec<usize>, Vec<usize>) {
    movers.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    (
        movers
            .iter()
            .take(TORCH_MOVING_MAX)
            .map(|(i, _)| *i)
            .collect(),
        movers
            .iter()
            .skip(TORCH_MOVING_MAX)
            .map(|(i, _)| *i)
            .collect(),
    )
}

/// MONKEY (static torch cache): the nearest promoted fixtures to the player receive entities.
/// Stable physical-slot tie breaking keeps coincident/equidistant candles deterministic; this
/// subset is independent of contribution ranking and never moves the resident static maps.
fn dynamic_set(fixtures: &[(usize, Vec3)], anchor: Vec3, budget: usize) -> u32 {
    let mut nearest = fixtures.to_vec();
    nearest.sort_by(|a, b| {
        a.1.distance_squared(anchor)
            .total_cmp(&b.1.distance_squared(anchor))
            .then(a.0.cmp(&b.0))
    });
    // MONKEY (live bank rank): also clamp resource values that bypass the cvar setter.
    nearest
        .iter()
        .take(budget.clamp(1, MAX_TORCH_DYNAMIC))
        .fold(0, |mask, (i, _)| mask | (1 << i))
}

/// MONKEY (torch lane perf): the radius the MOVING-caster gather admits parts inside, for a
/// fixture of effective reach `reach`. See [`TORCH_ENTITY_REACH_MARGIN`] for why `reach + margin`
/// is the honest bound and not a tightening of the effect; clamped to [`TORCH_RANGE`] because
/// nothing outside the cube projection can be recorded anyway.
fn entity_gather_radius(reach: f32) -> f32 {
    if !reach.is_finite() {
        return TORCH_RANGE;
    }
    (reach + TORCH_ENTITY_REACH_MARGIN).clamp(TORCH_ENTITY_REACH_MARGIN, TORCH_RANGE)
}

/// MONKEY (torch lane perf): is `origin` inside ANY of the gather's fixtures' own radii — the
/// UNION that replaced "48 yd of every dynamic fixture". Split out as a pure function because it is
/// the whole admission rule of the gather and the one part of it worth a test.
fn within_any_reach(fixtures: &[GatherFixture], origin: Vec3) -> bool {
    fixtures
        .iter()
        .any(|(_, p, r, _)| p.distance_squared(origin) <= r * r)
}

/// MONKEY (torch owner exclusion): is this entity-lane caster part the fixture's OWN body?
///
/// The entity half of `static_gx`'s `torch_item_is_emitter`, and deliberately the same two rules in
/// the same order — a part must not cast or not cast depending on which lane happens to be drawing
/// its placement that frame:
///
///  1. OWNERSHIP. A placed light names its placement by content ([`LightOwner::Placement`]), and
///     the entity path's parts carry a CLONE of that identity, so a doodad still in its feather
///     band (or one the retained lane never took) is excluded exactly as its retained twin is. A
///     carried light names its host model FRAME instead ([`LightOwner::Instance`] — a GameObject
///     brazier, an NPC's torch have no placement identity at all), and a part is its body iff it
///     hangs under that frame.
///  2. CONTAINMENT of the flame in the part's own bound, shared verbatim with the retained lane
///     ([`torch_flame_inside_bounds`], including the fixture-size cap that stops a room-sized batch
///     from claiming every candle in it).
///
/// Excluded if ANY fixture of the gather owns it: the moving gather feeds ONE mesh to up to eight
/// fixtures, so per-fixture exclusion is not representable there. The cost is that a self-lit
/// creature stops casting for the other fixtures of the same dynamic set — a shadow that is at
/// worst missing, against a self-shadow artefact that is always visible.
///
/// **Rule 2 never applies to a CREATURE.** A body standing against a wall torch has that torch's
/// flame inside its own bounding box perfectly often, and a player who walks up to a brazier and
/// loses their shadow is a worse bug than the one this function exists to fix. A creature that
/// really does carry its own light (an imp's hand fire, a fire elemental, a torch-bearing guard)
/// is covered by rule 1 instead, which is exact — `entities::carried_light` tags every one of
/// those lights with the frame it hangs under.
fn part_is_own_body(
    ents: &EntityCasters,
    fixtures: &[GatherFixture],
    entity: Entity,
    kind: ModelKind,
    object: Option<&WorldObject>,
    global: Option<&GlobalTransform>,
    aabb: Option<&Aabb>,
) -> bool {
    let bound = (kind != ModelKind::Creature)
        .then_some(())
        .and(global.zip(aabb));
    fixtures.iter().any(|(_, pos, _, owner)| {
        let owned = match owner.map(|o| (o, o.instance())) {
            Some((_, Some(frame))) => hangs_under(ents, entity, frame),
            Some((o, None)) => object.is_some_and(|obj| o.owns(obj)),
            None => false,
        };
        owned
            || bound.is_some_and(|(g, b)| {
                torch_flame_inside_bounds(g.affine(), b.center.into(), b.half_extents.into(), *pos)
            })
    })
}

/// MONKEY (torch owner exclusion): is `entity` `frame`, or a descendant of it? See
/// [`OWNER_WALK_DEPTH`] for the cap.
fn hangs_under(ents: &EntityCasters, entity: Entity, frame: Entity) -> bool {
    let mut at = entity;
    for _ in 0..OWNER_WALK_DEPTH {
        if at == frame {
            return true;
        }
        let Ok(parent) = ents.parents.get(at) else {
            return false;
        };
        at = parent.parent();
    }
    false
}

/// MONKEY (torch lane perf): is a REGATHER of the moving-caster mesh due this frame?
///
/// `rate == 0` is the pre-feature every-frame behaviour, kept as the live A/B. A changed dynamic
/// SET always forces one — the mesh is only valid for the fixtures it was gathered around, and
/// keeping a stale one there would put a shadow in a room the overlay no longer serves. `elapsed`
/// is compared against `1/rate` with a small tolerance so a 30 Hz cadence on a 60 fps frame lands
/// on every other frame rather than alternating 1 and 3 frames as the two clocks beat.
fn entity_gather_due(rate: u32, elapsed: f64, mask: u32, last_mask: u32) -> bool {
    rate == 0 || mask != last_mask || elapsed >= 1.0 / f64::from(rate) - 1e-4
}

/// MONKEY (torch caster selection): the entity-caster query trio, bundled as ONE `SystemParam`.
/// Not tidiness — [`update_torch_shadows`] sat at 15 of Bevy's 16 system params before this change
/// added the player anchor and the interior knobs, and three separate members would not compile.
#[allow(clippy::type_complexity)]
#[derive(bevy::ecs::system::SystemParam)]
struct EntityCasters<'w, 's> {
    /// The SAME query the world lane passes to `collect_entity_geometry` (written inline by that
    /// contract so a lane can pass its own).
    parts: Query<
        'w,
        's,
        (
            Entity,
            &'static PickMesh,
            &'static ModelPart,
            Option<&'static GlobalTransform>,
            Option<&'static RigPart>,
            &'static ShadowOccluder,
            Option<&'static MeshMaterial3d<WowModelMaterial>>,
            // MONKEY (torch owner exclusion): the placement identity the entity lane carries (a
            // CLONE of the retained lane's `Arc`, which is why the owner key compares by content),
            // and the model-local bound the containment fallback needs. Both `Option`: a part that
            // has neither is simply never excluded, which is the safe failure direction.
            Option<&'static WorldObject>,
            Option<&'static Aabb>,
        ),
        Without<BillboardCard>,
    >,
    /// MONKEY (torch owner exclusion): the hierarchy walk for a CARRIED light's owner — a brazier
    /// GameObject's light names its model frame, and its own meshes are that frame's descendants.
    parents: Query<'w, 's, &'static ChildOf>,
    rigs: Query<'w, 's, &'static RigSkin>,
    palettes: Res<'w, RigPalettes>,
    distance: Res<'w, ShadowDistance>,
    /// MONKEY (outdoor torch shadows): the live celestial sun, for the night gate. It rides THIS
    /// bundle rather than being a 17th system param because `update_torch_shadows` already sits at
    /// Bevy's 16-param ceiling — the same reason `distance` is here.
    lighting: Res<'w, WowLighting>,
    /// MONKEY (daylight: terrain torch casters): the resident ADT tiles, for the ground half of an
    /// exterior slot's static caster mesh (`torchTerrainShadows`). `Option` so a scene with no
    /// terrain streamer (the glue, a booth) is simply terrain-less.
    terrain: Option<Res<'w, benilla_world::terrain_stream::TerrainStreamer>>,
    adt_tiles: Option<Res<'w, Assets<benilla_assets::AdtTile>>>,
}

/// MONKEY (daylight: terrain torch casters): does an EXTERIOR slot's static map include the ground?
/// The cvar `torchTerrainShadows`, or `WOW_TORCH_TERRAIN=0|1` (read once) for hermetic captures.
fn terrain_casters_on(video: &VideoConfig) -> bool {
    static ENV: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    ENV.get_or_init(
        || match std::env::var("WOW_TORCH_TERRAIN").ok().as_deref() {
            Some("1") => Some(true),
            Some("0") => Some(false),
            _ => None,
        },
    )
    .unwrap_or(video.torch_terrain_shadows)
}

/// The INTERIOR lane is active when the dynamic-interior lane is on AND its shadow toggle is on.
fn interior_shadows_on(video: &VideoConfig) -> bool {
    video.interior_light && video.interior_shadows
}

/// MONKEY (outdoor torch shadows): the EXTERIOR lane's gate — its own cvar AND night.
///
/// Deliberately independent of `interiorLight`: the exterior receivers (`wmo_exterior_point_sum`
/// on a WMO's outdoor-class surfaces, `point_light_sum` on doodads/entities) have never been part
/// of the dynamic-interior feature and are drawn with `interiorLight 0` exactly as with it, so
/// tying their shadows to that dial would be surprising in both directions.
///
/// Night is a HARD gate, not a fade to zero, because it is what buys the "daytime is bit-identical"
/// claim on the CPU side as well as in the shaders: with the sun up there are no exterior
/// candidates, so no exterior slot, so no exterior caster mesh, no depth render, and the published
/// `exterior` bit is 0. `sun_shadow_strength` saturates at exactly 1.0 above ~12° of elevation, so
/// the test is the same threshold the receivers' own `night_w` crosses — the two cannot disagree.
fn exterior_shadows_on(video: &VideoConfig, sun_strength: f32) -> bool {
    video.exterior_shadows && sun_strength < 1.0
}

/// A reverse-Z perspective (near → 1, far → 0; wgpu clip, RH, looking down −Z) — matches the whole
/// engine's reverse-Z convention and the torch depth pipeline's `GreaterEqual` compare. Built in f64.
fn reverse_z_perspective(fov_y: f64, aspect: f64, near: f64, far: f64) -> DMat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    // Column-major columns.
    DMat4::from_cols(
        DVec4::new(f / aspect, 0.0, 0.0, 0.0),
        DVec4::new(0.0, f, 0.0, 0.0),
        DVec4::new(0.0, 0.0, near / (far - near), -1.0),
        DVec4::new(0.0, 0.0, (far * near) / (far - near), 0.0),
    )
}

/// MONKEY (Phase 5): the six cube-face `view_proj`s of a fixture — a 90°(+ε) reverse-Z perspective
/// looking down each of ±X, ±Y, ±Z — so the maps cover EVERY direction around it. This replaced the
/// single aimed cone: any single cone has an edge, and both the vertical (walk up to the forge) and
/// lateral (stand beside the player) shadow collapses were that edge being crossed. Face order is
/// the contract with `static_gx.wgsl`'s `torch_face`: 0 +X, 1 −X, 2 +Y, 3 −Y, 4 +Z, 5 −Z. Computed
/// in f64 because fixtures sit at absolute Bevy world coords (~9,300 out) where f32 view-matrix
/// arithmetic loses precision. Nothing here depends on the camera or the player any more.
fn cube_view_projs(fixture: Vec3) -> [Mat4; CUBE_FACES] {
    let eye = DVec3::new(fixture.x as f64, fixture.y as f64, fixture.z as f64);
    let proj = reverse_z_perspective(TORCH_FACE_FOV as f64, 1.0, 0.1, TORCH_RANGE as f64);
    let faces: [(DVec3, DVec3); CUBE_FACES] = [
        (DVec3::X, DVec3::Y),
        (DVec3::NEG_X, DVec3::Y),
        (DVec3::Y, DVec3::NEG_Z),
        (DVec3::NEG_Y, DVec3::NEG_Z),
        (DVec3::Z, DVec3::Y),
        (DVec3::NEG_Z, DVec3::Y),
    ];
    faces.map(|(dir, up)| {
        let vp = proj * DMat4::look_to_rh(eye, dir, up);
        Mat4::from_cols_array(&vp.to_cols_array().map(|v| v as f32))
    })
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn update_torch_shadows(
    video: Res<VideoConfig>,
    state: Res<State<ClientState>>,
    mut demand: ResMut<ShadowDemand>,
    frame: Res<ShadowFrame>,
    mut lane: ResMut<TorchLane>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut views: ResMut<TorchShadowViews>,
    cameras: Query<&GlobalTransform, With<WorldCamera>>,
    // MONKEY (torch caster selection): the RANKING ANCHOR. Shadows are for the player's own
    // surroundings — a free-look camera swung across the room must not re-pick the casters — so
    // scoring is done at the body, and only falls back to the camera when there is none (the glue
    // scene, a capture fixture, the frame before the player streams in).
    player: Query<&GlobalTransform, With<SelfPlayer>>,
    // Interior WMO fixtures: a `PointLight` carrying a room claim. MONKEY (torch caster selection):
    // the intensity/colour and the authored reach ride along now — the score is the light's own
    // contribution, so the lane must read exactly what the packer reads.
    // MONKEY (review fixes): containment is membership evidence even with no MOLR; keep the
    // visibility membership separate so the missing-lane fallback stays identical to the packer.
    // MONKEY (carried light stability): `Has<ChildOf>` + `Option<&CarriedLightMotion>` ride the
    // tuple so the filter below can refuse a MOVING carried light a slot. Both are cheap table
    // reads on a query that already walks these entities.
    // MONKEY (outdoor torch shadows): the FILTER moved into the closure. It used to be
    // `Or<(With<LightRooms>, With<LightLitRooms>)>` — "has a room claim" — which is exactly the set
    // of fixtures that can be interior, and therefore excluded every outdoor campfire from the
    // query outright. The filter is now the packer's own (`Without<ShadowProxyLight>`: a proxy is a
    // `PointLight` that exists only to cast, never to light), and the room-claim test is applied
    // per-candidate on the INTERIOR arm below, so the interior candidate set is unchanged
    // member-for-member. `Has<LightLitRooms>` rides the tuple to reproduce the other half of the
    // retired `Or`, and `Has<HeldLight>` to refuse a body-carried torch an EXTERIOR slot.
    // MONKEY (torch owner exclusion): `Option<&LightOwner>` — who this fixture is part of, written
    // at the two light spawn sites (`terrain_stream::spawn`'s `tag_light_owner`,
    // `entities::carried_light`). It rides the candidate and then the slot, because the caster
    // gathers need it and they run far downstream of this query.
    torches: Query<
        (
            Entity,
            &GlobalTransform,
            &WorldPointLight,
            Option<&LightReach>,
            Option<&LightLane>,
            Has<LightRooms>,
            Has<LightLitRooms>,
            Has<SyntheticFireLight>,
            Has<ChildOf>,
            Has<HeldLight>,
            Option<&crate::entities::CarriedLightMotion>,
            Option<&LightOwner>,
        ),
        // MONKEY (daylight fixtures): a doorway's daylight source is NOT a caster candidate. It is
        // an AREA source the width of the opening, standing in a hole in a wall — a point-cube
        // shadow of it would be wrong in kind (hard radial wedges from a soft sky) — and being
        // bright and close to the player the moment they walk in, it would outrank the room's real
        // fixtures for the twelve cube slots and take their shadows away.
        (Without<ShadowProxyLight>, Without<DaylightFixture>),
    >,
    // MONKEY (torch caster selection): `atten_scale` — a fixture's EFFECTIVE radius is its authored
    // end times this live cvar, and the score's window is a fraction of that radius. Reading the
    // same resource the packer folds into the shader's `.w` keeps ranking and rendering agreed.
    interiors: Res<DynamicInteriors>,
    fire_gain: Res<FireLightGain>,
    gx: Option<Res<StaticGx>>,
    // MONKEY (Phase 3B): the entity caster inputs, bundled (see [`EntityCasters`]).
    ents: EntityCasters,
    time: Res<Time>,
    mut last_trace: Local<f64>,
) {
    // Declare demand so the shared rig stays up while on.
    // MONKEY (outdoor torch shadows): EITHER lane keeps the lane alive — they share the sixteen
    // cube slots, the rebuild budget and the depth bank, so there is one system, not two.
    let sun_w = sun_shadow_strength(ents.lighting.celestial_dir().y);
    let interior_on = interior_shadows_on(&video);
    let exterior_on = exterior_shadows_on(&video, sun_w);
    let on = (interior_on || exterior_on) && *state.get() == ClientState::InWorld;
    demand.0 = demand.0 || on;

    if !(frame.active && on) {
        teardown(&mut lane, &mut meshes, &mut views);
        return;
    }
    let Some(cam) = cameras.iter().next().map(GlobalTransform::translation) else {
        return;
    };

    // (1) MONKEY (torch caster selection): who casts. Three rules, in this order.
    //
    //   RANK BY CONTRIBUTION, NOT DISTANCE. The old lane took the four fixtures nearest the camera
    //   and hard-switched the set every frame. In a candle-dense room that set churns every few
    //   steps — Northshire Abbey authors 42 candelabra about 5 yd apart — and every fixture NOT
    //   in it lit with no shadow at all, so shadows visibly popped in as you approached a candle
    //   and out again as you passed it. Scoring by the light's own direct term at the PLAYER makes
    //   the ranking a smooth function of position instead of a step function, and makes it right:
    //   a bright forge across the room outranks a guttering candle at your feet, which is what the
    //   eye expects of a shadow.
    //
    //   HYSTERESIS. A smooth score still crosses. An incumbent keeps its slot unless a challenger
    //   beats it by TORCH_SWAP_RATIO, and at most one contested swap starts per
    //   TORCH_SWAP_COOLDOWN — so walking the candle corridor does not trade slots every step.
    //
    //   FADE, NOT SWITCH. A promoted slot ramps its weight 0 to 1; a losing one ramps 1 to 0 and is
    //   only REPLACED at 0 (it keeps rendering while it fades). The receivers apply the weight as
    //   `mix(1, shadow, w)`, so a shadow dissolves rather than blinking.
    let anchor = player
        .iter()
        .next()
        .map(GlobalTransform::translation)
        .unwrap_or(cam);
    let want = (video.interior_shadow_casters as usize).clamp(1, MAX_TORCH_CASTERS);
    let dt = time.delta_secs();
    let now = time.elapsed_secs_f64();
    // MONKEY (moving fixture): who held a live rebuild budget LAST frame. Read before the candidate
    // closure because the closure needs it and `lane` is borrowed mutably from (1b) on; last
    // frame's answer is the right one anyway — the point of the test is incumbency.
    let live_held: Vec<Entity> = lane
        .slots
        .iter()
        .filter(|s| s.moving_live)
        .map(|s| s.fixture)
        .collect();

    // (1a) MONKEY (torch caster reach): each pool is eligible from 4R + slack away. Rank with the
    // same soft core/intensity so nearby contributors still win over pools seen across the room.
    // MONKEY (outdoor torch shadows): the candidate tuple carries its LANE (`true` = exterior).
    let mut cands: Vec<Candidate> = torches
        .iter()
        .filter_map(
            |(
                e,
                gt,
                pl,
                reach,
                light_lane,
                has_rooms,
                has_lit,
                synthetic,
                carried,
                held,
                motion,
                owner,
            )| {
                // MONKEY (carried light stability): a CARRIED light (`ChildOf` — a pet's hand flame,
                // an NPC's torch, a transport's deck brazier) used to cast ONLY while standing still,
                // because a cached cube is valid only while its fixture stays within
                // `TORCH_STALE_DRIFT_SQ` of where it was baked ([`map_publishable`]) and a moving one
                // toggled a full-strength shadow at frame rate.
                //
                // MONKEY (moving fixture): that ban is lifted, because the lane can now RE-RENDER a
                // moving fixture's whole cube every frame instead of republishing a frozen one (see
                // [`TorchSlot::moving`]). What survives of the old rule is its arithmetic: only
                // [`TORCH_MOVING_MAX`] fixtures can be afforded that way, so a moving carried light is
                // admitted as a candidate only if it already HOLDS one of those budgets or one is free.
                // Doing it here, at candidacy, rather than by promoting it and evicting it a moment
                // later, is what stops a third walking pet from churning the slot machinery: it never
                // enters `cands`, so nothing downstream has to fade it out and refill after it.
                //
                // The stationary members of the same `ChildOf` family are untouched — a placed brazier
                // GameObject or a campfire is `settled` and takes the ordinary cached path.
                if carried
                    && !motion.is_some_and(CarriedLightMotion::settled)
                    && !live_held.contains(&e)
                    && live_held.len() >= TORCH_MOVING_MAX
                {
                    return None;
                }
                let p = gt.translation();
                let d = p.distance(anchor);
                if p.distance(cam) > TORCH_SEARCH_RADIUS.max(ents.distance.0) {
                    return None;
                }
                // The packer's own recovery of authored colour x intensity (`spawn_point_light`
                // premultiplied 4*pi), and the packer's own reach: the authored MOLT end where there is
                // one, the intensity bucket otherwise, times the live `interiorAttenScale`.
                let base = pl.intensity / (4.0 * std::f32::consts::PI);
                let authored = reach
                    .map(|r| r.0)
                    .filter(|r| *r > 0.5)
                    .unwrap_or_else(|| m2_light_reach(base));
                let r = interior_reach(authored, interiors.atten_scale);
                // MONKEY (outdoor torch shadows): the `4R + slack` eligibility test moved DOWN into the
                // interior arm (unchanged there). It cannot be shared: `r` is an INTERIOR reach — the
                // authored MOLT end or the M2 intensity bucket times `interiorAttenScale` — and an
                // exterior entry packs no reach at all, so applying it out here would size an outdoor
                // campfire's promotion window off a number no exterior receiver ever reads.
                let c = pl.color;
                // Rec.709 luminance of the committed colour: ONE scalar for "how much light is this",
                // so a dim blue magic brazier cannot outrank a bright hearth on channel count.
                let lum = (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]).max(0.0) * base.max(0.0);
                // MONKEY (outdoor torch shadows): the lane split. It is the SAME verdict the light
                // packer writes into the colour row's `.w` (`LightLane` — a light is interior iff it
                // physically stands in an interior-class WMO group, with the old `LightRooms` rule as
                // the pre-classification fallback), because that is what decides which receiver family
                // reads it and therefore which shadow lane can ever be sampled for it.
                let interior = light_lane.map_or(has_rooms, |l| l.interior);
                if interior {
                    // The interior arm, unchanged — including the retired query filter's own
                    // membership test, so this candidate set is identical member-for-member to the one
                    // `Or<(With<LightRooms>, With<LightLitRooms>)>` produced.
                    if !interior_on || !(has_rooms || has_lit) {
                        return None;
                    }
                    if !fixture_eligible(d, p.distance(cam), r, ents.distance.0) {
                        return None;
                    }
                    let score =
                        candidate_score(lum, d, r, light_lane, has_rooms, synthetic, fire_gain.0);
                    // MONKEY (torch lane perf): `r` rides along — the moving-caster gather sizes its
                    // radius off the light that will be sampled, not off the cube's 48 yd far plane.
                    return (score > 0.0).then_some((e, p, score, false, r, owner.copied()));
                }
                // The EXTERIOR arm — a campfire, a brazier, a lamppost, a bonfire.
                if !exterior_on || held {
                    return None;
                }
                if !exterior_eligible(d, p.distance(cam), ents.distance.0) {
                    return None;
                }
                // Same synthetic-gain fold as the interior arm (and as the packer): dimming the
                // invented lights with `fireLightGain 0` must take their shadows with them, or a fire
                // that lights nothing would still carve a black wedge across the ground.
                let gain = if synthetic { fire_gain.0.max(0.0) } else { 1.0 };
                let score = exterior_fixture_score(lum * gain, d);
                // MONKEY (torch lane perf): an exterior entry packs no reach and its receivers apply an
                // un-windowed falloff, so the cube's own range is the only honest gather radius here.
                (score > 0.0).then_some((e, p, score, true, TORCH_RANGE, owner.copied()))
            },
        )
        .collect();
    cands.sort_by(|a, b| b.2.total_cmp(&a.2));
    let ext_cap = exterior_budget(want);
    cap_exterior_candidates(&mut cands, ext_cap);

    // (1b) Refresh the incumbents against this frame's candidates. A fixture that dropped out of
    // the set (it moved away, despawned, its room streamed out) starts fading rather than
    // vanishing. A slot past the live budget does the same, so lowering `interiorShadowCasters`
    // fades its extra shadows out instead of cutting them.
    for (i, slot) in lane.slots.iter_mut().enumerate() {
        match cands.iter().find(|(e, ..)| *e == slot.fixture) {
            Some((_, p, sc, _, r, owner)) => {
                slot.pos = *p;
                slot.score = *sc;
                slot.reach = *r;
                // The owner tag can only ever LAND (it is written at spawn and never removed), so
                // refreshing it here is what lets a fixture promoted on its very first frame pick
                // it up rather than casting its own body's shadow until it is next re-promoted.
                slot.owner = *owner;
            }
            None => {
                slot.score = 0.0;
                slot.evicting = true;
            }
        }
        if i >= want {
            slot.evicting = true;
        }
    }

    // (1b2) MONKEY (moving fixture): the motion test and the live-rebuild budget.
    //
    // The bug this answers: a carried light's slot kept publishing the cube it baked at `built_at`
    // while the light pool followed the pet, so the imp walked around the Darkshire inn dragging a
    // dark smear that belonged to where it had been. The eviction rule that produced it ("a drifted
    // slot on its way out publishes from `built_at` and fades") is correct for a fixture that is
    // LEAVING; it is simply the wrong rule for one that is merely walking, and the only right
    // answer for that one is to re-render it.
    //
    // Ordering matters: this sits between the incumbent refresh (which wrote this frame's `pos` and
    // `score`) and the weight ramp below (which needs to know who fades fast), so every slot's
    // verdict is made from fresh positions and acted on in the same frame it is made.
    for slot in lane.slots.iter_mut() {
        if slot.pos.distance_squared(slot.last_pos) > TORCH_MOVING_EPS_SQ {
            slot.still_since = now;
        }
        slot.last_pos = slot.pos;
    }
    let (live, dropped) = moving_budget(
        lane.slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.moving(now))
            .map(|(i, s)| (i, s.score))
            .collect(),
    );
    for (i, slot) in lane.slots.iter_mut().enumerate() {
        slot.moving_live = live.contains(&i);
        // A mover past the cap is withdrawn rather than kept: a frozen map on a fixture that is
        // still walking is a shadow in the wrong place, which is worse than no shadow. It still
        // FADES (over 0.15 s) rather than blinking, and it still publishes from `built_at` while it
        // does, which is what the `evicting` flag buys it in the publish step below.
        slot.fast_fade = dropped.contains(&i);
        if slot.fast_fade {
            slot.evicting = true;
        }
    }

    // (1c) Ramp the weights, then release the slots that finished fading out.
    let step = TORCH_FADE_RATE * dt;
    for slot in lane.slots.iter_mut() {
        // MONKEY (moving fixture): a dropped mover leaves five times faster — see the const.
        let step = if slot.fast_fade {
            TORCH_MOVING_FADE_RATE * dt
        } else {
            step
        };
        slot.w = if slot.evicting {
            (slot.w - step).max(0.0)
        } else {
            (slot.w + step).min(1.0)
        };
    }
    lane.slots.retain(|s| {
        let release = s.evicting && s.w <= 0.0;
        if release {
            if let Some(mesh) = &s.mesh {
                meshes.remove(mesh.id());
            }
        }
        !release
    });

    // (1d) Fill free slots from the best unassigned candidates. A newly filled slot starts at
    // w = 0 and fades IN. Note this can only run once a fading slot has fully drained, which is
    // what stops a churn from being visible even when the SELECTION churns.
    // MONKEY (static torch cache): at most two new residents per frame, including startup.
    let mut promotions = 0;
    while lane.slots.len() < want && promotions < 2 {
        let Some(&(e, p, sc, exterior, reach, owner)) = cands
            .iter()
            .find(|(e, ..)| !lane.slots.iter().any(|s| s.fixture == *e))
        else {
            break;
        };
        let cache_slot = (0..MAX_TORCH_CASTERS)
            .find(|i| !lane.slots.iter().any(|s| s.cache_slot == *i))
            .unwrap();
        promotions += 1;
        lane.slots.push(TorchSlot {
            cache_slot,
            mesh: None,
            geometry_key: None,
            built_at: None,
            fixture: e,
            pos: p,
            score: sc,
            reach,
            checked: None,
            exterior,
            owner,
            trace_excluded: (0, 0),
            // MONKEY (moving fixture): born settled — `still_since` at 0 is "moved long ago", so
            // the first frame takes the ordinary (queue-jumping) cached build and only an observed
            // move promotes the slot onto the live budget.
            last_pos: p,
            still_since: 0.0,
            moving_live: false,
            fast_fade: false,
            w: 0.0,
            evicting: false,
        });
    }

    // (1e) The contested swap: the best unassigned candidate takes the weakest incumbent's slot
    // only if it beats it by the ratio AND the cooldown has elapsed. Marking `evicting` rather
    // than replacing is the whole of "fade, not switch" — (1c) drains it and (1d) refills it a
    // fade later, by which time the challenger has been re-scored at the player's new position.
    if now - lane.last_swap >= TORCH_SWAP_COOLDOWN {
        let weakest = lane
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.evicting)
            .min_by(|a, b| a.1.score.total_cmp(&b.1.score))
            .map(|(i, s)| (i, s.score));
        let challenger = cands
            .iter()
            .find(|(e, ..)| !lane.slots.iter().any(|s| s.fixture == *e))
            .map(|(_, _, sc, ..)| *sc);
        if let (Some((idx, weak)), Some(best)) = (weakest, challenger) {
            if best > weak * TORCH_SWAP_RATIO {
                lane.slots[idx].evicting = true;
                lane.last_swap = now;
            }
        }
    }

    // The published set, in slot order: fixture, position, score (trace only) and fade weight,
    // capped at the table's allocation.
    // MONKEY (outdoor torch shadows): …and each slot's LANE, for the caster gather's self-exclusion
    // radius, the published `exterior` bit and the trace.
    let promoted: Vec<(usize, Entity, Vec3, f32, f32, bool)> = lane
        .slots
        .iter()
        .take(MAX_TORCH_CASTERS)
        .map(|s| (s.cache_slot, s.fixture, s.pos, s.score, s.w, s.exterior))
        .collect();

    // (2) MONKEY (static torch cache): fingerprint resident SOURCE geometry per fixture, not
    // StaticGx's change tick (frame bookkeeping changes it continuously). This catches arrivals,
    // removals and replacements even with a stationary camera. Entity-resident rigid furniture
    // joins the static mesh; creatures and skinned/animated parts belong to the dynamic overlay.
    let fixture_positions: Vec<_> = promoted.iter().map(|(i, _, p, ..)| (*i, *p)).collect();
    // MONKEY (moving fixture): a live slot is dynamic BY DEFINITION — the moving entities are half
    // of what makes its map wrong when it is frozen (the pet's own body is usually excluded as its
    // own owner, but everyone it walks past is not), and the overlay is where they are drawn. OR'd
    // in rather than folded into `dynamic_set`'s nearest-N so the cvar keeps meaning exactly what
    // it says; `torch_cache_plan` still clamps the union to the eight-cube live bank.
    let moving_mask = lane
        .slots
        .iter()
        .filter(|s| s.moving_live)
        .fold(0u32, |mask, s| mask | (1 << s.cache_slot));
    let dynamic = dynamic_set(
        &fixture_positions,
        anchor,
        video.interior_shadow_dynamic as usize,
    ) | moving_mask;
    let mut rebuilt = 0;
    // MONKEY (moving fixture): the live slots' own rebuild allowance, kept apart from `rebuilt`.
    let mut moving_rebuilt = 0;
    let mut static_rebuilt = 0u32;
    // MONKEY (torch lane perf): how many fingerprint walks the gate actually let through this
    // frame, folded into the lane's per-second trace below (`lane` is mutably borrowed inside the
    // loop, so the counter cannot live on it until the loop has finished).
    let mut lane_scans = 0u32;
    let mut published = TorchShadowViews {
        count: promoted
            .iter()
            .map(|(i, ..)| *i as u32 + 1)
            .max()
            .unwrap_or(0),
        soft: video.interior_shadow_soft,
        // MONKEY (shadow floor): the direct-term floor, packed beside `soft` in `count.y`.
        strength: video.torch_shadow_strength,
        dynamic_mask: dynamic,
        // MONKEY (moving fixture): the render plan's own budget follows the same set.
        moving_mask,
        // MONKEY (outdoor torch shadows): the receivers' one-bit gate — the cvar, the sun AND an
        // actual exterior slot to sample. Keyed on the SLOTS rather than on `exterior_on` alone so
        // that a night with the cvar on but nothing promoted (no fires in range) costs the
        // receivers nothing at all, which is most of the outdoor world.
        exterior: exterior_on && promoted.iter().any(|(.., ext)| *ext),
        ..Default::default()
    };
    // MONKEY (static torch cache): rotate service order so two moving/carried fixtures cannot
    // consume the rebuild budget forever and starve a newly resident room's static maps. The
    // rotation bounds the SCAN as well as the rebuild (see TORCH_SCAN_PER_FRAME): fingerprinting a
    // fixture's surroundings is the one part of this lane that is O(resident scene), so doing it
    // for every slot every frame would be the new cost ceiling — and at most two of those slots
    // could act on the answer anyway.
    let slot_count = lane.slots.len();
    // MONKEY (torch lane perf): the scene-wide census, computed ONCE for the whole lane rather
    // than implicitly once per scanned slot. Everything a fingerprint hashes lives in exactly two
    // places, and both are summarised here far more cheaply than they are hashed: the retained
    // regions' change stamps and populations, and one branch-free mix per rigid opaque model part.
    // While this pair is unchanged the fixture-local source set cannot have changed either, so the
    // slot's walk is skipped outright — which is what turns "a full scene scan every frame,
    // forever, in a room where nothing ever moves" into "a scan the frame something does".
    // MONKEY (daylight: terrain torch casters): the resident terrain set joins the retained
    // scene's stamp, so a tile streaming in re-checks the exterior slots near it; the switch itself
    // is folded in so toggling the cvar re-keys every slot.
    let terrain_on = terrain_casters_on(&video);
    let terrain_gen = match (&ents.terrain, &ents.adt_tiles) {
        (Some(t), Some(a)) if terrain_on => {
            benilla_world::terrain_stream::terrain_torch_generation(t, a) ^ 0x7e44_a1d0
        }
        _ => 0,
    };
    let census = if slot_count > 0 {
        PartCensus {
            gx: gx
                .as_ref()
                .map_or(0, |gx| gx.torch_residency_generation())
                .wrapping_add(terrain_gen),
            parts: static_part_census(&ents),
        }
    } else {
        PartCensus::default()
    };
    // A slot that has NEVER been fingerprinted jumps the queue: with one scan a frame a fresh
    // promotion would otherwise wait a whole sweep for its first map, and a slot with no map
    // publishes none (`map_publishable`), so that wait is a shadow that is simply missing. There
    // are at most two promotions a frame, so this can never crowd the rotation out for long.
    let start = lane
        .slots
        .iter()
        .position(|s| s.checked.is_none())
        .unwrap_or_else(|| {
            if slot_count == 0 {
                0
            } else {
                lane.rebuild_cursor % slot_count
            }
        });
    let scan = slot_count.min(TORCH_SCAN_PER_FRAME);
    // MONKEY (moving fixture): a live slot is scanned EVERY frame, in addition to (and ahead of)
    // the cursor's window. It cannot wait its turn in a rotation that takes a dozen frames to come
    // round: its fingerprint changes every frame by construction (`slot.pos` is hashed into it), so
    // "wait for the cursor" and "publish a map baked somewhere else for twelve frames" are the same
    // sentence. The two budgets stay separate below, so this never costs the resident slots their
    // own rebuild.
    let mut scanning: Vec<usize> = lane
        .slots
        .iter()
        .enumerate()
        .filter(|(_, s)| s.moving_live)
        .map(|(i, _)| i)
        .collect();
    for offset in 0..scan {
        let index = (start + offset) % slot_count;
        if !scanning.contains(&index) {
            scanning.push(index);
        }
    }
    for index in scanning {
        let slot = &mut lane.slots[index];
        let i = slot.cache_slot;
        // MONKEY (moving fixture): this slot's verdict, read once — it picks the gather radius,
        // the rebuild budget and whether the mesh asset is reused in place.
        let moving = slot.moving_live;
        // MONKEY (moving fixture): a moving fixture gathers static geometry inside its OWN reach
        // rather than the cube's flat 48 yd. The receivers' direct term is exactly zero past
        // `reach` (`interior_window`), so everything between the two radii could only ever shadow
        // fragments this fixture does not light — for a cached slot that waste is paid once and
        // does not matter, but this slot pays it EVERY frame, and an imp's flame reaches ~10 yd
        // where the cube reaches 48 (a hundredth of the volume). Settled slots keep the full range
        // verbatim, so no cached key or mesh changes meaning.
        let range = if moving {
            entity_gather_radius(slot.reach)
        } else {
            TORCH_RANGE
        };
        // MONKEY (torch lane perf): the skip. The fixture's own position joins the census because
        // it is hashed INTO the fingerprint (a carried torch moves without the world changing),
        // and a slot is only ever stamped once its key is actually committed below — a rebuild
        // deferred by the budget must come back and try again, not be certified as looked at.
        if slot.checked == Some((census.gx, census.parts, slot.pos)) {
            continue;
        }
        lane_scans += 1;
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        slot.fixture.hash(&mut hash);
        slot.pos.to_array().map(f32::to_bits).hash(&mut hash);
        // MONKEY (torch owner exclusion): the owner is part of the ADMISSION, so it is part of the
        // FINGERPRINT by construction — a key computed over a different source set than the mesh
        // would certify the wrong geometry as current. (It needs no separate hash lane: the owner
        // is a function of `slot.fixture`, which is hashed above, and it never changes for a slot.)
        let owner = slot.owner;
        let gather = [(i, slot.pos, range, owner)];
        gx.as_ref()
            .map(|gx| gx.torch_geometry_key(slot.pos, range, owner))
            .hash(&mut hash);
        let mut entity_key = 0u64;
        for (entity, pick, part, global, rig, occluder, _, object, aabb) in &ents.parts {
            if !static_part(part, rig.is_some()) || !occluder.0 {
                continue;
            }
            let Some(global) = global else { continue };
            if global.translation().distance_squared(slot.pos) > range * range {
                continue;
            }
            // The same exclusion the gather below applies, or the key would certify a mesh that
            // was never built from this set.
            if part_is_own_body(
                &ents,
                &gather,
                entity,
                part.kind,
                object,
                Some(global),
                aabb,
            ) {
                continue;
            }
            // Arc identity catches geometry replacement, transform bits catch moved furniture.
            let mut part_hash = std::collections::hash_map::DefaultHasher::new();
            entity.hash(&mut part_hash);
            (std::sync::Arc::as_ptr(&pick.0) as usize).hash(&mut part_hash);
            global
                .to_matrix()
                .to_cols_array()
                .map(f32::to_bits)
                .hash(&mut part_hash);
            entity_key = entity_key.wrapping_add(part_hash.finish());
        }
        entity_key.hash(&mut hash);
        // MONKEY (daylight: terrain torch casters): settled EXTERIOR slots only. An interior
        // fixture's room already occludes the ground, and a moving slot regathers every frame.
        let with_terrain = terrain_gen != 0 && slot.exterior && !moving;
        if with_terrain {
            // MONKEY (fix-daylight): only the tiles near THIS slot key it; the global stamp
            // re-rendered every exterior slot on any stream event anywhere.
            if let (Some(t), Some(a)) = (&ents.terrain, &ents.adt_tiles) {
                (benilla_world::terrain_stream::terrain_torch_generation_near(
                    t,
                    a,
                    slot.pos,
                    TORCH_RANGE,
                ) ^ 0x7e44_a1d0)
                    .hash(&mut hash);
            }
        }
        let key = hash.finish();
        let dirty = slot.geometry_key != Some(key);
        if !dirty {
            slot.checked = Some((census.gx, census.parts, slot.pos));
        }
        // MONKEY (moving fixture): two budgets, not one. A moving slot asks to rebuild every
        // frame, so sharing the resident slots' two would let one walking pet freeze every static
        // map in the room for as long as it walked.
        let afford = if moving {
            moving_rebuilt < TORCH_MOVING_MAX
        } else {
            rebuilt < 2
        };
        if dirty && afford {
            // MONKEY (moving fixture): recycle THIS slot's own buffers when it is live, so a
            // fixture that rebuilds sixty times a second does not allocate two Vecs a frame.
            let (mut positions, mut indices) = slot
                .mesh
                .as_ref()
                .filter(|_| moving)
                .and_then(|h| meshes.get_mut(h))
                .map(take_mesh_buffers)
                .unwrap_or_default();
            let gx_excluded = gx.as_ref().map_or(0, |gx| {
                gx.append_torch_triangles(slot.pos, range, owner, &mut positions, &mut indices)
            });
            let (_, part_excluded) =
                collect_torch_entities(&ents, true, &gather, &mut positions, &mut indices);
            if with_terrain {
                if let (Some(t), Some(a)) = (&ents.terrain, &ents.adt_tiles) {
                    let t0 = std::time::Instant::now();
                    let tris = benilla_world::terrain_stream::append_terrain_torch_triangles(
                        t,
                        a,
                        slot.pos,
                        range,
                        &mut positions,
                        &mut indices,
                    );
                    if std::env::var_os("WOW_TORCH_TRACE").is_some() {
                        info!("torch-terrain: slot {i} +{tris} ground tris within {range:.0} yd in {:.3} ms",
                            t0.elapsed().as_secs_f64() * 1e3);
                    }
                }
            }
            slot.trace_excluded = (gx_excluded, part_excluded);
            // MONKEY (moving fixture): a live slot MUTATES its mesh in place and keeps its asset
            // id; only a settled rebuild takes a fresh one. Two reasons. A new `Mesh` asset every
            // frame churns a vertex+index GPU allocation per frame per mover. And, worse, a brand
            // new asset is not certified resident on the frame it is added — `torch_cache_plan`
            // answers an unready mesh by clearing the slot's ready bit, which `TorchTable::pack`
            // turns into a zero cross-fade weight: the shadow would blink off on exactly the frames
            // the fixture is moving. With the id stable the mesh is always ready, and the render
            // plan rebuilds a `moving_mask` slot unconditionally instead of on an id change.
            match slot
                .mesh
                .as_ref()
                .filter(|_| moving)
                .and_then(|h| meshes.get_mut(h))
            {
                Some(mesh) => restore_mesh_buffers(mesh, positions, indices),
                None => {
                    let mut mesh = empty_shadow_mesh();
                    restore_mesh_buffers(&mut mesh, positions, indices);
                    if let Some(old) = slot.mesh.replace(meshes.add(mesh)) {
                        meshes.remove(old.id());
                    }
                }
            }
            slot.geometry_key = Some(key);
            slot.built_at = Some(slot.pos);
            slot.checked = Some((census.gx, census.parts, slot.pos));
            if moving {
                moving_rebuilt += 1
            } else {
                rebuilt += 1
            }
            static_rebuilt |= 1 << i;
        }
    }
    // Advance past everything scanned, not just what was rebuilt, so the window sweeps the whole
    // resident set at a steady rate instead of re-examining the head of it.
    lane.rebuild_cursor = (start + scan) % slot_count.max(1);

    // MONKEY (static torch cache): publish EVERY slot every frame — position, fade weight and the
    // six projections all move with the fixture and the cross-fade, and only the caster MESH is
    // cached. A slot whose map is stale (dirty, awaiting its rebuild turn) still publishes it; a
    // slot whose fixture has drifted away from where its map was baked publishes none, which forces
    // its weight to zero rather than showing shadows cast from a position the light has left.
    for slot in &lane.slots {
        let i = slot.cache_slot;
        // MONKEY (carried light stability): where this slot casts FROM this frame. Normally the
        // live fixture position. A slot that has drifted off its baked map AND is on its way out
        // publishes from `built_at` instead of withdrawing: withdrawing clears the slot's
        // `ready_mask` bit, and `TorchTable::pack` answers that by forcing the cross-fade weight
        // to 0 with no ramp — a hard shadow-off in one frame, which is the very pop the "fade,
        // not switch" law of this lane exists to remove. Frozen, the shadow dissolves over
        // `TORCH_FADE_RATE` from the last place it was correct; a third of a second later the
        // fixture is a stride away and the shadow is already gone, which reads as a shadow being
        // left behind rather than as a blink. A slot that is NOT evicting still withdraws — a
        // resident fixture mid-rebuild must never show shadows cast from a position it has left.
        let origin = if map_publishable(slot.built_at, slot.pos) {
            Some(slot.pos)
        } else {
            slot.built_at.filter(|_| slot.evicting)
        };
        if origin.is_some() {
            published.caster_meshes[i] = slot.mesh.as_ref().map(Handle::id);
        }
        let at = origin.unwrap_or(slot.pos);
        published.positions[i] = at.extend(slot.w);
        for (f, vp) in cube_view_projs(at).into_iter().enumerate() {
            published.view_projs[i * CUBE_FACES + f] = vp;
        }
    }
    // MONKEY (static torch cache): gather only around the dynamic subset, not every promoted
    // fixture. A single mesh serves those N maps and contains no rigid furniture already cached.
    //
    // MONKEY (torch lane perf): …and only within each of those fixtures' OWN reach, not within the
    // cube projection's flat 48 yd. The receivers' direct term is exactly zero past `reach`
    // (`interior_window`), so a caster admitted beyond it could only ever shadow fragments this
    // fixture does not light — it was pure cost. A candle's ~12 yd pool is a twentieth of the
    // volume the 48 yd radius swept, and this gather CPU-SKINS every part it admits.
    let dynamic_positions: Vec<GatherFixture> = lane
        .slots
        .iter()
        .take(MAX_TORCH_CASTERS)
        .filter(|s| dynamic & (1 << s.cache_slot) != 0)
        .map(|s| (s.cache_slot, s.pos, entity_gather_radius(s.reach), s.owner))
        .collect();
    // MONKEY (torch lane perf): and only at `interiorShadowEntityRate` Hz. The gather + the
    // `Mesh` mutation it feeds (a full vertex/index re-extraction, a GPU re-upload and an
    // `AssetChanged<Mesh3d>` fan-out through material specialisation) were paid EVERY frame
    // regardless of every count dial — the fixed charge that made `interiorShadowCasters 8` and
    // `interiorShadowDynamic 2` measure identically to the defaults. The mesh is RETAINED between
    // regathers and the six overlay passes still draw it every frame, so this ages the pose the
    // casters were skinned at; it never removes a shadow.
    if dynamic != 0 {
        let fresh = lane.entity_mesh.is_none();
        if fresh {
            lane.entity_mesh = Some(meshes.add(empty_shadow_mesh()));
        }
        // MONKEY (moving fixture): a live slot overrides the cadence. `interiorShadowEntityRate`
        // ages the POSE of the moving casters, which is invisible on a fixture that is standing
        // still (the shadow is in the right place either way, a frame's stride out of date at
        // worst). On a fixture that is itself walking it is not: the overlay is re-rendered from a
        // NEW position every frame, so a mesh gathered two frames ago puts the room's other bodies
        // in the wrong place relative to the light, which is the smear this change is about. The
        // rate is untouched for the settled case, which is every other slot in the room.
        let due = fresh
            || moving_mask != 0
            || entity_gather_due(
                video.interior_shadow_entity_rate,
                now - lane.entity_at,
                dynamic,
                lane.entity_mask,
            );
        if due {
            if let Some(mesh) = lane.entity_mesh.as_ref().and_then(|h| meshes.get_mut(h)) {
                let (mut positions, mut indices) = take_mesh_buffers(mesh);
                (lane.trace_parts, _) = collect_torch_entities(
                    &ents,
                    false,
                    &dynamic_positions,
                    &mut positions,
                    &mut indices,
                );
                restore_mesh_buffers(mesh, positions, indices);
            }
            lane.trace_gathers += 1;
            lane.entity_at = now;
            lane.entity_mask = dynamic;
        }
        published.entity_mesh = lane.entity_mesh.as_ref().map(Handle::id);
    }
    lane.trace_scans += lane_scans;
    lane.trace_reach = dynamic_positions
        .iter()
        .map(|(_, _, r, _)| *r)
        .fold(0.0, f32::max);
    *views = published;

    // `WOW_TORCH_TRACE=1` — once a second, what the lane published: fixture distances + whether a
    // caster mesh exists. The app-side half of the debug (the shader group-3 sample is the GPU half).
    if std::env::var_os("WOW_TORCH_TRACE").is_some() && now - *last_trace >= 1.0 {
        *last_trace = now;
        // MONKEY (outdoor torch shadows): the header carries the lane budget too — "6 of the
        // 12 slots may be exterior, 3 currently are, the sun is at 0.00" is the whole state of
        // the split in one line, and it is the first thing to read when an outdoor shadow is
        // missing (no EXT slots at sun 1.00 is correct; no EXT slots at sun 0.00 is a bug).
        info!(
                "torch-trace: {}/{} slots of {} (ext {}/{}, sun {:.2}, {} candidates, caster {}, {} entity casters)",
                promoted.len(),
                want,
                MAX_TORCH_CASTERS,
                promoted.iter().filter(|(.., ext)| *ext).count(),
                ext_cap,
                sun_w,
                cands.len(),
                if lane.slots.iter().any(|s| s.mesh.is_some()) {
                    "built"
                } else {
                    "none"
                },
                lane.trace_parts,
            );
        // MONKEY (torch lane perf): the FIXED per-frame cost, in the two counters that bound
        // it. `gather parts N (rate R Hz, reach X yd)` is the CPU-skinned moving-caster set and
        // the cadence + union radius it was taken at; `fingerprint scans S` is how many full
        // resident-scene walks the census gate actually let through in the last second (a
        // still room should read 0, a streaming one a handful). Both are per SECOND, so they
        // are directly comparable to the frame rate beside them: `gathers` equal to the fps is
        // `interiorShadowEntityRate 0`, and `scans` equal to the fps is the old behaviour with
        // the gate defeated.
        info!(
                "torch-perf: gather parts {} (rate {} Hz, {} gathers/s, reach {:.0} yd), fingerprint scans {}/s",
                lane.trace_parts,
                video.interior_shadow_entity_rate,
                lane.trace_gathers,
                lane.trace_reach,
                lane.trace_scans,
            );
        lane.trace_gathers = 0;
        lane.trace_scans = 0;
        // MONKEY (torch caster selection): per SLOT, the three numbers the selection is made
        // of — which fixture, what it scores at the player, and how far its cross-fade has
        // ramped. A shadow that pops is a `w` that jumped; a shadow that should be there and
        // is not is a fixture missing from this list (then read `cands`, above).
        // MONKEY (torch caster reach): scientific scores expose the small distant tail;
        // distance is still from the player, and eligibility now extends to 4R + 3 yd.
        for (i, e, p, sc, w, ext) in &promoted {
            // MONKEY (torch owner exclusion): `owner` + `own body` are the fix's own readout —
            // "who this fixture belongs to" and "how many retained items / entity parts its
            // LAST rebuild dropped as that owner's body". A fixture standing in its own black
            // square with `own body 0/0` is a missing owner tag; one with a plausible count and
            // a shadow still under it is a caster arriving by some third route.
            let slot = lane.slots.iter().find(|s| s.cache_slot == *i);
            let (gx_excl, part_excl) = slot.map_or((0, 0), |s| s.trace_excluded);
            // MONKEY (moving fixture): `moving` is this feature's readout — `live` is a slot
            // being re-rendered from scratch every frame, `dropped` one that is moving but past
            // TORCH_MOVING_MAX and fast-fading out, `settled` the ordinary cached path. An imp
            // whose shadow lags is a slot reading `settled` while it walks (then look at
            // `still_since` / the owner tag); a room whose static shadows freeze is two slots
            // stuck on `live`.
            info!(
                    "  slot {i}: lane {} {e} at [{:.1},{:.1},{:.1}] d(player) {:.1} score {:.4e} w {:.2} owner {:?} moving {} own body {gx_excl} gx + {part_excl} parts (camera <= shadowDistance AND player <= 4R+3) static {} dynamic {} requested_live_rank {:?}",
                    if *ext { "EXT" } else { "INT" },
                    p.x,
                    p.y,
                    p.z,
                    p.distance(anchor),
                    sc,
                    w,
                    slot.and_then(|s| s.owner),
                    match slot {
                        Some(s) if s.moving_live => "live",
                        Some(s) if s.fast_fade => "dropped",
                        _ => "settled",
                    },
                    if static_rebuilt & (1 << i) != 0 { "rebuilt (GPU pending)" } else { "cached/requested" },
                    if dynamic & (1 << i) != 0 { "yes" } else { "no" },
                    (dynamic & (1 << i) != 0).then(|| TorchShadowViews::live_rank(dynamic, *i)),
                );
        }
    }
}

// MONKEY (static torch cache): a rig can deform even while its root stands still. Cache only
// rigid non-creature parts; animated furniture follows the same nearest-N policy as characters.
fn static_part(part: &ModelPart, rigged: bool) -> bool {
    part.kind != ModelKind::Creature && !rigged && part.blend == ModelBlend::Opaque
}

/// MONKEY (torch lane perf): `fixtures` is `(slot, position, RADIUS, owner)` — each fixture carries
/// the radius it admits casters inside, instead of the whole gather sharing [`TORCH_RANGE`]. The
/// static (cached) caller passes `TORCH_RANGE`, so its admission WINDOW is the one it always was;
/// only the per-frame MOVING gather narrows, to `entity_gather_radius(reach)`.
///
/// MONKEY (torch owner exclusion): …and the owner, which drops the fixture's own body
/// ([`part_is_own_body`]). Returns `(admitted, excluded-as-own-body)` — the second is trace only.
fn collect_torch_entities(
    ents: &EntityCasters,
    want_static: bool,
    fixtures: &[GatherFixture],
    positions: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) -> (u32, u32) {
    let mut admitted = 0;
    let mut excluded = 0;
    for (entity, pick, part, global, rig_part, occluder, _, object, aabb) in &ents.parts {
        if !occluder.0 || static_part(part, rig_part.is_some()) != want_static {
            continue;
        }
        let solid = part.blend == ModelBlend::Opaque
            || (part.kind == ModelKind::Creature && part.blend == ModelBlend::AlphaTest);
        if !solid {
            continue;
        }
        let rig = rig_part.and_then(|r| ents.rigs.get(r.0).ok());
        let origin = global
            .map(GlobalTransform::translation)
            .or_else(|| rig.and_then(|r| ents.palettes.slot_origin(r.slot)));
        let Some(origin) = origin else { continue };
        if !within_any_reach(fixtures, origin) {
            continue;
        }
        // After the cheap window, before the expensive CPU skin: this is the lamp's own housing,
        // the brazier's own bowl, the lantern post the light hangs off.
        if part_is_own_body(ents, fixtures, entity, part.kind, object, global, aabb) {
            excluded += 1;
            continue;
        }
        let base = positions.len() as u32;
        if let Some(rig) = rig {
            let Some(palette) = ents.palettes.world_palette(rig.slot, rig.bones() as usize) else {
                continue;
            };
            for (v, p) in pick.0.positions.iter().enumerate() {
                let p = wow_to_bevy(*p);
                let (Some(joints), Some(weights)) = (pick.0.joints.get(v), pick.0.weights.get(v))
                else {
                    positions.push(
                        palette
                            .first()
                            .map(|m| m.transform_point3(p))
                            .unwrap_or(origin)
                            .to_array(),
                    );
                    continue;
                };
                let mut skinned = Vec3::ZERO;
                for lane in 0..4 {
                    if weights[lane] > 0.0 {
                        if let Some(m) = palette.get(joints[lane] as usize) {
                            skinned += m.transform_point3(p) * weights[lane];
                        }
                    }
                }
                positions.push(skinned.to_array());
            }
        } else if rig_part.is_none() {
            if let Some(global) = global {
                positions.extend(
                    pick.0
                        .positions
                        .iter()
                        .map(|p| global.transform_point(wow_to_bevy(*p)).to_array()),
                );
            }
        }
        let added = positions.len() as u32 - base;
        if added == 0 {
            continue;
        }
        for tri in pick.0.indices.as_chunks::<3>().0 {
            if tri.iter().all(|i| *i < added) {
                indices.extend(tri.iter().map(|i| base + *i));
            }
        }
        admitted += 1;
    }
    (admitted, excluded)
}

/// MONKEY (torch lane perf): the entity half of [`PartCensus`] — one branch-free mix per RIGID
/// OPAQUE model part (the exact set the fingerprint's own entity walk hashes), folded
/// order-independently so archetype reordering is invisible.
///
/// Deliberately much cheaper than the thing it guards: no `DefaultHasher` per part and no
/// `to_matrix()`, just the entity id, the geometry `Arc`'s identity, the affine's translation and
/// ONE basis column. Translation catches a moved prop, the basis column catches a rotated one (a
/// door on its hinge barely moves its origin), the `Arc` catches a geometry swap, and the id sum
/// catches an arrival or a despawn. What it cannot see — a part rotating exactly about the axis of
/// the column sampled — has no bearing on a shadow's silhouette to the precision this lane renders.
fn static_part_census(ents: &EntityCasters) -> u64 {
    let mut census = 0u64;
    for (entity, pick, part, global, rig, occluder, ..) in &ents.parts {
        if !occluder.0 || !static_part(part, rig.is_some()) {
            continue;
        }
        let Some(global) = global else { continue };
        let a = global.affine();
        let mut mix = entity.to_bits() ^ (std::sync::Arc::as_ptr(&pick.0) as usize as u64);
        for v in [
            a.translation.x,
            a.translation.y,
            a.translation.z,
            a.matrix3.x_axis.x,
            a.matrix3.x_axis.y,
            a.matrix3.x_axis.z,
        ] {
            mix = mix.rotate_left(11) ^ u64::from(v.to_bits());
        }
        census = census.wrapping_add(mix);
    }
    census
}

/// Tear down the caster mesh and clear the published views (the lane went off, or we left the world).
fn teardown(lane: &mut TorchLane, meshes: &mut Assets<Mesh>, views: &mut TorchShadowViews) {
    for slot in &mut lane.slots {
        if let Some(handle) = slot.mesh.take() {
            meshes.remove(handle.id());
        }
    }
    if let Some(handle) = lane.entity_mesh.take() {
        meshes.remove(handle.id());
    }
    // MONKEY (torch caster selection): the slot table goes with them — an incumbent kept across a
    // teardown would come back at full weight in a room it is no longer in.
    lane.slots.clear();
    // MONKEY (torch lane perf): the gather bookkeeping goes with them. A lane that comes back up in
    // a different room must regather on its first frame, not wait out a cadence tick against a
    // timestamp from before the teardown.
    lane.entity_at = 0.0;
    lane.entity_mask = 0;
    *views = TorchShadowViews::default();
}

// MONKEY (torch caster reach): regression boundaries and the unfloored distant ranking tail.
#[cfg(test)]
mod tests {
    use super::*;

    // MONKEY (review fixes): useful interior casters win regardless of an exterior fixture's
    // brightness; the fire dial scales only invented sources, including its live kill switch.
    #[test]
    fn candidates_follow_the_packed_lane_and_fire_gain() {
        let interior = LightLane::carried(true);
        let exterior = LightLane::carried(false);
        let authored = candidate_score(1.0, 10.0, 10.0, Some(&interior), true, false, 0.0);
        assert!(authored > 0.0);
        assert_eq!(
            candidate_score(100.0, 1.0, 10.0, Some(&exterior), true, false, 1.0),
            0.0
        );
        for gain in [0.0, -1.0] {
            assert_eq!(
                candidate_score(100.0, 1.0, 10.0, Some(&interior), true, true, gain),
                0.0
            );
        }
        assert_eq!(
            candidate_score(1.0, 10.0, 10.0, Some(&interior), true, true, 0.5),
            authored * 0.5
        );
        // No MOLR is needed once a containment-only fixture resolves interior; before resolution
        // the old LightRooms fallback is shared with build_light_data, in both directions.
        assert_eq!(
            candidate_score(1.0, 10.0, 10.0, Some(&interior), false, false, 0.0),
            authored
        );
        assert_eq!(
            candidate_score(1.0, 10.0, 10.0, None, true, false, 0.0),
            authored
        );
        assert_eq!(
            candidate_score(1.0, 10.0, 10.0, None, false, false, 0.0),
            0.0
        );
    }

    #[test]
    fn eligibility_covers_four_reaches_with_slack() {
        for r in [7.0, 15.0, 48.0] {
            assert!(fixture_eligible(3.5 * r, 80.0, r, 80.0));
            assert!(fixture_eligible(4.0 * r + 3.0, 80.0, r, 80.0));
            assert!(!fixture_eligible(4.2 * r + 3.0, 80.0, r, 80.0));
        }
        assert!(!fixture_eligible(1.0, 80.01, 48.0, 80.0));
        assert!(fixture_eligible(1.0, 250.0, 48.0, 300.0));
        assert!(!fixture_eligible(32.0, 1.0, 7.0, 80.0));
        assert!(!fixture_eligible(1.0, f32::NAN, 48.0, 80.0));
    }

    // MONKEY (live bank rank): sparse slots, tie stability and both ends of the live budget.
    #[test]
    fn dynamic_subset_is_nearest_and_deterministic() {
        let fixtures = [
            (15, Vec3::X),
            (8, -Vec3::X),
            (2, Vec3::X * 20.0),
            (4, Vec3::ZERO),
        ];
        assert_eq!(dynamic_set(&fixtures, Vec3::ZERO, 2), (1 << 4) | (1 << 8));
        assert_eq!(dynamic_set(&fixtures, Vec3::X * 20.0, 1), 1 << 2);
        assert_eq!(dynamic_set(&fixtures, Vec3::ZERO, 0), 1 << 4);
        assert_eq!(dynamic_set(&fixtures, Vec3::ZERO, 16).count_ones(), 4);
        let all: Vec<_> = (0..MAX_TORCH_CASTERS)
            .map(|i| (i, Vec3::X * i as f32))
            .collect();
        assert_eq!(dynamic_set(&all, Vec3::ZERO, 16), 0xff);
        assert_eq!(dynamic_set(&[], Vec3::ZERO, 1), 0);
    }

    // MONKEY (static torch cache): a never-built slot publishes nothing; a merely STALE map keeps
    // being published (its shadow must not blink off while the rebuild cursor reaches it); a map
    // whose fixture has MOVED is withdrawn, because the projections no longer address it.
    #[test]
    fn a_cached_map_survives_staleness_but_not_fixture_drift() {
        assert!(!map_publishable(None, Vec3::ZERO));
        assert!(map_publishable(Some(Vec3::ZERO), Vec3::ZERO));
        assert!(map_publishable(Some(Vec3::ZERO), Vec3::X * 0.09));
        assert!(!map_publishable(Some(Vec3::ZERO), Vec3::X * 0.2));
        assert!(!map_publishable(Some(Vec3::ZERO), Vec3::splat(10.0)));
    }

    // MONKEY (outdoor torch shadows): the exterior ranking key is the EXTERIOR receivers' own
    // profile, not the interior soft core — monotone in distance, linear in intensity, floored off
    // the falloff's pole at 0, and reaching exactly zero at the eligibility edge so a candidate
    // never steps out of the list at a finite score.
    #[test]
    fn exterior_scores_follow_the_outdoor_falloff() {
        let near = exterior_fixture_score(1.0, 5.0);
        let mid = exterior_fixture_score(1.0, 15.0);
        let far = exterior_fixture_score(1.0, 40.0);
        assert!(near > mid && mid > far && far > 0.0);
        assert_eq!(exterior_fixture_score(1.0, TORCH_EXT_ELIGIBLE_YD), 0.0);
        assert_eq!(exterior_fixture_score(0.0, 5.0), 0.0);
        assert!(exterior_fixture_score(2.0, 15.0) > mid);
        // The falloff's pole at d = 0 is floored, so a fixture the player is standing on cannot
        // score infinity, take a slot for one frame and hand it straight back.
        let at_zero = exterior_fixture_score(1.0, 0.0);
        assert!(
            at_zero.is_finite() && at_zero <= 1.0 / 0.73 + 1e-3,
            "{at_zero}"
        );
        // …and inside the window it IS the receivers' falloff: 1/(0.7d + 0.03d^2) at 1 yd.
        assert!((exterior_fixture_score(1.0, 1.0) - 1.0 / 0.73).abs() < 1e-2);
    }

    // MONKEY (outdoor torch shadows): the camera tests are shared with the interior lane (same
    // budget, same `shadowDistance` dial); the player window is flat, because an exterior entry
    // packs no reach for a `4R` window to be four of.
    #[test]
    fn exterior_eligibility_is_the_cube_range_at_the_player() {
        assert!(exterior_eligible(0.0, 0.0, 80.0));
        assert!(exterior_eligible(TORCH_EXT_ELIGIBLE_YD, 10.0, 80.0));
        assert!(!exterior_eligible(TORCH_EXT_ELIGIBLE_YD + 0.1, 10.0, 80.0));
        assert!(!exterior_eligible(1.0, 80.01, 80.0));
        assert!(exterior_eligible(1.0, 250.0, 300.0));
        assert!(!exterior_eligible(1.0, f32::NAN, 80.0));
    }

    // MONKEY (outdoor torch shadows): the lane gate is the cvar AND night, and "night" is the
    // receivers' own threshold — `sun_shadow_strength` saturates at exactly 1.0 in daylight, so the
    // CPU lane and the shaders' `night_w` can never disagree about which it is.
    #[test]
    fn the_exterior_lane_is_night_only_and_cvar_gated() {
        let mut video = VideoConfig::default();
        assert!(video.exterior_shadows, "shipped on");
        assert!(exterior_shadows_on(&video, 0.0), "midnight");
        assert!(exterior_shadows_on(&video, 0.99), "dusk still counts");
        assert!(!exterior_shadows_on(&video, 1.0), "full daylight never");
        video.exterior_shadows = false;
        assert!(!exterior_shadows_on(&video, 0.0));
        // The receivers' threshold, verbatim: 1.0 the moment the sun clears ~12 degrees.
        assert_eq!(sun_shadow_strength(0.208), 1.0);
        assert_eq!(sun_shadow_strength(0.9), 1.0);
        assert_eq!(sun_shadow_strength(-0.3), 0.0);
        assert!(sun_shadow_strength(0.1) > 0.0 && sun_shadow_strength(0.1) < 1.0);
    }

    // MONKEY (outdoor torch shadows): half the budget, rounded up — and the cap only ever removes
    // EXTERIOR candidates, keeping the best ones (the list is score-sorted).
    #[test]
    fn the_exterior_budget_is_half_and_caps_only_its_own_lane() {
        assert_eq!(exterior_budget(12), 6);
        assert_eq!(exterior_budget(16), 8);
        assert_eq!(
            exterior_budget(1),
            1,
            "the smallest budget still gets one outdoor shadow"
        );
        let mut cands: Vec<Candidate> = (0..8)
            .map(|i| {
                (
                    Entity::from_raw_u32(i).unwrap(),
                    Vec3::ZERO,
                    8.0 - i as f32,
                    i % 2 == 0,
                    10.0,
                    None,
                )
            })
            .collect();
        cap_exterior_candidates(&mut cands, 2);
        // Identified by score, which is also their rank: 8,7,6,5,4,3,2,1 with the even ranks
        // exterior. Capped at two, ranks 8 and 6 (the best two fires) survive; 4 and 2 do not.
        let kept: Vec<(u32, bool)> = cands
            .iter()
            .map(|(_, _, sc, x, ..)| (*sc as u32, *x))
            .collect();
        assert_eq!(
            kept,
            vec![
                (8, true),
                (7, false),
                (6, true),
                (5, false),
                (3, false),
                (1, false)
            ],
            "the two best exterior fires survive; every interior candidate does"
        );
        // A cap of zero is a lane switched off, and it takes nothing else with it.
        let mut all_ext: Vec<Candidate> = vec![(
            Entity::from_raw_u32(0).unwrap(),
            Vec3::ZERO,
            1.0,
            true,
            10.0,
            None,
        )];
        cap_exterior_candidates(&mut all_ext, 0);
        assert!(all_ext.is_empty());
    }

    // MONKEY (torch lane perf): the moving-caster gather is sized off each fixture's OWN reach.
    // The margin covers the part's extent (the test is on its ORIGIN), the clamp keeps the radius
    // inside the cube projection that has to record it, and a degenerate reach can only ever widen
    // the window — losing a caster must never be the failure direction here.
    #[test]
    fn the_entity_gather_radius_is_the_fixtures_own_reach() {
        assert_eq!(entity_gather_radius(12.0), 12.0 + TORCH_ENTITY_REACH_MARGIN);
        assert!(
            entity_gather_radius(12.0) < TORCH_RANGE,
            "a candle sweeps far less than 48 yd"
        );
        assert_eq!(
            entity_gather_radius(TORCH_RANGE),
            TORCH_RANGE,
            "clamped to the cube range"
        );
        assert_eq!(entity_gather_radius(1e9), TORCH_RANGE);
        assert_eq!(entity_gather_radius(0.0), TORCH_ENTITY_REACH_MARGIN);
        assert_eq!(entity_gather_radius(-5.0), TORCH_ENTITY_REACH_MARGIN);
        assert_eq!(
            entity_gather_radius(f32::NAN),
            TORCH_RANGE,
            "never a silent zero radius"
        );
    }

    // …and the gather admits the UNION of those radii, not one shared radius: a tight candle and a
    // wide hearth in the same dynamic set each keep their own window, and a part inside either is
    // in. The reach that matters is the fixture's, so a body 20 yd from the candle and 20 yd from
    // the hearth is a caster for the hearth alone.
    #[test]
    fn the_gather_admits_the_union_of_the_fixture_reaches() {
        let candle = entity_gather_radius(8.0);
        let hearth = entity_gather_radius(30.0);
        let set = [
            (0usize, Vec3::ZERO, candle, None),
            (1usize, Vec3::X * 60.0, hearth, None),
        ];
        assert!(within_any_reach(&set, Vec3::X * 5.0), "inside the candle");
        assert!(
            within_any_reach(&set, Vec3::X * 40.0),
            "inside the hearth, not the candle"
        );
        assert!(
            !within_any_reach(&set, Vec3::X * 20.0),
            "between the two, lit by neither"
        );
        assert!(
            !within_any_reach(&[], Vec3::ZERO),
            "an empty dynamic set admits nothing"
        );
        // The margin is what keeps a body whose ORIGIN sits just outside the pool casting.
        assert!(within_any_reach(&set, Vec3::X * (8.0 + 1.0)));
    }

    // MONKEY (torch lane perf): the regather cadence. `0` is the every-frame escape hatch, a
    // changed dynamic set always overrides the clock (the mesh is only valid for the fixtures it
    // was gathered around), and the tolerance stops a 30 Hz cadence from beating against a 60 fps
    // frame into an alternating 1/3-frame stutter.
    // MONKEY (moving fixture): a slot for the motion tests — everything but the four fields the
    // verdict reads is inert.
    fn moving_slot(
        owner: Option<LightOwner>,
        built_at: Vec3,
        pos: Vec3,
        still_since: f64,
    ) -> TorchSlot {
        TorchSlot {
            cache_slot: 0,
            mesh: None,
            geometry_key: None,
            built_at: Some(built_at),
            fixture: Entity::PLACEHOLDER,
            pos,
            score: 1.0,
            reach: 10.0,
            checked: None,
            exterior: false,
            owner,
            trace_excluded: (0, 0),
            last_pos: pos,
            still_since,
            moving_live: false,
            fast_fade: false,
            w: 1.0,
            evicting: false,
        }
    }

    // MONKEY (moving fixture): WHO may be re-rendered every frame, and for HOW LONG after it stops.
    #[test]
    fn a_fixture_is_moving_only_while_it_is_actually_moving() {
        let pet = Some(LightOwner::Instance(Entity::PLACEHOLDER));
        let wall = Some(LightOwner::Placement {
            kind: ModelKind::Wmo,
            id: 7,
            label: 0,
        });
        let now = 100.0;
        // A pet's flame that moved this frame: live.
        assert!(moving_slot(pet, Vec3::ZERO, Vec3::ZERO, now).moving(now));
        // …still live a moment later (a walk cycle pauses; the map must not thrash back and forth).
        assert!(moving_slot(pet, Vec3::ZERO, Vec3::ZERO, now - 0.5).moving(now));
        // …and settled once it has held still past the threshold — back on the cached path.
        assert!(!moving_slot(pet, Vec3::ZERO, Vec3::ZERO, now - 1.0).moving(now));
        // A placed brazier that has never moved is never live, however recently it was promoted.
        assert!(!moving_slot(wall, Vec3::ZERO, Vec3::ZERO, now).moving(now));
        // …unless it has physically drifted off its own baked map, which IS proof of motion:
        // exactly TORCH_STALE_DRIFT_SQ (0.1 yd) worth, the same threshold `map_publishable` uses.
        assert!(!moving_slot(wall, Vec3::ZERO, Vec3::X * 0.09, now).moving(now));
        assert!(moving_slot(wall, Vec3::ZERO, Vec3::X * 0.2, now).moving(now));
        // A slot with no map yet is NEW, not drifted — it belongs to the promotion path.
        let mut fresh = moving_slot(wall, Vec3::ZERO, Vec3::splat(50.0), now);
        fresh.built_at = None;
        assert!(!fresh.drifted());
        assert!(!fresh.moving(now));
    }

    // MONKEY (moving fixture): the budget is by contribution, capped, and deterministic.
    #[test]
    fn the_moving_budget_keeps_the_best_two_and_drops_the_rest() {
        assert_eq!(moving_budget(vec![]), (vec![], vec![]));
        assert_eq!(moving_budget(vec![(3, 0.5)]), (vec![3], vec![]));
        let (live, dropped) = moving_budget(vec![(0, 0.1), (1, 9.0), (2, 4.0), (3, 0.2)]);
        assert_eq!(live, vec![1, 2]);
        assert_eq!(dropped, vec![3, 0]);
        // Equal scores resolve by slot order, both ways round, so the set cannot alternate.
        let tied = vec![(5, 1.0), (2, 1.0), (9, 1.0)];
        assert_eq!(moving_budget(tied.clone()).0, vec![2, 5]);
        assert_eq!(
            moving_budget(tied.into_iter().rev().collect()).0,
            vec![2, 5]
        );
        assert_eq!(TORCH_MOVING_MAX, 2, "the cap the render plan budgets for");
    }

    #[test]
    fn the_entity_gather_cadence_is_a_rate_not_a_gate() {
        assert!(entity_gather_due(0, 0.0, 1, 1), "rate 0 = every frame");
        assert!(
            !entity_gather_due(30, 0.0, 1, 1),
            "same set, no time passed"
        );
        assert!(!entity_gather_due(30, 0.02, 1, 1), "still inside 1/30 s");
        assert!(
            entity_gather_due(30, 1.0 / 30.0, 1, 1),
            "exactly on the tick"
        );
        assert!(
            entity_gather_due(30, 1.0 / 60.0 * 2.0, 1, 1),
            "two 60 fps frames make the tick"
        );
        assert!(
            entity_gather_due(30, 0.0, 0b11, 0b01),
            "a changed dynamic set overrides the clock"
        );
        assert!(entity_gather_due(15, 0.07, 1, 1));
        assert!(!entity_gather_due(15, 0.05, 1, 1));
        // A rate at or above the frame rate is "every frame" without being a special case.
        assert!(entity_gather_due(240, 1.0 / 46.0, 1, 1));
    }

    #[test]
    fn distant_scores_stay_positive_and_ordered() {
        let r = 10.0;
        let near = fixture_score(1.0, r, r);
        let far = fixture_score(1.0, 3.0 * r, r);
        let farther = fixture_score(1.0, 3.5 * r, r);
        let slack = fixture_score(1.0, 4.0 * r + 2.0, r);
        assert!(near > far && far > farther && farther > slack && slack > 0.0);
        assert!(far < near * 0.2);
        assert_eq!(fixture_score(0.0, 3.0 * r, r), 0.0);
        assert_eq!(fixture_score(1.0, 4.0 * r + 3.0, r), 0.0);
        assert!(fixture_score(2.0, 3.0 * r, r) > far);
    }
}
