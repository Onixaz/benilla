//! Carried M2 lights: the point lights an entity's models bring, beside the placed ones
//! (`benilla_world::terrain_stream`'s `spawn_lights_for`). The reference makes no distinction:
//! `0x718960` gathers every drawn model's `type == 1` lights each frame, attached children included
//! (`0x7191b9`, `0x719286`), at the live bone matrix into the one scene light DB (`0x71b650`,
//! `0x71bb60`), so a held torch lights the ground. Each light here rides its host bone's joint.

use benilla_assets::coords::wow_to_bevy;
use benilla_assets::ModelLight;
use bevy::prelude::*;

use benilla_world::interior::{InteriorAnchor, WmoResidency};
use benilla_world::lighting::{
    LightLane, LightLitRooms, LightRooms, ShadowProxyLight, SpellFxLight, SyntheticFireLight,
    WorldPointLight,
};
use benilla_world::static_gx::LightOwner;
use benilla_world::terrain_stream::{carried_light_claims, point_light, CarriedClaimSet};
use benilla_world::wmo_portal::{WmoGroupVis, WmoPortalInstance, WmoRoom};

// MONKEY (spell light): the effect-side lifecycle this file's spawn helper stamps on — the
// envelope, the budget and the kill switch all live beside the rest of the effect lifecycle
// (`spell_fx::lifecycle`), because that is what a spell light's lifetime IS.
// MONKEY (area spell light): `AreaSpellLight` too — the persistent ground lane's marker, stamped
// here beside the rest of a spell light's tags.
use super::spell_fx::{
    spell_lights_enabled, AreaSpellLight, SpellLight, SpellLightMode, SPELL_BURST_SPAN,
};

/// MONKEY (outdoor torch shadows): this light is carried BY A BODY — a held torch, a creature's
/// own glow (an imp's hand flame, a fire elemental), a player's lantern. It is a marker, not a
/// gate: the light still lights the world exactly as it did, and it is still eligible for an
/// INTERIOR cube-shadow slot (that lane's behaviour is unchanged by this feature).
///
/// What it refuses is the EXTERIOR shadow lane. Three separate reasons, any one of which is enough:
///  · the caster gather cannot exclude the owner's own body from its own map (the emitter-owner
///    exclusion below works on retained `GxItem` bounds; a skinned creature is not one), so a
///    torch-bearing guard would stand in his own shadow, at full outdoor strength;
///  · a body walks, and this lane caches a depth cube per fixture and withdraws it the moment the
///    fixture leaves the 0.1 yd it was baked at ([`super::super::torch_shadow`]'s
///    `map_publishable`) — indoors a held torch is one of twelve resident candles and its churn is
///    hidden, outdoors it would be one of six casters for a whole village;
///  · the effect being built is "the CAMPFIRE throws the fence's shadow", and a shadow that walks
///    with the light source is the one thing that reads as a bug rather than as lighting.
/// Placed GameObject braziers and campfires are NOT held — they keep their slot (the
/// [`CarriedLightMotion::settled`] gate is what covers a brazier riding a moving transport).
#[derive(Component, Clone, Copy)]
pub(crate) struct HeldLight;

/// Spawn a point light child for each casting M2 light of an entity's model. A light whose bone
/// `joint` resolves rides that joint at `position − bone_pivot`; one without (a held item has no
/// skeleton, a `-1` bone) hangs off `frame` in model space. As children, the lights leave with a
/// gear change, a despawn or a mount transition.
pub(super) fn spawn_carried_lights(
    commands: &mut Commands,
    lights: &[ModelLight],
    frame: Entity,
    // MONKEY (outdoor torch shadows): does a BODY carry these lights? See [`HeldLight`].
    held: bool,
    joint: impl Fn(i16) -> Option<Entity>,
) {
    for l in lights {
        if !l.def.casts() {
            continue; // directionals feed an ambient term; a static `0` visibility key is dark
        }
        // MONKEY (carried light stability): a SYNTHESISED light (`fire_light` — the imp's hand
        // flames, a brazier GameObject, a fire elemental) does NOT ride its host bone; an
        // AUTHORED one still does.
        //
        // The distinction is what the position MEANS. An authored light block is a fixture an
        // artist put on a bone, and `0x718960` re-registering it through the live bone matrix is
        // the reference's own behaviour — the held torch that tracks the swinging hand, and that
        // case takes the boneless arm below anyway (a held item spawns no skeleton). A synthesised
        // one is a stand-in for a whole flame VOLUME whose position we GUESSED off the emitter
        // record: it is already approximate to within the flame's own size, and riding a hand bone
        // through an idle animation adds high-frequency motion that every consumer downstream
        // reads as an EVENT rather than as a wobble. The per-vertex nearest-3 in `point_light_sum`
        // re-ranks; and the cube-shadow cache tripped its 0.1 yd staleness test
        // (`torch_shadow::map_publishable`) every single frame and withdrew its map, which
        // `TorchTable::pack` answers by forcing that slot's cross-fade weight to 0 with no ramp —
        // a full-strength shadow toggling at frame rate. That is the reported "the imp produces a
        // DISCO with its light".
        //
        // The rest pose is identity about the pivot ([`ModelLight::bone_pivot`]), so hanging the
        // light off `frame` at the raw model-space `def.position` puts it EXACTLY where the bone
        // ride puts it in the rest pose — and `frame` is the model root, so the light still
        // follows the unit's own motion with zero lag. Only the animation is dropped.
        let host = if l.synthetic { None } else { joint(l.def.bone) };
        let (parent, local) = match host {
            Some(j) => (
                j,
                [
                    l.def.position[0] - l.bone_pivot[0],
                    l.def.position[1] - l.bone_pivot[1],
                    l.def.position[2] - l.bone_pivot[2],
                ],
            ),
            None => (frame, l.def.position),
        };
        // MONKEY (spell light): a SPELL/FIREWORK-derived light is not a carried FIXTURE and does
        // not take this lane's rules. It has an onset, a lifecycle envelope and a budget, and it
        // never flickers — all of which live on [`SpellLight`]. The branch is here because the
        // firework GameObject arrives HERE: its shell (`World\Goober\G_Firework0*`) is a GO model
        // like any other, so the entity lane is the only place its light can be born.
        if let Some(fx) = l.spell {
            spawn_spell_light_child(
                commands,
                parent,
                local,
                SpellLightSource {
                    color: l.def.diffuse_color,
                    intensity: l.def.diffuse_intensity,
                    kind: fx.kind,
                    onset: fx.onset,
                },
                // A GameObject that carries a spell light IS a one-shot: a firework shell, a
                // trigger's effect model. Nothing on this lane holds.
                SpellLightMode::Burst {
                    span: SPELL_BURST_SPAN,
                },
            );
            continue;
        }
        let mut glow = commands.spawn((
            point_light(l.def.diffuse_color, l.def.diffuse_intensity),
            Transform::from_translation(wow_to_bevy(local)),
            Visibility::default(),
        ));
        // MONKEY (fire GO lights): a light we DERIVED from the model's flame emitter is tagged, so
        // the packer can ride the live `fireLightGain` over it (and only over it). The entity lanes
        // take synthetic lights: a placed brazier GameObject, a fire elemental, a flaming helm have
        // no MOLT fixture standing beside them the way a WMO's own wall torches do.
        if l.synthetic {
            glow.insert(SyntheticFireLight);
        }
        // MONKEY (outdoor torch shadows): the exterior shadow lane's refusal marker (see
        // [`HeldLight`]). Tagged at the SPAWN site because that is the only place that knows who
        // owns the light — downstream all a query sees is a `PointLight` with a `ChildOf`, which a
        // placed brazier GameObject has just as much as a guard's torch does.
        if held {
            glow.insert(HeldLight);
        }
        // MONKEY (torch owner exclusion): and the light's OWNER — the model frame it belongs to.
        // Same bug as the placed lane's (`terrain_stream::spawn`'s `tag_light_owner`): the flame
        // sits INSIDE the thing that burns it, so a placed brazier GameObject was the nearest
        // occluder on all six of its own cube faces and stamped a black disc on the ground under
        // itself. No `WorldObject` exists at this spawn site — a carried light's host is a net
        // entity, not a placement — so the identity is the FRAME, and the caster gather asks
        // "is this part a descendant of it?" (`LightOwner::Instance`).
        glow.insert(LightOwner::Instance(frame));
        // MONKEY (flame flicker): a carried flame breathes like any other — the imp's hand fire, a
        // held torch, a brazier GameObject. It is safe on THIS lane specifically because the
        // modulation touches neither position nor reach: the settle/claim logic
        // ([`CarriedLightMotion`], `carried_light_claims`) and the shadow slot's staleness test all
        // key on where the light IS, and none of them can see a brightness change. The seed mixes
        // the light's model-space offset with its PARENT, so two imps in one room never burn in
        // step (position alone would give every copy of a model the same phase).
        if let Some(kind) = benilla_world::lighting::flame_kind_for(
            l.flame,
            l.synthetic,
            l.def.diffuse_color,
            l.def.diffuse_intensity,
        ) {
            let seed = benilla_world::lighting::flicker_seed(wow_to_bevy(local))
                ^ parent.to_bits().rotate_left(11) as u32;
            glow.insert(benilla_world::lighting::FlameFlicker::new(kind, seed));
        }
        let glow = glow.id();
        commands.entity(parent).add_child(glow);
    }
}

/// MONKEY (spell light): spawn the ONE spell light a luminous effect model carries, as a child of
/// that effect's root. `false` when the model carries none — which is the great majority of the
/// corpus (frost, nature, arcane, shadow and everything unnamed synthesise nothing at all), and
/// when the env kill switch `WOW_SPELL_LIGHT=0` is set.
///
/// `root` is the effect INSTANCE's root entity — the kit instance's attach node, the missile, the
/// dest-anchored plant, the firework's GameObject frame. That parentage is the whole lifetime
/// contract: every one of those lanes despawns its root when the effect ends (the reap, the
/// arrival, the expiry), and a child light goes with it. There is no separate reaping path to get
/// wrong, and no `on_owner_loss` case to answer — a light is not a particle pool and never drains.
///
/// The light hangs at the emitter's own model-space position, exactly as a synthesised fire light
/// does, and does NOT ride a bone: a synthesised light stands for a whole flame VOLUME whose
/// position was inferred, and riding an animating joint adds motion that the shadow/claim lanes
/// read as an event (the same reasoning [`spawn_carried_lights`] gives for its `l.synthetic` arm).
pub(super) fn spawn_spell_light(
    commands: &mut Commands,
    lights: &[benilla_assets::ModelLight],
    root: Entity,
    mode: SpellLightMode,
) -> bool {
    let Some((l, fx)) = lights.iter().find_map(|l| l.spell.map(|fx| (l, fx))) else {
        return false;
    };
    spawn_spell_light_child(
        commands,
        root,
        l.def.position,
        SpellLightSource {
            color: l.def.diffuse_color,
            intensity: l.def.diffuse_intensity,
            kind: fx.kind,
            onset: fx.onset,
        },
        mode,
    )
    .is_some()
}

/// MONKEY (area spell light): the ONE light a persistent ground effect throws, hung on the
/// DynamicObject anchor itself so it lives and dies with the area object.
///
/// `kind` is the area lane's own verdict ([`benilla_formats::area_light_kind`] — the model path,
/// then the spell's school, then the model's hue), and it is handed in rather than read back off
/// the model because **the model routinely has nothing to read**: Flamestrike's burning patch and
/// Explosive Trap's are flat animated decals with zero particle emitters, so
/// `synthesize_spell_light` returns `None` for exactly the effects this feature is about. Where the
/// model DOES carry a synthesised light we take its colour, intensity and onset (the artist's own
/// ramp beats any default); where it does not we light it from the school's hue at the area
/// fallback rung.
///
/// The light hangs [`AREA_LIGHT_LIFT`](benilla_formats::AREA_LIGHT_LIFT) above the emitter's own
/// point — a dynobj sits ON the ground, and the faithful `1/(0.7d + 0.03d²)` falloff at `d → 0`
/// would blow out the terrain under the centre while the rim of the same patch got nothing.
///
/// `radius` is the wire `DYNAMICOBJECT_RADIUS`: it sizes the interior pool
/// ([`benilla_formats::area_reach`], not the shared intensity bucket — an area effect's footprint
/// is stated on the wire and guessing it from particle size would be a worse answer than the one we
/// were handed) and it is the impact dedupe's test radius.
pub(super) fn spawn_area_spell_light(
    commands: &mut Commands,
    lights: &[benilla_assets::ModelLight],
    kind: benilla_formats::SpellLightKind,
    anchor: Entity,
    radius: f32,
) -> bool {
    if !kind.lights() {
        return false;
    }
    // The model's own synthesised light, if it has one — its `SpellLightInfo` is deliberately NOT
    // read: the school is the area lane's (handed in above, school column included) and the onset
    // is a fact about an emitter's FIRST pass, i.e. a fuse, on an object that loops until the
    // server destroys it. The ground catches when the object appears.
    let synth = lights.iter().find(|l| l.spell.is_some());
    let (local, source) = match synth {
        Some(l) => (
            l.def.position,
            SpellLightSource {
                color: benilla_formats::area_color(kind, Some(l.def.diffuse_color)),
                intensity: l.def.diffuse_intensity,
                kind,
                onset: 0.0,
            },
        ),
        // The decal-only half: no emitter to read, so the school's own hue at the area rung, hung
        // over the object's origin (its model has no emitter position either).
        None => (
            [0.0, 0.0, 0.0],
            SpellLightSource {
                color: benilla_formats::area_color(kind, None),
                intensity: benilla_formats::AREA_FALLBACK_INTENSITY,
                kind,
                onset: 0.0,
            },
        ),
    };
    let lifted = [
        local[0],
        local[1],
        local[2] + benilla_formats::AREA_LIGHT_LIFT,
    ];
    // The breathing phase: seeded from the anchor's bits so two overlapping patches swell out of
    // step, and stable for the light's whole life (it is baked into the mode, never re-rolled).
    let phase = (anchor.to_bits() % 1000) as f32 * 0.006_283_2;
    let Some(glow) = spawn_spell_light_child(
        commands,
        anchor,
        lifted,
        source,
        SpellLightMode::Area { phase },
    ) else {
        return false;
    };
    let reach = benilla_formats::area_reach(radius);
    commands.entity(glow).insert((
        AreaSpellLight { radius },
        // The pool's own size, stated rather than bucketed — see [`benilla_formats::area_reach`].
        benilla_world::lighting::LightReach(reach),
    ));
    if benilla_assets::trace::enabled() {
        // Beside the shared `spell light` line the helper already wrote: the two numbers only an
        // area light has, so "did the patch light, and how big was its pool" is answered from the
        // trace instead of by eye.
        benilla_assets::trace::line(
            "fx",
            &format!("area light e={glow} radius={radius:.1} reach={reach:.1}"),
        );
    }
    true
}

/// What one spell light is made of, as its spawn site knows it — the synthesised model light's
/// numbers, or (MONKEY (area spell light)) the area lane's school-derived stand-in for a model that
/// carries none.
struct SpellLightSource {
    /// Linear RGB, hue preserved.
    color: [f32; 3],
    /// The authored `diffuse_intensity` the [`point_light`] recipe scales.
    intensity: f32,
    /// The school, for the trace and any future per-school gain.
    kind: benilla_formats::SpellLightKind,
    /// Seconds from the instance's birth before the light comes up (a firework's fuse).
    onset: f32,
}

/// The shared body of the two spawn sites (the effect lanes' [`spawn_spell_light`] and the
/// entity lane's firework branch): one `PointLight` child at `local` (WoW model space, relative to
/// `parent`), carrying the tags every spell light must have.
///
/// The tag set is three separate promises, and each one is load-bearing:
/// - [`SyntheticFireLight`] — this light was INVENTED, not authored, so the packer's synthetic gain
///   owns it and the `WOW_POINTS_DUMP` census counts it as one.
/// - [`SpellFxLight`] — MONKEY (spellLightGain): the world-side twin of [`SpellLight`], and the
///   one thing benilla-world is told about a spell light. It OVERRIDES the tag above in the
///   packer's gain fold: a spell row takes `spellLightGain`, never `fireLightGain` (they compose
///   nowhere — see [`benilla_world::lighting::SpellFxLight`] for why the marker is world-side
///   rather than `SpellLight` itself moving down).
/// - [`HeldLight`] — **never** an exterior torch-shadow caster. A spell light moves (a missile), is
///   born and dies inside a second, and would churn the cube-shadow cache at frame rate for a
///   shadow nobody could resolve in the time it exists. The marker is that lane's refusal.
/// - [`SpellLight`] — the envelope and the budget ([`advance_spell_lights`] /
///   [`budget_spell_lights`]), the handle any future per-school gain targets, and (MONKEY (spell
///   light lane)) what [`claim_carried_light_rooms`] reads to let this light claim rooms while it
///   MOVES: a missile can never settle, and a light with no claims takes the exterior lane, which
///   the room shaders never read.
///
/// No [`FlameFlicker`](benilla_world::lighting::FlameFlicker), deliberately: the flicker makes a
/// fire breathe, and a spell light already has a shape of its own. The two together read as the
/// effect stuttering rather than as fire.
///
/// `None` when the kill switch is off — so a caller can tell "disabled" from "this model has no
/// light", though neither has any further consequence.
fn spawn_spell_light_child(
    commands: &mut Commands,
    parent: Entity,
    local: [f32; 3],
    fx: SpellLightSource,
    mode: SpellLightMode,
) -> Option<Entity> {
    if !spell_lights_enabled() {
        return None;
    }
    let lit = point_light(fx.color, fx.intensity);
    // Born DARK and raised by the envelope: the onset is real time (a firework's fuse), and a
    // light that showed its full strength on its first frame and only then ramped would flash
    // once before every effect it belongs to.
    let base = lit.intensity;
    let glow = commands
        .spawn((
            WorldPointLight {
                intensity: 0.0,
                ..lit
            },
            Transform::from_translation(wow_to_bevy(local)),
            Visibility::default(),
            SyntheticFireLight,
            SpellFxLight,
            HeldLight,
            SpellLight::new(base, fx.onset, mode),
            // The same owner exclusion every carried light takes: the flame sits INSIDE the thing
            // that burns it, so the effect's own meshes must not occlude their own light.
            LightOwner::Instance(parent),
        ))
        .id();
    commands.entity(parent).add_child(glow);
    if benilla_assets::trace::enabled() {
        benilla_assets::trace::line(
            "fx",
            &format!(
                "spell light e={glow} school={} onset={:.2} base={base:.1}",
                fx.kind.label(),
                fx.onset
            ),
        );
    }
    Some(glow)
}

/// MONKEY (fire GO lights): how far up the hierarchy the room walk looks before giving up. A
/// carried light's anchor is at most `light → joint → … → net entity`, and a rig's joint chain is
/// the deep part; 64 clears every shipped skeleton with room to spare, and the cap is there so a
/// malformed cycle can never hang the frame.
const ANCHOR_WALK_DEPTH: usize = 64;

/// MONKEY (carried light stability): the bearer crossfade weight at which a carried light COMMITS
/// to its owner's room. [`InteriorAnchor::lane`] ramps at 2/s, so crossing this midpoint needs
/// ~0.25 s of a HELD verdict; one frame of a wrong law moves the weight by `dt * 2` and can never
/// cross it. That IS the hysteresis — no second timer, and because it is the very signal the
/// bearer's own meshes crossfade their lighting on, the light changes lane with the body.
const LANE_INTERIOR: f32 = 0.5;

/// MONKEY (carried light stability): how far a carried light may drift and still count as standing
/// still. Deliberately HALF `torch_shadow`'s 0.1 yd cache-staleness radius: the settle test has to
/// be the stricter of the two, or a light could qualify as a shadow caster while already sitting
/// far enough from its baked map to have that map withdrawn again on the next step.
const STILL_DRIFT: f32 = 0.05;

/// MONKEY (carried light stability): how long a carried light must hold still before it may cast.
/// Long enough that a pet fidgeting between waypoints never qualifies, short enough that a placed
/// brazier GameObject is eligible well inside the first second it exists.
const STILL_HOLD: f32 = 0.75;

/// MONKEY (carried light stability): how long a carried light has been standing still, so the
/// torch-shadow lane can refuse a MOVING one.
///
/// The cube-shadow cache is built for fixtures that do not move: a slot keeps its baked depth only
/// while its fixture stays within 0.1 yd of where the map was baked
/// (`torch_shadow::map_publishable`), and a slot whose map is withdrawn has its cross-fade weight
/// forced to 0 with no ramp (`static_gx::torch_depth`'s `TorchTable::pack`). A walking bearer
/// crossed that threshold every frame, which is a full-strength shadow blinking at frame rate —
/// and worse, it burned the lane's global two-rebuilds-a-frame budget re-baking a mesh that was
/// stale again immediately, starving the STATIC candles around it so those flickered too. One
/// pet's shadow is not worth that.
///
/// A carried light is not excluded outright, because the same `ChildOf` spawn path carries the
/// stationary cases the shadow lane was written for: a GM-placed brazier GameObject in the Lion's
/// Pride Inn, a campfire. Those settle within [`STILL_HOLD`] and cast normally; only things that
/// actually move are refused. Refusal is a *candidacy* verdict, so an incumbent that starts
/// walking is cross-faded out by the lane's own eviction path rather than cut.
#[derive(Component)]
pub(crate) struct CarriedLightMotion {
    /// World position at the last frame the light was judged to have MOVED. Not re-anchored while
    /// it is still, so a slow creep accumulates and eventually trips the test — which is right,
    /// because the shadow cache's own staleness test is likewise absolute from the bake point.
    last: Vec3,
    /// Seconds held within [`STILL_DRIFT`] of `last`, capped at [`STILL_HOLD`]. The cap is what
    /// keeps a settled light a pure READ: no field write, so no change tick, so nothing
    /// downstream is woken by a light that is doing nothing.
    still_for: f32,
}

impl CarriedLightMotion {
    /// Has this light held still long enough to be trusted with a cached cube shadow?
    pub(crate) fn settled(&self) -> bool {
        self.still_for >= STILL_HOLD
    }
}

/// MONKEY (carried light stability): maintain [`CarriedLightMotion`] for every carried light.
///
/// `PostUpdate`, **after transform propagation**, for the same reason the packer is there: a
/// carried light is a child of a moving parent, so its `GlobalTransform` is only this frame's once
/// `Propagate` has run. `update_torch_shadows` reads the verdict in `Last`, i.e. later in the same
/// frame — never one stale.
#[allow(clippy::type_complexity)]
pub(crate) fn track_carried_light_motion(
    mut commands: Commands,
    time: Res<Time>,
    mut lights: Query<
        (Entity, &GlobalTransform, Option<&mut CarriedLightMotion>),
        (
            With<WorldPointLight>,
            With<ChildOf>,
            Without<ShadowProxyLight>,
        ),
    >,
) {
    let dt = time.delta_secs();
    for (light, gt, motion) in &mut lights {
        let p = gt.translation();
        match motion {
            Some(mut m) => {
                if p.distance_squared(m.last) > STILL_DRIFT * STILL_DRIFT {
                    m.last = p;
                    m.still_for = 0.0;
                } else if m.still_for < STILL_HOLD {
                    m.still_for += dt;
                }
            }
            // Born moving: a light that spawns mid-stride must not be eligible on its first frame
            // merely because it has no history yet.
            None => {
                commands.entity(light).try_insert(CarriedLightMotion {
                    last: p,
                    still_for: 0.0,
                });
            }
        }
    }
}

/// The bearer state a carried light's claim is derived from — the room its [`InteriorAnchor`]
/// names and that anchor's crossfade WEIGHT — walked up the hierarchy from the light itself,
/// because the light hangs off a BONE JOINT (or the model frame) and the anchor is the model's
/// net-entity root several links above it (`interior::part_interior_lit`: "`anchor` is the model's
/// NET ENTITY root for every caller").
fn owner_lane(
    light: Entity,
    parents: &Query<&ChildOf>,
    anchors: &Query<&InteriorAnchor>,
) -> (Option<WmoRoom>, f32) {
    let mut e = light;
    for _ in 0..ANCHOR_WALK_DEPTH {
        if let Ok(anchor) = anchors.get(e) {
            return (anchor.room(), anchor.lane());
        }
        match parents.get(e) {
            Ok(c) => e = c.parent(),
            Err(_) => break,
        }
    }
    // No anchor above us at all (a bearer with neither a lit part nor a lit emitter): exterior,
    // and settled there — the same answer the old walk's `None` produced.
    (None, 0.0)
}

/// The room a light's [`LightRooms`] claim currently stands for — our own record of it, because
/// `LightRooms` wraps a `WmoGroupVis` whose fields are benilla-world-private and cannot be read
/// back for a comparison. Absent = never claimed.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ClaimedRoom {
    /// What the light's `LightRooms` / `LightLane` currently stand for. `None` = exterior lane.
    applied: Option<WmoRoom>,
    /// MONKEY (carried light stability): the last room the bearer's anchor actually NAMED. Held
    /// across the exterior ramp: the frame a bearer's law flips to `Exterior` its `room` goes
    /// `None` at once, while the lane weight the claim is judged on takes ~0.25 s to fall past
    /// [`LANE_INTERIOR`]. Without this there would be no room left to keep claiming during that
    /// window, and the light would drop to the exterior lane on the raw flip — exactly the
    /// one-frame verdict the weight exists to smooth.
    seen: Option<WmoRoom>,
    /// MONKEY (GO room claims): the [`LightLane`] currently on the light. The lane used to be a
    /// pure function of `applied` (`is_some()`), so it needed no record; it is now also a function
    /// of the CLAIM SET (a light standing in an exterior-class group that claims a real room moves
    /// onto the interior lane — see [`CarriedClaims`]), and those two move independently. Recorded
    /// so the `LightLane` insert stays change-gated: an insert is an archetype write, and the
    /// render extraction behind it wakes on every one.
    lane: bool,
}

/// MONKEY (GO room claims): how far a carried light may drift (yd) before its claim set is
/// rebuilt. Ten times [`STILL_DRIFT`] on purpose — the settle test is about whether a CACHE of the
/// light's surroundings is still valid frame to frame, this one is about whether a set of ROOMS is
/// still the right set, and a room is yards across. Half a yard is well inside the smallest shipped
/// doorway, so a light cannot cross a portal without tripping it.
const CLAIM_DRIFT: f32 = 0.5;

/// MONKEY (GO room claims): a carried light's [`LightLitRooms`] claim set, plus the three inputs
/// the rebuild is gated on. Our own record, for the same reason [`ClaimedRoom`] is one:
/// `LightLitRooms` wraps a crate-private `WmoGroupVis` and cannot be read back for a comparison.
#[derive(Component)]
pub(crate) struct CarriedClaims {
    /// The set currently applied — `None` when the light carries no claims at all (outdoors, or
    /// moving, or standing in no building's authored box).
    set: Option<CarriedClaimSet>,
    /// The world position the set was decided at. Absolute, not re-anchored: a light that creeps
    /// out of the room it claimed must eventually rebuild, and the drift budget is the room's.
    at: Vec3,
    /// The [`WmoResidency`] generation it was decided under — a light that claimed nothing because
    /// its building had not streamed in yet has to be re-asked once it has, and the reverse (the
    /// placement entity a claim is keyed to is gone) is a stale instance id in the GPU claim head.
    generation: u32,
    /// The bearer room it was decided under: the light walked through a door, so the claim seed
    /// (and the preferred placement) moved even if the light itself barely did.
    room: Option<WmoRoom>,
}

impl CarriedClaims {
    /// MONKEY (GO room claims): **the recompute gate.** `true` = the applied set still stands for
    /// this position, this residency and this bearer room, so the whole claim walk is skipped.
    ///
    /// Split out as a pure function so it can be tested without a world: the walk it guards is the
    /// expensive half of this system (an O(groups) box scan per resident placement), and a gate
    /// that silently stopped firing would show up not as a wrong picture but as a stale one.
    fn fresh(&self, at: Vec3, generation: u32, room: Option<WmoRoom>) -> bool {
        self.generation == generation
            && self.room == room
            && self.at.distance_squared(at) <= CLAIM_DRIFT * CLAIM_DRIFT
    }
}

/// MONKEY (GO room claims): does this claim set justify moving the light onto the INTERIOR lane?
///
/// Two conditions, and the second is the safety half.
///
/// 1. **It must name an interior-class group.** Claims are read by exactly one function,
///    `static_gx.wgsl`'s `interior_room_light`, and that function skips every table entry whose
///    colour-row `.w < 0.5` — i.e. every EXTERIOR-lane light. So on the exterior lane a claim set
///    is inert: it is packed as the UNGATED head (`build_light_data` only calls `RoomClaim::build`
///    on the interior arm) and no shader ever looks at it. A light that claims a real room and
///    stays exterior has therefore gained nothing, which is the whole reason the lane moves.
/// 2. **Every claim must be exterior-lane eligible** (no `LIT_ROOM_EXT_DENY` bit). Moving onto the
///    interior lane is a TRADE: the light leaves `point_light_sum` (terrain, doodads, NPCs) and
///    `wmo_exterior_point_sum` (a WMO's exterior-class walls and cobbles) and joins
///    `interior_room_light`, which reaches an exterior-class surface only through the strict
///    exterior-batch room term — and that term admits a claim only when it carries `CLAIM_EXT_OK`.
///    A district-scale shell claim (Orgrimmar's whole valley, Stormwind's districts) carries the
///    deny bit, so a light standing in one would be taken off the very cobbles it lights today and
///    given nothing back that reached them. Those keep the exterior lane and behave exactly as
///    before; a BUILDING-scale exterior group (the Goldshire inn's shell, the canyon nook a shop
///    door opens onto) is eligible, so its floor keeps the light through the strict term while the
///    interior rooms one portal hop away gain it.
///
/// The trade the rule still accepts, stated plainly: a brazier that moves onto the interior lane
/// stops feeding the nearest-3 exterior term, so ADT TERRAIN under it and the NPCs standing beside
/// it lose that light and gain the room lane instead. That is the correct side for a fixture
/// genuinely inside a building's claimable volume — it is what a MOLT fixture in the same spot
/// already does — and condition 2 is what keeps it from happening out on a district-scale street.
fn lane_worthy(set: &CarriedClaimSet) -> bool {
    set.any_interior && set.all_ext_ok()
}

/// MONKEY (GO room claims): the claim set of a light at `at`, from the first resident WMO placement
/// whose authored MOGI boxes hold it.
///
/// `prefer` is tried in order before the scan — the bearer's own anchor placement and the one the
/// light claimed last time. In the steady state one of those two always wins on the first probe,
/// which is what keeps a residency-generation tick (a building streaming in three streets away
/// re-gates every carried light in the world) from turning into a full scan per light. The scan is
/// the cold path: a GameObject standing in an EXTERIOR-class group has no anchor room at all, so
/// its very first claim has nowhere else to come from.
fn claims_at(
    wmos: &Assets<benilla_assets::WmoModel>,
    placements: &Query<(Entity, &WmoPortalInstance)>,
    prefer: [Option<Entity>; 2],
    at: Vec3,
    reach: f32,
) -> Option<CarriedClaimSet> {
    let one = |e: Entity| {
        let (e, inst) = placements.get(e).ok()?;
        carried_light_claims(wmos.get(&inst.handle)?, e, inst.world_from_local, at, reach)
    };
    for e in prefer.into_iter().flatten() {
        if let Some(set) = one(e) {
            return Some(set);
        }
    }
    placements.iter().find_map(|(e, inst)| {
        if prefer.contains(&Some(e)) {
            return None; // already probed above
        }
        carried_light_claims(wmos.get(&inst.handle)?, e, inst.world_from_local, at, reach)
    })
}

/// MONKEY (spell light lane): **may this light claim rooms this frame?**
///
/// The settle rule ([`CarriedLightMotion::settled`]) exists to keep an O(groups) box scan per
/// resident placement off a WALKING bearer: a pet's torch would rebuild its set every frame for a
/// set that is about to be wrong again, so an unsettled carried light simply carries no claims and
/// falls back to the single-anchor-room behaviour it had before claims existed.
///
/// A SPELL light inverts every term of that trade, so it is exempt:
///
/// * **It never settles, by construction.** A missile flies from the caster's hand to the target
///   and is despawned on arrival; it can never hold still for [`STILL_HOLD`], so under the settle
///   rule it could never claim a room for a single frame of its life. The gap is not a tuning
///   question — an indoor fireball lit *nothing* on the way down a corridor, and only the impact's
///   own burst (which IS momentarily still, at a dest anchor) ever reached a wall.
/// * **The fallback it would take is empty.** An unsettled carried light still has its bearer's
///   anchor room to fall back on; a missile is a FREE world entity with no `InteriorAnchor`
///   anywhere above it ([`owner_lane`] walks to the top and answers `(None, 0.0)`), so "no claims"
///   means the EXTERIOR lane — and the exterior lane is the one `interior_room_light` never reads,
///   which is why the bolt lit no walls at all.
/// * **The cost it was protecting against is bounded.** Spell lights are capped at
///   [`SPELL_LIGHTS_MAX`](super::spell_fx::SPELL_LIGHTS_MAX) = 24 across every lane, and each lives
///   for about a second. The claim walk is still change-gated by [`CarriedClaims::fresh`] — a light
///   that has not drifted past [`CLAIM_DRIFT`] since it was measured keeps its set — so the worst
///   case is 24 lights × (resident placements) containment scans in a frame, each of which
///   early-outs on an AABB test per group and does no portal work at all unless some box holds the
///   light. That is the same walk ~150 of Orgrimmar's brazier GameObjects run on their first frame.
///
/// Settling is still what the SHADOW lane reads, and this function does not touch it: a spell light
/// is [`HeldLight`] and unsettled, so it is refused a cube-shadow slot exactly as before.
fn may_claim(spell: bool, motion: Option<&CarriedLightMotion>) -> bool {
    spell || motion.is_some_and(CarriedLightMotion::settled)
}

/// MONKEY (fire GO lights): give a CARRIED light the room its owner was classified into.
///
/// A world-baked light gets its rooms at spawn from the building's own MODR/MOLR tables. An
/// entity's light cannot: a GameObject walks (or is placed) into a building the tables know nothing
/// about, and its room is only ever known from the interior classifier's down-ray. So this system
/// copies that verdict onto the light, and two things follow from it:
///
/// - **The faithful gate.** A torch in a portal-culled room stops lighting the hillside outside it
///   ([`LightRooms`], decision 0689) — until now a placed brazier GameObject lit through walls
///   because it claimed no room at all and `room_admits` admits `None` unconditionally.
/// - **Shadow eligibility.** `torch_shadow`'s caster query is `With<WorldPointLight>, With<LightRooms>`,
///   so a roomless light can never be promoted to a cube-map caster however deep indoors it stands.
///   A GM-placed brazier in the Lion's Pride Inn now throws real shadows like a MOLT fixture.
///
/// Applies to synthetic and authored carried lights alike — the room is a fact about the OWNER,
/// not about where the light's colour came from.
///
/// MONKEY (carried light stability): the verdict is taken from the anchor's crossfade WEIGHT, not
/// from its raw per-frame law. The law alternates: `WOW_INTERIOR_LOG` from the 2026-09-09 run has
/// anchor `320v2` resolving `matte`↔`bake` on ten consecutive frames (t 73.75 … t 74.14), and
/// the same threshold noise puts an `Exterior` between two interior verdicts at a doorway. A light
/// that re-decided on that raw answer would hand itself back and forth between two DIFFERENT
/// consumer families every frame — the exterior `point_light_sum` and the interior room lane —
/// which is an on/off strobe of the pool, not a subtle difference. Reading `lane >= LANE_INTERIOR`
/// costs nothing and cannot be crossed by a blip (see [`LANE_INTERIOR`]).
///
/// MONKEY (GO room claims): **and it now builds the light's [`LightLitRooms`] too** — the rooms it
/// may LIGHT, as opposed to the one it must be VISIBLE from. The bearer's anchor answers "which
/// group is this standing in", singular, and one group was the whole gate: a brazier inside the
/// Lion's Pride Inn lit its own room and stopped dead at the doorway, and a brazier in one of
/// Orgrimmar's 123 canyon placements named no group at all (`MOGP & 0x48` ⇒ the anchor is `None`)
/// so it could never reach the shops the canyon opens onto. The fix is not a second classifier: it
/// is the SAME claim rule the WMO's own MOLT fixtures and MODD props run
/// (`benilla_world::terrain_stream::carried_light_claims` → `benilla_formats::room_claims`:
/// containment tightest-first, then one portal hop within reach), asked at the light's world
/// position. Reach is the M2 intensity bucket scaled by the DEFAULT `interiorAttenScale`, exactly
/// as the prop lane sizes its own hop — the live cvar must not be able to open a doorway that was
/// gated when the set was built, or every claim in the world would rebuild on a slider drag.
///
/// **Gated on four things**, because the walk is an O(groups) box scan per resident placement and
/// Orgrimmar has ~150 of these lights: the light must be allowed to claim ([`may_claim`] — SETTLED
/// for an ordinary carried light, so a walking pet's light re-claims no more often than the settle
/// cadence and while it is unsettled and off its build point it carries NO claims, i.e. exactly the
/// single-anchor-room behaviour it had before this change; a SPELL light is exempt and claims while
/// it flies), and the set is rebuilt only when the light drifts past [`CLAIM_DRIFT`], when
/// [`WmoResidency`] ticks, or when the bearer's room changes ([`CarriedClaims::fresh`]).
///
/// MONKEY (spell light lane): and the LANE half is what makes a moving claim worth anything. A
/// missile has no [`InteriorAnchor`] above it at all, so `want` is `None` and the bearer half of
/// the lane verdict can never fire; it reaches the interior lane purely through [`lane_worthy`]'s
/// claim half, which is the path an anchorless GameObject in an exterior-class canyon group already
/// took. Without it the bolt packs on the EXTERIOR lane, whose entries `interior_room_light` skips
/// outright (`static_gx.wgsl` drops every row with colour-row `.w < 0.5`) — the room's walls never
/// see it, and only the impact's own burst, which lands still at a dest anchor, ever lit anything.
///
/// Every component write below is change-gated on [`ClaimedRoom`]/[`CarriedClaims`]: a settled
/// light is one hierarchy walk and three compares per frame, and nothing is written, so the light's
/// archetype (and the render extraction behind it) stays quiet while an NPC stands still in a room.
#[allow(clippy::type_complexity)]
pub(crate) fn claim_carried_light_rooms(
    mut commands: Commands,
    // Carried lights only — `With<ChildOf>` IS the distinction: a world-baked doodad/MOLT light
    // spawns as a free root with its rooms already correct, and re-deciding them from a hierarchy
    // it isn't in would clear them.
    mut lights: Query<
        (
            Entity,
            // MONKEY (GO room claims): the light's own WORLD position (the claim rule's input) and
            // its intensity (which sizes the portal hop). `Update` reads last frame's propagation,
            // which is this frame's answer for anything settled.
            // MONKEY (spell light lane): a MOVING spell light now claims too, so for that one case
            // this position is one frame stale — at a missile's ~25 yd/s that is ~0.4 yd at 60 Hz.
            // Accepted rather than re-scheduled: the answer it feeds is ROOM MEMBERSHIP, whose
            // smallest feature is a doorway a yard wide, and the light's brightness at a wall
            // changes by nothing measurable over that distance. The alternative — moving this
            // system into `PostUpdate` between `Propagate` and the packer — would put a full
            // O(placements) walk for every carried light in the world (Orgrimmar: ~150) into the
            // frame's tightest stage to buy 0.4 yd on 24 of them.
            &GlobalTransform,
            &WorldPointLight,
            Option<&CarriedLightMotion>,
            Option<&mut ClaimedRoom>,
            Option<&mut CarriedClaims>,
            // MONKEY (spell light lane): is this a spell light? It is the one carried light that
            // claims while MOVING — see [`may_claim`].
            Has<SpellLight>,
        ),
        (With<ChildOf>, Without<ShadowProxyLight>),
    >,
    parents: Query<&ChildOf>,
    anchors: Query<&InteriorAnchor>,
    // MONKEY (GO room claims): the resident placements + their models — the claim rule's tables.
    placements: Query<(Entity, &WmoPortalInstance)>,
    wmos: Res<Assets<benilla_assets::WmoModel>>,
    residency: Res<WmoResidency>,
) {
    let generation = residency.generation();
    for (light, gt, pl, motion, mut claimed, mut claims, spell) in &mut lights {
        let (room, lane) = owner_lane(light, &parents, &anchors);
        // The room to claim IF we are on the interior lane: the one the anchor names this frame,
        // else the last one it named (the ramp-down window described on `ClaimedRoom::seen`).
        let seen = room.or_else(|| claimed.as_deref().and_then(|c| c.seen));
        let want = seen.filter(|_| lane >= LANE_INTERIOR);

        // MONKEY (GO room claims): the LIT-room claim set, rebuilt only on the gates above.
        // `None` below means "keep what is already applied"; `Some(x)` means "apply x".
        let at = gt.translation();
        let fresh = claims
            .as_deref()
            .is_some_and(|c| c.fresh(at, generation, want));
        let set = if fresh {
            None
        } else if may_claim(spell, motion) {
            // The prop lane's own sizing: the intensity bucket (`PointLight` premultiplied 4π at
            // spawn, so this inverts it exactly) through the DEFAULT atten scale.
            let reach = benilla_formats::room_claim::claim_reach(
                benilla_formats::room_claim::m2_light_reach(
                    pl.intensity / (4.0 * std::f32::consts::PI),
                ),
            );
            // MONKEY (spell light lane): for a spell light the FIRST entry is usually `None` —
            // a missile has no bearer anchor to name a placement — so the steady-state probe is
            // the SECOND: the building it claimed last frame. A bolt flying down a corridor stays
            // in the building it was cast in, so that probe hits on every frame after the first
            // and the full placement scan runs once per spell light, at birth.
            let prefer = [
                want.map(|r| r.instance),
                claims
                    .as_deref()
                    .and_then(|c| c.set.as_ref())
                    .map(|s| s.instance),
            ];
            Some(claims_at(&wmos, &placements, prefer, at, reach))
        } else {
            // Moved off its build point and not standing still: drop to the pre-claims behaviour
            // rather than assert a set for a position it was not measured at. A SPELL light never
            // reaches here ([`may_claim`]) — for it the "position it was measured at" is simply
            // re-measured, one frame late (see the `GlobalTransform` note on the query).
            Some(None)
        };
        // The set that stands THIS frame: the one just decided, else the one already applied. Read
        // from the local rather than back off the component, because on a light's first frame the
        // component is still a queued command and re-reading it would see `None` and remove the
        // very `LightLitRooms` this pass is inserting.
        let had = claims.as_deref().and_then(|c| c.set.as_ref());
        let next_set: Option<&CarriedClaimSet> = match &set {
            Some(next) => next.as_ref(),
            None => had,
        };
        let set_changed = match (next_set, had) {
            (None, None) => false,
            // MONKEY (soft portal claims): the FADES are part of the set. A light that drifts
            // across a room keeps claiming the same groups while the reach it has left past each
            // doorway shrinks — comparing ids alone would leave it faded by the distances it had
            // at its first build point, for as long as it never changed rooms.
            (Some(next), Some(had)) => {
                next.instance != had.instance
                    || next.groups != had.groups
                    || next.fades != had.fades
            }
            _ => true,
        };
        // MONKEY (GO room claims): the lane is the OR of the two halves — the bearer's own room,
        // and (new) a claim set that earns the interior lane on its own (see [`lane_worthy`]).
        let want_lane = want.is_some() || next_set.is_some_and(lane_worthy);

        // What the light already carries, so each insert below is earned.
        let (had_room, had_lane) = claimed
            .as_deref()
            .map_or((None, None), |c| (Some(c.applied), Some(c.lane)));

        let mut e = commands.entity(light);
        // MONKEY (light lane by position): the carried lane's own lane verdict. A bearer's room
        // came from the interior classifier's ray at the bearer's own feet, so "my owner is in a
        // room" IS "I am physically inside one" — no second ray, and the static classifier
        // (`classify_light_lanes`) deliberately skips `With<ChildOf>` so the two never fight.
        // MONKEY (GO room claims): plus the claim half — a light standing in an EXTERIOR-class
        // group (so: no anchor room) that nonetheless claims a real interior room through a portal
        // has to be on the interior lane for those claims to be read at all (see [`lane_worthy`]).
        if had_lane != Some(want_lane) {
            e.insert(LightLane::carried(want_lane));
        }
        if had_room != Some(want) {
            match want {
                Some(room) => {
                    e.insert(LightRooms::new(WmoGroupVis::single(
                        room.instance,
                        room.group,
                    )));
                }
                // Back outdoors (or the anchor went away): drop the claim rather than leave a
                // stale room, which would gate the light on a building it has walked out of.
                None => {
                    e.remove::<LightRooms>();
                }
            }
        }
        if set_changed {
            match next_set {
                Some(next) => {
                    e.insert(next.component());
                }
                None => {
                    e.remove::<LightLitRooms>();
                }
            }
        }

        // Bookkeeping LAST — every verdict above borrows it. `seen` is a record, not a component:
        // updating it must not touch the light's COMPONENTS, or a bearer walking a corridor would
        // churn the archetype for nothing.
        match claimed.as_mut() {
            Some(c) => {
                let (applied, lane) = (c.applied, c.lane);
                c.set_if_neq(ClaimedRoom {
                    applied,
                    seen,
                    lane,
                });
                c.applied = want;
                c.lane = want_lane;
            }
            None => {
                e.try_insert(ClaimedRoom {
                    applied: want,
                    seen,
                    lane: want_lane,
                });
            }
        }
        if let Some(next) = set {
            let next = CarriedClaims {
                set: next,
                at,
                generation,
                room: want,
            };
            match claims.as_mut() {
                Some(c) => **c = next,
                None => {
                    e.try_insert(next);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use benilla_formats::M2Light;
    use std::time::Duration;

    fn light(light_type: u16, bone: i16, position: [f32; 3], visibility_off: bool) -> ModelLight {
        ModelLight {
            def: M2Light {
                light_type,
                bone,
                position,
                bone_z: [0.0, 0.0, 1.0],
                ambient_color: [1.0; 3],
                ambient_intensity: 0.0,
                diffuse_color: [0.466_666_7, 0.290_196_1, 0.133_333_34], // the real torch's warm orange
                diffuse_intensity: 3.0,
                attenuation_start: 1.388_889,
                attenuation_end: 2.222_222_3,
                visibility_off,
            },
            bone_pivot: [1.0, 0.0, 0.5],
            synthetic: false,
            flame: false,
            spell: None,
        }
    }

    /// Only casting lights spawn (the gather's gate in `0x718960`), each a child of its bone's
    /// joint at `position − bone_pivot`; colour × intensity survives the packer's `intensity / 4π`.
    #[test]
    fn only_casting_lights_spawn_and_they_ride_their_bone() {
        let mut app = App::new();
        let frame = app.world_mut().spawn(Transform::IDENTITY).id();
        let joint = app.world_mut().spawn(Transform::IDENTITY).id();
        let lights = [
            light(1, 3, [2.0, 0.0, 1.5], false),  // casts, on bone 3
            light(0, 3, [2.0, 0.0, 1.5], false),  // directional → ambient term, never a GL light
            light(1, 3, [2.0, 0.0, 1.5], true),   // point but authored dark
            light(1, -1, [0.0, 0.0, 4.0], false), // casts, boneless → the frame itself
        ];
        app.world_mut().commands().queue(move |world: &mut World| {
            let mut q = world.commands();
            spawn_carried_lights(&mut q, &lights, frame, true, move |bone| {
                (bone == 3).then_some(joint)
            });
        });
        app.world_mut().flush();

        let mut spawned: Vec<(Entity, Vec3, Entity)> = app
            .world_mut()
            .query::<(Entity, &WorldPointLight, &Transform, &ChildOf)>()
            .iter(app.world())
            .map(|(e, _, t, c)| (e, t.translation, c.parent()))
            .collect();
        // Spawn order, i.e. light-table order: `Entity`'s own `Ord` is not index-ascending.
        spawned.sort_by_key(|(e, ..)| e.index());
        assert_eq!(
            spawned.len(),
            2,
            "the directional and the dark one stay out"
        );

        assert_eq!(spawned[0].2, joint);
        assert_eq!(spawned[0].1, wow_to_bevy([1.0, 0.0, 1.0]));
        // Boneless: off the frame in model space, as a held item's light always is.
        assert_eq!(spawned[1].2, frame);
        assert_eq!(spawned[1].1, wow_to_bevy([0.0, 0.0, 4.0]));

        let pl = app
            .world()
            .entity(spawned[0].0)
            .get::<WorldPointLight>()
            .unwrap();
        let recovered = pl.intensity / (4.0 * std::f32::consts::PI);
        assert!(
            (pl.color[0] * recovered - 1.4).abs() < 1e-3,
            "colour × intensity survives the packing"
        );
    }

    /// MONKEY (outdoor torch shadows). The [`HeldLight`] marker is written at the SPAWN site and
    /// nowhere else, because the spawn site is the only place that knows who owns the light: after
    /// this, a guard's torch and a placed brazier are both just a `PointLight` under a `ChildOf`.
    /// It is a pure ADDITION — the light's own components (colour, intensity, flicker, synthetic
    /// tag) and its parenting are identical either way, so nothing about how it LIGHTS the world
    /// moves; only the exterior shadow lane reads it.
    #[test]
    fn a_body_carried_light_is_marked_and_a_placed_one_is_not() {
        for held in [true, false] {
            let mut app = App::new();
            let frame = app.world_mut().spawn(Transform::IDENTITY).id();
            let lights = [light(1, -1, [0.0, 0.0, 4.0], false)];
            app.world_mut().commands().queue(move |world: &mut World| {
                let mut q = world.commands();
                spawn_carried_lights(&mut q, &lights, frame, held, |_| None);
            });
            app.world_mut().flush();
            let marked = app
                .world_mut()
                .query::<(&WorldPointLight, Has<HeldLight>)>()
                .iter(app.world())
                .map(|(_, h)| h)
                .collect::<Vec<_>>();
            assert_eq!(marked, vec![held], "held = {held}");
        }
    }

    /// GOLDEN — MONKEY (carried light stability). A SYNTHESISED light does NOT ride its host
    /// bone even when the instance carries that joint: it hangs off the model FRAME at the raw
    /// model-space position, i.e. exactly where the bone ride would put it in the REST pose. The
    /// frame is the model root, so the light still follows the unit with zero lag; what is dropped
    /// is the per-frame animation swing that made the imp's green pool strobe (it re-ranked the
    /// shaders' per-vertex nearest-3 and invalidated the cube-shadow cache every frame).
    #[test]
    fn a_synthetic_light_stays_on_the_rest_pose_pivot() {
        let mut app = App::new();
        let frame = app.world_mut().spawn(Transform::IDENTITY).id();
        let joint = app.world_mut().spawn(Transform::IDENTITY).id();
        let mut synth = light(1, 3, [2.0, 0.0, 1.5], false);
        synth.synthetic = true;
        synth.flame = true; // MONKEY (flame flicker): the flame route — the imp's hand fire
        let lights = [synth];
        app.world_mut().commands().queue(move |world: &mut World| {
            let mut q = world.commands();
            spawn_carried_lights(&mut q, &lights, frame, true, move |bone| {
                (bone == 3).then_some(joint)
            });
        });
        app.world_mut().flush();

        let spawned: Vec<(Vec3, Entity)> = app
            .world_mut()
            .query::<(&WorldPointLight, &Transform, &ChildOf)>()
            .iter(app.world())
            .map(|(_, t, c)| (t.translation, c.parent()))
            .collect();
        assert_eq!(spawned.len(), 1);
        assert_eq!(spawned[0].1, frame, "the frame, never the joint");
        assert_eq!(
            spawned[0].0,
            wow_to_bevy([2.0, 0.0, 1.5]),
            "raw model-space position, NOT rebased by the bone pivot"
        );
        // The tag still rides it: the packer must keep folding `fireLightGain` over this one.
        assert!(app
            .world_mut()
            .query::<&SyntheticFireLight>()
            .iter(app.world())
            .next()
            .is_some());
    }

    /// GOLDEN — MONKEY (carried light stability). A carried light earns its cube-shadow
    /// eligibility by HOLDING STILL for [`STILL_HOLD`], and loses it the instant it moves further
    /// than [`STILL_DRIFT`]. This is what keeps a walking bearer out of `torch_shadow`'s candidate
    /// set: a moving fixture invalidates its cached depth map every frame, and a withdrawn map
    /// zeroes that slot's cross-fade weight with no ramp — a shadow blinking at frame rate.
    #[test]
    fn a_carried_light_casts_only_after_it_holds_still() {
        let mut app = App::new();
        app.insert_resource(Time::<()>::default());
        app.add_systems(Update, track_carried_light_motion);
        let bearer = app.world_mut().spawn(Transform::IDENTITY).id();
        let light = app
            .world_mut()
            .spawn((
                WorldPointLight {
                    color: [1.0, 1.0, 1.0],
                    intensity: 0.0,
                    range: 48.0,
                },
                Transform::IDENTITY,
                GlobalTransform::IDENTITY,
                ChildOf(bearer),
            ))
            .id();

        // First sight of it: history exists, but nothing has been HELD yet.
        app.update();
        let settled = |app: &App, e: Entity| {
            app.world()
                .entity(e)
                .get::<CarriedLightMotion>()
                .is_some_and(CarriedLightMotion::settled)
        };
        assert!(!settled(&app, light), "born moving is not settled");

        // A second of standing still clears the hold.
        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(Duration::from_secs_f32(1.0));
        app.update();
        assert!(settled(&app, light), "held still past STILL_HOLD");

        // One stride and it is disqualified again, with no credit for the second it banked.
        *app.world_mut()
            .entity_mut(light)
            .get_mut::<GlobalTransform>()
            .unwrap() = GlobalTransform::from_translation(Vec3::new(1.0, 0.0, 0.0));
        app.world_mut()
            .resource_mut::<Time>()
            .advance_by(Duration::from_secs_f32(1.0 / 60.0));
        app.update();
        assert!(!settled(&app, light), "moved further than STILL_DRIFT");
    }

    /// GOLDEN — MONKEY (GO room claims). The claim-set RECOMPUTE GATE. Building a carried light's
    /// lit-room set is an O(groups) authored-box scan per resident WMO placement, and Orgrimmar
    /// alone places ~150 of these lights, so the walk must run on a change and never on a frame.
    /// All three inputs invalidate, and none of them alone is enough to be left out:
    /// residency (a building streamed in under a light that had claimed nothing), the bearer's room
    /// (it walked through a door, so the claim seed moved even though the light barely did), and
    /// drift past [`CLAIM_DRIFT`] (it is no longer where the set was measured).
    #[test]
    fn a_claim_set_is_rebuilt_only_on_residency_room_or_drift() {
        let room = |group| {
            Some(WmoRoom {
                instance: Entity::PLACEHOLDER,
                group,
            })
        };
        let held = CarriedClaims {
            set: None,
            at: Vec3::new(10.0, 2.0, -4.0),
            generation: 7,
            room: room(3),
        };
        assert!(
            held.fresh(held.at, 7, room(3)),
            "nothing moved: no walk this frame"
        );
        assert!(
            held.fresh(held.at + Vec3::X * (CLAIM_DRIFT * 0.9), 7, room(3)),
            "a wobble inside the drift budget still stands"
        );
        assert!(
            !held.fresh(held.at + Vec3::X * (CLAIM_DRIFT * 1.1), 7, room(3)),
            "past CLAIM_DRIFT the set was measured somewhere else"
        );
        assert!(
            !held.fresh(held.at, 8, room(3)),
            "a placement streamed in or out: the claim's instance key may be stale"
        );
        assert!(!held.fresh(held.at, 7, room(4)), "the bearer changed room");
        assert!(
            !held.fresh(held.at, 7, None),
            "the bearer left the building"
        );
    }

    /// GOLDEN — MONKEY (spell light lane). A SPELL light claims rooms **while it is moving**; every
    /// other carried light must hold still first.
    ///
    /// This is the whole of the missile fix in one predicate. A projectile is despawned on arrival
    /// and can never satisfy [`STILL_HOLD`], so under the settle rule it claimed nothing for its
    /// entire life — and with no [`InteriorAnchor`] above it either, "no claims" left it on the
    /// EXTERIOR lane, which `interior_room_light` never reads. An indoor fireball therefore lit no
    /// wall at any point of its flight; only its impact, which lands still, ever did.
    ///
    /// The exemption is deliberately NOT "everything unsettled claims": the settle rule still keeps
    /// a walking pet's torch from rebuilding an O(groups) scan per resident placement every frame.
    /// It is bought for the spell lane alone, where the population is capped
    /// ([`super::spell_fx::SPELL_LIGHTS_MAX`]) and each light lives about a second.
    #[test]
    fn a_spell_light_claims_while_it_flies_and_a_torch_still_must_settle() {
        let moving = CarriedLightMotion {
            last: Vec3::ZERO,
            still_for: 0.0,
        };
        let parked = CarriedLightMotion {
            last: Vec3::ZERO,
            still_for: STILL_HOLD,
        };
        assert!(
            may_claim(true, Some(&moving)),
            "a missile's light claims mid-flight — the whole point"
        );
        assert!(
            may_claim(true, None),
            "and on its very first frame, before the motion tracker has seen it at all"
        );
        assert!(
            !may_claim(false, Some(&moving)),
            "a walking bearer's torch still waits: the scan it would run is about to be stale"
        );
        assert!(
            may_claim(false, Some(&parked)),
            "a placed brazier settles and claims exactly as before"
        );
        assert!(
            !may_claim(false, None),
            "born moving is not a licence to claim"
        );
    }

    /// GOLDEN — MONKEY (spell light lane). The LANE half: a claim set with no bearer room behind it
    /// still earns the INTERIOR lane, which is the only lane whose entries the room shaders read.
    ///
    /// A missile is a free world entity — [`owner_lane`] finds no [`InteriorAnchor`] above its
    /// light and answers `(None, 0.0)` — so `want` is `None` and the bearer half of the lane
    /// verdict can never fire for it. Everything therefore rests on [`lane_worthy`], and on both of
    /// its conditions holding their meaning: a real room moves the light onto the interior lane,
    /// and a district-scale exterior shell (Stormwind's districts, Orgrimmar's valley) does NOT —
    /// a bolt flying over open cobbles must keep lighting them through `point_light_sum` rather
    /// than be handed to a lane that would not reach them.
    #[test]
    fn an_anchorless_spell_light_takes_the_interior_lane_from_its_claims_alone() {
        let set = |groups: &[u16], any_interior| CarriedClaimSet {
            instance: Entity::PLACEHOLDER,
            groups: groups.to_vec().into(),
            fades: Vec::new().into(),
            any_interior,
        };
        assert!(
            lane_worthy(&set(&[3, 4], true)),
            "a corridor and the room past its doorway: the walls must see the bolt"
        );
        assert!(
            !lane_worthy(&set(&[3, 4], false)),
            "an all-exterior set reaches no room lane, so moving onto it would only lose the pool"
        );
        assert!(
            !lane_worthy(&set(
                &[3, 4 | benilla_world::lighting::LIT_ROOM_EXT_DENY],
                true
            )),
            "one district-scale shell in the set and the light stays on the cobbles it lights"
        );
    }
}
