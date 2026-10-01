//! A placed model's point lights, particle emitters and ribbon trails, baked at its placement.

use benilla_assets::coords::{bevy_to_wow, wow_to_bevy};
use benilla_assets::{ModelEmitter, ModelLight, WmoModel};
use benilla_formats::room_claim::{self, PortalGraph};
use benilla_formats::{WmoBatchClass, WmoGroupInfo, WmoLight};
use bevy::prelude::*;

use crate::lighting::WorldPointLight;
use crate::particles;

/// Unit intensity in the `PointLight` convention, which the packer's `/(4π)` undoes to exactly
/// `diffuse_color × diffuse_intensity`; `wow_model.wgsl` applies the falloff `1/(0.7d + 0.03d²)`.
const POINT_LIGHT_INTENSITY: f32 = 4.0 * std::f32::consts::PI;
/// The ≤3-nearest selection radius (yd) around the lit unit's anchor, not a cutoff (a GL light has
/// none; att(48 yd) ≈ 0.01), wide enough to reach a hall's fixtures ~20 yd from its centre.
const POINT_LIGHT_RANGE: f32 = 48.0;
/// MONKEY (wmo exterior points): how close an authored MOLT fixture has to stand to a prop's flame
/// (yd) for that flame's SYNTHESISED light to be dropped as a duplicate. The vanilla convention is
/// a MOLT authored at the flame itself, so the real pairs sit well under a yard apart; 2.5 yd
/// covers a bracket-mounted torch whose fixture the artist put at the wall instead, without
/// reaching the next prop along a street (Stormwind's street lamps are ≥ 8 yd apart).
const MODD_SYNTH_DEDUPE: f32 = 2.5;

/// The [`WorldPointLight`] for an authored M2 or WMO MOLT light, `color` its linear RGB. As in the
/// reference it lights terrain, M2s and WMO walls, each taking its unit's ≤3 nearest, diffuse only.
pub fn point_light(color: [f32; 3], intensity_scale: f32) -> WorldPointLight {
    WorldPointLight {
        color,
        intensity: POINT_LIGHT_INTENSITY * intensity_scale.max(0.0),
        range: POINT_LIGHT_RANGE,
    }
}

fn spawn_point_light(
    commands: &mut Commands,
    world: Vec3,
    color: [f32; 3],
    intensity_scale: f32,
    // The light's rooms inside a building, `None` outdoors: a torch prop in a culled room is not
    // in the reference's scene that frame, so it lights nothing.
    room: Option<crate::wmo_portal::WmoGroupVis>,
    // MONKEY (fire GO lights): tag a SYNTHESISED source so the packer can ride the live
    // `fireLightGain` cvar over it without respawning the world
    // ([`crate::lighting::SyntheticFireLight`]).
    synthetic: bool,
    // MONKEY (flame flicker): the fire this source IS, or `None` for anything that merely shines
    // (a lamp behind glass, a neutral fill light). Decided once by
    // [`crate::lighting::flame_kind_for`] — one rule, every lane — and the phase SEED is taken here
    // from the light's own world position, so two candles on one table are never in step and a
    // fixture keeps its phase across a stream-out/stream-in.
    flame: Option<crate::lighting::FlameKind>,
    // MONKEY (interior attenuation): the source's AUTHORED attenuation end (yd) where the record
    // carries a usable one — i.e. a WMO MOLT fixture, and nothing else. `None` for every M2 light,
    // whose authored pair is a template default rather than a reach; the packer buckets those from
    // intensity instead ([`crate::lighting::m2_light_reach`], and see [`crate::lighting::LightReach`]
    // for the corpus evidence).
    reach: Option<f32>,
    // MONKEY (room gate): the rooms this source may LIGHT, over and above the ones that must be
    // visible for it to exist ([`crate::lighting::LightLitRooms`] carries the whole argument).
    // MONKEY (review fixes): MOLT and MODD claims keep their placement even with no MOLR/MODR.
    // MONKEY (soft portal claims): the component, built whole by [`lit_rooms`] — the group ids
    // and their per-claim fades travel together or not at all.
    lit_rooms: Option<crate::lighting::LightLitRooms>,
    out: &mut Vec<Entity>,
) {
    let mut e = commands.spawn((
        point_light(color, intensity_scale),
        Transform::from_translation(world),
    ));
    if let Some(room) = room {
        e.insert(crate::lighting::LightRooms(room));
    }
    if let Some(lit) = lit_rooms.filter(|l| !l.rooms.groups.is_empty()) {
        e.insert(lit);
    }
    if synthetic {
        e.insert(crate::lighting::SyntheticFireLight);
    }
    if let Some(kind) = flame {
        e.insert(crate::lighting::FlameFlicker::new(
            kind,
            crate::lighting::flicker_seed(world),
        ));
    }
    if let Some(reach) = reach.filter(|r| *r > 0.5) {
        e.insert(crate::lighting::LightReach(reach));
    }
    out.push(e.id());
}

/// Spawn a [`WorldPointLight`] for each casting `type==1` (point) **M2** light of a prop at `transform` (the
/// campfire/torch/brazier/lantern sources). Position is M2 model space at rest (bone matrix identity at
/// rest), placed via the prop's transform like the emitters — a placed prop never animates its light
/// bone, so the rest pose is exact here (the entity path, whose bones DO move, rides joints instead).
///
/// `room` is the prop's own rooms for a WMO prop (`None` for an ADT map doodad) — taken straight off
/// the [`emitter_fade`] built for this placement, so the prop's mesh, its flames, its streamers and
/// its glow are all admitted or refused by one value.
///
/// MONKEY (fire GO lights): `synthetic` says whether this lane accepts lights DERIVED from a
/// model's flame emitter ([`benilla_assets::ModelLight::synthetic`]). The ADT map-doodad lane does
/// — an outdoor campfire or wall torch has nothing else to light it.
///
/// MONKEY (wmo exterior points) / MONKEY (interior prop lights): the **WMO MODD prop lane now does
/// too — everywhere, indoors included.** It was first opened for EXTERIOR-class groups only (a
/// street, a courtyard, a porch), on the reasoning that a building's artists put a companion MOLT
/// fixture at each of its indoor torch flames, so a synthesised twin would double-light exactly the
/// rooms the interior lane is calibrated against. That assumption is FALSE for most buildings: the
/// `benilla-extract wmolamps` sweep counts 14,937 props across 506 roots whose model would
/// synthesise a light, and only 4,477 of them (30%) have an authored MOLT fixture within
/// [`MODD_SYNTH_DEDUPE`]. The Stormwind Bank's ceiling lanterns, the Crawford Winery's lanterns and
/// the Wizard's Sanctum's candelabras are all in the other 70%, and they lit nothing at all — the
/// reported bug. The DEDUPE, not the group class, is what stops the double-lighting where the
/// assumption did hold (the abbey pairs 41 of its 42 flames with a fixture; the inn 8 of 33).
///
/// `molt_world` is the same placement's authored MOLT fixtures in world space — that DUPLICATE
/// GUARD, applied to the synthesised light's own world position (the flame, not the prop origin,
/// which is why the check is here and not at resolve time).
///
/// MONKEY (portal claims): `claims` is the owning WMO's claim inputs (`None` for an ADT map
/// doodad). With it, a synthesised prop light gets the SAME room-claim set an authored MOLT fixture
/// gets — its own MODR rooms, unioned with the groups whose MOGI box holds the flame and the ones
/// one portal away — so the room gate admits a hanging lantern in the room it hangs in. Without it
/// the light would be packed with the prop's MODR rooms alone, which for a prop no group names is
/// no claim at all, i.e. UNGATED: one lantern lighting every room of the building through its
/// floors. That is why the claims travel with the lane rather than being optional dressing.
pub(super) fn spawn_lights_for(
    commands: &mut Commands,
    lights: &[ModelLight],
    transform: Transform,
    room: Option<&crate::wmo_portal::WmoGroupVis>,
    synthetic: bool,
    molt_world: &[Vec3],
    claims: Option<&PropClaims<'_>>,
    out: &mut Vec<Entity>,
) {
    // MONKEY (spell light): `l.spell.is_none()` keeps the PLACED lane exactly as it was. The spell
    // route (`fire_light::synthesize_spell_light`) lights effect models and fireworks, whose whole
    // premise — a light that exists for the length of one cast, budgeted and enveloped — belongs to
    // the entity lanes that reap it with its effect. A spell model placed as scenery must not enter
    // a table built for fixtures that stand still for minutes.
    for l in lights
        .iter()
        .filter(|l| (synthetic || !l.synthetic) && l.spell.is_none())
    {
        let def = &l.def;
        if !def.casts() {
            continue; // directional lights feed an ambient term; a static `0` visibility key is dark
        }
        let world = transform.transform_point(wow_to_bevy(def.position));
        if l.synthetic
            && molt_world
                .iter()
                .any(|p| p.distance_squared(world) < MODD_SYNTH_DEDUPE * MODD_SYNTH_DEDUPE)
        {
            continue; // an authored MOLT fixture already stands at this flame
        }
        spawn_point_light(
            commands,
            world,
            def.diffuse_color,
            def.diffuse_intensity,
            room.cloned(),
            l.synthetic,
            // MONKEY (flame flicker): the flame route burns whatever colour it is; the lamp route
            // never does; an authored block burns iff it was authored warm.
            crate::lighting::flame_kind_for(
                l.flame,
                l.synthetic,
                def.diffuse_color,
                def.diffuse_intensity,
            ),
            None, // an M2 light authors no usable reach — the packer buckets it
            // MONKEY (portal claims): the same claim set a MOLT fixture gets, measured at the
            // FLAME. An M2 source's reach is the intensity bucket (`room_claim::m2_light_reach`,
            // the packer's own ladder), scaled by the default `interiorAttenScale` — the portal
            // hop has to be decided at spawn, and the live cvar is not knowable here.
            claims.map(|c| {
                c.claim_set(
                    world,
                    room_claim::claim_reach(room_claim::m2_light_reach(def.diffuse_intensity)),
                    room.map_or(&[][..], |r| &r.groups[..]),
                )
            }),
            out,
        );
    }
}

/// The claim inputs of one placed WMO, borrowed straight off the loaded asset — no clone, because
/// this is rebuilt every spawn wave while a city's props are still landing.
pub(super) struct PropClaims<'a> {
    // MONKEY (review fixes): lighting identity must survive an empty MODR relation.
    instance: Entity,
    groups: &'a [WmoGroupInfo],
    portals: PortalGraph<'a>,
    /// World (Bevy) → this placement's WMO model space, so a prop light's WORLD position (the
    /// flame, resolved through the prop's own transform) can be asked the model-space question the
    /// MOGI boxes and MOPT polygons answer.
    to_model: bevy::math::Affine3A,
    /// MONKEY (soft portal claims): and back out again — a claim's fade centre is a PORTAL, which
    /// the rule answers in model space and the shader can only use in world space.
    to_world: Transform,
}

impl<'a> PropClaims<'a> {
    /// The claim inputs of `model`, with `slices` the caller-owned per-group portal-ref ranges
    /// ([`portal_slices`] — owned outside because [`PortalGraph`] borrows them).
    pub(super) fn new(
        model: &'a WmoModel,
        slices: &'a [(u16, u16)],
        transform: Transform,
        instance: Entity,
    ) -> Self {
        Self {
            instance,
            groups: &model.group_bounds,
            portals: PortalGraph {
                vertices: &model.portal_vertices,
                infos: &model.portal_infos,
                refs: &model.portal_refs,
                slices,
            },
            to_model: transform.compute_affine().inverse(),
            to_world: transform,
        }
    }

    /// The claim set of a source at `world` (Bevy) with effective radius `reach`, in the packed
    /// [`crate::lighting::LightLitRooms`] form.
    fn claim_set(&self, world: Vec3, reach: f32, molr: &[u16]) -> crate::lighting::LightLitRooms {
        let pos = bevy_to_wow(self.to_model.transform_point3(world));
        lit_rooms(
            self.instance,
            &room_claim::room_claims(self.groups, self.portals, pos, reach, molr),
            self.to_world,
        )
    }
}

/// MONKEY (GO room claims): the claim set of ONE **carried** light — a server-placed brazier
/// GameObject, a campfire, an NPC's torch — standing inside a placed WMO, in the packed
/// [`crate::lighting::LightLitRooms`] form its caller applies.
///
/// The same rule the placed lanes above run ([`benilla_formats::room_claim::room_claims`]), reached
/// from the OTHER side. A MOLT fixture and a MODD prop are authored INSIDE the building, so the
/// spawner already holds the model, the placement transform and the instance entity when it makes
/// the light. A GameObject is not: it walks (or is `.gobject add`-ed) into a building the WMO's own
/// MODR/MOLR tables know nothing about, and until now the only room it could ever claim was the
/// single one the interior classifier's down-ray put its BEARER in
/// (`benilla_app::entities::carried_light`'s [`crate::lighting::LightRooms`]). One room is one
/// group: the brazier lit the room it stood in and stopped dead at every doorway, and standing in
/// an EXTERIOR-class group (Orgrimmar's canyon "rooms", MOGP `& 0x48`) it named no room at all.
///
/// So the app asks HERE instead, with the light's own WORLD position, and gets the identical
/// containment/MOLR/portal-hop set an authored fixture gets. `molr` is empty by construction — no
/// group's MOLR can name a light that is not in the file — which is exactly the case the
/// containment source was added for (`room_claim`'s module doc: the dense supplement to the sparse
/// relation).
///
/// `None` when no group's authored MOGI box holds the light. That is not merely "the set is empty":
/// containment is the ONLY base source available here, and with no base claim the portal hop has
/// nothing to seed from, so the answer could not be anything else — returning `None` lets the
/// caller move on to the next resident placement with one test instead of a wasted allocation.
pub fn carried_light_claims(
    model: &WmoModel,
    instance: Entity,
    // The placement's Bevy model→world (`WmoPortalInstance::world_from_local`). Inverted here, per
    // call: a carried light's claims are rebuilt only on the gates its caller change-gates on
    // (settle / 0.5 yd drift / residency generation / bearer room), never per frame, so caching the
    // inverse would be state kept for an operation that does not repeat.
    world_from_local: bevy::math::Affine3A,
    world: Vec3,
    reach: f32,
) -> Option<CarriedClaimSet> {
    let pos = bevy_to_wow(world_from_local.inverse().transform_point3(world));
    let slices = portal_slices(model);
    let claims = room_claim::room_claims(
        &model.group_bounds,
        PortalGraph {
            vertices: &model.portal_vertices,
            infos: &model.portal_infos,
            refs: &model.portal_refs,
            slices: &slices,
        },
        pos,
        reach,
        &[],
    );
    if !claims
        .iter()
        .any(|c| c.how == room_claim::ClaimHow::Contains)
    {
        return None; // not in this building at all
    }
    Some(CarriedClaimSet {
        instance,
        any_interior: claims.iter().any(|c| c.interior),
        groups: pack_claims(&claims),
        // MONKEY (soft portal claims): the placement's own transform, rebuilt from the affine the
        // caller already holds — the fade centres are portals of THIS building and must reach the
        // shader in world space (see [`crate::lighting::ClaimFade`]).
        fades: pack_fades(&claims, Transform::from_matrix(world_from_local.into())),
    })
}

/// MONKEY (GO room claims): what [`carried_light_claims`] hands back — the packed claim list plus
/// the two facts the caller's LANE decision needs, which the packed ids alone cannot answer.
#[derive(Clone)]
pub struct CarriedClaimSet {
    /// The claiming placement's [`crate::wmo_portal::WmoPortalInstance`] entity — the head word of
    /// the GPU claim record, and the identity a claim is worthless without.
    pub instance: Entity,
    /// The packed group ids (`group | `[`crate::lighting::LIT_ROOM_EXT_DENY`]) — the
    /// [`crate::lighting::LightLitRooms`] payload. Exposed raw so the caller can change-gate its
    /// component write against the set it applied last time (a `LightLitRooms` cannot be read back:
    /// its field is crate-private, and re-inserting an identical one would churn the archetype).
    pub groups: std::sync::Arc<[u16]>,
    /// MONKEY (soft portal claims): index-parallel with `groups` — each claim's doorway + fade
    /// (world space). Exposed for the same change-gate reason `groups` is: a light that drifts
    /// without changing WHICH rooms it claims still changes HOW FAR past each doorway it reaches,
    /// and re-applying an identical component would churn the archetype for nothing.
    pub fades: std::sync::Arc<[crate::lighting::ClaimFade]>,
    /// Does the set name at least one INTERIOR-class group? The lane question: claims are read by
    /// `interior_room_light` alone, and that function is only ever reached by a light the packer
    /// filed on the INTERIOR lane (`static_gx.wgsl` skips every entry whose colour-row `.w < 0.5`).
    /// So a light that claims a real room has to be moved onto that lane for its claims to mean
    /// anything at all — see the caller for why an all-exterior set is deliberately left alone.
    pub any_interior: bool,
}

impl CarriedClaimSet {
    /// The component form.
    pub fn component(&self) -> crate::lighting::LightLitRooms {
        crate::lighting::LightLitRooms {
            rooms: crate::wmo_portal::WmoGroupVis {
                instance: self.instance,
                groups: self.groups.clone(),
            },
            fades: self.fades.clone(),
        }
    }

    /// Is EVERY claim exterior-lane eligible — i.e. does the strict exterior-batch room term
    /// (`static_gx.wgsl`, `interior_room_admits(strict = true)`) admit this light on all of them?
    ///
    /// The caller's safety condition for moving a light onto the interior lane. A
    /// [`crate::lighting::LIT_ROOM_EXT_DENY`] claim is a district-scale EXTERIOR shell, whose
    /// surfaces the strict term refuses; moving such a light off the exterior lane would take its
    /// pool off the very cobbles it stands on and give nothing back that reached them.
    pub fn all_ext_ok(&self) -> bool {
        self.groups
            .iter()
            .all(|g| g & crate::lighting::LIT_ROOM_EXT_DENY == 0)
    }
}

/// The per-group `(portal_ref_start, portal_ref_count)` ranges of a WMO's MOPR list — the shape
/// [`PortalGraph`] wants, which the loaded asset carries spread across [`WmoGroupNav`]
/// (`benilla_assets`). One small Vec per placement per spawn wave; the graph itself is borrowed.
pub(super) fn portal_slices(model: &WmoModel) -> Vec<(u16, u16)> {
    model
        .group_nav
        .iter()
        .map(|n| (n.ref_start, n.ref_count))
        .collect()
}

/// MONKEY (portal claims): fold each claim's EXTERIOR-LANE eligibility into its group id — bit 15,
/// which no group index can reach (the record's room key is 12 bits). A district-scale exterior
/// shell must stay in the set (dropping it would leave some fixtures with no claims at all, and the
/// packer's no-claims arm fails OPEN) while being invisible to the exterior batch lane, and one
/// `Arc<[u16]>` carrying both facts is cheaper than a second parallel component.
/// **Decoded by `RoomClaim::write` → `static_gx.wgsl`'s `CLAIM_EXT_OK`.**
fn pack_claims(claims: &[room_claim::Claim]) -> std::sync::Arc<[u16]> {
    claims
        .iter()
        .map(|c| {
            c.group
                | if c.ext_ok {
                    0
                } else {
                    crate::lighting::LIT_ROOM_EXT_DENY
                }
        })
        .collect()
}

/// MONKEY (soft portal claims): the per-claim FADE array, index-parallel with [`pack_claims`].
///
/// The rule answers in the WMO's own model space (WoW axes) because that is where the MOGI boxes
/// and MOPT polygons live; the shader has only a Bevy WORLD fragment position. So the doorway is
/// converted here, once per claim at spawn, rather than by handing the shader a matrix to invert
/// per fragment. A HARD claim (containment/MOLR, `fade_radius == 0`) converts to `ClaimFade`'s
/// default and reads as "weight 1 everywhere" — the binary gate, preserved exactly where it was
/// right.
///
/// The SLACK is a length and the transform is a rigid placement plus a uniform scale, so it scales
/// with `transform.scale.x` rather than riding the full affine: a WMO placement's scale is uniform
/// by construction (ADT/WDT store one factor), and taking the x component keeps the doorway's
/// tolerance in the same units as the world distance the shader measures against it.
fn pack_fades(
    claims: &[room_claim::Claim],
    transform: Transform,
) -> std::sync::Arc<[crate::lighting::ClaimFade]> {
    let scale = transform.scale.x.abs().max(1e-4);
    claims
        .iter()
        .map(|c| {
            if c.fade_radius <= 0.0 {
                return crate::lighting::ClaimFade::default();
            }
            crate::lighting::ClaimFade {
                center: transform.transform_point(wow_to_bevy(c.fade_center)),
                slack: c.fade_slack * scale,
                radius: c.fade_radius * scale,
                // Dimensionless — the placement's scale does not touch it.
                entry: c.fade_entry,
            }
        })
        .collect()
}

/// MONKEY (soft portal claims): the whole [`crate::lighting::LightLitRooms`] component of one
/// claim set — ids and fades built in one place so the two arrays cannot be produced at different
/// lengths (the shader indexes them at a fixed stride: a short fade array would gate a claim by
/// another claim's doorway).
fn lit_rooms(
    instance: Entity,
    claims: &[room_claim::Claim],
    transform: Transform,
) -> crate::lighting::LightLitRooms {
    crate::lighting::LightLitRooms {
        rooms: crate::wmo_portal::WmoGroupVis {
            instance,
            groups: pack_claims(claims),
        },
        fades: pack_fades(claims, transform),
    }
}

/// Spawn a [`WorldPointLight`] for each omni (`type==0`) **WMO MOLT** light at the building's `transform` (the
/// interior fixture sources — forge fire, inn fireplaces, chapel candles/chandeliers). These radiate onto
/// the NPCs/doodads near each fixture and onto the building's own walls/floor (decision 0273). Position
/// is WMO model space. The MOLT curve is the same fixed WoW falloff as the M2 path — byte-verified
/// (wow-re wmo-molt-runtime: the disk attenStart/End do not shape the GL falloff) and confirmed live in
/// the reference GL trace (committed c/l/q = 0/0.7/0.03, diffuse = colour × intensity, ambient 0).
///
/// A fixture belongs to the rooms whose **MOLR** names it (`group_light_refs`, per absolute group
/// index — the same list an interior GameObject's committed light folds), so a dungeon's brazier
/// stops lighting the hillside above it the moment the flood leaves its room. The refs are inverted
/// here, once per placement. MONKEY (review fixes): a MOLT light no group names keeps no VISIBILITY
/// rooms (the prop path's empty-MODR fail-open PVS arm); its lighting claims below still gate which
/// rooms it may illuminate, keyed to the placement independently of that sparse MOLR relation.
///
/// MONKEY (room gate) / MONKEY (portal claims): the rooms a fixture may LIGHT are a wider set than
/// that, and it is built by [`benilla_formats::room_claims`] — the one rule, shared with the
/// `benilla-extract wmolights`/`wmolamps` audits so what the instrument prints is what the shader
/// enforces. Containment (the interior groups whose authored MOGI box holds the fixture) is the
/// dense supplement the sparse MOLR relation needs — the Goldshire inn authors a MOLR on 2 of its
/// 12 groups and NSabbey leaves seven rooms unnamed, so a MOLR-only gate blacks both buildings out;
/// the PORTAL hop is what carries a fixture's light through an open doorway into the next room,
/// without which a continuous floor changes brightness in a straight line at the group boundary.
/// See that module for the whole argument, including why a district-scale exterior shell is kept in
/// the set but barred from the exterior lane.
pub(super) fn spawn_wmo_lights_for(
    commands: &mut Commands,
    lights: &[WmoLight],
    group_light_refs: &[Vec<u16>],
    // MONKEY (room gate) / MONKEY (portal claims): the root's group table + portal graph, for
    // [`benilla_formats::room_claims`]. A MOLT record's position is already in the space they are
    // in, so this lane needs no inverse transform (the prop lane, whose flame is resolved in world
    // space, does — see [`PropClaims`]).
    groups: &[WmoGroupInfo],
    portals: PortalGraph<'_>,
    instance: Option<Entity>,
    transform: Transform,
    out: &mut Vec<Entity>,
) {
    for (i, l) in lights.iter().enumerate() {
        if !l.is_omni() {
            continue; // spot/directional/ambient MOLT types are not point sources
        }
        let i = i as u16;
        let molr: std::sync::Arc<[u16]> = group_light_refs
            .iter()
            .enumerate()
            .filter(|(_, refs)| refs.contains(&i))
            .map(|(g, _)| g as u16)
            .collect();
        let world = transform.transform_point(wow_to_bevy(l.position));
        spawn_point_light(
            commands,
            world,
            l.color,
            l.intensity,
            instance
                .filter(|_| !molr.is_empty())
                .map(|instance| crate::wmo_portal::WmoGroupVis {
                    instance,
                    groups: molr.clone(),
                }),
            false, // a MOLT fixture is authored, never synthesised
            // MONKEY (flame flicker): an authored fixture burns iff it was authored WARM. The MOLT
            // corpus is overwhelmingly candles and wall torches at that hue; what the warmth test
            // holds back is the neutral-white fill lights artists park in halls (which must stay
            // dead steady) and the cold magic sources.
            crate::lighting::flame_kind_for(false, false, l.color, l.intensity),
            // MONKEY (interior attenuation): the fixture's authored `attenuation_end` (`+0x2c`) —
            // the reference's own fold window (wow-re `trace-forensics-abbey-interior-d3d` §4), and
            // the number that turns a 10-candle inn from a uniform warm wash into ten pools. The
            // 48 yd `PointLight::range` beside it stays what it always was: candidacy, not reach.
            Some(l.attenuation_end),
            // MONKEY (room gate) / MONKEY (portal claims): the rooms this fixture may LIGHT — its
            // MOLR rooms, the ones whose authored box holds it, and the ones one open doorway away
            // within its reach, in that priority. Dropped when the placement has no portal instance
            // — without one there is nothing for a claim to be keyed to, and the packer leaves the
            // fixture ungated.
            instance.map(|instance| {
                lit_rooms(
                    instance,
                    &room_claim::room_claims(
                        groups,
                        portals,
                        l.position,
                        room_claim::claim_reach(l.attenuation_end),
                        &molr,
                    ),
                    transform,
                )
            }),
            out,
        );
    }
}

/// MONKEY (daylight fixtures): spawn one interior-lane point light per EXTERIOR-FACING OPENING of
/// this placement — the doorways and windows whose sunlit threshold the room law otherwise meets
/// with a hard line (see `crate::lighting`'s `daylight` module for the whole argument, the two
/// seeds and the corpus measurement behind them).
///
/// It is deliberately the SAME entity shape a MOLT fixture gets — [`crate::lighting::LightRooms`]
/// for the visibility gate, the `room_claims` claim set through [`lit_rooms`],
/// [`crate::lighting::LightReach`] for the packer's window, an interior `LightLane` — so the
/// packer, the claim table, the shader and the std430 layout all need no knowledge that this source
/// is the sun rather than a candle. What it does NOT get is a `FlameFlicker` (daylight is steady)
/// or `SyntheticFireLight` (`fireLightGain` must not be able to dim the sun).
///
/// The `PointLight` itself is NOT spawned here: `update_daylight_fixtures` inserts it on the first
/// frame it is day, with the sun-calibrated colour and intensity. A placeholder would pack one
/// uncalibrated frame — a flash of the wrong brightness in every doorway of a building that just
/// streamed in — and a placement streaming in at night would spawn a light only to have it removed.
///
/// `instance` absent (a placement with no portal instance) means there is nothing to key a room
/// claim to, and an unclaimed interior fixture is packed UNGATED, which fails OPEN — one doorway
/// lighting every room of the building through its walls. Such a placement gets no daylight at all,
/// exactly as the MOLT lane drops its own claims there.
pub(super) fn spawn_daylight_fixtures_for(
    commands: &mut Commands,
    model: &WmoModel,
    portals: PortalGraph<'_>,
    instance: Option<Entity>,
    transform: Transform,
    out: &mut Vec<Entity>,
) {
    let Some(instance) = instance else {
        return;
    };
    // The APERTURE seed's input: every render batch of the root as `(group, EXT-class?, points)`,
    // borrowed straight off the loaded asset's two parallel arrays. The selection rule consumes it
    // lazily, so a root whose groups are all exterior walks the batches once and allocates nothing.
    let batches = model
        .submeshes
        .iter()
        .zip(model.submesh_group.iter())
        .map(|(s, g)| {
            (
                *g,
                matches!(s.wmo_batch, Some(WmoBatchClass::Ext)),
                &s.geometry.positions[..],
            )
        });
    // MONKEY (portal bleed): one selection call for both lanes -- the exterior-facing openings that
    // get the SUN and the interior<->interior doorways that get the ROOM NEXT DOOR -- ranked by area
    // against each other, so the eight slots go to this building's biggest openings whichever kind
    // they are.
    let (day, bleed) = crate::lighting::placement_openings(&model.group_bounds, portals, batches);
    for seed in day {
        let fixture = seed.fixture(instance);
        let claims = crate::lighting::daylight_claims(&model.group_bounds, portals, &seed);
        let e = commands
            .spawn((
                Transform::from_translation(transform.transform_point(wow_to_bevy(seed.pos))),
                fixture,
                crate::lighting::daylight_rooms(instance, seed.group),
                lit_rooms(instance, &claims, transform),
                crate::lighting::LightReach(fixture.reach),
                crate::lighting::daylight_lane(),
            ))
            .id();
        out.push(e);
    }
    // MONKEY (portal bleed): the doorway fixtures. Deliberately the SAME bundle as above plus one
    // component -- a bleed IS a daylight fixture as far as the packer, the claim table, the shader
    // and `torch_shadow` are concerned, and `BleedFixture` only names the two rooms it joins and the
    // plane it is calibrated at (in WORLD space: the per-frame evaluation has world positions and no
    // placement matrix). Its visibility gate takes BOTH rooms, not one. The `PointLight` is again NOT
    // spawned here -- a doorway between two unlit rooms never gets one at all.
    for seed in bleed {
        let fixture = seed.fixture(instance);
        let claims = seed.claims(&model.group_bounds, portals);
        let world = transform.transform_point(wow_to_bevy(seed.pos));
        let e = commands
            .spawn((
                Transform::from_translation(world),
                fixture,
                crate::lighting::BleedFixture {
                    sides: seed.sides,
                    probe: world,
                },
                seed.rooms(instance),
                lit_rooms(instance, &claims, transform),
                crate::lighting::LightReach(fixture.reach),
                crate::lighting::daylight_lane(),
            ))
            .id();
        out.push(e);
    }
}

/// The **one draw-set gate a placed model's emitters share** — the quad clouds and the ribbon
/// trails alike, built once per placement so the two families can never be handed different
/// answers about whether their model is in the frame's scene.
///
/// `fade` is the owner's bounding sphere as `(radius, MODEL-space centre)`; the world centre is the
/// placement applied to it — the same point the doodad's own visibility gate measures to.
/// `instance` is the WMO placement this model's doodad set belongs to (`None` for an ADT map
/// doodad), carrying the exterior-window exemption for a prop of the building the camera is
/// standing in (0786). `groups` are the rooms of that placement which name this prop — `None` or
/// empty ⇒ **no room gate**, which is the mesh path's own `d.groups.is_empty()` arm, spelled once
/// here so the two cannot diverge (0689/1289). It is borrowed rather than owned because the shared
/// `Arc` is exactly what wants cloning, and an ADT map doodad — the tens-of-thousands case — must
/// not allocate an empty one per placement to say "I have no rooms".
pub(super) fn emitter_fade(
    transform: Transform,
    fade: (f32, Vec3),
    instance: Option<Entity>,
    groups: Option<&std::sync::Arc<[u16]>>,
) -> particles::EmitterFade {
    let room = instance.zip(groups.filter(|g| !g.is_empty()));
    particles::EmitterFade {
        instance,
        room: room.map(|(instance, groups)| crate::wmo_portal::WmoGroupVis {
            instance,
            groups: groups.clone(),
        }),
        ..particles::EmitterFade::sphere(fade.0, transform.transform_point(fade.1))
    }
}

/// Spawns a model's emitters into `out`. With an anim `host` each rides its bone's anchor, which
/// holds a bone that never animates at its rest pivot, so riding is exact for every emitter.
pub(super) fn spawn_emitters_for(
    commands: &mut Commands,
    emitters: &[ModelEmitter],
    transform: Transform,
    host: Option<&super::assemble::PlacementHost>,
    fade: &particles::EmitterFade,
    out: &mut Vec<Entity>,
) {
    let arm = host.and_then(|h| h.arm);
    for em in emitters {
        let owner = host
            .and_then(|h| h.anchor(em.def.bone))
            .map(|a| (a, em.bone_pivot));
        if let Some(e) = particles::spawn_emitter(
            commands,
            em,
            transform,
            particles::EmitterFrames {
                owner,
                anchor: None, // anchor at the placement: an animated bone never drags the cloud
                // An animated doodad's rig goes only when its placement unloads, the model's end,
                // so the pool is freed rather than left draining at the old spot.
                on_owner_loss: particles::OwnerLoss::Free,
                ..default()
            },
            // A placed doodad re-rolls its variation every play window, so its emitters read the
            // slot and clip time off the host's live player, as the reference's animate kernel
            // (`0x714260`) does for every model.
            match arm {
                Some(root) => particles::EmitClock::Host(root),
                None => particles::EmitClock::Pinned,
            },
        ) {
            commands.entity(e).insert(fade.clone());
            out.push(e);
        }
    }
}

/// Spawns a model's ribbon trails, each riding its host bone's anchor or, on a static placement,
/// `fallback_owner`; a trail despawns with its owner, so none joins a despawn list.
pub(super) fn spawn_ribbons_for(
    commands: &mut Commands,
    ribbons: &[benilla_assets::ModelRibbon],
    transform: Transform,
    host: Option<&super::assemble::PlacementHost>,
    fallback_owner: Option<Entity>,
    // The emitters' gate, carried because a joint or submesh owner states no fade sphere or rooms.
    fade: &particles::EmitterFade,
) {
    let arm = host.and_then(|h| h.arm);
    for rb in ribbons {
        let (owner, use_pivot) = match host.and_then(|h| h.anchor(rb.def.bone)) {
            Some(a) => (a, true),
            None => match fallback_owner {
                Some(e) => (e, false),
                None => continue, // nothing to ride (an entity-less placement)
            },
        };
        // The `+0xc0` enable gate reads the live sequence, as the emitters do; an unanimated
        // placement has no clock and rests in `Stand`.
        crate::ribbons::spawn_ribbon(
            commands,
            rb,
            owner,
            use_pivot,
            transform.scale.max_element(),
            arm.map_or(
                crate::ribbons::RibbonSeq::Fixed(0),
                crate::ribbons::RibbonSeq::Host,
            ),
            // No model-alpha source: a placed prop / effect instance is always drawn.
            None,
            Some(fade.clone()),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{carried_light_claims, pack_claims};
    use benilla_assets::WmoModel;
    use benilla_formats::room_claim::{room_claims, PortalGraph};
    use benilla_formats::WmoGroupInfo;
    use bevy::prelude::*;

    fn g(interior: bool, min: [f32; 3], max: [f32; 3]) -> WmoGroupInfo {
        WmoGroupInfo {
            interior,
            show_skybox: false,
            bbox_min: min,
            bbox_max: max,
        }
    }

    /// MONKEY (room gate): the Goldshire inn's real numbers — the measurement the whole gate rests
    /// on. Its ground-floor fixture L0 stands at model-space `(-9.13, -1.13, 4.48)`; the basement
    /// (`room03`, `z -6.76 .. 0.45`) does not contain it and so takes no claim, which is exactly
    /// the reported leak (a basement lit bright through its own floor). The `kitchen` group, whose
    /// authored box does contain it, gets it — first, because it is the tightest box that does.
    ///
    /// MONKEY (portal claims): the building's `upstairs` SHELL is claimed too now, and that is the
    /// change. It is EXTERIOR-flagged, so it can never call the interior lane; the claim exists for
    /// the exterior batch lane, which is what stops that shell reading night-sky green a step away
    /// from a candle-lit room. At `57.7 x 32.2 yd` it is building-scale, so it is exterior-lane
    /// eligible (no `LIT_ROOM_EXT_DENY` bit) — a city's district shell would not be.
    #[test]
    fn a_fixture_claims_the_rooms_whose_authored_box_holds_it() {
        let groups = [
            g(true, [14.11, -4.86, 0.40], [20.47, -0.38, 4.37]), // g0, far corner room
            g(true, [-17.07, -11.08, -0.21], [14.39, 15.54, 16.32]), // g1 (inn "kitchen")
            g(true, [-36.75, -8.67, -6.76], [-17.01, 12.68, 0.45]), // g2 (inn "room03", basement)
            g(false, [-37.77, -13.53, -0.18], [19.91, 18.68, 23.28]), // g3 (inn "upstairs" SHELL)
        ];
        let claim = |p: [f32; 3]| room_claims(&groups, PortalGraph::default(), p, 11.2, &[]);
        assert_eq!(
            &pack_claims(&claim([-9.13, -1.13, 4.48]))[..],
            &[1, 3],
            "the room it stands in first, then the shell around it — and NOT the basement"
        );
        assert!(
            claim([-31.84, 1.48, -3.0]).iter().any(|c| c.group == 2),
            "a fixture actually IN the basement does claim it"
        );
        assert!(
            claim([500.0, 0.0, 0.0]).is_empty(),
            "a fixture outside every box claims nothing (the packer then leaves it ungated)"
        );
    }

    /// MONKEY (portal claims): a DISTRICT-scale exterior shell (Stormwind's, 200-660 yd) still
    /// gates the fixture — dropping it would leave some fixtures claimless, and no claims fails
    /// OPEN — but carries the deny bit so the exterior lane cannot light a whole city block from
    /// one tavern candle.
    #[test]
    fn a_district_shell_is_claimed_but_barred_from_the_exterior_lane() {
        let groups = [g(
            false,
            [-510.81, -82.83, -5.78],
            [-315.12, 101.35, 126.18],
        )];
        let packed = pack_claims(&room_claims(
            &groups,
            PortalGraph::default(),
            [-400.0, 0.0, 10.0],
            11.2,
            &[],
        ));
        assert_eq!(&packed[..], &[crate::lighting::LIT_ROOM_EXT_DENY]);
    }

    /// GOLDEN — MONKEY (GO room claims). A CARRIED light (a brazier GameObject) asks the placed
    /// lanes' claim rule from the outside, at its own WORLD position, and gets the identical
    /// answer an authored MOLT fixture standing there would: the tightest containing room first,
    /// then the shell around it. It also reports the two facts the caller's LANE decision needs and
    /// the packed ids cannot carry — whether any claim is on an INTERIOR-class group, and whether
    /// every claim is exterior-lane eligible.
    ///
    /// `None` outside every authored box is the contract the caller's placement scan rides: with no
    /// containment there is no base claim, so the portal hop has nothing to seed from and the
    /// answer could not have been anything else. One test, and the scan moves to the next building.
    #[test]
    fn a_carried_light_claims_the_building_it_stands_in() {
        // The Goldshire inn again (the numbers the whole gate was measured on), placed 100 yd east
        // so the world→model inverse is doing real work rather than being identity.
        let mut model = WmoModel {
            group_bounds: vec![
                g(true, [-17.07, -11.08, -0.21], [14.39, 15.54, 16.32]), // g0 "kitchen"
                g(true, [-36.75, -8.67, -6.76], [-17.01, 12.68, 0.45]),  // g1 "room03", basement
                g(false, [-37.77, -13.53, -0.18], [19.91, 18.68, 23.28]), // g2 "upstairs" SHELL
            ],
            ..default()
        };
        let placement = Transform::from_xyz(100.0, 0.0, 0.0);
        let world_from_local = placement.compute_affine();
        // Model-space (-9.13, -1.13, 4.48) — the inn's real ground-floor fixture L0.
        let at =
            placement.transform_point(benilla_assets::coords::wow_to_bevy([-9.13, -1.13, 4.48]));
        let inst = Entity::from_raw_u32(12).unwrap();

        let set = carried_light_claims(&model, inst, world_from_local, at, 11.2)
            .expect("the inn's authored boxes hold it");
        assert_eq!(set.instance, inst);
        assert_eq!(
            &set.groups[..],
            &[0, 2],
            "the room it stands in, then the building shell — never the basement below it"
        );
        assert!(set.any_interior, "g0 is interior-class: the lane may move");
        assert!(
            set.all_ext_ok(),
            "the inn's shell is building-scale, so the strict exterior term still admits it"
        );

        // A light 500 yd away is in no building: the caller tries the next placement.
        assert!(
            carried_light_claims(&model, inst, world_from_local, at + Vec3::X * 500.0, 11.2)
                .is_none()
        );

        // A DISTRICT-scale exterior shell (a city's canyon) still gates the light — dropping the
        // claim would pack it UNGATED, which fails open — but it is barred from the exterior lane,
        // so `all_ext_ok` is false and the caller leaves the light on the EXTERIOR lane rather than
        // taking its pool off the very cobbles it stands on.
        model.group_bounds = vec![g(
            false,
            [-510.81, -82.83, -5.78],
            [-315.12, 101.35, 126.18],
        )];
        let canyon =
            placement.transform_point(benilla_assets::coords::wow_to_bevy([-400.0, 0.0, 10.0]));
        let set = carried_light_claims(&model, inst, world_from_local, canyon, 11.2)
            .expect("the district shell contains it");
        assert!(!set.any_interior);
        assert!(!set.all_ext_ok());
    }
}
