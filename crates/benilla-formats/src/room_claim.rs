//! MONKEY (portal claims): **which rooms one light fixture may light** — the single rule, shared by
//! the runtime spawner (`benilla_world::terrain_stream::spawn`'s MOLT and MODD-prop lanes, which
//! turn it into the per-light claim list the room gate packs) and the offline audit
//! (`benilla-extract wmolights` / `wmolamps`, which is the only instrument that can measure it
//! across the shipped corpus). One rule, one implementation — the `fire_light` precedent.
//!
//! The gate itself lives in `static_gx.wgsl`: a WMO surface of group `G` takes an interior fixture
//! only if that fixture's claim set contains `G`. The set has three sources, in this priority:
//!
//! 1. **CONTAINMENT** — the interior groups whose authored MOGI bounding box holds the fixture.
//!    The dense supplement to the sparse MOLR relation (the Goldshire inn authors a MOLR on 2 of
//!    its 12 groups; Northshire abbey leaves 7 of 14 unnamed), ordered TIGHTEST BOX FIRST so the
//!    room a fixture actually stands in outranks the building-spanning shell that also contains it.
//! 2. **MOLR** — the groups whose authored light-ref list names the fixture. Authored for the
//!    reference's own purpose (register a GL light while drawing a visible group's doodads), which
//!    is why it cannot carry the gate alone, but it is the artist speaking and it is never wrong.
//!    2b. **SPLIT-FLOOR SIBLINGS** (MONKEY (split-floor claims)) — the group across a portal too
//!    LARGE to be a doorway ([`SPLIT_PORTAL_MIN_AREA`]): one room the artist cut in two for
//!    rendering. It joins the base at weight 1, because there is no threshold between the halves
//!    to fade across — see [`ClaimHow::Split`].
//! 3. **PORTAL-ADJACENT** — the groups up to [`ROOM_CLAIM_HOPS`] portal hops from a claimed group,
//!    when the doorways lie within the fixture's effective radius. This is the "light crosses an
//!    open doorway" rule: without it a fixture stops dead at its room's edge and a continuous floor
//!    changes brightness in a straight line at the group boundary, which is exactly the seam the
//!    owner reported. Nearest portal first, so a doorway three rooms away never takes the slot a
//!    doorway in this wall wants.
//!
//! MONKEY (soft portal claims): a portal claim is **not binary**. It carries the doorway it came
//! through ([`Claim::fade_center`]/[`Claim::fade_slack`]) and the reach left over on the far side
//! ([`Claim::fade_radius`]), and the shader turns those into a WEIGHT in `[0, 1]` — 1 in the
//! doorway, 0 where the reach runs out. That is what actually closes the seam: a binary claim
//! merely moved the step from "this room" to "the last room the hop reached", and admitting a
//! fixture on one side of a group plane while refusing it on the other is a step wherever the
//! FLOOR crosses that plane, which is precisely what a doorway is. Containment and MOLR claims
//! stay hard (weight 1) — a fixture standing in a room lights all of it.
//!
//! Sources 1+2 are the **base**: they must never be truncated (a dropped containment claim is a
//! black room, and the packer's overflow arm deliberately fails OPEN for that reason). Source 3 only
//! ever fills slots the base left free — a dropped portal claim is one doorway that does not carry
//! light, which is the behaviour we already had.

use crate::{WmoGroupInfo, WmoPortalInfo, WmoPortalRef};

/// Claims one fixture can carry. **Must equal `benilla_world::lighting::ROOM_CLAIM_MAX` and
/// `ROOM_CLAIM_MAX` in `static_gx.wgsl`** — past it the packer gives up and packs the fixture
/// UNGATED, so a claim list built here longer than this is not a wider gate, it is no gate at all.
pub const ROOM_CLAIM_MAX: usize = 6;

/// The portal graph of ONE WMO root, as the claim rule reads it: the MOPV vertex pool, the MOPT
/// portal records (each a slice of that pool plus its plane), the MOPR edges, and the per-group
/// slice of those edges (`MOGP +0x24/+0x26`, i.e. [`crate::WmoGroupHeader::portal_ref_start`] /
/// `portal_ref_count`) indexed by ABSOLUTE group index. All in WMO model space, WoW axes — the same
/// space [`WmoGroupInfo`]'s boxes and a MOLT record's position are in, which is why the whole rule
/// can run before anything is placed in the world.
#[derive(Clone, Copy, Default)]
pub struct PortalGraph<'a> {
    pub vertices: &'a [[f32; 3]],
    pub infos: &'a [WmoPortalInfo],
    pub refs: &'a [WmoPortalRef],
    pub slices: &'a [(u16, u16)],
}

/// How a group came to be claimed — the audit's readout, and the reason the list is ordered.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ClaimHow {
    /// The group's MOGI box contains the fixture.
    Contains,
    /// The group's MOLR names the fixture.
    Molr,
    /// One portal hop from a claimed group, the portal within the fixture's reach.
    Portal,
    /// MONKEY (split-floor claims): the group on the far side of a portal too LARGE to be a
    /// doorway ([`SPLIT_PORTAL_MIN_AREA`]) — one half of a room the artist SPLIT for rendering.
    /// Ranks and behaves as a BASE claim: hard (weight 1 everywhere), hop 0, and part of the
    /// wavefront the portal walk seeds from.
    Split,
}

impl ClaimHow {
    pub fn tag(self) -> &'static str {
        match self {
            ClaimHow::Contains => "in",
            ClaimHow::Molr => "molr",
            ClaimHow::Portal => "portal",
            ClaimHow::Split => "split",
        }
    }
}

/// One claimed group.
#[derive(Clone, Copy, Debug)]
pub struct Claim {
    pub group: u16,
    pub how: ClaimHow,
    /// Yards from the fixture to the portal it came through ([`ClaimHow::Portal`] only; 0 else).
    pub distance: f32,
    /// The claimed group's own class — `false` for an EXTERIOR-flagged group (MOGP `& 0x48`).
    /// Carried because an exterior-class group inside a building (an inn's basement stairwell, a
    /// covered porch) is drawn by the exterior law and needs a claim to be lit at all.
    pub interior: bool,
    /// May the EXTERIOR batch lane honour this claim? See [`CLAIM_EXT_SHELL_YD`]: a district-scale
    /// shell is kept in the set (dropping it would leave the fixture with NO claims, and the
    /// packer's no-claims arm fails OPEN — a far worse leak than the one being avoided) but is
    /// barred from the exterior lane, where it would put a tavern candle on the street outside.
    /// The interior lane is unaffected either way: it never runs on an exterior-class group's
    /// surfaces, so an exterior claim exists for the exterior lane and for nothing else.
    pub ext_ok: bool,
    /// MONKEY (soft portal claims): how many doorways the light had to cross to reach this group —
    /// `0` for the base (containment/MOLR), `1`/`2` for a portal hop. Ordering and audit readout
    /// only; the shader never sees it (the fade radius below already encodes "how much reach is
    /// left", which is the same fact measured in yards instead of doors).
    pub hops: u8,
    /// MONKEY (soft portal claims): the centre of the portal this claim came through, in the SAME
    /// space `pos` is in (WMO model space, WoW axes) — the point the shader's weight fades away
    /// from. Meaningless when `fade_radius == 0`.
    pub fade_center: [f32; 3],
    /// MONKEY (soft portal claims): the portal polygon's own bounding-SPHERE radius (yd). Distance
    /// measured from `fade_center` is reduced by this before the fade, so every fragment standing
    /// IN the doorway reads weight 1 — which is what makes the far side of the threshold
    /// continuous with the near side (see [`Claim::fade_radius`]).
    pub fade_slack: f32,
    /// MONKEY (soft portal claims): the yards of the fixture's reach still unspent when it arrived
    /// at `fade_center` — the distance over which the shader smoothsteps this claim's weight from
    /// 1 (at the doorway) to 0. **`0` means a HARD claim: weight 1 everywhere**, which is what
    /// every containment/MOLR claim carries, because a fixture standing in a room lights all of it.
    ///
    /// This is what turns the binary gate soft. The gate used to answer yes/no per group, so a
    /// fixture admitted on one side of a group boundary and refused on the other put a straight
    /// brightness STEP across any floor that crossed it — the scar on the Lion's Pride Inn's
    /// upstairs landing, where two groups meet mid-plank. With a weight the near side is 1 by
    /// containment, the far side is ~1 at the threshold (distance to the doorway ~ 0) and decays
    /// to 0 where the fixture's reach runs out anyway, so nothing steps at the plane.
    pub fade_radius: f32,
    /// MONKEY (soft portal claims): the weight this claim ALREADY carries at its own doorway — 1
    /// for a base claim and for a first hop, and for a second hop the weight the FIRST hop's fade
    /// had already fallen to by the time it reached this door. The shader multiplies its
    /// smoothstep by it.
    ///
    /// Without it the fade is continuous at the first threshold and discontinuous at the second,
    /// which is the same bug one room further in: the near room reads (say) 0.4 at its far door
    /// because the first fade has been decaying across it, while the room beyond restarts its own
    /// fade at 1.0 — a step, and a step the WRONG WAY (the further room brighter than the nearer
    /// one). Seeding each hop with its predecessor's value at that point makes the whole chain
    /// continuous by construction, and monotone: light never gets stronger further from its source.
    pub fade_entry: f32,
}

/// The `interiorAttenScale` DEFAULT. A claim set is built ONCE, at spawn, and never rebuilt — but
/// the reach the shader windows a fixture with is live (the cvar). So the portal hop is measured
/// against the DEFAULT reach rather than the live one: turning the cvar up widens every pool but
/// cannot open a doorway that was gated at spawn, and turning it down cannot slam one shut
/// mid-frame. That is the stable half of the trade, and it keeps the claim table free of per-frame
/// work; the alternative (re-claiming every fixture whenever the cvar moves) buys a doorway's worth
/// of light for a rebuild of the whole table. **Mirrors `DynamicInteriors::atten_scale`'s default.**
pub const CLAIM_ATTEN_SCALE: f32 = 1.6;

/// The cap on any claim reach — `benilla_world::lighting`'s `INTERIOR_LEGACY_REACH` /
/// `POINT_LIGHT_RANGE`, i.e. the widest a fixture's window is ever packed.
pub const CLAIM_REACH_MAX: f32 = 48.0;

/// The effective radius (yd) a fixture's portal hop is measured against: its AUTHORED MOLT
/// attenuation end scaled by [`CLAIM_ATTEN_SCALE`], clamped exactly as the packer's
/// `interior_reach` clamps it.
pub fn claim_reach(authored_end: f32) -> f32 {
    (authored_end * CLAIM_ATTEN_SCALE).clamp(1.0, CLAIM_REACH_MAX)
}

/// The interior reach (yd) of an **M2** point light of committed intensity `i` — the bucket ladder
/// [`crate::fire_light::fire_intensity`] buckets a synthesised flame onto (candle 0.6 / torch 1.5 /
/// brazier 2.0 / bonfire 3.0), because the authored M2 attenuation pair is demonstrably not a reach
/// in this corpus (a Karazhan BONFIRE authors `end = 0.97` yd). Lives here rather than in the
/// packer that reads it so the offline audit buckets a prop's synthesised light onto the very same
/// rung the runtime will; `benilla_world::lighting::m2_light_reach` delegates to it.
pub fn m2_light_reach(intensity: f32) -> f32 {
    match intensity {
        i if i < 1.0 => 6.0,   // candle
        i if i < 1.75 => 12.0, // torch / campfire
        i if i < 2.5 => 16.0,  // brazier
        _ => 24.0,             // bonfire / forge
    }
}

/// The AABB of one portal's polygon, or `None` when the record names no usable vertices.
fn portal_box(portals: &PortalGraph, portal: u16) -> Option<([f32; 3], [f32; 3])> {
    let info = portals.infos.get(usize::from(portal))?;
    let start = usize::from(info.start_vertex);
    let end = start.checked_add(usize::from(info.count))?;
    let verts = portals.vertices.get(start..end)?;
    let (first, rest) = verts.split_first()?;
    let mut lo = *first;
    let mut hi = *first;
    for v in rest {
        for a in 0..3 {
            lo[a] = lo[a].min(v[a]);
            hi[a] = hi[a].max(v[a]);
        }
    }
    Some((lo, hi))
}

/// MONKEY (split-floor claims): the AREA of one portal's polygon (yd²), fan-triangulated over the
/// MOPV vertices in authored order. The measure that separates a DOORWAY from a room the artist cut
/// in half: see [`SPLIT_PORTAL_MIN_AREA`] for the two populations it sits between.
pub fn portal_area(portals: &PortalGraph, portal: u16) -> Option<f32> {
    let info = portals.infos.get(usize::from(portal))?;
    let start = usize::from(info.start_vertex);
    let end = start.checked_add(usize::from(info.count))?;
    let verts = portals.vertices.get(start..end)?;
    let (first, rest) = verts.split_first()?;
    let sub = |a: &[f32; 3], b: &[f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let mut area = 0.0f32;
    for pair in rest.windows(2) {
        let (u, v) = (sub(&pair[0], first), sub(&pair[1], first));
        let c = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        area += 0.5 * (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
    }
    Some(area)
}

/// A portal as the claim rule measures it, from `pos`: `(distance, centre, slack)`.
///
/// **distance** — yards from `pos` to the portal's polygon, approximated by its AABB (the closest
/// point on the box), which is exact for the axis-aligned doorways most portals are and never
/// over-states the distance for the rest (erring toward admitting a doorway, which is the safe
/// direction: an extra claim is a room lit one doorway too far, a missing one is the seam we are
/// fixing).
///
/// MONKEY (soft portal claims): **centre** and **slack** (the box's half-diagonal) are the fade
/// geometry the shader gets. Fading from the centre alone would put a step back across the WIDTH
/// of the door — a fragment at the edge of a 1.2 x 2.5 yd doorway is ~1.4 yd from its centre, so
/// with a short remaining reach it would read half-lit one plank away from a fully lit fragment on
/// the other side of the plane. Subtracting the slack collapses the whole doorway to distance 0, so
/// the threshold is weight 1 on BOTH sides by construction and the fade starts where the door frame
/// ends. A sphere rather than the box because it is one number: slightly generous just outside the
/// frame, which is the direction that cannot re-introduce an edge.
fn portal_reach(portals: &PortalGraph, portal: u16, pos: [f32; 3]) -> Option<(f32, [f32; 3], f32)> {
    let (lo, hi) = portal_box(portals, portal)?;
    let mut d2 = 0.0f32;
    let mut center = [0.0f32; 3];
    let mut diag2 = 0.0f32;
    for a in 0..3 {
        let over = (lo[a] - pos[a]).max(pos[a] - hi[a]).max(0.0);
        d2 += over * over;
        center[a] = 0.5 * (lo[a] + hi[a]);
        let half = 0.5 * (hi[a] - lo[a]).max(0.0);
        diag2 += half * half;
    }
    Some((d2.sqrt(), center, diag2.sqrt()))
}

/// MONKEY (soft portal claims): how many portal hops a claim may travel. **1 is not enough**: a
/// fixture two rooms from a fragment is claimed on the near side of a group boundary (one hop) and
/// not at all on the far side, which is a hard step again — the very seam the fade exists to
/// remove, moved one room outward. Two hops means the far side also carries the fixture, at the
/// small weight its remaining reach affords. Three would buy nothing measurable: by the second
/// doorway the unspent reach is already at [`CLAIM_FADE_MIN_YD`] in most vanilla interiors, and
/// each extra hop multiplies the frontier while competing for the same six slots.
pub const ROOM_CLAIM_HOPS: u8 = 2;

/// MONKEY (soft portal claims): the floor on a portal claim's fade radius (yd). A fixture that only
/// just reaches a doorway has ~0 reach left, and a 0-yard smoothstep is a step function — i.e. the
/// hard edge again, at the doorway instead of at the group plane. Two yards is the shortest fade
/// that reads as a fade at walking speed, and a fixture that far gone contributes almost nothing
/// anyway (its own `interior_window` is at the end of its curve), so the floor cannot leak light.
pub const CLAIM_FADE_MIN_YD: f32 = 2.0;

/// MONKEY (soft portal claims): a hop whose [`Claim::fade_entry`] is already below this is not
/// worth one of the six slots — it can contribute at most this fraction of one fixture's already
/// attenuated pool, and the slot is better left to a doorway that can be seen. Dropping it cannot
/// re-open the seam it exists to close: the room on the NEAR side of that door is, by definition,
/// this dim there too, so both sides read ~0 and the boundary stays continuous.
pub const CLAIM_FADE_MIN_ENTRY: f32 = 0.02;

/// MONKEY (split-floor claims): the polygon area (yd²) at or above which a portal is read as a
/// ROOM SPLIT rather than a doorway, so containment claims cross it at FULL weight instead of
/// fading (see [`ClaimHow::Split`]).
///
/// The soft portal fade is continuous only at the door it came THROUGH — `fade_center` is that
/// doorway, and the weight decays away from it. A room the artist cut in two (the Lion's Pride
/// Inn's gallery, split from the common room below it by a 10.5 x 9.1 yd opening; its cellar,
/// reached through a 10.2 x 3.4 yd hole in the kitchen floor) therefore reads the SAME fixture at
/// weight 1 on the near half (containment) and at whatever the fade has decayed to on the far half
/// — a straight brightness line across one continuous floor, at a plane that is not a wall. Worse,
/// the far half then re-seeds its own portal hops at that decayed weight, so every real doorway off
/// it inherits the step (the inn's upstairs rooms took the gallery's fixtures at 0.2–0.5 while the
/// gallery took them at 1.0, and the boundary between them is DIAGONAL — the reported scar).
///
/// **28 yd², from the corpus's own bimodal split.** Measured over the two buildings the seam was
/// reported in, sorted: the Goldshire inn's ten portals run 9.9, 10.8, 11.9, 11.9, 15.6, 17.3,
/// 17.3, 20.5 (every one of them a door or an arch) and then 35.1 (the cellar floor hole) and 95.9
/// (the gallery opening); Northshire abbey's fourteen run 14.3 … 16.5 for nine doorways and then
/// 40.3, 47.1, 50.7, 50.9, 252.6 for its nave splits. The union gap is 20.5 → 35.1 and 28 sits in
/// the middle of it, so no doorway in either building is merged and every split is.
///
/// AREA alone, deliberately — not "horizontal-ish", which is the shape a floor split usually has:
/// the abbey authors a HORIZONTAL portal of 14.3 yd² (g10↔g5, a hatch), and merging a real hatch
/// would light a closed room from the one below it. The two horizontal portals that ARE splits (the
/// inn's 35.1, the abbey's 252.6) both clear the area bar on their own, so the extra criterion buys
/// nothing and costs a false positive.
pub const SPLIT_PORTAL_MIN_AREA: f32 = 28.0;

/// `smoothstep(0, edge, x)` — the shader's own curve, so the entry weight this rule bakes and the
/// weight `static_gx.wgsl` evaluates are the same function sampled at the same point. (Rust has no
/// `smoothstep`; WGSL's is the Hermite `t*t*(3 - 2t)` on the clamped ratio.)
fn smoothstep0(edge: f32, x: f32) -> f32 {
    let t = (x / edge.max(1e-4)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The largest HORIZONTAL extent (yd) an EXTERIOR-class group may have and still be honoured by the
/// EXTERIOR batch lane — "a building, not a district".
///
/// An exterior-class group is drawn by the exterior law and takes no interior fixture, so a claim on
/// one is consumed by exactly one thing: the exterior batch's claim-gated room term (bug B's second
/// cause — the inn's basement stairwell is an EXTERIOR group inside a building and reads night-sky
/// green next to its candle-lit neighbours). That term must reach an annex like the inn's shell
/// (`57.7 x 32.2 yd`) and must NOT reach a city's district shell, whose MOGI box swallows the
/// interiors inside it: measured on `Stormwind.wmo`, 494 of its 606 fixtures stand inside one of
/// ~8 exterior shells spanning 122-660 yd (g107 alone contains 90 of them), and admitting those
/// would put a tavern candle's pool on the cobbles outside the tavern wall. The shipped corpus
/// leaves a wide gap between the two populations — no Stormwind shell is under 122 yd — so the cut
/// sits in the middle of it rather than on either edge.
///
/// A PORTAL claim is exempt: a portal is the artist saying these two groups share a doorway, and
/// light through a door onto the street is right (it is how a lit doorway reads at night).
pub const CLAIM_EXT_SHELL_YD: f32 = 96.0;

/// Is a BOX claim (containment or MOLR) on this group eligible for the exterior lane? Interior
/// groups trivially are (the exterior lane never draws them); an exterior group only at building
/// scale (see [`CLAIM_EXT_SHELL_YD`]).
pub fn claimable_by_box(g: &WmoGroupInfo) -> bool {
    g.interior || (0..2).all(|a| (g.bbox_max[a] - g.bbox_min[a]) <= CLAIM_EXT_SHELL_YD)
}

/// MONKEY (ext-class night law): is this group EXTERIOR-class but at BUILDING scale — an inn's
/// shell, a basement stairwell, a covered porch, as opposed to a city district's shell?
///
/// The same cut [`claimable_by_box`] makes, read from the other side, and deliberately the SAME
/// one: a group the exterior lane will honour a candle's claim on is a group that ought to render
/// by candle-light at night, and a group whose claims that lane refuses (a district shell) must
/// keep the sky. Two thresholds here would put a group in the gap between them — lit by fixtures
/// but still sky-based, or sky-less with no fixtures allowed to reach it.
pub fn ext_building_scale(g: &WmoGroupInfo) -> bool {
    !g.interior && claimable_by_box(g)
}

/// MONKEY (enclosed day floor): the slack (yd) an exterior shell's authored MOGI box is grown by
/// before it is asked whether it CONTAINS a room. An authored box is not always tight around its
/// own geometry (NSabbey g3's bottom floats 1.5 yd above its own floor), and a room whose centre
/// sits a hand's breadth outside its own building's shell is a measurement artefact, not a cave.
pub const SHELL_CONTAIN_SLACK: f32 = 0.5;

/// MONKEY (enclosed day floor): is this INTERIOR-class group a room INSIDE A BUILDING — i.e. is its
/// box centre contained in some EXTERIOR-class group of the same root at building scale
/// ([`ext_building_scale`])?
///
/// The question the daylight floor turns on, and the one thing that separates "a room whose door
/// this renderer cannot find" from "a cave that has no door". A building authors a shell: the
/// Goldshire inn's `g4 upstairs` (57.7 x 32.2 yd, exterior-class) swallows all ten of its rooms,
/// and Northshire abbey's `g5 mainlobby2` its nave. A dungeon authors none — `az_deadmines_*`,
/// Stratholme and the rest are interior groups end to end — so nothing there is ever flagged and
/// the sunless look of a cave is preserved by the DATA rather than by a special case.
///
/// A city's district shell is deliberately refused by [`ext_building_scale`]'s
/// [`CLAIM_EXT_SHELL_YD`] cut, for the same reason the claim lane refuses it: a 660 yd box is not a
/// building, and flagging every interior group in Stormwind because one shell spans the district
/// would daylight its sealed cellars.
pub fn enclosed_by_building_shell(groups: &[WmoGroupInfo], group: u16) -> bool {
    let Some(g) = groups.get(usize::from(group)) else {
        return false;
    };
    if !g.interior {
        return false; // an exterior-class group is lit by the sky already
    }
    let c = [
        0.5 * (g.bbox_min[0] + g.bbox_max[0]),
        0.5 * (g.bbox_min[1] + g.bbox_max[1]),
        0.5 * (g.bbox_min[2] + g.bbox_max[2]),
    ];
    groups.iter().enumerate().any(|(i, e)| {
        i != usize::from(group)
            && ext_building_scale(e)
            && (0..3).all(|a| {
                c[a] >= e.bbox_min[a] - SHELL_CONTAIN_SLACK
                    && c[a] <= e.bbox_max[a] + SHELL_CONTAIN_SLACK
            })
    })
}

/// `pos` inside a group's authored MOGI box.
fn contains(g: &WmoGroupInfo, pos: [f32; 3]) -> bool {
    (0..3).all(|a| pos[a] >= g.bbox_min[a] && pos[a] <= g.bbox_max[a])
}

/// The box's volume — the containment ORDER key. A room is small, the shell that spans the whole
/// building is not, and when six slots have to be spent the room the fixture stands in must get one.
fn volume(g: &WmoGroupInfo) -> f32 {
    (0..3)
        .map(|a| (g.bbox_max[a] - g.bbox_min[a]).max(0.0))
        .product()
}

/// The claim set of ONE fixture at `pos` (WMO model space, WoW axes) with effective radius `reach`
/// yards, whose MOLR referrers are `molr`. Ordered by the priority in the module doc and capped at
/// [`ROOM_CLAIM_MAX`] — **except** that the base (containment ∪ MOLR) is returned whole even when it
/// is longer: the packer's own overflow arm decides that case (MONKEY (GO room claims): it keeps
/// the first [`ROOM_CLAIM_MAX`] in this priority order — it used to fail open, which lit the whole
/// building through its walls), so this function keeps the full ordered set for the audits.
///
/// `reach` of 0 or less disables the portal hop entirely (a fixture with no usable window).
pub fn room_claims(
    groups: &[WmoGroupInfo],
    portals: PortalGraph<'_>,
    pos: [f32; 3],
    reach: f32,
    molr: &[u16],
) -> Vec<Claim> {
    let mut out: Vec<Claim> = Vec::new();
    // MONKEY (soft portal claims): `fade` is `None` for a HARD claim (containment/MOLR — weight 1
    // everywhere) and `Some((centre, slack, radius))` for a portal hop.
    let push = |out: &mut Vec<Claim>,
                group: u16,
                how: ClaimHow,
                distance: f32,
                hops: u8,
                fade: Option<([f32; 3], f32, f32, f32)>| {
        if out.iter().any(|c| c.group == group) {
            return;
        }
        let g = groups.get(usize::from(group));
        let (fade_center, fade_slack, fade_radius, fade_entry) =
            fade.unwrap_or(([0.0; 3], 0.0, 0.0, 1.0));
        out.push(Claim {
            group,
            how,
            distance,
            interior: g.is_some_and(|g| g.interior),
            // A portal is the artist saying these two groups share a doorway, so it is always
            // exterior-lane eligible; a box claim has to be building-scale to be.
            ext_ok: how == ClaimHow::Portal || g.is_none_or(claimable_by_box),
            hops,
            fade_center,
            fade_slack,
            fade_radius,
            fade_entry,
        });
    };
    // 1. CONTAINMENT, tightest box first.
    let mut inside: Vec<(usize, bool, f32)> = groups
        .iter()
        .enumerate()
        .filter(|(_, g)| contains(g, pos))
        .map(|(i, g)| (i, !claimable_by_box(g), volume(g)))
        .collect();
    // Tightest box first, and a district-scale shell (which no lane will light from) after every
    // real room, so it can never take the slot a portal hop into the next room wanted.
    inside.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.total_cmp(&b.2)));
    for (i, _, _) in inside {
        push(&mut out, i as u16, ClaimHow::Contains, 0.0, 0, None);
    }
    // 2. MOLR.
    for &g in molr {
        push(&mut out, g, ClaimHow::Molr, 0.0, 0, None);
    }
    // 2b. SPLIT-FLOOR SIBLINGS — MONKEY (split-floor claims). A portal too big to be a doorway
    //     ([`SPLIT_PORTAL_MIN_AREA`]) is the artist cutting ONE room in two for rendering, and the
    //     two halves must light as one: the far half takes this fixture at weight 1, exactly as the
    //     half it physically stands in does. Pushed HERE — after the base, before the hop — because
    //     it is base-equivalent in every way that matters downstream:
    //
    //     * HARD (`fade = None`): the fade exists to make a THRESHOLD continuous, and a split has
    //       no threshold. Fading across it is what put the step on the continuous floor.
    //     * hop 0, and therefore in the wavefront the portal walk seeds from below at spend 0. That
    //       is the half of this rule that closes the *second-order* seam: the inn's upstairs rooms
    //       used to reach their gallery fixtures as hop-TWO claims (through the gallery opening
    //       first), entering at the 0.2–0.5 the first fade had already decayed to, while the
    //       gallery itself held them at 1.0 — a step at every one of those doorways, the wrong way
    //       round. Re-seeded from the merged half they are hop-ONE claims entering at 1.0, which is
    //       exactly the weight the gallery side reads there.
    //     * ranked above portal claims when the six slots are contested, for the same reason
    //       containment is: this is the room the fixture is in, spelt with two group ids.
    //
    //     Transitive (the queue below re-walks each merged group), so a room cut into three merges
    //     whole. Both ends must be [`claimable_by_box`] — a district-scale shell is never one half
    //     of a room — and the opening must be within the fixture's own reach, so a split it cannot
    //     light is not merged into its set at the cost of a slot.
    if reach > 0.0 && !portals.infos.is_empty() && out.len() < ROOM_CLAIM_MAX {
        let mut queue: Vec<u16> = out.iter().map(|c| c.group).collect();
        let mut head = 0usize;
        while head < queue.len() && out.len() < ROOM_CLAIM_MAX {
            let g = queue[head];
            head += 1;
            if groups
                .get(usize::from(g))
                .is_none_or(|gi| !claimable_by_box(gi))
            {
                continue;
            }
            let Some(&(start, count)) = portals.slices.get(usize::from(g)) else {
                continue;
            };
            let from = usize::from(start);
            let to = from.saturating_add(usize::from(count));
            for r in portals.refs.get(from..to).unwrap_or(&[]) {
                if out.len() >= ROOM_CLAIM_MAX {
                    break;
                }
                if out.iter().any(|c| c.group == r.group)
                    || groups
                        .get(usize::from(r.group))
                        .is_none_or(|gi| !claimable_by_box(gi))
                    || portal_area(&portals, r.portal).unwrap_or(0.0) < SPLIT_PORTAL_MIN_AREA
                {
                    continue;
                }
                let Some((d, ..)) = portal_reach(&portals, r.portal, pos) else {
                    continue;
                };
                if d > reach {
                    continue;
                }
                push(&mut out, r.group, ClaimHow::Split, d, 0, None);
                queue.push(r.group);
            }
        }
    }
    // 3. PORTAL-ADJACENT — up to [`ROOM_CLAIM_HOPS`] doorways off the base, nearest portal first
    //    within each hop, and every hop's claim carries the fade that makes it SOFT.
    //
    //    MONKEY (soft portal claims): this used to stop at ONE hop, and stopping there was itself a
    //    hard edge. A fixture two rooms from a group boundary is claimed on the near side (one hop)
    //    and refused on the far side, so the binary gate's step simply moved one room outward —
    //    the same scar, a doorway further along. The wavefront below SPENDS the fixture's reach as
    //    it walks: each hop's remaining reach becomes the fade radius past that doorway, so by the
    //    second door the weight is already small and a third door would not be worth a slot.
    if out.len() >= ROOM_CLAIM_MAX || reach <= 0.0 || portals.infos.is_empty() {
        return out;
    }
    // The wavefront: `(group, yards ALREADY SPENT getting there)`. The base is at zero — the
    // fixture physically stands in those groups.
    // `(group, yards spent, index of the claim in `out`)` — the index is what lets the next hop
    // sample its predecessor's fade at the door it is about to cross (see [`Claim::fade_entry`]).
    let mut frontier: Vec<(u16, f32, usize)> = out
        .iter()
        .enumerate()
        .map(|(i, c)| (c.group, 0.0, i))
        .collect();
    for hop in 1..=ROOM_CLAIM_HOPS {
        if out.len() >= ROOM_CLAIM_MAX {
            break;
        }
        // `(group, spent, portal centre, portal slack, seed claim)`, deduped to the CHEAPEST
        // way in.
        let mut next: Vec<(u16, f32, [f32; 3], f32, usize)> = Vec::new();
        for &(g, spent, seed) in &frontier {
            let Some(&(start, count)) = portals.slices.get(usize::from(g)) else {
                continue;
            };
            let from = usize::from(start);
            let to = from.saturating_add(usize::from(count));
            for r in portals.refs.get(from..to).unwrap_or(&[]) {
                if out.iter().any(|c| c.group == r.group) {
                    continue; // already claimed, by a tighter source or a cheaper hop
                }
                let Some((d, center, slack)) = portal_reach(&portals, r.portal, pos) else {
                    continue;
                };
                // The straight line to the SECOND door can be shorter than to the first (an
                // L-shaped pair of rooms folds back toward the fixture), but the light still had
                // to cross the first one. `max` keeps the spend monotone along the path, so a
                // second-hop room can never be claimed more cheaply than the room it hides behind.
                let d = d.max(spent);
                if d > reach {
                    continue;
                }
                match next.iter_mut().find(|(n, ..)| *n == r.group) {
                    Some(slot) if slot.1 <= d => {}
                    Some(slot) => *slot = (r.group, d, center, slack, seed),
                    None => next.push((r.group, d, center, slack, seed)),
                }
            }
        }
        next.sort_by(|a, b| a.1.total_cmp(&b.1));
        frontier.clear();
        for (g, d, center, slack, seed) in next {
            if out.len() >= ROOM_CLAIM_MAX {
                break;
            }
            // The weight the SEEDING claim still has at this doorway — 1 off a base claim (which
            // is hard), and the first hop's decayed value off a first hop. This is the whole of
            // the multi-hop continuity argument: hop N+1 starts at exactly the value hop N ends
            // at, so the chain has no step at any threshold and never brightens with distance.
            let seed = &out[seed];
            let entry = if seed.fade_radius <= 0.0 {
                1.0
            } else {
                let gap = (0..3)
                    .map(|a| (center[a] - seed.fade_center[a]).powi(2))
                    .sum::<f32>()
                    .sqrt();
                seed.fade_entry
                    * (1.0 - smoothstep0(seed.fade_radius, (gap - seed.fade_slack).max(0.0)))
            };
            if entry < CLAIM_FADE_MIN_ENTRY {
                continue; // too dim to be worth a slot (see the constant)
            }
            // The fade radius is what is LEFT of the reach past this doorway, floored (see
            // [`CLAIM_FADE_MIN_YD`]: a zero-length smoothstep is a step function again).
            let radius = (reach - d).max(CLAIM_FADE_MIN_YD);
            push(
                &mut out,
                g,
                ClaimHow::Portal,
                d,
                hop,
                Some((center, slack, radius, entry)),
            );
            frontier.push((g, d, out.len() - 1));
        }
    }
    out
}

/// [`room_claims`] reduced to the group ids the packer wants.
pub fn claim_groups(
    groups: &[WmoGroupInfo],
    portals: PortalGraph<'_>,
    pos: [f32; 3],
    reach: f32,
    molr: &[u16],
) -> Vec<u16> {
    room_claims(groups, portals, pos, reach, molr)
        .into_iter()
        .map(|c| c.group)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(interior: bool, lo: [f32; 3], hi: [f32; 3]) -> WmoGroupInfo {
        WmoGroupInfo {
            interior,
            show_skybox: false,
            bbox_min: lo,
            bbox_max: hi,
        }
    }

    /// MONKEY (enclosed day floor): a room inside a building's shell is flagged; a cave, a room
    /// under a CITY-scale shell, and the shell itself are not.
    #[test]
    fn enclosed_is_a_room_in_a_building_and_nothing_else() {
        // 0 the room, 1 the inn's own shell (57 x 32 yd, building scale), 2 a second room just
        // OUTSIDE that shell, 3 a district-scale shell (200 yd) containing everything.
        let groups = [
            g(true, [0.0, 0.0, 0.0], [10.0, 10.0, 5.0]),
            g(false, [-5.0, -5.0, -1.0], [52.0, 27.0, 10.0]),
            g(true, [90.0, 90.0, 0.0], [95.0, 95.0, 5.0]),
            g(false, [-100.0, -100.0, -10.0], [100.0, 100.0, 40.0]),
        ];
        assert!(enclosed_by_building_shell(&groups, 0));
        assert!(
            !enclosed_by_building_shell(&groups, 1),
            "a shell is not a room"
        );
        assert!(
            !enclosed_by_building_shell(&groups, 2),
            "only the DISTRICT shell holds it, and that is not a building"
        );
        assert!(!enclosed_by_building_shell(&groups, 3));
        assert!(!enclosed_by_building_shell(&groups, 9), "out of range");
        // A dungeon: interior groups end to end, no shell anywhere -> nothing is ever flagged.
        let cave = [
            g(true, [0.0, 0.0, 0.0], [10.0, 10.0, 5.0]),
            g(true, [10.0, 0.0, 0.0], [20.0, 10.0, 5.0]),
        ];
        assert!(!enclosed_by_building_shell(&cave, 0));
        assert!(!enclosed_by_building_shell(&cave, 1));
        // The slack really is applied: a room whose centre sits a hand's breadth outside its own
        // shell is still in the building.
        let tight = [
            g(true, [0.0, 0.0, 0.0], [10.0, 10.0, 5.0]),
            g(false, [-5.0, -5.0, -1.0], [52.0, 27.0, 2.3]),
        ];
        assert!(
            enclosed_by_building_shell(&tight, 0),
            "centre z 2.5 vs shell top 2.3 + 0.5"
        );
    }

    /// A quad portal in the x = 5 plane, spanning y 0..2, z 0..3.
    fn one_portal() -> (Vec<[f32; 3]>, Vec<WmoPortalInfo>) {
        (
            vec![
                [5.0, 0.0, 0.0],
                [5.0, 2.0, 0.0],
                [5.0, 2.0, 3.0],
                [5.0, 0.0, 3.0],
            ],
            vec![WmoPortalInfo {
                start_vertex: 0,
                count: 4,
                plane: [1.0, 0.0, 0.0, -5.0],
            }],
        )
    }

    #[test]
    fn containment_is_tightest_first_and_molr_follows() {
        let groups = [
            g(true, [-50.0, -50.0, -10.0], [50.0, 50.0, 30.0]), // g0 the building shell
            g(true, [-2.0, -2.0, 0.0], [4.0, 4.0, 5.0]),        // g1 the room it stands in
        ];
        let c = room_claims(&groups, PortalGraph::default(), [0.0, 0.0, 1.0], 8.0, &[7]);
        assert_eq!(
            c.iter().map(|c| c.group).collect::<Vec<_>>(),
            vec![1, 0, 7],
            "the tight room outranks the shell, and MOLR follows both"
        );
        assert_eq!(c[0].how, ClaimHow::Contains);
        assert_eq!(c[2].how, ClaimHow::Molr);
    }

    #[test]
    fn a_portal_within_reach_claims_the_next_room() {
        let groups = [
            g(true, [-5.0, -5.0, 0.0], [5.0, 5.0, 6.0]), // g0, holds the fixture
            g(true, [5.0, -5.0, 0.0], [15.0, 5.0, 6.0]), // g1, through the doorway
        ];
        let (verts, infos) = one_portal();
        let refs = [
            WmoPortalRef {
                portal: 0,
                group: 1,
                side: 1,
            },
            WmoPortalRef {
                portal: 0,
                group: 0,
                side: -1,
            },
        ];
        let slices = [(0u16, 1u16), (1u16, 1u16)];
        let portals = PortalGraph {
            vertices: &verts,
            infos: &infos,
            refs: &refs,
            slices: &slices,
        };
        // 3 yd from the doorway, reach 8 ⇒ the next room is claimed…
        let c = room_claims(&groups, portals, [2.0, 1.0, 1.0], 8.0, &[]);
        assert_eq!(c.iter().map(|c| c.group).collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(c[1].how, ClaimHow::Portal);
        // …and out of reach it is not: the far room stays gated, which is the whole point.
        let c = room_claims(&groups, portals, [-4.0, 1.0, 1.0], 5.0, &[]);
        assert_eq!(c.iter().map(|c| c.group).collect::<Vec<_>>(), vec![0]);
    }

    #[test]
    fn portal_claims_never_displace_the_base() {
        // Six containing groups fill every slot; the doorway beyond gets nothing.
        let mut groups: Vec<WmoGroupInfo> = (0..6)
            .map(|i| g(true, [-5.0, -5.0, 0.0], [5.0 + i as f32, 5.0, 6.0]))
            .collect();
        groups.push(g(true, [5.0, -5.0, 0.0], [15.0, 5.0, 6.0]));
        let (verts, infos) = one_portal();
        let refs = [WmoPortalRef {
            portal: 0,
            group: 6,
            side: 1,
        }];
        let mut slices = vec![(0u16, 1u16)];
        slices.resize(7, (0, 0));
        let portals = PortalGraph {
            vertices: &verts,
            infos: &infos,
            refs: &refs,
            slices: &slices,
        };
        let c = room_claims(&groups, portals, [0.0, 0.0, 1.0], 20.0, &[]);
        assert_eq!(c.len(), ROOM_CLAIM_MAX);
        assert!(c.iter().all(|c| c.how == ClaimHow::Contains));
    }

    /// MONKEY (soft portal claims): a portal claim carries the DOORWAY it came through and the
    /// reach left past it — the two numbers the shader's weight is built from. A base claim
    /// carries neither, which is how the shader tells "hard" from "faded".
    #[test]
    fn a_portal_claim_carries_its_doorway_and_the_reach_left_past_it() {
        let groups = [
            g(true, [-5.0, -5.0, 0.0], [5.0, 5.0, 6.0]),
            g(true, [5.0, -5.0, 0.0], [15.0, 5.0, 6.0]),
        ];
        let (verts, infos) = one_portal();
        let refs = [
            WmoPortalRef {
                portal: 0,
                group: 1,
                side: 1,
            },
            WmoPortalRef {
                portal: 0,
                group: 0,
                side: -1,
            },
        ];
        let slices = [(0u16, 1u16), (1u16, 1u16)];
        let portals = PortalGraph {
            vertices: &verts,
            infos: &infos,
            refs: &refs,
            slices: &slices,
        };
        // 3 yd short of the doorway (x = 5), reach 8 ⇒ 5 yd of reach left on the far side.
        let c = room_claims(&groups, portals, [2.0, 1.0, 1.0], 8.0, &[]);
        assert_eq!(
            c[0].fade_radius, 0.0,
            "a containment claim is HARD: weight 1 everywhere"
        );
        assert_eq!(c[0].hops, 0);
        let door = c[1];
        assert_eq!(door.how, ClaimHow::Portal);
        assert_eq!(door.hops, 1);
        // The quad spans y 0..2, z 0..3 in the x = 5 plane: centre (5, 1, 1.5), half-diagonal
        // sqrt(0 + 1 + 1.5^2) = 1.803.
        assert_eq!(door.fade_center, [5.0, 1.0, 1.5]);
        assert!((door.fade_slack - 1.802_775).abs() < 1e-4, "{door:?}");
        assert!(
            (door.distance - 3.0).abs() < 1e-5,
            "closest point on the quad, not its centre"
        );
        assert!(
            (door.fade_radius - 5.0).abs() < 1e-5,
            "reach 8 minus the 3 yd spent"
        );
        assert_eq!(
            door.fade_entry, 1.0,
            "a first hop is at full weight in its own doorway"
        );

        // The floor on the fade: a fixture that only just reaches the doorway must still FADE
        // there, not step. Reach 3.0 leaves 0 yd, and a 0-yard smoothstep is the hard edge again.
        let c = room_claims(&groups, portals, [2.0, 1.0, 1.0], 3.0, &[]);
        assert_eq!(c[1].fade_radius, CLAIM_FADE_MIN_YD, "clamped, never zero");
    }

    /// MONKEY (split-floor claims): a portal too LARGE to be a doorway merges the two halves of the
    /// room it cuts — the far half takes the fixture HARD (weight 1), and the walk re-seeds from it
    /// so a real doorway off that half is a FIRST hop at full entry, not a decayed second one.
    #[test]
    fn a_split_portal_merges_both_halves_and_reseeds_the_walk() {
        // g0 holds the fixture, g1 is the other half of the same room (a 5 x 6 = 30 yd² opening,
        // over SPLIT_PORTAL_MIN_AREA), g2 is a real room off g1 through a 2 x 3 = 6 yd² door.
        let groups = [
            g(true, [-5.0, -5.0, 0.0], [5.0, 5.0, 6.0]),
            g(true, [5.0, -5.0, 0.0], [15.0, 5.0, 6.0]),
            g(true, [15.0, -5.0, 0.0], [25.0, 5.0, 6.0]),
        ];
        let verts = vec![
            [5.0, -2.5, 0.0],
            [5.0, 2.5, 0.0],
            [5.0, 2.5, 6.0],
            [5.0, -2.5, 6.0],
            [15.0, 0.0, 0.0],
            [15.0, 2.0, 0.0],
            [15.0, 2.0, 3.0],
            [15.0, 0.0, 3.0],
        ];
        let infos = vec![
            WmoPortalInfo {
                start_vertex: 0,
                count: 4,
                plane: [1.0, 0.0, 0.0, -5.0],
            },
            WmoPortalInfo {
                start_vertex: 4,
                count: 4,
                plane: [1.0, 0.0, 0.0, -15.0],
            },
        ];
        let refs = [
            WmoPortalRef {
                portal: 0,
                group: 1,
                side: 1,
            },
            WmoPortalRef {
                portal: 0,
                group: 0,
                side: -1,
            },
            WmoPortalRef {
                portal: 1,
                group: 2,
                side: 1,
            },
            WmoPortalRef {
                portal: 1,
                group: 1,
                side: -1,
            },
        ];
        let slices = [(0u16, 1u16), (1u16, 2u16), (3u16, 1u16)];
        let portals = PortalGraph {
            vertices: &verts,
            infos: &infos,
            refs: &refs,
            slices: &slices,
        };
        assert!((portal_area(&portals, 0).unwrap() - 30.0).abs() < 1e-3);
        assert!((portal_area(&portals, 1).unwrap() - 6.0).abs() < 1e-3);
        let c = room_claims(&groups, portals, [2.0, 0.0, 1.0], 20.0, &[]);
        assert_eq!(
            c.iter()
                .map(|c| (c.group, c.how, c.hops))
                .collect::<Vec<_>>(),
            vec![
                (0, ClaimHow::Contains, 0),
                (1, ClaimHow::Split, 0),
                (2, ClaimHow::Portal, 1),
            ],
        );
        assert_eq!(
            c[1].fade_radius, 0.0,
            "a split is HARD — weight 1 across the far half"
        );
        assert_eq!(
            c[2].fade_entry, 1.0,
            "…so the door off it is a FIRST hop at full entry"
        );
        // Under the bar the same portal is an ordinary doorway again: g1 fades, g2 is hop TWO.
        let small = vec![
            [5.0, -1.0, 0.0],
            [5.0, 1.0, 0.0],
            [5.0, 1.0, 3.0],
            [5.0, -1.0, 3.0],
            [15.0, 0.0, 0.0],
            [15.0, 2.0, 0.0],
            [15.0, 2.0, 3.0],
            [15.0, 0.0, 3.0],
        ];
        let portals = PortalGraph {
            vertices: &small,
            ..portals
        };
        let c = room_claims(&groups, portals, [2.0, 0.0, 1.0], 20.0, &[]);
        assert_eq!(
            c.iter()
                .map(|c| (c.group, c.how, c.hops))
                .collect::<Vec<_>>(),
            vec![
                (0, ClaimHow::Contains, 0),
                (1, ClaimHow::Portal, 1),
                (2, ClaimHow::Portal, 2),
            ],
        );
    }

    /// MONKEY (soft portal claims): TWO hops. A fixture two rooms away has to be claimed on BOTH
    /// sides of the far group boundary or the seam simply moves one room outward — and the
    /// second hop's fade is measured from the SECOND doorway, with what is left of the reach.
    #[test]
    fn the_second_hop_claims_the_room_beyond_and_fades_from_its_own_door() {
        // Three rooms in a row along x, doors at x = 5 and x = 15.
        let groups = [
            g(true, [-5.0, -5.0, 0.0], [5.0, 5.0, 6.0]),
            g(true, [5.0, -5.0, 0.0], [15.0, 5.0, 6.0]),
            g(true, [15.0, -5.0, 0.0], [25.0, 5.0, 6.0]),
        ];
        let verts = vec![
            [5.0, 0.0, 0.0],
            [5.0, 2.0, 0.0],
            [5.0, 2.0, 3.0],
            [5.0, 0.0, 3.0],
            [15.0, 0.0, 0.0],
            [15.0, 2.0, 0.0],
            [15.0, 2.0, 3.0],
            [15.0, 0.0, 3.0],
        ];
        let infos = vec![
            WmoPortalInfo {
                start_vertex: 0,
                count: 4,
                plane: [1.0, 0.0, 0.0, -5.0],
            },
            WmoPortalInfo {
                start_vertex: 4,
                count: 4,
                plane: [1.0, 0.0, 0.0, -15.0],
            },
        ];
        let refs = [
            WmoPortalRef {
                portal: 0,
                group: 1,
                side: 1,
            }, // g0 -> g1
            WmoPortalRef {
                portal: 0,
                group: 0,
                side: -1,
            }, // g1 -> g0
            WmoPortalRef {
                portal: 1,
                group: 2,
                side: 1,
            }, // g1 -> g2
            WmoPortalRef {
                portal: 1,
                group: 1,
                side: -1,
            }, // g2 -> g1
        ];
        let slices = [(0u16, 1u16), (1u16, 2u16), (3u16, 1u16)];
        let portals = PortalGraph {
            vertices: &verts,
            infos: &infos,
            refs: &refs,
            slices: &slices,
        };
        // The fixture stands in g0 at x = 2, reach 20: 3 yd to the first door, 13 to the second.
        let c = room_claims(&groups, portals, [2.0, 1.0, 1.0], 20.0, &[]);
        assert_eq!(
            c.iter().map(|c| (c.group, c.hops)).collect::<Vec<_>>(),
            vec![(0, 0), (1, 1), (2, 2)],
            "one claim per room, in hop order"
        );
        assert_eq!(
            c[2].fade_center[0], 15.0,
            "the SECOND door is what g2 fades from"
        );
        assert!(
            (c[2].fade_radius - 7.0).abs() < 1e-5,
            "20 yd reach, 13 spent: {:?}",
            c[2]
        );
        // CONTINUITY at the second threshold: g2 must START at exactly the weight g1's own fade
        // has reached there, or the far room reads BRIGHTER than the near one across one plank.
        // g1's door is at (5, 1, 1.5) with slack 1.803 and radius 17; g2's is at (15, 1, 1.5),
        // 10 yd away, so g1 is at 1 - smoothstep(0, 17, 10 - 1.803) = 1 - 0.3835.
        let g1 = c[1];
        let gap = 10.0f32 - g1.fade_slack;
        let t = gap / g1.fade_radius;
        let expect = 1.0 - t * t * (3.0 - 2.0 * t);
        assert_eq!(c[0].fade_entry, 1.0, "a base claim enters at full weight");
        assert_eq!(g1.fade_entry, 1.0, "so does the first hop, at its own door");
        assert!(
            (c[2].fade_entry - expect).abs() < 1e-4,
            "the second hop enters at the first hop's value there: {} vs {expect}",
            c[2].fade_entry,
        );

        // Out of the reach of the second door, the third room stays unclaimed — the walk spends
        // reach, it does not flood.
        let c = room_claims(&groups, portals, [2.0, 1.0, 1.0], 10.0, &[]);
        assert_eq!(c.iter().map(|c| c.group).collect::<Vec<_>>(), vec![0, 1]);
    }
}
