//! The effect-model animation lifecycle, `Stand` → `Hold` → `Decay`, of a spell-visual `CEffect`.
//!
//! The model-load bootstrap (`0x710153`–`0x71019b`) arms `AnimationData.dbc` id 0 `Stand`, not the
//! file-order-first sequence. At each span's end the scene tick runs the model's completion
//! callback with the completed id (`0x719370`, `0x707595`), picked per [`FxStage`] by
//! `PlaySpellVisualKit 0x60edf0`: stages 0/1 destroy (`0x5fbf50`), which the span clock in
//! [`super::attach_spell_fx`] models; stage 2 hands over to `Hold` if the model authors it, else
//! stays on the birth (`0x5ff170`); stages 3/4 re-arm the completed id forever (`0x60ed00`). The
//! reap plays `Decay` out (`0x614150`); the destructor `0x6203e0` plays nothing.
//!
//! Deviation: `Hold` is one repeating play where the reference re-arms it each pass with variation
//! -1, a weighted random pick, because at most one `Spells\` model authors several variations.

use benilla_assets::ModelAnimations;
use bevy::animation::{graph::AnimationNodeIndex, RepeatAnimation};
use bevy::prelude::*;

use benilla_world::lighting::WorldPointLight;

use crate::creature_anim::FxStage;

/// `AnimationData.dbc` 158 `Hold`, the sustained pulse leg (`0x9e` at `0x5ff188`/`0x5ff1bb`).
pub(crate) const ANIM_HOLD: u16 = 158;
/// `AnimationData.dbc` 159 `Decay`, the fade-out leg (`0x9f` at `0x5ff233`/`0x6141c0`).
pub(crate) const ANIM_DECAY: u16 = 159;

/// The completion callback an effect instance carries (the `model+0x70` registration `0x711bb0`
/// writes). On every kit-effect root that armed a rig, since the reap needs its player; a missile,
/// an item glow or the `fxview` fixture is not a `CEffect` and carries none.
#[derive(Component)]
pub(crate) enum FxAnimLife {
    /// Stage 2's birth, waiting for `0x5ff170`'s handover to `Hold`.
    Birth(AnimationNodeIndex),
    /// Nothing left to advance; the node is kept so a reap can stop it before arming `Decay`.
    Settled(AnimationNodeIndex),
}

impl FxAnimLife {
    /// Arm `clip` for `stage`. [`FxStage::Relive`] repeats whatever the sequence flag says
    /// (`0x60ed00` reads none); the other stages wrap a bit-0-clear sequence (`0x71462a`) and clamp
    /// a bit-0-set one (`0x7145db`), as the M2 sampler does.
    pub(super) fn arm(
        player: &mut AnimationPlayer,
        clip: &benilla_assets::AnimClip,
        stage: FxStage,
    ) -> Self {
        let play = player.play(clip.node);
        if clip.looping || stage == FxStage::Relive {
            play.repeat();
        }
        match stage {
            FxStage::State => Self::Birth(clip.node),
            FxStage::OneShot | FxStage::Relive => Self::Settled(clip.node),
        }
    }

    fn armed(&self) -> AnimationNodeIndex {
        match self {
            Self::Birth(n) | Self::Settled(n) => *n,
        }
    }
}

/// Marks a reaped instance that plays its `Decay` out (`0x6141c0`); its expiry is already set to
/// the decay span.
#[derive(Component)]
pub(crate) struct FxDecay;

/// Run each instance's completion callback, the `Hold` handover or the reap's `Decay`. The
/// reference drains them at the end of the scene tick (`0x707595` in `0x7074b0`); run in `Update`,
/// after `PreUpdate` advances the players, this fires in the frame the span ends. Completion is
/// `completions() >= 1`, not `is_finished()`: a wrapping birth never finishes, and the reference
/// latches at the first span end either way (`0x7194bc`).
pub(crate) fn advance_fx_anim(
    mut commands: Commands,
    mut instances: Query<(
        Entity,
        &mut FxAnimLife,
        &mut AnimationPlayer,
        &ModelAnimations,
        Has<FxDecay>,
    )>,
) {
    for (root, mut life, mut player, anims, decaying) in &mut instances {
        if decaying {
            arm_decay(&mut player, life.armed(), anims);
            // The expiry clock owns the despawn now, and nothing may re-arm over the decay.
            commands.entity(root).try_remove::<FxAnimLife>();
            continue;
        }
        let FxAnimLife::Birth(armed) = *life else {
            continue; // settled: nothing changes on completion
        };
        // `0x719370`'s fire-once latch: one notification per authored span.
        if !player
            .animation(armed)
            .is_some_and(|a| a.completions() >= 1)
        {
            continue;
        }
        // `0x5ff170`: arm `Hold` if the model authors it and keep it running (`0x5ff1d0`'s deadline
        // `node+0x58` is set only at stage 3); otherwise stay parked on the birth.
        match anims.find(ANIM_HOLD) {
            Some(hold) => {
                player.stop(armed);
                player.play(hold.node).set_repeat(RepeatAnimation::Forever);
                *life = FxAnimLife::Settled(hold.node);
                trace_leg("hold", root, ANIM_HOLD);
            }
            None => {
                *life = FxAnimLife::Settled(armed);
                trace_leg("park", root, 0);
            }
        }
    }
}

/// The lifecycle's trace line (`WOW_MOVE_TRACE=<path>`, tag `fx`), one per leg change, beside the
/// instance lane's `kit spawn` and `kit expire`.
pub(super) fn trace_leg(leg: &str, root: Entity, anim_id: u16) {
    if !benilla_assets::trace::enabled() {
        return;
    }
    benilla_assets::trace::line("fx", &format!("leg {leg} e={root} anim={anim_id}"));
}

/// The reap's decay-out (`0x614150`, gates `0x614187`–`0x6141a1`): arm `Decay` if the model authors
/// it; otherwise the caller's immediate expiry stands (`0x6141d6`, straight to the destructor).
fn arm_decay(player: &mut AnimationPlayer, armed: AnimationNodeIndex, anims: &ModelAnimations) {
    let Some(decay) = anims.find(ANIM_DECAY) else {
        return;
    };
    player.stop(armed);
    // Never repeated: the instance is destroyed at this sequence's completion.
    player
        .play(decay.node)
        .set_repeat(RepeatAnimation::Never)
        .replay();
}

/// How long a reaped instance keeps rendering (`0x6141f0`): its model's `Decay` span, or `None`
/// when it authors none and is destroyed at once.
pub(crate) fn decay_span(anims: Option<&ModelAnimations>) -> Option<f32> {
    anims?.find(ANIM_DECAY).map(|c| c.duration)
}

// -------------------------------------------------------------------------------------------
// MONKEY (spell light): the LIGHT a luminous effect throws, and the envelope it throws it on.
// -------------------------------------------------------------------------------------------
//
// The synthesis — which effects are luminous at all (fire / holy / fel, never frost, nature,
// arcane or shadow), what colour they burn and how far they reach — is
// [`benilla_formats::fire_light`]'s spell route, decided once per MODEL at asset load. What lives
// here is everything that is per-INSTANCE: when the light comes up, how it goes out, and how many
// of them may exist at once.
//
// It sits in this file rather than beside the other carried lights because it is a LIFECYCLE, and
// this file is where an effect instance's lifecycle lives (`Stand` → `Hold` → `Decay`). The light
// is a child of the effect's own root, so the reap that despawns the root takes the light with it
// and there is no orphaning path to get wrong; the envelope below only shapes the brightness
// inside that life.

/// Seconds a spell light takes to come up. Short enough to read as a flash rather than a fade-in —
/// the point is only that a light never appears at full strength on a single frame, which reads as
/// a rendering fault rather than as a spell.
pub(crate) const SPELL_LIGHT_RAMP: f32 = 0.1;

/// The default burst fade ([`SpellLightMode::Burst`]) when the caller knows no better span: an
/// impact's flash. Long enough to see, short enough that a volley of them never overlaps into a
/// standing glow.
pub(crate) const SPELL_BURST_SPAN: f32 = 0.6;

/// How fast a KIT light goes dark once its instance is reaped ([`FxDecay`]). Deliberately quicker
/// than the model's own `Decay` sequence: the aura is over the moment the server said so, and a
/// light that outlived the visual by a second would read as a stuck effect.
pub(crate) const SPELL_REAP_FADE: f32 = 0.25;

/// MONKEY (area spell light): the AREA mode's ramp — longer than [`SPELL_LIGHT_RAMP`] on purpose.
/// The 0.1 s ramp exists to keep a FLASH from appearing on one frame; a ground effect is a thing
/// that CATCHES, and a patch of terrain that reaches full brightness in a tenth of a second reads
/// as a light being switched on rather than as a fire spreading over it.
pub(crate) const SPELL_AREA_RAMP: f32 = 0.25;

/// MONKEY (area spell light): how fast an area light goes dark once the DynamicObject it belongs to
/// is destroyed. Slower than the kit's [`SPELL_REAP_FADE`] because the thing ending is bigger — a
/// 24 yd pool snapping off in a quarter second is a visible pop across a whole clearing — and still
/// far inside the ~2 s the anchor's own teardown fade takes, so the light is long gone before the
/// burning decal it lit has finished fading out.
pub(crate) const SPELL_AREA_FADE: f32 = 0.4;

/// MONKEY (area spell light): the area light's BREATHING — depth (fraction of base) and rate (Hz).
///
/// Explicitly NOT [`FlameFlicker`](benilla_world::lighting::FlameFlicker), which is the fire lane's
/// fast noisy wobble: at 8 % and 0.7 Hz this is a slow swell, under the threshold where the eye
/// reads it as the light *changing* rather than as the fire on the ground being alive. Something is
/// needed, though — an 8-second Flamestrike patch whose light is a perfectly constant disc is the
/// one thing that gives the whole effect away as a light source, and the model under it is
/// animating the entire time.
pub(crate) const SPELL_AREA_BREATH_DEPTH: f32 = 0.08;
/// See [`SPELL_AREA_BREATH_DEPTH`].
pub(crate) const SPELL_AREA_BREATH_HZ: f32 = 0.7;

/// How many spell lights may be alive at once, across every lane. The shared point-light table is
/// 512 rows packed from the 256 nearest sources every frame, and the whole feature is worth
/// nothing if a raid's worth of casts can evict a city's torches: 24 is a generous ceiling for
/// what one camera can see cast at once and still a small fraction of the pack.
pub(crate) const SPELL_LIGHTS_MAX: usize = 24;

/// MONKEY (spell light): the env kill switch, `WOW_SPELL_LIGHT=0`. Read ONCE — the value is
/// latched on first call, so it cannot change under a running frame and costs one atomic load at
/// each spawn site thereafter.
///
/// An env var and STILL not a cvar, now that `spellLightGain` exists beside `fireLightGain`
/// (MONKEY (spellLightGain) — `cvars.rs`, folded at pack time in
/// `benilla_world::lighting::global_light::build_light_data`). The two are different tools and both
/// are wanted: the cvar is the live BRIGHTNESS dial, and its `0` darkens a spell light that is
/// still spawned, still parented, still aged and still counted against
/// [`SPELL_LIGHTS_MAX`]; this switch stops the lights being CREATED at all, which is what an A/B
/// against the pre-feature build needs (`WOW_SPELL_LIGHT=0` costs the frame nothing, and takes the
/// claim walk in `carried_light::claim_carried_light_rooms` with it).
pub(crate) fn spell_lights_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("WOW_SPELL_LIGHT").as_deref(), Ok("0")))
}

/// Which lifecycle a spell light follows — the one thing its spawn site knows that the model
/// cannot.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum SpellLightMode {
    /// A kit/aura effect attached to a unit: up on the ramp, HELD for as long as the instance
    /// stands, out on the reap ([`FxDecay`] → [`SPELL_REAP_FADE`]). Ice Barrier's shield holds,
    /// Immolate burns for its duration, a precast glows while it is cast.
    Kit,
    /// A missile: constant while it flies. The light rides the projectile root, so it sweeps the
    /// ground and the walls on the way — and the missile's own arrival despawn ends it, which is
    /// the only ending a projectile has.
    Missile,
    /// An impact / destination effect / firework shell: full at onset, gone `span` seconds later.
    /// The one shape whose *whole point* is that it does not persist.
    Burst { span: f32 },
    /// MONKEY (area spell light): a PERSISTENT GROUND EFFECT — one light for a whole
    /// DynamicObject's life (Flamestrike's burning patch, Rain of Fire's storm, Consecration's
    /// ground, a hunter's Flare). Up on the slower [`SPELL_AREA_RAMP`], held with a slow
    /// [`SPELL_AREA_BREATH_DEPTH`] swell for as long as the area object stands, out over
    /// [`SPELL_AREA_FADE`] when the server destroys it.
    ///
    /// It is a mode of its own rather than a [`Self::Kit`] with a different fade because the two
    /// differ in every part: the ending is driven by a `DespawnFade` on a net entity instead of an
    /// `FxDecay` on an effect instance, the hold is modulated, and the ceiling treats it as the one
    /// spell light that must NOT be evicted (see [`budget_spell_lights`]) — a strobe of impact
    /// flashes must never be able to push the standing pool out of the table.
    ///
    /// `phase` staggers the breathing per light (seeded from the anchor), so two overlapping
    /// patches swell out of step rather than pulsing the clearing in unison.
    Area { phase: f32 },
}

impl SpellLightMode {
    /// Seconds this mode takes to come up.
    fn ramp(self) -> f32 {
        match self {
            Self::Area { .. } => SPELL_AREA_RAMP,
            _ => SPELL_LIGHT_RAMP,
        }
    }

    /// Seconds this mode takes to go dark once its host has been reaped.
    fn reap_fade(self) -> f32 {
        match self {
            Self::Area { .. } => SPELL_AREA_FADE,
            _ => SPELL_REAP_FADE,
        }
    }
}

/// MONKEY (area spell light): the marker on a live area light, carrying the AoE footprint
/// (`DYNAMICOBJECT_RADIUS`, yd) the wire gave it.
///
/// Two readers, and neither wants it on [`SpellLight`] itself:
/// - the DEDUPE ([`super::super::dest_fx`]) — a repeating impact that lands inside a live area
///   light's footprint spawns no light of its own, which is the whole of why Rain of Fire is one
///   steady pool instead of five flashes a second;
/// - the BUDGET below, which evicts area lights last.
#[derive(Component)]
pub(crate) struct AreaSpellLight {
    /// The wire AoE radius (yd), horizontal — the dedupe's own test radius.
    pub(crate) radius: f32,
}

/// MONKEY (spell light): one live spell light. A child of the effect root it belongs to, so the
/// effect's own despawn reaps it; this component only shapes its brightness while it lives.
#[derive(Component)]
pub(crate) struct SpellLight {
    /// The `PointLight::intensity` this light sits at when the envelope is at full — captured at
    /// spawn from the synthesised colour/intensity, never recomputed. The envelope only ever
    /// SCALES this, so nothing downstream can drift the calibration.
    base: f32,
    /// Seconds from spawn before the light comes up — the emitter's own rate-track onset
    /// (`benilla_formats::fire_light::emit_onset`). A firework shell detonates half a second into
    /// its model's clip and a rocket that lit the sky on the way UP would be exactly backwards.
    onset: f32,
    /// The lifecycle ([`SpellLightMode`]).
    mode: SpellLightMode,
    /// Seconds alive. Also the budget's age ordering — the OLDEST spell light is the one dropped
    /// when the ceiling is hit, because it is the one whose moment has most passed.
    age: f32,
    /// Seconds this light's HOST instance has been reaped ([`FxDecay`]), i.e. how far into
    /// [`SPELL_REAP_FADE`] it is. Only ticks in [`SpellLightMode::Kit`].
    reaped: f32,
}

impl SpellLight {
    /// A fresh light at `base` intensity, coming up `onset` seconds from now.
    pub(crate) fn new(base: f32, onset: f32, mode: SpellLightMode) -> Self {
        Self {
            base,
            onset,
            mode,
            age: 0.0,
            reaped: 0.0,
        }
    }

    /// The envelope's current multiplier, `0..=1` — pure, so the shape is testable without a
    /// world. Ramp in, then the mode's own decay, then the reap fade, multiplied.
    fn envelope(&self) -> f32 {
        let t = self.age - self.onset;
        if t <= 0.0 {
            return 0.0; // the flame this light stands for does not exist yet
        }
        let ramp = self.mode.ramp();
        let up = (t / ramp).min(1.0);
        let down = match self.mode {
            // The fade starts where the ramp ended, so a burst's peak is a real (if brief) plateau
            // rather than a single-frame spike that a low frame rate could skip entirely.
            SpellLightMode::Burst { span } => {
                (1.0 - (t - ramp).max(0.0) / span.max(1e-3)).clamp(0.0, 1.0)
            }
            // MONKEY (area spell light): a hold that BREATHES. Deliberately allowed above 1.0 —
            // the swell is centred on the calibrated base, so clamping the top half would turn a
            // symmetric ±8 % into a one-sided dimming and leave every area light reading 4 %
            // darker than the rung it was filed on.
            SpellLightMode::Area { phase } => {
                1.0 + SPELL_AREA_BREATH_DEPTH
                    * (std::f32::consts::TAU * SPELL_AREA_BREATH_HZ * t + phase).sin()
            }
            SpellLightMode::Kit | SpellLightMode::Missile => 1.0,
        };
        let reap = (1.0 - self.reaped / self.mode.reap_fade()).clamp(0.0, 1.0);
        up * down * reap
    }

    /// MONKEY (area spell light): has this light finished going dark after its host was reaped?
    /// Only an AREA light asks — see the despawn note in [`advance_spell_lights`].
    fn reap_complete(&self) -> bool {
        self.reaped >= self.mode.reap_fade()
    }
}

/// MONKEY (spell light): run every live spell light's envelope.
///
/// `Update`, beside the rest of the effect lane, and it writes `PointLight::intensity` ONLY — the
/// packer reads that in `PostUpdate` (`lighting::build_light_data`), so a light's brightness is
/// always this frame's. Deliberately NOT a `FlameFlicker`: that modulation exists to make a fire
/// breathe, and a spell light already has a shape of its own; the two together read as the effect
/// stuttering.
///
/// The KIT arm is the only one that has to look outside itself: a reaped instance
/// ([`FxDecay`] on the effect root, written by the reap) must take its light down with it, and the
/// light's parent IS that root.
pub(crate) fn advance_spell_lights(
    mut commands: Commands,
    time: Res<Time>,
    mut lights: Query<(
        Entity,
        &mut SpellLight,
        &mut WorldPointLight,
        Option<&ChildOf>,
    )>,
    // MONKEY (area spell light): the reap signal now has TWO sources, one per persistent mode.
    // `FxDecay` is written on an effect INSTANCE by the kit lane's own reap; `DespawnFade` is
    // written on a NET ENTITY by `SMSG_DESTROY_OBJECT` (`net::apply::objects`), which is the only
    // notice a DynamicObject's end ever gives. That the anchor fades rather than popping is what
    // makes an area fade possible at all: the child light outlives the server's destroy by the
    // ~2 s the anchor's geometry teardown takes, and needs only 0.4 of them.
    reaped: Query<(Has<FxDecay>, Has<benilla_world::model_fade::DespawnFade>)>,
) {
    let dt = time.delta_secs();
    for (entity, mut light, mut point, parent) in &mut lights {
        light.age += dt;
        let host = parent.and_then(|c| reaped.get(c.parent()).ok());
        let ended = match light.mode {
            SpellLightMode::Kit => host.is_some_and(|(decay, _)| decay),
            SpellLightMode::Area { .. } => host.is_some_and(|(_, fade)| fade),
            SpellLightMode::Missile | SpellLightMode::Burst { .. } => false,
        };
        if ended {
            light.reaped += dt;
            // An area light's anchor lingers for its own teardown fade, several times longer than
            // the light's. Give the slot back the moment the pool is dark rather than holding one
            // of the 24 for a light nobody can see — the kit lane has no such gap, because its
            // instance despawns on the heels of its `FxDecay`.
            if matches!(light.mode, SpellLightMode::Area { .. }) && light.reap_complete() {
                commands.entity(entity).try_despawn();
            }
        }
        let want = light.base * light.envelope();
        // Write only on a real change: a held kit light sits at its plateau for the whole aura,
        // and a per-frame write there would wake every change-detection consumer of `PointLight`
        // for nothing.
        if (point.intensity - want).abs() > 1e-3 {
            point.intensity = want;
        }
    }
}

/// MONKEY (spell light): the ceiling ([`SPELL_LIGHTS_MAX`]) — drop the OLDEST beyond it.
///
/// Oldest rather than dimmest or furthest, for one reason: the lane exists to light what is
/// HAPPENING, and the newest cast is the one the player is looking at. A dimness rule would evict
/// the burst that is mid-fade (i.e. exactly the flash being watched) and a distance rule would
/// fight the packer, which already sorts by distance and would then be given a hole it had no say
/// in. Age is also the only ordering that is stable frame to frame, so the set does not churn.
///
/// Despawning the light alone never disturbs its effect: the light is a leaf child of the effect
/// root and nothing reads back from it.
pub(crate) fn budget_spell_lights(
    mut commands: Commands,
    lights: Query<(Entity, &SpellLight, Has<AreaSpellLight>)>,
) {
    let live = lights.iter().count();
    if live <= SPELL_LIGHTS_MAX {
        return;
    }
    let mut order: Vec<(Entity, bool, f32)> =
        lights.iter().map(|(e, s, area)| (e, area, s.age)).collect();
    // MONKEY (area spell light): AREA lights go LAST, whatever their age — and an area light is
    // always among the oldest, because it is the only spell light that stands for seconds. Under
    // pure age ordering a Rain of Fire's own impact flashes would evict the very pool they are
    // landing in, i.e. the ceiling would delete the steady light and keep the strobe. Within each
    // class the rule is unchanged: oldest first, for the reasons above.
    order.sort_by(|a, b| a.1.cmp(&b.1).then(b.2.total_cmp(&a.2)));
    for (light, _, _) in order.into_iter().take(live - SPELL_LIGHTS_MAX) {
        commands.entity(light).try_despawn();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use benilla_assets::AnimClip;
    use bevy::animation::RepeatAnimation;

    /// One clip of `anim_id` with the M2 loop flag `looping`, on graph node `node`.
    fn clip(anim_id: u16, node: usize, looping: bool) -> AnimClip {
        AnimClip {
            anim_id,
            seq_index: node,
            node: AnimationNodeIndex::new(node),
            looping,
            duration: 1.0,
            move_speed: 0.0,
            blend_time: 0.0,
            bounds_center: Vec3::ZERO,
            bounds_radius: 0.0,
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::ZERO,
            events: Vec::new().into(),
            arm_nodes: None,
            upper_node: None,
            frequency: 0,
            replay: (0, 0),
            poses_bones: true,
        }
    }

    fn repeat(player: &AnimationPlayer, node: usize) -> RepeatAnimation {
        player
            .animation(AnimationNodeIndex::new(node))
            .expect("armed")
            .repeat_mode()
    }

    /// `IceShield_State`'s birth clamps: armed unrepeated, watching for `0x5ff170`'s handover.
    #[test]
    fn state_arms_the_birth_and_watches_for_the_handover() {
        let mut player = AnimationPlayer::default();
        let life = FxAnimLife::arm(&mut player, &clip(0, 0, false), FxStage::State);
        assert!(matches!(life, FxAnimLife::Birth(n) if n == AnimationNodeIndex::new(0)));
        assert_eq!(repeat(&player, 0), RepeatAnimation::Never);
    }

    /// `0x60ed00` re-arms without reading the clamp bit, so a clamping precast repeats.
    #[test]
    fn relive_repeats_even_a_clamping_sequence() {
        let mut player = AnimationPlayer::default();
        let life = FxAnimLife::arm(&mut player, &clip(0, 0, false), FxStage::Relive);
        assert!(matches!(life, FxAnimLife::Settled(_)));
        assert_eq!(repeat(&player, 0), RepeatAnimation::Forever);
    }

    /// The clamp bit decides; a one-shot ends on the instance's span clock, not here.
    #[test]
    fn oneshot_follows_the_sequence_flag_only() {
        let mut clamped = AnimationPlayer::default();
        let life = FxAnimLife::arm(&mut clamped, &clip(0, 0, false), FxStage::OneShot);
        assert!(matches!(life, FxAnimLife::Settled(_)));
        assert_eq!(repeat(&clamped, 0), RepeatAnimation::Never);

        let mut wrapping = AnimationPlayer::default();
        FxAnimLife::arm(&mut wrapping, &clip(0, 0, true), FxStage::OneShot);
        assert_eq!(repeat(&wrapping, 0), RepeatAnimation::Forever);
    }

    /// The reap's gate (`0x6141a1`): no `Decay` is `None`, the straight-to-destructor branch.
    #[test]
    fn decay_span_is_the_gate_and_the_lifetime() {
        let with = test_anims(&[clip(0, 0, false), clip(ANIM_HOLD, 1, true), {
            let mut c = clip(ANIM_DECAY, 2, false);
            c.duration = 1.1;
            c
        }]);
        assert_eq!(decay_span(Some(&with)), Some(1.1));

        let without = test_anims(&[clip(0, 0, false), clip(ANIM_HOLD, 1, true)]);
        assert_eq!(decay_span(Some(&without)), None);
        assert_eq!(decay_span(None), None);
    }

    /// MONKEY (spell light) — GOLDEN: the envelope's four legs, on the pure function so the shape
    /// is pinned without a world. The ONSET leg is the one with a real bug behind it: a firework
    /// shell detonates half a second into its clip, and a light lit at spawn flashes the rocket's
    /// flight instead of its burst.
    #[test]
    fn the_spell_envelope_ramps_holds_and_bursts() {
        let at = |mut l: SpellLight, age: f32| {
            l.age = age;
            l.envelope()
        };

        // A kit light: dark before its onset, ramped over SPELL_LIGHT_RAMP, then HELD.
        let kit = || SpellLight::new(100.0, 0.0, SpellLightMode::Kit);
        assert_eq!(at(kit(), 0.0), 0.0, "nothing on the spawn frame");
        assert!((at(kit(), SPELL_LIGHT_RAMP * 0.5) - 0.5).abs() < 1e-3);
        assert_eq!(at(kit(), SPELL_LIGHT_RAMP), 1.0);
        assert_eq!(
            at(kit(), 30.0),
            1.0,
            "an aura holds for as long as it stands"
        );

        // A missile is the same minus any ending of its own — the arrival despawn is the ending.
        assert_eq!(
            at(SpellLight::new(1.0, 0.0, SpellLightMode::Missile), 5.0),
            1.0
        );

        // A burst peaks at the ramp's end and is gone `span` later; the plateau is real, not a
        // one-frame spike a low frame rate could step over.
        let burst = || SpellLight::new(1.0, 0.0, SpellLightMode::Burst { span: 0.6 });
        assert_eq!(at(burst(), SPELL_LIGHT_RAMP), 1.0);
        assert!((at(burst(), SPELL_LIGHT_RAMP + 0.3) - 0.5).abs() < 1e-3);
        assert_eq!(at(burst(), SPELL_LIGHT_RAMP + 0.6), 0.0);
        assert_eq!(at(burst(), 10.0), 0.0, "and stays gone");

        // The ONSET: a firework's fuse. Dark for the whole flight, up at the detonation.
        let fuse = || SpellLight::new(1.0, 1.5, SpellLightMode::Burst { span: 0.6 });
        assert_eq!(at(fuse(), 1.4), 0.0);
        // MONKEY (integration): (1.5 + ramp) - 1.5 is not exactly `ramp` in f32.
        assert!((at(fuse(), 1.5 + SPELL_LIGHT_RAMP) - 1.0).abs() < 1e-6);

        // The reap fade: a kit light goes dark on its instance's `FxDecay`, quicker than the
        // model's own Decay sequence — the aura is over the moment the server said so.
        let mut reaped = kit();
        reaped.age = 5.0;
        reaped.reaped = SPELL_REAP_FADE * 0.5;
        assert!((reaped.envelope() - 0.5).abs() < 1e-3);
        reaped.reaped = SPELL_REAP_FADE;
        assert_eq!(reaped.envelope(), 0.0);
    }

    /// MONKEY (area spell light) — GOLDEN: the AREA envelope's four legs, which differ from every
    /// other mode's in all four. The slower ramp, the BREATHING hold (the leg that exists so an
    /// 8-second Flamestrike patch is not a frozen disc), and the slower removal fade that the
    /// anchor's own ~2 s teardown leaves room for.
    #[test]
    fn an_area_light_ramps_slowly_breathes_and_fades_out() {
        let at = |mut l: SpellLight, age: f32| {
            l.age = age;
            l.envelope()
        };
        let area = || SpellLight::new(100.0, 0.0, SpellLightMode::Area { phase: 0.0 });
        // The swell is a factor on the WHOLE envelope, ramp included — it is what the light is
        // doing, not a decoration on the hold — so the expected values carry it.
        let swell = |t: f32| {
            1.0 + SPELL_AREA_BREATH_DEPTH * (std::f32::consts::TAU * SPELL_AREA_BREATH_HZ * t).sin()
        };

        // The ramp is the AREA one — at the flash ramp's 0.1 s an area light is well short of up.
        assert!(at(area(), SPELL_LIGHT_RAMP) < 0.5, "not a flash");
        let half = SPELL_AREA_RAMP * 0.5;
        assert!((at(area(), half) - 0.5 * swell(half)).abs() < 1e-3);
        assert!((at(area(), SPELL_AREA_RAMP) - swell(SPELL_AREA_RAMP)).abs() < 1e-3);

        // The hold BREATHES: a full period later it is back at its base, a quarter period past
        // that it is at the top of the swell, and it never wanders outside ±depth. (Every mark
        // below is past the ramp, so `up` is 1 and the swell is the whole of the value.)
        let period = 1.0 / SPELL_AREA_BREATH_HZ;
        assert!(
            (at(area(), period) - 1.0).abs() < 1e-3,
            "one period, back to base"
        );
        assert!(
            (at(area(), period * 1.25) - (1.0 + SPELL_AREA_BREATH_DEPTH)).abs() < 1e-3,
            "the swell is centred on the base, not clamped below it"
        );
        assert!((at(area(), period * 1.75) - (1.0 - SPELL_AREA_BREATH_DEPTH)).abs() < 1e-3);
        for step in 0..200 {
            let v = at(area(), 1.0 + step as f32 * 0.05);
            assert!(
                (1.0 - SPELL_AREA_BREATH_DEPTH - 1e-3..=1.0 + SPELL_AREA_BREATH_DEPTH + 1e-3)
                    .contains(&v),
                "the hold never drifts off its base: {v}"
            );
        }

        // Removal: the area fade, not the kit's — and `reap_complete` is what hands the budget
        // slot back rather than waiting out the anchor's own teardown.
        let mut ending = area();
        ending.age = 8.0;
        ending.reaped = SPELL_AREA_FADE * 0.5;
        assert!(!ending.reap_complete());
        assert!(ending.envelope() < 0.55 && ending.envelope() > 0.45);
        ending.reaped = SPELL_AREA_FADE;
        assert_eq!(ending.envelope(), 0.0);
        assert!(ending.reap_complete());
        // A KIT light of the same age is NOT finished at the area fade's length — the two clocks
        // are genuinely separate.
        let mut kit_ending = SpellLight::new(1.0, 0.0, SpellLightMode::Kit);
        kit_ending.reaped = SPELL_REAP_FADE * 0.5;
        assert!(!kit_ending.reap_complete());
    }

    fn test_anims(clips: &[AnimClip]) -> ModelAnimations {
        ModelAnimations {
            graph: Handle::default(),
            clips: clips.to_vec(),
            hand_close: [None, None],
            playable_animation_lookup: Vec::new(),
            animation_lookup: Vec::new(),
            global_bones: Vec::new(),
            first_seq: None,
            pose: Default::default(),
        }
    }
}
