//! MONKEY (dynamic interiors): the app-side half of the fixture-lit WMO interior lane — a
//! plug-and-play bridge from the `interiorLight` / `interiorAmbient` / `interiorFill` /
//! `interiorExposure` cvars ([`VideoConfig`]) to benilla-world's [`DynamicInteriors`] resource,
//! which `global_light` packs for `static_gx.wgsl`'s `interior_room_light` (WMO interior surfaces
//! AND interior props light from the room's live fixtures instead of the MOCV bake / the baked prop
//! probe). Independent of the shadow lanes: removing this plugin leaves the resource at its
//! shipped defaults; `interiorLight 0` restores the faithful baked interior path.
//!
//! The knobs are LIVE — `/script SetCVar("interiorExposure", 2)` in chat retunes the room without
//! a rebuild, which is how the defaults were found.
//!
//! MONKEY (fire GO lights): the same bridge also carries `fireLightGain` →
//! [`benilla_world::lighting::FireLightGain`]. Same shape, same guard, same liveness requirement —
//! and `fireLightGain 0` is the kill switch for the synthesised-fire lane the way `interiorLight 0`
//! is for this one.
//!
//! MONKEY (spellLightGain): and `spellLightGain` → [`benilla_world::lighting::SpellLightGain`],
//! the third of the same shape.
//!
//! MONKEY (moon shadows): and `moonShadowStrength` →
//! [`benilla_world::lighting::MoonShadowStrength`], the fourth. Same shape, same guard, same
//! liveness requirement — and `0` is that lane's faithful null the way `fireLightGain 0` is the
//! synthesised-fire lane's kill switch. It is what the Advanced Graphics page's Moon Shadows row
//! and the Off/Low presets reach.
//!
//! Water quality and lava glow use the same guarded bridge into their renderer resources.
use benilla_assets::WaterQuality;
use benilla_world::lighting::{
    DynamicInteriors, EmissiveTier, FireLightGain, LavaLightGain, MoonShadowStrength,
    SpellLightGain,
};
use bevy::prelude::*;

use crate::video::VideoConfig;

pub(crate) struct DynamicInteriorPlugin;

impl Plugin for DynamicInteriorPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, bridge);
    }
}

/// Publish the cvars to benilla-world when they change. Guarded so the resource's change ticks
/// stay quiet on idle frames.
///
/// MONKEY (fire GO lights): `fireLightGain` rides along here rather than in a module of its own —
/// it is the same shape (a live video cvar → a benilla-world resource the light packer reads) and
/// the same one-line guard, and a second plugin to carry one `f32` is not worth its wiring.
fn bridge(
    video: Res<VideoConfig>,
    mut out: ResMut<DynamicInteriors>,
    mut fire: ResMut<FireLightGain>,
    // MONKEY (spellLightGain): the spell lane's own gain rides the same bridge as the fire gain,
    // for the same reasons — same shape (a live video cvar → a benilla-world resource the packer
    // reads), same one-line guard, and a third plugin to carry a second `f32` is not worth its
    // wiring. Kept a SEPARATE resource rather than a field on `DynamicInteriors` because it is not
    // an interior knob at all: a fireball down a street takes it exactly as one down a corridor
    // does, and `FireLightGain` beside it already stands as the precedent for that.
    mut spell: ResMut<SpellLightGain>,
    // MONKEY (moon shadows): the night lane's shadow darkness, on the same bridge and for the same
    // reasons as the two gains above it — one live `f32`, one benilla-world resource, one guard.
    mut moon: ResMut<MoonShadowStrength>,
    mut water: ResMut<WaterQuality>,
    mut lava: ResMut<LavaLightGain>,
    // MONKEY (post): bloom also arms the HDR source hooks through the shared light blob.
    mut emissive: ResMut<EmissiveTier>,
    // GFX (moonlight): the additive moon term's dial (Option: absent in headless tests).
    moon_light: Option<ResMut<benilla_world::lighting::MoonLight>>,
) {
    if let Some(mut moon_light) = moon_light {
        if moon_light.0 != video.moon_light {
            moon_light.0 = video.moon_light;
        }
    }
    if water.0 != video.water_quality {
        water.0 = video.water_quality;
    }
    if lava.0 != video.lava_light_gain {
        lava.0 = video.lava_light_gain;
    }
    if emissive.0 != video.bloom {
        emissive.0 = video.bloom;
    }
    if fire.0 != video.fire_light_gain {
        fire.0 = video.fire_light_gain;
    }
    if spell.0 != video.spell_light_gain {
        spell.0 = video.spell_light_gain;
    }
    if moon.0 != video.moon_shadow_strength {
        moon.0 = video.moon_shadow_strength;
    }
    // MONKEY (fix-daylight): `daylightWindowSplit` (an atomic store; seeds read it at load).
    benilla_world::lighting::set_window_split(video.daylight_window_split);
    let want = DynamicInteriors {
        enabled: video.interior_light,
        ambient: video.interior_ambient,
        fill: video.interior_fill,
        exposure: video.interior_exposure,
        // MONKEY (interior attenuation): the authored-window scale rides the same bridge — the
        // light packer folds it into every interior entry's packed reach, so
        // `SetCVar("interiorAttenScale", 2)` widens every pool on the very next frame.
        atten_scale: video.interior_atten_scale,
        // MONKEY (room gate): the packer turns this into a per-light claim head, so
        // `SetCVar("interiorRoomGate", 0)` restores the ungated look on the very next frame.
        room_gate: video.interior_room_gate,
        debug: video.interior_debug,
        // MONKEY (flame flicker): the wobble gain rides the same bridge — it is consumed entirely
        // on the CPU in `build_light_data`, so `SetCVar("fireFlicker", 0)` freezes every flame on
        // the very next frame with no respawn and no shader change.
        flicker: video.fire_flicker,
        // MONKEY (darkness gains): both dim dials ride this same bridge — they are consumed
        // entirely on the CPU in `build_light_data` (folded into the packed rows / the packed
        // colours), so `SetCVar("nightGain", 1)` restores the reference night on the very next
        // frame with no respawn and no shader change. `night_gain` is an EXTERIOR knob living on an
        // interior-named resource for the sake of one bridge rather than two; the packer reads it
        // ungated by `enabled`.
        night_gain: video.night_gain,
        interior_gain: video.interior_gain,
        // MONKEY (enclosed day floor): the daylight floor rides the same bridge, but is consumed in
        // the SHADER (it rides the packed `wmo_fog_params.w` fraction) rather than folded on the
        // CPU — the packer's only job is to put it in the lane.
        // MONKEY (daylight): `WOW_INTERIOR_DAYLIGHT=<0..1>` overrides it in a hermetic capture,
        // which reads no config.toml (dev builds only; read once).
        daylight: capture_daylight().unwrap_or(video.interior_daylight),
        // MONKEY (bake floor): the bake floor rides the same bridge. Half CPU, half shader: the
        // packer folds `interiorGain` in and puts the product in the `sh_c16.w` fraction, and the
        // two interior lanes read it from there — so `SetCVar("interiorBakeFloor", 0)` restores
        // the pre-feature look on the very next frame with no respawn and no pipeline rebuild.
        bake_floor: video.interior_bake_floor,
    };
    if *out != want {
        *out = want;
    }
}

/// MONKEY (daylight): the capture-only `interiorDaylight` override (see the bridge above).
fn capture_daylight() -> Option<f32> {
    static V: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        // MONKEY (integration): the dev door is `run_mode` (decision 1179).
        if !crate::run_mode::dev_affordances() || std::env::var_os("WOW_CAPTURE").is_none() {
            return None;
        }
        std::env::var("WOW_INTERIOR_DAYLIGHT")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|v| v.is_finite())
            .map(|v| v.clamp(0.0, 1.0))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_and_lava_bridge_publishes_changes_without_dirtying_idle_frames() {
        let mut app = App::new();
        app.init_resource::<VideoConfig>()
            .init_resource::<DynamicInteriors>()
            .init_resource::<FireLightGain>()
            .init_resource::<SpellLightGain>()
            .init_resource::<MoonShadowStrength>()
            .init_resource::<WaterQuality>()
            .init_resource::<LavaLightGain>()
            .init_resource::<EmissiveTier>()
            .add_plugins(DynamicInteriorPlugin);
        assert_eq!(
            app.world().resource::<WaterQuality>().0,
            VideoConfig::default().water_quality
        );
        assert_eq!(
            app.world().resource::<LavaLightGain>().0,
            VideoConfig::default().lava_light_gain
        );
        app.update();
        for (water, lava) in [(0, 0.0), (2, 3.25), (1, 1.0)] {
            {
                let mut video = app.world_mut().resource_mut::<VideoConfig>();
                video.water_quality = water;
                video.lava_light_gain = lava;
            }
            app.world_mut().run_schedule(Update);
            assert_eq!(app.world().resource::<WaterQuality>().0, water);
            assert_eq!(app.world().resource::<LavaLightGain>().0, lava);
            app.world_mut().clear_trackers();
            app.world_mut().run_schedule(Update);
            assert!(!app.world().resource_ref::<WaterQuality>().is_changed());
            assert!(!app.world().resource_ref::<LavaLightGain>().is_changed());
        }
    }
}
