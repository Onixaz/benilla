//! MONKEY (p0 graphics programme): the cvar bridges of the graphics programme's own settings, each
//! landing on the benilla-world resource its renderer reads. One observer, one arm per cvar;
//! lanes add their rows here rather than to `video.rs`'s reference block.
//!
//! | cvar | values | default | Graphics preset High | resource |
//! |---|---|---|---|---|
//! | `skyDither` | 0 Off / 1 On | 0 | 1 | `benilla_world::ffx_glow::SkyDither` |
//! | `foliageWind` | 0 Off / 1 Grass / 2 Grass + trees | 2 | 2 | `benilla_world::wind::FoliageWind` |
//! | `foliageWindStrength` | 0.25..3 sway gain | 1 | not governed | `benilla_world::wind::FoliageWindStrength` |
//! | `fogModel` | 0 Classic / 1 Modern | 0 | 1 | `benilla_world::lighting::FogModelSetting` |
//! | `rainSurfaces` | 0 Off / 1 On | 1 | 1 | `benilla_world::weather::RainSurfaces` |

use benilla_world::ffx_glow::SkyDither;
use benilla_world::lighting::FogModelSetting;
use benilla_world::weather::RainSurfaces;
use benilla_world::wind::{FoliageWind, FoliageWindStrength};
use bevy::prelude::*; // MONKEY (wet)

/// The programme's cvar observer (registered by [`MonkeyGfxPlugin`]).
pub(crate) fn on_cvar(
    ev: On<crate::cvars::CvarChanged>,
    mut dither: ResMut<SkyDither>,
    mut foliage_wind: ResMut<FoliageWind>,
    mut wind_strength: ResMut<FoliageWindStrength>, // MONKEY (fix-wind)
    // MONKEY (fog)
    mut fog_model: ResMut<FogModelSetting>,
    mut rain: ResMut<RainSurfaces>, // MONKEY (wet)
) {
    match ev.key().as_str() {
        "skydither" => {
            let want = ev.num() >= 0.5;
            if dither.0 != want {
                dither.0 = want;
            }
        }
        // MONKEY (wind): numeric tier, live; 0 is the exact no-displacement path.
        "foliagewind" => {
            // Capture-only A/B pin (MONKEY reviewfix-a: gated like `capture_daylight` — dev
            // build AND `WOW_CAPTURE`). Player runs obey the live CVar exactly.
            let requested = foliage_wind_pin()
                .map(f32::from)
                .unwrap_or_else(|| ev.num());
            let want = requested.clamp(0.0, 2.0) as u8;
            if foliage_wind.0 != want {
                foliage_wind.0 = want;
            }
        }
        // MONKEY (fix-wind): the sway gain, live, clamped to the slider's 0.25..3.
        "foliagewindstrength" => {
            let want = FoliageWindStrength::from_cvar(ev.num());
            if *wind_strength != want {
                *wind_strength = want;
            }
        }
        // MONKEY (fog): `WOW_FOGMODEL` still wins for the session (captures).
        "fogmodel" => fog_model.set_from_cvar(ev.num()),
        // MONKEY (wet): wet surfaces + rain ripples.
        "rainsurfaces" => {
            let want = ev.num() >= 0.5;
            if rain.0 != want {
                rain.0 = want;
            }
        }
        _ => {}
    }
}

/// MONKEY (reviewfix-a): the `WOW_FOLIAGE_WIND` capture pin, only in a dev build under `WOW_CAPTURE`.
fn foliage_wind_pin() -> Option<u8> {
    if !crate::run_mode::dev_affordances() {
        return None;
    }
    FoliageWind::capture_override()
}

/// Registers the programme's cvar bridges. Must be added before `CvarPlugin` so the saved values
/// applied at Startup reach it (the edge `game_plugins.rs` documents for every knob plugin).
pub(crate) struct MonkeyGfxPlugin;

impl Plugin for MonkeyGfxPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SkyDither>()
            // MONKEY (reviewfix-a): the resource is seeded here so a player build never takes the
            // capture pin (the world plugin's `init_resource` keeps whatever is already present).
            .insert_resource(FoliageWind(
                foliage_wind_pin().unwrap_or(FoliageWind::REGISTERED),
            ))
            .init_resource::<FoliageWindStrength>() // MONKEY (fix-wind)
            .init_resource::<FogModelSetting>()
            .init_resource::<RainSurfaces>() // MONKEY (wet)
            .add_observer(on_cvar);
    }
}
