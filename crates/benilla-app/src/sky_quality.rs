//! MONKEY (sky): the `skyQuality` cvar → [`benilla_world::sky_fx::SkyQuality`] bridge.
//!
//! 0 Classic (the reference's sky, unchanged), 1 Enhanced (smooth gradient, sun glow, night sky),
//! 2 High (+ detailed, sun-lit clouds). `$WOW_SKY_QUALITY` pins the tier for the session, so
//! captures stay independent of `config.toml`.

use benilla_world::sky_fx::SkyQuality;
use bevy::prelude::*;

use crate::video::VideoConfig;

pub(crate) struct SkyQualityPlugin;

impl Plugin for SkyQualityPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SkyQuality>().add_systems(
            Update,
            // MONKEY (integration): the tier lands before the sky and cloud readers.
            bridge.before(benilla_world::lighting::LightingConsumeSet),
        );
    }
}

fn bridge(
    video: Res<VideoConfig>,
    mut out: ResMut<SkyQuality>,
    mut pinned: Local<Option<Option<u8>>>,
) {
    let pin = *pinned.get_or_insert_with(SkyQuality::env_override);
    let want = SkyQuality(pin.unwrap_or(video.sky_quality).min(SkyQuality::MAX));
    if *out != want {
        *out = want;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_publishes_the_tier_without_dirtying_idle_frames() {
        if SkyQuality::env_override().is_some() {
            return; // pinned by the environment; nothing to observe
        }
        let mut app = App::new();
        app.init_resource::<VideoConfig>()
            .add_plugins(SkyQualityPlugin);
        app.update();
        assert_eq!(
            app.world().resource::<SkyQuality>().0,
            VideoConfig::default().sky_quality
        );
        for tier in [2u8, 1, 0] {
            app.world_mut().resource_mut::<VideoConfig>().sky_quality = tier;
            app.update();
            assert_eq!(app.world().resource::<SkyQuality>().0, tier);
            app.update();
            assert!(!app.world().resource_ref::<SkyQuality>().is_changed());
        }
    }
}
