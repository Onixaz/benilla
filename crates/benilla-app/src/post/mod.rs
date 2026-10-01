//! MONKEY (post): optional world-camera effects ahead of FFXGlow's gamma clamp.

mod bloom;
// MONKEY (visualfix): crate-visible for `rendered_minute` (volumetric fog).
pub(crate) mod grading;
mod sun_shafts;

pub(crate) struct PostPlugin;

impl bevy::prelude::Plugin for PostPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_plugins(bloom::BloomPlugin);
        app.add_plugins(grading::GradingPlugin);
        app.add_plugins(sun_shafts::SunShaftsPlugin);
    }
}
