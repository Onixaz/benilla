//! MONKEY (skybox): the `zoneSkyboxes` cvar's bridge to [`benilla_world::skybox::ZoneSkyboxes`].
//! `WOW_ZONE_SKYBOXES=0|1` overrides it for the session, which is how a hermetic capture (no config
//! file, registry defaults) turns the lane on.

use bevy::prelude::*;

/// The session override, read once.
#[derive(Resource, Clone, Copy)]
pub(crate) struct ZoneSkyboxOverride(pub(crate) Option<bool>); // MONKEY (reviewfix-a): test census

pub(crate) struct ZoneSkyboxPlugin;

impl Plugin for ZoneSkyboxPlugin {
    fn build(&self, app: &mut App) {
        let forced = std::env::var("WOW_ZONE_SKYBOXES")
            .ok()
            .and_then(|v| v.trim().parse::<u8>().ok())
            .map(|v| v != 0);
        app.insert_resource(ZoneSkyboxOverride(forced))
            .add_observer(on_cvar)
            .add_systems(Startup, apply_override);
    }
}

fn apply_override(
    forced: Res<ZoneSkyboxOverride>,
    zone: Option<ResMut<benilla_world::skybox::ZoneSkyboxes>>,
) {
    if let (Some(on), Some(mut zone)) = (forced.0, zone) {
        zone.set_if_neq(benilla_world::skybox::ZoneSkyboxes(on));
    }
}

pub(crate) fn on_cvar(
    ev: On<crate::cvars::CvarChanged>,
    forced: Res<ZoneSkyboxOverride>,
    zone: Option<ResMut<benilla_world::skybox::ZoneSkyboxes>>,
) {
    if !ev.is("zoneSkyboxes") || forced.0.is_some() {
        return;
    }
    if let Some(mut zone) = zone {
        zone.set_if_neq(benilla_world::skybox::ZoneSkyboxes(ev.flag()));
    }
}
