//! The deliberately narrow Bevy DLSS Super Resolution experiment. It is compile-time opt-in
//! (`--features dlss`) because NVIDIA's SDK is licensed separately, then runtime opt-in with
//! `WOW_DLSS=1`; the stock renderer remains the fallback in every other case.

use bevy::prelude::*;

/// Whether this launch asks for the DLSS proof of life. This stays a developer switch rather than
/// a CVar: DLSS is not a 1.12 graphics setting and its runtime files are externally licensed.
pub fn requested() -> bool {
    requested_from(std::env::var("WOW_DLSS").ok().as_deref())
}

fn requested_from(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Whether this binary can honour the developer switch. A normal executable only reports the
/// request and leaves its existing backend, MSAA, static-GX and liquid behavior untouched.
pub fn enabled_build_requested() -> bool {
    #[cfg(feature = "dlss")]
    {
        requested()
    }
    #[cfg(not(feature = "dlss"))]
    {
        false
    }
}

/// The small amount of state other world systems need without coupling to Bevy's private NGX
/// objects. `active` flips only after Bevy confirms Super Resolution support and a world camera
/// receives its `Dlss` component.
#[derive(Resource, Default)]
pub struct DlssPrototypeState {
    active: bool,
    #[cfg(feature = "dlss")]
    support_reported: bool,
    #[cfg(feature = "dlss")]
    camera_reported: bool,
}

impl DlssPrototypeState {
    /// True once the world camera is running the DLSS path this launch.
    pub fn active(&self) -> bool {
        self.active
    }
}

/// Configures Bevy's raw Vulkan initialization before `DefaultPlugins` creates `RenderPlugin`.
///
/// `tuned_default_plugins` removes Bevy's static `DlssInitPlugin` entry so a feature-built binary
/// still has a normal renderer unless this developer switch requests DLSS.
#[cfg(feature = "dlss")]
pub fn install_before_default_plugins(app: &mut App) {
    use bevy::anti_alias::dlss::{DlssInitPlugin, DlssProjectId};

    if requested() {
        // This is Benilla's generated application ID, not Bevy's example ID or a machine path.
        app.insert_resource(DlssProjectId(bevy::asset::uuid::uuid!(
            "0ee66521-930b-472e-9b1d-82f6a7fa4f5b"
        )))
        .add_plugins(DlssInitPlugin);
    }
}

/// A normal build has no raw-Vulkan DLSS initializer to register.
#[cfg(not(feature = "dlss"))]
pub fn install_before_default_plugins(_app: &mut App) {}

/// Adds the built-in Bevy DLSS renderer only after `DefaultPlugins` has created its render app.
#[cfg(feature = "dlss")]
pub fn install_after_default_plugins(app: &mut App) {
    use bevy::anti_alias::dlss::DlssPlugin;

    if requested() {
        app.add_plugins((DlssPlugin, DlssPrototypePlugin));
    }
}

/// Normal launches carry no DLSS systems or resources.
#[cfg(not(feature = "dlss"))]
pub fn install_after_default_plugins(_app: &mut App) {
    // This runs after LogPlugin, so an opt-in request always leaves an actionable fallback line.
    if requested() {
        warn!(
            "DLSS: requested, but this executable lacks the `dlss` Cargo feature; using the normal renderer"
        );
    }
}

#[cfg(feature = "dlss")]
struct DlssPrototypePlugin;

#[cfg(feature = "dlss")]
impl Plugin for DlssPrototypePlugin {
    fn build(&self, app: &mut App) {
        use bevy::render::{Render, RenderApp, RenderSystems};

        app.init_resource::<DlssPrototypeState>()
            .add_systems(Update, configure_world_camera);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.add_systems(
            Render,
            log_dlss_resolution.after(RenderSystems::ManageViews),
        );
    }

    fn finish(&self, app: &mut App) {
        use bevy::anti_alias::dlss::DlssSuperResolutionSupported;
        use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
        use bevy::render::{render_graph::RenderGraphExt, RenderApp};

        // `DlssPlugin::finish` inserts the support resource and the two DLSS nodes first. Only
        // then can we safely require FFXGlow to consume the reconstructed full-size world image.
        if !app
            .world()
            .contains_resource::<DlssSuperResolutionSupported>()
        {
            return;
        }
        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_render_graph_edges(
            Core3d,
            (
                Node3d::DlssRayReconstruction,
                crate::ffx_glow::FfxGlowLabel,
                Node3d::Bloom,
            ),
        );
    }
}

#[cfg(feature = "dlss")]
fn configure_world_camera(
    supported: Option<Res<bevy::anti_alias::dlss::DlssSuperResolutionSupported>>,
    mut state: ResMut<DlssPrototypeState>,
    cameras: Query<(Entity, Option<&bevy::anti_alias::dlss::Dlss>), With<crate::view::WorldCamera>>,
    mut commands: Commands,
) {
    use bevy::anti_alias::dlss::{Dlss, DlssPerfQualityMode};
    use bevy::render::view::Msaa;

    if supported.is_none() {
        if !state.support_reported {
            warn!(
                "DLSS: Super Resolution is unavailable; using the normal renderer (check RTX, Vulkan, NGX runtime, and SDK setup)"
            );
            state.support_reported = true;
        }
        return;
    }

    if !state.support_reported {
        info!("DLSS: Super Resolution supported; configuring WorldCamera in Quality mode");
        state.support_reported = true;
    }
    for (entity, dlss) in &cameras {
        if dlss.is_some() {
            continue;
        }
        commands.entity(entity).insert((
            Dlss {
                perf_quality_mode: DlssPerfQualityMode::Quality,
                reset: true,
                ..default()
            },
            // DLSS is an alternative AA/upscaling path for this view, never a global UI policy.
            Msaa::Off,
        ));
        state.active = true;
        if !state.camera_reported {
            info!(
                "DLSS prototype: WorldCamera only; MSAA=Off, static_gx=disabled, liquid surfaces=disabled"
            );
            state.camera_reported = true;
        }
    }
}

/// Emits the render-world dimensions after Bevy creates the DLSS context and its main-pass
/// override. A changed window reprints the pair, proving the internal world pass stays smaller.
#[cfg(feature = "dlss")]
fn log_dlss_resolution(
    cameras: Query<
        (
            &bevy::render::view::ExtractedView,
            &bevy::camera::MainPassResolutionOverride,
            &bevy::anti_alias::dlss::Dlss,
        ),
        With<bevy::camera::Camera3d>,
    >,
    adapter: Res<bevy::render::renderer::RenderAdapterInfo>,
    mut last: Local<Option<(UVec2, UVec2)>>,
) {
    for (view, internal, _) in &cameras {
        let output = UVec2::new(view.viewport.z, view.viewport.w);
        let pair = (output, internal.0);
        if *last == Some(pair) {
            continue;
        }
        *last = Some(pair);
        info!(
            "DLSS: supported; mode=Quality; output={}x{}; internal={}x{}; backend={:?}; MSAA=Off; static_gx=disabled; liquid surfaces=disabled",
            output.x,
            output.y,
            internal.0.x,
            internal.0.y,
            adapter.backend,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::requested_from;

    #[test]
    fn only_the_explicit_developer_switch_requests_dlss() {
        assert!(requested_from(Some("1")));
        for value in [None, Some(""), Some("0"), Some("true"), Some("quality")] {
            assert!(!requested_from(value));
        }
    }

    #[cfg(not(feature = "dlss"))]
    #[test]
    fn a_normal_build_never_activates_the_prototype() {
        assert!(!super::enabled_build_requested());
    }
}
