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

/// Opt into the DLAA camera requirements without initializing NGX or adding Bevy's DLSS render
/// nodes. It is deliberately mutually exclusive with [`requested`], so one launch answers
/// whether temporal camera state alone changes the world image.
fn camera_only_requested() -> bool {
    camera_only_requested_from(
        std::env::var("WOW_DLSS").ok().as_deref(),
        std::env::var("WOW_DLSS_CAMERA_ONLY").ok().as_deref(),
    )
}

fn camera_only_requested_from(dlss: Option<&str>, camera_only: Option<&str>) -> bool {
    !requested_from(dlss) && requested_from(camera_only)
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

/// Adds either the non-NGX camera-only diagnostic or the built-in DLSS renderer after
/// `DefaultPlugins` has created its render app.
pub fn install_after_default_plugins(app: &mut App) {
    if camera_only_requested() {
        app.add_plugins(DlssCameraOnlyPlugin);
    }

    #[cfg(feature = "dlss")]
    {
        use bevy::anti_alias::dlss::DlssPlugin;

        if requested() {
            app.add_plugins((DlssPlugin, DlssPrototypePlugin));
        }
    }

    #[cfg(not(feature = "dlss"))]
    {
        // This runs after LogPlugin, so an opt-in request always leaves an actionable fallback line.
        if requested() {
            warn!(
                "DLSS: requested, but this executable lacks the `dlss` Cargo feature; using the normal renderer"
            );
        }
    }
}

/// Marker for the one temporary camera-only diagnostic. It keeps the test state off every other
/// camera, and makes the test's jitter driver independent of NVIDIA initialization.
#[derive(Component)]
struct DlssCameraOnly;

struct DlssCameraOnlyPlugin;

impl Plugin for DlssCameraOnlyPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (configure_dlss_camera_only, advance_dlss_camera_only_jitter),
        );
    }
}

/// Reproduces the camera-side state Bevy DLSS uses for a DLAA view, but has no DLSS component,
/// context, resolution override, or render-graph node. World cameras already receive `Hdr` and
/// the two prepasses in `view::plugin`; inserting them again here makes the diagnostic explicit.
fn configure_dlss_camera_only(
    mut commands: Commands,
    cameras: Query<Entity, (With<crate::view::WorldCamera>, Without<DlssCameraOnly>)>,
) {
    use bevy::{
        core_pipeline::prepass::{DepthPrepass, MotionVectorPrepass},
        render::{
            camera::{MipBias, TemporalJitter},
            view::{Hdr, Msaa},
        },
    };

    for entity in &cameras {
        commands.entity(entity).insert((
            DlssCameraOnly,
            Hdr,
            TemporalJitter::default(),
            // DLAA's internal and output resolutions match, so `dlss_wgpu` suggests -1.0.
            MipBias::default(),
            DepthPrepass,
            MotionVectorPrepass,
            Msaa::Off,
        ));
        warn!(
            "DLSS camera-only: WorldCamera has DLAA jitter, mip bias, depth/motion prepasses, and MSAA=Off; NGX evaluation is not running"
        );
    }
}

/// `dlss_wgpu` uses this eight-phase Halton sequence at a DLAA ratio of one. Keeping it here lets
/// the camera-only run exercise the same moving projection without opening an NGX context.
fn advance_dlss_camera_only_jitter(
    mut cameras: Query<&mut bevy::render::camera::TemporalJitter, With<DlssCameraOnly>>,
    mut frame: Local<u32>,
) {
    let offset = dlss_dlaa_jitter(*frame);
    *frame = frame.wrapping_add(1);
    for mut jitter in &mut cameras {
        jitter.offset = offset;
    }
}

fn dlss_dlaa_jitter(frame: u32) -> Vec2 {
    let phase = frame % 8;
    Vec2::new(halton(phase, 2), halton(phase, 3)) - 0.5
}

fn halton(mut index: u32, base: u32) -> f32 {
    let mut value = 0.0;
    let mut factor = 1.0;
    while index > 0 {
        factor /= base as f32;
        value += factor * (index % base) as f32;
        index /= base;
    }
    value
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
        info!("DLSS: Super Resolution supported; configuring WorldCamera in DLAA mode");
        state.support_reported = true;
    }
    for (entity, dlss) in &cameras {
        if dlss.is_some() {
            continue;
        }
        commands.entity(entity).insert((
            Dlss {
                perf_quality_mode: DlssPerfQualityMode::Dlaa,
                reset: true,
                ..default()
            },
            // DLSS is an alternative AA/upscaling path for this view, never a global UI policy.
            Msaa::Off,
        ));
        state.active = true;
        if !state.camera_reported {
            info!(
            "DLSS prototype: WorldCamera only; mode=DLAA; MSAA=Off, static_gx=disabled, liquid surfaces=disabled"
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
            "DLSS: supported; mode=DLAA; output={}x{}; internal={}x{}; backend={:?}; MSAA=Off; static_gx=disabled; liquid surfaces=disabled",
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
    use super::{camera_only_requested_from, dlss_dlaa_jitter, requested_from};
    use bevy::math::Vec2;

    #[test]
    fn only_the_explicit_developer_switch_requests_dlss() {
        assert!(requested_from(Some("1")));
        for value in [None, Some(""), Some("0"), Some("true"), Some("quality")] {
            assert!(!requested_from(value));
        }
    }

    #[test]
    fn camera_only_is_explicit_and_never_runs_beside_dlss() {
        assert!(camera_only_requested_from(None, Some("1")));
        assert!(!camera_only_requested_from(Some("1"), Some("1")));
        assert!(!camera_only_requested_from(None, Some("0")));
    }

    #[test]
    fn camera_only_uses_dlaas_eight_phase_halton_sequence() {
        assert_eq!(dlss_dlaa_jitter(0), Vec2::splat(-0.5));
        assert_eq!(dlss_dlaa_jitter(8), dlss_dlaa_jitter(0));
        assert_ne!(dlss_dlaa_jitter(1), dlss_dlaa_jitter(0));
    }

    #[cfg(not(feature = "dlss"))]
    #[test]
    fn a_normal_build_never_activates_the_prototype() {
        assert!(!super::enabled_build_requested());
    }
}
