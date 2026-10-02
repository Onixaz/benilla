//! Opt-in NVIDIA NGX Feature-18 initialization for Benilla.
//!
//! This crate evaluates only the world camera's Feature-18 view. Its bridge preserves Benilla's
//! gamma-authored HDR lane byte-for-byte, so FFXGlow and the native-resolution FrameXML UI retain
//! their existing colour contract.

mod bridge;
mod ngx;

use ash::vk;
use bevy::app::AppExit;
use bevy::camera::{Camera3d, CameraMainTextureUsages};
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::core_pipeline::prepass::ViewPrepassTextures;
use bevy::ecs::query::QueryItem;
use bevy::log::{info, warn};
use bevy::prelude::*;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::render_graph::{
    NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::TextureUsages;
use bevy::render::renderer::raw_vulkan_init::RawVulkanInitSettings;
use bevy::render::renderer::{RenderAdapter, RenderContext, RenderDevice};
use bevy::render::sync_world::MainEntity;
use bevy::render::view::ViewTarget;
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderSystems};
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

const DEFAULT_PROJECT_ID: &str = "a0f57b54-1daf-4934-90ae-c4035c19df04";
const RUNTIME_DLL: &str = "nvngx_dlssnr.dll";

/// Marker for the world camera that owns and evaluates Feature 18.
#[derive(Component, Clone, Copy, Default, ExtractComponent)]
pub struct DlssNrCamera;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DlssNrState {
    Unavailable,
    Initializing,
    Active,
    Failed(String),
}

/// Shared status. `Active` means the world camera's Feature 18 is evaluating successfully.
#[derive(Resource, Clone, Debug)]
pub struct DlssNrStatus(Arc<Mutex<DlssNrState>>);

impl DlssNrStatus {
    pub fn state(&self) -> DlssNrState {
        self.0.lock().expect("DLSSNR status mutex poisoned").clone()
    }

    fn set(&self, state: DlssNrState) {
        *self.0.lock().expect("DLSSNR status mutex poisoned") = state;
    }
}

/// Shared by the dev-only key system and render node; true leaves the world unprocessed for A/B.
#[derive(Resource, Clone, Default)]
struct DlssNrBypass(Arc<AtomicBool>);

/// True replaces the world with an amplified raw-versus-NR difference image for verification.
#[cfg(feature = "dev")]
#[derive(Resource, Clone, Default)]
struct DlssNrDifference(Arc<AtomicBool>);

/// Registers the raw Vulkan device requirements before Bevy creates its render device.
pub struct DlssNrPlugin {
    data_dir: Option<PathBuf>,
}

impl DlssNrPlugin {
    /// `data_dir` is provided by the host application's local-state policy.
    pub fn new(data_dir: Option<PathBuf>) -> Self {
        Self { data_dir }
    }
}

impl Plugin for DlssNrPlugin {
    fn build(&self, app: &mut App) {
        let mut settings = app
            .world_mut()
            .get_resource_or_init::<RawVulkanInitSettings>();
        // SAFETY: extensions are appended only when this adapter advertises them; no existing
        // feature is removed or changed. The leaked feature struct must outlive device creation.
        unsafe {
            settings.add_create_device_callback(|args, adapter, _| {
                let capabilities = adapter.physical_device_capabilities();
                for extension in [
                    vk::NVX_BINARY_IMPORT_NAME,
                    vk::NVX_IMAGE_VIEW_HANDLE_NAME,
                    vk::KHR_PUSH_DESCRIPTOR_NAME,
                    vk::KHR_MAINTENANCE4_NAME,
                ] {
                    if capabilities.supports_extension(extension)
                        && !args.extensions.contains(&extension)
                    {
                        args.extensions.push(extension);
                    }
                }

                let has_bda = args.extensions.iter().any(|extension| {
                    *extension == vk::KHR_BUFFER_DEVICE_ADDRESS_NAME
                        || *extension == vk::EXT_BUFFER_DEVICE_ADDRESS_NAME
                });
                if !has_bda && capabilities.supports_extension(vk::EXT_BUFFER_DEVICE_ADDRESS_NAME) {
                    let features =
                        Box::leak(Box::new(vk::PhysicalDeviceBufferDeviceAddressFeatures {
                            buffer_device_address: vk::TRUE,
                            p_next: args.create_info.p_next as *mut c_void,
                            ..default()
                        }));
                    args.extensions.push(vk::EXT_BUFFER_DEVICE_ADDRESS_NAME);
                    args.create_info.p_next = features as *mut _ as *const c_void;
                }
            });
        }

        app.insert_resource(DlssNrStatus(Arc::new(Mutex::new(
            DlssNrState::Initializing,
        ))));
        app.init_resource::<DlssNrBypass>();
        #[cfg(feature = "dev")]
        app.init_resource::<DlssNrDifference>();
        app.add_systems(Update, ensure_camera_texture_usages);
        #[cfg(feature = "dev")]
        app.add_systems(Update, toggle_ab_bypass);
    }

    fn finish(&self, app: &mut App) {
        // `DlssNrPlugin` builds before `DefaultPlugins` so NGX can amend Vulkan device creation.
        // `ExtractComponentPlugin` instead needs the render sub-app, which only exists by finish.
        app.add_plugins(ExtractComponentPlugin::<DlssNrCamera>::default());
        let status = app.world().resource::<DlssNrStatus>().clone();
        let bypass = app.world().resource::<DlssNrBypass>().clone();
        #[cfg(feature = "dev")]
        let difference = app.world().resource::<DlssNrDifference>().clone();
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            status.set(DlssNrState::Unavailable);
            warn!("dlssnr: no render application; continuing without Neural Rendering");
            return;
        };
        render_app.insert_resource(DlssNrRuntime {
            inner: Mutex::new(Runtime::new(
                status,
                bypass,
                #[cfg(feature = "dev")]
                difference,
                self.data_dir.clone(),
            )),
        });
        render_app.add_systems(ExtractSchedule, extract_shutdown);
        render_app.add_systems(Render, shutdown_runtime.in_set(RenderSystems::Cleanup));
        render_app
            .add_render_graph_node::<ViewNodeRunner<DlssNrEvalNode>>(Core3d, DlssNrEvalLabel)
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::StartMainPassPostProcessing,
                    DlssNrEvalLabel,
                    Node3d::Bloom,
                ),
            );
    }
}

/// Development-only world diagnostics: `Ctrl+Alt+N` switches raw/NR; `Ctrl+Alt+D` displays the
/// amplified per-pixel difference. Neither touches the FFX or UI lanes.
#[cfg(feature = "dev")]
fn toggle_ab_bypass(
    keys: Res<ButtonInput<KeyCode>>,
    bypass: Res<DlssNrBypass>,
    difference: Res<DlssNrDifference>,
) {
    let control = keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);
    let alt = keys.pressed(KeyCode::AltLeft) || keys.pressed(KeyCode::AltRight);
    if control && alt && keys.just_pressed(KeyCode::KeyN) {
        difference.0.store(false, Ordering::Relaxed);
        let raw = !bypass.0.fetch_xor(true, Ordering::Relaxed);
        info!(
            "dlssnr: A/B switched to {}; press Ctrl+Alt+N again to switch back",
            if raw { "RAW world" } else { "Feature 18" }
        );
    }
    if control && alt && keys.just_pressed(KeyCode::KeyD) {
        bypass.0.store(false, Ordering::Relaxed);
        let enabled = !difference.0.fetch_xor(true, Ordering::Relaxed);
        info!(
            "dlssnr: difference view {}; Ctrl+Alt+D returns to Feature 18",
            if enabled {
                "enabled (128x)"
            } else {
                "disabled"
            }
        );
    }
}

/// Feature 18 reads the same main colour, depth and motion-vector images Bevy renders, then its
/// bridge writes the result back into that main colour image. Make both usages explicit before
/// extraction creates the sole world camera's images.
fn ensure_camera_texture_usages(
    mut cameras: Query<(&mut CameraMainTextureUsages, &mut Camera3d), With<DlssNrCamera>>,
) {
    for (mut main, mut camera) in &mut cameras {
        main.0 |= TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING;
        let mut depth = TextureUsages::from(camera.depth_texture_usages);
        depth |= TextureUsages::TEXTURE_BINDING;
        camera.depth_texture_usages = depth.into();
    }
}

#[derive(Resource)]
struct DlssNrRuntime {
    inner: Mutex<Runtime>,
}

#[derive(Resource)]
struct DlssNrShutdown;

struct Runtime {
    core: Option<ngx::NgxCore>,
    features: HashMap<Entity, Feature>,
    bridge: Option<bridge::Bridge>,
    device: Option<RenderDevice>,
    status: DlssNrStatus,
    bypass: DlssNrBypass,
    #[cfg(feature = "dev")]
    difference: DlssNrDifference,
    #[cfg(feature = "dev")]
    presentation: Option<DebugPresentation>,
    data_dir: Option<PathBuf>,
    failed: bool,
}

/// The render-side result last presented for the developer A/B instrument.
#[cfg(feature = "dev")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum DebugPresentation {
    Raw,
    Feature18,
    Difference,
}

struct Feature {
    handle: usize,
    width: u32,
    height: u32,
    evaluated: bool,
}

impl Runtime {
    fn new(
        status: DlssNrStatus,
        bypass: DlssNrBypass,
        #[cfg(feature = "dev")] difference: DlssNrDifference,
        data_dir: Option<PathBuf>,
    ) -> Self {
        Self {
            core: None,
            features: HashMap::new(),
            bridge: None,
            device: None,
            status,
            bypass,
            #[cfg(feature = "dev")]
            difference,
            #[cfg(feature = "dev")]
            presentation: None,
            data_dir,
            failed: false,
        }
    }

    fn fail(&mut self, reason: String) {
        warn!("dlssnr: {reason}; continuing without Neural Rendering");
        self.status.set(DlssNrState::Failed(reason));
        self.failed = true;
    }

    #[cfg(feature = "dev")]
    fn report_presentation(&mut self, presentation: DebugPresentation, width: u32, height: u32) {
        if self.presentation == Some(presentation) {
            return;
        }
        self.presentation = Some(presentation);
        let label = match presentation {
            DebugPresentation::Raw => "RAW world (Feature 18 bypassed)",
            DebugPresentation::Feature18 => "Feature 18 output",
            DebugPresentation::Difference => "128x raw-versus-Feature-18 difference",
        };
        info!("dlssnr: render A/B is presenting {label} at {width}x{height}");
    }

    fn shutdown(&mut self) {
        let Some(core) = &self.core else {
            return;
        };
        if let Some(device) = &self.device {
            if let Err(error) = device.poll(wgpu::PollType::wait_indefinitely()) {
                warn!("dlssnr: GPU wait during shutdown failed: {error}");
            }
        }
        for (_, feature) in self.features.drain() {
            // SAFETY: `core` created this handle, queued work is complete, and shutdown is
            // exclusively ordered in render cleanup before the runtime or DLL are dropped.
            unsafe { core.release_feature(feature.handle as *mut c_void) };
        }
        self.core = None;
        self.bridge = None;
        self.device = None;
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn extract_shutdown(mut commands: Commands, exits: Extract<Res<Messages<AppExit>>>) {
    if !exits.is_empty() {
        commands.insert_resource(DlssNrShutdown);
    }
}

fn shutdown_runtime(shutdown: Option<Res<DlssNrShutdown>>, mut runtime: ResMut<DlssNrRuntime>) {
    if shutdown.is_none() {
        return;
    }
    match runtime.inner.get_mut() {
        Ok(runtime) => runtime.shutdown(),
        Err(poisoned) => poisoned.into_inner().shutdown(),
    }
}

/// World-only Feature-18 evaluation, explicitly ordered before FFXGlow by `benilla-world`.
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct DlssNrEvalLabel;

#[derive(Default)]
struct DlssNrEvalNode;

impl ViewNode for DlssNrEvalNode {
    type ViewQuery = (
        &'static MainEntity,
        &'static DlssNrCamera,
        &'static ViewTarget,
        &'static ViewPrepassTextures,
    );

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        context: &mut RenderContext<'w>,
        (main_entity, _, target, prepass): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let device = world.resource::<RenderDevice>();
        let adapter = world.resource::<RenderAdapter>();
        let runtime = world.resource::<DlssNrRuntime>();
        let (Some(depth), Some(motion)) = (&prepass.depth, &prepass.motion_vectors) else {
            return Ok(());
        };
        let width = target.main_texture().width();
        let height = target.main_texture().height();
        if depth.texture.texture.width() != width
            || depth.texture.texture.height() != height
            || motion.texture.texture.width() != width
            || motion.texture.texture.height() != height
        {
            return Ok(());
        }
        let key = main_entity.id();
        let mut runtime = runtime.inner.lock().expect("DLSSNR runtime mutex poisoned");
        if runtime.failed {
            return Ok(());
        }
        if runtime.bypass.0.load(Ordering::Relaxed) {
            #[cfg(feature = "dev")]
            runtime.report_presentation(DebugPresentation::Raw, width, height);
            return Ok(());
        }

        if runtime.core.is_none() {
            let info = adapter.get_info();
            if unsafe { ngx::device_handles(device.wgpu_device()) }.is_none() {
                runtime.status.set(DlssNrState::Unavailable);
                runtime.failed = true;
                warn!(
                    "dlssnr: adapter '{}' uses {:?}, not Vulkan; continuing without Neural Rendering",
                    info.name, info.backend
                );
                return Ok(());
            }
            let architecture = Architecture::detect(&info.name);
            let Some(data_dir) = runtime.data_dir.as_deref() else {
                runtime.fail(
                    "Benilla local state is unavailable, so NGX has nowhere safe to write".into(),
                );
                return Ok(());
            };
            let dll = match resolve_runtime(&info.name) {
                Ok(dll) => dll,
                Err(reason) => {
                    runtime.fail(reason);
                    return Ok(());
                }
            };
            info!(
                "dlssnr: Vulkan adapter='{}' architecture={} runtime={}",
                info.name,
                architecture.map_or("unknown", Architecture::folder),
                dll.display()
            );
            let handles = unsafe { ngx::device_handles(device.wgpu_device()) }
                .expect("Vulkan handles checked immediately above");
            match unsafe { ngx::NgxCore::init(DEFAULT_PROJECT_ID, &dll, data_dir, handles) } {
                Ok(core) => {
                    info!("dlssnr: NGX initialized");
                    runtime.device = Some((*device).clone());
                    runtime.core = Some(core);
                }
                Err(reason) => {
                    runtime.fail(reason);
                    return Ok(());
                }
            }
        }

        let needs_create = runtime
            .features
            .get(&key)
            .is_none_or(|feature| feature.width != width || feature.height != height);
        if needs_create {
            if let Some(old) = runtime.features.remove(&key) {
                // A resized view replaces a feature only after all work using its old images is
                // done. Resizes are rare; correctness is preferable to one stretched frame.
                if let Err(error) = device.poll(wgpu::PollType::wait_indefinitely()) {
                    runtime.fail(format!("GPU wait before Feature 18 resize failed: {error}"));
                    return Ok(());
                }
                unsafe {
                    runtime
                        .core
                        .as_ref()
                        .expect("core initialized")
                        .release_feature(old.handle as *mut c_void)
                };
            }
        }
        let bridge_needs_resize = runtime
            .bridge
            .as_ref()
            .is_none_or(|bridge| bridge.width != width || bridge.height != height);
        if bridge_needs_resize {
            runtime.bridge = Some(bridge::Bridge::new(device.wgpu_device(), width, height));
        }

        let Some(command_buffer) = ngx::raw_command_buffer(context.command_encoder()) else {
            runtime.fail("could not obtain a Vulkan command buffer".into());
            return Ok(());
        };
        let Some(raw_device) = ngx::raw_device(device.wgpu_device()) else {
            runtime.status.set(DlssNrState::Unavailable);
            runtime.failed = true;
            return Ok(());
        };
        if needs_create {
            let created = unsafe {
                runtime
                    .core
                    .as_ref()
                    .expect("core initialized")
                    .create_feature(raw_device, command_buffer, width, height)
            };
            match created {
                Ok(feature) => {
                    runtime.features.insert(
                        key,
                        Feature {
                            handle: feature as usize,
                            width,
                            height,
                            evaluated: false,
                        },
                    );
                    info!(
                        "dlssnr: Feature 18 created at {width}x{height}; depth={:?}, motion={:?}",
                        depth.texture.texture.format(),
                        motion.texture.texture.format()
                    );
                }
                Err(code) => {
                    runtime.fail(format!("Feature 18 creation failed (0x{code:08X})"));
                    return Ok(());
                }
            }
        }

        let bridge = runtime.bridge.as_ref().expect("bridge initialized above");
        bridge.copy(
            device.wgpu_device(),
            context.command_encoder(),
            target.main_texture_view(),
            &bridge.input_view,
        );
        let (Some(color), Some(output), Some(depth), Some(motion)) = (
            unsafe { ngx::resource(adapter, &bridge.input, &bridge.input_view) },
            unsafe { ngx::resource(adapter, &bridge.output, &bridge.output_view) },
            unsafe { ngx::resource(adapter, &depth.texture.texture, &depth.texture.default_view) },
            unsafe {
                ngx::resource(
                    adapter,
                    &motion.texture.texture,
                    &motion.texture.default_view,
                )
            },
        ) else {
            runtime.fail("could not expose Vulkan image handles for Feature 18".into());
            return Ok(());
        };
        let Some(command_buffer) = ngx::raw_command_buffer(context.command_encoder()) else {
            runtime.fail("could not obtain the Vulkan command buffer for Feature 18".into());
            return Ok(());
        };
        let feature = runtime
            .features
            .get(&key)
            .expect("Feature 18 created above or already resident");
        let first_evaluation = !feature.evaluated;
        match unsafe {
            runtime.core.as_ref().expect("core initialized").evaluate(
                command_buffer,
                feature.handle as *mut c_void,
                color,
                output,
                depth,
                motion,
            )
        } {
            Ok(()) => {
                #[cfg(feature = "dev")]
                let difference = runtime.difference.0.load(Ordering::Relaxed);
                #[cfg(feature = "dev")]
                if difference {
                    bridge.difference(
                        device.wgpu_device(),
                        context.command_encoder(),
                        &bridge.input_view,
                        &bridge.output_view,
                    );
                    bridge.copy(
                        device.wgpu_device(),
                        context.command_encoder(),
                        &bridge.comparison_view,
                        target.main_texture_view(),
                    );
                }
                #[cfg(not(feature = "dev"))]
                bridge.copy(
                    device.wgpu_device(),
                    context.command_encoder(),
                    &bridge.output_view,
                    target.main_texture_view(),
                );
                #[cfg(feature = "dev")]
                if !difference {
                    bridge.copy(
                        device.wgpu_device(),
                        context.command_encoder(),
                        &bridge.output_view,
                        target.main_texture_view(),
                    );
                }
                runtime.status.set(DlssNrState::Active);
                #[cfg(feature = "dev")]
                runtime.report_presentation(
                    if difference {
                        DebugPresentation::Difference
                    } else {
                        DebugPresentation::Feature18
                    },
                    width,
                    height,
                );
                if first_evaluation {
                    runtime
                        .features
                        .get_mut(&key)
                        .expect("Feature 18 remains resident while evaluating")
                        .evaluated = true;
                    info!(
                        "dlssnr: Feature 18 is evaluating the {}x{} world view before FFXGlow (intensity={})",
                        width,
                        height,
                        ngx::DLSSNR_INTENSITY,
                    );
                }
            }
            Err(code) => runtime.fail(format!("Feature 18 evaluation failed (0x{code:08X})")),
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Architecture {
    Turing,
    AdaLovelace,
    Blackwell,
}

impl Architecture {
    fn folder(self) -> &'static str {
        match self {
            Self::Turing => "Turing+",
            Self::AdaLovelace => "Ada Lovelace+",
            Self::Blackwell => "Blackwell+",
        }
    }

    fn detect(name: &str) -> Option<Self> {
        let name = name.to_ascii_uppercase();
        if name.contains("BLACKWELL") {
            return Some(Self::Blackwell);
        }
        if name.contains("ADA") {
            return Some(Self::AdaLovelace);
        }
        ["RTX", "GTX"].into_iter().find_map(|prefix| {
            let start = name.find(prefix)? + prefix.len();
            let digits: String = name[start..]
                .chars()
                .skip_while(|character| !character.is_ascii_digit())
                .take_while(char::is_ascii_digit)
                .collect();
            match digits.parse::<u32>().ok()? {
                5000..=5999 => Some(Self::Blackwell),
                4000..=4999 => Some(Self::AdaLovelace),
                1600..=3999 => Some(Self::Turing),
                _ => None,
            }
        })
    }
}

fn resolve_runtime(adapter_name: &str) -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("WOW_DLSSNR_DLL") {
        let path = PathBuf::from(path);
        return path
            .is_file()
            .then_some(path.clone())
            .ok_or_else(|| format!("WOW_DLSSNR_DLL is not a file: {}", path.display()));
    }
    let root = match std::env::var_os("WOW_DLSSNR_DIR") {
        Some(path) => PathBuf::from(path),
        None => std::env::current_exe()
            .ok()
            .and_then(|path| path.parent().map(|path| path.join("dlssnr")))
            .ok_or_else(|| "could not find the executable directory for dlssnr/".to_string())?,
    };
    let tier = Architecture::detect(adapter_name).ok_or_else(|| {
        format!("could not detect a supported RTX architecture from '{adapter_name}'")
    })?;
    let path = root.join(tier.folder()).join(RUNTIME_DLL);
    path.is_file().then_some(path.clone()).ok_or_else(|| {
        format!(
            "DLSSNR runtime is missing: place {} at {} or set WOW_DLSSNR_DLL",
            RUNTIME_DLL,
            path.display()
        )
    })
}

#[cfg(all(test, feature = "dev"))]
mod tests {
    use super::*;

    #[test]
    fn ctrl_alt_n_selects_the_raw_ab_side() {
        let mut app = App::new();
        app.init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<DlssNrBypass>()
            .init_resource::<DlssNrDifference>()
            .add_systems(Update, toggle_ab_bypass);
        {
            let mut keys = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
            keys.press(KeyCode::ControlLeft);
            keys.press(KeyCode::AltLeft);
            keys.press(KeyCode::KeyN);
        }
        app.update();
        assert!(app
            .world()
            .resource::<DlssNrBypass>()
            .0
            .load(Ordering::Relaxed));
    }
}
