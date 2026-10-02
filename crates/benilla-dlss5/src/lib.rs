//! Opt-in NVIDIA NGX Feature-18 initialization for Benilla.
//!
//! This crate deliberately stops after initializing NGX and creating the feature. It does not
//! evaluate the feature or change a frame's pixels; the colour/depth/motion bridge belongs to the
//! later integration phase once Benilla's gamma lane has been validated end-to-end.

mod ngx;

use ash::vk;
use bevy::app::AppExit;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::core_pipeline::prepass::ViewPrepassTextures;
use bevy::ecs::query::QueryItem;
use bevy::log::{info, warn};
use bevy::prelude::*;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::render_graph::{
    NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::CommandEncoderDescriptor;
use bevy::render::renderer::raw_vulkan_init::RawVulkanInitSettings;
use bevy::render::renderer::{RenderAdapter, RenderContext, RenderDevice};
use bevy::render::sync_world::MainEntity;
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderSystems};
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

const DEFAULT_PROJECT_ID: &str = "a0f57b54-1daf-4934-90ae-c4035c19df04";
const RUNTIME_DLL: &str = "nvngx_dlssnr.dll";

/// Marker for the world camera that owns the Feature-18 lifetime. It does not evaluate NR yet.
#[derive(Component, Clone, Copy, Default, ExtractComponent)]
pub struct DlssNrCamera;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DlssNrState {
    Unavailable,
    Initializing,
    Active,
    Failed(String),
}

/// Shared status. `Active` means NGX initialized and Feature 18 was created; it does not claim
/// that the scene is being neural-rendered yet.
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
        ))))
        .add_plugins(ExtractComponentPlugin::<DlssNrCamera>::default());
    }

    fn finish(&self, app: &mut App) {
        let status = app.world().resource::<DlssNrStatus>().clone();
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            status.set(DlssNrState::Unavailable);
            warn!("dlssnr: no render application; continuing without Neural Rendering");
            return;
        };
        render_app.insert_resource(DlssNrRuntime {
            inner: Mutex::new(Runtime::new(status, self.data_dir.clone())),
        });
        render_app.add_systems(ExtractSchedule, extract_shutdown);
        render_app.add_systems(Render, shutdown_runtime.in_set(RenderSystems::Cleanup));
        // Creation is render-only in this phase, so its exact pixel order is intentionally
        // irrelevant. Evaluation will become an explicit graph node before FFXGlow in Phase 4.
        render_app
            .add_render_graph_node::<ViewNodeRunner<CreateFeatureNode>>(Core3d, CreateFeatureLabel)
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::StartMainPassPostProcessing,
                    CreateFeatureLabel,
                    Node3d::Bloom,
                ),
            );
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
    device: Option<RenderDevice>,
    status: DlssNrStatus,
    data_dir: Option<PathBuf>,
    failed: bool,
}

struct Feature {
    handle: usize,
    width: u32,
    height: u32,
}

impl Runtime {
    fn new(status: DlssNrStatus, data_dir: Option<PathBuf>) -> Self {
        Self {
            core: None,
            features: HashMap::new(),
            device: None,
            status,
            data_dir,
            failed: false,
        }
    }

    fn fail(&mut self, reason: String) {
        warn!("dlssnr: {reason}; continuing without Neural Rendering");
        self.status.set(DlssNrState::Failed(reason));
        self.failed = true;
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

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct CreateFeatureLabel;

#[derive(Default)]
struct CreateFeatureNode;

impl ViewNode for CreateFeatureNode {
    type ViewQuery = (
        &'static MainEntity,
        &'static DlssNrCamera,
        &'static ViewPrepassTextures,
    );

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        context: &mut RenderContext<'w>,
        (main_entity, _, prepass): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let device = world.resource::<RenderDevice>();
        let adapter = world.resource::<RenderAdapter>();
        let runtime = world.resource::<DlssNrRuntime>();
        let (Some(depth), Some(motion)) = (&prepass.depth, &prepass.motion_vectors) else {
            return Ok(());
        };
        let width = depth.texture.texture.width();
        let height = depth.texture.texture.height();
        let key = main_entity.id();
        let mut runtime = runtime.inner.lock().expect("DLSSNR runtime mutex poisoned");
        if runtime.failed {
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
        if !needs_create {
            return Ok(());
        }
        if let Some(old) = runtime.features.remove(&key) {
            // No evaluation has occurred in this phase. The completed creation command is the
            // only possible work before a resize can reach this point.
            unsafe {
                runtime
                    .core
                    .as_ref()
                    .expect("core initialized")
                    .release_feature(old.handle as *mut c_void)
            };
        }

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("dlssnr_feature18_create"),
        });
        let Some(command_buffer) = ngx::raw_command_buffer(&mut encoder) else {
            runtime.fail("could not obtain a Vulkan command buffer".into());
            return Ok(());
        };
        let Some(raw_device) = ngx::raw_device(device.wgpu_device()) else {
            runtime.status.set(DlssNrState::Unavailable);
            runtime.failed = true;
            return Ok(());
        };
        let created = unsafe {
            runtime
                .core
                .as_ref()
                .expect("core initialized")
                .create_feature(raw_device, command_buffer, width, height)
        };
        match created {
            Ok(feature) => {
                context.add_command_buffer(encoder.finish());
                runtime.features.insert(
                    key,
                    Feature {
                        handle: feature as usize,
                        width,
                        height,
                    },
                );
                runtime.status.set(DlssNrState::Active);
                info!(
                    "dlssnr: Feature 18 created at {width}x{height}; depth={:?}, motion={:?}",
                    depth.texture.texture.format(),
                    motion.texture.texture.format()
                );
            }
            Err(code) => runtime.fail(format!("Feature 18 creation failed (0x{code:08X})")),
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
