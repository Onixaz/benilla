//! Small, hand-written NGX bindings. Feature 18 is not declared by the public SDK, so bindgen
//! would add tooling without covering the one ABI this crate needs.

use ash::vk::Handle as _;
use std::ffi::{c_void, CString};
use std::os::raw::c_char;
use std::path::Path;
use std::ptr;

pub const FEATURE_DLSSNR: u32 = 18;
const NGX_VERSION_API: u32 = 0x0000_0015;
const NGX_SUCCESS: u32 = 1;
type VkHandle = *const c_void;
type Pfn = *const c_void;

unsafe extern "C" {
    fn NVSDK_NGX_VULKAN_Init_with_ProjectID(
        project_id: *const c_char,
        engine_type: u32,
        engine_version: *const c_char,
        app_data_path: *const u16,
        instance: VkHandle,
        physical_device: VkHandle,
        device: VkHandle,
        get_instance_proc_addr: Pfn,
        get_device_proc_addr: Pfn,
        feature_info: *const c_void,
        sdk_version: u32,
    ) -> u32;
    fn NVSDK_NGX_VULKAN_GetCapabilityParameters(out: *mut *mut c_void) -> u32;
    fn NVSDK_NGX_Parameter_SetUI(parameters: *mut c_void, name: *const c_char, value: u32);
    fn NVSDK_NGX_Parameter_SetI(parameters: *mut c_void, name: *const c_char, value: i32);
}

type InitExt2 = unsafe extern "C" fn(
    u64,
    *const u16,
    VkHandle,
    VkHandle,
    VkHandle,
    Pfn,
    Pfn,
    u32,
    *const c_void,
) -> u32;
type CreateFeature =
    unsafe extern "C" fn(VkHandle, VkHandle, u32, *const c_void, *mut *mut c_void) -> u32;
type ReleaseFeature = unsafe extern "C" fn(*mut c_void) -> u32;

#[derive(Clone, Copy)]
pub struct DeviceHandles {
    instance: VkHandle,
    physical_device: VkHandle,
    device: VkHandle,
    get_instance_proc_addr: Pfn,
    get_device_proc_addr: Pfn,
}

pub struct NgxCore {
    parameters: *mut c_void,
    _runtime: libloading::Library,
    create_feature: CreateFeature,
    release_feature: ReleaseFeature,
}

// The render-world runtime serializes every call and holds the render device until after release.
unsafe impl Send for NgxCore {}
unsafe impl Sync for NgxCore {}

impl NgxCore {
    /// Initialize the linked NGX core and the externally supplied Feature-18 runtime.
    ///
    /// # Safety
    /// `handles` must belong to a live Vulkan device for this thread's renderer.
    pub unsafe fn init(
        project_id: &str,
        runtime: &Path,
        data_dir: &Path,
        handles: DeviceHandles,
    ) -> Result<Self, String> {
        let project_id = CString::new(project_id).map_err(|_| "NGX project id contains NUL")?;
        let engine = CString::new(env!("CARGO_PKG_VERSION")).expect("crate version has no NUL");
        std::fs::create_dir_all(data_dir).map_err(|error| {
            format!(
                "could not create NGX data directory {}: {error}",
                data_dir.display()
            )
        })?;
        let data_dir: Vec<u16> = data_dir
            .as_os_str()
            .to_string_lossy()
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();

        let result = NVSDK_NGX_VULKAN_Init_with_ProjectID(
            project_id.as_ptr(),
            0,
            engine.as_ptr(),
            data_dir.as_ptr(),
            handles.instance,
            handles.physical_device,
            handles.device,
            handles.get_instance_proc_addr,
            handles.get_device_proc_addr,
            ptr::null(),
            NGX_VERSION_API,
        );
        if result != NGX_SUCCESS {
            return Err(format!("NGX core initialization failed (0x{result:08X})"));
        }

        let mut parameters = ptr::null_mut();
        let result = NVSDK_NGX_VULKAN_GetCapabilityParameters(&mut parameters);
        if result != NGX_SUCCESS || parameters.is_null() {
            return Err(format!(
                "NGX did not return capability parameters (0x{result:08X})"
            ));
        }

        let library = libloading::Library::new(runtime)
            .map_err(|error| format!("could not load {}: {error}", runtime.display()))?;
        let init: libloading::Symbol<InitExt2> = library
            .get(b"NVSDK_NGX_VULKAN_Init_Ext2\0")
            .map_err(|error| format!("DLSSNR Init_Ext2 export missing: {error}"))?;
        let create_feature: libloading::Symbol<CreateFeature> = library
            .get(b"NVSDK_NGX_VULKAN_CreateFeature1\0")
            .map_err(|error| format!("DLSSNR CreateFeature1 export missing: {error}"))?;
        let release_feature: libloading::Symbol<ReleaseFeature> = library
            .get(b"NVSDK_NGX_VULKAN_ReleaseFeature\0")
            .map_err(|error| format!("DLSSNR ReleaseFeature export missing: {error}"))?;
        let (init, create_feature, release_feature) = (*init, *create_feature, *release_feature);
        let result = init(
            0x1122_3344_5566_7788,
            data_dir.as_ptr(),
            handles.instance,
            handles.physical_device,
            handles.device,
            handles.get_instance_proc_addr,
            handles.get_device_proc_addr,
            NGX_VERSION_API,
            parameters,
        );
        if result != NGX_SUCCESS {
            return Err(format!(
                "DLSSNR snippet initialization failed (0x{result:08X}); launch the main executable as nvngx.dll"
            ));
        }

        Ok(Self {
            parameters,
            _runtime: library,
            create_feature,
            release_feature,
        })
    }

    /// # Safety
    /// `command_buffer` must be recording on the device that initialized this runtime.
    pub unsafe fn create_feature(
        &self,
        device: VkHandle,
        command_buffer: VkHandle,
        width: u32,
        height: u32,
    ) -> Result<*mut c_void, u32> {
        let set_u32 = |name: &str, value| {
            let name = CString::new(name).expect("NGX parameter names contain no NUL");
            NVSDK_NGX_Parameter_SetUI(self.parameters, name.as_ptr(), value);
        };
        set_u32("CreationNodeMask", 1);
        set_u32("VisibilityNodeMask", 1);
        set_u32("Width", width);
        set_u32("Height", height);
        set_u32("OutWidth", width);
        set_u32("OutHeight", height);
        set_u32("DLSSNR.Width", width);
        set_u32("DLSSNR.Height", height);
        set_u32("DLSSNR.OutputWidth", width);
        set_u32("DLSSNR.OutputHeight", height);
        set_u32("DLSSNR.Enabled", 1);
        let preset = CString::new("DLSSNR.Hint.Render.Preset").expect("literal has no NUL");
        NVSDK_NGX_Parameter_SetI(self.parameters, preset.as_ptr(), 0);

        let mut feature = ptr::null_mut();
        let result = (self.create_feature)(
            device,
            command_buffer,
            FEATURE_DLSSNR,
            self.parameters,
            &mut feature,
        );
        if result == NGX_SUCCESS && !feature.is_null() {
            Ok(feature)
        } else {
            Err(result)
        }
    }

    /// # Safety
    /// `feature` must have been created by this runtime after GPU work using it has completed.
    pub unsafe fn release_feature(&self, feature: *mut c_void) {
        let _ = (self.release_feature)(feature);
    }
}

/// # Safety
/// The supplied device must be live; returned handles borrow it.
pub unsafe fn device_handles(device: &wgpu::Device) -> Option<DeviceHandles> {
    use wgpu::hal::api::Vulkan;

    let device = device.as_hal::<Vulkan>()?;
    let shared = device.shared_instance();
    let instance = shared.raw_instance();
    Some(DeviceHandles {
        instance: instance.handle().as_raw() as usize as VkHandle,
        physical_device: device.raw_physical_device().as_raw() as usize as VkHandle,
        device: device.raw_device().handle().as_raw() as usize as VkHandle,
        get_instance_proc_addr: shared.entry().static_fn().get_instance_proc_addr as usize as Pfn,
        get_device_proc_addr: instance.fp_v1_0().get_device_proc_addr as usize as Pfn,
    })
}

pub fn raw_command_buffer(encoder: &mut wgpu::CommandEncoder) -> Option<VkHandle> {
    use wgpu::hal::api::Vulkan;

    let mut raw = ptr::null();
    // SAFETY: called from the render thread while this encoder is live and recording.
    unsafe {
        encoder.as_hal_mut::<Vulkan, _, _>(|encoder| {
            raw = encoder
                .map(|encoder| encoder.raw_handle().as_raw() as usize as VkHandle)
                .unwrap_or(ptr::null());
        });
    }
    (!raw.is_null()).then_some(raw)
}

pub fn raw_device(device: &wgpu::Device) -> Option<VkHandle> {
    // SAFETY: the render device remains alive for the runtime's entire lifetime.
    unsafe { device_handles(device).map(|handles| handles.device) }
}
