//! The NGX Feature-18 snippet creates its feature only for a caller whose module path contains
//! `nvngx.dll`. This library is that caller: `benilla-dlss5`'s build script compiles it, the
//! executable embeds it, writes it to the NGX data folder on use and routes the snippet's load and
//! its four entry points through it, so `benilla.exe` keeps its name.
//!
//! Each export calls through and then uses the result, never a tail call, so the return address
//! the snippet sees lies inside this module.

#[cfg(windows)]
mod bridge {
    use std::ffi::c_void;
    use std::hint::black_box;

    type Handle = *const c_void;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
    }

    type InitExt2 = unsafe extern "C" fn(
        u64,
        *const u16,
        Handle,
        Handle,
        Handle,
        Handle,
        Handle,
        u32,
        *const c_void,
    ) -> u32;
    type CreateFeature =
        unsafe extern "C" fn(Handle, Handle, u32, *const c_void, *mut *mut c_void) -> u32;
    type ReleaseFeature = unsafe extern "C" fn(*mut c_void) -> u32;
    type EvaluateFeature =
        unsafe extern "C" fn(Handle, *const c_void, *const c_void, *const c_void) -> u32;

    /// Loads the snippet from this module; `path` is a NUL-terminated UTF-16 path.
    ///
    /// # Safety
    /// `path` must be a valid NUL-terminated wide string.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn benilla_nvngx_load(path: *const u16) -> *mut c_void {
        black_box(unsafe { LoadLibraryW(path) })
    }

    /// # Safety
    /// `f` is the snippet's `NVSDK_NGX_VULKAN_Init_Ext2`; the rest are its arguments.
    #[unsafe(no_mangle)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "C" fn benilla_nvngx_init_ext2(
        f: InitExt2,
        app_id: u64,
        data_dir: *const u16,
        instance: Handle,
        physical_device: Handle,
        device: Handle,
        get_instance_proc_addr: Handle,
        get_device_proc_addr: Handle,
        version: u32,
        parameters: *const c_void,
    ) -> u32 {
        black_box(unsafe {
            f(
                app_id,
                data_dir,
                instance,
                physical_device,
                device,
                get_instance_proc_addr,
                get_device_proc_addr,
                version,
                parameters,
            )
        })
    }

    /// # Safety
    /// `f` is the snippet's `NVSDK_NGX_VULKAN_CreateFeature1`; the rest are its arguments.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn benilla_nvngx_create_feature(
        f: CreateFeature,
        device: Handle,
        command_buffer: Handle,
        feature: u32,
        parameters: *const c_void,
        handle: *mut *mut c_void,
    ) -> u32 {
        black_box(unsafe { f(device, command_buffer, feature, parameters, handle) })
    }

    /// # Safety
    /// `f` is the snippet's `NVSDK_NGX_VULKAN_EvaluateFeature`; the rest are its arguments.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn benilla_nvngx_evaluate_feature(
        f: EvaluateFeature,
        command_buffer: Handle,
        feature: *const c_void,
        parameters: *const c_void,
        callback: *const c_void,
    ) -> u32 {
        black_box(unsafe { f(command_buffer, feature, parameters, callback) })
    }

    /// # Safety
    /// `f` is the snippet's `NVSDK_NGX_VULKAN_ReleaseFeature`; `feature` a live feature handle.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn benilla_nvngx_release_feature(
        f: ReleaseFeature,
        feature: *mut c_void,
    ) -> u32 {
        black_box(unsafe { f(feature) })
    }
}
