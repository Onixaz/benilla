# DLSS prototype

This branch has a narrow proof-of-life path for NVIDIA DLSS Super Resolution. It is not a stock
1.12 setting and remains both a Cargo feature and a developer-only launch switch.

## Build and launch

Benilla 0.18.1 resolves Bevy's `dlss` feature to `dlss_wgpu` 2.0.0 and wgpu 27. Its fixed
16-slot NGX G-buffer literal does not compile with the installed SDK's 17-slot header layout, so
Benilla vendors a narrow compatibility patch while retaining the same Bevy/wgpu API. Obtain the
SDK under its license, then provide absolute paths for its root and the Vulkan SDK. `bindgen` also
needs a working clang/libclang installation.

Download Nvidia DLSS SDK - https://github.com/NVIDIA/DLSS/archive/refs/tags/v310.4.0.zip
VulkanSDK - https://sdk.lunarg.com/sdk/download/1.4.363.0
LLVM for clang - https://github.com/llvm/llvm-project/releases/

Then set the env variables e.g on Windows:

```powershell
$env:DLSS_SDK = 'E:\SDK\DLSS-310.4.0'
$env:VULKAN_SDK = 'E:\SDK\VulkanSDK\1.4.363.0'
$env:LIBCLANG_PATH = "E:\SDK\LLVM\bin"   
$env:WOW_DLSS = '1'
cargo run -p benilla --features dlss
```

`DLSS_SDK` must provide `include/` and the Windows `lib/Windows_x86_64/x64` import library used by
`dlss_wgpu`. The build deliberately does not carry a machine-specific SDK path. Bevy normally
puts `DlssInitPlugin` in `DefaultPlugins`; Benilla removes that static entry and adds it manually
with its `DlssProjectId` only for `WOW_DLSS=1`, avoiding both duplicate registration and NGX
initialization in a feature-built normal launch. A normal
`cargo run -p benilla` builds and launches without the NVIDIA SDK; setting `WOW_DLSS=1` there logs
the normal-renderer fallback and does not alter its existing backend, MSAA, static-GX or liquid
configuration.

For Windows distribution or a direct executable run, copy the SDK's
`lib/Windows_x86_64/rel/nvngx_dlss.dll` beside the executable and include the DLSS license/copyright
text required by the NVIDIA SDK. This prototype does not use Ray Reconstruction, so it does not
need `nvngx_dlssd.dll`.

## Runtime behavior

`WOW_DLSS=1` requests Vulkan before Bevy creates the renderer. If Bevy does not publish
`DlssSuperResolutionSupported` (no supported RTX/NGX/Vulkan path), no world camera receives a
`Dlss` component and rendering continues without DLSS.

On a supported system, only `WorldCamera` receives Bevy's `Dlss` component in Quality mode and
`Msaa::Off`. Bevy supplies jitter, depth/motion inputs and `MainPassResolutionOverride`; the render
log prints the native output and lower internal world resolution. FrameXML and the 2D/UI cameras
remain at native resolution.

The current proof disables retained `static_gx` for a requested DLSS launch because it lacks the
prepass data DLSS needs. It hides liquid surfaces only after DLSS support is confirmed, because the
liquid renderer has not yet been adapted to the lower main-pass resolution. These are intentional,
temporary safeguards, not implementations of static-GX or water temporal support.

The render graph is explicitly ordered as world/depth-motion, Bevy DLSS Super Resolution, the
existing FFXGlow node, Bloom/tonemapping, then UI composition. No FFXGlow shader or effect behavior
changes in this phase.
