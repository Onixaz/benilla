# dlss_wgpu SDK-header compatibility patch
Origin: `dlss_wgpu` 2.0.0 from crates.io, dual-licensed MIT or Apache-2.0. The original license
files are retained beside this note.

Benilla 0.18.1 uses this version because it matches Bevy's wgpu 27 integration. Its original NGX
initializers hard-code a 16-entry `NVSDK_NGX_VK_GBuffer::pInAttrib` array. The installed NVIDIA
headers expose 17 entries and add two optional Ray Reconstruction responsivity-mask fields, which
makes bindgen-generated Rust bindings fail to compile against the original literals.

The local patch changes only those initializers and exposes one missing upstream exposure form:

- zero-initialize the G-buffer structure, so its pointer-array length is inferred from the active
  SDK headers;
- use a zeroed struct-update tail for the two newer optional Ray Reconstruction fields. Zero/null
  means the inputs are unused, matching the original behavior. Headers without those fields have
  no omitted fields.
- add `DlssSuperResolutionExposure::Fixed`, which sends explicit scalar exposure and
  pre-exposure values with no exposure texture. Benilla uses the neutral `(1.0, 1.0)` form because
  its gamma-space world path has no exposure pass; upstream's `Automatic` form remains unchanged.

Remove the SDK-header portion only when upgrading Bevy and its `dlss_wgpu` dependency together to
a version that supports the installed NVIDIA SDK headers. Keep or upstream the fixed-exposure form
while Benilla needs its non-HDR DLSS experiment.
