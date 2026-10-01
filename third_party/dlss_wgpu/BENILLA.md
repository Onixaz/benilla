# dlss_wgpu SDK-header compatibility patch

Origin: `dlss_wgpu` 2.0.0 from crates.io, dual-licensed MIT or Apache-2.0. The original license
files are retained beside this note.

Benilla 0.18.1 uses this version because it matches Bevy's wgpu 27 integration. Its original NGX
initializers hard-code a 16-entry `NVSDK_NGX_VK_GBuffer::pInAttrib` array. The installed NVIDIA
headers expose 17 entries and add two optional Ray Reconstruction responsivity-mask fields, which
makes bindgen-generated Rust bindings fail to compile against the original literals.

The local patch changes only those initializers:

- zero-initialize the G-buffer structure, so its pointer-array length is inferred from the active
  SDK headers;
- use a zeroed struct-update tail for the two newer optional Ray Reconstruction fields. Zero/null
  means the inputs are unused, matching the original behavior. Headers without those fields have
  no omitted fields.

No DLSS feature behavior is added or changed. Remove this patch only when upgrading Bevy and its
`dlss_wgpu` dependency together to a version that supports the installed NVIDIA SDK headers.
