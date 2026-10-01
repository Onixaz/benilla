// MONKEY (torch shadows, Phase 1): the vertex-only depth pass for one interior fixture.
//
// Renders the retained interior caster geometry (position only, stride 12) from a fixture's
// down-looking reverse-Z `view_proj` into one layer of the torch depth 2D-array texture. No
// fragment stage: only depth is written (clear 0.0, GreaterEqual — reverse-Z). See
// `static_gx/torch_depth.rs` for the pipeline and `static_gx.wgsl` for the sampling side.

struct TorchView {
    view_proj: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> torch_view: TorchView;

@vertex
fn vertex(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return torch_view.view_proj * vec4<f32>(position, 1.0);
}
