// The world-only temporal diagnostic. The prepass stores UV-space motion in Rg16Float; preserve
// the conventional signed R/G display (0.5 is zero) and use blue for magnitude so quiet pixels
// remain obvious rather than looking like missing output.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var motion_vectors: texture_2d<f32>;

@fragment
fn fs_motion_vectors(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let motion = textureLoad(motion_vectors, vec2<i32>(in.position.xy), 0).xy;
    // Eight pixels at a 256-pixel render target fills one signed colour channel. Larger movement
    // intentionally saturates, while small idle/limb changes remain visible.
    let scaled = motion * 32.0;
    let direction = clamp(vec2<f32>(0.5) + scaled, vec2<f32>(0.0), vec2<f32>(1.0));
    let magnitude = clamp(length(scaled), 0.0, 1.0);
    return vec4<f32>(direction.x, direction.y, 0.12 + 0.88 * magnitude, 1.0);
}
