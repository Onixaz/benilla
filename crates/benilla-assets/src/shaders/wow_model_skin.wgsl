#define_import_path benilla_assets::wow_model_skin

// The temporal prepass's affine M2 blend mirrors the forward pass exactly. It owns neither the
// buffer declaration nor the palette-bank choice, so its caller keeps that state explicit while
// the established forward shader stays unchanged.
fn wow_skin_affine(
    r00: vec4<f32>, r01: vec4<f32>, r02: vec4<f32>,
    r10: vec4<f32>, r11: vec4<f32>, r12: vec4<f32>,
    r20: vec4<f32>, r21: vec4<f32>, r22: vec4<f32>,
    r30: vec4<f32>, r31: vec4<f32>, r32: vec4<f32>,
    weights: vec4<f32>,
) -> mat4x4<f32> {
    let r0 = weights.x * r00 + weights.y * r10 + weights.z * r20 + weights.w * r30;
    let r1 = weights.x * r01 + weights.y * r11 + weights.z * r21 + weights.w * r31;
    let r2 = weights.x * r02 + weights.y * r12 + weights.z * r22 + weights.w * r32;
    // r0/r1/r2 are affine rows; WGSL matrices are column-major.
    return mat4x4<f32>(
        vec4<f32>(r0.x, r1.x, r2.x, 0.0),
        vec4<f32>(r0.y, r1.y, r2.y, 0.0),
        vec4<f32>(r0.z, r1.z, r2.z, 0.0),
        vec4<f32>(r0.w, r1.w, r2.w, 1.0),
    );
}
