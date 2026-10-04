// The character-shadow proxy's shadow-pass vertex stage: a unit part's own GPU mesh, skinned from
// the same palette rows `wow_model.wgsl` draws it with, so the caster is the visible pose with no
// CPU skin and no upload. Bevy's prepass I/O (`prepass_io::VertexOutput`), so the default prepass
// fragment still links where unclipped depth is emulated.
//
// The shared light buffer is read as raw words at the region offsets `rig_palette` publishes,
// rather than redeclaring `wow_model.wgsl`'s whole `WowLight` struct here.

#import bevy_pbr::{
    mesh_functions,
    prepass_io::VertexOutput,
    view_transformations::position_world_to_clip,
}

// Word offsets into `light`: x = the rig slot table, y = the rig origins, z = the palette rows.
@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> regions: vec4<u32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<storage, read> light: array<u32>;

struct SkinVertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
#ifdef WOW_RIG_SKIN
    @location(10) joint_indices: vec4<u32>,
    @location(11) joint_weights: vec4<f32>,
#endif
}

fn light_vec4(word: u32) -> vec4<f32> {
    return vec4<f32>(
        bitcast<f32>(light[word]),
        bitcast<f32>(light[word + 1u]),
        bitcast<f32>(light[word + 2u]),
        bitcast<f32>(light[word + 3u]),
    );
}

#ifdef WOW_RIG_SKIN
// Palette row `row` (3 rows a bone, each one `vec4` of the rig-relative affine).
fn palette_row(row: u32) -> vec4<f32> {
    return light_vec4(regions.z + 4u * row);
}
#endif

@vertex
fn vertex(vertex: SkinVertex) -> VertexOutput {
    var out: VertexOutput;
    let p = vec4<f32>(vertex.position, 1.0);
#ifdef WOW_RIG_SKIN
    // `wow_model.wgsl`'s `wow_skin_model`: the slot rides MeshTag bits 19..=29.
    let slot = (mesh_functions::get_tag(vertex.instance_index) >> 19u) & 0x7ffu;
    let base = light[regions.x + slot];
    let i = vertex.joint_indices;
    let w = vertex.joint_weights;
    let b0 = 3u * (base + i.x);
    let b1 = 3u * (base + i.y);
    let b2 = 3u * (base + i.z);
    let b3 = 3u * (base + i.w);
    let r0 = w.x * palette_row(b0) + w.y * palette_row(b1)
        + w.z * palette_row(b2) + w.w * palette_row(b3);
    let r1 = w.x * palette_row(b0 + 1u) + w.y * palette_row(b1 + 1u)
        + w.z * palette_row(b2 + 1u) + w.w * palette_row(b3 + 1u);
    let r2 = w.x * palette_row(b0 + 2u) + w.y * palette_row(b1 + 2u)
        + w.z * palette_row(b2 + 2u) + w.w * palette_row(b3 + 2u);
    // The rows are relative to the slot's origin; add it back for the light's world-space view.
    let origin = light_vec4(regions.y + 4u * slot).xyz;
    let world = vec3<f32>(dot(r0, p), dot(r1, p), dot(r2, p)) + origin;
#else
    let world = (mesh_functions::get_world_from_local(vertex.instance_index) * p).xyz;
#endif
    out.world_position = vec4<f32>(world, 1.0);
    out.position = position_world_to_clip(world);
#ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
    out.unclipped_depth = out.position.z;
    out.position.z = min(out.position.z, 1.0);
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex.instance_index;
#endif
    return out;
}
