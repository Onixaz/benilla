// The M2/WMO prepass deformation. Bevy owns target allocation and the PBR prepass alpha path;
// this shader supplies the paired current and previous positions for animation that Bevy cannot
// infer from an entity transform.

#import benilla_assets::wow_model_skin::wow_skin_affine
#import bevy_pbr::{
    mesh_functions,
    mesh_view_bindings::view,
    pbr_prepass_functions,
    prepass_bindings::previous_view_uniforms,
    prepass_io,
    view_transformations::position_world_to_clip,
}

// This must match WowLight in wow_model.wgsl. It has the same full prefix because the shared
// storage binding contains unrelated lighting data before its rig-table tail.
struct WowLight {
    light_ambient: vec4<f32>,
    light_diffuse: vec4<f32>,
    light_sun: vec4<f32>,
    light_spec: vec4<f32>,
    fog_color: vec4<f32>,
    fog_params: vec4<f32>,
    sh_c10_r: vec4<f32>,
    sh_c10_g: vec4<f32>,
    sh_c10_b: vec4<f32>,
    sh_c13_r: vec4<f32>,
    sh_c13_g: vec4<f32>,
    sh_c13_b: vec4<f32>,
    sh_c16: vec4<f32>,
    _water: array<vec4<f32>, 4>,
    grade: vec4<f32>,
    wmo_fog_color: vec4<f32>,
    wmo_fog_params: vec4<f32>,
    point_count: vec4<f32>,
    points: array<vec4<f32>, 512>,
    prop_probes: array<vec4<f32>, 57344>,
    rig_table: array<u32, 2048>,
    rig_tint: array<u32, 2048>,
    rig_origin: array<vec4<f32>, 2048>,
    matanim: array<vec4<f32>, 2048>,
    water_clip: array<vec2<f32>, 2048>,
    previous_rig_origin: array<vec4<f32>, 2048>,
    palettes: array<vec4<f32>>,
};
@group(#{MATERIAL_BIND_GROUP}) @binding(90) var<storage, read> wow_light: WowLight;

const WOW_RIG_CURRENT_PALETTE_ROW_OFFSET: u32 = 0u;
const WOW_RIG_PREVIOUS_PALETTE_ROW_OFFSET: u32 = 3u * 131072u;

struct WowPrepassVertex {
    @builtin(instance_index) instance_index: u32,
#ifdef VERTEX_POSITIONS
    @location(0) position: vec3<f32>,
#endif
#ifdef VERTEX_UVS_A
    @location(2) uv: vec2<f32>,
#endif
#ifdef VERTEX_UVS_B
    @location(3) uv_b: vec2<f32>,
#endif
#ifdef VERTEX_COLORS
    @location(5) color: vec4<f32>,
#endif
#ifdef WOW_RIG_SKIN
    @location(10) joint_indices: vec4<u32>,
    @location(11) joint_weights: vec4<f32>,
#endif
}

// The stock prepass output layout, plus two values used only by the rigged motion-vector branch.
struct WowPrepassOut {
    @builtin(position) position: vec4<f32>,
#ifdef VERTEX_UVS_A
    @location(0) uv: vec2<f32>,
#endif
#ifdef VERTEX_UVS_B
    @location(1) uv_b: vec2<f32>,
#endif
    @location(4) world_position: vec4<f32>,
#ifdef MOTION_VECTOR_PREPASS
    @location(5) previous_world_position: vec4<f32>,
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    @location(7) @interpolate(flat) instance_index: u32,
#endif
#ifdef WOW_RIG_SKIN
#ifdef MOTION_VECTOR_PREPASS
    @location(10) current_view_position: vec3<f32>,
    @location(11) previous_view_position: vec3<f32>,
#endif
#endif
}

fn wow_rig_slot(instance_index: u32) -> u32 {
    return (mesh_functions::get_tag(instance_index) >> 19u) & 0x7ffu;
}

fn wow_skin_model(
    instance_index: u32,
    indices: vec4<u32>,
    weights: vec4<f32>,
    palette_row_offset: u32,
) -> mat4x4<f32> {
    let base = wow_light.rig_table[wow_rig_slot(instance_index)];
    let b0 = palette_row_offset + 3u * (base + indices.x);
    let b1 = palette_row_offset + 3u * (base + indices.y);
    let b2 = palette_row_offset + 3u * (base + indices.z);
    let b3 = palette_row_offset + 3u * (base + indices.w);
    return wow_skin_affine(
        wow_light.palettes[b0], wow_light.palettes[b0 + 1u], wow_light.palettes[b0 + 2u],
        wow_light.palettes[b1], wow_light.palettes[b1 + 1u], wow_light.palettes[b1 + 2u],
        wow_light.palettes[b2], wow_light.palettes[b2 + 1u], wow_light.palettes[b2 + 2u],
        wow_light.palettes[b3], wow_light.palettes[b3 + 1u], wow_light.palettes[b3 + 2u],
        weights,
    );
}

fn view_rotation(view_from_world: mat4x4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(
        view_from_world[0].xyz,
        view_from_world[1].xyz,
        view_from_world[2].xyz,
    );
}

@vertex
fn vertex(vertex: WowPrepassVertex) -> WowPrepassOut {
    var out: WowPrepassOut;
    let mesh_world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);

#ifdef WOW_RIG_SKIN
    let slot = wow_rig_slot(vertex.instance_index);
    let current_frame_from_local = wow_skin_model(
        vertex.instance_index,
        vertex.joint_indices,
        vertex.joint_weights,
        WOW_RIG_CURRENT_PALETTE_ROW_OFFSET,
    );
    let current_camera_relative =
        (current_frame_from_local * vec4<f32>(vertex.position, 1.0)).xyz
        + (wow_light.rig_origin[slot].xyz - view.world_position);
    let current_view_position = view_rotation(view.view_from_world) * current_camera_relative;
    // Identical to wow_model.wgsl: only a nearby value reaches the projection multiplication.
    out.position = view.clip_from_view * vec4<f32>(current_view_position, 1.0);
    out.world_position = vec4<f32>(current_camera_relative + view.world_position, 1.0);

#ifdef MOTION_VECTOR_PREPASS
    let previous_frame_from_local = wow_skin_model(
        vertex.instance_index,
        vertex.joint_indices,
        vertex.joint_weights,
        WOW_RIG_PREVIOUS_PALETTE_ROW_OFFSET,
    );
    let previous_world_from_view = inverse(previous_view_uniforms.view_from_world);
    let previous_camera_position = previous_world_from_view[3].xyz;
    let previous_camera_relative =
        (previous_frame_from_local * vec4<f32>(vertex.position, 1.0)).xyz
        + (wow_light.previous_rig_origin[slot].xyz - previous_camera_position);
    out.previous_world_position =
        vec4<f32>(previous_camera_relative + previous_camera_position, 1.0);
    out.current_view_position = current_view_position;
    out.previous_view_position =
        view_rotation(previous_view_uniforms.view_from_world) * previous_camera_relative;
#endif
#else
    out.world_position = mesh_functions::mesh_position_local_to_world(
        mesh_world_from_local,
        vec4<f32>(vertex.position, 1.0),
    );
    out.position = position_world_to_clip(out.world_position.xyz);
#ifdef MOTION_VECTOR_PREPASS
    out.previous_world_position = mesh_functions::mesh_position_local_to_world(
        mesh_functions::get_previous_world_from_local(vertex.instance_index),
        vec4<f32>(vertex.position, 1.0),
    );
#endif
#endif

#ifdef VERTEX_UVS_A
    out.uv = vertex.uv;
#endif
#ifdef VERTEX_UVS_B
    out.uv_b = vertex.uv_b;
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex.instance_index;
#endif
    return out;
}

// The standard PBR prepass fragment is retained for alpha discard. Its absolute-world motion
// helper is replaced only for custom rigs: world-space cancellation at WoW map coordinates would
// discard the precision the forward path deliberately protects.
#ifdef PREPASS_FRAGMENT
@fragment
fn fragment(in: WowPrepassOut) -> prepass_io::FragmentOutput {
    var stock: prepass_io::VertexOutput;
    stock.position = in.position;
    stock.world_position = in.world_position;
#ifdef VERTEX_UVS_A
    stock.uv = in.uv;
#endif
#ifdef VERTEX_UVS_B
    stock.uv_b = in.uv_b;
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    stock.instance_index = in.instance_index;
#endif
    pbr_prepass_functions::prepass_alpha_discard(stock);

    var out: prepass_io::FragmentOutput;
#ifdef MOTION_VECTOR_PREPASS
#ifdef WOW_RIG_SKIN
    let current_unjittered_clip_from_view =
        view.unjittered_clip_from_world * inverse(view.view_from_world);
    let current_clip_t = current_unjittered_clip_from_view
        * vec4<f32>(in.current_view_position, 1.0);
    let previous_clip_t = previous_view_uniforms.clip_from_view
        * vec4<f32>(in.previous_view_position, 1.0);
    out.motion_vector = (current_clip_t.xy / current_clip_t.w
        - previous_clip_t.xy / previous_clip_t.w) * vec2<f32>(0.5, -0.5);
#else
    out.motion_vector = pbr_prepass_functions::calculate_motion_vector(
        in.world_position,
        in.previous_world_position,
    );
#endif
#endif
    return out;
}
#endif
