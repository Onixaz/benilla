// MONKEY (world shadows): alpha-tested foliage shadow caster — the PREPASS fragment.
//
// The proxy caster is invisible in the camera's forward pass (its forward fragment `discard`s every
// fragment — `shadow_caster.wgsl`). Its material is `AlphaMode::Mask`, which sets Bevy's
// `MAY_DISCARD` pipeline key; paired with this EXPLICIT prepass fragment shader that decides Bevy
// runs a fragment during the directional-light SHADOW pass (which is otherwise depth-only). So a
// leaf card can discard its transparent texels and cast a leaf-SHAPED silhouette instead of the
// solid rectangular box a positions-only proxy throws.
//
// The leaf sheet is bound as the material's own texture at group `MATERIAL_BIND_GROUP` — the
// prepass material slot (index 3; substituted by the prepass pipeline). Sampling the SAME image the
// forward pass draws keeps the clamp/repeat address mode identical, so the shadow silhouette
// matches the visible foliage exactly (a cutout card's UVs run outside 0..1 into the transparent
// border — sampled with the wrong wrap mode the margin wraps back into the opaque middle and the
// card casts solid again).
//
// Structure mirrors Bevy's own `pbr_prepass.wgsl`: a `FragmentOutput`-returning entry under
// `PREPASS_FRAGMENT` (the unclipped-depth-emulation path for GPUs without `depth_clip_control`) and
// a void entry otherwise (the common desktop depth-only path); both run the same cutout discard.

#import bevy_pbr::prepass_io

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var leaf_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var leaf_sampler: sampler;

// WoW ≤ WotLK alpha-test key (`benilla_assets::materials::VANILLA_ALPHA_KEY_REF` = 224/255): the
// value below which a cutout fragment is discarded. The exact ref the forward cutout pass uses.
const VANILLA_ALPHA_KEY: f32 = 0.8784314;

// Sample the leaf sheet at the interpolated UV and discard transparent texels, exactly as the
// forward cutout does. Sampled BEFORE any `discard` so the texture access stays in uniform control
// flow. A caster mesh always carries `UV_0` (`VERTEX_UVS_A`); lacking it, nothing discards (the
// card falls back to a solid cast rather than vanishing).
fn cutout_discard(in: prepass_io::VertexOutput) {
#ifdef VERTEX_UVS_A
    let alpha = textureSample(leaf_texture, leaf_sampler, in.uv).a;
    if alpha < VANILLA_ALPHA_KEY {
        discard;
    }
#endif
}

#ifdef PREPASS_FRAGMENT
// A GPU without `depth_clip_control`: Bevy emulates the directional-shadow view's unclipped depth
// by writing `frag_depth` in the fragment (`prepass/mod.rs`, `UNCLIPPED_DEPTH_ORTHO_EMULATION`).
// Mirror the default prepass fragment's single field so that depth-clamp fix still applies here.
@fragment
fn fragment(in: prepass_io::VertexOutput) -> prepass_io::FragmentOutput {
    cutout_discard(in);
    var out: prepass_io::FragmentOutput;
#ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
    out.frag_depth = in.unclipped_depth;
#endif // UNCLIPPED_DEPTH_ORTHO_EMULATION
    return out;
}
#else
// The common desktop path: the shadow pass is depth-only (no color targets), so the fragment
// returns nothing and exists ONLY to run the cutout discard against the depth write.
@fragment
fn fragment(in: prepass_io::VertexOutput) {
    cutout_discard(in);
}
#endif // PREPASS_FRAGMENT
