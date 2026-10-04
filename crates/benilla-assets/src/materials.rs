//! The render materials: the four `ExtendedMaterial`s the world is drawn with (terrain splat,
//! M2/WMO model, WDL far band, liquid surface) and the WGSL each one binds.
//!
//! The WGSL is embedded (`embedded://benilla_assets/shaders/…`), not served: a relative path would
//! resolve against the host binary's `AssetPlugin::file_path` and render nothing elsewhere.

use bevy::image::Image;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{
    ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline,
    MeshPipelineKey,
};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, BlendComponent, BlendFactor, BlendOperation, BlendState, Buffer, ColorWrites,
    CompareFunction, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::{load_shader_library, ShaderRef};

/// Compile the WGSL files into the binary and register them under
/// `embedded://benilla_assets/shaders/…`. Call **after** Bevy's `AssetPlugin` (it fills the
/// registry that plugin creates); [`crate::register_asset_loaders`] already does.
pub fn register_shaders(app: &mut App) {
    bevy::asset::embedded_asset!(app, "shaders/terrain.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/wow_model_skin.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/wow_model.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/wow_model_prepass.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/wdl.wgsl");
    bevy::asset::embedded_asset!(app, "shaders/liquid.wgsl");
    // MONKEY (shadow hook): register the importable shadow-contribution library `benilla::shadow_hook`
    // (loaded eagerly so `#import benilla::shadow_hook` resolves in terrain/model/static_gx). This is
    // the ONE place the realtime directional-shadow term lives; the receivers just call into it.
    load_shader_library!(app, "shaders/shadow_hook.wgsl");
    // MONKEY (p0 MonkeyFrame): `benilla::monkey_frame`, the struct every `WowLight` mirror declares
    // after its point table (LIGHTING.md "Invariants").
    load_shader_library!(app, "shaders/monkey_frame.wgsl");
    // MONKEY (p0 fog hook): `benilla::fog_hook`, the one distance-fog law all seven receivers call.
    load_shader_library!(app, "shaders/fog_hook.wgsl");
    // MONKEY (post): shared opt-in HDR multipliers; tier 0 is an exact identity.
    load_shader_library!(app, "shaders/emissive_hook.wgsl");
    // MONKEY (wind): `benilla::wind_hook`, shared vertex displacement for grass and foliage.
    load_shader_library!(app, "shaders/wind_hook.wgsl");
    // MONKEY (wet): `benilla::wet_hook`, the rain-on-surfaces look terrain/model/static_gx call.
    load_shader_library!(app, "shaders/wet_hook.wgsl");
    // GFX (moonlight): `benilla::moonlight_hook`, the additive moon term the receivers call.
    load_shader_library!(app, "shaders/moonlight_hook.wgsl");
    // MONKEY (enhanced water): the optional water module, `benilla::enhanced_water` (WATER.md).
    load_shader_library!(app, "shaders/enhanced_water.wgsl");
}

/// The WDL far-band shader's source, for the tests that live beside the renderer (`wdl.rs`).
pub const WDL_WGSL: &str = include_str!("shaders/wdl.wgsl");

/// Alpha-test reference for blend mode 1 (`Blend_AlphaKey`): 224 in the reference's
/// per-blend-mode table at `0x85ad20`, `{0, 224, 1, 1, 1, 1, 1, 0, 0, 0, 0}`. Must stay in sync
/// with `VANILLA_ALPHA_KEY` in `shaders/wow_model.wgsl`.
pub const VANILLA_ALPHA_KEY_REF: f32 = 224.0 / 255.0;

/// `StandardMaterial` plus the per-tile layer-blend extension.
pub type TerrainMaterial = ExtendedMaterial<StandardMaterial, TerrainExtension>;

/// World models (doodads, WMOs, creatures, GameObjects) lit in gamma space like terrain, not by
/// PBR; `StandardMaterial` carries only the texture, alpha mode and culling.
pub type WowModelMaterial = ExtendedMaterial<StandardMaterial, WowModelExt>;

/// Pipeline-specialization key for [`WowModelExt`]: the depth and blend states Bevy's `AlphaMode`
/// cannot express. `fade` (`model_flags.y`) is the M2 doodad fade twin and `clutter`
/// (`clutter_fade.w`) ground clutter; both fade opacity with depth-write on, as the reference does.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct WowModelKey {
    fade: bool,
    clutter: bool,
    /// Additive glow card (`clutter_fade.z` bit 2).
    additive: bool,
    /// M2 render flag 0x10, no depth write (`clutter_fade.z` bit 0).
    no_depth_write: bool,
    /// M2 render flag 0x08, no depth test (`clutter_fade.z` bit 1).
    no_depth_test: bool,
    /// The multiply blends, Mod and Mod2x (`clutter_fade.z` bits 7/8).
    modulate: bool,
    modulate2x: bool,
    /// The depth-prime twin (`clutter_fade.z` bit 9, `model_render::zfill_material`).
    zfill: bool,
    /// Far side of the water plane (`clutter_fade.z` bit 11).
    far_side: bool,
    /// The WMO-skybox lane (`clutter_fade.z` bit 13, `model_render::SKY_DEPTH_MARKER`), a key axis
    /// because only this model's depth is pinned.
    sky_depth: bool,
    /// MONKEY (skybox): a second texture stage (`stage1.x`), the `WOW_STAGE1` shader def.
    stage1: bool,
    // The WMO batch order is deliberately not a key axis (a pipeline per batch index stalls a
    // city's first sight): the file-order layering (`0x6b4f10`/`0x6b5190`) rides `sun_scale.y`.
}

impl From<&WowModelExt> for WowModelKey {
    fn from(e: &WowModelExt) -> Self {
        let markers = e.clutter_fade.z as u32;
        Self {
            fade: e.model_flags.y > 0.5,
            clutter: e.clutter_fade.w > 0.5,
            additive: markers & 4 != 0,
            no_depth_write: markers & 1 != 0,
            no_depth_test: markers & 2 != 0,
            modulate: markers & 0x80 != 0,
            modulate2x: markers & 0x100 != 0,
            zfill: markers & 0x200 != 0,
            far_side: markers & 0x800 != 0,
            sky_depth: markers & 0x2000 != 0,
            stage1: e.stage1.x > 0.5,
        }
    }
}

/// The model material's extension. Every `Vec4` uniform packs into binding 100, one buffer entry,
/// to stay under Metal's 16-buffer vertex-stage cap (Bevy 0.18 ignores `visibility` on uniforms);
/// `ModelParams` in `wow_model.wgsl` must match the field order.
#[derive(Asset, AsBindGroup, Clone, TypePath)]
#[bind_group_data(WowModelKey)]
pub struct WowModelExt {
    /// Clutter fade: `x`, `y` = full-opacity and fully-faded view depth (yd; the ~70 yd horizon),
    /// `w` = clutter, all zero off clutter; `z` = the batch marker bits.
    #[uniform(100)]
    pub clutter_fade: Vec4,
    /// `x` = WMO, `y` = the fade blend twin, `z` = interior, `w` = unlit fullbright.
    #[uniform(100)]
    pub model_flags: Vec4,
    /// `x` = the terrain-shade selector, `y` = the WMO batch order (the clip-z layering nudge),
    /// `zw` = the UV-animation offset.
    #[uniform(100)]
    pub sun_scale: Vec4,
    /// `xyz` = the M2Color tint of a batch whose colour track animates, identity otherwise; `w` =
    /// the WMO interior batch class (`0x6b5190`): `0` exterior, `1` INT, `2` TRANS.
    #[uniform(100)]
    pub tint: Vec4,
    /// WMO glass, `0` on M2: `xyz` = the MOMT SIDN night emissive (ramp `0x6b4090`), `w` = the MOMT
    /// WINDOW flag, the brighter interior light (`0x6d37e0`, ambient +16/255).
    #[uniform(100)]
    pub sidn: Vec4,
    /// Rows of the shared light buffer's `matanim` region, `0` = identity: `x` = UV scroll, `y` =
    /// tint, `z` = the texture-transform affine, `w` = the UI tile's cell clip.
    #[uniform(100)]
    pub anim_slots: Vec4,
    /// MONKEY (skybox): a two-texture batch's stage 1: `x` = 0 none, 1 Mod, 2 Mod2x; `z`/`w` = the
    /// `matanim` rows of its translation and affine, read on UV set B. `0` on every other batch.
    #[uniform(100)]
    pub stage1: Vec4,
    /// MONKEY (skybox): stage 1's texture; `None` (the fallback image) unless `stage1.x > 0`.
    #[texture(94)]
    #[sampler(95)]
    pub stage1_texture: Option<Handle<Image>>,
    /// The shared global light (`lighting::global_light`), updated in place once a frame; the
    /// vertex stage reads its point-light table, since Bevy's clusterable lights are fragment-only.
    #[storage(90, read_only, buffer, visibility(vertex, fragment))]
    pub light_buf: Buffer,
    /// MONKEY (torch shadows Phase 3A): the interior torch depth-map ARRAY — the one shared
    /// `Depth32Float` 512×512×24 `Image` (4 fixtures × 6 cube faces) the torch depth node renders
    /// into each frame (`benilla_world::static_gx::torch_depth`, `TorchDepthImage`). Bound as a
    /// depth 2D-array + the image's own `GreaterEqual` comparison sampler (the `Image`'s
    /// `ImageSamplerDescriptor { compare: Some(GreaterEqual) }` — bevy builds a real comparison
    /// sampler from it; the derive's prepare-time check requires the format's sample type to be
    /// `Depth`, which `Depth32Float` is). The image is created ONCE at app startup regardless of
    /// the cvars — an absent image would stall every model material's bind group on
    /// `RetryNextUpdate` and blank every model. Fragment-only, like static_gx's group 3.
    #[texture(
        91,
        dimension = "2d_array",
        sample_type = "depth",
        visibility(fragment)
    )]
    #[sampler(92, sampler_type = "comparison", visibility(fragment))]
    pub torch_depth: Handle<Image>,
    /// MONKEY (static torch cache): the ≤16-fixture torch TABLE — the same 6416-byte
    /// `TorchTableUniform` bytes static_gx's group-3 uniform carries (count / positions[16] /
    /// view_projs[96]), as a SEPARATE shared raw buffer (`SharedTorchBuffer`) rewritten in place
    /// every frame from `TorchShadowViews`. A raw `Buffer` like `light_buf`, deliberately NOT a
    /// `#[uniform]` field (a per-frame-mutated uniform re-prepares every material every frame)
    /// and NOT a region of the shared light blob (whose `LightStd430` mirror + the booth packer
    /// would all have to grow in lock-step). `wow_model.wgsl` reads it as
    /// `var<storage, read> torch_table` (std430 of this struct == the std140 6416 bytes: every
    /// member is 16-aligned). Set once, never mutated.
    #[storage(93, read_only, buffer, visibility(fragment))]
    pub torch_buf: Buffer,
}

/// MONKEY (torch shadows Phase 3A): the two torch-receiver bindings every [`WowModelExt`] shares —
/// threaded into the material builders beside the shared light `Buffer`, and stored on
/// `WorldAssets` the same way. Both come from the client's startup resources
/// (`benilla_world::static_gx::{TorchDepthImage, SharedTorchBuffer}`); this crate only clones them
/// into materials, which is why it takes the raw handle + buffer and not those resources (the same
/// severing `WorldAssets::open` does for the light buffer, decision 1164).
#[derive(Clone)]
pub struct TorchBinds {
    /// The shared torch depth array image (→ `WowModelExt::torch_depth`).
    pub depth: Handle<Image>,
    /// The shared 6416-byte torch table buffer (→ `WowModelExt::torch_buf`).
    pub table: Buffer,
}

impl MaterialExtension for WowModelExt {
    /// Bevy's mesh vertex plus the point-light term, per vertex like the reference's FFP.
    fn vertex_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/wow_model.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/wow_model.wgsl".into()
    }

    fn prepass_vertex_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/wow_model_prepass.wgsl".into()
    }

    fn prepass_fragment_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/wow_model_prepass.wgsl".into()
    }

    /// The reference writes depth for every M2 batch, transparent ones too, and tests `LEQUAL`,
    /// unless render flag 0x10 (no write) or 0x08 (no test) clears it (`0x70c190`).
    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // `WOW_RIG_SKIN` skins from the shared buffer's palette. The base layout lacks our joint
        // attributes: rebuild it with Bevy's conditionals (mesh.rs, locations 0-5) plus 10/11.
        if layout.0.contains(crate::ATTRIBUTE_WOW_JOINT_INDEX) {
            descriptor.vertex.shader_defs.push("WOW_RIG_SKIN".into());
            // The FRAGMENT stage keys on the same def: a skinned unit takes the whole-unit
            // ground-point shadow sample (one fetch at its rig origin) instead of the
            // per-fragment fetch, which would read the unit's own caster copy as an occluder.
            if let Some(fragment) = descriptor.fragment.as_mut() {
                fragment.shader_defs.push("WOW_RIG_SKIN".into());
            }
            let mut attrs = Vec::with_capacity(7);
            for (attr, loc) in [
                (Mesh::ATTRIBUTE_POSITION, 0),
                (Mesh::ATTRIBUTE_NORMAL, 1),
                (Mesh::ATTRIBUTE_UV_0, 2),
                (Mesh::ATTRIBUTE_UV_1, 3),
                (Mesh::ATTRIBUTE_TANGENT, 4),
                (Mesh::ATTRIBUTE_COLOR, 5),
            ] {
                if layout.0.contains(attr) {
                    attrs.push(attr.at_shader_location(loc));
                }
            }
            attrs.push(crate::ATTRIBUTE_WOW_JOINT_INDEX.at_shader_location(10));
            attrs.push(crate::ATTRIBUTE_WOW_JOINT_WEIGHT.at_shader_location(11));
            descriptor.vertex.buffers = vec![layout.0.get_layout(&attrs)?];
        }
        // A merged blob computes the doodad fade per vertex from its fade sphere; same rebuild,
        // and merged blobs never skin.
        if layout.0.contains(crate::ATTRIBUTE_WOW_FADE_SPHERE) {
            descriptor.vertex.shader_defs.push("WOW_MERGED_FADE".into());
            if let Some(fragment) = descriptor.fragment.as_mut() {
                fragment.shader_defs.push("WOW_MERGED_FADE".into());
            }
            let mut attrs = Vec::with_capacity(6);
            for (attr, loc) in [
                (Mesh::ATTRIBUTE_POSITION, 0),
                (Mesh::ATTRIBUTE_NORMAL, 1),
                (Mesh::ATTRIBUTE_UV_0, 2),
                (Mesh::ATTRIBUTE_UV_1, 3),
                (Mesh::ATTRIBUTE_TANGENT, 4),
                (Mesh::ATTRIBUTE_COLOR, 5),
            ] {
                if layout.0.contains(attr) {
                    attrs.push(attr.at_shader_location(loc));
                }
            }
            attrs.push(crate::ATTRIBUTE_WOW_FADE_SPHERE.at_shader_location(12));
            // An interior-prop blob's baked SH-probe slot replaces the per-entity MeshTag payload.
            if layout.0.contains(crate::ATTRIBUTE_WOW_MERGED_SLOT) {
                descriptor.vertex.shader_defs.push("WOW_MERGED_SLOT".into());
                if let Some(fragment) = descriptor.fragment.as_mut() {
                    fragment.shader_defs.push("WOW_MERGED_SLOT".into());
                }
                attrs.push(crate::ATTRIBUTE_WOW_MERGED_SLOT.at_shader_location(13));
            }
            descriptor.vertex.buffers = vec![layout.0.get_layout(&attrs)?];
        }
        if let Some(ds) = descriptor.depth_stencil.as_mut() {
            ds.depth_write_enabled = !key.bind_group_data.no_depth_write;
            if key.bind_group_data.no_depth_test {
                ds.depth_compare = CompareFunction::Always;
            }
        }
        // A sort bias is not a raster bias: the base `StandardMaterial::specialize` also packs it
        // into `depth_stencil.bias.constant`, where the far side's −4e4 would clip coplanar layers.
        if key.bind_group_data.far_side {
            if let Some(ds) = descriptor.depth_stencil.as_mut() {
                ds.bias.constant = 0;
            }
        }
        // MONKEY (skybox): the second texture stage compiles only where a batch has one.
        if key.bind_group_data.stage1 {
            descriptor.vertex.shader_defs.push("WOW_STAGE1".into());
            if let Some(fragment) = descriptor.fragment.as_mut() {
                fragment.shader_defs.push("WOW_STAGE1".into());
            }
        }
        // The skybox pins clip z to 0, reverse-Z far, in the vertex stage; its bias is sort-only.
        if key.bind_group_data.sky_depth {
            descriptor.vertex.shader_defs.push("WOW_SKY_DEPTH".into());
            if let Some(fragment) = descriptor.fragment.as_mut() {
                fragment.shader_defs.push("WOW_SKY_DEPTH".into());
            }
            if let Some(ds) = descriptor.depth_stencil.as_mut() {
                ds.bias.constant = 0;
            }
        }
        if key.bind_group_data.fade && !key.bind_group_data.sky_depth {
            // A fading model's near cards occlude its far ones; the sky never writes depth.
            if let Some(ds) = descriptor.depth_stencil.as_mut() {
                ds.depth_write_enabled = true;
            }
        }
        if key.bind_group_data.clutter {
            // Clutter blends in the alpha-mask pass, as the reference does, so the ~70 yd ramp
            // fades opacity; the shader keeps the 128/255 discard.
            if let Some(target) = descriptor
                .fragment
                .as_mut()
                .and_then(|f| f.targets.get_mut(0))
                .and_then(|t| t.as_mut())
            {
                target.blend = Some(BlendState::ALPHA_BLENDING);
            }
        }
        if key.bind_group_data.additive {
            // A pure (ONE, ONE) add: the shader weights by alpha in gamma space, where a `SrcAlpha`
            // factor or `AlphaMode::Add` would weight after linearising and fatten soft edges.
            if let Some(target) = descriptor
                .fragment
                .as_mut()
                .and_then(|f| f.targets.get_mut(0))
                .and_then(|t| t.as_mut())
            {
                target.blend = Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::Zero,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                });
            }
        }
        if key.bind_group_data.zfill {
            // The reference's `M2UseZFill` pre-pass (`0x707f7d`-`0x708072`): depth only, so a
            // translucent model shows one blended layer. The colour is computed and masked, not
            // skipped by an early return, which naga's MSL backend miscompiles.
            if let Some(frag) = descriptor.fragment.as_mut() {
                if let Some(target) = frag.targets.get_mut(0).and_then(|t| t.as_mut()) {
                    target.blend = None;
                    target.write_mask = ColorWrites::empty();
                }
            }
            if let Some(ds) = descriptor.depth_stencil.as_mut() {
                ds.depth_write_enabled = true;
            }
        }
        // The waterline clip (`benilla_world::straddle`) keys on the blend pass: no new pipelines.
        if key
            .mesh_key
            .intersection(MeshPipelineKey::BLEND_RESERVED_BITS)
            == MeshPipelineKey::BLEND_ALPHA
        {
            if let Some(fragment) = descriptor.fragment.as_mut() {
                fragment.shader_defs.push("WOW_WATER_CLIP".into());
            }
        }
        let key = &key.bind_group_data;
        if key.modulate || key.modulate2x {
            // `0x70c190`: Mod (M2 mode 5, WMO 4) is `DST_COLOR/ZERO`, Mod2x (M2 6, WMO 5)
            // `DST_COLOR/SRC_COLOR`; on a gamma framebuffer this is the reference's byte math.
            if let Some(target) = descriptor
                .fragment
                .as_mut()
                .and_then(|f| f.targets.get_mut(0))
                .and_then(|t| t.as_mut())
            {
                target.blend = Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::Dst,
                        dst_factor: if key.modulate2x {
                            BlendFactor::Src
                        } else {
                            BlendFactor::Zero
                        },
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::Zero,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                });
            }
        }
        Ok(())
    }
}

/// Distant low-detail terrain (WDL), the horizon hills beyond the streamed tiles: opaque white
/// geometry under the scene fog colour at the hull pass's own start 0 and end 1.0.
pub type WdlMaterial = ExtendedMaterial<StandardMaterial, WdlExt>;

/// WDL has no per-material input: its whole colour is the scene fog, off the shared global light.
#[derive(Asset, AsBindGroup, Clone, TypePath)]
pub struct WdlExt {
    /// The shared global light, rows 4/5 only (scene fog colour, farclip wall).
    #[storage(90, read_only, buffer)]
    pub light_buf: Buffer,
}

impl MaterialExtension for WdlExt {
    // `wdl.wgsl` writes its horizon-depth clamp from the fragment stage. The generic prepass
    // cannot reproduce that clamp and would let the coarse hull occlude detailed ADT terrain.
    // It is a static backdrop, not a temporal-vector producer, so retain its forward-only pass.
    fn enable_prepass() -> bool {
        false
    }

    fn vertex_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/wdl.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/wdl.wgsl".into()
    }
}

/// Liquid surfaces, one `liquid.wgsl` arm per reference liquid renderer: ADT MCLQ (the
/// `ocean0_s.bls` combine), WMO exterior and interior water, and magma and slime.
pub type LiquidMaterial = ExtendedMaterial<StandardMaterial, LiquidExt>;

/// Liquid inputs: the animated frames and the per-material lanes; all light comes off the shared
/// global light. The `Vec4` uniforms pack into binding 102, in the WGSL struct's field order.
#[derive(Asset, AsBindGroup, Clone, TypePath)]
pub struct LiquidExt {
    /// MONKEY (enhanced water): the water module's own bindings (103-106, `enhanced_water.wgsl`).
    /// The opaque world depth, resolved after the main opaque pass (`liquid/scene_depth.rs`).
    #[texture(103, sample_type = "float", filterable = false, visibility(fragment))]
    pub scene_depth: Handle<Image>,
    /// The module's uniform: quality, wave energy, clock, lane, sky rows, celestial body.
    #[uniform(104)]
    pub water: crate::WaterUniform,
    /// The shared light buffer again (the same `Buffer` as `light_buf`), for the module's own
    /// view of it: read-only storage may be bound twice in one group.
    #[storage(105, read_only, buffer, visibility(fragment))]
    pub water_light: Buffer,
    /// The opaque scene colour, copied beside the depth (`liquid/scene_depth.rs`): what the
    /// refraction looks through. Loaded, never sampled, so no sampler.
    #[texture(106, sample_type = "float", filterable = false, visibility(fragment))]
    pub scene_colour: Handle<Image>,
    /// The kind's animated frames (`lake_a`/`fast_a`/`ocean_h`), stacked as `2d_array` layers
    /// (`Rgba8Unorm`, repeat-sampled). RGB near-black; alpha = the ripple/wave → transparency.
    #[texture(100, dimension = "2d_array", visibility(fragment))]
    #[sampler(101, visibility(fragment))]
    pub frames: Handle<Image>,
    /// Which lanes of the shared light this surface reads:
    /// - `x` = fullbright (magma, slime): the sheet is the opaque body.
    /// - `y` = ocean: rows 15/16 (`Light.dbc` IntBand 14/15), else rows 13/14 (IntBand 16/17).
    /// - `z` = a WMO interior group's liquid, fogged with the interior block (rows 18/19): the
    ///   reference gates its fog (`0x6b6323`-`0x6b6342`) on the same `[0xca7f00]` as the group's
    ///   geometry (`0x6b51d9`/`0x6b51ea`).
    /// - `w` = the sun-sheen shininess (`lighting::WATER_SHININESS`).
    #[uniform(102)]
    pub kind: Vec4,
    /// `x` = the renderer (`liquid::surface::LiquidPath`): `0` ADT MCLQ, `1` WMO exterior, `2` WMO
    /// interior.
    #[uniform(102)]
    pub path: Vec4,
    /// `y` = frame count, `z` = scroll flag, `w` = clock enable.
    #[uniform(102)]
    pub anim: Vec4,
    /// The shared global light, read in both stages (the vertex stage evaluates the sun sheen).
    #[storage(90, read_only, buffer, visibility(vertex, fragment))]
    pub light_buf: Buffer,
}

impl MaterialExtension for LiquidExt {
    fn vertex_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/liquid.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/liquid.wgsl".into()
    }

    /// `sky_order::WATER_BIAS` (−2e4) is a sort rung, kept out of the rasterizer: as a depth-bias
    /// constant it would move the waterline, by an amount that doubles at every float exponent
    /// boundary, so neighbouring triangles' shorelines would disagree.
    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        if let Some(ds) = descriptor.depth_stencil.as_mut() {
            ds.bias.constant = 0;
        }
        Ok(())
    }
}

/// Per-tile terrain splat, one material per ADT tile: each merged-mesh vertex carries its chunk's
/// four layer indices (`COLOR`) and alpha layer (`UV1.x`), so a tile is one draw.
///
/// One repeating sampler (105) serves every array, to stay under Metal's 16 fragment samplers;
/// `terrain.wgsl` insets the alpha UVs by half a texel so they do not wrap at a chunk edge.
#[derive(Asset, AsBindGroup, Clone, TypePath)]
pub struct TerrainExtension {
    #[texture(100, dimension = "2d_array", visibility(fragment))]
    #[sampler(105, visibility(fragment))]
    pub layer_array: Handle<Image>,
    #[texture(104, dimension = "2d_array", visibility(fragment))]
    pub alpha_array: Handle<Image>,
    /// Per-chunk MCSH shadow maps (`R` = shadowed); `UV1.y` is the layer, `-1` for none.
    #[texture(110, dimension = "2d_array", visibility(fragment))]
    pub shadow_array: Handle<Image>,

    // Binding 106's uniforms, in the field order of the WGSL `TerrainParams`.
    /// `x` = texture tiling factor (repeats per chunk); other lanes unused.
    #[uniform(106)]
    pub params: Vec4,

    /// The shared global light, rows 0-5 (light, fog, farclip).
    #[storage(90, read_only, buffer)]
    pub light_buf: Buffer,
    /// MONKEY (outdoor torch shadows: terrain): the torch receiver's three bindings, at the SAME
    /// numbers [`WowModelExt`] uses (91 depth array / 92 comparison sampler / 93 table) and for the
    /// same reason - a Bevy material draw sets groups 0/1/2 only, so a receiver's shadow inputs have
    /// to ride its own material group. Both resources are created ONCE at startup
    /// (`benilla_world::assets`, beside the shared light buffer) and are ALWAYS bound: a missing
    /// image would stall the tile's bind group on `RetryNextUpdate` and blank the GROUND, so the
    /// spawn site idles until `TorchShared::binds()` is `Some` rather than baking a default.
    ///
    /// Fragment-only, like the splat/alpha/shadow arrays above (Bevy 0.18 narrows textures and
    /// samplers but not uniforms) - the vertex stage only picks WHICH lights, it never samples.
    #[texture(
        91,
        dimension = "2d_array",
        sample_type = "depth",
        visibility(fragment)
    )]
    #[sampler(92, sampler_type = "comparison", visibility(fragment))]
    pub torch_depth: Handle<Image>,
    /// The 6416-byte `TorchTableUniform` bytes (count / positions[16] / view_projs[96]) as the same
    /// shared raw storage buffer the model lane binds, rewritten in place each frame - read by
    /// `terrain.wgsl` as the byte-identical `var<storage, read> torch_table: TorchTable`.
    #[storage(93, read_only, buffer, visibility(fragment))]
    pub torch_buf: Buffer,
}

impl MaterialExtension for TerrainExtension {
    // The sun specular is per vertex (the reference's light flush `0x59c820`).
    fn vertex_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/terrain.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "embedded://benilla_assets/shaders/terrain.wgsl".into()
    }
}

#[cfg(test)]
mod tests {
    /// MONKEY (outdoor torch shadows: terrain): the GROUND receiver's three contracts with the
    /// model lane, all of which fail SILENTLY at runtime rather than at compile time.
    ///
    /// 1. **The bindings.** `TerrainExtension` and [`WowModelExt`] must claim the SAME numbers
    ///    (91 depth / 92 comparison sampler / 93 table). The Rust attributes and the WGSL `@binding`
    ///    lines are two independent declarations of one layout, and a mismatch is not a build error
    ///    — it is a bind-group validation failure at first tile spawn, i.e. no ground.
    /// 2. **The table layout.** All three receivers (`terrain.wgsl`, `wow_model.wgsl`, and
    ///    `static_gx.wgsl`'s std140 twin) mirror ONE 6416-byte buffer. Comments differ per lane on
    ///    purpose; the member lines must not, or terrain reads matrices at the wrong offsets and
    ///    projects shadows to the wrong place with nothing in any log.
    /// 3. **Daytime identity.** The night lane must stay behind `ext_night_w > 0.0`. `fog_params.z`
    ///    saturates to exactly 1.0 above the daylight threshold, so that guard is what makes the
    ///    day render the same instructions AND the same bits it was before this feature — the whole
    ///    "cost zero by day" claim, in one line of shader.
    #[test]
    fn the_terrain_lane_receives_the_torch_maps() {
        let terrain = include_str!("shaders/terrain.wgsl");
        let model = include_str!("shaders/wow_model.wgsl");

        for binding in [
            "@binding(91) var torch_depth: texture_depth_2d_array;",
            "@binding(92) var torch_samp: sampler_comparison;",
            "@binding(93) var<storage, read> torch_table: TorchTable;",
        ] {
            assert!(
                terrain.contains(binding) && model.contains(binding),
                "the terrain and model torch receivers disagree on `{binding}` — the WGSL and the \
                 AsBindGroup attributes are two declarations of one layout, and this one blanks \
                 the ground at the first tile spawn"
            );
        }

        // The struct's MEMBER lines only: each copy carries its own prose about its own lane.
        let layout = |src: &str| -> Vec<String> {
            let body = src
                .split_once("struct TorchTable {")
                .expect("every torch receiver declares the table")
                .1;
            body.split_once("\n}")
                .expect("the struct is closed")
                .0
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with("//"))
                .map(str::to_owned)
                .collect()
        };
        assert_eq!(
            layout(terrain),
            layout(model),
            "the terrain and model `TorchTable` copies have drifted — they mirror ONE 6416-byte \
             buffer (count / positions[16] / view_projs[96]); a member added, reordered or resized \
             on one side moves every matrix offset on that side only"
        );

        assert!(
            terrain.contains(
                "let ext_night_w = select(0.0, clamp(1.0 - sun_shadow_strength, 0.0, 1.0), \
                 torch_ext_on());"
            ) && terrain.contains("if (ext_night_w > 0.0) {"),
            "terrain's exterior torch lane is no longer gated on `ext_night_w > 0.0` — daylight is \
             now paying for (and possibly rendering) the night blend, and the day look is free to \
             drift by a rounding step"
        );
    }

    /// WDL writes a custom, far-pushed fragment depth. Rendering its raw mesh through Bevy's
    /// generic prepass would occlude detailed terrain before the forward pass applies that clamp.
    #[test]
    fn wdl_stays_out_of_the_generic_prepass() {
        assert!(
            !<super::WdlExt as bevy::pbr::MaterialExtension>::enable_prepass(),
            "the WDL backdrop re-entered Bevy's generic prepass and can occlude detailed terrain"
        );
    }

    /// Rigid models retain Bevy's exact helper; only the custom palette-rig path needs paired
    /// camera-relative positions. This catches a prepass edit that accidentally regresses one
    /// lane while changing the other.
    #[test]
    fn model_prepass_keeps_rigid_motion_stock_and_pairs_rig_history() {
        let src = include_str!("shaders/wow_model_prepass.wgsl");
        assert!(
            src.contains("WOW_RIG_PREVIOUS_PALETTE_ROW_OFFSET")
                && src.contains("previous_rig_origin")
                && src.contains("previous_view_uniforms.view_from_world"),
            "the rigged prepass lost one half of the prior-frame state"
        );
        assert!(
            src.contains("pbr_prepass_functions::calculate_motion_vector(\n        in.world_position,\n        in.previous_world_position,\n    )"),
            "the rigid prepass must continue to use Bevy's stock motion-vector helper"
        );
        assert!(
            src.contains("view.unjittered_clip_from_world * inverse(view.view_from_world)"),
            "the rigged branch must project camera-relative positions through the unjittered view"
        );
    }

    /// The prepass and the forward pass read one `wow_light` binding through two `WowLight`
    /// declarations. A member missing on one side moves the rig table and palettes on that side
    /// only: the prepass writes depth for scrambled geometry and the forward pass fails against it.
    #[test]
    fn model_prepass_mirrors_the_forward_light_struct() {
        let layout = |src: &str| -> Vec<String> {
            let body = src
                .split_once("struct WowLight {")
                .expect("both model passes declare the light struct")
                .1;
            body.split_once("\n}")
                .expect("the struct is closed")
                .0
                .lines()
                .map(|l| l.split_once("//").map_or(l, |(code, _)| code).trim())
                .filter(|l| !l.is_empty())
                .map(str::to_owned)
                .collect()
        };
        assert_eq!(
            layout(include_str!("shaders/wow_model_prepass.wgsl")),
            layout(include_str!("shaders/wow_model.wgsl")),
            "the prepass `WowLight` has drifted from wow_model.wgsl's"
        );
    }

    /// The sky depth law, for the one sky element that draws on the MODEL lane: the WMO skybox
    /// ([`WowModelKey::sky_depth`]). Every other sky shader is checked the same way, together, in
    /// `benilla_world::sky_order::the_sky_depth_is_pinned_at_the_vertex_and_nowhere_else` — this
    /// half lives here because the shader does. Without it a skybox silently goes back to being
    /// occluded by its own 94-yard shell radius instead of by world geometry (the regression
    /// decision 0588 fixed) — or, the other way, a `frag_depth` write comes back and every
    /// doodad, creature and wall in the frame loses its early-Z with it (decision 2016).
    #[test]
    fn the_sky_lane_pins_the_far_depth_at_the_vertex() {
        let src = include_str!("shaders/wow_model.wgsl");
        // The pin, matched behind its ifdef: ungated, every model draw sits at the far plane.
        let pin = src
            .find("#ifdef WOW_SKY_DEPTH")
            .expect("the sky-depth branch is gone");
        let branch = &src[pin..src[pin..].find("#endif").map_or(src.len(), |e| pin + e)];
        assert!(
            branch.contains("out.position.z = 0.0;"),
            "the model lane's sky branch no longer pins the far depth at the vertex — a WMO \
             skybox's shell radius is deciding occlusion again (benilla_world::sky_order, \"The \
             depth law\")"
        );
        assert!(
            !src.contains("@builtin(frag_depth)"),
            "the model lane writes a fragment depth again — the sky pin is the vertex stage's, \
             and a fragment write costs every draw on this lane its early-Z"
        );
    }

    /// The WGSL carries the reference's swatch row arithmetic, and a Rust mirror of it reproduces
    /// the row fill `0x68a830` on `Light.dbc` id 4, map 0, t = 1440, ocean `LightIntBand` 14 to 15.
    #[test]
    fn liquid_swatch_reproduces_the_reference_row_ramp() {
        let src = include_str!("shaders/liquid.wgsl");
        // The row accumulator `c0 + floor(i*(c1 - c0)/64)`, not a lerp to the deep endpoint.
        assert!(
            src.contains("let row = c0 + floor(i * (c1 - c0) / 64.0);"),
            "the swatch stopped building its rows the way `0x68a830` does: a plain lerp to the \
             deep endpoint runs a 64th of a ramp that does not exist"
        );
        // The ocean-only tail: floor(0.9*byte) on the last row, alpha forced opaque.
        assert!(
            src.contains("if ocean && i >= 63.0 {")
                && src.contains("vec4<f32>(floor(row.rgb * 0.9), 255.0)"),
            "the ocean's last-row darkening is gone — ~80% of the world's ocean vertices sample \
             that row, so this is the open sea's colour"
        );
        // Linear sampling across the two rows V falls between, at texel `V*64 - 0.5`.
        assert!(
            src.contains("let t = clamp(v * 64.0 - 0.5, 0.0, 63.0);"),
            "the swatch stopped sampling as an 8x64 LINEAR/CLAMP texture — the ocean darkening \
             would step instead of ramping across the final 1/64 of V"
        );

        /// `row(i)` for one channel, `swatch_row`'s byte arithmetic.
        fn row(c0: i32, c1: i32, i: i32) -> i32 {
            c0 + (i * (c1 - c0)).div_euclid(64)
        }
        /// The ocean tail's colour half.
        fn tail(b: i32) -> i32 {
            (f64::from(b) * 0.9).floor() as i32
        }

        // Ocean shallow (sub-14) 0x00457b63 and deep (sub-15) 0x000b2228, as R/G/B.
        let (shallow, deep) = ([69, 123, 99], [11, 34, 40]);
        let at = |i| [0, 1, 2].map(|k| row(shallow[k], deep[k], i));

        // Row 63 is one short of the endpoint on G: the ramp never arrives.
        assert_eq!(at(63), [11, 35, 40], "row 63 before the ocean tail");
        assert_ne!(
            at(63)[1],
            deep[1],
            "the ramp must NOT reach the deep endpoint"
        );
        assert_eq!(
            at(62),
            [12, 36, 41],
            "row 62, the other half of the last blend"
        );
        assert_eq!(
            at(63).map(tail),
            [9, 31, 36],
            "row 63 after the ocean tail (HSV V *= 0.9, which is floor(0.9*byte) per channel)"
        );
        // The tail and the shortfall are deep-end only.
        assert_eq!(at(0), shallow, "row 0 is the shallow endpoint verbatim");
    }
}
