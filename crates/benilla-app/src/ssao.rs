//! MONKEY (ao): depth-only screen-space ambient occlusion (soft contact shadows).
//!
//! No depth or normal prepass exists, and the static world draws in its own node, so Bevy's
//! SSAO cannot see it. This lane reads the main-pass depth after the opaque pass instead:
//! half-resolution occlusion from depth-reconstructed positions and normals, a 4x4 depth-aware
//! blur, then a joint-bilateral upsample MULTIPLIED into the main colour attachment (the MSAA
//! attachment when multisampled, so the resolve and every later pass see it). It runs before
//! the water depth/colour copy and before transparents, so water, particles and UI are
//! untouched and refraction sees the darkened bed. Sky, distant pixels and
//! bright pixels (lit windows, flames, sunlit sand) are protected; fade 45-90 yd (Low) / 60-120 yd (High).
//!
//! cvar `ambientOcclusion`: 0 Off (pass not scheduled, image unchanged), 1 Low, 2 High.
//! `WOW_AO=0|1|2` overrides it for the session; `WOW_AO_DEBUG=1..5` writes a diagnostic view
//! (factor, protection, distance, raw occlusion); `WOW_AO_GAIN/RADIUS/BIAS/STRENGTH` tune it.
use crate::video::VideoConfig;
use benilla_world::{liquid::WaterDepthLabel, view::WorldCamera};
use bevy::{
    core_pipeline::{
        core_3d::graph::{Core3d, Node3d},
        FullscreenShader,
    },
    ecs::query::QueryItem,
    image::ToExtents,
    prelude::*,
    render::{
        camera::ExtractedCamera,
        diagnostic::RecordDiagnostics,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_graph::{
            NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::{CachedTexture, TextureCache},
        view::{Msaa, ViewDepthTexture, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
        Render, RenderApp, RenderStartup, RenderSystems,
    },
    shader::ShaderDefVal,
};

pub(crate) struct AmbientOcclusionPlugin;

#[derive(Resource)]
struct AoOverride(Option<u8>);

/// Per-tier constants, uploaded as one small uniform.
#[derive(Component, Clone, Copy, PartialEq, Debug, ExtractComponent, ShaderType)]
struct AoView {
    /// Radius (yd), darkening strength, sample count, cosine bias.
    params: Vec4,
    /// Distance fade start and end (yd), bright-pixel protection ramp (gamma max channel).
    fade: Vec4,
    /// x = debug view (0 off, 1 AO factor, 2 protection, 3 distance/50, 4 raw occlusion,
    /// 5 surface smoothness, MONKEY (followups)), y = occlusion gain.
    debug: Vec4,
}

/// Dev tuning knob `WOW_AO_<name>=<f32>`, read once per name; `None` keeps the tier value.
fn tuning(name: &'static str) -> Option<f32> {
    static KNOBS: std::sync::Mutex<Vec<(&'static str, Option<f32>)>> =
        std::sync::Mutex::new(Vec::new());
    let mut knobs = KNOBS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some((_, v)) = knobs.iter().find(|(n, _)| *n == name) {
        return *v;
    }
    let v = std::env::var(format!("WOW_AO_{name}"))
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0);
    knobs.push((name, v));
    v
}

impl AoView {
    fn tier(tier: u8) -> Self {
        let (radius, strength, samples) = if tier >= 2 {
            (1.5, 0.7, 12.0)
        } else {
            (1.2, 0.65, 6.0)
        };
        // Outdoor contacts (house bases, trunks) sit 30-60 yd out; Low stops sooner.
        let fade = if tier >= 2 {
            (60.0, 120.0)
        } else {
            (45.0, 90.0)
        };
        let radius = tuning("RADIUS").unwrap_or(radius);
        let strength = tuning("STRENGTH").unwrap_or(strength).min(1.0);
        let bias = tuning("BIAS").unwrap_or(0.05);
        let gain = tuning("GAIN").unwrap_or(6.0);
        Self {
            params: Vec4::new(radius, strength, samples, bias),
            fade: Vec4::new(fade.0, fade.1, 0.7, 0.95),
            debug: Vec4::new(debug_mode() as f32, gain, 0.0, 0.0),
        }
    }
}

/// `WOW_AO_DEBUG=1..5`: write a diagnostic term instead of darkening the scene.
fn debug_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        std::env::var("WOW_AO_DEBUG")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
            .min(5)
    })
}

fn debug_view() -> bool {
    debug_mode() != 0
}

impl Plugin for AmbientOcclusionPlugin {
    fn build(&self, app: &mut App) {
        let value = std::env::var("WOW_AO")
            .ok()
            .and_then(|v| v.parse::<u8>().ok())
            .filter(|v| *v <= 2);
        app.insert_resource(AoOverride(value))
            .add_plugins(ExtractComponentPlugin::<AoView>::default())
            .add_systems(Last, update_ao);
        // Keep headless policy tests independent of the renderer.
        if !app.is_plugin_added::<AssetPlugin>() {
            return;
        }
        // MONKEY (integration): the crate's shader convention — embedded by `shaders::plugin`.
        let shader = app
            .world()
            .resource::<AssetServer>()
            .load("embedded://benilla_app/shaders/ssao.wgsl");
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(AoShader(shader))
            .init_resource::<SpecializedRenderPipelines<AoPipeline>>()
            .add_systems(RenderStartup, init_pipeline)
            .add_systems(
                Render,
                (
                    prepare_pipelines.in_set(RenderSystems::Prepare),
                    prepare_textures.in_set(RenderSystems::PrepareResources),
                    // MONKEY (reviewfix-a): one uniform per view, rewritten, not allocated per frame.
                    prepare_ao_uniforms.in_set(RenderSystems::PrepareResources),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<AoNode>>(Core3d, AoLabel)
            // After every solid path (static_gx precedes MainOpaquePass), before the water copy.
            .add_render_graph_edges(Core3d, (Node3d::MainOpaquePass, AoLabel, WaterDepthLabel));
    }

    /// Feature 18 evaluates the occluded opaque scene, and the water refracts its result.
    /// `DlssNrPlugin` registers its node in its own `finish`, which runs before this one.
    #[cfg(feature = "dlss")]
    fn finish(&self, app: &mut App) {
        // `build` registered no node without a renderer.
        if !app.is_plugin_added::<AssetPlugin>() {
            return;
        }
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_render_graph_edges(
                Core3d,
                (AoLabel, benilla_dlss5::DlssNrEvalLabel, WaterDepthLabel),
            );
        }
    }
}

fn update_ao(
    mut commands: Commands,
    video: Res<VideoConfig>,
    override_value: Res<AoOverride>,
    mut cameras: Query<(Entity, &Camera, &mut Camera3d, Option<&AoView>), With<WorldCamera>>,
) {
    let tier = override_value.0.unwrap_or(video.ambient_occlusion).min(2);
    for (entity, camera, mut camera3d, current) in &mut cameras {
        if tier == 0 || !camera.is_active {
            if current.is_some() {
                commands.entity(entity).remove::<AoView>();
            }
            continue;
        }
        let binding = TextureUsages::TEXTURE_BINDING.bits();
        if camera3d.depth_texture_usages.0 & binding == 0 {
            camera3d.depth_texture_usages.0 |= binding;
        }
        let want = AoView::tier(tier);
        if current != Some(&want) {
            commands.entity(entity).insert(want);
        }
    }
}

#[derive(Resource)]
struct AoShader(Handle<Shader>);

#[derive(Resource)]
struct AoPipeline {
    ao_layouts: [BindGroupLayoutDescriptor; 2],
    blur_layout: BindGroupLayoutDescriptor,
    apply_layouts: [BindGroupLayoutDescriptor; 2],
    shader: Handle<Shader>,
    fullscreen: FullscreenShader,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Stage {
    Ao,
    Blur,
    Apply,
}

const AO_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

fn init_pipeline(mut commands: Commands, shader: Res<AoShader>, fullscreen: Res<FullscreenShader>) {
    let with_depth = |label: &'static str, multisampled: bool| {
        BindGroupLayoutDescriptor::new(
            label,
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    if multisampled {
                        texture_depth_2d_multisampled()
                    } else {
                        texture_depth_2d()
                    },
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    uniform_buffer::<ViewUniform>(true),
                    uniform_buffer::<AoView>(false),
                ),
            ),
        )
    };
    let blur_layout = BindGroupLayoutDescriptor::new(
        "ssao_blur_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            (
                (
                    0,
                    texture_2d(TextureSampleType::Float { filterable: false }),
                ),
                (3, uniform_buffer::<AoView>(false)),
            ),
        ),
    );
    commands.insert_resource(AoPipeline {
        ao_layouts: [false, true].map(|ms| with_depth("ssao_layout", ms)),
        blur_layout,
        apply_layouts: [false, true].map(|ms| with_depth("ssao_apply_layout", ms)),
        shader: shader.0.clone(),
        fullscreen: fullscreen.clone(),
    });
}

impl SpecializedRenderPipeline for AoPipeline {
    /// Stage, target format, main-pass sample count, debug view.
    type Key = (Stage, TextureFormat, u32, bool);
    fn specialize(&self, (stage, format, samples, debug): Self::Key) -> RenderPipelineDescriptor {
        let multisampled = samples > 1;
        let mut defs: Vec<ShaderDefVal> = vec![match stage {
            Stage::Ao => "STAGE_AO",
            Stage::Blur => "STAGE_BLUR",
            Stage::Apply => "STAGE_APPLY",
        }
        .into()];
        if multisampled && stage != Stage::Blur {
            defs.push("MULTISAMPLED".into());
        }
        let (label, entry, layout) = match stage {
            Stage::Ao => (
                "ssao",
                "ao_main",
                self.ao_layouts[multisampled as usize].clone(),
            ),
            Stage::Blur => ("ssao_blur", "blur_main", self.blur_layout.clone()),
            Stage::Apply => (
                "ssao_apply",
                "apply_main",
                self.apply_layouts[multisampled as usize].clone(),
            ),
        };
        // The apply stage multiplies into the scene (dst * src) and leaves alpha alone.
        let blend = (stage == Stage::Apply && !debug).then_some(BlendState {
            color: BlendComponent {
                src_factor: BlendFactor::Dst,
                dst_factor: BlendFactor::Zero,
                operation: BlendOperation::Add,
            },
            alpha: BlendComponent {
                src_factor: BlendFactor::Zero,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
        });
        RenderPipelineDescriptor {
            label: Some(label.into()),
            layout: vec![layout],
            vertex: self.fullscreen.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: defs,
                entry_point: Some(entry.into()),
                targets: vec![Some(ColorTargetState {
                    format: if stage == Stage::Apply {
                        format
                    } else {
                        AO_FORMAT
                    },
                    blend,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multisample: MultisampleState {
                count: if stage == Stage::Apply { samples } else { 1 },
                ..default()
            },
            ..default()
        }
    }
}

#[derive(Component)]
struct ViewAoPipelines([CachedRenderPipelineId; 3]);

fn prepare_pipelines(
    mut commands: Commands,
    cache: Res<PipelineCache>,
    pipeline: Res<AoPipeline>,
    mut specialized: ResMut<SpecializedRenderPipelines<AoPipeline>>,
    views: Query<(Entity, &ViewTarget, &Msaa), With<AoView>>,
    all_views: Query<(&ViewTarget, &Msaa), With<Camera3d>>,
) {
    let debug = debug_view();
    // MONKEY (integration): warm every reachable key on every 3-D view, feature on or off, so the
    // compile happens under the entry cover and never live when the player turns the row on.
    // (Named in `pipe_warm/menagerie.rs`'s custom-lane census.)
    for (target, msaa) in &all_views {
        for stage in [Stage::Ao, Stage::Blur, Stage::Apply] {
            specialized.specialize(
                &cache,
                &pipeline,
                (stage, target.main_texture_format(), msaa.samples(), debug),
            );
        }
    }
    for (entity, target, msaa) in &views {
        let format = target.main_texture_format();
        let samples = msaa.samples();
        let ids = [Stage::Ao, Stage::Blur, Stage::Apply].map(|stage| {
            specialized.specialize(&cache, &pipeline, (stage, format, samples, debug))
        });
        commands.entity(entity).insert(ViewAoPipelines(ids));
    }
}

#[derive(Component)]
struct AoTextures {
    raw: CachedTexture,
    blurred: CachedTexture,
}

fn prepare_textures(
    mut commands: Commands,
    mut cache: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedCamera), With<AoView>>,
) {
    for (entity, camera) in &views {
        // The depth texture is sized to the physical target.
        let Some(size) = camera.physical_target_size else {
            continue;
        };
        let half = UVec2::new(size.x.div_ceil(2), size.y.div_ceil(2)).max(UVec2::ONE);
        let mut texture = |label: &'static str| {
            cache.get(
                &device,
                TextureDescriptor {
                    label: Some(label),
                    size: half.to_extents(),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: AO_FORMAT,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
            )
        };
        let raw = texture("ssao_raw");
        let blurred = texture("ssao_blurred");
        commands.entity(entity).insert(AoTextures { raw, blurred });
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct AoLabel;

/// MONKEY (reviewfix-a): the view's AO uniform, created once and rewritten in prepare.
#[derive(Component)]
struct AoUniform(UniformBuffer<AoView>);

fn prepare_ao_uniforms(
    mut commands: Commands,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut views: Query<(Entity, &AoView, Option<&mut AoUniform>)>,
) {
    for (entity, settings, uniform) in &mut views {
        match uniform {
            Some(mut uniform) => {
                uniform.0.set(*settings);
                uniform.0.write_buffer(&device, &queue);
            }
            None => {
                let mut uniform = UniformBuffer::from(*settings);
                uniform.write_buffer(&device, &queue);
                commands.entity(entity).insert(AoUniform(uniform));
            }
        }
    }
}

#[derive(Default)]
struct AoNode;

impl ViewNode for AoNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ViewDepthTexture,
        &'static ViewUniformOffset,
        &'static AoUniform,
        &'static ViewAoPipelines,
        &'static AoTextures,
    );
    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        context: &mut RenderContext<'w>,
        (target, depth, view_offset, settings, ids, textures): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let cache = world.resource::<PipelineCache>();
        let (Some(ao), Some(blur), Some(apply)) = (
            cache.get_render_pipeline(ids.0[0]),
            cache.get_render_pipeline(ids.0[1]),
            cache.get_render_pipeline(ids.0[2]),
        ) else {
            return Ok(());
        };
        if !depth
            .texture
            .usage()
            .contains(TextureUsages::TEXTURE_BINDING)
        {
            return Ok(());
        }
        let Some(view) = world.resource::<ViewUniforms>().uniforms.binding() else {
            return Ok(());
        };
        let layouts = world.resource::<AoPipeline>();
        let ms = (depth.texture.sample_count() > 1) as usize;
        let device = context.render_device().clone();
        let params = settings.0.binding().unwrap();

        // `main_texture_view` is the single-sample texture the opaque pass resolved into.
        let ao_bind = device.create_bind_group(
            "ssao",
            &cache.get_bind_group_layout(&layouts.ao_layouts[ms]),
            &BindGroupEntries::sequential((
                depth.view(),
                target.main_texture_view(),
                view.clone(),
                params.clone(),
            )),
        );
        let blur_bind = device.create_bind_group(
            "ssao_blur",
            &cache.get_bind_group_layout(&layouts.blur_layout),
            &BindGroupEntries::with_indices(((0, &textures.raw.default_view), (3, params.clone()))),
        );
        let apply_bind = device.create_bind_group(
            "ssao_apply",
            &cache.get_bind_group_layout(&layouts.apply_layouts[ms]),
            &BindGroupEntries::sequential((
                depth.view(),
                &textures.blurred.default_view,
                view,
                params,
            )),
        );
        let diagnostics = context.diagnostic_recorder();
        let half_pass = |encoder: &mut CommandEncoder,
                         label: &'static str,
                         out: &TextureView,
                         pipeline: &RenderPipeline,
                         bind: &BindGroup,
                         offsets: &[u32]| {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: out,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Default::default()),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            let span = diagnostics.pass_span(&mut pass, label);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind, offsets);
            pass.draw(0..3, 0..1);
            span.end(&mut pass);
        };
        let encoder = context.command_encoder();
        half_pass(
            encoder,
            "ssao",
            &textures.raw.default_view,
            ao,
            &ao_bind,
            &[view_offset.offset],
        );
        half_pass(
            encoder,
            "ssao_blur",
            &textures.blurred.default_view,
            blur,
            &blur_bind,
            &[],
        );
        // The main attachment (MSAA + resolve when multisampled), so later passes load it darkened.
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("ssao_apply"),
            color_attachments: &[Some(target.get_color_attachment())],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        let span = diagnostics.pass_span(&mut pass, "ssao_apply");
        pass.set_pipeline(apply);
        pass.set_bind_group(0, &apply_bind, &[view_offset.offset]);
        pass.draw(0..3, 0..1);
        span.end(&mut pass);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiers_follow_the_cvar_override_and_camera_state() {
        let mut app = App::new();
        app.init_resource::<VideoConfig>()
            .insert_resource(AoOverride(None))
            .add_systems(Update, update_ao);
        let camera = app
            .world_mut()
            .spawn((WorldCamera, Camera::default(), Camera3d::default()))
            .id();
        let preview = app
            .world_mut()
            .spawn((Camera::default(), Camera3d::default()))
            .id();
        // The registered default is Off: no pass, image unchanged.
        app.update();
        assert!(app.world().get::<AoView>(camera).is_none());
        app.world_mut()
            .resource_mut::<VideoConfig>()
            .ambient_occlusion = 1;
        app.update();
        let low = *app.world().get::<AoView>(camera).unwrap();
        assert_eq!(low.params.z, 6.0);
        assert!(app.world().get::<AoView>(preview).is_none());
        let usage = app
            .world()
            .get::<Camera3d>(camera)
            .unwrap()
            .depth_texture_usages
            .0;
        assert!(usage & TextureUsages::TEXTURE_BINDING.bits() != 0);
        app.world_mut()
            .resource_mut::<VideoConfig>()
            .ambient_occlusion = 2;
        app.update();
        let high = *app.world().get::<AoView>(camera).unwrap();
        assert_eq!(high.params.z, 12.0);
        assert!(high.params.x > low.params.x);
        app.world_mut().get_mut::<Camera>(camera).unwrap().is_active = false;
        app.update();
        assert!(app.world().get::<AoView>(camera).is_none());
        app.world_mut().get_mut::<Camera>(camera).unwrap().is_active = true;
        app.world_mut().resource_mut::<AoOverride>().0 = Some(0);
        app.update();
        assert!(app.world().get::<AoView>(camera).is_none());
    }

    #[test]
    fn tiers_keep_the_fade_and_protection_ramps_ordered() {
        for tier in [1, 2] {
            let v = AoView::tier(tier);
            assert!(v.fade.x < v.fade.y, "distance fade runs outward");
            assert!(v.fade.z < v.fade.w, "protection ramps up with brightness");
            assert!(v.params.y > 0.0 && v.params.y < 1.0, "never a black halo");
        }
    }
}
