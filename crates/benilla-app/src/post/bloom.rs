//! MONKEY (post): HDR-excess bloom. This runs on the world view before FFXGlow clamps it;
//! the later UI camera therefore never enters either the extraction or halo combine.

use crate::video::VideoConfig;
use benilla_world::{ffx_glow::FfxGlowLabel, view::WorldCamera};
use bevy::{
    core_pipeline::{
        core_3d::graph::{Core3d, Node3d},
        FullscreenShader,
    },
    ecs::query::QueryItem,
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
        view::ViewTarget,
        Render, RenderApp, RenderStartup, RenderSystems,
    },
};

pub(super) struct BloomPlugin;

#[derive(Component, Clone, Copy, ExtractComponent)]
struct BloomView {
    tier: u32,
}

impl Plugin for BloomPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<BloomView>::default())
            .add_systems(Last, update_views);
        if !app.is_plugin_added::<AssetPlugin>() {
            return;
        }
        let shader = app
            .world_mut()
            .resource_mut::<Assets<Shader>>()
            .add(Shader::from_wgsl(
                include_str!("bloom.wgsl"),
                "post/bloom.wgsl",
            ));
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(BloomShader(shader))
            .add_systems(RenderStartup, init_pipeline)
            .add_systems(
                Render,
                prepare_textures.in_set(RenderSystems::PrepareResources),
            )
            .add_render_graph_node::<ViewNodeRunner<BloomNode>>(Core3d, BloomLabel)
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::StartMainPassPostProcessing,
                    BloomLabel,
                    FfxGlowLabel,
                ),
            );
    }
}

fn update_views(
    mut commands: Commands,
    video: Res<VideoConfig>,
    cameras: Query<(Entity, &Camera, Option<&BloomView>), With<WorldCamera>>,
) {
    let tier = u32::from(video.bloom.min(2));
    for (entity, camera, old) in &cameras {
        if tier == 0 || !camera.is_active {
            if old.is_some() {
                commands.entity(entity).remove::<BloomView>();
            }
        } else if old.is_none_or(|old| old.tier != tier) {
            commands.entity(entity).insert(BloomView { tier });
        }
    }
}

#[derive(Resource)]
struct BloomShader(Handle<Shader>);

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(super) struct BloomLabel;

#[derive(Resource)]
struct BloomPipelines {
    layout: BindGroupLayoutDescriptor,
    sampler: Sampler,
    extract: CachedRenderPipelineId,
    blur_h: CachedRenderPipelineId,
    blur_v: CachedRenderPipelineId,
    combine: CachedRenderPipelineId,
}

fn init_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    fullscreen: Res<FullscreenShader>,
    shader: Res<BloomShader>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "post_bloom_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                texture_2d(TextureSampleType::Float { filterable: true }),
                uniform_buffer_sized(false, std::num::NonZeroU64::new(16)),
            ),
        ),
    );
    let sampler = render_device.create_sampler(&SamplerDescriptor {
        min_filter: FilterMode::Linear,
        mag_filter: FilterMode::Linear,
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        ..default()
    });
    let pipeline = |label: &'static str, entry: &'static str| RenderPipelineDescriptor {
        label: Some(label.into()),
        layout: vec![layout.clone()],
        vertex: fullscreen.to_vertex_state(),
        fragment: Some(FragmentState {
            shader: shader.0.clone(),
            shader_defs: vec![],
            entry_point: Some(entry.into()),
            targets: vec![Some(ColorTargetState {
                format: ViewTarget::TEXTURE_FORMAT_HDR,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        ..default()
    };
    let extract =
        pipeline_cache.queue_render_pipeline(pipeline("post_bloom_extract", "fs_extract"));
    let blur_h = pipeline_cache.queue_render_pipeline(pipeline("post_bloom_blur_h", "fs_blur_h"));
    let blur_v = pipeline_cache.queue_render_pipeline(pipeline("post_bloom_blur_v", "fs_blur_v"));
    let combine =
        pipeline_cache.queue_render_pipeline(pipeline("post_bloom_combine", "fs_combine"));
    commands.insert_resource(BloomPipelines {
        layout,
        sampler,
        extract,
        blur_h,
        blur_v,
        combine,
    });
}

#[derive(Component)]
struct BloomTextures {
    a: CachedTexture,
    b: CachedTexture,
    uniform: Buffer,
    bind_ab: BindGroup,
    bind_ba: BindGroup,
    tier: u32,
}

fn prepare_textures(
    mut commands: Commands,
    mut cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    pipelines: Res<BloomPipelines>,
    views: Query<(Entity, &ExtractedCamera, &BloomView, Option<&BloomTextures>)>,
) {
    let layout = pipeline_cache.get_bind_group_layout(&pipelines.layout);
    for (entity, camera, view, old) in &views {
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        let divisor = if view.tier == 1 { 4 } else { 2 };
        let mut texture = |label: &'static str, width: u32, height: u32| {
            cache.get(
                &render_device,
                TextureDescriptor {
                    label: Some(label),
                    size: Extent3d {
                        width: width.max(8),
                        height: height.max(8),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: ViewTarget::TEXTURE_FORMAT_HDR,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
            )
        };
        let a = texture("post_bloom_a", size.x / divisor, size.y / divisor);
        let b = texture("post_bloom_b", size.x / divisor, size.y / divisor);
        if old.is_some_and(|old| {
            old.tier == view.tier
                && old.a.texture.id() == a.texture.id()
                && old.b.texture.id() == b.texture.id()
        }) {
            continue;
        }
        let uniform = render_device.create_buffer(&BufferDescriptor {
            label: Some("post_bloom_uniform"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = |label, source: &TextureView, auxiliary: &TextureView| {
            render_device.create_bind_group(
                label,
                &layout,
                &BindGroupEntries::sequential((
                    source,
                    &pipelines.sampler,
                    auxiliary,
                    uniform.as_entire_binding(),
                )),
            )
        };
        let bind_ab = bind("post_bloom_a_to_b", &a.default_view, &a.default_view);
        let bind_ba = bind("post_bloom_b_to_a", &b.default_view, &b.default_view);
        commands.entity(entity).insert(BloomTextures {
            a,
            b,
            uniform,
            bind_ab,
            bind_ba,
            tier: view.tier,
        });
    }
}

#[derive(Default)]
struct BloomNode;

impl ViewNode for BloomNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static BloomView,
        &'static BloomTextures,
    );

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (target, view, textures): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let pipelines = world.resource::<BloomPipelines>();
        let cache = world.resource::<PipelineCache>();
        let (Some(extract), Some(blur_h), Some(blur_v), Some(combine)) = (
            cache.get_render_pipeline(pipelines.extract),
            cache.get_render_pipeline(pipelines.blur_h),
            cache.get_render_pipeline(pipelines.blur_v),
            cache.get_render_pipeline(pipelines.combine),
        ) else {
            return Ok(());
        };
        let gain = if view.tier == 1 { 0.55 } else { 0.78 };
        let params = [view.tier as f32, gain, 0.0, 0.0];
        world.resource::<RenderQueue>().write_buffer(
            &textures.uniform,
            0,
            bytemuck::cast_slice(&params),
        );
        let device = render_context.render_device().clone();
        let layout = cache.get_bind_group_layout(&pipelines.layout);
        let source = target.main_texture_view();
        let extract_bind = device.create_bind_group(
            "post_bloom_extract",
            &layout,
            &BindGroupEntries::sequential((
                source,
                &pipelines.sampler,
                &textures.b.default_view,
                textures.uniform.as_entire_binding(),
            )),
        );
        let diagnostics = render_context.diagnostic_recorder();
        for (label, pipeline, bind, destination) in [
            (
                "post_bloom_extract",
                extract,
                &extract_bind,
                &textures.a.default_view,
            ),
            (
                "post_bloom_blur_h",
                blur_h,
                &textures.bind_ab,
                &textures.b.default_view,
            ),
            (
                "post_bloom_blur_v",
                blur_v,
                &textures.bind_ba,
                &textures.a.default_view,
            ),
        ] {
            let mut pass =
                render_context
                    .command_encoder()
                    .begin_render_pass(&RenderPassDescriptor {
                        label: Some(label),
                        color_attachments: &[Some(RenderPassColorAttachment {
                            view: destination,
                            depth_slice: None,
                            resolve_target: None,
                            ops: Operations::default(),
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
            let span = diagnostics.pass_span(&mut pass, label);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.draw(0..3, 0..1);
            span.end(&mut pass);
        }
        let output = target.post_process_write();
        let combine_bind = device.create_bind_group(
            "post_bloom_combine",
            &layout,
            &BindGroupEntries::sequential((
                output.source,
                &pipelines.sampler,
                &textures.a.default_view,
                textures.uniform.as_entire_binding(),
            )),
        );
        let mut pass = render_context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("post_bloom_combine"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: output.destination,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations::default(),
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        let span = diagnostics.pass_span(&mut pass, "post_bloom_combine");
        pass.set_pipeline(combine);
        pass.set_bind_group(0, &combine_bind, &[]);
        pass.draw(0..3, 0..1);
        span.end(&mut pass);
        Ok(())
    }
}
