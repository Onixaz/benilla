//! The world camera's motion-vector prepass and a deliberately narrow developer visualizer. It
//! is independent of FFXGlow's ordering: FFX completes first, this last diagnostic replaces only
//! the world source the UI backdrop presents.

use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::core_pipeline::prepass::ViewPrepassTextures;
use bevy::core_pipeline::FullscreenShader;
use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::render_graph::{
    NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::binding_types::texture_2d;
use bevy::render::render_resource::*;
use bevy::render::renderer::RenderContext;
use bevy::render::view::ViewTarget;
use bevy::render::{RenderApp, RenderStartup};

use crate::dev_state::DebugState;
use crate::view::WorldCamera;

/// Extracted only from a [`WorldCamera`]. When present, the view's final world colour is replaced
/// with the raw prepass diagnostic after FFXGlow has done its ordinary work.
#[derive(Component, Clone, Copy, ExtractComponent)]
pub struct MotionVectorDebug;

/// Installs the world-only diagnostic and keeps it in step with the developer panel state.
pub struct MotionVectorDebugPlugin;

impl Plugin for MotionVectorDebugPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<MotionVectorDebug>::default())
            .add_systems(Update, sync_debug_view);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .add_systems(RenderStartup, init_pipeline)
            .add_render_graph_node::<ViewNodeRunner<MotionVectorDebugNode>>(
                Core3d,
                MotionVectorDebugLabel,
            )
            // FFXGlow retains its Start -> FFX -> Bloom order. The diagnostic is strictly after
            // that chain and before the always-present tonemapper. (AutoExposure is optional,
            // so it cannot be an unconditional graph-edge target.)
            .add_render_graph_edges(
                Core3d,
                (Node3d::Bloom, MotionVectorDebugLabel, Node3d::Tonemapping),
            );
    }
}

fn sync_debug_view(
    debug: Res<DebugState>,
    mut commands: Commands,
    cameras: Query<(Entity, Option<&MotionVectorDebug>), With<WorldCamera>>,
) {
    for (entity, present) in &cameras {
        match (debug.models.motion_vectors, present.is_some()) {
            (true, false) => {
                commands.entity(entity).insert(MotionVectorDebug);
            }
            (false, true) => {
                commands.entity(entity).remove::<MotionVectorDebug>();
            }
            _ => {}
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct MotionVectorDebugLabel;

#[derive(Resource)]
struct MotionVectorDebugPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedRenderPipelineId,
}

fn init_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    fullscreen: Res<FullscreenShader>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "motion_vector_debug_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (texture_2d(TextureSampleType::Float { filterable: false }),),
        ),
    );
    let shader = asset_server.load("embedded://benilla_world/shaders/motion_vector_debug.wgsl");
    let pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("motion_vector_debug".into()),
        layout: vec![layout.clone()],
        vertex: fullscreen.to_vertex_state(),
        fragment: Some(FragmentState {
            shader,
            shader_defs: vec![],
            entry_point: Some("fs_motion_vectors".into()),
            targets: vec![Some(ColorTargetState {
                format: ViewTarget::TEXTURE_FORMAT_HDR,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        ..default()
    });
    commands.insert_resource(MotionVectorDebugPipeline { layout, pipeline });
}

#[derive(Default)]
struct MotionVectorDebugNode;

impl ViewNode for MotionVectorDebugNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ViewPrepassTextures,
        &'static MotionVectorDebug,
    );

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (target, prepass, _debug): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let pipelines = world.resource::<MotionVectorDebugPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_render_pipeline(pipelines.pipeline) else {
            return Ok(());
        };
        let Some(motion_vectors) = prepass.motion_vectors_view() else {
            return Ok(());
        };

        let layout = pipeline_cache.get_bind_group_layout(&pipelines.layout);
        let bind_group = render_context.render_device().create_bind_group(
            "motion_vector_debug_bind_group",
            &layout,
            &BindGroupEntries::sequential((motion_vectors,)),
        );
        // Do not take `post_process_write`: the active player world is normally a backdrop source
        // whose UI camera samples this main texture directly. This full-screen pass safely replaces
        // it in place and lets the unchanged UI/FFX composition present the diagnostic.
        let mut pass = render_context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("motion_vector_debug"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: target.main_texture_view(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}
