//! MONKEY (post): screen-space sun shafts from the visible sky depth.
//!
//! MONKEY (polish): the depth is read once per half-resolution texel into an R8 sky mask
//! (`fs_mask`, the only MSAA-specialised stage); the 28-tap radial blur then samples the mask.

use super::bloom::BloomLabel;
use crate::video::VideoConfig;
use benilla_world::{lighting::WowLighting, view::WorldCamera, wmo_portal::CameraInteriorClaim};
// MONKEY (leftovers): LightSkybox flag 0x8.
use benilla_world::skybox::CameraSkybox;
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
        view::{ViewDepthTexture, ViewTarget},
        Render, RenderApp, RenderStartup, RenderSystems,
    },
};

pub(super) struct SunShaftsPlugin;

#[derive(Component, Clone, Copy, PartialEq, ExtractComponent, ShaderType)]
struct ShaftView {
    // xy = sun UV, z = faded strength, w = sample count.
    sun: Vec4,
    // Gamma-space sun colour; the framebuffer is still gamma-space here.
    color: Vec4,
    // MONKEY (leftovers): the view ray's world height is `dot(horizon.xyz, (ndc.x, ndc.y, 1))`
    // over `length((ndc.x * ray.x, ndc.y * ray.y, 1))`; the mask counts a depth-0 pixel as sky
    // only above `horizon.w` (sin elevation). Water writes no depth, so the sea read as sky.
    horizon: Vec4,
    // xy = tan half-FOV per axis (1/P00, 1/P11); zw unused.
    ray: Vec4,
}

impl Plugin for SunShaftsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<ShaftView>::default())
            .add_systems(Last, update_views);
        if !app.is_plugin_added::<AssetPlugin>() {
            return;
        }
        let shader = app
            .world_mut()
            .resource_mut::<Assets<Shader>>()
            .add(Shader::from_wgsl(
                include_str!("sun_shafts.wgsl"),
                "post/sun_shafts.wgsl",
            ));
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(ShaftShader(shader))
            .init_resource::<SpecializedRenderPipelines<ShaftPipeline>>()
            .add_systems(RenderStartup, init_pipeline)
            .add_systems(
                Render,
                (
                    prepare_pipelines.in_set(RenderSystems::Prepare),
                    prepare_masks.in_set(RenderSystems::PrepareResources),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<ShaftNode>>(Core3d, ShaftLabel)
            .add_render_graph_edges(
                Core3d,
                (Node3d::StartMainPassPostProcessing, ShaftLabel, BloomLabel),
            );
    }
}

#[allow(clippy::type_complexity)]
fn update_views(
    mut commands: Commands,
    video: Res<VideoConfig>,
    lighting: Res<WowLighting>,
    interior: Res<CameraInteriorClaim>,
    sky: Option<Res<CameraSkybox>>,
    mut cameras: Query<
        (
            Entity,
            &Camera,
            &GlobalTransform,
            &Projection,
            &mut Camera3d,
            Option<&ShaftView>,
        ),
        With<WorldCamera>,
    >,
) {
    let to_sun = lighting.celestial_dir().normalize_or_zero();
    // MONKEY (leftovers): a `0x8` sky forces the shafts: its weight lifts the daylight ramp, so
    // a low or just-set sun keeps full shafts (a sun far below the horizon still ends them).
    let force = sky.map_or(0.0, |s| {
        s.flag_weight(benilla_formats::SKYBOX_FORCE_SUN_SHAFTS)
    });
    for (entity, camera, transform, projection, mut camera3d, old) in &mut cameras {
        let disabled =
            !video.sun_shafts || !camera.is_active || interior.0.is_some() || to_sun == Vec3::ZERO;
        if disabled {
            if old.is_some() {
                commands.entity(entity).remove::<ShaftView>();
            }
            continue;
        }
        let far = match projection {
            Projection::Perspective(p) => p.far,
            _ => 3000.0,
        };
        let world = transform.translation() + to_sun * far * 0.85;
        let (Ok(screen), Some(size)) = (
            camera.world_to_viewport(transform, world),
            camera.logical_viewport_size(),
        ) else {
            commands.entity(entity).remove::<ShaftView>();
            continue;
        };
        let uv = screen / size;
        // Keep a small off-screen skirt so shafts leave the frame continuously.
        let outside = Vec2::new(
            (-uv.x).max(uv.x - 1.0).max(0.0),
            (-uv.y).max(uv.y - 1.0).max(0.0),
        )
        .max_element();
        let edge_fade = 1.0 - smoothstep(0.0, 0.18, outside);
        let daylight = smoothstep(-0.02, 0.16, to_sun.y)
            .max(force.clamp(0.0, 1.0) * smoothstep(-0.12, -0.02, to_sun.y));
        let strength = edge_fade * daylight;
        if strength <= 0.001 {
            commands.entity(entity).remove::<ShaftView>();
            continue;
        }
        camera3d.depth_texture_usages.0 |= TextureUsages::TEXTURE_BINDING.bits();
        let (horizon, ray) = horizon_row(transform, projection);
        let next = ShaftView {
            sun: uv.extend(strength).extend(28.0),
            color: Vec3::from_array(lighting.diffuse)
                .lerp(Vec3::ONE, 0.18)
                .extend(0.0),
            horizon,
            ray,
        };
        if old != Some(&next) {
            commands.entity(entity).insert(next);
        }
    }
}

/// MONKEY (leftovers): the view ray's world-up component as a linear form in NDC, plus the per-axis
/// ray scale to normalise it. A pixel is sky only above `-0.01` (≈ 0.6° below the horizon): a sun
/// just under the horizon still finds sky next to it, the sea below it never does.
fn horizon_row(transform: &GlobalTransform, projection: &Projection) -> (Vec4, Vec4) {
    let Projection::Perspective(p) = projection else {
        // No horizon test: `(0, 0, 1)` is always "up".
        return (
            Vec4::new(0.0, 0.0, 1.0, -2.0),
            Vec4::new(0.0, 0.0, 0.0, 0.0),
        );
    };
    let ty = (p.fov * 0.5).tan();
    let tx = ty * p.aspect_ratio;
    let rot = transform.affine().matrix3;
    // View dir = right·x·tx + up·y·ty − back; only its world Y matters.
    let row = Vec3::new(rot.x_axis.y * tx, rot.y_axis.y * ty, -rot.z_axis.y);
    (row.extend(-0.01), Vec4::new(tx, ty, 0.0, 0.0))
}

fn smoothstep(a: f32, b: f32, value: f32) -> f32 {
    let t = ((value - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[derive(Resource)]
struct ShaftShader(Handle<Shader>);

#[derive(Resource)]
struct ShaftPipeline {
    /// The blur's layout; the mask's per MSAA (`[single, multisampled]`).
    layout: BindGroupLayoutDescriptor,
    mask_layouts: [BindGroupLayoutDescriptor; 2],
    masks: [CachedRenderPipelineId; 2],
    shader: Handle<Shader>,
    fullscreen: FullscreenShader,
    sampler: Sampler,
}

/// The half-resolution sky mask the blur marches through.
const MASK_FORMAT: TextureFormat = TextureFormat::R8Unorm;

#[derive(Component)]
struct ShaftMask(CachedTexture);

/// MONKEY (polish): the view's uniform, created once and rewritten in prepare, like bloom's.
#[derive(Component)]
struct ShaftUniform(Buffer);

#[derive(Component)]
struct ViewShaftPipeline(CachedRenderPipelineId);

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct ShaftLabel;

fn init_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    shader: Res<ShaftShader>,
    fullscreen: Res<FullscreenShader>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "post_sun_shafts_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                texture_2d(TextureSampleType::Float { filterable: true }),
                uniform_buffer::<ShaftView>(false),
            ),
        ),
    );
    let mask_layouts = [false, true].map(|multisampled| {
        BindGroupLayoutDescriptor::new(
            "post_sun_shafts_mask_layout",
            // MONKEY (leftovers): + the view uniform, for the horizon row.
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    if multisampled {
                        texture_depth_2d_multisampled()
                    } else {
                        texture_depth_2d()
                    },
                    uniform_buffer::<ShaftView>(false),
                ),
            ),
        )
    });
    let masks = [false, true].map(|multisampled| {
        let mut shader_defs = vec!["SHAFT_MASK".into()];
        if multisampled {
            shader_defs.push("MULTISAMPLED".into());
        }
        cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("post_sun_shafts_mask".into()),
            layout: vec![mask_layouts[multisampled as usize].clone()],
            vertex: fullscreen.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: shader.0.clone(),
                shader_defs,
                entry_point: Some("fs_mask".into()),
                targets: vec![Some(ColorTargetState {
                    format: MASK_FORMAT,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            ..default()
        })
    });
    commands.insert_resource(ShaftPipeline {
        layout,
        mask_layouts,
        masks,
        shader: shader.0.clone(),
        fullscreen: fullscreen.clone(),
        sampler: device.create_sampler(&SamplerDescriptor {
            min_filter: FilterMode::Linear,
            mag_filter: FilterMode::Linear,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            ..default()
        }),
    });
}

impl SpecializedRenderPipeline for ShaftPipeline {
    type Key = TextureFormat;

    fn specialize(&self, format: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("post_sun_shafts".into()),
            layout: vec![self.layout.clone()],
            vertex: self.fullscreen.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            ..default()
        }
    }
}

fn prepare_pipelines(
    mut commands: Commands,
    cache: Res<PipelineCache>,
    pipeline: Res<ShaftPipeline>,
    mut specialized: ResMut<SpecializedRenderPipelines<ShaftPipeline>>,
    views: Query<(Entity, &ViewTarget), With<ShaftView>>,
    all_views: Query<&ViewTarget, With<Camera3d>>,
) {
    // MONKEY (integration): warm every reachable key on every 3-D view, feature on or off, so the
    // compile happens under the entry cover and never live when the player turns the row on.
    // (Named in `pipe_warm/menagerie.rs`'s custom-lane census.) The mask pipelines (per MSAA) are
    // queued once at init, so only the blur's format key needs warming here.
    for target in &all_views {
        specialized.specialize(&cache, &pipeline, target.main_texture_format());
    }
    for (entity, target) in &views {
        let id = specialized.specialize(&cache, &pipeline, target.main_texture_format());
        commands.entity(entity).insert(ViewShaftPipeline(id));
    }
}

/// The half-size mask target, rounded up so an odd edge column still has a texel, and the
/// view's uniform, written in place.
fn prepare_masks(
    mut commands: Commands,
    mut textures: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    views: Query<(Entity, &ExtractedCamera, &ShaftView, Option<&ShaftUniform>)>,
) {
    for (entity, camera, shaft, uniform) in &views {
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        let rows = [
            shaft.sun.to_array(),
            shaft.color.to_array(),
            shaft.horizon.to_array(),
            shaft.ray.to_array(),
        ];
        match uniform {
            Some(uniform) => queue.write_buffer(&uniform.0, 0, bytemuck::cast_slice(&rows)),
            None => {
                let buffer = device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("post_sun_shafts_uniform"),
                    contents: bytemuck::cast_slice(&rows),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                });
                commands.entity(entity).insert(ShaftUniform(buffer));
            }
        }
        let mask = textures.get(
            &device,
            TextureDescriptor {
                label: Some("post_sun_shafts_mask"),
                size: Extent3d {
                    width: size.x.div_ceil(2).max(1),
                    height: size.y.div_ceil(2).max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: MASK_FORMAT,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );
        commands.entity(entity).insert(ShaftMask(mask));
    }
}

#[derive(Default)]
struct ShaftNode;

impl ViewNode for ShaftNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ViewDepthTexture,
        // Gates the pass: the mask and uniform outlive a disabled frame.
        &'static ShaftView,
        &'static ViewShaftPipeline,
        &'static ShaftMask,
        &'static ShaftUniform,
    );

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        context: &mut RenderContext<'w>,
        (target, depth, _, id, mask, uniform): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let cache = world.resource::<PipelineCache>();
        let settings = world.resource::<ShaftPipeline>();
        let multisampled = (depth.texture.sample_count() > 1) as usize;
        let (Some(pipeline), Some(mask_pipeline)) = (
            cache.get_render_pipeline(id.0),
            cache.get_render_pipeline(settings.masks[multisampled]),
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
        let device = context.render_device();
        let mask_bind = device.create_bind_group(
            "post_sun_shafts_mask",
            &cache.get_bind_group_layout(&settings.mask_layouts[multisampled]),
            &BindGroupEntries::sequential((depth.view(), uniform.0.as_entire_binding())),
        );
        let out = target.post_process_write();
        let bind = device.create_bind_group(
            "post_sun_shafts",
            &cache.get_bind_group_layout(&settings.layout),
            &BindGroupEntries::sequential((
                out.source,
                &settings.sampler,
                &mask.0.default_view,
                uniform.0.as_entire_binding(),
            )),
        );
        let diagnostics = context.diagnostic_recorder();
        {
            let mut pass = context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("post_sun_shafts_mask"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &mask.0.default_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations::default(),
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            let span = diagnostics.pass_span(&mut pass, "post_sun_shafts_mask");
            pass.set_pipeline(mask_pipeline);
            pass.set_bind_group(0, &mask_bind, &[]);
            pass.draw(0..3, 0..1);
            span.end(&mut pass);
        }
        let mut pass = context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("post_sun_shafts"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: out.destination,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations::default(),
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        let span = diagnostics.pass_span(&mut pass, "post_sun_shafts");
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.draw(0..3, 0..1);
        span.end(&mut pass);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_and_night_fades_are_bounded() {
        assert_eq!(smoothstep(0.0, 1.0, -1.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 2.0), 1.0);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1.0e-6);
    }

    /// MONKEY (leftovers): the shader's horizon form, evaluated on the CPU, is the view ray's
    /// world-up sine: the screen centre of a level camera sits on the horizon, a pitched-down one
    /// sees below it, and the top/bottom edges sit at ±half the vertical FOV.
    #[test]
    fn horizon_row_is_the_view_ray_elevation() {
        let p = PerspectiveProjection {
            fov: 1.0,
            aspect_ratio: 16.0 / 9.0,
            ..default()
        };
        let proj = Projection::Perspective(p.clone());
        let up = |t: &Transform, ndc: Vec2| {
            let (h, r) = horizon_row(&GlobalTransform::from(*t), &proj);
            h.truncate().dot(ndc.extend(1.0)) / (ndc * r.truncate().truncate()).extend(1.0).length()
        };
        let level = Transform::default().looking_to(Vec3::X, Vec3::Y);
        assert!(up(&level, Vec2::ZERO).abs() < 1e-5);
        assert!((up(&level, Vec2::new(0.0, 1.0)) - 0.5f32.sin()).abs() < 1e-4);
        assert!((up(&level, Vec2::new(0.0, -1.0)) + 0.5f32.sin()).abs() < 1e-4);
        let down = Transform::default().looking_to(Vec3::new(1.0, -0.3, 0.0), Vec3::Y);
        let dir = Vec3::new(1.0, -0.3, 0.0).normalize();
        assert!((up(&down, Vec2::ZERO) - dir.y).abs() < 1e-5);
        assert!(
            up(&down, Vec2::ZERO) < -0.01,
            "sea below the horizon is not sky"
        );
    }
}
