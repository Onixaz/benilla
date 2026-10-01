//! MONKEY (post): zone/day-night colour grading through a real 32³ GPU texture.
//!
//! Ported from WarcraftXL (https://github.com/WarcraftXL) by iThorgrim — module wxl-retail-grading, Grading.cpp, shaders/Grading.ps.hlsl.
//! Used with the author's permission; attribution required. Its 1024×32 BLP strip/two-bilinear-tap cube lookup becomes the equivalent hardware-trilinear
//! lookup after upload to a 3D texture. See `THIRD-PARTY.md`.

use super::bloom::BloomLabel;
use crate::video::VideoConfig;
use benilla_assets::{AssetSet, LockRecover, WorldAssets};
use benilla_world::{lighting::WorldTime, terrain_stream::CurrentArea, view::WorldCamera};
use bevy::{
    asset::RenderAssetUsages,
    core_pipeline::{core_3d::graph::Core3d, FullscreenShader},
    ecs::query::QueryItem,
    image::Image,
    prelude::*,
    render::{
        diagnostic::RecordDiagnostics,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_asset::RenderAssets,
        render_graph::{
            NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        view::ViewTarget,
        Render, RenderApp, RenderStartup, RenderSystems,
    },
};
use std::collections::{HashMap, HashSet};

pub(super) struct GradingPlugin;

#[derive(Resource)]
struct GradeCatalog {
    rows: benilla_formats::MonkeyZoneGrades,
    luts: HashMap<String, Handle<Image>>,
    identity: Handle<Image>,
}

#[derive(Component, Clone, PartialEq, ExtractComponent)]
struct GradeView {
    day: Handle<Image>,
    night: Handle<Image>,
    /// MONKEY (polish): the pair being faded out (identity once the fade is done).
    prev_day: Handle<Image>,
    prev_night: Handle<Image>,
    // x = authored strength, y = night blend, z = crossfade weight of the current pair (1 = done),
    // w = the previous pair's strength.
    control: Vec4,
}

/// MONKEY (polish): the zone crossfade's length; a border crossing no longer snaps the grade.
const GRADE_FADE_S: f32 = 2.0;

/// One zone's LUT pair at its authored strength; a zone with no row grades at strength 0.
#[derive(Clone, PartialEq, Debug)]
struct GradePair {
    day: Handle<Image>,
    night: Handle<Image>,
    strength: f32,
}

/// MONKEY (polish): the crossfade from the previous zone's pair to the current one.
#[derive(Resource, Default)]
struct GradeBlend {
    current: Option<GradePair>,
    previous: Option<GradePair>,
    /// The current pair's weight, 0..1.
    fade: f32,
    primed: bool,
}

impl GradeBlend {
    /// Steps toward `target`. The first graded zone after login (or re-enabling) and a disabled
    /// grade snap; a return to the
    /// pair still fading out reverses the fade instead of restarting it.
    fn step(&mut self, target: Option<GradePair>, dt: f32, snap: bool) {
        if snap || !self.primed {
            *self = Self {
                primed: !snap && target.is_some(),
                current: target,
                previous: None,
                fade: 1.0,
            };
            return;
        }
        if target != self.current {
            if self.fade < 1.0 && target == self.previous {
                std::mem::swap(&mut self.current, &mut self.previous);
                self.fade = 1.0 - self.fade;
            } else {
                self.previous = self.current.take();
                self.current = target;
                self.fade = 0.0;
            }
        }
        self.fade = (self.fade + dt / GRADE_FADE_S).min(1.0);
        if self.fade >= 1.0 {
            self.previous = None;
        }
    }
}

#[derive(Clone, Copy, ShaderType)]
struct GradeUniform {
    control: Vec4,
}

impl Plugin for GradingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<GradeView>::default())
            .init_resource::<GradeBlend>()
            .add_systems(Last, update_views);
        if !app.is_plugin_added::<AssetPlugin>() {
            return;
        }
        app.add_systems(Startup, load_catalog.after(AssetSet::Open));
        let shader = app
            .world_mut()
            .resource_mut::<Assets<Shader>>()
            .add(Shader::from_wgsl(
                include_str!("grading.wgsl"),
                "post/grading.wgsl",
            ));
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(GradeShader(shader))
            .add_systems(RenderStartup, init_pipeline)
            .add_systems(
                Render,
                prepare_uniforms.in_set(RenderSystems::PrepareResources),
            )
            .add_render_graph_node::<ViewNodeRunner<GradeNode>>(Core3d, GradeLabel)
            .add_render_graph_edges(
                Core3d,
                (
                    BloomLabel,
                    GradeLabel,
                    benilla_world::ffx_glow::FfxGlowLabel,
                ),
            );
    }
}

fn load_catalog(
    mut commands: Commands,
    assets: Option<Res<WorldAssets>>,
    mut images: ResMut<Assets<Image>>,
) {
    let identity = images.add(volume_image(identity_cube()));
    let Some(assets) = assets else {
        commands.insert_resource(GradeCatalog {
            rows: default(),
            luts: default(),
            identity,
        });
        return;
    };
    let mut chain = assets.chain.lock_recover();
    let loose_root = benilla_formats::wow_data();
    let loaded_rows = loose_root
        .as_ref()
        .map(|root| root.join("DBFilesClient").join("MonkeyZoneGrade.dbc"))
        .filter(|path| path.is_file())
        .map(|path| {
            std::fs::read(path)
                .map_err(anyhow::Error::from)
                .and_then(|bytes| benilla_formats::parse_monkey_zone_grades(&bytes))
        })
        .unwrap_or_else(|| benilla_formats::load_monkey_zone_grades(&mut chain));
    let rows = match loaded_rows {
        Ok(rows) => rows,
        Err(error) => {
            info!("post grading: no MonkeyZoneGrade.dbc; identity fallback ({error:#})");
            commands.insert_resource(GradeCatalog {
                rows: default(),
                luts: default(),
                identity,
            });
            return;
        }
    };
    let paths: HashSet<String> = rows
        .rows()
        .flat_map(|row| [row.day_lut.clone(), row.night_lut.clone()])
        .collect();
    let mut luts = HashMap::new();
    for path in paths {
        let loose = loose_root
            .as_ref()
            .map(|root| root.join(path.replace('\\', "/")))
            .filter(|path| path.is_file())
            .map(std::fs::read)
            .transpose()
            .map_err(anyhow::Error::from);
        let result = loose
            .and_then(|bytes| match bytes {
                Some(bytes) => Ok(bytes),
                None => chain.read_file(&path.replace('/', "\\")),
            })
            .and_then(|bytes| benilla_formats::blp_bytes_to_mip_chain(&bytes));
        match result.and_then(|blp| strip_to_cube(&blp)) {
            Ok(cube) => {
                luts.insert(path, images.add(volume_image(cube)));
            }
            Err(error) => warn!("post grading: {path} rejected; identity fallback: {error:#}"),
        }
    }
    info!(
        "post grading: {} area rows, {} resident LUTs",
        rows.rows().count(),
        luts.len()
    );
    commands.insert_resource(GradeCatalog {
        rows,
        luts,
        identity,
    });
}

fn strip_to_cube(blp: &benilla_formats::BlpMipChain) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(
        blp.width == 1024 && blp.height == 32,
        "expected 1024x32, got {}x{}",
        blp.width,
        blp.height
    );
    anyhow::ensure!(!blp.mips.is_empty(), "BLP has no base mip");
    let strip = &blp.mips[0];
    anyhow::ensure!(
        strip.len() == 1024 * 32 * 4,
        "base mip has {} bytes",
        strip.len()
    );
    let mut cube = vec![0; 32 * 32 * 32 * 4];
    for b in 0..32usize {
        for g in 0..32usize {
            for r in 0..32usize {
                let source = (g * 1024 + b * 32 + r) * 4;
                let destination = ((b * 32 + g) * 32 + r) * 4;
                cube[destination..destination + 4].copy_from_slice(&strip[source..source + 4]);
            }
        }
    }
    Ok(cube)
}

fn identity_cube() -> Vec<u8> {
    let mut cube = Vec::with_capacity(32 * 32 * 32 * 4);
    for b in 0..32u32 {
        for g in 0..32u32 {
            for r in 0..32u32 {
                cube.extend([r, g, b].map(|v| ((v * 255 + 15) / 31) as u8));
                cube.push(255);
            }
        }
    }
    cube
}

fn volume_image(cube: Vec<u8>) -> Image {
    Image::new(
        Extent3d {
            width: 32,
            height: 32,
            depth_or_array_layers: 32,
        },
        TextureDimension::D3,
        cube,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    )
}

#[allow(clippy::too_many_arguments)]
fn update_views(
    mut commands: Commands,
    video: Res<VideoConfig>,
    catalog: Option<Res<GradeCatalog>>,
    area: Res<CurrentArea>,
    // MONKEY (reviewfix-a): a capture has no avatar; it grades by the area under the camera.
    capture_area: Option<Res<benilla_world::terrain_stream::CaptureCameraArea>>,
    areas: Option<Res<crate::area::AreaTableRes>>,
    time: Res<WorldTime>,
    // MONKEY (reviewfix-a): the minute the lighting actually rendered (server, or the manual /
    // capture clock); `WorldTime` alone is the server's and stays at its noon default offline.
    rendered: Option<Res<benilla_world::lighting::GameClock>>,
    clock: Res<Time>,
    mut blend: ResMut<GradeBlend>,
    cameras: Query<(Entity, &Camera, Option<&GradeView>), With<WorldCamera>>,
) {
    let selected = catalog.as_ref().and_then(|catalog| {
        let leaf = area.0.or_else(|| capture_area.as_ref().and_then(|a| a.0))?;
        let zone = areas
            .as_ref()
            .and_then(|areas| areas.0.top_zone(leaf))
            .unwrap_or(leaf);
        let row = catalog
            .rows
            .for_area(leaf)
            .or_else(|| catalog.rows.for_area(zone))?;
        (row.strength > 0.0).then(|| GradePair {
            day: catalog
                .luts
                .get(&row.day_lut)
                .unwrap_or(&catalog.identity)
                .clone(),
            night: catalog
                .luts
                .get(&row.night_lut)
                .unwrap_or(&catalog.identity)
                .clone(),
            strength: row.strength,
        })
    });
    blend.step(selected, clock.delta_secs(), !video.color_grading);
    let minute = rendered_minute(&time, rendered.as_deref());
    let selected = catalog.as_ref().and_then(|catalog| {
        if blend.current.is_none() && blend.previous.is_none() {
            return None;
        }
        let pair = |p: &Option<GradePair>| {
            p.as_ref().map_or_else(
                || (catalog.identity.clone(), catalog.identity.clone(), 0.0),
                |p| (p.day.clone(), p.night.clone(), p.strength),
            )
        };
        let (day, night, strength) = pair(&blend.current);
        let (prev_day, prev_night, prev_strength) = pair(&blend.previous);
        Some(GradeView {
            day,
            night,
            prev_day,
            prev_night,
            control: Vec4::new(strength, night_weight(minute), blend.fade, prev_strength),
        })
    });
    for (entity, camera, old) in &cameras {
        let next = (video.color_grading && camera.is_active)
            .then(|| selected.clone())
            .flatten();
        match next {
            Some(next) if old != Some(&next) => {
                commands.entity(entity).insert(next);
            }
            None if old.is_some() => {
                commands.entity(entity).remove::<GradeView>();
            }
            _ => {}
        }
    }
}

/// MONKEY (reviewfix-a): the server's fractional minute while it is the minute being rendered,
/// else the rendered (manual or capture) minute.
pub(crate) fn rendered_minute(
    time: &WorldTime,
    rendered: Option<&benilla_world::lighting::GameClock>,
) -> f32 {
    match rendered {
        Some(c) if time.minute_f.floor() as u32 != c.minute => c.minute as f32,
        _ => time.minute_f,
    }
}

fn night_weight(minute: f32) -> f32 {
    let dawn = smoothstep(330.0, 450.0, minute);
    let dusk = smoothstep(1110.0, 1230.0, minute);
    1.0 - dawn * (1.0 - dusk)
}

fn smoothstep(a: f32, b: f32, value: f32) -> f32 {
    let t = ((value - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[derive(Resource)]
struct GradeShader(Handle<Shader>);

#[derive(Resource)]
struct GradePipeline {
    layout: BindGroupLayoutDescriptor,
    scene_sampler: Sampler,
    lut_sampler: Sampler,
    pipeline: CachedRenderPipelineId,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct GradeLabel;

fn init_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<PipelineCache>,
    shader: Res<GradeShader>,
    fullscreen: Res<FullscreenShader>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "post_grading_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                texture_3d(TextureSampleType::Float { filterable: true }),
                texture_3d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                uniform_buffer::<GradeUniform>(false),
                // MONKEY (polish): the previous zone's pair, crossfaded out.
                texture_3d(TextureSampleType::Float { filterable: true }),
                texture_3d(TextureSampleType::Float { filterable: true }),
            ),
        ),
    );
    let pipeline = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("post_grading".into()),
        layout: vec![layout.clone()],
        vertex: fullscreen.to_vertex_state(),
        fragment: Some(FragmentState {
            shader: shader.0.clone(),
            shader_defs: vec![],
            entry_point: Some("fragment".into()),
            targets: vec![Some(ColorTargetState {
                format: ViewTarget::TEXTURE_FORMAT_HDR,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        ..default()
    });
    let sampler = |filter| {
        device.create_sampler(&SamplerDescriptor {
            min_filter: filter,
            mag_filter: filter,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            ..default()
        })
    };
    commands.insert_resource(GradePipeline {
        layout,
        scene_sampler: sampler(FilterMode::Nearest),
        lut_sampler: sampler(FilterMode::Linear),
        pipeline,
    });
}

/// MONKEY (polish): the view's uniform, created once and rewritten in prepare, like bloom's.
#[derive(Component)]
struct GradeUniformBuffer(Buffer);

fn prepare_uniforms(
    mut commands: Commands,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    views: Query<(Entity, &GradeView, Option<&GradeUniformBuffer>)>,
) {
    for (entity, grade, buffer) in &views {
        let rows = [grade.control.to_array()];
        match buffer {
            Some(buffer) => queue.write_buffer(&buffer.0, 0, bytemuck::cast_slice(&rows)),
            None => {
                let buffer = device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("post_grading_uniform"),
                    contents: bytemuck::cast_slice(&rows),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                });
                commands.entity(entity).insert(GradeUniformBuffer(buffer));
            }
        }
    }
}

#[derive(Default)]
struct GradeNode;

impl ViewNode for GradeNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static GradeView,
        &'static GradeUniformBuffer,
    );

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        context: &mut RenderContext<'w>,
        (target, grade, uniform): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let settings = world.resource::<GradePipeline>();
        let cache = world.resource::<PipelineCache>();
        let Some(pipeline) = cache.get_render_pipeline(settings.pipeline) else {
            return Ok(());
        };
        let images = world.resource::<RenderAssets<GpuImage>>();
        let (Some(day), Some(night), Some(prev_day), Some(prev_night)) = (
            images.get(&grade.day),
            images.get(&grade.night),
            images.get(&grade.prev_day),
            images.get(&grade.prev_night),
        ) else {
            return Ok(());
        };
        let device = context.render_device();
        let out = target.post_process_write();
        let bind = device.create_bind_group(
            "post_grading",
            &cache.get_bind_group_layout(&settings.layout),
            &BindGroupEntries::sequential((
                out.source,
                &settings.scene_sampler,
                &day.texture_view,
                &night.texture_view,
                &settings.lut_sampler,
                uniform.0.as_entire_binding(),
                &prev_day.texture_view,
                &prev_night.texture_view,
            )),
        );
        let diagnostics = context.diagnostic_recorder();
        let mut pass = context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("post_grading"),
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
        let span = diagnostics.pass_span(&mut pass, "post_grading");
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

    /// MONKEY (reviewfix-a): offline (capture / manual clock) the grade follows the rendered
    /// minute, not the server's noon default; live it keeps the server's fraction.
    #[test]
    fn grade_night_weight_follows_the_rendered_minute() {
        let server = WorldTime::default();
        let manual = benilla_world::lighting::GameClock {
            minute: 1260,
            ..Default::default()
        };
        assert_eq!(rendered_minute(&server, Some(&manual)), 1260.0);
        assert_eq!(night_weight(rendered_minute(&server, Some(&manual))), 1.0);
        let live = WorldTime {
            minute_f: 1260.5,
            ..Default::default()
        };
        assert_eq!(rendered_minute(&live, Some(&manual)), 1260.5);
        assert_eq!(rendered_minute(&live, None), 1260.5);
    }

    #[test]
    fn identity_volume_has_exact_axes() {
        let cube = identity_cube();
        let at = |r: usize, g: usize, b: usize| &cube[((b * 32 + g) * 32 + r) * 4..][..4];
        assert_eq!(at(0, 0, 0), &[0, 0, 0, 255]);
        assert_eq!(at(31, 0, 0), &[255, 0, 0, 255]);
        assert_eq!(at(0, 31, 0), &[0, 255, 0, 255]);
        assert_eq!(at(0, 0, 31), &[0, 0, 255, 255]);
    }

    #[test]
    fn zone_change_crossfades_and_a_quick_return_reverses() {
        let pair = |id: u128, strength| GradePair {
            day: Handle::Uuid(bevy::asset::uuid::Uuid::from_u128(id), default()),
            night: Handle::Uuid(bevy::asset::uuid::Uuid::from_u128(id + 1), default()),
            strength,
        };
        let (a, b) = (pair(10, 1.0), pair(20, 0.5));
        let mut blend = GradeBlend::default();
        // No zone yet, then the first graded zone snaps in.
        blend.step(None, 0.016, false);
        blend.step(Some(a.clone()), 0.016, false);
        assert_eq!((blend.current.clone(), blend.fade), (Some(a.clone()), 1.0));
        // A border crossing fades over GRADE_FADE_S.
        blend.step(Some(b.clone()), 0.5, false);
        assert_eq!(blend.previous, Some(a.clone()));
        assert!((blend.fade - 0.25).abs() < 1e-6);
        // Stepping back mid-fade reverses from the same image.
        blend.step(Some(a.clone()), 0.0, false);
        assert_eq!(
            (blend.current.clone(), blend.previous.clone()),
            (Some(a.clone()), Some(b))
        );
        assert!((blend.fade - 0.75).abs() < 1e-6);
        blend.step(Some(a.clone()), 1.0, false);
        assert_eq!((blend.fade, blend.previous.is_none()), (1.0, true));
        // An ungraded zone fades out to identity rather than cutting.
        blend.step(None, 1.0, false);
        assert_eq!(
            (blend.current.is_none(), blend.previous.clone()),
            (true, Some(a))
        );
        assert!((blend.fade - 0.5).abs() < 1e-6);
        // Disabling snaps.
        blend.step(None, 0.0, true);
        assert!(blend.previous.is_none() && blend.current.is_none());
    }

    #[test]
    fn day_night_blend_is_continuous() {
        assert_eq!(night_weight(0.0), 1.0);
        assert_eq!(night_weight(720.0), 0.0);
        assert_eq!(night_weight(1439.0), 1.0);
        assert!((night_weight(390.0) - 0.5).abs() < 1.0e-6);
    }
}
