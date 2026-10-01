//! Main-pass depth for water. The shared image fits ExtendedMaterial's existing bind group;
//! only the world camera writes it (portrait/preview views never overwrite it). This belongs
//! in world/liquid because visibility and the WorldCamera marker are world responsibilities.
//!
//! MONKEY (enhanced water: refraction): the same node also copies the opaque COLOUR, the frame
//! as it stood before any water drew, into `WaterColourImage` for the module to look through.
use benilla_assets::{materials::LiquidMaterial, WaterColourImage, WaterDepthImage, WaterQuality};
use bevy::camera::visibility::VisibleEntities;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::ecs::query::QueryItem;
use bevy::image::ToExtents;
use bevy::prelude::*;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::*;
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::texture::GpuImage;
use bevy::render::view::{ViewDepthTexture, ViewTarget};
use bevy::render::{RenderApp, RenderStartup};

#[derive(Component, Clone, ExtractComponent)]
struct WaterView;

#[derive(Resource, Clone, ExtractResource)]
struct DepthSource {
    image: Handle<Image>,
    colour: Handle<Image>,
    active: bool,
}

/// `WOW_WATER=0|1|2` overrides settings for this run, read once at plugin startup.
pub struct WaterDepthPlugin;

#[derive(Resource)]
struct WaterOverride(Option<u8>);

impl Plugin for WaterDepthPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WaterQuality>()
            .init_resource::<WaterDepthImage>()
            .init_resource::<WaterColourImage>();
        let value = std::env::var("WOW_WATER")
            .ok()
            .and_then(|v| v.parse::<u8>().ok())
            .filter(|v| *v <= 2);
        if let Some(value) = value {
            app.insert_resource(WaterQuality(value));
        }
        let image = app.world().resource::<WaterDepthImage>().0.clone();
        let colour = app.world().resource::<WaterColourImage>().0.clone();
        app.insert_resource(WaterOverride(value))
            .insert_resource(DepthSource {
                image,
                colour,
                active: false,
            })
            .add_plugins((
                ExtractComponentPlugin::<WaterView>::default(),
                ExtractResourcePlugin::<DepthSource>::default(),
            ))
            // Camera sizes and the per-view visible list are final by Last, before extraction.
            .add_systems(Last, update_water_depth);
        let shader = app
            .world_mut()
            .resource_mut::<Assets<Shader>>()
            .add(Shader::from_wgsl(
                RESOLVE_SHADER,
                "water_depth_resolve.wgsl",
            ));
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(DepthShader(shader))
            .add_systems(RenderStartup, init_pipeline)
            .add_render_graph_node::<ViewNodeRunner<DepthNode>>(Core3d, WaterDepthLabel)
            // static_gx precedes MainOpaquePass; this sees BOTH solid-geometry paths.
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::MainOpaquePass,
                    WaterDepthLabel,
                    Node3d::MainTransparentPass,
                ),
            );
    }
}

fn update_water_depth(
    mut commands: Commands,
    mut cameras: Query<
        (Entity, &Camera, &mut Camera3d, &VisibleEntities),
        With<crate::view::WorldCamera>,
    >,
    liquids: Query<&MeshMaterial3d<LiquidMaterial>>,
    mut materials: ResMut<Assets<LiquidMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut source: ResMut<DepthSource>,
    mut quality: ResMut<WaterQuality>,
    override_value: Res<WaterOverride>,
    light: Res<crate::lighting::WowLighting>,
) {
    if let Some(value) = override_value.0 {
        if quality.0 != value {
            quality.0 = value;
        }
    }
    // The dome's resolved, byte-quantized sky rows, converted to linear RGB.
    let linear = |row: [f32; 3]| {
        let c = benilla_assets::quant255(row).map(|v| {
            if v <= 0.04045 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        });
        Vec4::new(c[0], c[1], c[2], 0.0)
    };
    let zenith = linear(light.sky[0]);
    let horizon = linear(light.sky[4]);
    let celestial = if light.celestial_dir.y > 0.0 {
        light.celestial_dir.extend(0.0)
    } else {
        light.moon_dir_white.extend(1.0)
    };
    // Evaluated after lighting resolves each frame; mutate only on actual changes.
    let tier = quality.0.min(2) as f32;
    let changed: Vec<_> = materials
        .iter()
        .filter(|(_, m)| {
            let water = &m.extension.water;
            water.lane.z < 0.5
                && (water.mode.x != tier
                    || (quality.0 > 0
                        && (water.sky_zenith != zenith
                            || water.sky_horizon != horizon
                            || water.celestial != celestial)))
        })
        .map(|(id, _)| id)
        .collect();
    for id in changed {
        let water = &mut materials.get_mut(id).unwrap().extension.water;
        water.mode.x = tier;
        water.sky_zenith = zenith;
        water.sky_horizon = horizon;
        water.celestial = celestial;
    }
    source.active = false;
    for (entity, camera, mut camera3d, visible) in &mut cameras {
        commands.entity(entity).insert(WaterView);
        let usage = TextureUsages::from(camera3d.depth_texture_usages);
        if !usage.contains(TextureUsages::TEXTURE_BINDING) {
            camera3d.depth_texture_usages = (usage | TextureUsages::TEXTURE_BINDING).into();
        }
        if quality.0 == 0 || !camera.is_active {
            continue;
        }
        source.active = visible
            .get(std::any::TypeId::of::<Mesh3d>())
            .iter()
            .any(|entity| {
                liquids
                    .get(*entity)
                    .ok()
                    .and_then(|m| materials.get(&m.0))
                    .is_some_and(|m| m.extension.water.lane.z < 0.5)
            });
        if let Some(size) = camera.physical_target_size() {
            if size.x > 0 && size.y > 0 {
                let stale = |handle: &Handle<Image>| {
                    images.get(handle).is_some_and(|image| image.size() != size)
                };
                if stale(&source.image) || stale(&source.colour) {
                    for handle in [source.image.clone(), source.colour.clone()] {
                        if let Some(image) = images.get_mut(&handle) {
                            if image.size() != size {
                                image.resize(size.to_extents());
                            }
                        }
                    }
                    // A resize replaces the GPU view behind this stable handle. Rebuild all
                    // material bindings, including opaque liquids sharing the fallback binding.
                    let ids: Vec<_> = materials.ids().collect();
                    for id in ids {
                        let _ = materials.get_mut(id);
                    }
                }
            }
        }
    }
}

#[derive(Resource)]
struct DepthPipelines {
    layouts: [BindGroupLayoutDescriptor; 2],
    pipelines: [CachedRenderPipelineId; 2],
    colour_layout: BindGroupLayoutDescriptor,
    colour_pipeline: CachedRenderPipelineId,
}

#[derive(Resource)]
struct DepthShader(Handle<Shader>);

fn init_pipeline(mut commands: Commands, shader: Res<DepthShader>, cache: Res<PipelineCache>) {
    let shader = &shader.0;
    let layouts = [false, true].map(|multisampled| {
        BindGroupLayoutDescriptor::new(
            "water depth source",
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Depth,
                    view_dimension: TextureViewDimension::D2,
                    multisampled,
                },
                count: None,
            }],
        )
    });
    let pipelines = [0, 1].map(|i| {
        cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("water depth resolve".into()),
            layout: vec![layouts[i].clone()],
            vertex: VertexState {
                shader: shader.clone(),
                entry_point: Some("vertex".into()),
                ..default()
            },
            fragment: Some(FragmentState {
                shader: shader.clone(),
                entry_point: Some("fragment".into()),
                shader_defs: if i == 1 {
                    vec!["MULTISAMPLED".into()]
                } else {
                    vec![]
                },
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::R32Float,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            ..default()
        })
    });
    let colour_layout = BindGroupLayoutDescriptor::new(
        "water colour source",
        &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: false },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }],
    );
    let colour_pipeline = cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("water colour copy".into()),
        layout: vec![colour_layout.clone()],
        vertex: VertexState {
            shader: shader.clone(),
            entry_point: Some("vertex".into()),
            ..default()
        },
        fragment: Some(FragmentState {
            shader: shader.clone(),
            entry_point: Some("colour".into()),
            shader_defs: vec!["COLOUR".into()],
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        ..default()
    });
    commands.insert_resource(DepthPipelines {
        layouts,
        pipelines,
        colour_layout,
        colour_pipeline,
    });
}

// MONKEY (ao): public so the AO pass can order itself before the opaque copies.
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct WaterDepthLabel;
#[derive(Default)]
struct DepthNode;

impl ViewNode for DepthNode {
    type ViewQuery = (
        &'static WaterView,
        &'static ViewDepthTexture,
        &'static ViewTarget,
    );
    fn run<'w>(
        &self,
        _: &mut RenderGraphContext,
        context: &mut RenderContext<'w>,
        (_, depth, target): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let source = world.resource::<DepthSource>();
        if !source.active {
            return Ok(());
        }
        let images = world.resource::<RenderAssets<GpuImage>>();
        let Some(image) = images.get(&source.image) else {
            return Ok(());
        };
        if image.texture.size() != depth.texture.size() {
            return Ok(());
        }
        let pipelines = world.resource::<DepthPipelines>();
        let cache = world.resource::<PipelineCache>();
        let index = usize::from(depth.texture.sample_count() > 1);
        let Some(pipeline) = cache.get_render_pipeline(pipelines.pipelines[index]) else {
            return Ok(());
        };
        // Both copies or neither: a depth copy without its colour would show the module a real bed
        // through a black frame (shallows flash black while the colour pipeline compiles). With
        // neither, the cleared depth reads as sky, the water is opaque, and nothing shows.
        let Some(colour_pipeline) = cache.get_render_pipeline(pipelines.colour_pipeline) else {
            return Ok(());
        };
        let Some(colour) = images.get(&source.colour) else {
            return Ok(());
        };
        if colour.texture.size() != depth.texture.size() {
            return Ok(());
        }
        let device = world.resource::<RenderDevice>();
        let bind = device.create_bind_group(
            "water depth source",
            &cache.get_bind_group_layout(&pipelines.layouts[index]),
            &BindGroupEntries::single(depth.view()),
        );
        let mut pass = context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("water opaque depth copy"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &image.texture_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: Operations {
                        load: LoadOp::Clear(Default::default()),
                        store: StoreOp::Store,
                    },
                })],
                ..default()
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);

        // The opaque colour, for the refraction. `main_texture_view` is the single-sample texture
        // the opaque pass resolved into, so no MSAA variant is needed here.
        let bind = device.create_bind_group(
            "water colour source",
            &cache.get_bind_group_layout(&pipelines.colour_layout),
            &BindGroupEntries::single(target.main_texture_view()),
        );
        let mut pass = context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("water opaque colour copy"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &colour.texture_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: Operations {
                        load: LoadOp::Clear(Default::default()),
                        store: StoreOp::Store,
                    },
                })],
                ..default()
            });
        pass.set_pipeline(colour_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}

// Sample zero is intentional: averaging reverse-Z samples invents surfaces at silhouettes.
const RESOLVE_SHADER: &str = r#"
#ifdef COLOUR
@group(0) @binding(0) var source: texture_2d<f32>;
#else ifdef MULTISAMPLED
@group(0) @binding(0) var depth: texture_depth_multisampled_2d;
#else
@group(0) @binding(0) var depth: texture_depth_2d;
#endif
@vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}
#ifdef COLOUR
@fragment fn colour(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(source, vec2<i32>(p.xy), 0);
}
#else
@fragment fn fragment(@builtin(position) p: vec4<f32>) -> @location(0) f32 {
    return textureLoad(depth, vec2<i32>(p.xy), 0);
}
#endif
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstructed_water_depth_is_height_not_view_ray_length() {
        let projection = Mat4::perspective_infinite_reverse_rh(1.0, 1.6, 0.1);
        // Same submerged wall point from a low grazing view and an overhead view.
        // Opacity must see the same 2 yd depth even though ray lengths differ greatly.
        let wall = Vec3::new(0.0, 8.0, 0.0);
        for eye in [Vec3::new(0.0, 12.0, 30.0), Vec3::new(0.0, 40.0, 3.0)] {
            let world_from_view = Mat4::look_at_rh(eye, wall, Vec3::Y).inverse();
            let clip_from_world = projection * world_from_view.inverse();
            let clip = clip_from_world * wall.extend(1.0);
            let ndc = clip / clip.w;
            let viewport = Vec4::new(32.0, 16.0, 1600.0, 900.0);
            let pixel = (ndc.truncate().truncate() * Vec2::new(0.5, -0.5) + Vec2::splat(0.5))
                * Vec2::new(viewport.z, viewport.w)
                + Vec2::new(viewport.x, viewport.y);
            let xy = ((pixel - Vec2::new(viewport.x, viewport.y))
                / Vec2::new(viewport.z, viewport.w))
                * Vec2::new(2.0, -2.0)
                + Vec2::new(-1.0, 1.0);
            let scene_h = clip_from_world.inverse() * Vec4::new(xy.x, xy.y, ndc.z, 1.0);
            let vertical_depth = 10.0 - scene_h.y / scene_h.w;
            assert!((vertical_depth - 2.0).abs() < 0.001);
        }
    }

    #[test]
    fn reverse_z_thickness_handles_finite_and_infinite_projections() {
        // Project real view-space positions, then exercise the WGSL reconstruction law.
        // Off-axis rays must return ray length, not merely the eye-Z gap.
        for projection in [
            Mat4::perspective_infinite_reverse_rh(1.0, 1.6, 0.1),
            Mat4::perspective_rh(1.0, 1.6, 1000.0, 0.1),
        ] {
            for ray in [Vec3::NEG_Z, Vec3::new(0.4, -0.3, -1.0).normalize()] {
                for (water, bed) in [(2.0, 2.2), (40.0, 52.0), (200.0, 240.0)] {
                    let linearise = |distance: f32| {
                        let clip = projection * (ray * distance).extend(1.0);
                        projection.w_axis.z / (clip.z / clip.w + projection.z_axis.z)
                    };
                    let thickness = (linearise(bed) - linearise(water)) / ray.z.abs();
                    assert!((thickness - (bed - water)).abs() < 0.002);
                }
            }
        }
    }
}
