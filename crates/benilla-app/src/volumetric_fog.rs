//! MONKEY (volumetric fog): gamma-space distance haze and shadow-map light shafts.
//! Runs after world transparencies and before the world's gamma decode. Bevy's physical
//! fog loses energy in shade; this artistic lane preserves luminance while converging
//! toward the zone fog chromaticity. All shader code lives here with its pipeline.
use crate::{
    shadow_core::{ShadowSet, ShadowSun},
    video::VideoConfig,
};
use benilla_world::{
    lighting::{ResolvedPointLight, ResolvedPointLights, WorldTime, WowLighting},
    view::WorldCamera,
    weather::WeatherState,
    wmo_portal::CameraInteriorClaim,
};
use bevy::{
    camera::primitives::{Frustum, Sphere as CullSphere},
    core_pipeline::{
        core_3d::graph::{Core3d, Node3d},
        FullscreenShader,
    },
    ecs::query::QueryItem,
    light::VolumetricLight,
    pbr::{
        GpuLights, LightMeta, ViewLightsUniformOffset, ViewShadowBindings, MAX_CASCADES_PER_LIGHT,
        MAX_DIRECTIONAL_LIGHTS,
    },
    prelude::*,
    render::{
        diagnostic::RecordDiagnostics,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_graph::{
            NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderQueue},
        view::{Msaa, ViewDepthTexture, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
        Render, RenderApp, RenderStartup, RenderSystems,
    },
    shader::ShaderDefVal,
};

pub(crate) struct VolumetricFogPlugin;
#[derive(Resource)]
struct FogOverride {
    fog: Option<u8>,
    lamp: Option<u8>,
    shaft_gain: f32,
}

const MAX_FOG_LAMPS: usize = 32;
const LOW_FOG_LAMPS: usize = 16;
const FOG_LAMP_RADIUS: f32 = 100.0;

#[derive(Clone, Copy)]
struct FogLamps {
    positions: [Vec4; MAX_FOG_LAMPS],
    colors: [Vec4; MAX_FOG_LAMPS],
    count: usize,
}

impl Default for FogLamps {
    fn default() -> Self {
        Self {
            positions: [Vec4::ZERO; MAX_FOG_LAMPS],
            colors: [Vec4::ZERO; MAX_FOG_LAMPS],
            count: 0,
        }
    }
}

#[derive(Component, Clone, Copy, ExtractComponent, ShaderType)]
struct FogView {
    // Gamma byte-space colour and extinction per yard.
    color_density: Vec4,
    // Sun colour and strength, independent of PBR lux/exposure.
    sun_strength: Vec4,
    // Actual celestial direction and raymarch step count.
    direction_steps: Vec4,
    // MONKEY (fog): MonkeyFrame fog rows 1-3. Modern (fog3.x > 0) colours the haze with the shared
    // fog colour function and keeps sky pixels (depth 0) out of the distance haze.
    mf_fog1: Vec4,
    mf_fog2: Vec4,
    mf_fog3: Vec4,
    // MONKEY (lampfog): count + strength, followed by 32 resolved point-table rows. Colour already
    // includes intensity, flame flicker and (for synthesised fire) fireLightGain.
    lamp_meta: Vec4,
    lamp_positions: [Vec4; MAX_FOG_LAMPS],
    lamp_colors: [Vec4; MAX_FOG_LAMPS],
}

impl Plugin for VolumetricFogPlugin {
    fn build(&self, app: &mut App) {
        let value = std::env::var("WOW_VOLFOG")
            .ok()
            .and_then(|v| v.parse::<u8>().ok())
            .filter(|v| *v <= 2);
        // MONKEY (lampfog): session-only capture override; the persistent setting is `lampFog`.
        let lamp = std::env::var("WOW_LAMPFOG")
            .ok()
            .and_then(|v| v.parse::<u8>().ok())
            .filter(|v| *v <= 2);
        // Dev capture instrument: isolate shafts without changing surface shadows or haze.
        // MONKEY (integration): the dev door is `run_mode` (decision 1179).
        let shaft_gain =
            if crate::run_mode::dev_affordances() && std::env::var_os("WOW_CAPTURE").is_some() {
                std::env::var("WOW_VOLFOG_SHAFT_GAIN")
                    .ok()
                    .and_then(|v| v.parse::<f32>().ok())
                    .filter(|v| v.is_finite())
                    .unwrap_or(2.0)
                    .clamp(0.0, 8.0)
            } else {
                2.0
            };
        app.insert_resource(FogOverride {
            fog: value,
            lamp,
            shaft_gain,
        })
        .add_plugins(ExtractComponentPlugin::<FogView>::default())
        // MONKEY (followups): the streamed-caster refresh moved to `world_shadow.rs`; it no longer
        // depends on this cvar.
        .add_systems(Last, update_fog.after(ShadowSet::Lanes));
        // Keep headless policy tests independent of the renderer.
        if !app.is_plugin_added::<AssetPlugin>() {
            return;
        }
        let shader = app
            .world_mut()
            .resource_mut::<Assets<Shader>>()
            .add(Shader::from_wgsl(FOG_SHADER, "volumetric_fog.rs"));
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(FogShader(shader))
            .init_resource::<SpecializedRenderPipelines<FogPipeline>>()
            .add_systems(RenderStartup, init_pipeline)
            .add_systems(
                Render,
                (
                    prepare_pipelines.in_set(RenderSystems::Prepare),
                    // MONKEY (reviewfix-a): one uniform per view, rewritten, not allocated per frame.
                    prepare_fog_uniforms.in_set(RenderSystems::PrepareResources),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<FogNode>>(Core3d, FogLabel)
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::EndMainPass,
                    FogLabel,
                    Node3d::StartMainPassPostProcessing,
                ),
            );
    }
}

/// MONKEY (visualfix): how near (yd, WoW XY) an ocean footprint must come to arm the sea plane.
const SEA_PLANE_REACH: f32 = 400.0;
/// MONKEY (visualfix): the eye must be this far above sea level (yd) for the plane to apply.
const SEA_LEVEL_MIN_EYE: f32 = 0.25;

pub(crate) fn density(minute: f32, weather: f32, indoors: bool) -> f32 {
    let dawn = (1.0 - ((minute - 390.0) / 90.0).abs()).clamp(0.0, 1.0);
    let dawn = dawn * dawn * (3.0 - 2.0 * dawn);
    0.004 * (1.0 + dawn * 1.5 + weather.clamp(0.0, 1.0) * 2.0) * if indoors { 0.18 } else { 1.0 }
}

// MONKEY (lampfog): independent near-field haze floor. Full daylight is exactly zero; weather
// only strengthens an already-night-time halo, and interiors keep a restrained local version.
fn lamp_haze_strength(sun_height: f32, weather: f32, indoors: bool) -> f32 {
    let daylight = ((sun_height + 0.02) / 0.15).clamp(0.0, 1.0);
    (1.0 - daylight) * (0.8 + 1.2 * weather.clamp(0.0, 1.0)) * if indoors { 0.55 } else { 1.0 }
}

// MONKEY (lampfog): pick from the exact CPU mirror of the packed point table. A sphere test keeps
// off-screen fixtures whose halo can still cross the view. Both exterior and interior rows are
// eligible: a street camera can see facade/room fixtures, and scene depth—not a camera-lane guess—
// cuts the view segment. This pass is deliberately unshadowed, so the remaining leakage is the
// documented tradeoff until point-light cube shadowing exists.
fn select_fog_lamps(
    points: &[ResolvedPointLight],
    camera_position: Vec3,
    frustum: Option<&Frustum>,
    limit: usize,
) -> FogLamps {
    let mut candidates: Vec<(f32, ResolvedPointLight)> = points
        .iter()
        .copied()
        .filter_map(|point| {
            let point_is_indoors = point.lane > 0.5;
            if point.color.max_element() <= 0.0001 {
                return None;
            }
            let distance_squared = point.position.distance_squared(camera_position);
            if distance_squared > FOG_LAMP_RADIUS * FOG_LAMP_RADIUS {
                return None;
            }
            let reach = if point_is_indoors {
                point.lane
            } else {
                point.range
            }
            .clamp(1.0, 48.0);
            let visible = frustum.is_none_or(|frustum| {
                frustum.intersects_sphere(
                    &CullSphere {
                        center: point.position.into(),
                        radius: reach,
                    },
                    false,
                )
            });
            visible.then_some((distance_squared, point))
        })
        .collect();
    candidates.sort_by(|left, right| left.0.total_cmp(&right.0));

    let mut selected = FogLamps::default();
    for (_, point) in candidates.into_iter().take(limit.min(MAX_FOG_LAMPS)) {
        let reach = if point.lane > 0.5 {
            point.lane
        } else {
            point.range
        }
        .clamp(1.0, 48.0);
        selected.positions[selected.count] = point.position.extend(reach);
        selected.colors[selected.count] = point.color.extend(0.0);
        selected.count += 1;
    }
    selected
}

#[allow(clippy::type_complexity)]
fn update_fog(
    mut commands: Commands,
    video: Res<VideoConfig>,
    override_value: Res<FogOverride>,
    lighting: Res<WowLighting>,
    clock: Res<WorldTime>,
    weather: Res<WeatherState>,
    interior: Res<CameraInteriorClaim>,
    // MONKEY (lampfog): the resolved point-table view published by global_light's packer.
    point_lights: Res<ResolvedPointLights>,
    // MONKEY (visualfix): the minute the lighting rendered (manual / capture clock offline).
    rendered: Option<Res<benilla_world::lighting::GameClock>>,
    // MONKEY (fog): the Modern fog rows (Option: absent in unit tests).
    monkey: Option<Res<benilla_world::lighting::MonkeyFrame>>,
    mut cameras: Query<
        (
            Entity,
            &Camera,
            &mut Camera3d,
            Option<&GlobalTransform>,
            Option<&Frustum>,
        ),
        With<WorldCamera>,
    >,
    suns: Query<(Entity, &DirectionalLight), With<ShadowSun>>,
    // MONKEY (visualfix): ocean surfaces, for the haze's sea-level plane.
    water: Query<&benilla_world::liquid::WaterChunkInfo>,
    // GFX (volumetric light): while that lane is live it owns the shafts; the haze keeps its own.
    vol_light: Option<Res<crate::volumetric_light::VolLightOverride>>,
) {
    let tier = override_value.fog.unwrap_or(video.volumetric_fog).min(2);
    let shafts_retired = vol_light.is_some_and(|v| v.tier(&video) > 0);
    let lamp_tier = override_value.lamp.unwrap_or(video.lamp_fog).min(2);
    let enabled = (tier != 0 || lamp_tier != 0)
        && cameras.iter().any(|(_, camera, _, _, _)| camera.is_active);
    for (entity, sun) in &suns {
        // Only a tag for selecting this light in our shader. No Bevy VolumetricFog
        // camera/volume is created, and the rig's lux and surface lighting stay intact.
        if tier != 0 && enabled && sun.shadows_enabled {
            commands.entity(entity).insert(VolumetricLight);
        } else {
            commands.entity(entity).remove::<VolumetricLight>();
        }
    }
    for (entity, camera, mut camera3d, camera_transform, frustum) in &mut cameras {
        if !enabled || !camera.is_active {
            commands.entity(entity).remove::<FogView>();
            continue;
        }
        camera3d.depth_texture_usages.0 |= TextureUsages::TEXTURE_BINDING.bits();
        let sun = lighting.celestial_dir();
        let daylight = ((sun.y + 0.02) / 0.15).clamp(0.0, 1.0);
        let weather_amount = weather
            .sky_density
            .max(weather.effect_density)
            .clamp(0.0, 1.0);
        // Exact zero by day is both the intended look and the A/B invariant. At night a constant
        // floor makes nearby halos survive the zone haze's ten-yard dead zone; rain/fog raises it.
        let lamp_strength = if lamp_tier == 0 {
            0.0
        } else {
            lamp_haze_strength(sun.y, weather_amount, interior.0.is_some())
        };
        let lamps = if lamp_strength > 0.0 {
            select_fog_lamps(
                point_lights.as_slice(),
                camera_transform.map_or(Vec3::ZERO, GlobalTransform::translation),
                frustum,
                if lamp_tier == 1 {
                    LOW_FOG_LAMPS
                } else {
                    MAX_FOG_LAMPS
                },
            )
        } else {
            FogLamps::default()
        };
        // MONKEY (reviewfix-a): no haze tier and no lamp in range is an identity pass — skip the
        // node (and its depth bind and ping-pong) rather than run it every frame.
        if tier == 0 && lamps.count == 0 {
            commands.entity(entity).remove::<FogView>();
            continue;
        }
        let fog = Vec3::from_array(lighting.fog_color);
        // MONKEY (visualfix): 1 when the eye is above an ocean in reach. Water writes no depth, so
        // without this the haze measured the SEABED behind it, and water over no seabed read as
        // sky and took none: the underwater slope of an island drew a hard wedge across the sea.
        let sea = camera_transform.map_or(0.0, |t| {
            let eye = benilla_assets::coords::bevy_to_wow(t.translation());
            let over = eye[2] > SEA_LEVEL_MIN_EYE
                && water
                    .iter()
                    .any(|w| w.ocean_within(eye[0], eye[1], SEA_PLANE_REACH));
            if over && tier != 0 {
                1.0
            } else {
                0.0
            }
        });
        // MONKEY (fog): rows 1-3 as packed for the light buffer.
        let rows = monkey.as_ref().map_or(
            [[0.0; 4]; benilla_world::lighting::MONKEY_FRAME_ROWS],
            |m| m.pack(0.0, 0.0),
        );
        commands.entity(entity).insert(FogView {
            mf_fog1: Vec4::from_array(rows[1]),
            mf_fog2: Vec4::from_array(rows[2]),
            mf_fog3: Vec4::from_array(rows[3]),
            lamp_meta: Vec4::new(lamps.count as f32, lamp_strength, 0.0, sea),
            lamp_positions: lamps.positions,
            lamp_colors: lamps.colors,
            color_density: fog.extend(
                density(
                    crate::post::grading::rendered_minute(&clock, rendered.as_deref()),
                    weather_amount,
                    interior.0.is_some(),
                ) * if tier == 0 {
                    0.0
                } else if tier == 2 {
                    1.6
                } else {
                    1.0
                },
            ),
            sun_strength: Vec3::from_array(lighting.diffuse)
                .lerp(Vec3::ONE, 0.5)
                .extend(if tier == 0 || shafts_retired {
                    0.0
                } else {
                    override_value.shaft_gain * daylight
                }),
            direction_steps: sun.extend(if tier == 1 { 24.0 } else { 64.0 }),
        });
    }
}

#[derive(Resource)]
struct FogShader(Handle<Shader>);

/// MONKEY (reviewfix-a): the view's fog uniform, created once and rewritten in prepare.
#[derive(Component)]
struct FogUniform(UniformBuffer<FogView>);

fn prepare_fog_uniforms(
    mut commands: Commands,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut views: Query<(Entity, &FogView, Option<&mut FogUniform>)>,
) {
    for (entity, fog, uniform) in &mut views {
        match uniform {
            Some(mut uniform) => {
                uniform.0.set(*fog);
                uniform.0.write_buffer(&device, &queue);
            }
            None => {
                let mut uniform = UniformBuffer::from(*fog);
                uniform.write_buffer(&device, &queue);
                commands.entity(entity).insert(FogUniform(uniform));
            }
        }
    }
}
#[derive(Resource)]
struct FogPipeline {
    layouts: [BindGroupLayoutDescriptor; 2],
    shader: Handle<Shader>,
    fullscreen: FullscreenShader,
    comparison: Sampler,
}
#[derive(Component)]
struct ViewFogPipeline(CachedRenderPipelineId);
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub(crate) struct FogLabel;

fn init_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    shader: Res<FogShader>,
    fullscreen: Res<FullscreenShader>,
) {
    let layouts = [false, true].map(|multisampled| {
        BindGroupLayoutDescriptor::new(
            "gamma_fog_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    if multisampled {
                        texture_depth_2d_multisampled()
                    } else {
                        texture_depth_2d()
                    },
                    uniform_buffer::<ViewUniform>(true),
                    uniform_buffer::<GpuLights>(true),
                    texture_2d_array(TextureSampleType::Depth),
                    sampler(SamplerBindingType::Comparison),
                    uniform_buffer::<FogView>(false),
                ),
            ),
        )
    });
    commands.insert_resource(FogPipeline {
        layouts,
        shader: shader.0.clone(),
        fullscreen: fullscreen.clone(),
        comparison: device.create_sampler(&SamplerDescriptor {
            compare: Some(CompareFunction::GreaterEqual),
            min_filter: FilterMode::Linear,
            mag_filter: FilterMode::Linear,
            ..default()
        }),
    });
}
impl SpecializedRenderPipeline for FogPipeline {
    type Key = (TextureFormat, bool);
    fn specialize(&self, (format, multisampled): Self::Key) -> RenderPipelineDescriptor {
        let mut defs = vec![
            ShaderDefVal::UInt(
                "MAX_DIRECTIONAL_LIGHTS".into(),
                MAX_DIRECTIONAL_LIGHTS as u32,
            ),
            ShaderDefVal::UInt(
                "MAX_CASCADES_PER_LIGHT".into(),
                MAX_CASCADES_PER_LIGHT as u32,
            ),
            ShaderDefVal::UInt("AVAILABLE_STORAGE_BUFFER_BINDINGS".into(), 0),
        ];
        if multisampled {
            defs.push("MULTISAMPLED".into());
        }
        RenderPipelineDescriptor {
            label: Some("gamma_volumetric_fog".into()),
            layout: vec![self.layouts[multisampled as usize].clone()],
            vertex: self.fullscreen.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: defs,
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
    pipeline: Res<FogPipeline>,
    mut specialized: ResMut<SpecializedRenderPipelines<FogPipeline>>,
    views: Query<(Entity, &ViewTarget, &Msaa), With<FogView>>,
    all_views: Query<(&ViewTarget, &Msaa), With<Camera3d>>,
) {
    // MONKEY (integration): warm every reachable key on every 3-D view, feature on or off, so the
    // compile happens under the entry cover and never live when the player turns the row on.
    // (Named in `pipe_warm/menagerie.rs`'s custom-lane census.)
    for (target, msaa) in &all_views {
        specialized.specialize(
            &cache,
            &pipeline,
            (target.main_texture_format(), msaa.samples() > 1),
        );
    }
    for (entity, target, msaa) in &views {
        let id = specialized.specialize(
            &cache,
            &pipeline,
            (target.main_texture_format(), msaa.samples() > 1),
        );
        commands.entity(entity).insert(ViewFogPipeline(id));
    }
}
#[derive(Default)]
struct FogNode;
impl ViewNode for FogNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ViewDepthTexture,
        &'static ViewUniformOffset,
        &'static ViewLightsUniformOffset,
        &'static ViewShadowBindings,
        &'static FogUniform,
        &'static ViewFogPipeline,
    );
    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        context: &mut RenderContext<'w>,
        (target, depth, view_offset, light_offset, shadows, fog, id): QueryItem<
            'w,
            '_,
            Self::ViewQuery,
        >,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let cache = world.resource::<PipelineCache>();
        let Some(pipeline) = cache.get_render_pipeline(id.0) else {
            return Ok(());
        };
        if !depth
            .texture
            .usage()
            .contains(TextureUsages::TEXTURE_BINDING)
        {
            return Ok(());
        }
        let (Some(view), Some(lights)) = (
            world.resource::<ViewUniforms>().uniforms.binding(),
            world.resource::<LightMeta>().view_gpu_lights.binding(),
        ) else {
            return Ok(());
        };
        let settings = world.resource::<FogPipeline>();
        let device = context.render_device();
        let out = target.post_process_write();
        let bind = device.create_bind_group(
            "gamma_fog",
            &cache.get_bind_group_layout(
                &settings.layouts[(depth.texture.sample_count() > 1) as usize],
            ),
            &BindGroupEntries::sequential((
                out.source,
                depth.view(),
                view,
                lights,
                &shadows.directional_light_depth_texture_view,
                &settings.comparison,
                fog.0.binding().unwrap(),
            )),
        );
        let diagnostics = context.diagnostic_recorder();
        let mut pass = context
            .command_encoder()
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("gamma_volumetric_fog"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: out.destination,
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
        let span = diagnostics.pass_span(&mut pass, "gamma_volumetric_fog");
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[view_offset.offset, light_offset.offset]);
        pass.draw(0..3, 0..1);
        span.end(&mut pass);
        Ok(())
    }
}

const FOG_SHADER: &str = r#"
#import bevy_render::view::View
#import bevy_pbr::mesh_view_types::Lights
#import benilla::fog_hook
@group(0) @binding(0) var scene: texture_2d<f32>;
#ifdef MULTISAMPLED
@group(0) @binding(1) var depth: texture_depth_multisampled_2d;
#else
@group(0) @binding(1) var depth: texture_depth_2d;
#endif
@group(0) @binding(2) var<uniform> view: View;
@group(0) @binding(3) var<uniform> lights: Lights;
@group(0) @binding(4) var shadows: texture_depth_2d_array;
@group(0) @binding(5) var shadow_sampler: sampler_comparison;
struct Fog {
    color_density: vec4<f32>, sun_strength: vec4<f32>, direction_steps: vec4<f32>,
    // MONKEY (fog): MonkeyFrame fog1..fog3.
    mf_fog1: vec4<f32>, mf_fog2: vec4<f32>, mf_fog3: vec4<f32>,
    // MONKEY (lampfog): x=count, y=night/weather strength; packed point-table rows follow.
    lamp_meta: vec4<f32>,
    lamp_positions: array<vec4<f32>, 32>,
    lamp_colors: array<vec4<f32>, 32>,
}
@group(0) @binding(6) var<uniform> fog: Fog;

// Integral of smoothstep(10, 30, distance): zero in the first ten yards,
// continuous density and derivative at both ends, bounded at 150 yd.
fn fog_path(distance: f32) -> f32 {
    let d = clamp(distance, 10.0, 150.0);
    let u = clamp((d - 10.0) / 20.0, 0.0, 1.0);
    return 20.0 * (u*u*u - 0.5*u*u*u*u) + max(d - 30.0, 0.0);
}
fn visibility(light_index: u32, p: vec3<f32>) -> f32 {
    // Borrow uniform records instead of copying all cascade matrices per sample.
    let light = &lights.directional_lights[light_index];
    let view_distance = -(view.view_from_world * vec4(p, 1.0)).z;
    for (var cascade = 0u; cascade < (*light).num_cascades; cascade += 1u) {
        let shadow_cascade = &(*light).cascades[cascade];
        if (view_distance > (*shadow_cascade).far_bound) { continue; }
        let q = (*shadow_cascade).clip_from_world * vec4(p + (*light).direction_to_light * (*light).shadow_depth_bias, 1.0);
        let ndc = q.xyz / q.w;
        let uv = ndc.xy * vec2(0.5, -0.5) + 0.5;
        if (any(uv < vec2(0.0)) || any(uv > vec2(1.0)) || ndc.z < 0.0 || ndc.z > 1.0) { continue; }
        return textureSampleCompareLevel(shadows, shadow_sampler, uv,
            i32((*light).depth_texture_base_index + cascade), ndc.z);
    }
    // Unknown outside caster coverage is not evidence of direct sunlight.
    // MONKEY (visualfix): -1 marks "unknown"; the march fills it from the ray's covered samples.
    return -1.0;
}

// MONKEY (lampfog): a mild Henyey-Greenstein lobe. The inverse-square term is integrated exactly;
// evaluating this slow angular term at the closest admitted point keeps the closed form intact.
fn lamp_phase(cosine: f32) -> f32 {
    let anisotropy = 0.2;
    let denom = 1.0 + anisotropy * anisotropy - 2.0 * anisotropy * cosine;
    return 0.07957747 * (1.0 - anisotropy * anisotropy) / (denom * sqrt(denom));
}

// MONKEY (lampfog): analytic line integral of a softened inverse-square source over the visible
// camera-to-depth segment. It is intentionally UNSHADOWED; scene depth cuts the segment, but no
// point-light cube map is sampled. Range is a smooth screen-space gate around the exact sphere
// intersection. The exponential soft cap plus destination headroom keeps halos below 1.0 so a
// later bloom pass cannot turn a stacked row of lamps into white slabs.
fn lamp_halos(ray: vec3<f32>, distance: f32) -> vec3<f32> {
    var energy = vec3<f32>(0.0);
    let lamp_count = u32(fog.lamp_meta.x);
    for (var lamp_index = 0u; lamp_index < 32u; lamp_index += 1u) {
        if (lamp_index >= lamp_count) { break; }
        let position_range = fog.lamp_positions[lamp_index];
        let to_lamp = position_range.xyz - view.world_position;
        let projection = dot(to_lamp, ray);
        let perpendicular = to_lamp - ray * projection;
        let perpendicular_sq = dot(perpendicular, perpendicular);
        let range_sq = position_range.w * position_range.w;
        if (perpendicular_sq >= range_sq) { continue; }

        let half_span = sqrt(max(range_sq - perpendicular_sq, 0.0));
        let segment_start = max(0.0, projection - half_span);
        let segment_end = min(distance, projection + half_span);
        if (segment_end <= segment_start) { continue; }

        // Integral dt / (core^2 + perpendicular^2 + (t - projection)^2).
        let core_radius = 1.25;
        let height = sqrt(perpendicular_sq + core_radius * core_radius);
        let integral = (atan((segment_end - projection) / height)
            - atan((segment_start - projection) / height)) / height;
        let closest = clamp(projection, segment_start, segment_end);
        let sample_to_lamp = view.world_position + ray * closest - position_range.xyz;
        let light_path = sample_to_lamp / max(length(sample_to_lamp), 0.0001);
        let phase = lamp_phase(dot(light_path, -ray));
        let edge = 1.0 - smoothstep(position_range.w * 0.72, position_range.w,
            sqrt(perpendicular_sq));
        energy += max(fog.lamp_colors[lamp_index].rgb, vec3<f32>(0.0))
            * (integral * phase * edge);
    }
    let capped = min(energy * fog.lamp_meta.y * 0.9, vec3<f32>(0.75));
    return vec3<f32>(1.0) - exp(-capped);
}

@fragment
fn fragment(@builtin(position) pixel: vec4<f32>) -> @location(0) vec4<f32> {
    let xy = vec2<i32>(pixel.xy);
    let source = textureLoad(scene, xy, 0);
    let z = textureLoad(depth, xy, 0);
    let uv = (pixel.xy - view.viewport.xy) / view.viewport.zw;
    let q = view.world_from_clip * vec4(uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), max(z, 0.000001), 1.0);
    var delta = q.xyz / q.w - view.world_position;
    // MONKEY (visualfix): over an ocean, a point below sea level (seabed, or no depth at all under
    // the horizon) is hazed at the ocean surface (Bevy y = 0), where the eye actually meets it.
    var sea_surface = false;
    if (fog.lamp_meta.w > 0.5 && view.world_position.y > 0.0) {
        let down = normalize(delta);
        let below = (view.world_position + delta).y < 0.0;
        if (down.y < -0.0001 && (below || z <= 0.0)) {
            let to_sea = view.world_position.y / -down.y;
            if (to_sea < length(delta)) {
                delta = down * to_sea;
                sea_surface = true;
            }
        }
    }
    let distance = min(length(delta), 150.0);
    if (distance <= 10.0 && fog.lamp_meta.x < 0.5) { return source; }
    let ray = normalize(delta);
    let luma = vec3(0.2126, 0.7152, 0.0722);
    var haze = source.rgb;
    if (distance > 10.0) {
        let optical_depth = fog.color_density.w * fog_path(distance);
        var opacity = 1.0 - exp(-optical_depth);
        var haze_rgb = fog.color_density.rgb;
        if (fog_hook::fog_is_modern(fog.mf_fog3)) {
            // MONKEY (fog): the sky (depth 0, clamped to 150 yd above) already carries its own
            // horizon fog, so it takes no distance haze; the haze colour is the shared fog colour.
            if (z <= 0.0 && !sea_surface) { opacity = 0.0; }
            haze_rgb = fog_hook::fog_modern_colour(fog.color_density.rgb, ray, length(delta),
                fog.mf_fog1, fog.mf_fog2, fog.mf_fog3);
        }
        haze = mix(source.rgb, haze_rgb, opacity);
        // A darker zone colour may change hue, never the pixel's luminance. Add
        // neutral headroom proportionally so saturated channels cannot clip dark.
        let missing = max(0.0, dot(source.rgb - haze, luma));
        let headroom = max(vec3(0.0), vec3(1.0) - haze);
        haze += headroom * clamp(missing / max(dot(headroom, luma), 0.00001), 0.0, 1.0);
        if (fog.sun_strength.w > 0.0) {
            let g = 0.65;
            let cosine = dot(ray, fog.direction_steps.xyz);
            let denom = 1.0 + g*g - 2.0*g*cosine;
            let phase = 0.07957747 * (1.0-g*g) / (denom * sqrt(denom));
            let count = u32(fog.direction_steps.w);
            let step_size = (distance - 10.0) / f32(count);
            var scatter = 0.0;
            for (var light_index = 0u; light_index < lights.n_directional_lights; light_index += 1u) {
                if ((lights.directional_lights[light_index].flags & 3u) != 3u) { continue; }
                // MONKEY (visualfix): a sample past the cascade (beyond shadowDistance) takes the
                // mean of this ray's covered samples, not 0: a hard 0 cut the sun lobe into rings
                // of the far-bound step. A ray that starts in a dark room stays dark.
                var known = 0.0;
                var known_count = 0.0;
                for (var step = 0u; step < count; step += 1u) {
                    // Fixed midpoint samples: no temporal noise in this gamma lane.
                    let t = 10.0 + (f32(step) + 0.5) * step_size;
                    let density = fog.color_density.w * smoothstep(10.0, 30.0, t);
                    var seen = visibility(light_index, view.world_position + ray*t);
                    if (seen >= 0.0) {
                        known += seen;
                        known_count += 1.0;
                    } else {
                        seen = select(0.0, known / known_count, known_count > 0.0);
                    }
                    scatter += seen * exp(-fog.color_density.w * fog_path(t)) * density * step_size;
                }
            }
            let shafts = fog.sun_strength.rgb * fog.sun_strength.w * phase * scatter;
            haze += max(vec3(0.0), vec3(1.0) - haze) * (vec3(1.0) - exp(-shafts));
        }
    }
    if (fog.lamp_meta.x > 0.5) {
        let halo = lamp_halos(ray, distance);
        haze += max(vec3(0.0), vec3(1.0) - haze) * halo;
    }
    return vec4(haze, source.a);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dawn_weather_and_rooms_control_density() {
        let noon = density(720.0, 0.0, false);
        assert!(density(390.0, 0.0, false) > noon * 2.0);
        assert_eq!(density(300.0, 0.0, false), noon);
        assert_eq!(density(480.0, 0.0, false), noon);
        assert!(density(720.0, 1.0, false) > noon * 2.0);
        assert!(density(390.0, 1.0, true) < noon);
    }
    #[test]
    fn lamp_haze_is_exactly_off_by_day_and_weather_only_strengthens_night() {
        assert_eq!(lamp_haze_strength(1.0, 1.0, false), 0.0);
        let clear = lamp_haze_strength(-1.0, 0.0, false);
        assert!(lamp_haze_strength(-1.0, 1.0, false) > clear * 2.0);
        assert!(lamp_haze_strength(-1.0, 0.0, true) < clear);
    }
    #[test]
    fn lamp_picker_keeps_the_nearest_across_both_lanes_and_tier_limit() {
        let mut points: Vec<_> = (1..=40)
            .rev()
            .map(|distance| ResolvedPointLight {
                position: Vec3::X * distance as f32,
                range: 48.0,
                color: Vec3::new(0.8, 0.4, 0.1),
                lane: 0.0,
            })
            .collect();
        // Interior and exterior table rows compete by distance; black rows never consume a slot.
        points.push(ResolvedPointLight {
            position: Vec3::X * 0.25,
            range: 48.0,
            color: Vec3::ONE,
            lane: 8.0,
        });
        points.push(ResolvedPointLight {
            position: Vec3::X * 0.5,
            range: 48.0,
            color: Vec3::ZERO,
            lane: 0.0,
        });
        let low = select_fog_lamps(&points, Vec3::ZERO, None, LOW_FOG_LAMPS);
        assert_eq!(low.count, LOW_FOG_LAMPS);
        assert_eq!(low.positions[0], Vec4::new(0.25, 0.0, 0.0, 8.0));
        assert_eq!(low.positions[1].x, 1.0);
        assert_eq!(low.positions[LOW_FOG_LAMPS - 1].x, 15.0);
        assert_eq!(low.colors[0].truncate(), Vec3::ONE);
        assert_eq!(low.colors[1].truncate(), Vec3::new(0.8, 0.4, 0.1));
    }
    #[test]
    fn live_tiers_shadow_loss_inactive_camera_and_off() {
        let mut app = App::new();
        app.init_resource::<VideoConfig>()
            .init_resource::<WowLighting>()
            .init_resource::<WorldTime>()
            .init_resource::<WeatherState>()
            .init_resource::<CameraInteriorClaim>()
            .init_resource::<ResolvedPointLights>()
            .insert_resource(FogOverride {
                fog: None,
                lamp: None,
                shaft_gain: 2.0,
            })
            .add_systems(Update, update_fog);
        let camera = app
            .world_mut()
            .spawn((WorldCamera, Camera::default(), Camera3d::default()))
            .id();
        let preview = app
            .world_mut()
            .spawn((Camera::default(), Camera3d::default()))
            .id();
        let sun = app
            .world_mut()
            .spawn((
                ShadowSun,
                DirectionalLight {
                    illuminance: 123.0,
                    shadows_enabled: true,
                    ..default()
                },
            ))
            .id();
        app.update();
        let low = *app.world().get::<FogView>(camera).unwrap();
        assert_eq!(low.direction_steps.w, 24.0);
        assert!(app.world().get::<FogView>(preview).is_none());
        assert!(app.world().get::<VolumetricLight>(sun).is_some());
        app.world_mut().resource_mut::<VideoConfig>().volumetric_fog = 2;
        app.world_mut()
            .get_mut::<DirectionalLight>(sun)
            .unwrap()
            .shadows_enabled = false;
        app.update();
        let high = *app.world().get::<FogView>(camera).unwrap();
        assert_eq!(high.direction_steps.w, 64.0);
        assert_eq!(high.color_density.w, low.color_density.w * 1.6);
        assert!(app.world().get::<VolumetricLight>(sun).is_none());
        assert_eq!(
            app.world()
                .get::<DirectionalLight>(sun)
                .unwrap()
                .illuminance,
            123.0
        );
        app.world_mut().get_mut::<Camera>(camera).unwrap().is_active = false;
        app.update();
        assert!(app.world().get::<FogView>(camera).is_none());
        app.world_mut().get_mut::<Camera>(camera).unwrap().is_active = true;
        app.world_mut().resource_mut::<FogOverride>().fog = Some(0);
        app.update();
        assert!(app.world().get::<FogView>(camera).is_none());
        app.world_mut().resource_mut::<FogOverride>().fog = None;
        app.update();
        assert!(app.world().get::<FogView>(camera).is_some());
    }
}
