//! GFX (volumetric light): shadow-mapped light shafts for the SUN and the MOON.
//!
//! The directional light that holds the realtime shadow map (`shadow_core`'s one rig, aimed at
//! the sun by day and at the moon at night) is ray-marched through the air in front of every
//! pixel: a few jittered samples along the view ray each ask the cascade "is this point of air
//! lit?", and the lit air in-scatters the light toward the eye with a forward-peaked phase. The
//! result is real light shafts (god rays) through tree canopies, between buildings and past
//! the player, including right in front of the camera, which the screen-space `sunShafts` lane
//! (sky pixels only) and the old midpoint march in `volumetric_fog` (10 yd dead zone, fixed
//! samples) could not show.
//!
//! Passes, all after the world's transparencies and the gamma fog pass (so this light sits on
//! top of the haze it is scattering through):
//! 1. **march**, half resolution: `steps` samples per ray, quadratic spacing (dense near the eye,
//!    where the shafts are), a per-pixel interleaved-gradient-noise start offset. Writes the
//!    in-scatter scalar and the marched distance.
//! 2. **blur**, half resolution: a depth-aware 4x4 gather that averages the jitter pattern out
//!    and never bleeds shafts across a depth edge.
//! 3. **composite**, full resolution: a depth-aware 2x2 upsample, coloured by the light (sun
//!    colour, or the cool moon tint and intensity from `benilla_world::lighting::moonlight`) and
//!    tinted toward the zone fog colour, added with the same soft screen curve the fog pass uses.
//!
//! Fog integration: the extinction per yard is the fog lane's own density (dawn mist, weather)
//! scaled by how short the zone's authored fog end is; the in-scatter colour leans toward the
//! zone fog colour. Interiors (the camera in a WMO room) take no shafts. The Graphics Preset turns
//! it off on Classic/Low, Medium on Medium, High on High/Ultra (`volumetricLight`), with a
//! `volumetricLightStrength` multiplier.
//!
//! No temporal accumulation: the jitter is fixed per pixel and removed by the blur, so a still
//! frame is stable and the pass has no history to invalidate on camera cuts.
use crate::{
    shadow_core::{ShadowSet, ShadowSun},
    video::VideoConfig,
    volumetric_fog::FogLabel,
};
use benilla_world::{
    lighting::{moon_light_intensity, ShadowHandover, WorldTime, WowLighting},
    view::WorldCamera,
    weather::WeatherState,
    wmo_portal::CameraInteriorClaim,
};
use bevy::{
    core_pipeline::{
        core_3d::graph::{Core3d, Node3d},
        FullscreenShader,
    },
    ecs::query::QueryItem,
    pbr::{
        GpuLights, LightMeta, ViewLightsUniformOffset, ViewShadowBindings, MAX_CASCADES_PER_LIGHT,
        MAX_DIRECTIONAL_LIGHTS,
    },
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

pub(crate) struct VolumetricLightPlugin;

/// Session-only capture/test override of the tier (`WOW_VOLLIGHT=0|1|2`); the persistent setting
/// is `volumetricLight`. Read by `volumetric_fog` too, which retires its own midpoint shafts
/// whenever this lane is live.
#[derive(Resource, Clone, Copy, Default)]
pub(crate) struct VolLightOverride(pub(crate) Option<u8>);

impl VolLightOverride {
    /// The live tier: the session override, else the setting.
    pub(crate) fn tier(&self, video: &VideoConfig) -> u8 {
        self.0.unwrap_or(video.volumetric_light).min(2)
    }
}

/// March samples per tier (index = tier).
const STEPS: [f32; 3] = [0.0, 16.0, 32.0];
/// The march starts this far from the eye (inside it the near plane clips geometry anyway).
const MARCH_START: f32 = 0.35;
/// Upper bound on the march length (yd); the cascade's own reach (`shadowDistance`) is lower.
const MARCH_MAX: f32 = 160.0;
/// Henyey-Greenstein anisotropy: forward-peaked, like air with a little haze.
const ANISOTROPY: f32 = 0.6;
/// The sun's shaft gain at `volumetricLightStrength 1` (gamma units per unit in-scatter).
const SUN_SHAFT_GAIN: f32 = 9.0;
/// The moon's shaft gain, applied on top of its (already dim) moonlight intensity: faint.
const MOON_SHAFT_GAIN: f32 = 30.0;
/// Cool moon tint; the same constant `benilla::moonlight_hook` lights surfaces with.
const MOON_TINT: Vec3 = Vec3::new(0.62, 0.74, 1.0);
/// How much the in-scatter colour leans toward the (luminance-normalised) zone fog colour.
const FOG_TINT: f32 = 0.35;
/// Reference fog end (yd): a zone authored foggier than this scatters more, clearer less.
const FOG_END_REFERENCE: f32 = 350.0;
/// Scattering air thins with height above the eye (e-folding yards): shafts live in the air
/// around and below the viewer (under canopies, between buildings), not in the open sky above.
const HEIGHT_FALLOFF: f32 = 10.0;

/// Dev/capture tuning levers (`WOW_VOLLIGHT_DENSITY`, `_HEIGHT`, `_GAIN`, `_DEBUG`), honoured only
/// in a dev capture, the same door `volumetric_fog`'s `WOW_VOLFOG_SHAFT_GAIN` uses.
#[derive(Resource, Clone, Copy)]
struct VolLightTune {
    density: f32,
    height: f32,
    gain: f32,
    debug: f32,
}

impl VolLightTune {
    fn from_env() -> Self {
        let dev = crate::run_mode::dev_affordances() && std::env::var_os("WOW_CAPTURE").is_some();
        let knob = |k: &str, d: f32| {
            if !dev {
                return d;
            }
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .filter(|v| v.is_finite())
                .unwrap_or(d)
        };
        Self {
            density: knob("WOW_VOLLIGHT_DENSITY", 1.0).clamp(0.0, 50.0),
            height: knob("WOW_VOLLIGHT_HEIGHT", HEIGHT_FALLOFF).clamp(1.0, 10000.0),
            gain: knob("WOW_VOLLIGHT_GAIN", 1.0).clamp(0.0, 50.0),
            debug: knob("WOW_VOLLIGHT_DEBUG", 0.0),
        }
    }
}

#[derive(Component, Clone, Copy, PartialEq, ExtractComponent, ShaderType)]
struct VolLightView {
    /// xyz = unit direction toward the light body (sun or moon, Bevy space); w = march steps.
    dir_steps: Vec4,
    /// rgb = in-scatter colour (gain, strength and hand-over fade folded in); w = extinction/yd.
    colour: Vec4,
    /// x = march length cap (yd), y = march start (yd), z = anisotropy g, w = height falloff (yd).
    range: Vec4,
    /// x = debug view (1 = the in-scatter alone, as grey), yzw unused.
    debug: Vec4,
}

impl Plugin for VolumetricLightPlugin {
    fn build(&self, app: &mut App) {
        let tier = std::env::var("WOW_VOLLIGHT")
            .ok()
            .and_then(|v| v.parse::<u8>().ok())
            .filter(|v| *v <= 2);
        app.insert_resource(VolLightOverride(tier))
            .insert_resource(VolLightTune::from_env())
            .add_plugins(ExtractComponentPlugin::<VolLightView>::default())
            .add_systems(Last, update_views.after(ShadowSet::Lanes));
        if !app.is_plugin_added::<AssetPlugin>() {
            return;
        }
        let shader = app
            .world_mut()
            .resource_mut::<Assets<Shader>>()
            .add(Shader::from_wgsl(
                include_str!("volumetric_light.wgsl"),
                "volumetric_light.wgsl",
            ));
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .insert_resource(VolLightShader(shader))
            .init_resource::<SpecializedRenderPipelines<VolLightPipeline>>()
            .add_systems(RenderStartup, init_pipeline)
            .add_systems(
                Render,
                (
                    prepare_pipelines.in_set(RenderSystems::Prepare),
                    prepare_resources.in_set(RenderSystems::PrepareResources),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<VolLightNode>>(Core3d, VolLightLabel)
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::EndMainPass,
                    FogLabel,
                    VolLightLabel,
                    Node3d::StartMainPassPostProcessing,
                ),
            );
    }
}

/// The in-scatter colour and fade of the body the shadow rig holds this frame, or `None` when
/// no body casts (daylight hand-over window, `moonShadowStrength 0` / `moonLight 0` at night).
fn light_body(
    lighting: &WowLighting,
    handover: &ShadowHandover,
    moon_gain: f32,
) -> Option<(Vec3, Vec3)> {
    if handover.weight > 0.0 {
        // The sun: its lane weight already carries the elevation ramp and the hand-over ramp.
        let dir = lighting.celestial_dir().normalize_or_zero();
        let colour = Vec3::from_array(lighting.diffuse).lerp(Vec3::ONE, 0.5)
            * (SUN_SHAFT_GAIN * handover.weight.clamp(0.0, 1.0));
        return (dir != Vec3::ZERO).then_some((dir, colour));
    }
    let ramp = handover.moon_ramp();
    if ramp <= 0.0 {
        return None;
    }
    let dir = lighting.moon_dir().normalize_or_zero();
    let intensity = moon_light_intensity(
        moon_gain,
        lighting.celestial_dir().y,
        dir.y,
        lighting.storm(),
    );
    (intensity > 0.0 && dir != Vec3::ZERO)
        .then(|| (dir, MOON_TINT * (MOON_SHAFT_GAIN * intensity * ramp)))
}

/// Extinction per yard: the fog lane's density law, scaled by the zone's authored fog end.
fn extinction(minute: f32, weather: f32, fog_end: f32) -> f32 {
    let zone = if fog_end > 1.0 {
        (FOG_END_REFERENCE / fog_end).clamp(0.7, 2.5)
    } else {
        1.0
    };
    crate::volumetric_fog::density(minute, weather, false) * zone
}

/// The in-scatter colour leaned toward the zone fog's hue (luminance kept).
fn fog_tinted(colour: Vec3, fog: Vec3) -> Vec3 {
    let luma = |c: Vec3| c.dot(Vec3::new(0.2126, 0.7152, 0.0722));
    let l = luma(fog);
    if l <= 1e-4 {
        return colour;
    }
    colour * Vec3::ONE.lerp(fog / l, FOG_TINT)
}

fn update_views(
    mut commands: Commands,
    video: Res<VideoConfig>,
    override_tier: Res<VolLightOverride>,
    tune: Option<Res<VolLightTune>>,
    lighting: Res<WowLighting>,
    handover: Res<ShadowHandover>,
    clock: Res<WorldTime>,
    weather: Res<WeatherState>,
    interior: Res<CameraInteriorClaim>,
    rendered: Option<Res<benilla_world::lighting::GameClock>>,
    suns: Query<&DirectionalLight, With<ShadowSun>>,
    mut cameras: Query<(Entity, &Camera, &mut Camera3d, Option<&VolLightView>), With<WorldCamera>>,
) {
    let tier = override_tier.tier(&video);
    let tune = tune.map_or(
        VolLightTune {
            density: 1.0,
            height: HEIGHT_FALLOFF,
            gain: 1.0,
            debug: 0.0,
        },
        |t| *t,
    );
    let strength = video.volumetric_light_strength.clamp(0.0, 2.0);
    let rig_live = suns.iter().any(|sun| sun.shadows_enabled);
    let body = (tier > 0 && strength > 0.0 && rig_live && interior.0.is_none())
        .then(|| light_body(&lighting, &handover, video.moon_light))
        .flatten();
    for (entity, camera, mut camera3d, old) in &mut cameras {
        let Some((dir, colour)) = body.filter(|_| camera.is_active) else {
            if old.is_some() {
                commands.entity(entity).remove::<VolLightView>();
            }
            continue;
        };
        camera3d.depth_texture_usages.0 |= TextureUsages::TEXTURE_BINDING.bits();
        let weather_amount = weather
            .sky_density
            .max(weather.effect_density)
            .clamp(0.0, 1.0);
        let sigma = extinction(
            crate::post::grading::rendered_minute(&clock, rendered.as_deref()),
            weather_amount,
            lighting.fog_end(),
        ) * tune.density;
        let colour =
            fog_tinted(colour, Vec3::from_array(lighting.fog_color)) * strength * tune.gain;
        let next = VolLightView {
            dir_steps: dir.extend(STEPS[tier as usize]),
            colour: colour.extend(sigma),
            range: Vec4::new(
                video.shadow_distance.min(MARCH_MAX),
                MARCH_START,
                ANISOTROPY,
                tune.height,
            ),
            debug: Vec4::new(tune.debug, 0.0, 0.0, 0.0),
        };
        if old != Some(&next) {
            commands.entity(entity).insert(next);
        }
    }
}

#[derive(Resource)]
struct VolLightShader(Handle<Shader>);

/// Half-resolution march / blur target: `.r` in-scatter, `.g` marched distance (yd).
const HALF_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

#[derive(Resource)]
struct VolLightPipeline {
    /// `[single, multisampled]` depth: the march's and the composite's layouts.
    march_layouts: [BindGroupLayoutDescriptor; 2],
    blur_layout: BindGroupLayoutDescriptor,
    composite_layouts: [BindGroupLayoutDescriptor; 2],
    shader: Handle<Shader>,
    fullscreen: FullscreenShader,
    comparison: Sampler,
}

/// Which of the three stages, plus the keys each is specialised on.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum VolLightKey {
    March {
        multisampled: bool,
    },
    Blur,
    Composite {
        format: TextureFormat,
        multisampled: bool,
    },
}

#[derive(Component)]
struct ViewVolLightPipelines {
    march: CachedRenderPipelineId,
    blur: CachedRenderPipelineId,
    composite: CachedRenderPipelineId,
}

#[derive(Component)]
struct VolLightTextures {
    march: CachedTexture,
    blur: CachedTexture,
}

#[derive(Component)]
struct VolLightUniform(UniformBuffer<VolLightView>);

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct VolLightLabel;

fn depth_entry(multisampled: bool) -> BindGroupLayoutEntryBuilder {
    if multisampled {
        texture_depth_2d_multisampled()
    } else {
        texture_depth_2d()
    }
}

fn init_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    shader: Res<VolLightShader>,
    fullscreen: Res<FullscreenShader>,
) {
    let march_layouts = [false, true].map(|multisampled| {
        BindGroupLayoutDescriptor::new(
            "vol_light_march_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    depth_entry(multisampled),
                    uniform_buffer::<ViewUniform>(true),
                    uniform_buffer::<GpuLights>(true),
                    texture_2d_array(TextureSampleType::Depth),
                    sampler(SamplerBindingType::Comparison),
                    uniform_buffer::<VolLightView>(false),
                ),
            ),
        )
    });
    let blur_layout = BindGroupLayoutDescriptor::new(
        "vol_light_blur_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (texture_2d(TextureSampleType::Float { filterable: false }),),
        ),
    );
    let composite_layouts = [false, true].map(|multisampled| {
        BindGroupLayoutDescriptor::new(
            "vol_light_composite_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    depth_entry(multisampled),
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    uniform_buffer::<ViewUniform>(true),
                    uniform_buffer::<VolLightView>(false),
                ),
            ),
        )
    });
    commands.insert_resource(VolLightPipeline {
        march_layouts,
        blur_layout,
        composite_layouts,
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

impl SpecializedRenderPipeline for VolLightPipeline {
    type Key = VolLightKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut defs: Vec<ShaderDefVal> = vec![
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
        let (label, layout, entry, format) = match key {
            VolLightKey::March { multisampled } => {
                defs.push("VL_MARCH".into());
                if multisampled {
                    defs.push("MULTISAMPLED".into());
                }
                (
                    "vol_light_march",
                    self.march_layouts[multisampled as usize].clone(),
                    "fs_march",
                    HALF_FORMAT,
                )
            }
            VolLightKey::Blur => {
                defs.push("VL_BLUR".into());
                (
                    "vol_light_blur",
                    self.blur_layout.clone(),
                    "fs_blur",
                    HALF_FORMAT,
                )
            }
            VolLightKey::Composite {
                format,
                multisampled,
            } => {
                defs.push("VL_COMPOSITE".into());
                if multisampled {
                    defs.push("MULTISAMPLED".into());
                }
                (
                    "vol_light_composite",
                    self.composite_layouts[multisampled as usize].clone(),
                    "fs_composite",
                    format,
                )
            }
        };
        RenderPipelineDescriptor {
            label: Some(label.into()),
            layout: vec![layout],
            vertex: self.fullscreen.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: defs,
                entry_point: Some(entry.into()),
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
    pipeline: Res<VolLightPipeline>,
    mut specialized: ResMut<SpecializedRenderPipelines<VolLightPipeline>>,
    views: Query<(Entity, &ViewTarget, &Msaa), With<VolLightView>>,
    all_views: Query<(&ViewTarget, &Msaa), With<Camera3d>>,
) {
    // Warm every reachable key on every 3-D view, feature on or off (the `pipe_warm` census
    // contract): enabling the row later hits the cache, never a live compile.
    let mut keys = |format: TextureFormat, multisampled: bool| ViewVolLightPipelines {
        march: specialized.specialize(&cache, &pipeline, VolLightKey::March { multisampled }),
        blur: specialized.specialize(&cache, &pipeline, VolLightKey::Blur),
        composite: specialized.specialize(
            &cache,
            &pipeline,
            VolLightKey::Composite {
                format,
                multisampled,
            },
        ),
    };
    for (target, msaa) in &all_views {
        keys(target.main_texture_format(), msaa.samples() > 1);
    }
    for (entity, target, msaa) in &views {
        let ids = keys(target.main_texture_format(), msaa.samples() > 1);
        commands.entity(entity).insert(ids);
    }
}

/// The two half-size targets (rounded up) and the view's uniform, rewritten in place.
fn prepare_resources(
    mut commands: Commands,
    mut textures: ResMut<TextureCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut views: Query<(
        Entity,
        &ExtractedCamera,
        &VolLightView,
        Option<&mut VolLightUniform>,
    )>,
) {
    for (entity, camera, view, uniform) in &mut views {
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        match uniform {
            Some(mut uniform) => {
                uniform.0.set(*view);
                uniform.0.write_buffer(&device, &queue);
            }
            None => {
                let mut uniform = UniformBuffer::from(*view);
                uniform.write_buffer(&device, &queue);
                commands.entity(entity).insert(VolLightUniform(uniform));
            }
        }
        let mut half = |label: &'static str| {
            textures.get(
                &device,
                TextureDescriptor {
                    label: Some(label),
                    size: Extent3d {
                        width: size.x.div_ceil(2).max(1),
                        height: size.y.div_ceil(2).max(1),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: HALF_FORMAT,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
            )
        };
        let march = half("vol_light_march");
        let blur = half("vol_light_blur");
        commands
            .entity(entity)
            .insert(VolLightTextures { march, blur });
    }
}

#[derive(Default)]
struct VolLightNode;

impl ViewNode for VolLightNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ViewDepthTexture,
        &'static ViewUniformOffset,
        &'static ViewLightsUniformOffset,
        &'static ViewShadowBindings,
        // Gates the pass: the uniform, textures and pipelines outlive a disabled frame.
        &'static VolLightView,
        &'static VolLightUniform,
        &'static VolLightTextures,
        &'static ViewVolLightPipelines,
    );

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        context: &mut RenderContext<'w>,
        (target, depth, view_offset, light_offset, shadows, _, uniform, half, ids): QueryItem<
            'w,
            '_,
            Self::ViewQuery,
        >,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let cache = world.resource::<PipelineCache>();
        let (Some(march), Some(blur), Some(composite)) = (
            cache.get_render_pipeline(ids.march),
            cache.get_render_pipeline(ids.blur),
            cache.get_render_pipeline(ids.composite),
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
        let (Some(view), Some(lights), Some(params)) = (
            world.resource::<ViewUniforms>().uniforms.binding(),
            world.resource::<LightMeta>().view_gpu_lights.binding(),
            uniform.0.binding(),
        ) else {
            return Ok(());
        };
        let settings = world.resource::<VolLightPipeline>();
        let multisampled = (depth.texture.sample_count() > 1) as usize;
        let device = context.render_device();
        let march_bind = device.create_bind_group(
            "vol_light_march",
            &cache.get_bind_group_layout(&settings.march_layouts[multisampled]),
            &BindGroupEntries::sequential((
                depth.view(),
                view.clone(),
                lights,
                &shadows.directional_light_depth_texture_view,
                &settings.comparison,
                params.clone(),
            )),
        );
        let blur_bind = device.create_bind_group(
            "vol_light_blur",
            &cache.get_bind_group_layout(&settings.blur_layout),
            &BindGroupEntries::sequential((&half.march.default_view,)),
        );
        let out = target.post_process_write();
        let composite_bind = device.create_bind_group(
            "vol_light_composite",
            &cache.get_bind_group_layout(&settings.composite_layouts[multisampled]),
            &BindGroupEntries::sequential((
                out.source,
                depth.view(),
                &half.blur.default_view,
                view,
                params,
            )),
        );
        let diagnostics = context.diagnostic_recorder();
        let stages: [(&str, &TextureView, &RenderPipeline, &BindGroup, &[u32]); 3] = [
            (
                "vol_light_march",
                &half.march.default_view,
                march,
                &march_bind,
                &[view_offset.offset, light_offset.offset],
            ),
            (
                "vol_light_blur",
                &half.blur.default_view,
                blur,
                &blur_bind,
                &[],
            ),
            (
                "vol_light_composite",
                out.destination,
                composite,
                &composite_bind,
                &[view_offset.offset],
            ),
        ];
        for (label, destination, pipeline, bind, offsets) in stages {
            let mut pass = context
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
            pass.set_bind_group(0, bind, offsets);
            pass.draw(0..3, 0..1);
            span.end(&mut pass);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn handover_sun(weight: f32) -> ShadowHandover {
        let mut h = ShadowHandover::default();
        h.weight = weight;
        h
    }

    #[test]
    fn the_sun_body_fades_with_its_lane_weight_and_none_casts_nothing() {
        let lighting = WowLighting::default();
        // The default lighting has no celestial direction: nothing to scatter.
        assert!(light_body(&lighting, &handover_sun(1.0), 1.0).is_none());
        assert!(light_body(&lighting, &handover_sun(0.0), 1.0).is_none());
    }

    #[test]
    fn foggier_zones_scatter_more_and_the_tint_keeps_luminance_order() {
        let clear = extinction(720.0, 0.0, 1400.0);
        let foggy = extinction(720.0, 0.0, 120.0);
        assert!(foggy > clear * 2.0);
        assert_eq!(
            extinction(720.0, 0.0, 0.0),
            crate::volumetric_fog::density(720.0, 0.0, false)
        );
        let white = fog_tinted(Vec3::ONE, Vec3::new(0.5, 0.5, 0.5));
        assert!(
            (white - Vec3::ONE).length() < 1e-5,
            "a grey fog does not tint"
        );
        let blue = fog_tinted(Vec3::ONE, Vec3::new(0.2, 0.3, 0.8));
        assert!(blue.z > blue.x, "a blue fog leans the shafts blue");
    }

    #[test]
    fn tiers_disable_classic_and_the_override_wins() {
        let mut video = VideoConfig::default();
        assert_eq!(
            VolLightOverride(None).tier(&video),
            0,
            "registered default is off"
        );
        video.volumetric_light = 2;
        assert_eq!(VolLightOverride(None).tier(&video), 2);
        assert_eq!(VolLightOverride(Some(1)).tier(&video), 1);
        assert_eq!(STEPS[0], 0.0);
        assert!(STEPS[2] > STEPS[1]);
    }
}
