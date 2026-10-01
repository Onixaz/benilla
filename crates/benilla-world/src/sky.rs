//! The sky dome, the gradient backdrop. The reference draws the sky first each frame as an unlit,
//! vertex-coloured dome without depth writes: a camera-centred cap of 1 apex and 5 rings × 24
//! segments, recentred by `−cos(45°)` so the rim sits at eye level (builder `0x6d0d10`, colours
//! `0x6d0f50`). The five `Light.dbc` `SkyColor` stops (LightIntBand rows 2 to 6) sit one per ring
//! at 90°, 16.8°, 9.8°, 3.7° and 1.8° of elevation, the horizon and below are the fog colour
//! (row 7), and colour is linear in elevation between rings.
//!
//! Here a world-aligned sphere around the camera interpolates the stops per fragment by view
//! elevation, at the far plane ([`crate::sky_order`]), so world geometry always paints over it.

use bevy::asset::RenderAssetUsages;
use bevy::camera::Projection;
use bevy::mesh::{Indices, MeshVertexBufferLayoutRef, PrimitiveTopology};
use bevy::pbr::{
    ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline,
    MaterialPlugin,
};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, Buffer, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use crate::dev_state::DebugState;
use crate::lighting::WowLighting;
use crate::view::WorldCamera;
use benilla_assets::AssetSet;

/// The unlit gradient sky; the `StandardMaterial` shell only supplies `unlit` and no culling.
pub type SkyMaterial = ExtendedMaterial<StandardMaterial, SkyExt>;

/// The five sky stops, zenith to horizon, and the fog colour on one binding (100); `sky.wgsl`'s
/// struct must match this field order.
#[derive(Asset, AsBindGroup, Clone, TypePath)]
#[bind_group_data(SkyFxKey)]
pub struct SkyExt {
    /// `SkyColor0`: zenith (90°).
    #[uniform(100)]
    pub(crate) sky0: Vec4,
    /// `SkyColor1`: 16.8°.
    #[uniform(100)]
    pub(crate) sky1: Vec4,
    /// `SkyColor2`: 9.8°.
    #[uniform(100)]
    pub(crate) sky2: Vec4,
    /// `SkyColor3`: 3.7°.
    #[uniform(100)]
    pub(crate) sky3: Vec4,
    /// `SkyColor4`: 1.8°, the last sky ring.
    #[uniform(100)]
    pub(crate) sky4: Vec4,
    /// Fog colour (LightIntBand row 7): the horizon (0°) and below.
    #[uniform(100)]
    pub(crate) fog: Vec4,
    /// Dawn/dusk azimuthal warp (`0x6d0f50`): `x` the strength `S` (`WowLighting.sky_warp`, 0 for
    /// none), `y` the sun azimuth `atan2(sun.z, sun.x)` in radians, `zw` reserved.
    #[uniform(100)]
    pub(crate) warp: Vec4,
    /// MONKEY (sky): `x` the [`crate::sky_fx::SkyQuality`] tier (0 = Classic, the fields below
    /// unused), `y` reserved (0; the clock is [`Self::clock`]), `z` the night-sky alpha, `w` the
    /// sun-glow strength.
    #[uniform(100)]
    pub(crate) fx: Vec4,
    /// MONKEY (sky): camera to the visible sun (`xyz`).
    #[uniform(100)]
    pub(crate) sun: Vec4,
    /// MONKEY (sky): the glow colour, gamma (`rgb`).
    #[uniform(100)]
    pub(crate) glow: Vec4,
    /// MONKEY (fog): MonkeyFrame fog rows B-D so the dome horizon uses the world's far colour.
    #[uniform(100)]
    pub(crate) mf_fog_b: Vec4,
    #[uniform(100)]
    pub(crate) mf_fog_c: Vec4,
    #[uniform(100)]
    pub(crate) mf_fog_d: Vec4,
    /// MONKEY (polish): the shared sky clock ([`crate::sky_fx::SkyClockBuffer`]), outside the
    /// uniform so the clock never rewrites the material.
    #[storage(101, read_only, buffer)]
    pub(crate) clock: Buffer,
}

/// MONKEY (sky): the pipeline key: Enhanced/High compile `SKY_FX` into `sky.wgsl`; Classic keeps
/// the reference's code with nothing added.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SkyFxKey {
    fx: bool,
}

impl From<&SkyExt> for SkyFxKey {
    fn from(e: &SkyExt) -> Self {
        Self { fx: e.fx.x >= 0.5 }
    }
}

impl MaterialExtension for SkyExt {
    /// The shared sky vertex stage, which pins depth to the far plane ([`crate::sky_order`]).
    fn vertex_shader() -> ShaderRef {
        crate::sky_order::SKY_VERTEX_SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        "embedded://benilla_world/shaders/sky.wgsl".into()
    }

    /// No depth write, as every sky element inherits from `CSky::Render` (`0x6d4940`); the depth
    /// test stays on. The glare quads are then occluded by world geometry only, never the dome.
    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // MONKEY (sky): the Enhanced/High branch.
        if key.bind_group_data.fx {
            if let Some(fragment) = descriptor.fragment.as_mut() {
                fragment.shader_defs.push("SKY_FX".into());
            }
        }
        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            depth.depth_write_enabled = false;
        }
        crate::sky_order::sky_pipeline_state(descriptor);
        Ok(())
    }
}

/// Marker for the sky-dome entity.
#[derive(Component)]
struct Sky;

/// The sky dome: spawned at startup, pinned to the camera and fed the `Light.dbc` colours each
/// frame.
pub(crate) struct SkyPlugin;

impl Plugin for SkyPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<SkyMaterial>::default())
            .add_systems(Startup, setup_sky.after(AssetSet::Open))
            .add_systems(
                Update,
                (
                    // Reads the resolved atmosphere: unordered, it would paint last frame's
                    // palette, the underwater one on a surfacing frame.
                    // MONKEY (fix-sky): after the cloud tile, so the glow reads this frame's cover.
                    update_sky_colors
                        .in_set(crate::lighting::LightingConsumeSet)
                        .after(crate::clouds::CloudTick),
                    // The skybox and submersion gates must read their settled resolves.
                    apply_sky_visibility
                        .after(crate::skybox::SkyboxResolve)
                        .after(crate::liquid::SubmersionVerdict),
                ),
            )
            // After propagation (`BillboardPlace`), from the camera's same-frame pose.
            .add_systems(
                PostUpdate,
                follow_camera.in_set(crate::billboard::BillboardPlace),
            );
    }
}

fn col(c: [f32; 3]) -> Vec4 {
    Vec4::new(c[0], c[1], c[2], 1.0)
}

/// A unit UV sphere (Y-up); the gradient is per fragment, so a modest tessellation suffices.
fn dome_mesh() -> Mesh {
    const STACKS: usize = 24;
    const SECTORS: usize = 48;
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    for i in 0..=STACKS {
        let v = i as f32 / STACKS as f32;
        let phi = v * std::f32::consts::PI; // 0 (zenith) .. π (nadir)
        let (sp, cp) = phi.sin_cos();
        for j in 0..=SECTORS {
            let u = j as f32 / SECTORS as f32;
            let theta = u * std::f32::consts::TAU;
            let (st, ct) = theta.sin_cos();
            let p = [sp * ct, cp, sp * st]; // y = cos(phi): +1 zenith, −1 nadir
            positions.push(p);
            normals.push([-p[0], -p[1], -p[2]]); // inward (unused: unlit)
            uvs.push([u, v]);
        }
    }
    let mut indices: Vec<u32> = Vec::new();
    let stride = SECTORS + 1;
    for i in 0..STACKS {
        for j in 0..SECTORS {
            let a = (i * stride + j) as u32;
            let b = a + 1;
            let c = a + stride as u32;
            let d = c + 1;
            indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

fn setup_sky(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<SkyMaterial>>,
    clock: Option<Res<crate::sky_fx::SkyClockBuffer>>,
) {
    let Some(clock) = clock else {
        return;
    };
    let mesh = meshes.add(dome_mesh());
    // Seeded with the neutral-day fallback; `update_sky_colors` overwrites from Light.dbc each frame.
    let material = materials.add(SkyMaterial {
        base: StandardMaterial {
            unlit: true,
            cull_mode: None, // viewed from inside the dome
            ..default()
        },
        extension: SkyExt {
            sky0: col([0.30, 0.50, 0.85]),
            sky1: col([0.35, 0.58, 0.88]),
            sky2: col([0.55, 0.72, 0.92]),
            sky3: col([0.68, 0.80, 0.93]),
            sky4: col([0.78, 0.86, 0.95]),
            fog: col([0.55, 0.72, 0.92]),
            warp: Vec4::ZERO, // update_sky_colors fills S + sun azimuth each frame
            // MONKEY (sky): Classic until update_sky_colors reads the tier.
            fx: Vec4::ZERO,
            sun: Vec4::Y,
            glow: Vec4::ZERO,
            // MONKEY (fog): zero rows leave Classic unchanged.
            mf_fog_b: Vec4::ZERO,
            mf_fog_c: Vec4::ZERO,
            mf_fog_d: Vec4::ZERO,
            clock: clock.0.clone(),
        },
    });
    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::default(),
        Sky,
    ));
}

/// Pin the dome to the camera, world-aligned, just inside the far plane; the radius only keeps it
/// unclipped, since occlusion is `sky_vertex.wgsl`'s far-depth pin.
#[allow(clippy::type_complexity)]
fn follow_camera(
    cam: Query<(&GlobalTransform, &Projection), With<WorldCamera>>,
    mut dome: Query<(&mut Transform, &mut GlobalTransform), (With<Sky>, Without<WorldCamera>)>,
) {
    let Some((cam_gt, proj)) = cam.iter().next() else {
        return;
    };
    let Ok((mut tf, mut gt)) = dome.single_mut() else {
        return;
    };
    let far = match proj {
        Projection::Perspective(p) => p.far,
        _ => 3000.0,
    };
    tf.translation = cam_gt.translation();
    tf.rotation = Quat::IDENTITY;
    tf.scale = Vec3::splat(far * 0.9);
    // Propagation already ran this frame, so the direct global write is what renders.
    *gt = GlobalTransform::from(*tf);
}

/// Hide the dome for the debug toggle, under a WMO skybox (its MOSB model replaces the gradient)
/// and while submerged: the reference's submerged test (`0x6812a4`) skips `CSky::Render`
/// (`0x6d4940`) whole. The skybox gate is its weight: below 0.99 the dome still draws under the
/// crossfading skybox ([`crate::skybox::SkyboxWeight`]).
fn apply_sky_visibility(
    debug: Res<DebugState>,
    skybox: Res<crate::skybox::SkyboxWeight>,
    underwater: Res<crate::liquid::Underwater>,
    mut dome: Query<&mut Visibility, With<Sky>>,
) {
    let Ok(mut vis) = dome.single_mut() else {
        return;
    };
    let want =
        if debug.lighting.disable_sky_dome || skybox.replaces_celestial() || underwater.0.any() {
            Visibility::Hidden
        } else {
            Visibility::Visible
        };
    if *vis != want {
        *vis = want;
    }
}

/// Push the time-of-day sky stops and fog colour (`WowLighting`) into the dome material.
#[allow(clippy::too_many_arguments)]
fn update_sky_colors(
    light: Res<WowLighting>,
    // MONKEY (sky): the tier and the cloud cover over the sun.
    quality: Res<crate::sky_fx::SkyQuality>,
    clouds: Res<crate::clouds::CloudCoverage>,
    // MONKEY (fog): the Modern fog rows for the shared horizon colour.
    monkey: Res<crate::lighting::MonkeyFrame>,
    dome: Query<&MeshMaterial3d<SkyMaterial>, With<Sky>>,
    mut materials: ResMut<Assets<SkyMaterial>>,
) {
    let Ok(handle) = dome.single() else {
        return;
    };
    // Quantized to bytes (Light.dbc stops are byte colours) and write-gated, since a `get_mut`
    // alone re-uploads the material.
    let sky: Vec<Vec4> = light
        .sky
        .iter()
        .map(|c| col(benilla_assets::quant255(*c)))
        .collect();
    let f = benilla_assets::quant255(light.fog_color);
    // `fog.w` is unused by the shader.
    let fog = Vec4::new(f[0], f[1], f[2], 0.0);
    // Dawn/dusk warp: strength S and the sun's azimuth `atan2(z, x)`, quantized to 1/4096; S = 0
    // (midday, or a `highlightSky` 0 zone) leaves the gradient untouched.
    let s = light.celestial_dir;
    let sun_az = s.z.atan2(s.x);
    let warp = Vec4::new(
        benilla_assets::quantize(light.sky_warp, 4096.0),
        benilla_assets::quantize(sun_az, 4096.0),
        0.0,
        0.0,
    );
    // MONKEY (sky): the Enhanced/High inputs; all zero at Classic so the write gate stays quiet.
    let (fx, sun_v, glow) = sky_fx_inputs(&light, *quality, &clouds);
    // MONKEY (fog): rows B-D as packed for the light buffer; clock lanes are not read here.
    let rows = monkey.pack(0.0, 0.0);
    let mf_fog = [
        Vec4::from_array(rows[1]),
        Vec4::from_array(rows[2]),
        Vec4::from_array(rows[3]),
    ];
    benilla_assets::write_gated(
        &mut materials,
        &handle.0,
        |m| {
            m.extension.fx != fx
                || m.extension.sun != sun_v
                || m.extension.glow != glow
                || m.extension.mf_fog_b != mf_fog[0]
                || m.extension.mf_fog_c != mf_fog[1]
                || m.extension.mf_fog_d != mf_fog[2]
                || m.extension.sky0 != sky[0]
                || m.extension.sky1 != sky[1]
                || m.extension.sky2 != sky[2]
                || m.extension.sky3 != sky[3]
                || m.extension.sky4 != sky[4]
                || m.extension.fog != fog
                || m.extension.warp != warp
        },
        |m| {
            m.extension.sky0 = sky[0];
            m.extension.sky1 = sky[1];
            m.extension.sky2 = sky[2];
            m.extension.sky3 = sky[3];
            m.extension.sky4 = sky[4];
            m.extension.fog = fog;
            m.extension.warp = warp;
            m.extension.fx = fx;
            m.extension.sun = sun_v;
            m.extension.glow = glow;
            m.extension.mf_fog_b = mf_fog[0];
            m.extension.mf_fog_c = mf_fog[1];
            m.extension.mf_fog_d = mf_fog[2];
        },
    );
}

/// MONKEY (sky): the dome's Enhanced/High uniforms, `(fx, sun, glow)`; zeros at Classic.
fn sky_fx_inputs(
    light: &WowLighting,
    quality: crate::sky_fx::SkyQuality,
    clouds: &crate::clouds::CloudCoverage,
) -> (Vec4, Vec4, Vec4) {
    if !quality.enhanced() {
        return (Vec4::ZERO, Vec4::Y, Vec4::ZERO);
    }
    let sun = light.celestial_dir.normalize_or(Vec3::Y);
    // The cover at the glare point, the same sample the flare's occlusion reads.
    let cover = clouds.coverage(sun * 12.0);
    let strength = crate::sky_fx::glow_strength(sun.y, cover, light.storm_bcc);
    // Warm toward the sun colour, grounded in the fog colour so dusk glows orange, noon pale.
    let tint = Vec3::from(light.celestial_tint);
    let fog = Vec3::from(light.fog_color);
    let glow = fog.lerp(tint, 0.6);
    // The night sky follows the star curve, thinned by storm weather.
    let night = (light.star_alpha * (1.0 - light.storm_bcc)).clamp(0.0, 1.0);
    // Quantized like every other lane here, so an idle frame writes nothing.
    let q = |v: f32| benilla_assets::quantize(v, 4096.0);
    (
        Vec4::new(f32::from(quality.0), 0.0, q(night), q(strength)),
        Vec4::new(q(sun.x), q(sun.y), q(sun.z), 0.0),
        Vec4::new(q(glow.x), q(glow.y), q(glow.z), 0.0),
    )
}
