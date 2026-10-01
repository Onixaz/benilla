//! MONKEY (sky): the sky quality tier and what the Enhanced/High sky reads each frame.
//!
//! `skyQuality` 0 is Classic, the reference's image byte for byte: the dome's piecewise-linear
//! gradient and the kernel's cloud texels, untouched. 1 (Enhanced) turns on the smooth gradient,
//! the sun glow and the procedural night sky in `sky.wgsl`; 2 (High) adds the cloud detail and the
//! sun-lit cloud shading in `cloud.wgsl`. The WGSL lives in `shaders/sky_fx.wgsl`.
//!
//! The High cloud detail and sun march (`sky_fx.wgsl`, `cloud.wgsl`, `clouds/layer.rs`):
//! Ported from WarcraftXL (https://github.com/WarcraftXL) by iThorgrim — module wxl-retail-clouds, Clouds.cpp, Clouds.hpp.

use bevy::prelude::*;
use bevy::render::{
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_resource::{Buffer, BufferDescriptor, BufferUsages},
    renderer::{RenderDevice, RenderQueue},
    Render, RenderApp, RenderSystems,
};

/// The sky tier: 0 Classic, 1 Enhanced, 2 High. Written by the app from the `skyQuality` cvar;
/// `$WOW_SKY_QUALITY` pins it for the session (captures).
#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct SkyQuality(pub u8);

impl SkyQuality {
    /// The highest tier.
    pub const MAX: u8 = 2;

    /// The session override, `$WOW_SKY_QUALITY` clamped to the tiers.
    pub fn env_override() -> Option<u8> {
        std::env::var("WOW_SKY_QUALITY")
            .ok()
            .and_then(|v| v.trim().parse::<u8>().ok())
            .map(|v| v.min(Self::MAX))
    }

    pub fn enhanced(self) -> bool {
        self.0 >= 1
    }

    pub fn high(self) -> bool {
        self.0 >= 2
    }
}

/// The sky's own animation clock (star twinkle, cloud detail drift), seconds wrapped to a day.
/// Frozen in capture mode, at `$WOW_CAPTURE_SKY_T` or 0.
///
/// MONKEY (fix-sky): it wrapped hourly in f32, and every star's twinkle jumped at the wrap. Now it
/// accumulates in f64 and wraps at [`SKY_CLOCK_WRAP_S`]; the shader's twinkle rates are whole
/// cycles per wrap (seamless). MONKEY (polish): the High cloud detail tiles and drifts whole
/// tiles per wrap (`CLOUD_TILE`, `CLOUD_DRIFT_TILES`), so it is seamless there too.
#[derive(Resource, Default)]
pub struct SkyClock {
    pub secs: f32,
    acc: f64,
    frozen: Option<f32>,
}

/// The sky clock's wrap in seconds; `sky_fx.wgsl`'s `SKY_WRAP` must match.
pub const SKY_CLOCK_WRAP_S: f64 = 86_400.0;

/// MONKEY (polish): the sky clock on the GPU, one `vec4` (`x` = [`SkyClock::secs`]) that the sky
/// and cloud materials bind read-only. It is rewritten in place each frame in the render world,
/// so the materials re-prepare only when a real input moves, not to advance the clock. Bevy's
/// `globals.time` is not used: it wraps hourly, and the twinkle rates are whole cycles per day.
#[derive(Resource, Clone, ExtractResource)]
pub struct SkyClockBuffer(pub Buffer);

/// The clock value the render world uploads.
#[derive(Resource, Clone, Copy)]
struct SkyClockSecs(f32);

impl ExtractResource for SkyClockSecs {
    type Source = SkyClock;

    fn extract_resource(source: &SkyClock) -> Self {
        Self(source.secs)
    }
}

/// How strongly the glow shows: a broad halo at `GLOW_GAIN` × the sun colour at the sun itself.
pub(crate) const GLOW_GAIN: f32 = 0.22;

/// The glow's day fade: full once the sun clears the horizon, gone a few degrees under it.
pub(crate) fn glow_day_fade(sun_y: f32) -> f32 {
    let t = ((sun_y + 0.10) / 0.16).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The glow strength: the day fade, dimmed by the cloud cover over the sun (the same coverage the
/// glare's occlusion reads) and by storm weather.
pub(crate) fn glow_strength(sun_y: f32, cover_at_sun: f32, bcc: f32) -> f32 {
    GLOW_GAIN
        * glow_day_fade(sun_y)
        * (1.0 - 0.8 * cover_at_sun.clamp(0.0, 1.0))
        * (1.0 - 0.85 * bcc.clamp(0.0, 1.0))
}

pub(crate) struct SkyFxPlugin;

impl Plugin for SkyFxPlugin {
    fn build(&self, app: &mut App) {
        // The library `sky.wgsl` and `cloud.wgsl` import; its embedded path is served by
        // `shaders::plugin`, this keeps it loaded so the import resolves.
        let lib: Handle<Shader> = bevy::asset::load_embedded_asset!(app, "shaders/sky_fx.wgsl");
        std::mem::forget(lib);
        let frozen = std::env::var_os("WOW_CAPTURE").map(|_| {
            std::env::var("WOW_CAPTURE_SKY_T")
                .ok()
                .and_then(|v| v.trim().parse::<f32>().ok())
                .unwrap_or(0.0)
        });
        app.insert_resource(SkyClock {
            secs: frozen.unwrap_or(0.0),
            acc: 0.0,
            frozen,
        });
        let quality = SkyQuality(SkyQuality::env_override().unwrap_or(0));
        app.insert_resource(quality)
            // MONKEY (integration): the clock ticks before its readers (sky colours, cloud FX).
            .add_systems(
                Update,
                tick_sky_clock.before(crate::lighting::LightingConsumeSet),
            )
            // Beside the shared light buffer, ahead of the sky and cloud materials built after it.
            .add_systems(
                Startup,
                init_sky_clock_buffer.in_set(benilla_assets::AssetSet::Open),
            )
            .add_plugins((
                ExtractResourcePlugin::<SkyClockBuffer>::default(),
                ExtractResourcePlugin::<SkyClockSecs>::default(),
            ));
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_systems(
                Render,
                upload_sky_clock.in_set(RenderSystems::PrepareResources),
            );
        }
    }
}

/// A headless build has no device, and then no sky or cloud dome either.
fn init_sky_clock_buffer(mut commands: Commands, device: Option<Res<RenderDevice>>) {
    let Some(device) = device else {
        return;
    };
    commands.insert_resource(SkyClockBuffer(device.create_buffer(&BufferDescriptor {
        label: Some("sky_clock"),
        size: 16,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })));
}

/// Render-world: the clock into the shared buffer before any sky draw reads it.
fn upload_sky_clock(
    queue: Res<RenderQueue>,
    buffer: Option<Res<SkyClockBuffer>>,
    secs: Option<Res<SkyClockSecs>>,
) {
    let (Some(buffer), Some(secs)) = (buffer, secs) else {
        return;
    };
    queue.write_buffer(&buffer.0, 0, bytemuck::cast_slice(&[secs.0, 0.0, 0.0, 0.0]));
}

fn tick_sky_clock(time: Res<Time>, mut clock: ResMut<SkyClock>) {
    if let Some(t) = clock.frozen {
        if clock.secs != t {
            clock.secs = t;
        }
        return;
    }
    clock.acc = (clock.acc + time.delta_secs_f64()) % SKY_CLOCK_WRAP_S;
    clock.secs = clock.acc as f32;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `const NAME ... = <value>;` line of `sky_fx.wgsl`, as text.
    fn wgsl_const(name: &str) -> String {
        let src = include_str!("shaders/sky_fx.wgsl");
        let line = src
            .lines()
            .find(|l| l.starts_with(&format!("const {name}:")))
            .unwrap_or_else(|| panic!("no const {name}"));
        let value = line.split('=').nth(1).unwrap().trim();
        value.trim_end_matches(';').trim().to_string()
    }

    #[test]
    fn the_shader_wrap_and_the_cloud_loop_match_the_clock() {
        let wrap: f64 = wgsl_const("SKY_WRAP").parse().unwrap();
        assert_eq!(wrap, SKY_CLOCK_WRAP_S);
        // The loop needs an even tile (the half-rate warp tiles at half of it) and whole tiles of
        // drift per wrap; the rates stay within 5% of WarcraftXL's 0.011 and 0.0043 cells/s.
        let tile: i32 = wgsl_const("CLOUD_TILE").parse().unwrap();
        assert!(tile > 0 && tile % 2 == 0);
        let drift = wgsl_const("CLOUD_DRIFT_TILES");
        let tiles: Vec<f64> = drift
            .trim_start_matches("vec2<f32>(")
            .trim_end_matches(')')
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();
        for (n, rate) in tiles.iter().zip([0.011, 0.0043]) {
            assert_eq!(n.fract(), 0.0, "whole tiles per wrap");
            let actual = n * f64::from(tile) / SKY_CLOCK_WRAP_S;
            assert!(
                (actual / rate - 1.0).abs() < 0.05,
                "drift {actual} vs {rate}"
            );
        }
    }

    #[test]
    fn glow_fades_out_below_the_horizon_and_under_cloud() {
        assert_eq!(glow_strength(-0.2, 0.0, 0.0), 0.0);
        assert!((glow_strength(0.5, 0.0, 0.0) - GLOW_GAIN).abs() < 1e-6);
        assert!(glow_strength(0.5, 1.0, 0.0) < 0.25 * GLOW_GAIN);
        assert!(glow_strength(0.5, 0.0, 1.0) < 0.2 * GLOW_GAIN);
        // Monotone across dusk.
        let mut prev = 0.0;
        for i in 0..40 {
            let y = -0.15 + i as f32 * 0.01;
            let g = glow_day_fade(y);
            assert!(g >= prev);
            prev = g;
        }
    }
}
