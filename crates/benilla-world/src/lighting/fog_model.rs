//! MONKEY (fog): the Modern fog model's CPU half — fills the FOG rows of [`MonkeyFrame`] every
//! frame. The law itself lives in `benilla-assets/src/shaders/fog_hook.wgsl`.
//!
//! Setting: cvar `fogModel` (0 Classic = the 1.12 linear fog, byte-identical; 1 Modern), bridged by
//! `benilla-app/src/monkey_gfx.rs` into [`FogModelSetting`]; `WOW_FOGMODEL=0|1` overrides it for a
//! session (captures).
//!
//! Modern, per frame:
//! - **Distances** stay the zone's: the shader's exponential curve is fitted to the 1.12 start/end
//!   pair it already receives. Only past the reference's farclip ceiling does the end grow with the
//!   view distance ([`modern_fog_end`], called from `resolve.rs`), so a larger `farclip` shows more.
//! - **Colours** without `LightFogBand.dbc` rows are derived from the 1.12 bands
//!   ([`derived_band`]): the sun-fog lobe leans the fog toward the sun colour (IntBand 9) around the
//!   sun and fades out at night; the far fog shifts toward the sky's lowest ring (1.8°) so the
//!   fogged world, the WDL hull and the sky dome's horizon meet in one colour. Height fog is off.
//! - With `LightFogBand.dbc` rows, those override the derivation per `LightParams`, blended over
//!   the same area-light chain as the stock bands.
//!
//! Packed rows (see `monkey_frame.rs`): `fog_a` height density / height (Bevy Y) / falloff / curve
//! blend; `fog_b` sun rgb / strength × day; `fog_c` end rgb / end distance; `fog_d` the scene fog end
//! (> 0 = Modern; the shader applies the model only to spans ending there, so an interior WMO fog
//! stays classic) / sun-fog cosine / the sun direction, octahedral-encoded.

use bevy::prelude::*;

use benilla_assets::{LockRecover, WorldAssets};
use benilla_formats::{FogBand, FogBandCatalog};

use super::{FogModel, LightSampler, MonkeyFrame, WowLighting};
use crate::terrain_stream::SPAWN_XY;
use crate::view::WorldCamera;
use crate::world_map::CurrentMap;
use benilla_assets::coords::bevy_to_wow;

/// The live fog-model choice (cvar `fogModel`). Default Classic; `WOW_FOGMODEL` overrides.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct FogModelSetting(pub FogModel);

impl Default for FogModelSetting {
    fn default() -> Self {
        FogModelSetting(env_override().unwrap_or(FogModel::Classic))
    }
}

/// `WOW_FOGMODEL=0|1`, read once.
fn env_override() -> Option<FogModel> {
    static ENV: std::sync::OnceLock<Option<FogModel>> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| {
        std::env::var("WOW_FOGMODEL")
            .ok()
            .and_then(|v| match v.trim() {
                "0" => Some(FogModel::Classic),
                "1" => Some(FogModel::Modern),
                _ => None,
            })
    })
}

impl FogModelSetting {
    /// Apply a cvar write; the env override, when set, wins for the session.
    pub fn set_from_cvar(&mut self, v: f32) {
        let want = env_override().unwrap_or(if v >= 0.5 {
            FogModel::Modern
        } else {
            FogModel::Classic
        });
        if self.0 != want {
            self.0 = want;
        }
    }

    pub fn modern(&self) -> bool {
        self.0 == FogModel::Modern
    }
}

/// `LightFogBand.dbc`, loaded once; empty without the file.
#[derive(Resource, Default)]
pub struct FogBandTable(pub FogBandCatalog);

/// The reference client's farclip ceiling (`FARCLIP_RANGE` end). Up to it Modern keeps the 1.12
/// fog pair exactly; beyond it the fog end scales with the view distance.
pub const FOG_REF_FARCLIP: f32 = 777.0;

/// The Modern scene fog pair: the zone's end, stretched by `farclip / 777` once the view distance
/// passes the reference ceiling, and never past `farclip` (the wall). Classic is
/// `(frac·min(end, farclip), min(end, farclip))`; at `farclip ≤ 777` the two are identical.
pub fn modern_fog_end(fog_end_raw: f32, fog_start_frac: f32, farclip: f32) -> (f32, f32) {
    let stretch = (farclip / FOG_REF_FARCLIP).max(1.0);
    let end = (fog_end_raw * stretch).min(farclip);
    (fog_start_frac * end, end)
}

/// Sun-fog lobe defaults (no `LightFogBand` row): strength and the cosine the lobe starts at.
pub const DERIVED_SUN_FOG_STRENGTH: f32 = 0.25;
pub const DERIVED_SUN_FOG_ANGLE: f32 = 0.4;
/// How far the sun-fog colour leans from the fog colour toward the sun colour (IntBand 9).
const DERIVED_SUN_FOG_LEAN: f32 = 0.4;
/// How far the end-fog colour leans from the fog colour toward the 1.8° sky ring.
const DERIVED_END_FOG_LEAN: f32 = 0.25;

fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// The Modern fields from 1.12 data alone: no height fog, the exponential curve, a sun-fog lobe in
/// the sun's colour and an end colour leaning to the sky's lowest ring, ending at the scene fog end.
pub fn derived_band(l: &WowLighting) -> FogBand {
    FogBand {
        height_density: 0.0,
        height: 0.0,
        height_falloff: 0.0,
        curve_blend: 0.0,
        sun_rgb: mix3(l.fog_color, l.spec, DERIVED_SUN_FOG_LEAN),
        sun_strength: DERIVED_SUN_FOG_STRENGTH,
        sun_angle: DERIVED_SUN_FOG_ANGLE,
        end_rgb: mix3(l.fog_color, l.sky[4], DERIVED_END_FOG_LEAN),
        end_distance: 0.0,
    }
}

/// The sun-fog day factor from the visible sun's height (`sin(elevation)`): full from ~3° up,
/// fading through the horizon to 0 by ~6° below it (the modern client's SunAngleBlend ramp).
pub fn sun_fog_day(sun_y: f32) -> f32 {
    let t = ((sun_y + 0.1) / 0.15).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Octahedral encoding of a unit vector into `[-1, 1]²` (the shader's `oct_decode`).
pub fn oct_encode(v: Vec3) -> [f32; 2] {
    let v = v.normalize_or(Vec3::Y);
    let n = v / (v.x.abs() + v.y.abs() + v.z.abs());
    if n.y >= 0.0 {
        [n.x, n.z]
    } else {
        let sx = if n.x >= 0.0 { 1.0 } else { -1.0 };
        let sz = if n.z >= 0.0 { 1.0 } else { -1.0 };
        [(1.0 - n.z.abs()) * sx, (1.0 - n.x.abs()) * sz]
    }
}

/// Write the FOG rows of `frame` for `band` under the resolved light. Classic zeroes them.
pub fn fill_frame(frame: &mut MonkeyFrame, model: FogModel, l: &WowLighting, band: &FogBand) {
    if model == FogModel::Classic {
        frame.fog_model = FogModel::Classic;
        frame.fog_scene_end = 0.0;
        frame.height_fog_density = 0.0;
        frame.height_fog_height = 0.0;
        frame.height_fog_falloff = 0.0;
        frame.fog_curve_blend = 0.0;
        frame.sun_fog_rgb = [0.0; 3];
        frame.sun_fog_strength = 0.0;
        frame.sun_fog_angle = 0.0;
        frame.sun_fog_dir = [0.0; 2];
        frame.end_fog_rgb = [0.0; 3];
        frame.end_fog_distance = 0.0;
        return;
    }
    let sun = l.celestial_dir;
    frame.fog_model = FogModel::Modern;
    frame.fog_scene_end = l.fog_end;
    frame.height_fog_density = band.height_density.max(0.0);
    // WoW Z is Bevy Y.
    frame.height_fog_height = band.height;
    frame.height_fog_falloff = band.height_falloff.max(0.0);
    frame.fog_curve_blend = band.curve_blend.clamp(0.0, 1.0);
    frame.sun_fog_rgb = band.sun_rgb;
    frame.sun_fog_strength = band.sun_strength.clamp(0.0, 1.0) * sun_fog_day(sun.y);
    frame.sun_fog_angle = band.sun_angle.clamp(-1.0, 0.999);
    frame.sun_fog_dir = oct_encode(sun);
    frame.end_fog_rgb = band.end_rgb;
    frame.end_fog_distance = if band.end_distance > 0.0 {
        band.end_distance
    } else {
        l.fog_end
    };
}

/// Startup: read `LightFogBand.dbc` (absent → empty).
pub(super) fn load_fog_bands(mut commands: Commands, world_assets: Option<ResMut<WorldAssets>>) {
    let Some(world_assets) = world_assets else {
        return;
    };
    let mut chain = world_assets.chain.lock_recover();
    let cat = FogBandCatalog::load(&mut chain);
    if !cat.is_empty() {
        info!("LightFogBand.dbc loaded: Modern fog uses its authored fields");
    }
    commands.insert_resource(FogBandTable(cat));
}

/// Per frame: the Modern fog fields into [`MonkeyFrame`] (after the light resolve).
#[allow(clippy::too_many_arguments)]
pub(super) fn update_fog_model(
    setting: Res<FogModelSetting>,
    lighting: Res<WowLighting>,
    table: Option<Res<FogBandTable>>,
    sampler: Option<Res<LightSampler>>,
    clock: Res<super::GameClock>,
    cam: Query<&GlobalTransform, With<WorldCamera>>,
    current_map: Option<Res<CurrentMap>>,
    eye_liquid: crate::liquid::EyeLiquid,
    viewer: Res<crate::view::Viewer>,
    weather: Option<Res<crate::weather::WeatherState>>,
    // MONKEY (leftovers): LightSkybox flag 0x10 (last frame's backdrop; the skybox resolves later).
    sky: Option<Res<crate::skybox::CameraSkybox>>,
    mut frame: ResMut<MonkeyFrame>,
) {
    let derived = derived_band(&lighting);
    let band = match (&table, &sampler, setting.modern()) {
        (Some(t), Some(s), true) if !t.0.is_empty() => {
            let pos = match cam.single() {
                Ok(t) => bevy_to_wow(t.translation()),
                Err(_) => [SPAWN_XY.0, SPAWN_XY.1, 83.5],
            };
            let map = current_map.as_ref().map_or(0, |m| m.0);
            let sub = eye_liquid.submersion();
            let time = clock.minute * 2;
            let storm = weather
                .as_ref()
                .map_or(0.0, |w| crate::weather::storm_blend(w.sky_density));
            let chain = |stormy| s.0.blend_chain(map, pos, stormy, sub, viewer.ghost);
            let clear = t.0.sample_chain(&chain(false), time, &derived);
            let stormy = (storm > 0.0)
                .then(|| t.0.sample_chain(&chain(true), time, &derived))
                .flatten();
            match (clear, stormy) {
                (None, None) => derived,
                (c, s) => c.unwrap_or(derived).lerp(&s.unwrap_or(derived), storm),
            }
        }
        _ => derived,
    };
    let mut next = *frame;
    fill_frame(&mut next, setting.0, &lighting, &band);
    // MONKEY (leftovers): a `0x10` skybox fades the sun-fog lobe out by its weight.
    let no_sun = sky.map_or(0.0, |s| s.flag_weight(benilla_formats::SKYBOX_NO_SUN_FOG));
    next.sun_fog_strength *= 1.0 - no_sun.clamp(0.0, 1.0);
    if *frame != next {
        *frame = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modern_end_matches_classic_up_to_the_reference_farclip() {
        for far in [177.0, 350.0, 500.0, 777.0] {
            for end in [120.0, 444.0, 900.0] {
                let classic_end: f32 = f32::min(end, far);
                let (s, e) = modern_fog_end(end, 0.25, far);
                assert_eq!(e, classic_end);
                assert_eq!(s, 0.25 * classic_end);
            }
        }
    }

    #[test]
    fn modern_end_grows_past_the_reference_farclip() {
        let (_, e) = modern_fog_end(444.0, 0.25, 1554.0);
        assert!((e - 888.0).abs() < 1e-3);
        // Never past the wall.
        let (_, e) = modern_fog_end(900.0, 0.25, 1000.0);
        assert_eq!(e, 1000.0);
    }

    #[test]
    fn classic_leaves_the_fog_rows_zero() {
        let mut f = MonkeyFrame::default();
        let l = WowLighting::default();
        fill_frame(&mut f, FogModel::Classic, &l, &derived_band(&l));
        assert_eq!(f, MonkeyFrame::default());
        assert!(f.pack(0.0, 0.0)[..4].iter().flatten().all(|v| *v == 0.0));
    }

    #[test]
    fn modern_marks_the_scene_end_and_fades_the_sun_at_night() {
        let mut l = WowLighting {
            fog_end: 444.0,
            celestial_dir: Vec3::new(0.0, 0.8, 0.6),
            ..Default::default()
        };
        let mut f = MonkeyFrame::default();
        fill_frame(&mut f, FogModel::Modern, &l, &derived_band(&l));
        assert_eq!(f.fog_scene_end, 444.0);
        assert_eq!(f.end_fog_distance, 444.0);
        assert!((f.sun_fog_strength - DERIVED_SUN_FOG_STRENGTH).abs() < 1e-6);
        l.celestial_dir = Vec3::new(0.0, -0.5, 0.86);
        fill_frame(&mut f, FogModel::Modern, &l, &derived_band(&l));
        assert_eq!(f.sun_fog_strength, 0.0);
    }

    #[test]
    fn oct_encoding_round_trips() {
        fn decode(e: [f32; 2]) -> Vec3 {
            let mut v = Vec3::new(e[0], 1.0 - e[0].abs() - e[1].abs(), e[1]);
            let t = (-v.y).max(0.0);
            v.x += if v.x >= 0.0 { -t } else { t };
            v.z += if v.z >= 0.0 { -t } else { t };
            v.normalize()
        }
        for d in [
            Vec3::Y,
            -Vec3::Y,
            Vec3::new(0.3, 0.2, -0.9),
            Vec3::new(-0.7, -0.1, 0.7),
            Vec3::new(0.1, -0.9, -0.4),
        ] {
            let d = d.normalize();
            assert!(decode(oct_encode(d)).dot(d) > 0.9999, "{d:?}");
        }
    }
}
