//! GFX (moonlight): the MOON as a light, on top of the stock night.
//!
//! 1.12 has no moon light. Its night is the `Light.dbc` night colours on the lighting sun, whose
//! direction never sets (`daynight::sun_direction`), so a night scene is lit from the same
//! compass point as noon, only darker and bluer. This module keeps that stock night as the base
//! and ADDS a cool, dim directional term from the visible white moon (`daynight::moon_direction`,
//! the disc the sky draws), which the receivers evaluate per fragment with N·L and shadow with the
//! moon's own shadow map when the shadow rig holds it (`ShadowHandover::moon_ramp`).
//!
//! Packed into the programme block (`MonkeyFrame` row 16 + `misc.w`, see `monkey_frame.rs`); the
//! WGSL half is `benilla::moonlight_hook`. [`MoonLight`] `0` is exactly the pre-feature render:
//! the intensity packs as 0 and every receiver's moon branch is skipped.
//!
//! Crossfades: the term ramps in with the moon's elevation over the same 0..12° smoothstep the
//! shadow hand-over uses ([`sun_shadow_strength`]) and out with the SUN's rise, so dusk (sun set,
//! moon not yet up) and dawn keep the stock night with no moon, and nothing ever steps.

use bevy::prelude::*;

use super::{sun_shadow_strength, MonkeyFrame, ShadowHandover, WowLighting};

/// The moonlight dial (`moonLight`, 0..2): a multiplier on [`MOON_LIGHT_BASE`]. 0 = off (the
/// pre-feature night, bit for bit), 1 = the shipped look. The settings bridge writes it.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct MoonLight(pub f32);

impl Default for MoonLight {
    /// Off until the settings bridge says otherwise, so a headless test or an off-world scene
    /// never grows a moon term it did not ask for.
    fn default() -> Self {
        Self(0.0)
    }
}

/// The moon's N·L = 1 contribution in gamma units at `moonLight 1`, a high full moon, clear sky.
/// Chosen against the Elwynn night (`nightGain 0.45`): enough to model a lit hillside and throw a
/// readable moon shadow, well short of the dawn ambient.
pub const MOON_LIGHT_BASE: f32 = 0.16;

/// How much a full storm (`storm_bcc = 1`) takes away: an overcast moon still glows a little.
const MOON_STORM_DIM: f32 = 0.75;

/// The moonlight intensity for one frame (before the tint the shader applies).
///
/// `sun_height` / `moon_height` are the SINE of each body's elevation (`celestial_dir.y`,
/// `moon_dir().y`); `storm` is the resolved storm blend `0..1`.
pub fn moon_light_intensity(gain: f32, sun_height: f32, moon_height: f32, storm: f32) -> f32 {
    if gain <= 0.0 {
        return 0.0;
    }
    let night = 1.0 - sun_shadow_strength(sun_height);
    let risen = sun_shadow_strength(moon_height);
    let sky = 1.0 - MOON_STORM_DIM * storm.clamp(0.0, 1.0);
    MOON_LIGHT_BASE * gain.clamp(0.0, 2.0) * night * risen * sky
}

/// Writes the moon row of [`MonkeyFrame`] (chained right before the packer).
pub(super) fn update_moonlight(
    gain: Res<MoonLight>,
    lighting: Res<WowLighting>,
    handover: Res<ShadowHandover>,
    mut frame: ResMut<MonkeyFrame>,
) {
    let moon = lighting.moon_dir().normalize_or_zero();
    let intensity = moon_light_intensity(
        gain.0,
        lighting.celestial_dir().y,
        moon.y,
        lighting.storm_bcc,
    );
    let (dir, intensity) = if intensity > 0.0 && moon != Vec3::ZERO {
        (moon.to_array(), intensity)
    } else {
        ([0.0; 3], 0.0)
    };
    let confidence = if intensity > 0.0 {
        handover.moon_ramp()
    } else {
        0.0
    };
    // Write through `ResMut` only on a change, so the packer's own change check stays quiet.
    if frame.moon_light_dir != dir
        || frame.moon_light != intensity
        || frame.moon_shadow_confidence != confidence
    {
        frame.moon_light_dir = dir;
        frame.moon_light = intensity;
        frame.moon_shadow_confidence = confidence;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_day_and_a_set_moon_are_exactly_zero() {
        // Midnight: sun far below, moon high.
        assert_eq!(
            moon_light_intensity(0.0, -0.9, 0.8, 0.0),
            0.0,
            "moonLight 0 is the null"
        );
        assert_eq!(
            moon_light_intensity(1.0, 0.7, 0.8, 0.0),
            0.0,
            "no moonlight by day"
        );
        assert_eq!(
            moon_light_intensity(1.0, -0.9, -0.1, 0.0),
            0.0,
            "a set moon lights nothing"
        );
        let full = moon_light_intensity(1.0, -0.9, 0.8, 0.0);
        assert!((full - MOON_LIGHT_BASE).abs() < 1e-6);
    }

    #[test]
    fn the_moon_ramps_in_with_its_elevation_and_storms_dim_it() {
        let mut prev = 0.0;
        for step in 0..=40 {
            let h = -0.05 + step as f32 * 0.01;
            let v = moon_light_intensity(1.0, -0.9, h, 0.0);
            assert!(v + 1e-7 >= prev, "moonrise must be monotonic");
            prev = v;
        }
        let clear = moon_light_intensity(1.0, -0.9, 0.8, 0.0);
        let storm = moon_light_intensity(1.0, -0.9, 0.8, 1.0);
        assert!(storm < clear * 0.5 && storm > 0.0);
        assert!(moon_light_intensity(2.0, -0.9, 0.8, 0.0) > clear * 1.9);
    }

    #[test]
    fn dawn_hands_the_night_back_as_the_sun_rises() {
        // The sun climbing through its ramp takes the moon term away continuously.
        let mut prev = f32::INFINITY;
        for step in 0..=25 {
            let sun = step as f32 * 0.01;
            let v = moon_light_intensity(1.0, sun, 0.3, 0.0);
            assert!(v <= prev + 1e-7);
            prev = v;
        }
        assert_eq!(prev, 0.0);
    }
}
