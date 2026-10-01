//! MONKEY (p0 MonkeyFrame): the graphics programme's per-frame parameter block.
//!
//! One resource, [`MonkeyFrame`], that any system may write (fog model, wind, wetness, benders).
//! The light packer (`global_light::pack_monkey_frame`) copies it every frame into 16 rows
//! appended to the shared light buffer right after the point-light table, so the offsets of every
//! row before it never move. The WGSL mirror is `benilla::monkey_frame::MonkeyFrame`
//! (`benilla-assets/src/shaders/monkey_frame.wgsl`); every receiver's `WowLight` struct declares
//! it as its `monkey` member.
//!
//! All zero (the default) is "every feature off": nothing reads a zero field as a change, so the
//! image is the same as before the block existed.
//!
//! Layout (17 × `vec4<f32>`, 272 B):
//!
//! | row | xyz / x,y,z | w |
//! |---|---|---|
//! | 0 `fog_a` | height_fog_density, height_fog_height, height_fog_falloff | curve_blend |
//! | 1 `fog_b` | sun_fog_rgb | sun_fog_strength |
//! | 2 `fog_c` | end_fog_rgb | end_fog_distance |
//! | 3 `fog_d` | fog_model (0 classic; Modern = the scene fog end, yd, ≥ 1), sun_fog_angle, sun dir oct.x | sun dir oct.y |
//! | 4 `wind_a` | dir_x, dir_y, base_heading (rad) | gust |
//! | 5 `wind_b` | travel (yd, wrapped), sway_strength, grass_strength | tree_strength |
//! | 6 `wet_a` | rain_rate, wetness, ripple_time_s | snow |
//! | 7 `misc` | bender_count, time_of_day 0..1, night 0..1 | moon shadow confidence 0..1 |
//! | 8-15 `benders` | world x, y, z | radius |
//! | 16 `moon` | moonlight direction (toward the moon, unit) | moonlight intensity (gamma, 0 = off) |
//!
//! GFX (moonlight): row 16 and `misc.w` belong to `lighting::moonlight`; the confidence is how far
//! the shadow rig's hand-over to the moon has ramped (0 = the map does not hold the moon).
//!
//! Coordinates are Bevy world space (Y up, 1 unit = 1 yd), the space the receivers'
//! `world_position` is in: WoW `(x, y, z)` → Bevy `(-y, z, -x)` ([`benilla_assets::coords`]).
//! The wind direction is a unit vector in the world XZ plane: `dir_x` = world x, `dir_y` = world z.
//! `time_of_day` and `night` are filled by the packer from the game clock and the celestial sun;
//! every other field belongs to the lane that owns the feature.

use bevy::prelude::*;

/// Rows in the appended block.
pub const MONKEY_FRAME_ROWS: usize = 17;

/// How many benders (grass/foliage pushers: the player and nearby units) the block carries.
pub const MAX_BENDERS: usize = 8;

/// The fog model selector, `fog_d.x`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FogModel {
    /// Today's linear 1.12 fog, byte-identical.
    #[default]
    Classic,
    /// The FOG lane's modern model (height / sun / end-colour / curve fog).
    Modern,
}

/// The per-frame programme parameters. Write the fields your lane owns through
/// `ResMut<MonkeyFrame>`; leave the rest alone. See the module docs for the packed layout.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct MonkeyFrame {
    // ── FOG lane (rows 0-3) ──
    pub fog_model: FogModel,
    pub height_fog_density: f32,
    /// World Y (Bevy space, yd) of the height-fog plane.
    pub height_fog_height: f32,
    pub height_fog_falloff: f32,
    pub fog_curve_blend: f32,
    /// Gamma 0..1.
    pub sun_fog_rgb: [f32; 3],
    pub sun_fog_strength: f32,
    /// Cosine threshold of the sun-fog lobe.
    pub sun_fog_angle: f32,
    /// MONKEY (fog): the sun direction (Bevy space), octahedral-encoded (`fog_model::oct_encode`).
    pub sun_fog_dir: [f32; 2],
    /// MONKEY (fog): the scene fog end (yd) the Modern law applies to; packed as `fog_d.x`, so an
    /// interior WMO span (a different end) stays classic.
    pub fog_scene_end: f32,
    /// Gamma 0..1.
    pub end_fog_rgb: [f32; 3],
    pub end_fog_distance: f32,
    // ── WIND lane (rows 4-5) ──
    /// Unit vector in the world XZ plane (`[x, z]`).
    pub wind_dir: [f32; 2],
    /// MONKEY (fix-wind): the profile's fixed heading in radians. The waves' spatial term uses
    /// it, so the veer never rotates the phase field about the world origin.
    pub wind_base_heading: f32,
    pub wind_gust: f32,
    /// MONKEY (fix-wind): the integrated wind travel (speed integrated over time) in yards, wrapped
    /// at `wind::TRAVEL_WRAP`; the shader's wave rates are whole cycles per wrap, so it is seamless.
    pub wind_travel: f32,
    pub sway_strength: f32,
    pub grass_strength: f32,
    pub tree_strength: f32,
    // ── WET lane (row 6) ──
    pub rain_rate: f32,
    pub wetness: f32,
    pub ripple_time_s: f32,
    pub snow: f32,
    // ── benders (rows 7.x, 8-15) ──
    /// Bender slots; only the first `min(bender_count, MAX_BENDERS)` are packed. `[x, y, z, radius]` in
    /// Bevy world space.
    pub benders: [[f32; 4]; MAX_BENDERS],
    pub bender_count: u32,
    // ── GFX (moonlight): row 16 + `misc.w` ──
    /// Unit direction toward the moon (Bevy space).
    pub moon_light_dir: [f32; 3],
    /// Moonlight intensity in gamma units at N·L = 1; 0 is exactly the pre-feature render.
    pub moon_light: f32,
    /// How far the shadow rig's hand-over to the moon has ramped (0..1).
    pub moon_shadow_confidence: f32,
}

impl MonkeyFrame {
    /// The 16 packed rows. `time_of_day` (0..1) and `night` (0..1) are the packer's own inputs.
    pub fn pack(&self, time_of_day: f32, night: f32) -> [[f32; 4]; MONKEY_FRAME_ROWS] {
        let mut rows = [[0.0f32; 4]; MONKEY_FRAME_ROWS];
        // MONKEY (fog): Modern packs the scene fog end (never below 1) as its flag.
        let model = match self.fog_model {
            FogModel::Classic => 0.0,
            FogModel::Modern => self.fog_scene_end.max(1.0),
        };
        rows[0] = [
            self.height_fog_density,
            self.height_fog_height,
            self.height_fog_falloff,
            self.fog_curve_blend,
        ];
        let s = self.sun_fog_rgb;
        rows[1] = [s[0], s[1], s[2], self.sun_fog_strength];
        let e = self.end_fog_rgb;
        rows[2] = [e[0], e[1], e[2], self.end_fog_distance];
        rows[3] = [
            model,
            self.sun_fog_angle,
            self.sun_fog_dir[0],
            self.sun_fog_dir[1],
        ];
        rows[4] = [
            self.wind_dir[0],
            self.wind_dir[1],
            self.wind_base_heading,
            self.wind_gust,
        ];
        rows[5] = [
            self.wind_travel,
            self.sway_strength,
            self.grass_strength,
            self.tree_strength,
        ];
        rows[6] = [self.rain_rate, self.wetness, self.ripple_time_s, self.snow];
        let n = (self.bender_count as usize).min(MAX_BENDERS);
        rows[7] = [n as f32, time_of_day, night, self.moon_shadow_confidence];
        rows[8..8 + n].copy_from_slice(&self.benders[..n]);
        let m = self.moon_light_dir;
        rows[16] = [m[0], m[1], m[2], self.moon_light];
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_frame_packs_to_zero_but_the_clock() {
        let rows = MonkeyFrame::default().pack(0.0, 0.0);
        assert!(rows.iter().flatten().all(|v| *v == 0.0));
        let rows = MonkeyFrame::default().pack(0.5, 1.0);
        assert_eq!(rows[7], [0.0, 0.5, 1.0, 0.0]);
    }

    #[test]
    fn fields_land_in_their_documented_rows() {
        let f = MonkeyFrame {
            fog_model: FogModel::Modern,
            height_fog_density: 1.0,
            height_fog_height: 2.0,
            height_fog_falloff: 3.0,
            fog_curve_blend: 4.0,
            sun_fog_rgb: [5.0, 6.0, 7.0],
            sun_fog_strength: 8.0,
            end_fog_rgb: [9.0, 10.0, 11.0],
            end_fog_distance: 12.0,
            sun_fog_angle: 13.0,
            sun_fog_dir: [0.5, -0.5],
            fog_scene_end: 444.0,
            wind_dir: [14.0, 15.0],
            wind_base_heading: 16.0,
            wind_gust: 17.0,
            wind_travel: 18.0,
            sway_strength: 19.0,
            grass_strength: 20.0,
            tree_strength: 21.0,
            rain_rate: 22.0,
            wetness: 23.0,
            ripple_time_s: 24.0,
            snow: 25.0,
            benders: [[26.0, 27.0, 28.0, 29.0]; MAX_BENDERS],
            bender_count: 20, // clamped to MAX_BENDERS
            moon_light_dir: [30.0, 31.0, 32.0],
            moon_light: 33.0,
            moon_shadow_confidence: 34.0,
        };
        let r = f.pack(0.25, 0.75);
        assert_eq!(r[0], [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(r[1], [5.0, 6.0, 7.0, 8.0]);
        assert_eq!(r[2], [9.0, 10.0, 11.0, 12.0]);
        assert_eq!(r[3], [444.0, 13.0, 0.5, -0.5]);
        assert_eq!(r[4], [14.0, 15.0, 16.0, 17.0]);
        assert_eq!(r[5], [18.0, 19.0, 20.0, 21.0]);
        assert_eq!(r[6], [22.0, 23.0, 24.0, 25.0]);
        assert_eq!(r[7], [8.0, 0.25, 0.75, 34.0]);
        assert_eq!(r[15], [26.0, 27.0, 28.0, 29.0]);
        assert_eq!(r[16], [30.0, 31.0, 32.0, 33.0]);
    }

    /// The WGSL mirror declares the same rows; a member added on one side only would shift
    /// every bender.
    #[test]
    fn the_wgsl_mirror_has_the_same_rows() {
        let src = include_str!("../../../benilla-assets/src/shaders/monkey_frame.wgsl");
        let body = src.split_once("struct MonkeyFrame {").unwrap().1;
        let body = body.split_once("\n}").unwrap().0;
        let mut rows = 0;
        for line in body
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with("//"))
        {
            if line.contains("array<vec4<f32>, 8>") {
                rows += 8;
            } else if line.contains("vec4<f32>") {
                rows += 1;
            }
        }
        assert_eq!(rows, MONKEY_FRAME_ROWS);
    }
}
