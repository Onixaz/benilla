//! MONKEY (fog): `LightFogBand.dbc`, the modern fog fields 1.12 has no column for.
//!
//! The file is ours (DESIGN-Benilla-Graphics §2a). It mirrors `LightFloatBand.dbc` exactly: the
//! same 34-field, 136-byte band record (`ID, num, time[16], value[16]`, times in half-minutes on
//! the 2880 day), interpolated the same way across midnight. `LightParams` id `P` keys
//! [`FOG_BANDS_PER_PARAM`] rows, band `b` at id `(P − 1)·13 + b + 1`, the `LightFloatBand`
//! indexing with its own stride, so the stock `paramID·6 + k` tables stay untouched.
//!
//! | band | field | unit |
//! |---|---|---|
//! | 0 | height-fog density | extra extinction below the plane, × the distance fog's (0 = off) |
//! | 1 | height-fog height | WoW world Z, yards |
//! | 2 | height-fog falloff | 1/yd: how fast the extra density fades above the plane |
//! | 3 | curve blend | 0 = exponential (modern), 1 = the 1.12 linear ramp |
//! | 4-6 | sun-fog colour r, g, b | gamma 0..1 |
//! | 7 | sun-fog strength | 0..1 |
//! | 8 | sun-fog angle | cosine threshold of the lobe around the sun |
//! | 9-11 | end-fog colour r, g, b | gamma 0..1 |
//! | 12 | end-fog distance | yards (0 = the scene fog end) |
//!
//! A missing file, or a `LightParams` id without rows, is not an error: the caller falls back to
//! the colours it derives from the 1.12 bands.

use std::collections::HashMap;

use benilla_dbc::FieldType;

use super::tables::{band_schema, load_float_bands, sample_float, Band};
use crate::Chain;

const FOG_BAND: &str = "DBFilesClient\\LightFogBand.dbc";

/// Rows per `LightParams` id.
pub const FOG_BANDS_PER_PARAM: u32 = 13;

/// One `LightParams` id's fog fields at one time of day (see the module table).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FogBand {
    pub height_density: f32,
    /// WoW world Z, yards.
    pub height: f32,
    pub height_falloff: f32,
    pub curve_blend: f32,
    pub sun_rgb: [f32; 3],
    pub sun_strength: f32,
    pub sun_angle: f32,
    pub end_rgb: [f32; 3],
    /// Yards; 0 = the scene fog end.
    pub end_distance: f32,
}

impl FogBand {
    /// Componentwise `self + (o − self)·t`, the area and storm blend.
    pub fn lerp(&self, o: &FogBand, t: f32) -> FogBand {
        let l = |a: f32, b: f32| a + (b - a) * t;
        let l3 = |a: [f32; 3], b: [f32; 3]| [l(a[0], b[0]), l(a[1], b[1]), l(a[2], b[2])];
        FogBand {
            height_density: l(self.height_density, o.height_density),
            height: l(self.height, o.height),
            height_falloff: l(self.height_falloff, o.height_falloff),
            curve_blend: l(self.curve_blend, o.curve_blend),
            sun_rgb: l3(self.sun_rgb, o.sun_rgb),
            sun_strength: l(self.sun_strength, o.sun_strength),
            sun_angle: l(self.sun_angle, o.sun_angle),
            end_rgb: l3(self.end_rgb, o.end_rgb),
            end_distance: l(self.end_distance, o.end_distance),
        }
    }
}

/// The parsed `LightFogBand.dbc`; empty when the file is absent.
#[derive(Default)]
pub struct FogBandCatalog {
    bands: HashMap<u32, Band<f32>>,
}

impl FogBandCatalog {
    /// Read `LightFogBand.dbc` off the chain. Absent or unreadable → an empty catalog (every
    /// lookup `None`), which is the 1.12 data set.
    pub fn load(chain: &mut Chain) -> Self {
        let bands = load_float_bands(
            chain,
            FOG_BAND,
            band_schema("LightFogBand", FieldType::Float32),
        )
        .unwrap_or_default();
        FogBandCatalog { bands }
    }

    /// Whether the file carried any row.
    pub fn is_empty(&self) -> bool {
        self.bands.is_empty()
    }

    /// `LightParams` id `p` at `time` (half-minutes), or `None` when `p` has no rows. A single
    /// missing band of a present id reads 0.
    pub fn sample(&self, p: u32, time: u32) -> Option<FogBand> {
        if p < 1 {
            return None;
        }
        let base = (p - 1) * FOG_BANDS_PER_PARAM + 1;
        // MONKEY (fix-fog): a fixed array, no per-frame Vec (review B11).
        let rows: [Option<f32>; FOG_BANDS_PER_PARAM as usize] = std::array::from_fn(|b| {
            self.bands
                .get(&(base + b as u32))
                .and_then(|band| sample_float(band, time))
        });
        if rows.iter().all(Option::is_none) {
            return None;
        }
        let v = |b: usize| rows[b].unwrap_or(0.0);
        Some(FogBand {
            height_density: v(0),
            height: v(1),
            height_falloff: v(2),
            curve_blend: v(3),
            sun_rgb: [v(4), v(5), v(6)],
            sun_strength: v(7),
            sun_angle: v(8),
            end_rgb: [v(9), v(10), v(11)],
            end_distance: v(12),
        })
    }

    /// The area/storm blend of [`super::LightCatalog::blend_chain`]: each id's sample, or
    /// `fallback` for an id without rows, lerped in chain order. `None` when no id in the chain has
    /// rows, so the caller keeps its own derivation untouched.
    pub fn sample_chain(
        &self,
        chain: &[(u32, f32)],
        time: u32,
        fallback: &FogBand,
    ) -> Option<FogBand> {
        let mut any = false;
        let mut acc = *fallback;
        for &(p, w) in chain {
            let v = match self.sample(p, time) {
                Some(v) => {
                    any = true;
                    v
                }
                None => *fallback,
            };
            acc = acc.lerp(&v, w);
        }
        any.then_some(acc)
    }

    /// Test/tool constructor: a catalog from `(row id, [(time, value)])` bands.
    pub fn from_rows(rows: &[(u32, &[(u32, f32)])]) -> Self {
        let mut bands = HashMap::new();
        for (id, keys) in rows {
            bands.insert(*id, Band::from_keys(keys));
        }
        FogBandCatalog { bands }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn neutral() -> FogBand {
        FogBand {
            height_density: 0.0,
            height: 0.0,
            height_falloff: 0.0,
            curve_blend: 0.0,
            sun_rgb: [0.0; 3],
            sun_strength: 0.0,
            sun_angle: 1.0,
            end_rgb: [0.0; 3],
            end_distance: 0.0,
        }
    }

    #[test]
    fn empty_catalog_is_absent() {
        let c = FogBandCatalog::default();
        assert!(c.is_empty());
        assert_eq!(c.sample(12, 1440), None);
        assert_eq!(c.sample_chain(&[(12, 1.0)], 1440, &neutral()), None);
    }

    #[test]
    fn rows_land_at_the_param_stride_and_interpolate() {
        // Param 12, band 7 (sun strength): 0 at 06:00, 1 at 18:00.
        let id = 11 * FOG_BANDS_PER_PARAM + 7 + 1;
        let c = FogBandCatalog::from_rows(&[(id, &[(720, 0.0), (2160, 1.0)])]);
        let noon = c.sample(12, 1440).unwrap();
        assert!((noon.sun_strength - 0.5).abs() < 1e-6);
        assert_eq!(noon.height_density, 0.0);
        assert_eq!(c.sample(13, 1440), None);
    }

    #[test]
    fn chain_blends_rows_against_the_fallback() {
        let id = 11 * FOG_BANDS_PER_PARAM + 7 + 1;
        let c = FogBandCatalog::from_rows(&[(id, &[(0, 1.0)])]);
        // Base 13 (no rows → fallback strength 0), then param 12 at weight 0.25.
        let out = c
            .sample_chain(&[(13, 1.0), (12, 0.25)], 100, &neutral())
            .unwrap();
        assert!((out.sun_strength - 0.25).abs() < 1e-6);
    }
}
