//! MONKEY (skybox): `LightSkybox.dbc` in both layouts and the zone skybox walk.
//!
//! The 1.12 table is `ID, Name` (8-byte records). The extended table the sky patch ships is the
//! modern column set with paths in place of FileDataIDs: `ID, Name, Flags, CelestialName`
//! (16-byte records). The layout is picked by the header's field count, so an unpatched data set
//! loads with flags 0 and no celestial model.
//!
//! [`LightCatalog::zone_skyboxes`] is the modern client's skybox collector
//! (`DayNightLightHolder::SkyBoxCollector::addSkyBox`) over the same global-then-farthest-first
//! walk as [`LightCatalog::sample_blended`]: the same model keeps its largest alpha, and every
//! other model collected so far is scaled by `1 − alpha`, so two skyboxes crossfade. A light whose
//! param names no skybox leaves the list alone, as in the modern client.

use std::collections::HashMap;

use anyhow::{Context, Result};
use benilla_dbc::{FieldType, Schema, SchemaField};

use super::{blend_alpha, weather_slot, LightCatalog, Submersion, FALLBACK_LIGHT_ID, SLOT_CLEAR};
use crate::dbc::{parse, str_at, u32_at};
use crate::Chain;

/// Flag `0x1`: the model's sequence 0 plays across the game day, `time = duration × day fraction`.
pub const SKYBOX_FULL_DAY: u32 = 0x1;
/// Flag `0x2`: the procedural sky (dome, sun, moons, stars, clouds) keeps drawing under the model.
pub const SKYBOX_KEEP_CELESTIAL: u32 = 0x2;
/// Flag `0x4`: an alpha cone in the final fog colour draws over the model's horizon.
pub const SKYBOX_FOG_BLEND: u32 = 0x4;
/// MONKEY (leftovers): flag `0x8`: sun shafts run under this sky even where they would fade.
pub const SKYBOX_FORCE_SUN_SHAFTS: u32 = 0x8;
/// MONKEY (leftovers): flag `0x10`: the fog's sun-colour lobe is off under this sky.
pub const SKYBOX_NO_SUN_FOG: u32 = 0x10;

/// One `LightSkybox.dbc` row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkyboxDef {
    /// The model's chain path, spelled as a WMO MOSB skybox is, so one model never builds twice.
    pub path: String,
    /// The modern flag word; 0 in the 1.12 layout.
    pub flags: u32,
    /// The second model the modern client names as the celestial layer; `None` when empty.
    pub celestial: Option<String>,
}

/// One weighted skybox the zone walk collected.
#[derive(Clone, Debug, PartialEq)]
pub struct ZoneSkybox {
    pub id: u32,
    pub weight: f32,
}

/// Read `LightSkybox.dbc`, choosing the schema by the header's field count.
pub(super) fn load_skyboxes(chain: &mut Chain, file: &str) -> Result<HashMap<u32, SkyboxDef>> {
    let bytes = chain
        .read_file(file)
        .with_context(|| format!("reading {file}"))?;
    let fields = bytes
        .get(8..12)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .unwrap_or(2);
    let extended = fields >= 4;
    let mut schema = Schema::new("LightSkybox");
    schema.add_field(SchemaField::new("ID", FieldType::UInt32));
    schema.add_field(SchemaField::new("Name", FieldType::String));
    if extended {
        schema.add_field(SchemaField::new("Flags", FieldType::UInt32));
        schema.add_field(SchemaField::new("CelestialName", FieldType::String));
        // Columns past the four this client reads, if a later table carries them.
        for i in 4..fields {
            schema.add_field(SchemaField::new(format!("extra{i}"), FieldType::UInt32));
        }
    }
    let rs = parse(&bytes, schema, "LightSkybox")?;
    let mut m = HashMap::with_capacity(rs.records().len());
    for r in rs.records() {
        let (Some(id), Some(name)) = (u32_at(r, 0), str_at(&rs, r, 1)) else {
            continue;
        };
        let (flags, celestial) = if extended {
            (
                u32_at(r, 2).unwrap_or(0),
                str_at(&rs, r, 3)
                    .filter(|c| !c.trim().is_empty())
                    .map(|c| crate::models::model_path(&c)),
            )
        } else {
            (0, None)
        };
        m.insert(
            id,
            SkyboxDef {
                path: crate::models::model_path(&name),
                flags,
                celestial,
            },
        );
    }
    Ok(m)
}

/// MONKEY (reviewfix): build the deterministic reverse index used by WMO and ghost path lookups.
/// The lowest id wins when patched tables alias one model with multiple rows.
pub(super) fn index_paths(skyboxes: &HashMap<u32, SkyboxDef>) -> HashMap<String, u32> {
    let mut by_path = HashMap::with_capacity(skyboxes.len());
    for (&id, def) in skyboxes {
        by_path
            .entry(def.path.to_ascii_lowercase())
            .and_modify(|old: &mut u32| *old = (*old).min(id))
            .or_insert(id);
    }
    by_path
}

/// The modern collector's step: dedupe by model with the larger alpha, then scale every other
/// entry by `1 − alpha` (`addSkyBox` steps 1 and 4).
pub(super) fn collect(list: &mut Vec<ZoneSkybox>, id: u32, alpha: f32) {
    if alpha <= 0.0 {
        return;
    }
    let (at, alpha) = match list.iter_mut().position(|e| e.id == id) {
        Some(i) => {
            let a = list[i].weight.max(alpha);
            list[i].weight = a;
            (i, a)
        }
        None => {
            list.push(ZoneSkybox { id, weight: alpha });
            (list.len() - 1, alpha)
        }
    };
    for (i, e) in list.iter_mut().enumerate() {
        if i != at {
            e.weight *= 1.0 - alpha;
        }
    }
}

impl LightCatalog {
    /// A `LightSkybox.dbc` row by id.
    pub fn skybox_def(&self, id: u32) -> Option<&SkyboxDef> {
        self.skyboxes.get(&id)
    }

    /// The row whose model is `path`, for a MOSB skybox the table also names.
    pub fn skybox_def_by_path(&self, path: &str) -> Option<&SkyboxDef> {
        let path = crate::models::model_path(path).to_ascii_lowercase();
        self.skybox_by_path
            .get(&path)
            .and_then(|id| self.skyboxes.get(id))
    }

    /// The skybox the slot's param names; a weather slot without one falls back to the clear
    /// slot's, because the sky patch authors the clear slot only.
    fn light_skybox(&self, params: &[u32; 5], slot: usize) -> Option<u32> {
        let of = |p: u32| self.light_params_skybox.get(&p).copied();
        let param = match params[slot] {
            0 => params[SLOT_CLEAR],
            p => p,
        };
        of(param).or_else(|| {
            (slot != SLOT_CLEAR)
                .then(|| of(params[SLOT_CLEAR]))
                .flatten()
        })
    }

    /// The weighted skyboxes at `pos` for the living: the global (or fallback) light at alpha 1,
    /// then each local sphere containing `pos` farthest first at its falloff alpha, through
    /// [`collect`]. The slot is the lighting resolve's ([`weather_slot`]), ghost excluded: the
    /// death sky is [`Self::ghost_skybox`].
    pub fn zone_skyboxes(
        &self,
        map: u32,
        pos: [f32; 3],
        stormy: bool,
        submersion: Submersion,
    ) -> Vec<ZoneSkybox> {
        let mut out = Vec::new();
        if submersion.fixed_param().is_some() {
            return out;
        }
        let slot = weather_slot(false, stormy, submersion.is_water());
        let map_has_no_light = !self.lights.iter().any(|l| l.map == map);
        let base = self
            .lights
            .iter()
            .find(|l| l.map == map && l.global)
            .or_else(|| {
                map_has_no_light
                    .then(|| self.lights.iter().find(|l| l.id == FALLBACK_LIGHT_ID))
                    .flatten()
            });
        if let Some(sky) = base.and_then(|l| self.light_skybox(&l.params, slot)) {
            collect(&mut out, sky, 1.0);
        }
        let mut locals: Vec<(f32, &super::Light)> = self
            .lights
            .iter()
            .filter(|l| l.map == map && !l.global)
            .filter_map(|l| {
                let d = (0..3)
                    .map(|i| (l.pos[i] - pos[i]).powi(2))
                    .sum::<f32>()
                    .sqrt();
                (d <= l.falloff_end).then_some((d, l))
            })
            .collect();
        locals.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (dist, l) in locals {
            if let Some(sky) = self.light_skybox(&l.params, slot) {
                collect(
                    &mut out,
                    sky,
                    blend_alpha(dist, l.falloff_start, l.falloff_end),
                );
            }
        }
        out.retain(|e| e.weight > 0.0 && self.skyboxes.contains_key(&e.id));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(list: &[ZoneSkybox], id: u32) -> f32 {
        list.iter().find(|e| e.id == id).map_or(0.0, |e| e.weight)
    }

    #[test]
    fn a_second_model_crossfades_the_first() {
        let mut l = Vec::new();
        collect(&mut l, 1, 1.0);
        collect(&mut l, 2, 0.25);
        assert!((w(&l, 1) - 0.75).abs() < 1e-6);
        assert!((w(&l, 2) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn the_same_model_keeps_its_largest_alpha() {
        let mut l = Vec::new();
        collect(&mut l, 1, 0.5);
        collect(&mut l, 2, 0.5);
        collect(&mut l, 1, 0.8);
        assert_eq!(l.len(), 2);
        assert!((w(&l, 1) - 0.8).abs() < 1e-6);
        assert!((w(&l, 2) - 0.5 * 0.2).abs() < 1e-6);
    }

    #[test]
    fn a_zero_alpha_leaves_the_list_alone() {
        let mut l = Vec::new();
        collect(&mut l, 1, 1.0);
        collect(&mut l, 2, 0.0);
        assert_eq!(l, vec![ZoneSkybox { id: 1, weight: 1.0 }]);
    }

    #[test]
    fn duplicate_paths_resolve_to_the_lowest_id() {
        let def = |path: &str| SkyboxDef {
            path: path.into(),
            flags: 0,
            celestial: None,
        };
        let skyboxes = HashMap::from([
            (17, def(r"environments\stars\shared.m2")),
            (3, def(r"ENVIRONMENTS\STARS\SHARED.M2")),
            (9, def(r"environments\stars\other.m2")),
        ]);
        let paths = index_paths(&skyboxes);
        assert_eq!(paths.get(r"environments\stars\shared.m2"), Some(&3));
        assert_eq!(paths.get(r"environments\stars\other.m2"), Some(&9));
    }

    /// The installed chain loads in either layout.
    #[test]
    fn the_installed_table_loads() {
        let data = crate::wow_data_or_skip!();
        let mut chain = crate::open_chain(&data).expect("open chain");
        let m = load_skyboxes(&mut chain, super::super::LIGHT_SKYBOX).expect("LightSkybox");
        assert_eq!(
            m.get(&3).map(|d| d.path.as_str()),
            Some(r"environments\stars\deathclouds.m2"),
            "the ghost sky row"
        );
    }
}
