//! MONKEY (world shadows): CPU collection of the RETAINED static world's geometry as shadow-caster
//! triangles — the same source-submesh reachback `pick.rs` does for ray hits, appended into a
//! world-space triangle list. The world-shadow lane (`benilla-app::world_shadow`) appends
//! these into a cached, layer-31 caster mesh so trees and buildings cast into the SAME Bevy
//! directional shadow map as characters, instead of only their baked shading.
//!
//! The retained pass recentres vertices per cell and bakes them into a `RENDER_WORLD`-only mesh
//! whose CPU data is gone after extract — so we go to the SOURCE the divert kept
//! ([`super::GxItem::geometry`] + [`super::GxItem::transform`]), exactly as `pick.rs` does, and
//! reconstruct world space directly: `transform.transform_point(wow_to_bevy(pos))`.

use benilla_assets::coords::wow_to_bevy;
use bevy::asset::AssetId;
use bevy::image::Image;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use super::{GxCell, StaticGx};

/// One alpha-tested foliage caster group: every resident cutout batch that shares a single leaf
/// texture, merged into one world-space triangle list carrying UVs. The character-shadow system
/// drives one layer-31 caster entity + [`super::super`]-side `CutoutShadowCasterMaterial` per
/// bucket, so the shadow pass can sample this texture and discard transparent texels — casting a
/// leaf-SHAPED silhouette instead of the solid box a positions-only proxy throws.
pub struct CutoutBucket {
    /// The grouping key: the leaf sheet's asset id (stable across rebuilds, so the caller keys a
    /// persistent caster entity off it rather than churning one per rebuild).
    pub texture_id: AssetId<Image>,
    /// A live strong handle to the same leaf image the forward pass draws — put straight on the
    /// caster material so its sampler inherits the image's own clamp/repeat address mode.
    pub texture: Handle<Image>,
    /// World-space (Bevy) caster positions.
    pub positions: Vec<[f32; 3]>,
    /// Per-vertex UVs, parallel to [`Self::positions`] — what the prepass fragment samples.
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}

impl StaticGx {
    /// Append world-space (Bevy) shadow-caster triangles for resident static geometry whose
    /// placement anchor is within `reach` of `center`, onto the caller's buffers (the character
    /// system recycles them across rebuilds). Whole triangles referencing out-of-range indices are
    /// dropped — a truncated submesh must never rewire a later triangle into garbage (mirrors
    /// `character_shadow::append_triangles`).
    ///
    /// FIRST CUT: a currently-EXILED (distance-faded) static item also draws as an entity, so it is
    /// collected by both lanes and its geometry lands in the caster twice — harmless (identical
    /// triangles at identical depth), a follow-up can gate on the fader state.
    pub fn append_shadow_triangles(
        &self,
        center: Vec3,
        reach: f32,
        positions: &mut Vec<[f32; 3]>,
        indices: &mut Vec<u32>,
    ) {
        let r2 = reach * reach;
        let mut push = |cell: &GxCell| {
            for item in &cell.items {
                // Distance gate on the placement anchor BEFORE any vertex work.
                if item.transform.translation.distance_squared(center) > r2 {
                    continue;
                }
                // MONKEY (world shadows): the SOLID caster skips alpha-cutout foliage — a leaf card
                // is a big flat quad, and cast solid here (this mesh has no texture) it would be an
                // ugly rectangular BOX on the ground. Trunks/buildings/fences (opaque) cast their
                // real silhouette here; cutout foliage is collected leaf-shaped by
                // [`Self::collect_cutout_shadow_triangles`] instead. (`item.cutout` == AlphaTest.)
                if item.cutout {
                    continue;
                }
                let base = positions.len() as u32;
                let t = item.transform;
                positions.extend(
                    item.geometry
                        .positions
                        .iter()
                        .map(|p| t.transform_point(wow_to_bevy(*p)).to_array()),
                );
                let added = positions.len() as u32 - base;
                for tri in item.geometry.indices.as_chunks::<3>().0 {
                    if tri.iter().all(|i| *i < added) {
                        indices.extend(tri.iter().map(|i| base + *i));
                    }
                }
            }
        };
        for cell in self.cells.values() {
            push(cell);
        }
        for cell in self.wmos.values() {
            push(cell);
        }
        for cell in self.props.values() {
            push(cell);
        }
    }

    /// Collect the resident ALPHA-CUTOUT geometry (tree canopies, bushes, cattails — the leaf
    /// cards the solid caster skips) as one [`CutoutBucket`] per distinct leaf texture, each a
    /// world-space triangle list with UVs. The character-shadow system drives one alpha-tested
    /// layer-31 caster per bucket, so a canopy discards its transparent texels in the shadow pass
    /// and casts a leaf-SHAPED silhouette. Grouping by texture is what lets a single draw carry
    /// one sheet: a zone's canopies draw from only a handful of sheets, so this yields a handful
    /// of buckets. Range-gates on the placement anchor exactly like [`Self::append_shadow_triangles`].
    ///
    /// A cutout batch without a texture handle, or whose UVs don't parallel its positions, is
    /// skipped (it can't be alpha-tested — better no shadow than a wrong solid box).
    pub fn collect_cutout_shadow_triangles(&self, center: Vec3, reach: f32) -> Vec<CutoutBucket> {
        let r2 = reach * reach;
        let mut buckets: HashMap<AssetId<Image>, CutoutBucket> = HashMap::new();
        let mut push = |cell: &GxCell| {
            for item in &cell.items {
                if !item.cutout {
                    continue;
                }
                let (Some(texture_id), Some(texture)) =
                    (item.texture, item.texture_handle.as_ref())
                else {
                    continue;
                };
                if item.transform.translation.distance_squared(center) > r2 {
                    continue;
                }
                let geometry = &item.geometry;
                // UVs must parallel positions to map per-vertex — a cutout card that lost its UVs
                // (should never happen) can't be sampled, so drop it rather than box it out.
                if geometry.uvs.len() != geometry.positions.len() {
                    continue;
                }
                let bucket = buckets.entry(texture_id).or_insert_with(|| CutoutBucket {
                    texture_id,
                    texture: texture.clone(),
                    positions: Vec::new(),
                    uvs: Vec::new(),
                    indices: Vec::new(),
                });
                let base = bucket.positions.len() as u32;
                let t = item.transform;
                bucket.positions.extend(
                    geometry
                        .positions
                        .iter()
                        .map(|p| t.transform_point(wow_to_bevy(*p)).to_array()),
                );
                bucket.uvs.extend_from_slice(&geometry.uvs);
                let added = bucket.positions.len() as u32 - base;
                for tri in geometry.indices.as_chunks::<3>().0 {
                    if tri.iter().all(|i| *i < added) {
                        bucket.indices.extend(tri.iter().map(|i| base + *i));
                    }
                }
            }
        };
        for cell in self.cells.values() {
            push(cell);
        }
        for cell in self.wmos.values() {
            push(cell);
        }
        for cell in self.props.values() {
            push(cell);
        }
        buckets.into_values().collect()
    }
}
