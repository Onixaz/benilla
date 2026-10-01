//! MONKEY (daylight: terrain torch casters): terrain as a CASTER in the torch cube maps.
//!
//! The torch lane's static caster mesh is gathered from the retained scene (`StaticGx`) and the
//! rigid model parts; the ground was never in it, so a hill or a bank between a campfire and the
//! slope behind it never blocked the fire. This module hands the gather the resident MCNK chunks
//! within a fixture's reach, as world-space triangles (hole-masked indices, draw space — the same
//! vertices `collider::terrain_collider_data` welds, so the depth map and the drawn ground agree).
//!
//! Budget: one chunk is 145 vertices / up to 256 triangles; a 48 yd sphere touches at most ~16
//! chunks, so a slot gathers at most ~2.3k vertices and ~4k triangles once, when its static map
//! rebuilds. Chunks are culled by their own box against the sphere, then per triangle.

use benilla_assets::coords::wow_to_bevy;
use benilla_assets::AdtTile;
use bevy::prelude::*;

use super::TerrainStreamer;

/// The terrain vertex bounds of one chunk in WoW axes. A chunk's positions are absolute world
/// yards, so the box is found without the tile index.
fn chunk_box(chunk: &benilla_formats::ChunkMesh) -> Option<(Vec3, Vec3)> {
    let mut it = chunk.positions.iter();
    let first = Vec3::from(*it.next()?);
    let (mut lo, mut hi) = (first, first);
    for p in it {
        let p = Vec3::from(*p);
        lo = lo.min(p);
        hi = hi.max(p);
    }
    Some((lo, hi))
}

fn box_sphere_d2(lo: Vec3, hi: Vec3, c: Vec3) -> f32 {
    (c.clamp(lo, hi) - c).length_squared()
}

/// Append every resident terrain triangle within `reach` of `center` (Bevy world space) to the
/// caster buffers, and return how many triangles were added. A triangle is kept when its own
/// box reaches the sphere — cheaper than an exact test and never drops a real occluder.
pub fn append_terrain_torch_triangles(
    streamer: &TerrainStreamer,
    adt_tiles: &Assets<AdtTile>,
    center: Vec3,
    reach: f32,
    positions: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) -> u32 {
    // Chunk boxes are in WoW axes, so test the sphere there (the conversion is a rigid swizzle).
    let c = Vec3::from(benilla_assets::coords::bevy_to_wow(center));
    let r2 = reach * reach;
    let mut added = 0u32;
    for ts in streamer.tiles.values() {
        let Some(adt) = adt_tiles.get(&ts.handle) else {
            continue;
        };
        for chunk in &adt.chunks {
            let Some((lo, hi)) = chunk_box(chunk) else {
                continue;
            };
            if box_sphere_d2(lo, hi, c) > r2 {
                continue;
            }
            let base = positions.len() as u32;
            let mut used = false;
            for tri in chunk.indices.as_chunks::<3>().0 {
                let (Some(a), Some(b), Some(d)) = (
                    chunk.positions.get(tri[0] as usize),
                    chunk.positions.get(tri[1] as usize),
                    chunk.positions.get(tri[2] as usize),
                ) else {
                    continue;
                };
                let (a, b, d) = (Vec3::from(*a), Vec3::from(*b), Vec3::from(*d));
                if box_sphere_d2(a.min(b).min(d), a.max(b).max(d), c) > r2 {
                    continue;
                }
                if !used {
                    positions.extend(chunk.positions.iter().map(|p| wow_to_bevy(*p).to_array()));
                    used = true;
                }
                indices.extend(tri.iter().map(|i| base + i));
                added += 1;
            }
        }
    }
    added
}

/// A cheap stamp of the resident terrain set: which tiles are loaded (decoded) right now. Folded
/// into the torch lane's census and slot fingerprint so a tile streaming in or out re-gathers the
/// fixtures near it. Order-independent.
pub fn terrain_torch_generation(streamer: &TerrainStreamer, adt_tiles: &Assets<AdtTile>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut sum = 0u64;
    for (key, ts) in &streamer.tiles {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut h);
        ts.handle.id().hash(&mut h);
        adt_tiles.contains(&ts.handle).hash(&mut h);
        sum = sum.wrapping_add(h.finish());
    }
    sum
}

/// MONKEY (fix-daylight): [`terrain_torch_generation`] over only the tiles whose footprint
/// reaches the sphere (`center` in Bevy space), so a tile streaming far away does not re-key (and
/// re-render) every settled exterior torch slot. Keys are `(tile_x, tile_y)`: tile_x from world y,
/// tile_y from world x, both counted down from `32 * TILE_SIZE`.
pub fn terrain_torch_generation_near(
    streamer: &TerrainStreamer,
    adt_tiles: &Assets<AdtTile>,
    center: Vec3,
    reach: f32,
) -> u64 {
    use std::hash::{Hash, Hasher};
    let c = Vec3::from(benilla_assets::coords::bevy_to_wow(center));
    let t = benilla_formats::TILE_SIZE;
    let offset = 32.0 * t;
    let mut sum = 0u64;
    for (key, ts) in &streamer.tiles {
        let (tx, ty) = (key.0 as f32, key.1 as f32);
        let (x_hi, y_hi) = (offset - ty * t, offset - tx * t);
        let lo = Vec3::new(x_hi - t, y_hi - t, c.z);
        let hi = Vec3::new(x_hi, y_hi, c.z);
        if box_sphere_d2(lo, hi, c) > reach * reach {
            continue;
        }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut h);
        ts.handle.id().hash(&mut h);
        adt_tiles.contains(&ts.handle).hash(&mut h);
        sum = sum.wrapping_add(h.finish());
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_sphere_distance_is_zero_inside_and_grows_outside() {
        let (lo, hi) = (Vec3::ZERO, Vec3::ONE);
        assert_eq!(box_sphere_d2(lo, hi, Vec3::splat(0.5)), 0.0);
        assert!((box_sphere_d2(lo, hi, Vec3::new(3.0, 0.5, 0.5)) - 4.0).abs() < 1e-6);
    }
}
