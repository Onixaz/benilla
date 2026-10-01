//! MONKEY (water LOD): High-quality liquid uses a denser near-camera lattice.
//!
//! Wgpu has no tessellation stage.  The coarse grid therefore remains the streamed source of
//! truth, while a 4x bilinear copy is built only when the camera enters the surface's near ring.
//! Every original edge vertex survives bit-for-bit. Non-corner fine vertices on an outer edge carry
//! a compact stitch tag in UV1.y, so the vertex shader lerps the displaced coarse endpoints instead
//! of evaluating a new Gerstner point that can leave the neighbour's straight edge. Hysteresis
//! avoids rebuilding while the camera hovers at the threshold.
//! A full MCLQ copy is 1,089 vertices + 6,144 indices (about 68 KiB without allocator overhead);
//! the 64 yd ring bounds ordinary terrain residency to roughly 3.5 MiB, and a per-surface ceiling
//! refuses pathological MLIQ grids rather than allowing one pool to consume an unbounded copy.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use bevy::prelude::*;

use benilla_assets::{coords::wow_to_bevy, WaterQuality};
use benilla_formats::LiquidMesh;

use crate::view::WorldCamera;

const SUBDIVISIONS: usize = 4;
const ENTER_YARDS: f32 = 64.0;
const EXIT_YARDS: f32 = 80.0;
const MAX_FINE_VERTICES: usize = 65_536;

/// The coarse asset is retained for the far swap. Only the tiny coverage mask/grid metadata is
/// duplicated per surface; source vertex attributes remain in `Assets<Mesh>` and are copied only
/// during an actual near-ring transition.
#[derive(Component)]
pub(super) struct LiquidLod {
    coarse: Handle<Mesh>,
    fine: Option<Handle<Mesh>>,
    grid: [u32; 2],
    wet: Vec<bool>,
    local_center: Vec3,
    local_radius_xz: f32,
}

impl LiquidLod {
    pub(super) fn new(coarse: Handle<Mesh>, source: &LiquidMesh) -> Self {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for &position in &source.positions {
            let p = wow_to_bevy(position);
            min = min.min(p);
            max = max.max(p);
        }
        let valid = min.is_finite() && max.is_finite();
        let local_center = if valid { (min + max) * 0.5 } else { Vec3::ZERO };
        let local_radius_xz = if valid {
            Vec2::new(max.x - min.x, max.z - min.z).length() * 0.5
        } else {
            0.0
        };
        Self {
            coarse,
            fine: None,
            grid: source.grid,
            wet: source.wet.clone(),
            local_center,
            local_radius_xz,
        }
    }
}

pub(super) fn register(app: &mut App) {
    app.add_systems(Update, update_liquid_lod);
}

fn update_liquid_lod(
    quality: Res<WaterQuality>,
    camera: Query<&GlobalTransform, With<WorldCamera>>,
    mut surfaces: Query<(&GlobalTransform, &mut Mesh3d, &mut LiquidLod)>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let Ok(camera) = camera.single() else { return };
    let camera = camera.translation().xz();

    for (transform, mut current, mut lod) in &mut surfaces {
        if quality.0 < 2 {
            if let Some(fine) = lod.fine.take() {
                current.0 = lod.coarse.clone();
                meshes.remove(fine.id());
            }
            continue;
        }

        let centre = transform.transform_point(lod.local_center).xz();
        let (scale, _, _) = transform.to_scale_rotation_translation();
        let world_radius = lod.local_radius_xz * scale.abs().max_element();
        let edge_distance = (camera.distance(centre) - world_radius).max(0.0);
        if lod.fine.is_some() {
            if edge_distance > EXIT_YARDS {
                let fine = lod.fine.take().expect("checked");
                current.0 = lod.coarse.clone();
                meshes.remove(fine.id());
            }
            continue;
        }
        if edge_distance > ENTER_YARDS {
            continue;
        }

        let Some(fine_mesh) = meshes
            .get(&lod.coarse)
            .and_then(|coarse| subdivide(coarse, lod.grid, &lod.wet))
        else {
            continue;
        };
        let fine = meshes.add(fine_mesh);
        current.0 = fine.clone();
        lod.fine = Some(fine);
    }
}

fn bilerp<const N: usize>(
    values: &[[f32; N]],
    cols: usize,
    x: usize,
    y: usize,
    tx: f32,
    ty: f32,
) -> [f32; N] {
    let [tl, tr, bl, br] = [
        values[y * cols + x],
        values[y * cols + x + 1],
        values[(y + 1) * cols + x],
        values[(y + 1) * cols + x + 1],
    ];
    let mut out = [0.0; N];
    for i in 0..N {
        let top = tl[i] + (tr[i] - tl[i]) * tx;
        let bottom = bl[i] + (br[i] - bl[i]) * tx;
        out[i] = top + (bottom - top) * ty;
    }
    out
}

/// MONKEY (reviewfix): encode an outer-edge fine vertex as `axis/sign + coarse-edge fraction`.
/// UV1.y is otherwise zero. Coarse corners and interior fine vertices need no stitch.
fn stitch_tag(
    positions: &[[f32; 3]],
    cols: usize,
    rows: usize,
    fine_cols: usize,
    fine_rows: usize,
    fx: usize,
    fy: usize,
    sx: usize,
    sy: usize,
    tx: f32,
    ty: f32,
) -> f32 {
    let (a, b, t) = if (fy == 0 || fy + 1 == fine_rows) && !fx.is_multiple_of(SUBDIVISIONS) {
        let y = if fy == 0 { 0 } else { rows - 1 };
        (y * cols + sx, y * cols + sx + 1, tx)
    } else if (fx == 0 || fx + 1 == fine_cols) && !fy.is_multiple_of(SUBDIVISIONS) {
        let x = if fx == 0 { 0 } else { cols - 1 };
        (sy * cols + x, (sy + 1) * cols + x, ty)
    } else {
        return 0.0;
    };
    let dx = positions[b][0] - positions[a][0];
    let dz = positions[b][2] - positions[a][2];
    let lane = if dx.abs() >= dz.abs() {
        if dx >= 0.0 {
            1.0
        } else {
            2.0
        }
    } else if dz >= 0.0 {
        3.0
    } else {
        4.0
    };
    lane + t
}

/// Four-way subdivision of every source cell. The output is still one indexed regular grid: there
/// are no duplicated cell-edge vertices, and dry source cells emit no triangles.
fn subdivide(coarse: &Mesh, grid: [u32; 2], wet: &[bool]) -> Option<Mesh> {
    let [cols, rows] = grid.map(|v| v as usize);
    if cols < 2 || rows < 2 || wet.len() != (cols - 1) * (rows - 1) {
        return None;
    }
    let VertexAttributeValues::Float32x3(positions) = coarse.attribute(Mesh::ATTRIBUTE_POSITION)?
    else {
        return None;
    };
    let VertexAttributeValues::Float32x2(uv0) = coarse.attribute(Mesh::ATTRIBUTE_UV_0)? else {
        return None;
    };
    let VertexAttributeValues::Float32x2(uv1) = coarse.attribute(Mesh::ATTRIBUTE_UV_1)? else {
        return None;
    };
    if positions.len() != cols * rows
        || uv0.len() != positions.len()
        || uv1.len() != positions.len()
    {
        return None;
    }
    let colours = match coarse.attribute(Mesh::ATTRIBUTE_COLOR) {
        Some(VertexAttributeValues::Float32x4(values)) if values.len() == positions.len() => {
            Some(values)
        }
        None => None,
        _ => return None,
    };

    let fine_cols = (cols - 1).checked_mul(SUBDIVISIONS)?.checked_add(1)?;
    let fine_rows = (rows - 1).checked_mul(SUBDIVISIONS)?.checked_add(1)?;
    let fine_count = fine_cols.checked_mul(fine_rows)?;
    if fine_count > MAX_FINE_VERTICES {
        return None;
    }
    let mut fine_positions = Vec::with_capacity(fine_count);
    let mut fine_uv0 = Vec::with_capacity(fine_count);
    let mut fine_uv1 = Vec::with_capacity(fine_count);
    let mut fine_colours = colours.map(|_| Vec::with_capacity(fine_count));

    for fy in 0..fine_rows {
        let sy = (fy / SUBDIVISIONS).min(rows - 2);
        let ty = (fy - sy * SUBDIVISIONS) as f32 / SUBDIVISIONS as f32;
        for fx in 0..fine_cols {
            let sx = (fx / SUBDIVISIONS).min(cols - 2);
            let tx = (fx - sx * SUBDIVISIONS) as f32 / SUBDIVISIONS as f32;
            fine_positions.push(bilerp(positions, cols, sx, sy, tx, ty));
            fine_uv0.push(bilerp(uv0, cols, sx, sy, tx, ty));
            let mut uv1 = bilerp(uv1, cols, sx, sy, tx, ty);
            uv1[1] = stitch_tag(
                positions, cols, rows, fine_cols, fine_rows, fx, fy, sx, sy, tx, ty,
            );
            fine_uv1.push(uv1);
            if let (Some(source), Some(output)) = (colours, fine_colours.as_mut()) {
                output.push(bilerp(source, cols, sx, sy, tx, ty));
            }
        }
    }

    let wet_fine_cells = wet.iter().filter(|&&cell| cell).count() * SUBDIVISIONS * SUBDIVISIONS;
    let mut indices = Vec::with_capacity(wet_fine_cells * 6);
    for y in 0..fine_rows - 1 {
        for x in 0..fine_cols - 1 {
            let source_x = x / SUBDIVISIONS;
            let source_y = y / SUBDIVISIONS;
            if !wet[source_y * (cols - 1) + source_x] {
                continue;
            }
            let tl = (y * fine_cols + x) as u32;
            let tr = tl + 1;
            let bl = ((y + 1) * fine_cols + x) as u32;
            let br = bl + 1;
            indices.extend_from_slice(&[tl, bl, br, tl, br, tr]);
        }
    }

    let mut fine = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    fine.insert_attribute(Mesh::ATTRIBUTE_POSITION, fine_positions);
    fine.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 1.0, 0.0]; fine_count]);
    fine.insert_attribute(Mesh::ATTRIBUTE_UV_0, fine_uv0);
    fine.insert_attribute(Mesh::ATTRIBUTE_UV_1, fine_uv1);
    if let Some(colours) = fine_colours {
        fine.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
    }
    fine.insert_indices(Indices::U32(indices));
    Some(fine)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coarse() -> Mesh {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 1.0, 0.0]; 4]);
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_UV_0,
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, vec![[0.0, 0.0]; 4]);
        mesh
    }

    #[test]
    fn subdivision_preserves_edges_and_splits_every_wet_cell_sixteen_ways() {
        let fine = subdivide(&coarse(), [2, 2], &[true]).expect("fine mesh");
        let Some(VertexAttributeValues::Float32x3(positions)) =
            fine.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("positions");
        };
        assert_eq!(positions.len(), 25);
        assert_eq!(positions[0], [0.0, 0.0, 0.0]);
        assert_eq!(positions[4], [1.0, 0.0, 0.0]);
        assert_eq!(positions[20], [0.0, 0.0, 1.0]);
        assert_eq!(positions[24], [1.0, 1.0, 1.0]);
        assert_eq!(positions[12], [0.5, 0.25, 0.5]);
        let Some(VertexAttributeValues::Float32x2(uv1)) = fine.attribute(Mesh::ATTRIBUTE_UV_1)
        else {
            panic!("uv1");
        };
        assert_eq!(uv1[0][1], 0.0, "coarse corner");
        assert_eq!(uv1[2][1], 1.5, "top-edge midpoint, +x at t=0.5");
        assert_eq!(uv1[10][1], 3.5, "left-edge midpoint, +z at t=0.5");
        assert_eq!(uv1[12][1], 0.0, "interior fine vertex");
        assert_eq!(fine.indices().expect("indices").len(), 16 * 6);
    }

    #[test]
    fn dry_parent_cells_stay_holes() {
        let mut mesh = coarse();
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
            ],
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 1.0, 0.0]; 6]);
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_UV_0,
            vec![
                [0.0, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 1.0],
                [1.0, 1.0],
            ],
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, vec![[0.0, 0.0]; 6]);
        let fine = subdivide(&mesh, [3, 2], &[true, false]).expect("fine mesh");
        assert_eq!(fine.indices().expect("indices").len(), 16 * 6);
    }
}
