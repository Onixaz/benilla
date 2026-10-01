//! MONKEY (skybox): the animated half of the skybox lane. A skybox draws camera-anchored at the far
//! depth ([`crate::skybox`]), so it cannot ride the joint-entity rig: its anchor is written after
//! transform propagation. Its bones are posed here on the CPU instead, each frame, from the armed
//! sequence (animation id 0) and the global sequences, as `0x714260` composes them:
//! `M = parent · T(pivot) · T(t) · R(q) · S(s) · T(−pivot)`. A batch wholly on one bone takes that
//! bone's matrix as its transform; a batch spread over several moving bones is skinned on the CPU
//! (skyboxes are a few thousand vertices).
//!
//! Its texture-transform, colour and alpha tracks run on the same clock through the shared
//! mat-anim table ([`crate::mat_anim_table`]), in rows this module owns, so a skybox flagged
//! `0x1` (sequence 0 across the game day) drives every channel from the day fraction.

use std::sync::Arc;

use bevy::math::Affine3A;
use bevy::prelude::*;

use benilla_assets::coords::{wow_rotation_to_bevy, wow_to_bevy};

/// A key list for one channel: `(seconds, value)` time-ascending, and whether it interpolates.
#[derive(Clone, Debug, Default)]
struct Channel<V> {
    keys: Vec<(f32, V)>,
    linear: bool,
}

impl<V: Copy> Channel<V> {
    /// The value at `t` (seconds inside the channel's own span): held before the first key and past
    /// the last, stepped or blended between, as the reference's key search clamps.
    fn sample(&self, t: f32, blend: impl Fn(V, V, f32) -> V) -> Option<V> {
        let (&(t0, v0), rest) = self.keys.split_first()?;
        if rest.is_empty() || t <= t0 {
            return Some(v0);
        }
        let mut k = 0;
        while k + 1 < self.keys.len() && self.keys[k + 1].0 <= t {
            k += 1;
        }
        let (ta, va) = self.keys[k];
        if !self.linear || k + 1 >= self.keys.len() {
            return Some(va);
        }
        let (tb, vb) = self.keys[k + 1];
        let f = if tb > ta { (t - ta) / (tb - ta) } else { 0.0 };
        Some(blend(va, vb, f))
    }
}

/// A global-sequence channel: keys in seconds over `period`, sampled at the free clock mod it.
#[derive(Clone, Debug, Default)]
struct GlobalChannel<V> {
    period: f32,
    channel: Channel<V>,
}

/// One bone's channels, Bevy space. A global-sequence channel replaces the sequence one.
#[derive(Clone, Debug, Default)]
struct BoneTracks {
    translation: Channel<Vec3>,
    rotation: Channel<Quat>,
    scale: Channel<Vec3>,
    g_translation: Option<GlobalChannel<Vec3>>,
    g_rotation: Option<GlobalChannel<Quat>>,
    g_scale: Option<GlobalChannel<Vec3>>,
}

impl BoneTracks {
    fn moves(&self) -> bool {
        !self.translation.keys.is_empty()
            || !self.rotation.keys.is_empty()
            || !self.scale.keys.is_empty()
            || self.g_translation.is_some()
            || self.g_rotation.is_some()
            || self.g_scale.is_some()
    }
}

/// A skybox model's rig: the bone tree and the armed sequence's tracks, Bevy space.
#[derive(Debug, Default)]
pub(crate) struct SkyRig {
    parents: Vec<i16>,
    pivots: Vec<Vec3>,
    tracks: Vec<BoneTracks>,
    /// The armed sequence's length in seconds and its file slot (the alpha bake's slot index).
    pub(crate) duration: f32,
    pub(crate) seq_slot: Option<usize>,
    looping: bool,
}

fn lerp3(a: Vec3, b: Vec3, f: f32) -> Vec3 {
    a.lerp(b, f)
}

fn slerp(a: Quat, b: Quat, f: f32) -> Quat {
    a.slerp(b, f)
}

/// The interpolation flag of the `M2Track` at `track` (`interp_type != 0`), linear when unreadable.
fn track_linear(bytes: &[u8], track: usize) -> bool {
    bytes
        .get(track..track + 2)
        .is_none_or(|b| u16::from_le_bytes([b[0], b[1]]) != 0)
}

/// WoW-space scale to Bevy space: [`wow_to_bevy`] permutes the axes, so a scale permutes too.
fn scale_to_bevy(s: [f32; 3]) -> Vec3 {
    Vec3::new(s[1], s[2], s[0])
}

impl SkyRig {
    /// Parse the rig out of the model bytes; an unparsable or boneless model has an empty rig.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Self {
        let Ok(skeleton) = benilla_formats::parse_m2_skeleton(bytes) else {
            return Self::default();
        };
        let n = skeleton.bones.len();
        let mut rig = SkyRig {
            parents: skeleton.bones.iter().map(|b| b.parent).collect(),
            pivots: skeleton
                .bones
                .iter()
                .map(|b| wow_to_bevy(b.pivot))
                .collect(),
            tracks: vec![BoneTracks::default(); n],
            duration: 0.0,
            seq_slot: None,
            looping: true,
        };
        let bone_ofs = bytes
            .get(0x38..0x3c)
            .map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize);
        let rec = |i: usize| bone_ofs + i * 0x6c;
        // The armed sequence: animation id 0 (`0x70ebd0`), else the file's first.
        let anims = benilla_formats::parse_m2_animations(bytes);
        if let Some(seq) = anims
            .iter()
            .find(|a| a.anim_id == 0)
            .or_else(|| anims.first())
        {
            rig.duration = seq.duration;
            rig.seq_slot = Some(seq.seq_index);
            rig.looping = seq.looping;
            for keys in &seq.bones {
                let i = usize::from(keys.bone);
                let Some(t) = rig.tracks.get_mut(i) else {
                    continue;
                };
                t.translation = Channel {
                    keys: keys
                        .translation
                        .iter()
                        .map(|&(s, v)| (s, wow_to_bevy(v)))
                        .collect(),
                    linear: track_linear(bytes, rec(i) + 0x0c),
                };
                t.rotation = Channel {
                    keys: keys
                        .rotation
                        .iter()
                        .map(|&(s, q)| (s, wow_rotation_to_bevy(q)))
                        .collect(),
                    linear: track_linear(bytes, rec(i) + 0x28),
                };
                t.scale = Channel {
                    keys: keys
                        .scale
                        .iter()
                        .map(|&(s, v)| (s, scale_to_bevy(v)))
                        .collect(),
                    linear: track_linear(bytes, rec(i) + 0x44),
                };
            }
        }
        fn global<T, V>(
            c: &Option<benilla_formats::GlobalSeqChannel<T>>,
            linear: bool,
            conv: impl Fn(T) -> V,
        ) -> Option<GlobalChannel<V>>
        where
            T: Copy,
        {
            let c = c.as_ref()?;
            Some(GlobalChannel {
                period: c.period_ms as f32 / 1000.0,
                channel: Channel {
                    keys: c
                        .keys
                        .iter()
                        .map(|&(ms, v)| (ms as f32 / 1000.0, conv(v)))
                        .collect(),
                    linear,
                },
            })
        }
        for g in benilla_formats::parse_m2_global_sequence_bones(bytes) {
            let i = usize::from(g.bone);
            let Some(t) = rig.tracks.get_mut(i) else {
                continue;
            };
            t.g_translation = global(&g.translation, track_linear(bytes, rec(i) + 0x0c), |v| {
                wow_to_bevy(v)
            });
            t.g_rotation = global(&g.rotation, track_linear(bytes, rec(i) + 0x28), |q| {
                wow_rotation_to_bevy(q)
            });
            t.g_scale = global(&g.scale, track_linear(bytes, rec(i) + 0x44), scale_to_bevy);
        }
        rig
    }

    /// Whether any bone has a channel, i.e. posing can move a vertex.
    pub(crate) fn animates(&self) -> bool {
        self.tracks.iter().any(BoneTracks::moves)
    }

    /// Whether `bone` or any bone above it has a channel.
    pub(crate) fn bone_moves(&self, bone: u16) -> bool {
        let mut i = usize::from(bone);
        for _ in 0..=self.parents.len() {
            let Some(t) = self.tracks.get(i) else {
                return false;
            };
            if t.moves() {
                return true;
            }
            match usize::try_from(self.parents[i]) {
                Ok(p) => i = p,
                Err(_) => return false,
            }
        }
        false
    }

    /// The sequence clock for `t` seconds on the scene or day clock: wrapped for a looping
    /// sequence, clamped for a one-shot, as the band bake does.
    pub(crate) fn band_time(&self, t: f32) -> f32 {
        if self.duration <= 0.0 {
            0.0
        } else if self.looping {
            t.rem_euclid(self.duration)
        } else {
            t.clamp(0.0, self.duration)
        }
    }

    /// Every bone's model-space matrix at band time `band_t` and free clock `gseq_now` seconds.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn pose(&self, band_t: f32, gseq_now: f64) -> Vec<Affine3A> {
        let mut out = Vec::new();
        self.pose_into(band_t, gseq_now, &mut Vec::new(), &mut out);
        out
    }

    /// MONKEY (visualfix): [`Self::pose`] into reused buffers (`scratch` is the resolve memo), so
    /// the per-frame skybox animation allocates nothing once warm.
    pub(crate) fn pose_into(
        &self,
        band_t: f32,
        gseq_now: f64,
        scratch: &mut Vec<Option<Affine3A>>,
        out: &mut Vec<Affine3A>,
    ) {
        let n = self.parents.len();
        scratch.clear();
        scratch.resize(n, None);
        for i in 0..n {
            self.resolve(i, band_t, gseq_now, scratch, 0);
        }
        out.clear();
        out.extend(scratch.iter().map(|m| m.unwrap_or(Affine3A::IDENTITY)));
    }

    fn resolve(
        &self,
        i: usize,
        band_t: f32,
        gseq_now: f64,
        out: &mut Vec<Option<Affine3A>>,
        depth: usize,
    ) -> Affine3A {
        if let Some(m) = out[i] {
            return m;
        }
        let local = self.local(i, band_t, gseq_now);
        // Bounded by the bone count: the guard against a malformed parent cycle.
        let m = match usize::try_from(self.parents[i]) {
            Ok(p) if p < out.len() && p != i && depth < out.len() => {
                self.resolve(p, band_t, gseq_now, out, depth + 1) * local
            }
            _ => local,
        };
        out[i] = Some(m);
        m
    }

    fn local(&self, i: usize, band_t: f32, gseq_now: f64) -> Affine3A {
        let t = &self.tracks[i];
        if !t.moves() {
            return Affine3A::IDENTITY;
        }
        fn at<V: Copy>(
            g: &Option<GlobalChannel<V>>,
            c: &Channel<V>,
            band_t: f32,
            gseq_now: f64,
            blend: impl Fn(V, V, f32) -> V,
        ) -> Option<V> {
            match g {
                Some(g) if g.period > 0.0 => {
                    let tt = (gseq_now % f64::from(g.period)) as f32;
                    g.channel.sample(tt, blend)
                }
                _ => c.sample(band_t, blend),
            }
        }
        let tr =
            at(&t.g_translation, &t.translation, band_t, gseq_now, lerp3).unwrap_or(Vec3::ZERO);
        let rot = at(&t.g_rotation, &t.rotation, band_t, gseq_now, slerp)
            .unwrap_or(Quat::IDENTITY)
            .normalize();
        let sc = at(&t.g_scale, &t.scale, band_t, gseq_now, lerp3).unwrap_or(Vec3::ONE);
        let pivot = self.pivots[i];
        Affine3A::from_translation(pivot + tr)
            * Affine3A::from_scale_rotation_translation(sc, rot, Vec3::ZERO)
            * Affine3A::from_translation(-pivot)
    }
}

/// The mat-anim rows a skybox batch owns: its texture transform (translation row, and an affine
/// row for rotation/scale) and its colour track, each seeded at `t = 0` like the shared lane.
#[derive(Default)]
pub(crate) struct SkyMatLane {
    pub(crate) uv: Option<(u16, [f32; 2], Arc<benilla_formats::UvAnim>)>,
    pub(crate) affine: Option<(u16, AffineLoops)>,
    pub(crate) tint: Option<(u16, [f32; 3], Arc<benilla_formats::RgbAnim>)>,
    /// Stage 1's translation row (absolute, no seed) and affine row.
    pub(crate) stage1_uv: Option<(u16, benilla_formats::UvAnim)>,
    pub(crate) stage1_affine: Option<(u16, AffineLoops)>,
}

/// The rotation and scaling channels of a texture transform, slot 0's loops.
pub(crate) struct AffineLoops {
    pub(crate) rot: Option<benilla_formats::KeyAnim<[f32; 4]>>,
    pub(crate) scale: Option<benilla_formats::KeyAnim<[f32; 2]>>,
}

impl SkyMatLane {
    /// Allocate the rows for a batch and write their slots into every material it draws with
    /// (`anim_slots.x` UV, `.y` tint, `.z` affine). A full table leaves the batch at its seed.
    pub(crate) fn register(
        sub: &benilla_formats::RenderSubmesh,
        uv: Option<Arc<benilla_formats::UvAnim>>,
        tint: Option<Arc<benilla_formats::RgbAnim>>,
        table: &mut crate::mat_anim_table::MatAnimTable,
        materials: &mut Assets<benilla_assets::materials::WowModelMaterial>,
        mats: &mut [Handle<benilla_assets::materials::WowModelMaterial>],
    ) -> Self {
        // MONKEY (reviewfix): `model_material` is a cache. Animation slots are per sky batch, so
        // detach its handles before writing any slot (notably affine-only `.z`) into the material.
        let originals = mats.to_vec();
        for i in 0..mats.len() {
            if let Some(prior) = (0..i).find(|&j| originals[j].id() == originals[i].id()) {
                mats[i] = mats[prior].clone();
                continue;
            }
            if let Some(material) =
                crate::model_render::lazy::with_material_mut(materials, originals[i].id(), |m| {
                    m.clone()
                })
            {
                mats[i] = materials.add(material);
            }
        }
        let mut lane = SkyMatLane::default();
        let write =
            |materials: &mut Assets<benilla_assets::materials::WowModelMaterial>,
             f: &dyn Fn(&mut benilla_assets::materials::WowModelMaterial)| {
                for id in mats.iter() {
                    crate::model_render::lazy::with_material_mut(materials, id.id(), |m| f(m));
                }
            };
        if let Some(uv) = uv.filter(|a| a.period > 0.0) {
            if let Some(slot) = table.alloc() {
                let seed = mats
                    .first()
                    .and_then(|handle| {
                        crate::model_render::lazy::with_material_mut(materials, handle.id(), |m| {
                            [m.extension.sun_scale.z, m.extension.sun_scale.w]
                        })
                    })
                    .unwrap_or([0.0, 0.0]);
                write(materials, &|m| m.extension.anim_slots.x = f32::from(slot));
                lane.uv = Some((slot, seed, uv));
            }
        }
        let rot = sub.uv_rot_seq.as_ref().and_then(|s| s.seq(None)).cloned();
        let scale = sub.uv_scale_seq.as_ref().and_then(|s| s.seq(None)).cloned();
        if rot.is_some() || scale.is_some() {
            if let Some(slot) = table.alloc() {
                write(materials, &|m| m.extension.anim_slots.z = f32::from(slot));
                lane.affine = Some((slot, AffineLoops { rot, scale }));
            }
        }
        if let Some(tint) = tint {
            if let Some(slot) = table.alloc() {
                let seed = mats
                    .first()
                    .and_then(|handle| {
                        crate::model_render::lazy::with_material_mut(materials, handle.id(), |m| {
                            [m.extension.tint.x, m.extension.tint.y, m.extension.tint.z]
                        })
                    })
                    .unwrap_or([1.0, 1.0, 1.0]);
                write(materials, &|m| m.extension.anim_slots.y = f32::from(slot));
                lane.tint = Some((slot, seed, tint));
            }
        }
        // Stage 1: rows only where it moves; row 0 is the identity.
        if let Some(st) = &sub.stage1 {
            if let Some(uv) = st.uv_anim.clone() {
                if let Some(slot) = table.alloc() {
                    write(materials, &|m| m.extension.stage1.z = f32::from(slot));
                    lane.stage1_uv = Some((slot, uv));
                }
            }
            if st.uv_rot.is_some() || st.uv_scale.is_some() {
                if let Some(slot) = table.alloc() {
                    write(materials, &|m| m.extension.stage1.w = f32::from(slot));
                    lane.stage1_affine = Some((
                        slot,
                        AffineLoops {
                            rot: st.uv_rot.clone(),
                            scale: st.uv_scale.clone(),
                        },
                    ));
                }
            }
        }
        lane
    }

    pub(crate) fn any(&self) -> bool {
        self.uv.is_some()
            || self.affine.is_some()
            || self.tint.is_some()
            || self.stage1_uv.is_some()
            || self.stage1_affine.is_some()
    }

    /// Write this frame's rows: the delta from each seed, quantized as the shared lane does.
    pub(crate) fn tick(
        &self,
        band_t: f32,
        gseq_now: f64,
        table: &mut crate::mat_anim_table::MatAnimTable,
    ) {
        if let Some((slot, seed, a)) = &self.uv {
            let v = a.sample(a.clock(band_t, gseq_now));
            table.set(
                *slot,
                [
                    benilla_assets::quantize(v[0], 4096.0) - seed[0],
                    benilla_assets::quantize(v[1], 4096.0) - seed[1],
                    0.0,
                    0.0,
                ],
            );
        }
        if let Some((slot, loops)) = &self.affine {
            let q = loops.rot.as_ref().map_or([0.0, 0.0, 0.0, 1.0], |l| {
                l.sample(l.clock(band_t, gseq_now))
            });
            let s = loops
                .scale
                .as_ref()
                .map_or([1.0, 1.0], |l| l.sample(l.clock(band_t, gseq_now)));
            table.set(*slot, crate::mat_anim_table::affine_row(q, s));
        }
        if let Some((slot, a)) = &self.stage1_uv {
            let v = a.sample(a.clock(band_t, gseq_now));
            table.set(
                *slot,
                [
                    benilla_assets::quantize(v[0], 4096.0),
                    benilla_assets::quantize(v[1], 4096.0),
                    0.0,
                    0.0,
                ],
            );
        }
        if let Some((slot, loops)) = &self.stage1_affine {
            let q = loops.rot.as_ref().map_or([0.0, 0.0, 0.0, 1.0], |l| {
                l.sample(l.clock(band_t, gseq_now))
            });
            let s = loops
                .scale
                .as_ref()
                .map_or([1.0, 1.0], |l| l.sample(l.clock(band_t, gseq_now)));
            table.set(*slot, crate::mat_anim_table::affine_row(q, s));
        }
        if let Some((slot, seed, a)) = &self.tint {
            let c = benilla_assets::quant255(a.sample(a.clock(band_t, gseq_now)));
            table.set(*slot, [c[0] - seed[0], c[1] - seed[1], c[2] - seed[2], 0.0]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spin_rig() -> SkyRig {
        // One parentless bone at pivot (1,0,0) turning 135° about +Y over 4 s, linear; the quarter
        // turn falls at 8/3 s.
        SkyRig {
            parents: vec![-1],
            pivots: vec![Vec3::X],
            tracks: vec![BoneTracks {
                rotation: Channel {
                    keys: vec![
                        (0.0, Quat::IDENTITY),
                        (
                            4.0,
                            Quat::from_rotation_y(std::f32::consts::FRAC_PI_2 * 1.5),
                        ),
                    ],
                    linear: true,
                },
                ..Default::default()
            }],
            duration: 4.0,
            seq_slot: Some(0),
            looping: true,
        }
    }

    #[test]
    fn a_bone_turns_about_its_pivot() {
        let rig = spin_rig();
        let m = rig.pose(rig.band_time(4.0 + 8.0 / 3.0), 0.0)[0];
        // A quarter turn about the pivot; the pivot itself stays put.
        assert!(m.transform_point3(Vec3::X).abs_diff_eq(Vec3::X, 1e-5));
        let p = m.transform_point3(Vec3::new(2.0, 0.0, 0.0));
        assert!(p.abs_diff_eq(Vec3::new(1.0, 0.0, -1.0), 1e-5), "{p}");
    }

    #[test]
    fn a_child_composes_onto_its_parent() {
        let mut rig = spin_rig();
        rig.parents.push(0);
        rig.pivots.push(Vec3::new(2.0, 0.0, 0.0));
        rig.tracks.push(BoneTracks {
            translation: Channel {
                keys: vec![(0.0, Vec3::Y)],
                linear: true,
            },
            ..Default::default()
        });
        let pose = rig.pose(8.0 / 3.0, 0.0);
        let p = pose[1].transform_point3(Vec3::new(2.0, 0.0, 0.0));
        // Lifted by the child, then turned a quarter by the parent.
        assert!(p.abs_diff_eq(Vec3::new(1.0, 1.0, -1.0), 1e-5), "{p}");
        assert!(rig.bone_moves(1) && rig.animates());
    }

    #[test]
    fn a_trackless_rig_is_the_bind_pose() {
        let rig = SkyRig {
            parents: vec![-1, 0],
            pivots: vec![Vec3::ZERO, Vec3::ONE],
            tracks: vec![BoneTracks::default(); 2],
            ..Default::default()
        };
        assert!(!rig.animates());
        assert!(rig.pose(1.0, 1.0).iter().all(|m| *m == Affine3A::IDENTITY));
    }

    /// The Caverns of Time belts: the general rig reproduces the rigid spin the old lane drew.
    #[test]
    fn the_caverns_belts_spin_as_the_bone_spin_does() {
        let data = benilla_formats::wow_data_or_skip!();
        let mut chain = benilla_formats::Chain::open(&data).expect("open chain");
        const SKY: &str = "Environments\\Stars\\CavernsOfTimeSky.m2";
        let bytes = chain.read_file(SKY).expect("read the sky");
        let rig = SkyRig::from_bytes(&bytes);
        let spins = benilla_formats::m2_bone_spins(&bytes);
        assert!(!spins.is_empty());
        for t in [0.0f32, 3.3, 17.0, 50.0] {
            let pose = rig.pose(rig.band_time(t), f64::from(t));
            for (bone, spin) in &spins {
                let pivot = wow_to_bevy(spin.pivot);
                let rot = wow_rotation_to_bevy(spin.sample(t));
                let old = Affine3A::from_translation(pivot)
                    * Affine3A::from_quat(rot)
                    * Affine3A::from_translation(-pivot);
                let p = Vec3::new(10.0, 3.0, -7.0) + pivot;
                let a = pose[usize::from(*bone)].transform_point3(p);
                let b = old.transform_point3(p);
                assert!(a.abs_diff_eq(b, 1e-2), "bone {bone} t {t}: {a} vs {b}");
            }
        }
    }
}
