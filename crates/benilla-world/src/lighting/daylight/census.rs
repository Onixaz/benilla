//! MONKEY (daylight): the city daylight census (test-only instrument). Loads a WMO root from the
//! install, runs the live selection rule over it and prints which interior rooms get a daylight
//! seed, which do not, and what each unseeded room authors (batch classes, WINDOW / SIDN
//! materials, portal neighbours), plus the cost of each seed pass.
//!
//! `WOW_DATA=<Data> cargo test -p benilla-world --lib lighting::daylight::census -- --ignored --nocapture`
//! `WOW_CENSUS_WMO=<internal path>` picks the root (default: Stormwind then Ironforge).

use super::*;
use benilla_formats::{open_chain, parse_wmo_portals, parse_wmo_root, wmo_group_header};

struct Loaded {
    groups: Vec<WmoGroupInfo>,
    names: Vec<String>,
    portals: benilla_formats::WmoPortals,
    slices: Vec<(u16, u16)>,
    /// `(group, class, window, sidn, positions)` per render batch.
    batches: Vec<(
        u16,
        benilla_formats::WmoBatchClass,
        bool,
        bool,
        Vec<[f32; 3]>,
    )>,
}

fn group_names(bytes: &[u8]) -> Vec<String> {
    let (mut mogn, mut mogi) = (None, None);
    let mut o = 0usize;
    while o + 8 <= bytes.len() {
        let tag = [bytes[o + 3], bytes[o + 2], bytes[o + 1], bytes[o]];
        let n = u32::from_le_bytes(bytes[o + 4..o + 8].try_into().unwrap()) as usize;
        let body = bytes.get(o + 8..o + 8 + n);
        match (&tag, body) {
            (b"MOGN", Some(b)) => mogn = Some(b),
            (b"MOGI", Some(b)) => mogi = Some(b),
            _ => {}
        }
        o += 8 + n;
    }
    let (Some(mogn), Some(mogi)) = (mogn, mogi) else {
        return Vec::new();
    };
    mogi.chunks_exact(32)
        .map(|r| {
            let off = i32::from_le_bytes(r[28..32].try_into().unwrap());
            usize::try_from(off)
                .ok()
                .and_then(|off| mogn.get(off..))
                .map(|rest| {
                    let end = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
                    String::from_utf8_lossy(&rest[..end]).into_owned()
                })
                .unwrap_or_default()
        })
        .collect()
}

fn load(path: &str) -> Loaded {
    let data = benilla_formats::wow_data().expect("no WoW install found (set $WOW_DATA)");
    let mut chain = open_chain(&data).expect("open MPQ chain");
    let root_path = path.replace('/', "\\").to_ascii_lowercase();
    let bytes = chain.read_file(&root_path).expect("read root");
    let root = parse_wmo_root(&bytes).expect("parse root");
    let stem = root_path.strip_suffix(".wmo").unwrap().to_string();
    let mut slices = Vec::new();
    let mut batches = Vec::new();
    for gi in 0..root.group_count() {
        let Ok(gbytes) = chain.read_file(&format!("{stem}_{gi:03}.wmo")) else {
            slices.push((0, 0));
            continue;
        };
        let h = wmo_group_header(&gbytes);
        slices.push(h.map_or((0, 0), |h| (h.portal_ref_start, h.portal_ref_count)));
        for s in benilla_formats::wmo_group_submeshes(&gbytes, &root).unwrap_or_default() {
            if let Some(class) = s.wmo_batch {
                batches.push((gi as u16, class, s.window, s.sidn.is_some(), s.positions));
            }
        }
    }
    Loaded {
        groups: root.group_infos().to_vec(),
        names: group_names(&bytes),
        portals: parse_wmo_portals(&bytes),
        slices,
        batches,
    }
}

fn census(path: &str) {
    let l = load(path);
    let graph = PortalGraph {
        vertices: &l.portals.vertices,
        infos: &l.portals.infos,
        refs: &l.portals.refs,
        slices: &l.slices,
    };
    let feed: Vec<(u16, bool, &[[f32; 3]])> = l
        .batches
        .iter()
        .map(|(g, c, _, _, p)| (*g, matches!(c, benilla_formats::WmoBatchClass::Ext), &p[..]))
        .collect();
    let verts: usize = feed.iter().map(|b| b.2.len()).sum();
    let rooms = l.groups.iter().filter(|g| g.interior).count();
    println!(
        "=== {path}: {} groups, {rooms} interior, {} batches, {verts} batch verts",
        l.groups.len(),
        feed.len()
    );

    let t = std::time::Instant::now();
    let ranked = daylight_seeds_ranked(&l.groups, graph, feed.iter().copied());
    let t_ranked = t.elapsed();
    let t = std::time::Instant::now();
    let bounds = boundary_clusters(&l.groups, &feed);
    let t_bound = t.elapsed();
    let (day, bleed) = placement_openings(&l.groups, graph, feed.iter().copied());
    println!(
        "live ranked seeds {} ({:.1} ms); boundary clusters if forced {} ({:.1} ms); kept {} day + {} bleed of budget {}",
        ranked.len(),
        t_ranked.as_secs_f64() * 1e3,
        bounds.len(),
        t_bound.as_secs_f64() * 1e3,
        day.len(),
        bleed.len(),
        daylight_budget(&l.groups),
    );
    let mut by_how: HashMap<&str, usize> = HashMap::new();
    for s in &day {
        *by_how.entry(s.how.tag()).or_default() += 1;
    }
    println!("kept daylight by rule: {by_how:?}");
    // Which rooms each lane actually REACHES: the fixture's own claim set (containment, portal hop,
    // split), the same set the packer's room gate enforces.
    let mut day_claim: HashMap<u16, Vec<String>> = HashMap::new();
    for s in &day {
        for c in daylight_claims(&l.groups, graph, s) {
            day_claim.entry(c.group).or_default().push(format!(
                "{}@g{}{:?}",
                s.how.tag(),
                s.group,
                c.how
            ));
        }
    }
    let mut bleed_claim: HashSet<u16> = HashSet::new();
    for s in &bleed {
        for c in s.claims(&l.groups, graph) {
            bleed_claim.insert(c.group);
        }
    }
    let bound_groups: HashSet<u16> = bounds.iter().map(|b| b.0).collect();
    let place = placement();
    let (mut n_day, mut n_bleed, mut n_none) = (0, 0, 0);
    for (gi, g) in l.groups.iter().enumerate() {
        if !g.interior {
            continue;
        }
        let gid = gi as u16;
        let tag = if day_claim.contains_key(&gid) {
            n_day += 1;
            "DAY  "
        } else if bleed_claim.contains(&gid) {
            n_bleed += 1;
            "bleed"
        } else {
            n_none += 1;
            "NONE "
        };
        let (start, count) = l.slices[gi];
        let mut nb = Vec::new();
        for r in l
            .portals
            .refs
            .iter()
            .skip(start as usize)
            .take(count as usize)
        {
            let other =
                l.groups
                    .get(r.group as usize)
                    .map_or('?', |o| if o.interior { 'i' } else { 'e' });
            let area = benilla_formats::room_claim::portal_area(&graph, r.portal).unwrap_or(0.0);
            nb.push(format!("p{}->g{}{}({:.0})", r.portal, r.group, other, area));
        }
        let (mut e, mut w) = (0, 0);
        for (bg, c, win, _, _) in &l.batches {
            if usize::from(*bg) == gi {
                e += usize::from(matches!(c, benilla_formats::WmoBatchClass::Ext));
                w += usize::from(*win);
            }
        }
        let c = [
            0.5 * (g.bbox_min[0] + g.bbox_max[0]),
            0.5 * (g.bbox_min[1] + g.bbox_max[1]),
            0.5 * (g.bbox_min[2] + g.bbox_max[2]),
        ];
        let wc = place.map_or(c, |m| {
            benilla_assets::coords::bevy_to_wow(
                m.transform_point3(benilla_assets::coords::wow_to_bevy(c)),
            )
        });
        println!(
            "  {tag} g{gi:<3} {:<18} at ({:8.1},{:7.1},{:6.1}) size ({:5.1},{:5.1},{:5.1}) E{e} W{w}{} {}  {}",
            l.names.get(gi).map(String::as_str).unwrap_or("-"),
            wc[0], wc[1], wc[2],
            g.bbox_max[0] - g.bbox_min[0],
            g.bbox_max[1] - g.bbox_min[1],
            g.bbox_max[2] - g.bbox_min[2],
            if bound_groups.contains(&gid) { " stitch" } else { "" },
            nb.join(" "),
            day_claim.get(&gid).map(|v| v.iter().take(3).cloned().collect::<Vec<_>>().join(",")).unwrap_or_default(),
        );
    }
    println!(
        "interior rooms: {n_day} reached by daylight, {n_bleed} by bleed only, {n_none} by neither"
    );
    let sky = district_sky_rooms(&l.groups, graph, feed.iter().copied());
    let enclosed = (0..l.groups.len() as u16)
        .filter(|g| benilla_formats::room_claim::enclosed_by_building_shell(&l.groups, *g))
        .count();
    let dark_sky = (0..l.groups.len())
        .filter(|g| sky.get(*g).copied().unwrap_or(false))
        .filter(|g| !day_claim.contains_key(&(*g as u16)))
        .count();
    println!(
        "day floor: {enclosed} rooms enclosed by a building shell; {} district sky rooms ({dark_sky} of them reached by no daylight fixture)",
        sky.iter().filter(|b| **b).count()
    );
}

/// `WOW_CENSUS_PLACE=map,tx,ty,uid`: the MODF placement, for world coordinates in the table.
fn placement() -> Option<bevy::math::Affine3A> {
    let spec = std::env::var("WOW_CENSUS_PLACE").ok()?;
    let p: Vec<&str> = spec.split(',').collect();
    let (map, tx, ty, uid) = (
        p[0],
        p[1].parse().ok()?,
        p[2].parse().ok()?,
        p[3].parse::<u32>().ok()?,
    );
    let data = benilla_formats::wow_data()?;
    let mut chain = open_chain(&data).ok()?;
    let tile = benilla_formats::load_tile_mesh(&mut chain, map, tx, ty).ok()?;
    let w = tile.wmos.iter().find(|w| w.unique_id == uid)?;
    println!(
        "placement uid {uid} pos {:?} rot {:?}",
        w.position, w.rotation
    );
    Some(bevy::math::Affine3A::from_scale_rotation_translation(
        Vec3::ONE,
        benilla_assets::coords::placement_rotation(w.rotation),
        benilla_assets::coords::wow_to_bevy(w.position),
    ))
}

#[test]
#[ignore = "instrument: needs WOW_DATA; run by hand with --ignored --nocapture"]
fn city_daylight_census() {
    match std::env::var("WOW_CENSUS_WMO") {
        Ok(p) => census(&p),
        Err(_) => {
            census(r"World\wmo\Azeroth\Buildings\Stormwind\Stormwind.wmo");
            census(r"World\wmo\KhazModan\Cities\Ironforge\ironforge.wmo");
        }
    }
}

/// MONKEY (daylight: terrain torch casters): which lampposts have ground they cannot see — the
/// share of terrain points within 25 yd whose line to the lamp head (3.5 yd up) is blocked by the
/// terrain itself. Picks a vantage for the B1 capture. `WOW_LAMP_SCAN=map,x,y,radius,filter`.
#[test]
#[ignore = "instrument: needs WOW_DATA; run by hand with --ignored --nocapture"]
fn lamp_terrain_occlusion_scan() {
    let spec =
        std::env::var("WOW_LAMP_SCAN").unwrap_or_else(|_| "Azeroth,-9300,150,1,lamppost".into());
    let p: Vec<&str> = spec.split(',').collect();
    let (map, cx, cy, r, filter) = (
        p[0],
        p[1].parse::<f32>().unwrap(),
        p[2].parse::<f32>().unwrap(),
        p[3].parse::<i32>().unwrap(),
        p[4].to_ascii_lowercase(),
    );
    let data = benilla_formats::wow_data().expect("WOW_DATA");
    let mut chain = open_chain(&data).expect("chain");
    let (tx, ty) = benilla_formats::world_to_tile(cx, cy);
    let mut chunks = Vec::new();
    let mut lamps = Vec::new();
    for x in tx as i32 - r..=tx as i32 + r {
        for y in ty as i32 - r..=ty as i32 + r {
            let Ok(t) = benilla_formats::load_tile_mesh(&mut chain, map, x as u32, y as u32) else {
                continue;
            };
            for d in &t.doodads {
                if d.model.to_ascii_lowercase().contains(&filter) {
                    lamps.push(d.position);
                }
            }
            chunks.push(t.chunks);
        }
    }
    let h = |x: f32, y: f32| {
        chunks
            .iter()
            .find_map(|c| benilla_formats::terrain_height_at(c, [x, y, 1.0e4]))
    };
    let mut rows = Vec::new();
    // MONKEY (leftovers): optional 6th field = light height above the doodad origin (default the
    // lamppost's 3.5 yd; a campfire's flame sits near 1 yd).
    let lift = p.get(5).and_then(|v| v.parse::<f32>().ok()).unwrap_or(3.5);
    // 7th field = how far the ground must rise above the ray to count (default 5 cm; the receiver's
    // normal offset is 0.15 yd, so a grazing 5 cm crest is below what a cube map can resolve).
    let clear = p.get(6).and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.05);
    for l in lamps {
        let head = Vec3::new(l[0], l[1], l[2] + lift);
        let (mut total, mut blocked) = (0u32, 0u32);
        let mut best = (0.0f32, 0.0f32);
        // MONKEY (leftovers): blocked samples inside 15 yd, where a fire's pool is still visible.
        let mut near = 0u32;
        for ring in 1..=10 {
            let d = ring as f32 * 2.5;
            for k in 0..36 {
                let a = k as f32 * std::f32::consts::TAU / 36.0;
                let (x, y) = (l[0] + d * a.cos(), l[1] + d * a.sin());
                let Some(z) = h(x, y) else { continue };
                total += 1;
                let target = Vec3::new(x, y, z + 0.3);
                let steps = (d / 0.5) as i32;
                let hit = (1..steps).any(|s| {
                    let q = head.lerp(target, s as f32 / steps as f32);
                    h(q.x, q.y).is_some_and(|g| g > q.z + clear)
                });
                if hit {
                    blocked += 1;
                    best = (a.to_degrees(), d);
                    near += (d <= 15.0) as u32;
                }
            }
        }
        if total > 0 {
            rows.push((
                blocked as f32 / total as f32,
                l,
                best,
                near,
                h(l[0], l[1]).unwrap_or(f32::NAN),
            ));
        }
    }
    rows.sort_by(|a, b| b.3.cmp(&a.3).then(b.0.total_cmp(&a.0)));
    for (f, l, (a, d), near, g) in rows.iter().take(12) {
        println!(
            "lamp ({:8.1},{:7.1},{:6.1}) ground {g:6.2}  terrain-blocked {:5.1}% ({near} within 15 yd)  e.g. bearing {a:5.0} deg at {d:4.1} yd",
            l[0], l[1], l[2], f * 100.0
        );
    }
}
