#define_import_path benilla::monkey_frame

// MONKEY (p0 MonkeyFrame): the graphics programme's per-frame block, 17 rows (272 B) appended to
// the shared light buffer right after the point-light table (`lighting::global_light`, packed from
// the `MonkeyFrame` resource in `lighting/monkey_frame.rs`). Every receiver's `WowLight` mirror
// declares it as `monkey: monkey_frame::MonkeyFrame` after `points`, so the offsets of everything
// before it never move. All zero = no visual change: every reader treats zero as "feature off".
//
// Coordinates are Bevy world space (Y up, 1 unit = 1 yd), the space of the receivers'
// `world_position`: WoW `(x, y, z)` is Bevy `(-y, z, -x)` (`benilla_assets::coords::wow_to_bevy`).
// Member names carry no trailing digit: naga_oil refuses composable-module identifiers that its
// namer would rewrite, so the rows are `fog_a..fog_d`, `wind_a/b`, `wet_a`.
struct MonkeyFrame {
    fog_a: vec4<f32>,  // height_fog_density, height_fog_height (world Y, yd), height_fog_falloff, curve_blend
    fog_b: vec4<f32>,  // sun_fog_rgb (gamma 0..1), sun_fog_strength
    fog_c: vec4<f32>,  // end_fog_rgb (gamma 0..1), end_fog_distance (yd)
    fog_d: vec4<f32>,  // 0 classic or scene fog end for Modern, sun_fog_angle, sun direction octahedral xy
    wind_a: vec4<f32>, // dir_x, dir_y (unit, world XZ: .x = world x, .y = world z), base_heading (rad, fixed profile heading), gust 0..1
    wind_b: vec4<f32>, // travel (yd, integrated speed, wrapped at 4096), sway_strength, grass_strength, tree_strength
    wet_a: vec4<f32>,  // rain_rate 0..1, wetness 0..1, ripple_time_s, snow 0..1
    misc: vec4<f32>,  // bender_count, time_of_day 0..1 (packer), night 0..1 (packer), moon shadow confidence 0..1
    benders: array<vec4<f32>, 8>, // world xyz (Bevy space) + radius (yd); the first bender_count are live
    // GFX (moonlight): unit direction toward the moon (Bevy space) + intensity (gamma; 0 = off).
    moon: vec4<f32>,
}
