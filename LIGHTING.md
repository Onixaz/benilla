# benilla-lighting

A dynamic light and shadow system for [benilla](https://github.com/samwhosung/benilla), the Rust +
Bevy reimplementation of the 1.12.1 client. This branch is upstream benilla (merged at
`fd386e75`) plus the lighting work, the enhanced water module and volumetric fog, and nothing else. It needs no data changes: everything is
derived from what a 1.12 install already carries (WMO `MOLT` lights, `MOCV` vertex colour, portals,
M2 particle emitters, `Light.dbc`).

Everything here is a departure from the reference client's look, so every lane has a live cvar and
most can be switched off. Code is tagged `MONKEY (<topic>)` with the rationale and the measurements
behind each constant.

## What it does

| Lane | What you see | Main switch |
|---|---|---|
| Sun shadows, characters | real silhouettes for units instead of the oval blob | `characterShadows` |
| Sun shadows, world | trees, buildings and alpha-tested foliage cast; baked terrain shadows step aside; the static caster cache also refreshes when streamed geometry arrives, so a standing camera picks up late trees (`world_shadow.rs::refresh_streamed_shadows`, independent of volumetric fog) | `worldShadows` |
| Dynamic interiors | WMO rooms are lit per fragment by their own fixtures with a soft falloff, gated per room so light does not leak through walls | `interiorLight`, `interiorRoomGate` |
| Synthesised lights | torches, braziers, lanterns, campfires and lampposts emit a light in the colour of their flame, with flicker | `fireLightGain`, `fireFlicker` |
| Torch shadows | cube-map shadows from point lights onto buildings, models and terrain, with a contact-hardening penumbra and a shadow floor | `interiorShadows`, `exteriorShadows`, `torchShadowStrength`, `interiorShadowSoft` |
| Daylight and doorways | calibrated fixtures at doors, windows and open boundaries carry daylight into a room, and doorways between rooms bleed light | `interiorDaylight`, env `WOW_DAYLIGHT`, `WOW_BLEED` |
| Terrain in torch shadows | the ground casts into an outdoor fire's cube map, so a hill or bank blocks the fire (settled exterior slots only; `torch_terrain.rs` gathers the resident MCNK chunks inside the cube's 48 yd; the gather and the ground receiver both work, but stock Elwynn lamps/campfires show no visible change: the terrain-blocked ground is grazing (N·L ≈ 0, inside the 0.15 yd normal offset) or at the pool rim, below one 8-bit step — measured by the `lamp_terrain_occlusion_scan` census) | `torchTerrainShadows` (default 0, High 1), env `WOW_TORCH_TERRAIN=0\|1` |
| City day floor | `interiorDaylight`'s room floor also reaches Stormwind/Ironforge rooms that the portal graph connects to the sky (an exterior-facing portal, or one hop from one; `district_sky_rooms` in `daylight.rs`); before this no city room could take it | `interiorDaylight` (default 0, suggested High 0.10) |
| Spell and firework lights | fire, holy and fel effects light their surroundings for their lifetime; frost, nature, arcane and shadow do not | `spellLightGain`, env `WOW_SPELL_LIGHT=0` |
| Moon shadows | at night the same shadow rig re-aims at the moon and casts a faint shadow; dims only the night sky term, never point lights | `moonShadowStrength` |
| Moonlight | GFX (moonlight): the moon as a light. The stock 1.12 night (Light.dbc night colours on the never-setting lighting sun) stays the base; a cool (0.62, 0.74, 1.0), dim directional term from the visible white moon (`daynight::moon_direction`, the disc the sky draws) is ADDED per fragment with a soft-wrapped N·L on terrain, models, WMO exteriors and retained doodads, shadowed by the moon's own map (`shadow_hook::realtime_shadow_moonlit` `.z`, faded in with the rig's hand-over ramp). It ramps with the moon's elevation (0-12°) and out with the sun, dims under storms, and never reaches interiors or the glue booth. Packed as MonkeyFrame row 16 (`moon`: direction + intensity, base 0.16 gamma at `moonLight 1`) and `misc.w` (hand-over confidence); the per-frame blob is 8800 B (`lighting/moonlight.rs`, `moonlight_hook.wgsl`) | Advanced Graphics → Moonlight, cvar `moonLight` 0..2 (0 = the reference night, bit for bit; Graphics preset Low..Ultra = 1) |
| Ground-effect spells | Flamestrike, Rain of Fire, Consecration, Flare and fire traps light the ground for their duration; frost and nature areas stay dark | `spellLightGain` |
| Volumetric fog | near-field haze that converges on the zone fog colour (clear within 10 yd, full by 150 yd; mistier at dawn and in bad weather, faint indoors; the dawn curve follows the rendered clock, so a manual or capture time sets it; over an ocean a point below sea level is hazed at the sea surface, since water writes no depth) and daytime sun shafts through gaps beyond 10 yd, sampled from the shadow map (never moon shafts: they are gated by daylight; retired while Volumetric Light is on); own fullscreen pass after the main pass (`benilla-app/src/volumetric_fog.rs`) | Advanced Graphics → Volumetric Fog (Off/Low/High), cvar `volumetricFog`, env `WOW_VOLFOG=0\|1\|2` |
| Volumetric light | GFX (volumetric light): shadow-mapped sun and moon shafts. The directional light the shadow rig holds (sun by day, moon at night) is ray-marched along each view ray at HALF resolution: 16 (Medium) or 32 (High) samples, quadratic spacing from 0.35 yd (dense near the eye), a per-pixel interleaved-gradient-noise offset, cascade lookups with the covered-sample mean past the cascades, air thinning with height above the eye (10 yd e-fold, so shafts live under canopies and between buildings rather than in the open sky), a Henyey-Greenstein g = 0.6 phase with a quarter isotropic. A depth-aware 4x4 half-res blur removes the jitter and a depth-aware 2x2 upsample composites it with the fog pass's soft screen curve. Extinction = the volumetric fog density law (dawn mist, weather) x the zone's fog end (350 yd reference, x0.7-2.5); colour = sun diffuse (or the moon tint x moonlight intensity, faint) leaned 35 % toward the zone fog hue. No shafts in WMO interiors. While it is on, the volumetric fog pass retires its own midpoint shafts and keeps only its haze and lamp halos. Three passes after the fog node (`benilla-app/src/volumetric_light.rs` + `.wgsl`); no temporal history. Dev capture levers `WOW_VOLLIGHT_DENSITY/_HEIGHT/_GAIN/_DEBUG` (1 = the in-scatter alone) | Advanced Graphics → Volumetric Light (Off/Medium/High) + Volumetric Light Strength 0..2, cvars `volumetricLight`, `volumetricLightStrength`, env `WOW_VOLLIGHT=0\|1\|2`; Graphics preset Medium = 1, High/Ultra = 2 |
| Sky dithering | a faint screen-space dither in the FFXGlow combine so smooth sky and fog gradients do not band (MONKEY p0; was env-only) | Advanced Graphics → Sky Dithering, cvar `skyDither` 0/1 (default 0, Graphics preset High = 1), env `WOW_DITHER=1` still forces it on; bridge in `benilla-app/src/monkey_gfx.rs` |
| HDR emission and bloom | lit WMO windows, additive models/particles and magma can exceed display white; a soft-capped fullscreen halo is added before the faithful FFX clamp | Advanced Graphics → Bloom (Off/Low/High), cvar `bloom`; Off is the unchanged reference image, suggested High preset value 2 |
| Sky quality | Enhanced: the five Light.dbc sky stops through a smooth monotone curve in linear light (no bands at the rings), a soft sun glow tinted by sun and fog colour (fades at night and under cloud), a procedural star field with twinkle and a faint Milky Way over the stock `Stars.m2`. High adds domain-warped cloud detail and sun-lit clouds (self-shadow, silver lining; technique from WarcraftXL, see `THIRD-PARTY.md`). Classic is the reference sky unchanged | Advanced Graphics → Sky Quality, cvar `skyQuality` 0 Classic / 1 Enhanced / 2 High, env `WOW_SKY_QUALITY` |
| Modern fog | MONKEY (fog): radial distance fog with a gradual exponential curve, a daylight sun lobe, and an end colour shared by the world, WDL hull, sky horizon and volumetric haze. Each zone keeps its authored fog end up to the reference view-distance limit; optional `LightFogBand.dbc` rows provide height, sun and end-fog controls. Interior WMO fog stays classic | Advanced Graphics → Modern Fog, cvar `fogModel` 0 Classic (default) / 1 Modern (High = 1), env `WOW_FOGMODEL=0\|1`; `fog_hook.wgsl`, `lighting/fog_model.rs`, `light/fog_band.rs` |
| Rain on surfaces | MONKEY (wet): rain gradually darkens and saturates exposed terrain and models, adds puddles and a sky/sun sheen, and produces rings on exterior Enhanced water. Wetness rises over roughly 90 seconds and dries over roughly four minutes; MONKEY (rainshelter): a camera-centred rain-occlusion map (128² cells of 0.75 yd, filled by vertical collision rays) keeps ground, walls and water under roofs, porches and bridges dry, with a ~1.5 yd drip margin and a soft edge; tree canopies without colliders do not shelter | Advanced Graphics → Wet Surfaces in Rain, cvar `rainSurfaces` 0/1 (default and High 1), env `WOW_RAIN_SURFACES`, `WOW_WETNESS`, `WOW_WET_T`; `weather/wetness.rs`, `weather/shelter.rs`, `wet_hook.wgsl` |
| Foliage wind | one weather-fed gust/veer field drives grass and classified tree/bush foliage; grass also parts around the player and nearby units | Advanced Graphics → Foliage Wind, cvar `foliageWind` 0 Off / 1 Grass / 2 Grass + Trees (High = 2), plus the slider Foliage Wind Strength, cvar `foliageWindStrength` 0.25..3 (default 1, not on the preset ladder; a gain on the sway, not the parting; a storm adds up to +60 %). Tuning at 1 (fix-wind): grass tips about ±0.2-0.3 yd on a 2.5 s wave with 1.1 s cross flutter, gust bands 55 yd wide rolling downwind; tree crowns about ±0.3 yd (lever-arm weight grows with height to 1.3x at 8.5 yd) on a 4 s sway, 5.5 s side wave, 0.8 s leaf flutter; bent tips dip slightly. capture override `WOW_FOLIAGE_WIND` (dev build + `WOW_CAPTURE` only); field and benders in `benilla-world/src/wind/`. The wave phase is the CPU-integrated travel (speed integrated over time, wrapped seamlessly), and the waves' spatial term uses the fixed profile heading; the veer turns only the bend. Tree exile copies sway through material marker bit 14 (`FOLIAGE_WIND_MARKER`). Capture offset `WOW_CAPTURE_WIND_T` (s). Known: shadow casters (world-shadow proxies, `torch_depth.wgsl`) do not apply the sway, so leaves sample a static shadow; a future world-camera DepthPrepass must call the same offsets |
| Ambient occlusion | soft contact shadows under props, in wall-floor corners and around feet; depth-only SSAO at half resolution (depth-reconstructed normals, 6/12 spiral taps, 4x4 depth-aware blur, bilateral upsample) multiplied into the opaque scene after the opaque pass and before the water copy, so water, particles and UI are untouched; fades out 45-90 yd (Low) / 60-120 yd (High), spares sky and bright pixels; on alpha-tested foliage (a dense window of two-sided depth spikes) the blur widens its depth tolerance and the AO drops 40 %, which removes the leaf grain while planes, silhouettes and contact creases keep full AO (`benilla-app/src/ssao.rs`, `benilla-app/src/shaders/ssao.wgsl`) | Advanced Graphics → Ambient Occlusion (Off/Low/High; default Off, High preset 2), cvar `ambientOcclusion`, env `WOW_AO=0\|1\|2`, `WOW_AO_DEBUG=1..5` diagnostic views (factor, protection, distance, raw occlusion, foliage mask), `WOW_AO_GAIN`/`_RADIUS`/`_BIAS`/`_STRENGTH` tuning |
| Lamp fog | MONKEY (lampfog): the fog fullscreen pass selects the nearest resolved exterior and interior point-table sources and analytically integrates their softened inverse-square light over the camera-to-scene-depth ray. A near-field haze floor makes lamps visible inside the ordinary volumetric haze's 10 yd clear zone; it is zero by day, stronger in rain/fog, and deliberately unshadowed. The table colour already includes intensity, flame flicker and `fireLightGain`. Halo output uses destination headroom and stays below 1.0, so a later bloom pass cannot blow it out | Advanced Graphics → Lamp Fog, cvar `lampFog` 0 Off (default, unchanged) / 1 Low (16 lamps) / 2 High (32 lamps; Graphics preset High = 2), env `WOW_LAMPFOG=0\|1\|2`; `benilla-app/src/volumetric_fog.rs` |
| City window daylight | MONKEY (fix-daylight/integration): a district WMO's oversized window batch (e.g. the Cathedral of Light's stained-glass walls) is split into window-sized daylight apertures, so the nave takes daylight pane by pane. Applies to WMOs loaded after a change | Advanced Graphics → City Window Daylight, cvar `daylightWindowSplit` 0/1 (default 1, the merged behaviour); `benilla-world/src/lighting/daylight.rs` (`set_window_split`), bridge `benilla-app/src/dynamic_interior.rs` |
| Zone skyboxes | the painted sky a zone's `LightParams` names (clear-weather slot, storm slot lerped by the storm blend) for the living, fading in by the Light sphere falloff with the modern client's crossfade (ghost > WMO > zone); extended `LightSkybox.dbc` flags `0x1` day cycle, `0x2` keep sun/moon/stars/clouds, `0x4` fog-colour horizon cone, `0x8` forces the screen-space sun shafts through a low or just-set sun (the sky's weight lifts their daylight fade; `post/sun_shafts.rs`), `0x10` fades the Modern fog's sun-colour lobe out by the sky's weight (`lighting/fog_model.rs`), plus a celestial second model; every skybox now plays its bones, texture transforms, colour and alpha tracks (`benilla-world/src/skybox.rs`, `skybox_anim.rs`, `benilla-formats/src/light/skybox.rs`) | Advanced Graphics → Zone Skyboxes, cvar `zoneSkyboxes` (0 Off default, 1 On; High = 1), env `WOW_ZONE_SKYBOXES=0\|1` |
| Night and interior level | global dimming of the night sky term and of interior ambient | `nightGain`, `interiorGain`, `interiorBakeFloor` |

`zoneSkyboxes = 0` disables only the added living-player zone slot. WMO and ghost skyboxes are
the reference 1.12 slots, so their fidelity fixes remain active in Classic: authored colour and
alpha tracks affect the layer, non-white M2 colours reach the mesh, deterministic captures pose at
their pinned time, and the batch-order base reserves the lower celestial band and final fog cone.

Asset-free local builder and bilingual install guide: [`Optional/sky-and-grading/`](Optional/sky-and-grading/README.md).

Players reach all of it from **Options -> Advanced Graphics** (a Graphics Preset over everything, a Render Distance slider, a Lighting Quality preset Off / Low / Medium / High / Ultra plus the individual rows; Classic / Off is the original client look). The dev build has a panel for all of it: **Ctrl+Shift+D → Lighting & shadows**, with Dim / Default /
Bright presets.

## Graphics Preset

MONKEY (presets). The top row of Advanced Graphics, cvar `graphicsQuality`: Classic / Low / Medium /
High / Ultra, or Custom when the rows match none of them. The table is `GRAPHICS_PRESETS` in
`benilla-app/src/cvars.rs` (one line per governed row); choosing a rung writes each row through
the ordinary cvar path, and the label is re-derived every frame, so editing any governed row
(or any Lighting Quality member) shows Custom. `lightingQuality` is itself a governed row, so
every row the lighting ladder owns (`LIGHTING_PRESETS`: both sun lanes, shadow resolution, the
room and torch lanes, moon shadows, `waterQuality`, `volumetricFog`, `bloom`, `sunShafts`,
`colorGrading`, `foliageWind`, the fire/spell/lava gains, `nightGain`, `interiorGain`) follows it.

| row | Classic | Low | Medium | High | Ultra |
|---|---|---|---|---|---|
| `lightingQuality` | Off | Low | Medium | High | Ultra |
| `farclip` (Render Distance, yd) | 350 | 350 | 477 | 777 | 1497 |
| `skyQuality` | 0 | 1 | 1 | 2 | 2 |
| `skyDither` | 0 | 1 | 1 | 1 | 1 |
| `fogModel` | 0 | 1 | 1 | 1 | 1 |
| `rainSurfaces` | 0 | 1 | 1 | 1 | 1 |
| `torchTerrainShadows` | 0 | 0 | 0 | 1 | 1 |
| `daylightWindowSplit` | 0 | 1 | 1 | 1 | 1 |
| `ambientOcclusion` | 0 | 0 | 1 | 2 | 2 |
| `zoneSkyboxes` | 0 | 0 | 1 | 1 | 1 |
| `lampFog` | 0 | 0 | 1 | 2 | 2 |

What the lighting rungs set, beyond Off (everything off, gains 1.0) and High (the registered
defaults): Low = character shadows only at a 1024 map, no torch or moon shadows, bloom Low;
Medium = both sun lanes, indoor torch shadows (6 resident / 2 moving), moon shadows, bloom Low,
no sun shafts or colour grading; Ultra = High with a 4096 sun map, 16 / 8 torch maps, High water
and High volumetric fog. No rung decides `shadowDistance` or `interiorShadowSoft`.

**Default: High.** A player whose `config.toml` names no `graphicsQuality` is seeded to the High
column at startup (`Cvars::seed_graphics_preset`) on every row the file does not carry; the rows'
registered defaults stay their per-lane ones (`farclip` the reference's 350), and captures and
tests, which never seed, keep them. The Advanced page's Defaults button lands on High too.
The seeded boot value that leaves the reference (`farclip` 777 against 350) is declared in
`cvars::SEEDED_DEVIATIONS`, and a test walks the whole High column against the reference rows.
Session-owned rows (env levers such as `WOW_FARCLIP`) are skipped when both labels are derived, so
an env override never persists `Custom` (MONKEY (reviewfix-a)).

**Capture fixtures** (MONKEY (reviewfix-a)). A capture reads CVars only from
`WOW_CAPTURE_CVARS=<config.toml path>` (read-only; nothing is saved), never from `BENILLA_HOME`.
`WOW_FOLIAGE_WIND` is honoured only in a dev build under `WOW_CAPTURE`. A capture has no avatar,
so zone grading reads `terrain_stream::CaptureCameraArea` (the area under the camera, written only
under `WOW_CAPTURE`) and the grade's day/night weight follows the rendered `GameClock` minute.

**Saved presets and new rows** (MONKEY (followups)). On load, `Cvars::reapply_saved_presets`
writes the saved `graphicsQuality` rung (and the saved `lightingQuality` rung, else the Graphics
rung's) over every governed row the file does NOT carry, so a row registered after the save gets
the preset's value, not its registered default. Rows the file carries and session-owned rows stay;
`Custom` writes nothing.

**Render Distance** is `farclip`, the same cvar as the Graphics page's Terrain Distance (which
keeps the stock 177–777 slider). Its clamp (`view::FARCLIP_RANGE`) now reaches `FARCLIP_MAX` =
1497 yd, the 1.12 grid of 60 carried on. Terrain residency (`terrain_stream/window.rs`), the
model/WMO and particle walls (`view::within_farclip`) and the art-cache sweep radius (now 7 tiles,
`art_scope.rs`) all follow it; at 1497 the residency window is at most 7×7 ADT tiles against
777's 5×5, which is the memory bound. Past 777 only Modern fog (`fog_model::modern_fog_end`)
stretches the fog end with it; under Classic fog the zone's fog end still hides the extra reach.

The optional enhanced water (Water Quality, refraction, caustics, High reflections, lava glow)
is its own module, documented in `WATER.md`.

## Where the code lives

- **Shaders**: `crates/benilla-assets/src/shaders/shadow_hook.wgsl` (shared shadow sampling),
  `fog_hook.wgsl` (MONKEY p0: the ONE distance-fog law — `fog_linear`, `fog_sample`, `apply_fog`;
  terrain, wow_model, static_gx, liquid, enhanced_water, wow_effect and wdl all call it; classic is
  bit-identical to the old per-shader copies, and `MonkeyFrame.fog_model` = 1 is the FOG lane's
  extension point), `monkey_frame.wgsl` (the MonkeyFrame struct),
  `crates/benilla-assets/src/shaders/emissive_hook.wgsl` (shared opt-in HDR multipliers),
  `crates/benilla-world/src/shaders/torch_depth.wgsl`, and the lighting lanes inside
  `static_gx.wgsl`, `wow_model.wgsl` and `terrain.wgsl`. The three receivers mirror each other; the
  comments say where.
- **World** (`crates/benilla-world/src`): `lighting/global_light.rs` (light table packer, lanes,
  room claims, gains), `lighting/daylight.rs`, `lighting/flicker.rs`, `static_gx/torch_depth.rs`
  (cube-map cache), `static_gx/shadow.rs`, and the light spawning in `terrain_stream/spawn/`.
- **Data rules** (`crates/benilla-formats/src`): `fire_light.rs` (which models emit light, their
  colour and reach, the spell school rules) and `room_claim.rs` (which rooms a light may light).
- **App** (`crates/benilla-app/src`): `torch_shadow.rs` (caster selection), `shadow_core.rs`,
  `character_shadow.rs`, `world_shadow.rs`, `blob_shadow.rs`, `entities/carried_light.rs`,
  `entities/spell_fx/lifecycle.rs`, `dynamic_interior.rs`, `debug_panel/lighting_controls.rs`, and
  the cvars in `cvars.rs` / `video.rs`.
- **Sky** (`crates/benilla-world/src`): `sky_fx.rs` (tier, clock, glow inputs), `shaders/sky_fx.wgsl` (gradient curve, glow, stars, cloud noise), the tier branches in `shaders/sky.wgsl` and `shaders/cloud.wgsl`, `clouds/layer.rs` (`update_cloud_fx`); the cvar bridge is `benilla-app/src/sky_quality.rs`.
- **Census**: `cargo test -p benilla-world --lib lighting::daylight::census -- --ignored --nocapture`
  (`WOW_CENSUS_WMO`, `WOW_CENSUS_PLACE=map,tx,ty,uid`) prints, per interior room of a city WMO,
  whether a daylight fixture, a bleed fixture or neither reaches it, and the sky-room count.
- **Wind**: `wind/mod.rs` resolves the WXL-derived shared field and nearest-unit benders;
  `clutter.rs` authors blade weights/phases; `static_gx` classifies and marks alpha-tested leaf
  batches. `wind_hook.wgsl` owns the displacement shared by clutter, retained trees and fade twins.
- **Tools**: `benilla-extract <Data> wmolights <wmo> [--verts <group>]`, `wmolamps`, `m2firescan`
  print the inputs the system works from (groups, batch classes, portals, claims, flame emitters).

## Invariants worth knowing before editing

- The shared light buffer's per-frame blob is 8784 bytes and the torch table 6416 bytes; both are
  mirrored in the shaders and pinned by tests. The blob is 21 header rows (336 B), the 256-slot
  point-light table (8192 B), then the MONKEY **MonkeyFrame** block (16 rows, 256 B; nothing before
  it moves). Every `WowLight` mirror (terrain, wow_model, static_gx, liquid, enhanced_water,
  wow_effect, wdl) declares it as `monkey: monkey_frame::MonkeyFrame` after `points`:

  | row | x | y | z | w |
  |---|---|---|---|---|
  | `fog_a` | height_fog_density | height_fog_height | height_fog_falloff | curve_blend |
  | `fog_b` | sun_fog r | g | b | sun_fog_strength |
  | `fog_c` | end_fog r | g | b | end_fog_distance |
  | `fog_d` | fog_model (0 classic; Modern = scene fog end in yd) | sun_fog_angle | sun direction octahedral x | y |
  | `wind_a` | dir_x | dir_y | base_heading (rad) | gust |
  | `wind_b` | travel (yd, wrapped at 4096) | sway_strength | grass_strength | tree_strength |
  | `wet_a` | rain_rate | wetness | ripple_time_s | snow |
  | `misc` | bender_count | time_of_day 0..1 | night 0..1 | 0 |
  | `benders[8]` | world x | world y | world z | radius |

  MONKEY (rainshelter): the rain-shelter region follows the MonkeyFrame block (row 549: two
  header rows, then 16384 packed `u32` cells, 65568 B; `weather/shelter.rs`), and the prop-probe
  region moved up by that much. `terrain`, `wow_model` and `static_gx` mirror it as
  `shelter_hdr`, `shelter_cfg`, `shelter`; `enhanced_water` reads it through its row view.

  Positions are Bevy world space (Y up, yards; WoW `(x, y, z)` = Bevy `(-y, z, -x)`), the wind
  direction is a unit vector in world XZ. Lanes write the `MonkeyFrame` resource
  (`benilla-world/src/lighting/monkey_frame.rs`); `global_light::pack_monkey_frame` packs it after
  the point table and fills `time_of_day` / `night` itself. All zero = no visual change.
- World lights use upstream's `WorldPointLight`, never Bevy's `PointLight`.
- Foliage wind is vertex-only. Tree and leaf shadow meshes stay static; the small mismatch is the
  accepted first-stage cost/complexity tradeoff.
- MONKEY (lampfog): `ResolvedPointLights` is a CPU-only read view of the live, truncated point table;
  it does not add a row to the shared GPU buffer. Lamp selection is capped at 100 yd and ranks both
  exterior and interior rows together before the 16/32 nearest limit is applied.
- WGSL only fails at pipeline creation, so a shader edit is verified by running a world scene, not
  by `cargo check`.
- Exterior point lights are selected per draw unit (12 slots, ranked against the chunk's box); the
  cube-map occlusion is evaluated for the nearest three.
- Portal bleed fixtures (`update_bleed_fixtures`, `daylight.rs`) are re-evaluated at 10 Hz, not per
  frame; a new doorway or a dial change runs it at once. Sources are bucketed per placement.

## Build

```bash
cargo build --release -p benilla
```

Debug tracing: `WOW_TORCH_TRACE=1`, `WOW_POINTS_DUMP=1`, `WOW_SHADOW_TRACE=1`, and the cvar
`interiorDebug` 1..4.

Performance: measure a `play`/release build, never `dev`. `WOW_CAPTURE=<scenario>
WOW_FPS_PROBE=600 WOW_FPS_JOURNAL=<csv> WOW_GPU_PASSES=1 WOW_WIN=2560x1440` prints frame, CPU and
per-pass GPU times; `BENILLA_HOME=<dir>` supplies the `config.toml` (a capture never seeds the
Graphics Preset, so write the rows out). `WOW_PERF_CROWD=<n>` stands n players at the look point.
Night fixtures: `perf-northshire-night`, `perf-abbey-night`. Per-system CPU: build with
`--features benilla-app/trace_chrome`.

## Licence

Same as upstream: MIT OR Apache-2.0. Code and techniques taken from other projects (WarcraftXL,
by iThorgrim) are credited file by file in [`THIRD-PARTY.md`](THIRD-PARTY.md).
