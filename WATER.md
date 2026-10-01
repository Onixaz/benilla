# benilla-water

An optional enhanced water renderer for [benilla](https://github.com/samwhosung/benilla), the Rust +
Bevy reimplementation of the 1.12.1 client. It needs no data changes: everything is derived from
what a 1.12 install already carries (MCLQ liquid depth and MLIQ opacity bytes, their liquid grids, the
`Light.dbc` water and sky rows, the opaque scene depth).

It is a **module on top of the reference water, not a replacement for it**. The reference path
(`Classic`) is untouched and stays byte-identical to upstream; the enhanced path only runs when the
player picks it, and every part of it can be switched off.

## Credits

- **Original project: [WarcraftXL](https://github.com/WarcraftXL)** — the `wxl-experimental-water`
  3.3.5a client extension.
- **Author: iThorgrim.**

The WarcraftXL water module is the design reference for this one, and its author has given
permission to reuse its code here on the condition that the author and the original project are
named. This section is that attribution; it must travel with the module (this file, and the
`enhanced_water.wgsl` header) wherever the module is copied or shared.

Rule for contributors: a file that ports code or a technique from WarcraftXL says so in its header
(`Ported from WarcraftXL (https://github.com/WarcraftXL) by iThorgrim — module <module>, <source
files>.`; blocks inside a file keep their `Portions derived from WarcraftXL ...` line), the
technique is listed in the table below, and the file is listed in
[`THIRD-PARTY.md`](THIRD-PARTY.md), the one table of every WarcraftXL-derived file across the
fork. Do not remove any of them.

| Technique | Status here | From WarcraftXL |
|---|---|---|
| Visual-only wave riding for swimmers (authoritative position never moves) | shipped (`water_fx/bob.rs`) | same principle as `world/Ride.cpp`, implemented independently |
| Shore surf limited by the TERRAIN column, so objects in the water never foam | shipped (authored depth gate, `on_bed`) | principle from `sea/Shore.hpp` |
| Per-channel extinction + scene-copy refraction | shipped (`enhanced_water.wgsl` DEPTH LOOK; bent samples landing on anything in front of the water are refused via the scene depth) | ported from `shaders/Surface.ps.hlsl`, `render/Refraction.cpp` |
| Caustics, two layers at incommensurate scales/rates | shipped (floor web; built from animated cell edges, not value noise) | ported from `shaders/Surface.ps.hlsl`, `render/Noise.cpp` |
| Screen-space reflection of the scenery | shipped on `High` (a bisected march against the scene depth, not WXL's single probe) | principle from `shaders/Surface.ps.hlsl` |
| Gerstner trains + breaker index, crest-fold foam | planned | `sea/Spectrum.*`, `sea/Shore.hpp`, `shaders/Wave.hlsli` |
| Specular lobe widened by the normal's screen derivative, distance glints | shipped / planned | `shaders/Surface.ps.hlsl` |

## What it does

| Lane | What you see | Main switch |
|---|---|---|
| Quality tiers | `Classic` = the reference water, `Enhanced`, `High` (= Enhanced + scenery reflections) | Video options → Water Quality, cvar `waterQuality`, env `WOW_WATER=0\|1\|2` |
| Procedural waves | multi-band analytic waves with exact normals; the long swell moves ocean vertices | tier |
| Open-sea whitecaps | High adds Gerstner crest gathering and fold-thresholded foam; calm canals remain clear. The fold mask adds the next two wave bands, a slow domain-warped gust field moves the break threshold and calms patches, and ~6 yd segments break each crest into short runs, so the caps do not form a lattice (`whitecap_fold`, `enhanced_water.wgsl`) | tier |
| Near mesh refinement | High builds a transient 4x liquid lattice within 64 yd, with an 80 yd release ring | tier |
| Ocean / inland profiles | the sea and lakes/rivers have their own colour, energy and reflectivity | tier |
| WMO pools | exterior canals use a calm drifting profile with outdoor glints; interiors use their authored MOMT colour and room-fog reflection | tier |
| Depth look | light fades per channel over the real path through the water; the bed is seen through it, bent by the waves; sunlit shallows carry a caustic web | tier |
| Reflection and glints | Fresnel sky reflection from the zone's sky rows, sun path, moon path, point-light glints | tier |
| Beach surf | a travelling wave front, a lace of foam that dissolves behind it, a swash sheet on the sand | tier |
| Swimmers | wake foam, treading rings and a visual-only bob with the swell | tier |
| Lava glow | magma surfaces emit warm point lights through the lighting module | Advanced Graphics → Lava Glow, `lavaLightGain` |
| Lava HDR | magma partially resists distance fog and feeds the screen-space halo | Advanced Graphics → Bloom, `bloom` (Off/Low/High) |

Rivers deliberately have NO flow effect: a derived current was tried and removed at the owner's
request. If it ever returns it should read the authored MCLQ flow records, not derive one.

## Where the code lives

- **Shader**: `crates/benilla-assets/src/shaders/enhanced_water.wgsl` - the whole module on the GPU
  side, an importable library (`benilla::enhanced_water`) with its own bindings: 103 scene depth,
  104 `WaterParams` (`benilla_assets::WaterUniform`), 105 the shared light buffer (a second
  read-only view of upstream's binding 90), 106 the scene colour (for refraction and reflection). Upstream's `liquid.wgsl` carries three hooks only: the
  `#import`, `water_swell` in its vertex stage, and the `water_active()` branch at the top of its
  fragment stage, which returns the module's result as-is (the module fogs its own surface terms
  only: the scene it refracts and reflects is already fogged). Keep it that way: new
  water code goes in the module, not in `liquid.wgsl`.
- **Assets** (`crates/benilla-assets/src`): `water_depth.rs` (`WaterUniform`, the scene-depth image),
  `materials.rs` (`LiquidExt` fields 103-106, shader registration), `lib.rs` (`WaterQuality`).
- **World** (`crates/benilla-world/src`): `liquid/scene_depth.rs` (copies the opaque depth AND colour after
  the main opaque pass), `liquid/waves.rs` (the CPU mirror of the vertex swell), `liquid/surface.rs`
  (per-kind material parameters and the opt-in `WOW_WATER_PROBE` WMO classification/depth log),
  `liquid/lod.rs` (High-only 4x near-grid swap),
  `water_fx/bob.rs` (swimmer bob), `lighting/lava_light.rs`.
- **App** (`crates/benilla-app/src`): the setting in `cvars.rs` / `video.rs`, the options rows in
  `assets/ui/OptionsFrame.xml`, the capture scenes `water-*` and `lava-*` in `capture/scenarios.rs`.

## Invariants worth knowing before editing

- `Classic` must stay byte-identical: capture `water-noon` with `WOW_WATER=0` and compare with the
  baseline before and after any edit.
- The CPU swell (`liquid/waves.rs`) and the shader's `WATER_WAVES` table are one contract.
- MLIQ's per-vertex byte is an opacity-ramp coordinate, not water depth; enhanced WMO water uses
  reconstructed scene depth for its column measurement.
- Every screen derivative in `enhanced_water()` is taken at the top of the function (WGSL
  uniformity); WGSL only fails at pipeline creation, so verify by running a `water-*` scene.
- Never rotate a noise domain by a per-pixel angle at world coordinates (~1e4 yd): it shears.
- The temporary water wind is `WATER_WIND_DIR`; all wind-aligned shader terms derive from it until
  the weather lane supplies shared wind.
- Run one GPU capture at a time, and run `cargo test` in a shell WITHOUT `WOW_DATA` exported.
