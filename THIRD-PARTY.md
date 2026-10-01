# Third-party credits

Code and techniques this fork takes from other projects, with the files they landed in. Vendored
components under `third_party/` carry their own licences beside them and are listed in `README.md`.

## WarcraftXL

- **Project: [WarcraftXL](https://github.com/WarcraftXL)** — client extensions for the **3.3.5a**
  (build 12340) client, written in C++ with D3D9 / HLSL.
- **Author: iThorgrim.**

WarcraftXL's code is reused here on the condition that the author and the original project are
named. This file, and the header line in each file below, is that attribution; keep both wherever
the code is copied or shared.

**Rule for contributors.** A file that ports code or a technique from WarcraftXL carries this line
in its header:

```
Ported from WarcraftXL (https://github.com/WarcraftXL) by iThorgrim — module <module>, <source files>.
```

and gets a row in the table below. Blocks inside a file that port a specific routine also say so
where they stand. Do not remove either.

| Benilla file | WarcraftXL module | Source files | What was taken |
|---|---|---|---|
| `crates/benilla-assets/src/shaders/enhanced_water.wgsl` (DEPTH LOOK block) | `wxl-experimental-water` | `shaders/Surface.ps.hlsl`, `render/Refraction.cpp` | Per-channel Beer-Lambert extinction over the view path through the water column, the body as a lerp from scattered colour to what is behind by the transmittance, normal-bent scene-copy refraction scaled by the water depth (ported) |
| `crates/benilla-assets/src/shaders/enhanced_water.wgsl` (caustics) | `wxl-experimental-water` | `shaders/Surface.ps.hlsl`, `render/Noise.cpp` | Two caustic layers at incommensurate scales and rates, multiplied (ported; our layers are built from animated cell edges, not value noise) |
| `crates/benilla-assets/src/shaders/enhanced_water.wgsl` (HIGH reflections) | `wxl-experimental-water` | `shaders/Surface.ps.hlsl` | Screen-space reflection of the scene copy along an almost-planar normal, masked by the screen edge and by rays turning back toward the eye (principle; ours is a bisected march against the scene depth, not a single probe) |
| `crates/benilla-assets/src/shaders/enhanced_water.wgsl` (beach surf, `on_bed`) | `wxl-experimental-water` | `sea/Shore.hpp`, `shaders/Shore.hlsli` | Shore surf limited by the terrain column, so objects standing in the water never foam (principle) |
| `crates/benilla-assets/src/shaders/enhanced_water.wgsl` (Gerstner crests, High) + `crates/benilla-world/src/liquid/waves.rs` | `wxl-experimental-water` | `shaders/Surface.ps.hlsl`, `sea/Spectrum.*`, `sea/Ocean.*` | Gerstner displacement and fold-driven crest whitecaps; the CPU swimmer sample inverts the same displacement (ported) |
| `crates/benilla-world/src/water_fx/bob.rs` | `wxl-experimental-water` | `world/Ride.cpp` | Visual-only wave riding for swimmers; the authoritative position never moves (principle, implemented independently) |
| `crates/benilla-app/src/post/grading.rs` + `crates/benilla-app/src/post/grading.wgsl` | `wxl-retail-grading` | `Grading.cpp`, `shaders/Grading.ps.hlsl` | 32³ LUT convention, strength control and strip lookup mathematics, with the strip uploaded as a native 3D texture (ported; used with the author's permission, attribution required) |
| `crates/benilla-world/src/shaders/sky_fx.wgsl` | `wxl-retail-clouds` | `Clouds.cpp`, `Clouds.hpp` | Domain-warped billow fbm (quintic value noise, 0.6 cotton blend, 0.55 octave gain) for the cloud detail |
| `crates/benilla-world/src/sky_fx.rs` | `wxl-retail-clouds` | `Clouds.cpp`, `Clouds.hpp` | The High cloud-detail tier and the sun-march inputs it feeds the sky and cloud materials (ported) |
| `crates/benilla-world/src/shaders/cloud.wgsl` | `wxl-retail-clouds` | `Clouds.cpp` | The 3-tap march toward the sun (self-shadow, ambient floor) and the one-tap silver lining |
| `crates/benilla-world/src/clouds/layer.rs` (`update_cloud_fx`) | `wxl-retail-clouds` | `Clouds.cpp` | March direction = the sun projected on the sheet, strength folded by its flatness (noon clouds evenly lit) |
| `crates/benilla-world/src/wind/mod.rs` | `wxl-experimental-wind` | `field/Wind.hpp`, `field/Wind.cpp` | Stateless three-sine gust and veer field, default profile, and weather gain (ported) |
| `crates/benilla-assets/src/shaders/wind_hook.wgsl` | `wxl-experimental-wind` | `grass/GrassWind.hpp`, `grass/GrassWind.cpp` | Two-wave grass sway, gust response, lean, blade phase/variance, distance fade and radial parting (ported; extended to eight benders) |
| `crates/benilla-assets/src/shaders/wow_model.wgsl` (MONKEY wind hook) | `wxl-experimental-wind` | `grass/GrassWind.cpp` | Grass vertex displacement call seam (technique) |
| `crates/benilla-world/src/clutter.rs` (MONKEY wind attributes) | `wxl-experimental-wind` | `grass/GrassWind.cpp` | Per-blade bend weight and per-tuft phase inputs (technique; height replaces WXL's unverified texture-V weight) |

Planned (not yet in the tree): breaker index from `sea/Shore.hpp`, `shaders/Wave.hlsli`.
The lane that ports one adds its row here.
