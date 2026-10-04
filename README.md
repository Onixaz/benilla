# benilla — enhanced

A fork of [**benilla**](https://github.com/samwhosung/benilla), the from-scratch World of Warcraft 1.12.1
client in Rust and Bevy by samwhosung. **[Read the original benilla README →](https://github.com/samwhosung/benilla/blob/main/README.md)**


Upstream benilla tracks the 1.12.1 client exactly, so changes that go beyond it live here instead. This fork brings together its own work and work from other forks in three areas:

1. **[Turtle WoW support](#turtle-wow-support)**: log in, create characters and play on Turtle WoW, including its races, auction house and transmog.
2. **[Enhanced graphics](#enhanced-graphics)**: an optional modern look (shadows, modern fog, volumetric light, water, weather, HD character textures). The **Classic** preset keeps the original 1.12 image.
3. **[DLSS 5 Neural Rendering](#dlss-5-neural-rendering-experimental)** *(experimental)*: NVIDIA's neural renderer applied to the world, on RTX cards.

Everything else (formats, networking, the stock interface, addons, audio) is upstream benilla, merged in
regularly.

**Want to modernize 1.12.1? This is the place.** New graphics, private-server support, quality-of-life
features: anything that goes beyond the original client is welcome here as an issue or a pull request.
This repository is maintained and will stay open source.

**Fixing benilla itself?** Bugs and changes that make benilla more like the real 1.12.1 client belong in
[upstream benilla](https://github.com/samwhosung/benilla/pulls): open the pull request there, and it
reaches this fork with the next merge.

## Enhanced graphics

An optional modern look: realtime sun and moon shadows, lit interiors and flickering torch light, a
modern sky and fog with volumetric light shafts, bloom and per-zone colour grading, enhanced water,
rain and wind, render distance up to 1497 yards, and HD character texture packs.

Pick a **Graphics Preset** (Classic / Low / Medium / High / Ultra) in the options, or switch each feature
yourself under **Options → Advanced Graphics**. **Classic** keeps the original 1.12 image.

Based on https://github.com/pkuzic/benilla-everwood_graphics

## Turtle WoW support

benilla plays on Turtle WoW: logging in, creating characters (including Turtle's races), the auction
house and transmog all work against Turtle's servers. Point it at your own Turtle WoW install; benilla
reads the game data from it and never writes into it.

Initial work from https://github.com/jhinzuo2/benilla-twow



Details: [`LIGHTING.md`](LIGHTING.md), [`WATER.md`](WATER.md), and an optional sky and grading pack in
[`Optional/sky-and-grading`](Optional/sky-and-grading/README.md). Third-party credits, including code
ported from [WarcraftXL](https://github.com/WarcraftXL) by iThorgrim: [`THIRD-PARTY.md`](THIRD-PARTY.md).

## DLSS 5 Neural Rendering (experimental)

benilla can run NVIDIA's DLSS 5 **Neural Rendering** (NGX Feature 18) over the world view. It is not
the DLSS upscaler: the image stays at native resolution and the network re-renders the lit scene.
It is off unless you build for it, and the interface, spell effects, water and fog are drawn on top
of its output unchanged.

**Requirements**
- Windows and an NVIDIA RTX GPU (40 or 50 series), Vulkan
- The [NVIDIA DLSS SDK](https://github.com/NVIDIA/DLSS) (tested with 310.4.0 and 310.9.1), for linking
- The Feature-18 runtime, `nvngx_dlssnr.dll`. It does not come with the SDK and NVIDIA does not
  distribute it; you provide it yourself.
- MSAA off (`gxMultisample 1`, the default). With MSAA on, Neural Rendering turns itself off and says
  so in the log

**Running it**

```powershell
$env:DLSS_SDK = 'C:\path\to\DLSS-SDK'                  
$env:WOW_DLSSNR_DLL = 'C:\path\to\nvngx_dlssnr.dll'   
cargo run -p benilla --features dlss
```

Tested on Windows 11 with an RTX 4070.

**Known limitations**
- The parameters for Feature 18 are not in NVIDIA's public SDK, so the tuning values are our own.
- Foliage motion vectors leave out wind sway, which can show as smearing on swaying trees and grass.

## Licence

Same as upstream benilla: MIT OR Apache-2.0. No original client code and no game assets are included;
you provide your own client.
