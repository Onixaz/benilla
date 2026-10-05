# benilla — enhanced

A fork of [**benilla**](https://github.com/samwhosung/benilla), the from-scratch World of Warcraft 1.12.1
client in Rust and Bevy by samwhosung. **[Read the original benilla README →](https://github.com/samwhosung/benilla/blob/main/README.md)**

Upstream benilla tracks the 1.12.1 client exactly, so changes that go beyond it are scattered across forks. This fork tries to bring them together in one place: work from other forks (credited in each section below) and its own, in four areas:

1. **[Turtle WoW support](#turtle-wow-support)**: log in, create characters and play on Turtle WoW, including its races, auction house and transmog.
2. **[Enhanced graphics](#enhanced-graphics)**: an optional modern look (shadows, modern fog, volumetric light, water, weather). The **Classic** preset keeps the original 1.12 image.
3. **[HD models and textures](#hd-models-and-textures)**: support for HD asset packs such as Project Reforged.
4. **[DLSS 5 Neural Rendering](#dlss-5-neural-rendering-experimental)** *(experimental)*: NVIDIA's neural renderer applied to the world, on RTX cards.


Everything else (formats, networking, the stock interface, addons, audio) is upstream benilla, merged in
regularly.

**Want to modernize 1.12.1? This is the place.** New graphics, private-server support, quality-of-life
features: anything that goes beyond the original client is welcome here as an issue or a pull request.
This repository is maintained and will stay open source.

**Fixing benilla itself?** Bugs and changes that make benilla more like the real 1.12.1 client belong in
[upstream benilla](https://github.com/samwhosung/benilla/pulls): open the pull request there, and it
reaches this fork with the next merge.

**Questions?** Find me as **Dzoziz** on the [benilla Discord](https://discord.gg/wJSJx467G4).

## Enhanced graphics

An optional modern look: realtime sun and moon shadows, lit interiors and flickering torch light, a
modern sky and fog with volumetric light shafts, bloom and per-zone colour grading, enhanced water,
rain and wind, and render distance up to 1497 yards.

Pick a **Graphics Preset** (Classic / Low / Medium / High / Ultra) in the options, or switch each feature
yourself under **Options → Advanced Graphics**. **Classic** keeps the original 1.12 image.

Based on https://github.com/pkuzic/benilla-everwood_graphics

## Turtle WoW support

benilla plays on Turtle WoW: logging in, creating characters (including Turtle's races), the auction
house and transmog all work against Turtle's servers. Point it at your own Turtle WoW install; benilla
reads the game data from it and never writes into it.

Turtle's login server only accepts its own client build, so set it before launching:

```powershell
$env:WOW_LOGIN_BUILD = 7272
```

Without it, benilla presents the stock 1.12.1 build (5875), as for any vanilla server.

Initial work from https://github.com/jhinzuo2/benilla-twow

## HD models and textures

benilla loads HD asset packs such as Project Reforged from your install's patch chain, like the game
does: higher-resolution character skins and overlays (VanillaHelpers' larger character atlas, at any
power-of-two size) and HD models. Install the
pack into your WoW folder as its instructions say; benilla needs no extra setting.

## DLSS 5 Neural Rendering (experimental)

benilla can run NVIDIA's DLSS 5 **Neural Rendering** (NGX Feature 18) over the world view. It is not
the DLSS upscaler: the image stays at native resolution and the network re-renders the lit scene.
It is off unless you build for it. It enhances the lit world, including its realtime shadows and ambient occlusion; the interface, spell effects, water and fog are drawn on top of its output unchanged.

Based on https://github.com/AlrikOlson/bevy_dlss5

**Requirements**
- Windows and an NVIDIA RTX GPU (40 or 50 series), Vulkan
- The [NVIDIA DLSS SDK](https://github.com/NVIDIA/DLSS) (tested with 310.4.0 and 310.9.1), at build time
  only, named by `DLSS_SDK`
- The Feature-18 runtime, `nvngx_dlssnr.dll`. It is not in the public SDK; NVIDIA ships it with games that support DLSS 5, so copy it from one you own.
  distribute it; you provide it yourself.
- MSAA off (`gxMultisample 1`, the default). With MSAA on, Neural Rendering turns itself off and says
  so in the log

**Running it**

```powershell
$env:DLSS_SDK = 'C:\path\to\DLSS-SDK'                  
$env:WOW_DLSSNR_DLL = 'C:\path\to\nvngx_dlssnr.dll'   
cargo run -p benilla --features dlss
```

`DLSS_SDK` is read only when building with `--features dlss`, and `WOW_DLSSNR_DLL` only when running
such a build. A normal build or `cargo run` needs neither, and never links or loads anything from
NVIDIA.

Tested on Windows 11 with an RTX 4070.

**Comparing**: press `Ctrl+Alt+N` in game to switch between Neural Rendering and the raw image (in
the default dev build; a `--no-default-features` player build leaves it out).

**Known limitations**
- The parameters for Feature 18 are not in NVIDIA's public SDK, so the tuning values are our own.
- Foliage motion vectors leave out wind sway, which can show as smearing on swaying trees and grass.

## Licence

Same as upstream benilla: MIT OR Apache-2.0. No original client code and no game assets are included;
you provide your own client.
