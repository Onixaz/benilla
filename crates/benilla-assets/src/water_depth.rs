//! Shared water settings and image binding, independent of world streaming.
use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;

/// The enhanced water module's uniform (binding 104, `WaterParams` in `enhanced_water.wgsl`).
/// Field order is the shader's; change the two together.
#[derive(ShaderType, Clone, Copy, Debug, PartialEq)]
pub struct WaterUniform {
    /// x = quality (0 Classic, 1 Enhanced, 2 High); y = wave energy (ocean 1.0, ADT inland 0.18,
    /// WMO pools 0.12); z = the pinned capture time; w = clock enable (0 on a deterministic run).
    pub mode: Vec4,
    /// x = which renderer (0 ADT MCLQ, 1 WMO exterior, 2 WMO interior); y = ocean; z = fullbright.
    pub lane: Vec4,
    /// Reflection endpoints, linear RGB from the dome's resolved LightIntBand 2 / 6.
    pub sky_zenith: Vec4,
    pub sky_horizon: Vec4,
    /// xyz toward the visible sun by day, the white moon by night; w = 0 sun, 1 moon.
    pub celestial: Vec4,
}

impl Default for WaterUniform {
    fn default() -> Self {
        Self {
            mode: Vec4::new(0.0, 0.0, 0.0, 1.0),
            lane: Vec4::ZERO,
            sky_zenith: Vec4::ZERO,
            sky_horizon: Vec4::ZERO,
            celestial: Vec3::Y.extend(0.0),
        }
    }
}

/// 0 = Classic, 1 = Enhanced, 2 = High (Enhanced + screen-space reflection of the scenery).
#[derive(Resource, Clone, Copy, PartialEq)]
pub struct WaterQuality(pub u8);

impl Default for WaterQuality {
    fn default() -> Self {
        Self(1)
    }
}

/// The world view's opaque depth, resolved to a sampleable R32Float image.
#[derive(Resource, Clone)]
pub struct WaterDepthImage(pub Handle<Image>);

impl FromWorld for WaterDepthImage {
    fn from_world(world: &mut World) -> Self {
        use bevy::{asset::RenderAssetUsages, render::render_resource::*};
        let mut image = Image::new_fill(
            Extent3d::default(),
            TextureDimension::D2,
            &0f32.to_le_bytes(),
            TextureFormat::R32Float,
            RenderAssetUsages::default(),
        );
        image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
            | TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::COPY_DST;
        // Render-only: no CPU copy, so a resize does not allocate (and upload) a screen of zeros.
        image.data = None;
        Self(world.resource_mut::<Assets<Image>>().add(image))
    }
}

/// The world view's opaque COLOUR, copied right after the main opaque pass (the frame as it stood
/// before any water drew), for the enhanced water's refraction. `Rgba16Float`: a load and a store,
/// never filtered, so the copy is exact whatever the view target's own format is.
#[derive(Resource, Clone)]
pub struct WaterColourImage(pub Handle<Image>);

impl FromWorld for WaterColourImage {
    fn from_world(world: &mut World) -> Self {
        use bevy::{asset::RenderAssetUsages, render::render_resource::*};
        let mut image = Image::new_fill(
            Extent3d::default(),
            TextureDimension::D2,
            &[0u8; 8],
            TextureFormat::Rgba16Float,
            RenderAssetUsages::default(),
        );
        image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
            | TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::COPY_DST;
        // Render-only: no CPU copy, so a resize does not allocate (and upload) a screen of zeros.
        image.data = None;
        Self(world.resource_mut::<Assets<Image>>().add(image))
    }
}
