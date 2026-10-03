//! The explicit bridge between Benilla's gamma-authored HDR world lane and Feature 18.
//!
//! Benilla's world values are deliberately gamma-coded even though the render target is
//! `Rgba16Float`; the UI/FFX lane relies on that contract. The bridge therefore copies values
//! verbatim into and out of NGX-owned storage images. Do not add an sRGB encode/decode here.

use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, ComputePassDescriptor, ComputePipelineDescriptor, Extent3d,
    PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, StorageTextureAccess, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsages, TextureView, TextureViewDimension,
};

const COPY_SHADER: &str = r#"
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var destination: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn copy(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(destination);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    textureStore(destination, vec2<i32>(id.xy), textureLoad(source, vec2<i32>(id.xy), 0));
}
"#;

#[cfg(feature = "dev")]
const DIFFERENCE_SHADER: &str = r#"
@group(0) @binding(0) var raw: texture_2d<f32>;
@group(0) @binding(1) var neural: texture_2d<f32>;
@group(0) @binding(2) var destination: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn difference(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(destination);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let pixel = vec2<i32>(id.xy);
    let error = abs(textureLoad(raw, pixel, 0).rgb - textureLoad(neural, pixel, 0).rgb);
    let amplified = min(max(error.r, max(error.g, error.b)) * 128.0, 1.0);
    textureStore(destination, pixel, vec4<f32>(vec3<f32>(amplified), 1.0));
}
"#;

pub struct Bridge {
    pub width: u32,
    pub height: u32,
    pub input: wgpu::Texture,
    pub input_view: TextureView,
    pub output: wgpu::Texture,
    pub output_view: TextureView,
    #[cfg(feature = "dev")]
    _comparison: wgpu::Texture,
    #[cfg(feature = "dev")]
    pub comparison_view: TextureView,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    #[cfg(feature = "dev")]
    difference_layout: wgpu::BindGroupLayout,
    #[cfg(feature = "dev")]
    difference_pipeline: wgpu::ComputePipeline,
}

impl Bridge {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("dlssnr_gamma_preserving_bridge_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("dlssnr_gamma_preserving_bridge"),
            source: ShaderSource::Wgsl(COPY_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("dlssnr_gamma_preserving_bridge_layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("dlssnr_gamma_preserving_bridge"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("copy"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        #[cfg(feature = "dev")]
        let difference_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("dlssnr_difference_bridge_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        #[cfg(feature = "dev")]
        let difference_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("dlssnr_difference_bridge"),
            source: ShaderSource::Wgsl(DIFFERENCE_SHADER.into()),
        });
        #[cfg(feature = "dev")]
        let difference_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("dlssnr_difference_bridge_layout"),
            bind_group_layouts: &[&difference_layout],
            push_constant_ranges: &[],
        });
        #[cfg(feature = "dev")]
        let difference_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("dlssnr_difference_bridge"),
            layout: Some(&difference_pipeline_layout),
            module: &difference_shader,
            entry_point: Some("difference"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        let make_texture = |label| {
            device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            })
        };
        let input = make_texture("dlssnr_input");
        let input_view = input.create_view(&wgpu::TextureViewDescriptor::default());
        let output = make_texture("dlssnr_output");
        let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());
        #[cfg(feature = "dev")]
        let comparison = make_texture("dlssnr_difference");
        #[cfg(feature = "dev")]
        let comparison_view = comparison.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            width,
            height,
            input,
            input_view,
            output,
            output_view,
            #[cfg(feature = "dev")]
            _comparison: comparison,
            #[cfg(feature = "dev")]
            comparison_view,
            layout,
            pipeline,
            #[cfg(feature = "dev")]
            difference_layout,
            #[cfg(feature = "dev")]
            difference_pipeline,
        }
    }

    pub fn copy(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source: &TextureView,
        destination: &TextureView,
    ) {
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("dlssnr_gamma_preserving_bridge_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(destination),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("dlssnr_gamma_preserving_bridge"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
    }

    #[cfg(feature = "dev")]
    pub fn difference(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        raw: &TextureView,
        neural: &TextureView,
    ) {
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("dlssnr_difference_bridge_bind_group"),
            layout: &self.difference_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(raw),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(neural),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.comparison_view),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("dlssnr_difference_bridge"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.difference_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
    }
}
