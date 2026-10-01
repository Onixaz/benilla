// MONKEY (post): world-only HDR excess extraction, dual-filter blur, and halo composite.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var source_tex: texture_2d<f32>;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var auxiliary_tex: texture_2d<f32>;

struct BloomParams {
    tier: f32,
    gain: f32,
    unused: vec2<f32>,
};
@group(0) @binding(3) var<uniform> bloom: BloomParams;

fn tap(tex: texture_2d<f32>, uv: vec2<f32>, offset: vec2<f32>) -> vec4<f32> {
    let size = vec2<f32>(textureDimensions(tex));
    return textureSampleLevel(tex, linear_sampler, uv + offset / size, 0.0);
}

@fragment
fn fs_extract(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    // 13-tap downfilter: a broad, stable footprint when Low uses a quarter-size target.
    // MONKEY (fix-post): the footprint scales with the target divisor (Low = quarter size, so
    // 2x offsets), else 16 source pixels fed each output from a 5x5 area and thin sparks shimmered.
    let spread = select(1.0, 2.0, bloom.tier < 1.5);
    var c = tap(source_tex, in.uv, vec2(0.0)) * 0.20;
    c += tap(source_tex, in.uv, vec2( 1.0,  0.0) * spread) * 0.10;
    c += tap(source_tex, in.uv, vec2(-1.0,  0.0) * spread) * 0.10;
    c += tap(source_tex, in.uv, vec2( 0.0,  1.0) * spread) * 0.10;
    c += tap(source_tex, in.uv, vec2( 0.0, -1.0) * spread) * 0.10;
    c += tap(source_tex, in.uv, vec2( 1.0,  1.0) * spread) * 0.06;
    c += tap(source_tex, in.uv, vec2(-1.0,  1.0) * spread) * 0.06;
    c += tap(source_tex, in.uv, vec2( 1.0, -1.0) * spread) * 0.06;
    c += tap(source_tex, in.uv, vec2(-1.0, -1.0) * spread) * 0.06;
    c += tap(source_tex, in.uv, vec2( 2.0,  0.0) * spread) * 0.04;
    c += tap(source_tex, in.uv, vec2(-2.0,  0.0) * spread) * 0.04;
    c += tap(source_tex, in.uv, vec2( 0.0,  2.0) * spread) * 0.04;
    c += tap(source_tex, in.uv, vec2( 0.0, -2.0) * spread) * 0.04;
    // MONKEY (fix-post): cap the excess so a +inf pixel cannot become an inf/inf NaN halo.
    let excess = min(max(c.rgb - vec3(1.0), vec3(0.0)), vec3(64.0));
    let knee = excess * excess / (excess + vec3(0.50));
    // Frost-Nova/firework stacks can reach 5-10x; compress them without hard clipping.
    let capped = vec3(5.0) * (vec3(1.0) - exp(-knee / vec3(5.0)));
    return vec4(capped, 1.0);
}

fn blur_axis(uv: vec2<f32>, axis: vec2<f32>) -> vec4<f32> {
    var c = tap(source_tex, uv, vec2(0.0)) * 0.227027;
    c += tap(source_tex, uv, axis * 1.384615) * 0.316216;
    c += tap(source_tex, uv, axis * -1.384615) * 0.316216;
    c += tap(source_tex, uv, axis * 3.230769) * 0.070270;
    c += tap(source_tex, uv, axis * -3.230769) * 0.070270;
    return c;
}

@fragment
fn fs_blur_h(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return blur_axis(in.uv, vec2(1.0, 0.0));
}

@fragment
fn fs_blur_v(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return blur_axis(in.uv, vec2(0.0, 1.0));
}

@fragment
fn fs_combine(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let scene = textureSample(source_tex, linear_sampler, in.uv);
    let halo = textureSample(auxiliary_tex, linear_sampler, in.uv).rgb;
    return vec4(scene.rgb + halo * bloom.gain, scene.a);
}
