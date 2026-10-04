// The character-shadow proxy's forward pass. The proxy exists only for the sun's shadow pass, but
// the camera must carry layer 31 for that pass to queue it, so the camera's draw collapses every
// vertex outside the clip volume: no triangle rasterises and no fragment runs.
@vertex
fn vertex() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, -1.0, 1.0);
}

@fragment
fn fragment() -> @location(0) vec4<f32> {
    discard;
    return vec4<f32>(0.0);
}
