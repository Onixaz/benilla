# Benilla patch notes

Vendored from `bevy_anti_alias` 0.18.1, under its included MIT OR Apache-2.0 license.

The only source changes are in the DLSS Super Resolution path:

- `src/dlss/prepare.rs` retains `LowResolutionMotionVectors` and `InvertedDepth`, but omits
  `HighDynamicRange` and `AutoExposure`.
- `src/dlss/node.rs` submits the neutral fixed exposure `(scale: 1.0, pre-exposure: 1.0)` instead
  of DLSS automatic exposure.

Benilla's world color texture is `Rgba16Float` for precision but carries the reference client's
gamma-space scene values. FFXGlow consumes the reconstructed image afterwards and performs the
frame's sole gamma-to-linear decode. This patch keeps DLSS's declared input and exposure behavior
consistent with that contract while leaving the rest of Bevy's DLSS implementation unchanged.
