# Changelog

All notable changes to this project will be documented in this file.

<details open><summary>1.8.1 - 6 July 2026</summary>

### Fixed

- Fixed post-quantization INT8 LoRA stack application for Krea2 and other adapter-quantized MODEL workflows by resolving pending INT8 object patches before falling back to live modules.
- Fixed stochastic and dynamic LoRA stack cache signatures so changed LoRA sets do not reuse stale adapter outputs.
- Fixed LoRA adapter wrapping detection so INT8 LoRA patches do not fall through to ComfyUI's generic quantized weight patching path.

</details>

<details><summary>1.8.0 - 2 July 2026</summary>

### Added

- Added `convrot` as an outlier method for Toolkit on-the-fly quantization and `Enable INT8 on MODEL`.
- Added Toolkit runtime support for native ComfyUI ConvRot INT8 checkpoints by reading `convrot` and `convrot_groupsize` from `.comfy_quant` metadata.
- Added native `.comfy_quant` export for Toolkit plain INT8 and ConvRot INT8 layers.
- Added `boogu` and `krea2` architecture presets.
- Added native-save diagnostics for full, partial, or missing `.comfy_quant` coverage.

### Changed

- Expanded Wan Animate exclusions with `face_adapter`, `face_encoder`, `motion_encoder`, and `pose_patch_embedding`.
- Broadened the `ltx2` exclusion preset for LTX 2.3-sensitive layers.
- Updated README guidance around ComfyUI core INT8, upstream INT8-Fast, and Toolkit-specific workflows.
- Clarified ConvRot attribution and native ComfyUI/comfy-kitchen compatibility framing.
- Moved release-history notes out of the README and into this changelog.

### Compatibility

- Plain INT8 and ConvRot Toolkit saves can include native ComfyUI metadata.
- QuaRot and HadaNorm remain Toolkit-specific because ComfyUI core does not implement their activation transforms.

</details>

<details><summary>1.7.0 - Development Runtime Update</summary>

### Added

- Added `runtime_backend` with `torch_int_mm`, `triton`, and diagnostic `triton_legacy_unsafe` modes.
- Added `small_batch_fallback` with `only_small_layers`, `always`, and `never`.
- Added CUDA-safe padding for `torch._int_mm` tiny-row and non-8-aligned output cases.
- Added experimental `prepack_int8_weights`.
- Added `INT8 Lazy Torch Compile`.
- Added stable Dynamic LoRA patch UUIDs to avoid unnecessary recomposition.
- Added `hidream o1`, `sdxl`, and opt-in `flux2_fast_unsafe` presets.

### Changed

- Changed the default backend to `torch_int_mm`.
- Reordered node inputs so `bake_loaded_loras` and logging controls are near the bottom.
- Alphabetized model type lists with `auto` first and `none` last where applicable.
- Improved cache-reuse logs and runtime diagnostics.
- Restored prior INT8 object patches before requantizing.

### Fixed

- Fixed Triton edge tiles so tail shapes no longer wrap reads with modulo offsets.
- Fixed module eligibility issues for Flux2.
- Added Torch Compile compatibility fixes.
- Added `Standard` LoRA mode for stock-style A/B comparisons.

</details>

<details><summary>0.1.1 - Early Maintenance</summary>

### Changed

- Added INT8 kernel config tuner and runtime optimization work.
- Published the extension to the Comfy Registry.

</details>

<details><summary>0.1.0 - Initial Release</summary>

### Added

- Initial Toolkit fork with INT8 tensorwise loading, on-the-fly quantization, and LoRA support.

</details>
