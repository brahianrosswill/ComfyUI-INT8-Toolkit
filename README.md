# ComfyUI-INT8-Toolkit

INT8 quantization stores model weights in 8-bit integers instead of higher-precision formats. On GPUs with strong INT8 throughput, this can reduce VRAM use and speed up transformer-heavy diffusion models. The tradeoff is that quantization changes model numerics, so layer targeting, LoRA order, runtime backend, outlier handling, and Torch Compile behavior all matter.

This project began as a fork of [ComfyUI-INT8-Fast](https://github.com/BobJohnson24/ComfyUI-INT8-Fast), but it is now maintained as its own INT8 toolkit for ComfyUI.

![workflow](example_workflows/load_workflow.png)

## Current INT8 Landscape

ComfyUI now has native INT8 checkpoint loading through the standard `Load Diffusion Model` node when a checkpoint contains native `.comfy_quant` metadata. That is the best path when you already have a native-format INT8 checkpoint and do not need extra conversion, INT8-specific LoRA behavior, or Toolkit runtime controls.

The Toolkit remains useful when you want:

- `Enable INT8 on MODEL`: convert a stock-loaded `MODEL` after regular loaders and stock LoRA nodes have run.
- On-the-fly INT8 quantization from float or FP8 checkpoints.
- Architecture-specific exclusion presets.
- INT8-aware LoRA modes: `Stochastic`, `Dynamic`, and `Standard`.
- Lazy Torch Compile behavior tuned around INT8 object patches.
- Runtime backend controls, small-batch fallback, diagnostics, and native-format export for compatible INT8 layers.

| Project | Best Fit | Notes |
| --- | --- | --- |
| ComfyUI core INT8 | Native-format pre-quantized checkpoints loaded with stock `Load Diffusion Model`. | No special Toolkit nodes required. Stock LoRA patching is generic and can disable the fast quantized path for patched layers. |
| ComfyUI-INT8-Fast | Historical upstream implementation and preset source. | Upstream is effectively retired now that ComfyUI has native INT8 loading. This Toolkit selectively adapts useful preset/runtime work. |
| ComfyUI-INT8-Toolkit | Conversion, stock-loader adaptation, INT8-aware LoRA handling, compile/runtime tuning, and experimentation. | Use this when you need more control than core exposes or want to quantize models locally. |

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Recommended Workflows

### Stock Loader Compatibility

Use this when your graph is built around ComfyUI's stock loaders.

```text
Load Diffusion Model
-> optional stock Load LoRA nodes
-> Enable INT8 on MODEL
-> optional INT8 Lazy Torch Compile
-> sampler
```

With `bake_loaded_loras` enabled, `Enable INT8 on MODEL` applies stock LoRA weight patches in float space, quantizes the resulting layer weights, and removes consumed patches so they are not applied twice. Bias patches and excluded-layer patches are left for ComfyUI to handle normally.

### Pre-Quantized Or On-The-Fly INT8

Use this when you are already loading an INT8 checkpoint, or when you want the extension to quantize eligible layers during model load.

```text
Load Diffusion Model INT8 (W8A8)
-> optional Load LoRA INT8
-> optional INT8 Lazy Torch Compile
-> sampler
```

### Add Or Swap LoRAs After INT8

Use this when the model is already INT8 and you want to add or change LoRAs without re-running the stock-loader bake step.

```text
INT8 model
-> Load LoRA INT8
-> sampler
```

`Load LoRA INT8` modes:

- `Stochastic`: applies the LoRA delta into INT8 weights with stochastic rounding.
- `Dynamic`: keeps compatible plain LoRAs as runtime additions instead of modifying INT8 weights.
- `Standard`: applies LoRA through ComfyUI's regular MODEL patch path without INT8-specific handling.

## Nodes

### Enable INT8 on MODEL

Converts an already-loaded diffusion `MODEL` to INT8 by object-patching eligible linear layers.

Key settings:

- `model_type`: defaults to `auto`, which inspects the loaded `MODEL` and selects a known exclusion preset when possible. Use `none` only for experiments.
- `outlier_method`: choose `none`, `convrot`, `quarot`, or `hadanorm`.
- `small_batch_fallback`: defaults to `only_small_layers`, which avoids slow full-weight dequantization on large layers.
- `runtime_backend`: defaults to `torch_int_mm`; `triton` is available for shape-dependent testing.
- `prepack_int8_weights`: experimental extra transposed INT8 weight buffer for the Triton path.
- `bake_loaded_loras`: applies current stock LoRA weight patches before quantization and removes consumed patches.

### Load Diffusion Model INT8 (W8A8)

Loads INT8 diffusion models using `Int8TensorwiseOps` and architecture-specific exclusion presets. It also supports on-the-fly quantization of eligible float or FP8 weights.

Supported `model_type` presets:

- `anima`
- `boogu`
- `chroma`
- `ernie`
- `flux2`
- `flux2_fast_unsafe`
- `hidream o1`
- `ideogram4`
- `krea2`
- `ltx2`
- `qwen`
- `sdxl`
- `wan`
- `z-image`

`flux2_fast_unsafe` is opt-in and less conservative. It is mainly useful for experiments where speed matters more than defensive layer targeting.

### INT8 Lazy Torch Compile

Lazily applies `torch.compile` at the first sampling call, after Comfy object patches such as INT8 module replacement are active.

Recommended placement:

```text
Enable INT8 on MODEL
-> INT8 Lazy Torch Compile
-> sampler
```

This node can compile recognized repeated transformer block lists instead of the whole diffusion model, apply Comfy-style guard filtering, and raise TorchDynamo cache limits for workflows with many compiled modules.

### Load LoRA INT8 And Load LoRA Stack INT8

Use these nodes when LoRAs need to be applied after the model is already INT8. In `Stochastic` stack mode, compatible LoRAs are combined before one stochastic rounding step, which is usually better than repeatedly rounding each LoRA one by one.

### INT8 Kernel Config

Applies fixed Triton kernel settings at runtime. Optional microbench mode tests candidate configs and prints environment variable values that can be reused later.

## Outlier Methods

| Method | Behavior | Native Core Export |
| --- | --- | --- |
| `none` | Direct per-row INT8 quantization. Fastest path and default. | Yes |
| `convrot` | ConvRot-style regular Hadamard rotation with 256-channel groups. Good quality/speed tradeoff and aligns with ComfyUI native metadata. | Yes |
| `quarot` | Toolkit legacy Hadamard rotation with 128-channel groups. | Toolkit loader required |
| `hadanorm` | Experimental per-channel scaling plus Hadamard mixing and runtime correction. | Toolkit loader required |

The Toolkit's `convrot` path is intended to match ComfyUI/comfy-kitchen native ConvRot INT8 semantics: compatible weights are rotated with grouped regular Hadamard blocks before quantization, and activations are rotated at runtime with the matching transform. This is compatibility work, not a new quantization method; it follows the ConvRot paper's group-wise regular Hadamard rotation and the broader QuaRot rotation-based quantization lineage.

For saved checkpoints, plain INT8 and ConvRot layers receive native `.comfy_quant` metadata. QuaRot and HadaNorm layers intentionally do not, because ComfyUI core does not know their Toolkit-specific activation transforms.

## LoRA Order

Some LoRA orders can temporarily materialize large float tensors. On lower-VRAM cards this can look like a sudden spike and may OOM even if normal sampling would fit.

| LoRA method | Before `Enable INT8 on MODEL` | After `Enable INT8 on MODEL` |
| --- | --- | --- |
| Stock `Load LoRA` | Recommended for stock workflows. Bake with `Enable INT8 on MODEL`. | Avoid for INT8 layers unless testing; core's generic patch path may dequantize patched weights. |
| `Load LoRA INT8` with `Standard` | Useful for pre-INT8 A/B testing. | Mainly for testing. It intentionally skips INT8-specific handling. |
| `Load LoRA INT8` with `Stochastic` | Carries deferred INT8-aware patches. | Preferred speed-oriented post-INT8 LoRA mode. |
| `Load LoRA INT8` with `Dynamic` | Not the preferred order. | Useful when compatible and runtime LoRA matmuls are acceptable. |

Practical defaults:

- Put stock `Load LoRA` before `Enable INT8 on MODEL`.
- Use `Load LoRA INT8` after INT8 is enabled.
- Leave `bake_loaded_loras` enabled unless you intentionally want patched layers skipped by the adapter.

## Runtime Guidance

- `torch_int_mm` is the default because it is simple, robust on Windows, and works well on current Toolkit test workflows.
- `triton` can still be faster on some architecture and shape mixes.
- `triton_legacy_unsafe` is only for diagnosis and may produce incorrect output on tail shapes.
- `small_batch_fallback=only_small_layers` is the recommended default.
- Keep `INT8_RUNTIME_STATS=0` for normal benchmarking; diagnostics add console overhead.

Torch Compile is often the difference between "INT8 works" and "INT8 is actually fast" in ComfyUI. Put `INT8 Lazy Torch Compile` after `Enable INT8 on MODEL`, use `compile_transformer_blocks_only=True` unless an architecture needs whole-model compilation, and restart ComfyUI after failed compile experiments before drawing conclusions.

## Model Save

`Save Model INT8 (DynamicVRAM Safe)` saves Toolkit INT8-patched `MODEL` outputs.

- Plain INT8 and ConvRot layers include native ComfyUI `.comfy_quant` metadata.
- QuaRot and HadaNorm layers are Toolkit-specific and should be reloaded with `Load Diffusion Model INT8 (W8A8)`.
- The save node prints counts for INT8 weights, `weight_scale` tensors, and native `.comfy_quant` layers so compatibility is visible.

## Checkpoint Notes

Pre-quantized checkpoints are still useful when available. On-the-fly quantization is more flexible, but it requires loading source weights and quantizing them locally.

Vistralis checkpoints:

| Model | Link |
| --- | --- |
| FLUX.2-klein-base-9b | [Download](https://huggingface.co/vistralis/FLUX.2-klein-base-9b-INT8-transformer) |
| FLUX.2-klein-base-4b | [Download](https://huggingface.co/vistralis/FLUX.2-klein-base-4b-INT8-transformer) |
| FLUX.2-klein-9b | [Download](https://huggingface.co/vistralis/FLUX.2-klein-9b-INT8-transformer) |
| FLUX.2-klein-4b | [Download](https://huggingface.co/vistralis/FLUX.2-klein-4b-INT8-transformer) |

Additional checkpoints:

| Model | Link |
| --- | --- |
| Chroma1-HD | [Download](https://huggingface.co/bertbobson/Chroma1-HD-INT8Tensorwise) |
| Z-Image-Turbo | [Download](https://huggingface.co/bertbobson/Z-Image-Turbo-INT8-Tensorwise) |
| Anima | [Download](https://huggingface.co/bertbobson/Anima-INT8-QUIP) |

## Requirements

- Recent ComfyUI
- NVIDIA GPU with useful INT8 throughput
- PyTorch build compatible with your ComfyUI install
- `triton-windows` for the optional fused Triton backend on Windows

Windows note: use the Triton build that matches your PyTorch/CUDA stack. In the tested Comfy Anaconda environment, PyTorch `2.8.0+cu126` imports Triton `3.4.0` from `triton-windows 3.4.0.post21`.

## Credits

- dxqb / OneTrainer INT8 work: https://github.com/Nerogar/OneTrainer/pull/1034
- ConvRot paper, "Rotation-Based Plug-and-Play 4-bit Quantization for Diffusion Transformers": https://arxiv.org/abs/2512.03673
- QuaRot paper, "Outlier-Free 4-Bit Inference in Rotated LLMs": https://arxiv.org/abs/2404.00456
- ComfyUI and comfy-kitchen native INT8/ConvRot compatibility references: https://github.com/Comfy-Org/ComfyUI and https://github.com/Comfy-Org/comfy-kitchen
- silveroxides / convert_to_quant: https://github.com/silveroxides/convert_to_quant
- silveroxides / ComfyUI-QuantOps: https://github.com/silveroxides/ComfyUI-QuantOps
- newgrit1004 / QuaRot reference code: https://github.com/newgrit1004/ComfyUI-ZImage-Triton
