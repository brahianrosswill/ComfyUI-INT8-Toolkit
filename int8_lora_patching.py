import comfy.lora
import torch

try:
	from comfy.weight_adapter.lora import LoRAAdapter
	_LORA_ADAPTER_AVAILABLE = True
except Exception:
	LoRAAdapter = None
	_LORA_ADAPTER_AVAILABLE = False

try:
	from comfy.weight_adapter.base import WeightAdapterBase
	_WEIGHT_ADAPTER_BASE_AVAILABLE = True
except Exception:
	WeightAdapterBase = None
	_WEIGHT_ADAPTER_BASE_AVAILABLE = False


INT8_LORA_SIGNATURE_ATTACHMENT_KEY = "int8_lora_signature"


def _get_lora_signature(model_patcher):
	if not hasattr(model_patcher, "get_attachment"):
		return ()
	signature = model_patcher.get_attachment(INT8_LORA_SIGNATURE_ATTACHMENT_KEY)
	return signature if isinstance(signature, tuple) else ()


def _append_lora_signature(model_patcher, mode, lora_name, strength, seed=None):
	if not hasattr(model_patcher, "set_attachments"):
		return

	entry = (
		str(mode),
		str(lora_name).replace("\\", "/"),
		float(strength),
	)
	if seed is not None:
		entry = entry + (int(seed),)

	model_patcher.set_attachments(
		INT8_LORA_SIGNATURE_ATTACHMENT_KEY,
		_get_lora_signature(model_patcher) + (entry,),
	)


def _is_plain_lora_adapter(adapter):
	return _LORA_ADAPTER_AVAILABLE and isinstance(adapter, LoRAAdapter)


def _is_weight_adapter(adapter):
	return _is_plain_lora_adapter(adapter) or (_WEIGHT_ADAPTER_BASE_AVAILABLE and isinstance(adapter, WeightAdapterBase))


def _extract_layer_name(key):
	layer_name = key[0] if isinstance(key, tuple) else key
	if isinstance(layer_name, str) and layer_name.endswith(".weight"):
		layer_name = layer_name[:-7]
	return layer_name


def _candidate_module_names(layer_name):
	candidates = []
	names_to_expand = [layer_name]
	if layer_name.startswith("model."):
		names_to_expand.append(layer_name[len("model."):])

	for name in names_to_expand:
		candidates.append(name)
		if not name.startswith("diffusion_model."):
			candidates.append(f"diffusion_model.{name}")

	seen_names = set()
	output = []
	for candidate_name in candidates:
		if candidate_name in seen_names:
			continue
		seen_names.add(candidate_name)
		output.append(candidate_name)
	return output


def _resolve_target_module_cached(model_patcher, key, module_cache):
	layer_name = _extract_layer_name(key)
	if not isinstance(layer_name, str):
		raise TypeError("Unsupported key type for layer resolution")
	if layer_name in module_cache:
		return module_cache[layer_name]

	for object_patch_map_name in ("object_patches", "object_patches_backup"):
		object_patch_map = getattr(model_patcher, object_patch_map_name, None)
		if not isinstance(object_patch_map, dict):
			continue

		for candidate_name in _candidate_module_names(layer_name):
			if candidate_name in object_patch_map:
				target_module = object_patch_map[candidate_name]
				module_cache[layer_name] = target_module
				return target_module

	try:
		target_module = model_patcher.get_model_object(layer_name)
		module_cache[layer_name] = target_module
		return target_module
	except Exception:
		pass

	traversal_name = layer_name[len("model."):] if layer_name.startswith("model.") else layer_name
	parts = traversal_name.split(".")
	target_module = model_patcher.model.diffusion_model
	for part in parts[1:] if parts and parts[0] == "diffusion_model" else parts:
		if part.isdigit():
			target_module = target_module[int(part)]
		else:
			target_module = getattr(target_module, part)

	module_cache[layer_name] = target_module
	return target_module


def _get_key_map(model_patcher):
	key_map = {}
	if model_patcher.model.model_type.name != "ModelType.CLIP":
		key_map = comfy.lora.model_lora_keys_unet(model_patcher.model, key_map)
	return key_map


def _get_weight_scale_for_module(target_module):
	weight_scale = target_module.weight_scale
	if isinstance(weight_scale, torch.Tensor):
		return weight_scale.item() if weight_scale.numel() == 1 else weight_scale
	return weight_scale


def _mark_deferred_int8_patch(adapter):
	setattr(adapter, "_int8_defer_until_quantized", True)
	return adapter


def _wrap_adapter_for_stochastic(adapter, weight_scale, seed, outlier_method=None, hadanorm_sigma=None, defer_until_quantized=False):
	from .int8_quant import INT8LoRAPatchAdapter, INT8WeightPatchAdapter

	if not _is_weight_adapter(adapter):
		return adapter

	if _is_plain_lora_adapter(adapter):
		wrapped_adapter = INT8LoRAPatchAdapter(
			adapter.loaded_keys,
			adapter.weights,
			weight_scale,
			seed=seed,
			outlier_method=outlier_method,
			hadanorm_sigma=hadanorm_sigma,
		)
	else:
		wrapped_adapter = INT8WeightPatchAdapter(
			adapter,
			weight_scale,
			seed=seed,
			outlier_method=outlier_method,
			hadanorm_sigma=hadanorm_sigma,
		)

	if defer_until_quantized:
		return _mark_deferred_int8_patch(wrapped_adapter)

	return wrapped_adapter


def _can_merge_stochastic_stack(patches):
	return all(hasattr(adapter, "loaded_keys") and hasattr(adapter, "weights") for adapter, _ in patches)


def _create_stochastic_stack_adapter(patches, weight_scale, seed, outlier_method=None, hadanorm_sigma=None, defer_until_quantized=False):
	from .int8_quant import INT8MergedLoRAPatchAdapter

	merged_adapter = INT8MergedLoRAPatchAdapter(
		patches,
		weight_scale,
		seed=seed,
		outlier_method=outlier_method,
		hadanorm_sigma=hadanorm_sigma,
	)

	if defer_until_quantized:
		return _mark_deferred_int8_patch(merged_adapter)

	return merged_adapter


def _model_has_quantized_int8_modules(model_patcher):
	for object_patch_map_name in ("object_patches", "object_patches_backup"):
		object_patch_map = getattr(model_patcher, object_patch_map_name, None)
		if not isinstance(object_patch_map, dict):
			continue

		for _patch_key, module in object_patch_map.items():
			if getattr(module, "_is_quantized", False):
				return True

	diffusion_model = getattr(model_patcher.model, "diffusion_model", None)
	if diffusion_model is None:
		return False

	for _module_name, module in diffusion_model.named_modules():
		if getattr(module, "_is_quantized", False):
			return True
	return False


def _upgrade_patch_dict_for_int8(model_patcher, patch_dict, seed, module_cache, defer_unquantized=True):
	final_patch_dict = {}
	applied_count = 0

	for key, adapter in patch_dict.items():
		try:
			target_module = _resolve_target_module_cached(model_patcher, key, module_cache)
			is_quantized = hasattr(target_module, "_is_quantized") and target_module._is_quantized

			if _is_weight_adapter(adapter):
				if is_quantized:
					weight_scale = _get_weight_scale_for_module(target_module)
					outlier_method = getattr(target_module, "_outlier_method", None)
					hadanorm_sigma = getattr(target_module, "hadanorm_sigma", None)
					final_patch_dict[key] = _wrap_adapter_for_stochastic(
						adapter,
						weight_scale,
						seed,
						outlier_method=outlier_method,
						hadanorm_sigma=hadanorm_sigma,
						defer_until_quantized=False,
					)
				elif defer_unquantized:
					final_patch_dict[key] = _wrap_adapter_for_stochastic(
						adapter,
						1.0,
						seed,
						defer_until_quantized=True,
					)
				else:
					final_patch_dict[key] = adapter
				applied_count += 1
			else:
				final_patch_dict[key] = adapter
		except Exception:
			final_patch_dict[key] = adapter

	return final_patch_dict, applied_count


def _wrap_static_int8_patches(model_patcher, patch_dict, seed=318008, module_cache=None):
	if module_cache is None:
		module_cache = {}

	wrapped_patch_dict = {}
	for key, adapter in patch_dict.items():
		if not _is_weight_adapter(adapter):
			wrapped_patch_dict[key] = adapter
			continue

		try:
			target_module = _resolve_target_module_cached(model_patcher, key, module_cache)
			if not (hasattr(target_module, "_is_quantized") and target_module._is_quantized):
				wrapped_patch_dict[key] = adapter
				continue

			weight_scale = _get_weight_scale_for_module(target_module)
			outlier_method = getattr(target_module, "_outlier_method", None)
			hadanorm_sigma = getattr(target_module, "hadanorm_sigma", None)
			wrapped_patch_dict[key] = _wrap_adapter_for_stochastic(
				adapter,
				weight_scale,
				seed,
				outlier_method=outlier_method,
				hadanorm_sigma=hadanorm_sigma,
			)
		except Exception:
			wrapped_patch_dict[key] = adapter

	return wrapped_patch_dict
