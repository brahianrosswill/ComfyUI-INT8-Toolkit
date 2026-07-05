"""ConvRot-style regular Hadamard helpers for INT8 outlier mitigation.

This follows ComfyUI/comfy-kitchen's native ConvRot INT8 convention: rotate
weights offline with grouped regular Hadamard blocks, then apply the matching
activation rotation at runtime. The method is based on ConvRot's group-wise
regular Hadamard rotation and the broader QuaRot rotation-based quantization
lineage.
"""

import math

import torch


_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def build_hadamard(
	size: int,
	device: str | torch.device = "cpu",
	dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
	"""Build a normalized regular Hadamard matrix for ConvRot."""
	cache_key = (size, str(device), dtype)
	if cache_key in _HADAMARD_CACHE:
		return _HADAMARD_CACHE[cache_key]

	if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
		raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")

	h4 = torch.tensor(
		[
			[1, 1, 1, -1],
			[1, 1, -1, 1],
			[1, -1, 1, 1],
			[-1, 1, 1, 1],
		],
		dtype=dtype,
		device=device,
	)

	h_matrix = h4
	current_size = 4
	while current_size < size:
		h_matrix = torch.kron(h_matrix, h4)
		current_size *= 4

	h_matrix = h_matrix / (size**0.5)
	_HADAMARD_CACHE[cache_key] = h_matrix
	return h_matrix


def rotate_weight(weight: torch.Tensor, h_matrix: torch.Tensor, group_size: int) -> torch.Tensor:
	out_features, in_features = weight.shape
	if in_features % group_size != 0:
		raise ValueError(f"in_features {in_features} not divisible by group_size {group_size}")

	group_count = in_features // group_size
	weight_grouped = weight.view(out_features, group_count, group_size)
	return torch.matmul(weight_grouped, h_matrix.T.to(dtype=weight.dtype, device=weight.device)).reshape(weight.shape)
