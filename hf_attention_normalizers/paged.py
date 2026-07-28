from typing import Callable, Optional, Union

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from .backends import resolve_softmax_fn
from .kernels.paged_sparse_entmax_hopper import paged_hopper_sparse_attention
from .kernels.paged_softmax1_hopper import paged_hopper_softmax1_attention


class DifferentiablePagedCacheLayer(CacheLayerMixin):
    """Functional paged cache layer that preserves gradients through writes."""

    def __init__(
        self,
        block_table: torch.Tensor,
        block_size: int,
        num_key_value_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        if block_table.ndim != 2:
            raise ValueError("block_table must have shape [batch, max_blocks].")
        self.block_table = block_table.to(device=device, dtype=torch.long)
        self.block_size = block_size
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.max_batch_size = block_table.shape[0]
        self.max_cache_len = block_table.shape[1] * block_size
        self.cumulative_length = 0
        self.lazy_initialization(
            torch.empty(
                self.max_batch_size,
                num_key_value_heads,
                0,
                head_dim,
                dtype=dtype,
                device=device,
            )
        )

    def lazy_initialization(self, key_states: torch.Tensor):
        num_physical_blocks = int(self.block_table.max().item()) + 1
        shape = (
            num_physical_blocks * self.block_size,
            self.num_key_value_heads,
            self.head_dim,
        )
        self.keys = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.values = torch.zeros_like(self.keys)
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key_states.shape != value_states.shape or key_states.ndim != 4:
            raise ValueError("K/V updates must have matching [batch, heads, sequence, dim] shapes.")
        if key_states.shape[:2] != (self.max_batch_size, self.num_key_value_heads):
            raise ValueError("K/V update batch or head count does not match the paged cache.")
        cache_kwargs = cache_kwargs or {}
        cache_position = cache_kwargs.get("cache_position")
        if cache_position is None:
            cache_position = torch.arange(
                self.cumulative_length,
                self.cumulative_length + key_states.shape[-2],
                device=self.device,
            )
        cache_position = cache_position.to(device=self.device, dtype=torch.long)
        if cache_position.ndim != 1 or cache_position.numel() != key_states.shape[-2]:
            raise ValueError("cache_position must have one logical position per update token.")
        if int(cache_position.max().item()) >= self.max_cache_len:
            raise ValueError("cache_position exceeds the logical paged-cache capacity.")

        physical_blocks = self.block_table[:, cache_position // self.block_size]
        physical_indices = physical_blocks * self.block_size + cache_position.remainder(self.block_size)
        flat_indices = physical_indices.reshape(-1)
        if torch.unique(flat_indices).numel() != flat_indices.numel():
            raise ValueError("Two batch entries cannot write the same physical cache page.")

        flat_keys = key_states.transpose(1, 2).reshape(
            -1,
            self.num_key_value_heads,
            self.head_dim,
        )
        flat_values = value_states.transpose(1, 2).reshape_as(flat_keys)
        self.keys = torch.index_copy(self.keys, 0, flat_indices, flat_keys)
        self.values = torch.index_copy(self.values, 0, flat_indices, flat_values)
        self.cumulative_length = max(
            self.cumulative_length,
            int(cache_position[-1].item()) + 1,
        )

        logical_positions = torch.arange(self.cumulative_length, device=self.device)
        read_indices = (
            self.block_table[:, logical_positions // self.block_size] * self.block_size
            + logical_positions.remainder(self.block_size)
        )
        keys = self.keys[read_indices].permute(0, 2, 1, 3)
        values = self.values[read_indices].permute(0, 2, 1, 3)
        return keys, values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        new_length = int(cache_position[-1].item()) + 1 if cache_position.numel() else self.cumulative_length
        return max(self.cumulative_length, new_length), 0

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len

    def reset(self) -> None:
        self.keys = torch.zeros_like(self.keys)
        self.values = torch.zeros_like(self.values)
        self.cumulative_length = 0


class DifferentiablePagedCache(Cache):
    """HuggingFace Cache using functional physical-page K/V updates."""

    def __init__(
        self,
        num_hidden_layers: int,
        block_table: torch.Tensor,
        block_size: int,
        num_key_value_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        layers = [
            DifferentiablePagedCacheLayer(
                block_table,
                block_size,
                num_key_value_heads,
                head_dim,
                dtype,
                device,
            )
            for _ in range(num_hidden_layers)
        ]
        super().__init__(layers=layers)


def paged_cache_indices(
    block_table: torch.Tensor,
    sequence_lengths: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build physical token indices for a paged KV cache.

    ``block_table`` has shape ``[batch, max_blocks]`` and contains physical
    block numbers. Returned indices and validity mask both have shape
    ``[batch, max_sequence_length]``.
    """
    if block_table.ndim != 2:
        raise ValueError("block_table must have shape [batch, max_blocks].")
    if sequence_lengths.ndim != 1 or sequence_lengths.shape[0] != block_table.shape[0]:
        raise ValueError("sequence_lengths must have shape [batch].")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if bool((sequence_lengths <= 0).any()):
        raise ValueError("Every sequence length must be positive.")

    max_sequence_length = int(sequence_lengths.max().item())
    logical_positions = torch.arange(max_sequence_length, device=block_table.device)
    logical_blocks = logical_positions // block_size
    if max_sequence_length > block_table.shape[1] * block_size:
        raise ValueError("block_table does not contain enough blocks for sequence_lengths.")

    physical_blocks = block_table[:, logical_blocks]
    physical_indices = physical_blocks * block_size + logical_positions.remainder(block_size)
    valid = logical_positions.unsqueeze(0) < sequence_lengths.unsqueeze(1)
    return physical_indices, valid


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    sequence_lengths: torch.Tensor,
    block_size: int,
    normalizer: Union[str, Callable[..., torch.Tensor]] = "softmax1",
    scale: Optional[float] = None,
    is_causal: bool = True,
) -> torch.Tensor:
    """Differentiable paged attention reference implementation.

    Args:
        query: ``[batch, query_heads, query_length, head_dim]``.
        key_cache/value_cache: flattened physical cache tensors with shape
            ``[num_blocks * block_size, kv_heads, head_dim]``.
        block_table: physical block numbers with shape ``[batch, max_blocks]``.
        sequence_lengths: logical KV lengths with shape ``[batch]``.

    Gradients for K/V are accumulated back into their physical cache pages by
    PyTorch's indexed-gather backward, including when a physical page is
    referenced more than once.
    """
    if query.ndim != 4:
        raise ValueError("query must have shape [batch, heads, sequence, dim].")
    if key_cache.ndim != 3 or value_cache.ndim != 3:
        raise ValueError("key_cache/value_cache must have shape [pages, heads, dim].")
    if key_cache.shape != value_cache.shape:
        raise ValueError("key_cache and value_cache shapes must match.")
    if query.shape[0] != block_table.shape[0]:
        raise ValueError("query and block_table batch sizes must match.")
    if query.shape[-1] != key_cache.shape[-1]:
        raise ValueError("query and cache head dimensions must match.")
    if query.shape[1] % key_cache.shape[1] != 0:
        raise ValueError("query heads must be divisible by KV heads.")

    physical_indices, valid_keys = paged_cache_indices(
        block_table,
        sequence_lengths,
        block_size,
    )
    if int(physical_indices.max().item()) >= key_cache.shape[0]:
        raise ValueError("block_table references a block outside the physical cache.")

    key = key_cache[physical_indices].permute(0, 2, 1, 3)
    value = value_cache[physical_indices].permute(0, 2, 1, 3)
    groups = query.shape[1] // key.shape[1]
    if groups != 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)

    actual_scale = query.shape[-1] ** -0.5 if scale is None else scale
    scores = torch.matmul(query, key.transpose(-2, -1)) * actual_scale
    valid = valid_keys[:, None, None, :]

    if is_causal:
        query_length = query.shape[-2]
        key_positions = torch.arange(key.shape[-2], device=query.device)[None, :]
        query_positions = torch.arange(query_length, device=query.device)[:, None]
        causal_limit = sequence_lengths[:, None, None] - query_length + query_positions[None, :, :]
        valid = valid & (key_positions[None, None, :, :] <= causal_limit[:, None, :, :])

    scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    probabilities = resolve_softmax_fn(normalizer)(scores, dim=-1)
    return torch.matmul(probabilities, value)


def paged_triton_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    sequence_lengths: torch.Tensor,
    block_size: int,
    normalizer: str,
    scale: Optional[float] = None,
    is_causal: bool = True,
    max_block_n: int = 4096,
) -> torch.Tensor:
    """Paged sparse attention using the fused Triton forward/backward kernels.

    This first implementation launches one fused kernel per variable-length
    sequence. K/V pages are gathered by physical index, but the quadratic
    attention matrix is never materialized. Gather backward accumulates K/V
    gradients into their original physical cache pages.
    """
    if normalizer not in {"softmax1", "softmax_1", "sparsemax", "entmax15"}:
        raise ValueError(
            "paged_triton_attention supports 'softmax1', 'sparsemax', or 'entmax15'."
        )
    if not query.is_cuda or not key_cache.is_cuda or not value_cache.is_cuda:
        raise ValueError("paged_triton_attention requires CUDA tensors.")
    if query.ndim != 4:
        raise ValueError("query must have shape [batch, heads, sequence, dim].")
    if key_cache.ndim != 3 or key_cache.shape != value_cache.shape:
        raise ValueError("key_cache/value_cache must have matching [pages, heads, dim] shapes.")
    if query.shape[0] != block_table.shape[0]:
        raise ValueError("query and block_table batch sizes must match.")
    if query.shape[-1] != key_cache.shape[-1]:
        raise ValueError("query and cache head dimensions must match.")
    if query.shape[1] % key_cache.shape[1] != 0:
        raise ValueError("query heads must be divisible by KV heads.")

    physical_indices, _ = paged_cache_indices(block_table, sequence_lengths, block_size)
    if int(physical_indices.max().item()) >= key_cache.shape[0]:
        raise ValueError("block_table references a block outside the physical cache.")
    common_kwargs = {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_table": block_table,
        "sequence_lengths": sequence_lengths,
        "page_size": block_size,
        "scale": scale,
        "is_causal": is_causal,
        # H100 measurements for the direct physical-page kernels show that a
        # 256-token K tile reduces loop/control overhead for both normalizers
        # at practical decode/prefill lengths. Callers can still cap it for
        # very small heads or constrained kernels through ``max_block_n``.
        "block_n": min(256, max_block_n),
    }
    if normalizer in {"softmax1", "softmax_1"}:
        return paged_hopper_softmax1_attention(**common_kwargs)
    return paged_hopper_sparse_attention(**common_kwargs, normalizer=normalizer)
