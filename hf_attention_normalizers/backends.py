import math
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.integrations.sdpa_attention import use_gqa_in_sdpa
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3RMSNorm
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.deprecation import deprecate_kwarg

from .vutils import entmax15, softmax_1, sparsemax


SoftmaxLike = Callable[[torch.Tensor, int], torch.Tensor]


SOFTMAX_FUNCTIONS: dict[str, Callable[..., torch.Tensor]] = {
    "vanilla": nn.functional.softmax,
    "softmax": nn.functional.softmax,
    "sparsemax": sparsemax,
    "entmax15": entmax15,
    "softmax1": softmax_1,
    "softmax_1": softmax_1,
}


def resolve_softmax_fn(softmax_fn: Union[str, Callable[..., torch.Tensor]]) -> Callable[..., torch.Tensor]:
    if callable(softmax_fn):
        return softmax_fn

    try:
        return SOFTMAX_FUNCTIONS[softmax_fn]
    except KeyError as exc:
        valid = ", ".join(sorted(SOFTMAX_FUNCTIONS))
        raise ValueError(f"Invalid softmax function: {softmax_fn}. Valid options: {valid}") from exc


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_fn: Callable[..., torch.Tensor],
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    training: bool = True,
) -> torch.Tensor:
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    target_length, source_length = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_weight = query @ key.transpose(-2, -1) * scale_factor

    if is_causal:
        if attn_mask is not None:
            raise ValueError("is_causal=True and attn_mask cannot be used together.")
        causal_mask = torch.ones(target_length, source_length, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_weight = attn_weight.masked_fill(causal_mask.logical_not(), torch.finfo(attn_weight.dtype).min)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weight = attn_weight.masked_fill(attn_mask.logical_not(), torch.finfo(attn_weight.dtype).min)
        else:
            attn_weight = attn_weight + attn_mask

    attn_weight = softmax_fn(attn_weight, dim=-1)
    attn_weight = nn.functional.dropout(attn_weight, p=dropout_p, training=training)
    return attn_weight @ value


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    softmax_fn: Callable[..., torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    if is_causal is None:
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_output = scaled_dot_product_attention(
        query,
        key,
        value,
        softmax_fn=softmax_fn,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        training=module.training,
        **sdpa_kwargs,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def flash_attention_softmax_n_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    softmax_n_param: Union[int, float] = 1,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    try:
        from flash_attention_softmax_n import flash_attention_n
    except ImportError as exc:
        raise ImportError(
            "softmax1_flash_attention_2 requires flash-attention-softmax-n. "
            "Install it with: pip install flash-attention-softmax-n"
        ) from exc

    if key.size(-3) != query.size(-3):
        if query.size(-3) % key.size(-3) != 0:
            raise ValueError(
                "FlashAttention requires the number of query heads to be divisible by "
                f"the number of key/value heads. Got {query.size(-3)} query heads and "
                f"{key.size(-3)} key/value heads."
            )
        # flash-attention-softmax-n does not implement PyTorch SDPA's
        # enable_gqa behavior, so expand grouped K/V heads explicitly.
        num_key_value_groups = query.size(-3) // key.size(-3)
        key = repeat_kv(key, num_key_value_groups)
        value = repeat_kv(value, num_key_value_groups)

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    if is_causal is None:
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_output = flash_attention_n(
        query=query,
        key=key,
        value=value,
        softmax_n_param=softmax_n_param,
        scale=scaling,
        dropout_p=dropout,
        attn_mask=attention_mask,
        attn_bias=None,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def _dense_mask_from_flex_block_mask(
    block_mask,
    query_length: int,
    key_length: int,
    batch_size: int,
    num_heads: int,
) -> torch.Tensor:
    block_dense = block_mask.to_dense().to(torch.bool)
    query_blocks, key_blocks = block_dense.shape[-2:]
    query_block_size = (query_length + query_blocks - 1) // query_blocks
    key_block_size = (key_length + key_blocks - 1) // key_blocks
    dense = block_dense.repeat_interleave(query_block_size, dim=-2).repeat_interleave(
        key_block_size,
        dim=-1,
    )[..., :query_length, :key_length]
    dense = dense.expand(batch_size, num_heads, query_length, key_length)

    batch = torch.arange(batch_size, device=dense.device)[:, None, None, None]
    head = torch.arange(num_heads, device=dense.device)[None, :, None, None]
    query_index = torch.arange(query_length, device=dense.device)[None, None, :, None]
    key_index = torch.arange(key_length, device=dense.device)[None, None, None, :]
    element_mask = block_mask.mask_mod(batch, head, query_index, key_index)
    return dense & element_mask


def _is_pure_causal_flex_mask(block_mask) -> bool:
    """Conservatively recognize HF's offset-wrapped causal mask function."""
    try:
        from transformers.masking_utils import causal_mask_function
    except ImportError:
        return False

    def unwrap(function, seen):
        if function is causal_mask_function:
            return True
        if id(function) in seen:
            return False
        seen.add(id(function))
        closure = getattr(function, "__closure__", None)
        if not closure:
            return False
        nested_functions = [
            cell.cell_contents
            for cell in closure
            if callable(cell.cell_contents)
        ]
        # Offset wrappers contain exactly one nested mask function. Combined
        # padding/custom masks contain multiple functions and must fall back.
        return len(nested_functions) == 1 and unwrap(nested_functions[0], seen)

    return unwrap(block_mask.mask_mod, set())


def _sliding_window_from_flex_mask(block_mask) -> Optional[int]:
    """Recognize HF's causal sliding-window mask and return its left width."""
    found_causal = False
    window = None

    def walk(value):
        nonlocal found_causal, window
        if isinstance(value, tuple):
            return all(walk(item) for item in value)
        if not callable(value):
            return True
        qualified_name = getattr(value, "__qualname__", "")
        if qualified_name == "causal_mask_function":
            found_causal = True
            return True
        if "sliding_window_overlay.<locals>.inner_mask" in qualified_name:
            integers = [
                cell.cell_contents
                for cell in (value.__closure__ or ())
                if isinstance(cell.cell_contents, int)
            ]
            if len(integers) != 1:
                return False
            window = integers[0]
            return True
        # Padding is handled separately by _right_padded_lengths_from_flex_mask.
        # Accept it here so a causal sliding-window mask combined with padding can
        # still use the native windowed kernel.
        if "padding_mask_function.<locals>.inner_mask" in qualified_name:
            return True
        if (
            "add_offsets_to_mask_function.<locals>.inner_mask" not in qualified_name
            and "and_masks.<locals>.and_mask" not in qualified_name
        ):
            return False
        return all(walk(cell.cell_contents) for cell in (value.__closure__ or ()))

    return window if walk(block_mask.mask_mod) and found_causal and window is not None else None


def _right_padded_lengths_from_flex_mask(block_mask) -> Optional[torch.Tensor]:
    """Return per-example K lengths for a causal HF padding BlockMask.

    This deliberately recognizes only the standard Transformers composition of
    causal/sliding-window masks and a 2-D *right-padded* attention mask.  Other
    custom masks continue through the exact dense compatibility path below.
    """
    padding_mask = None
    found_causal = False

    def walk(value) -> bool:
        nonlocal padding_mask, found_causal
        if isinstance(value, tuple):
            return all(walk(item) for item in value)
        if not callable(value):
            return True
        qualified_name = getattr(value, "__qualname__", "")
        if qualified_name == "causal_mask_function":
            found_causal = True
            return True
        if "padding_mask_function.<locals>.inner_mask" in qualified_name:
            tensors = [
                cell.cell_contents
                for cell in (value.__closure__ or ())
                if isinstance(cell.cell_contents, torch.Tensor)
            ]
            if len(tensors) != 1 or padding_mask is not None:
                return False
            padding_mask = tensors[0]
            return padding_mask.ndim == 2
        if "sliding_window_overlay.<locals>.inner_mask" in qualified_name:
            return True
        if (
            "add_offsets_to_mask_function.<locals>.inner_mask" not in qualified_name
            and "and_masks.<locals>.and_mask" not in qualified_name
        ):
            return False
        return all(walk(cell.cell_contents) for cell in (value.__closure__ or ()))

    if not walk(block_mask.mask_mod) or not found_causal or padding_mask is None:
        return None
    padding_mask = padding_mask.to(dtype=torch.bool)
    lengths = padding_mask.sum(dim=-1, dtype=torch.long)
    positions = torch.arange(padding_mask.shape[-1], device=padding_mask.device)
    if not torch.equal(padding_mask, positions[None, :] < lengths[:, None]):
        return None
    # An all-masked row has no well-defined sparse/entmax distribution. Preserve
    # the dense fallback's behavior in that unusual case.
    if bool((lengths == 0).any()):
        return None
    return lengths


def _block_csr_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask,
    softmax_fn: Callable[..., torch.Tensor],
    scale: Optional[float],
) -> Optional[torch.Tensor]:
    """Exact block-CSR Flex path for masks without element-level filtering.

    This avoids ``BlockMask.to_dense()`` entirely. It intentionally targets
    masks created from explicit full KV blocks (`mask_mod=noop_mask`); a custom
    element mask can make a listed block only partially valid and must use the
    dense compatibility fallback until the Triton CSR kernel is in place.
    """
    try:
        from torch.nn.attention.flex_attention import noop_mask
    except ImportError:
        return None
    if block_mask.mask_mod is not noop_mask:
        return None
    if key.shape[-3] != query.shape[-3]:
        if query.shape[-3] % key.shape[-3] != 0:
            raise ValueError("Query heads must be divisible by key/value heads.")
        groups = query.shape[-3] // key.shape[-3]
        key = repeat_kv(key, groups)
        value = repeat_kv(value, groups)
    batch_size, heads, query_length, head_dim = query.shape
    q_block_size, kv_block_size = block_mask.BLOCK_SIZE
    num_blocks = block_mask.kv_num_blocks
    block_indices = block_mask.kv_indices
    if num_blocks.ndim != 3 or block_indices.ndim != 4:
        return None
    if num_blocks.shape[0] not in {1, batch_size} or num_blocks.shape[1] not in {1, heads}:
        return None
    actual_scale = head_dim**-0.5 if scale is None else scale
    if softmax_fn is softmax_1:
        from .kernels.block_csr_softmax1_triton import block_csr_softmax1_attention

        csr_num_blocks = num_blocks.expand(batch_size, heads, -1)
        csr_block_indices = block_indices.expand(batch_size, heads, -1, -1)
        return block_csr_softmax1_attention(
            query,
            key,
            value,
            csr_num_blocks,
            csr_block_indices,
            q_block_size,
            kv_block_size,
            actual_scale,
        )
    output = torch.empty_like(query)
    for batch_index in range(batch_size):
        mask_batch = 0 if num_blocks.shape[0] == 1 else batch_index
        for head_index in range(heads):
            mask_head = 0 if num_blocks.shape[1] == 1 else head_index
            for q_block in range(num_blocks.shape[2]):
                q_start = q_block * q_block_size
                q_end = min(q_start + q_block_size, query_length)
                if q_start >= q_end:
                    continue
                count = int(num_blocks[mask_batch, mask_head, q_block].item())
                if count == 0:
                    # Match the dense path's all-masked behavior without
                    # constructing a full mask or an attention score matrix.
                    output[batch_index, head_index, q_start:q_end] = 0
                    continue
                selected_blocks = block_indices[mask_batch, mask_head, q_block, :count]
                offsets = torch.arange(kv_block_size, device=query.device)
                key_positions = (selected_blocks[:, None] * kv_block_size + offsets).reshape(-1)
                key_positions = key_positions[key_positions < key.shape[-2]]
                q_tile = query[batch_index, head_index, q_start:q_end]
                k_tile = key[batch_index, head_index, key_positions]
                v_tile = value[batch_index, head_index, key_positions]
                scores = torch.matmul(q_tile, k_tile.transpose(-2, -1)) * actual_scale
                probabilities = softmax_fn(scores, dim=-1)
                output[batch_index, head_index, q_start:q_end] = torch.matmul(probabilities, v_tile)
    return output


def _hopper_right_padded_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_lengths: torch.Tensor,
    softmax_fn: Callable[..., torch.Tensor],
    scale: Optional[float],
    sliding_window: Optional[int],
) -> torch.Tensor:
    """Run native Hopper attention without materializing a dense padding mask.

    The current Hopper kernels take a static K length.  We therefore launch one
    fused kernel per batch row with its valid prefix, retaining autograd and
    avoiding both dense mask materialization and padded K/V reads.
    """
    from .kernels.softmax1_hopper import hopper_softmax1_attention
    from .kernels.sparse_entmax_hopper import (
        hopper_entmax15_attention,
        hopper_sparsemax_attention,
    )

    if key.shape[-3] != query.shape[-3]:
        if query.shape[-3] % key.shape[-3] != 0:
            raise ValueError("Query heads must be divisible by key/value heads.")
        groups = query.shape[-3] // key.shape[-3]
        key = repeat_kv(key, groups)
        value = repeat_kv(value, groups)
    kernels = {
        softmax_1: hopper_softmax1_attention,
        sparsemax: hopper_sparsemax_attention,
        entmax15: hopper_entmax15_attention,
    }
    kernel = kernels[softmax_fn]
    outputs = []
    for batch_index, length in enumerate(key_lengths.tolist()):
        batch_query = query[batch_index : batch_index + 1]
        batch_key = key[batch_index : batch_index + 1, :, :length]
        batch_value = value[batch_index : batch_index + 1, :, :length]
        # The Hopper causal convention is bottom-right aligned.  For a padded
        # prefill row, Q is sized to the batch maximum while its valid K prefix
        # is shorter.  Split at the prefix: the first part is ordinary causal
        # attention, while later (padded) queries may see the whole valid prefix.
        valid_queries = min(length, batch_query.shape[-2])
        parts = [
            kernel(
                batch_query[:, :, :valid_queries],
                batch_key,
                batch_value,
                scale=scale,
                is_causal=True,
            )
        ]
        if valid_queries < batch_query.shape[-2]:
            parts.append(
                kernel(
                    batch_query[:, :, valid_queries:],
                    batch_key,
                    batch_value,
                    scale=scale,
                    is_causal=False,
                )
            )
        outputs.append(torch.cat(parts, dim=-2))
    return torch.cat(outputs, dim=0)


def flex_attention_normalizer_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    softmax_fn: Callable[..., torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    try:
        from torch.nn.attention.flex_attention import BlockMask
    except ImportError:
        BlockMask = ()  # type: ignore[assignment]

    mask_sliding_window = (
        _sliding_window_from_flex_mask(attention_mask)
        if isinstance(attention_mask, BlockMask)
        else None
    )
    padding_lengths = (
        _right_padded_lengths_from_flex_mask(attention_mask)
        if isinstance(attention_mask, BlockMask)
        else None
    )
    module_sliding_window = kwargs.pop("sliding_window", None)
    sliding_window = (
        mask_sliding_window
        if mask_sliding_window is not None
        else module_sliding_window
    )
    supports_native_padding = padding_lengths is not None and sliding_window is None
    if isinstance(attention_mask, BlockMask) and (
        _is_pure_causal_flex_mask(attention_mask)
        or (mask_sliding_window is not None and padding_lengths is None)
        or supports_native_padding
    ):
        if supports_native_padding and softmax_fn in (softmax_1, sparsemax, entmax15):
            output = _hopper_right_padded_attention(
                query,
                key,
                value,
                padding_lengths,
                softmax_fn,
                scaling,
                sliding_window,
            )
            return output.transpose(1, 2).contiguous(), None
        common_kwargs = {
            "module": module,
            "query": query,
            "key": key,
            "value": value,
            "attention_mask": None,
            "dropout": dropout,
            "scaling": scaling,
            "is_causal": True,
            "sliding_window": sliding_window,
        }
        if softmax_fn is softmax_1:
            return hopper_softmax1_attention_forward(**common_kwargs, **kwargs)
        if softmax_fn is sparsemax:
            return hopper_sparsemax_attention_forward(**common_kwargs, **kwargs)
        if softmax_fn is entmax15:
            return hopper_entmax15_attention_forward(**common_kwargs, **kwargs)

    if isinstance(attention_mask, BlockMask):
        output = _block_csr_flex_attention(
            query, key, value, attention_mask, softmax_fn, scaling
        )
        if output is not None:
            return output.transpose(1, 2).contiguous(), None

    if isinstance(attention_mask, BlockMask):
        attention_mask = _dense_mask_from_flex_block_mask(
            attention_mask,
            query.shape[-2],
            key.shape[-2],
            query.shape[0],
            query.shape[1],
        )
    return sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        softmax_fn=softmax_fn,
        dropout=dropout,
        scaling=scaling,
        is_causal=False if attention_mask is not None else kwargs.pop("is_causal", None),
        **kwargs,
    )


def triton_sparsemax_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    max_block_n: int = 4096,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if attention_mask is not None or dropout > 0 or not query.is_cuda:
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            softmax_fn=sparsemax,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )

    try:
        from .kernels.sparse_entmax_triton import triton_sparsemax_attention
    except ImportError as exc:
        raise ImportError("sparsemax_triton requires Triton. Install it with: pip install triton") from exc

    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    if is_causal is None:
        is_causal = query.shape[2] > 1 and getattr(module, "is_causal", True)

    attn_output = triton_sparsemax_attention(
        query,
        key,
        value,
        scale=scaling,
        is_causal=is_causal,
        max_block_n=max_block_n,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def triton_entmax15_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    max_block_n: int = 4096,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if attention_mask is not None or dropout > 0 or not query.is_cuda:
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            softmax_fn=entmax15,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )

    try:
        from .kernels.sparse_entmax_triton import triton_entmax15_attention
    except ImportError as exc:
        raise ImportError("entmax15_triton requires Triton. Install it with: pip install triton") from exc

    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    if is_causal is None:
        is_causal = query.shape[2] > 1 and getattr(module, "is_causal", True)

    attn_output = triton_entmax15_attention(
        query,
        key,
        value,
        scale=scaling,
        is_causal=is_causal,
        max_block_n=max_block_n,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def _hopper_sparse_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    normalizer: str,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    block_n: int = 128,
    bisection_steps: int = 24,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    kwargs.pop("softmax_fn", None)
    kwargs.pop("softmax_n_param", None)
    sliding_window = kwargs.pop("sliding_window", None)
    normalizer_fn = sparsemax if normalizer == "sparsemax" else entmax15
    if attention_mask is not None or not query.is_cuda:
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            softmax_fn=normalizer_fn,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )

    from .kernels.sparse_entmax_hopper import (
        hopper_entmax15_attention,
        hopper_sparsemax_attention,
    )

    if key.shape[-3] != query.shape[-3]:
        if query.shape[-3] % key.shape[-3] != 0:
            raise ValueError("Query heads must be divisible by key/value heads.")
        groups = query.shape[-3] // key.shape[-3]
        key = repeat_kv(key, groups)
        value = repeat_kv(value, groups)
    if is_causal is None:
        is_causal = query.shape[-2] > 1 and getattr(module, "is_causal", True)
    kernel = hopper_sparsemax_attention if normalizer == "sparsemax" else hopper_entmax15_attention
    output = kernel(
        query,
        key,
        value,
        scale=scaling,
        is_causal=is_causal,
        block_n=block_n,
        bisection_steps=bisection_steps,
        window_left=sliding_window,
        dropout_p=dropout if module.training else 0.0,
    )
    return output.transpose(1, 2).contiguous(), None


def hopper_sparsemax_attention_forward(*args, **kwargs) -> tuple[torch.Tensor, None]:
    return _hopper_sparse_attention_forward(*args, normalizer="sparsemax", **kwargs)


def hopper_entmax15_attention_forward(*args, **kwargs) -> tuple[torch.Tensor, None]:
    return _hopper_sparse_attention_forward(*args, normalizer="entmax15", **kwargs)


def hopper_softmax1_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    block_n: int = 128,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    kwargs.pop("softmax_fn", None)
    kwargs.pop("softmax_n_param", None)
    sliding_window = kwargs.pop("sliding_window", None)
    if attention_mask is not None or not query.is_cuda:
        return sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            softmax_fn=softmax_1,
            dropout=dropout,
            scaling=scaling,
            is_causal=is_causal,
            **kwargs,
        )

    from .kernels.softmax1_hopper import hopper_softmax1_attention

    if key.shape[-3] != query.shape[-3]:
        if query.shape[-3] % key.shape[-3] != 0:
            raise ValueError("Query heads must be divisible by key/value heads.")
        groups = query.shape[-3] // key.shape[-3]
        key = repeat_kv(key, groups)
        value = repeat_kv(value, groups)
    if is_causal is None:
        is_causal = query.shape[-2] > 1 and getattr(module, "is_causal", True)
    output = hopper_softmax1_attention(
        query,
        key,
        value,
        scale=scaling,
        is_causal=is_causal,
        block_n=block_n,
        window_left=sliding_window,
        dropout_p=dropout if module.training else 0.0,
    )
    return output.transpose(1, 2).contiguous(), None


def paged_normalizer_attention_forward(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    cache=None,
    cu_seq_lens_q: Optional[torch.Tensor] = None,
    cu_seq_lens_k=None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k=None,
    implementation=None,
    softmax_fn: Callable[..., torch.Tensor] = softmax_1,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """HuggingFace ``paged_attention`` interface for custom normalizers.

    Transformers continuous batching supplies packed Q plus cumulative lengths.
    Its cache owns physical-page allocation and returns the logical packed K/V
    view after update; the custom varlen kernel then performs one grid over the
    entire continuous batch.  This is intentionally separate from
    :func:`paged_triton_attention`, whose public API operates directly on a
    physical block table and supports cache-page gradients.
    """
    if attention_mask is not None:
        raise NotImplementedError(
            "Custom paged_attention supports continuous-batching cumulative-length masks; "
            "arbitrary attention_mask is not supported."
        )
    if cache is not None:
        k, v = cache.update(k, v, module.layer_idx, **kwargs)
    if isinstance(cu_seq_lens_k, dict):
        layer_type = "sliding_attention" if getattr(module, "sliding_window", None) else "full_attention"
        cu_seq_lens_k = cu_seq_lens_k[layer_type]
        max_seqlen_k = max_seqlen_k[layer_type]
    if cu_seq_lens_q is None or cu_seq_lens_k is None or max_seqlen_q is None or max_seqlen_k is None:
        raise ValueError("paged_attention requires cu_seq_lens_q/k and max_seqlen_q/k.")
    if q.ndim != 4:
        raise ValueError("paged_attention query must have shape [1, heads, total_q, head_dim].")
    packed_query = q.transpose(1, 2).squeeze(0).contiguous()
    if k.ndim == 4:
        packed_key = k.transpose(1, 2).squeeze(0).contiguous()
        packed_value = v.transpose(1, 2).squeeze(0).contiguous()
    else:
        packed_key, packed_value = k.contiguous(), v.contiguous()
    if softmax_fn is softmax_1:
        normalizer = "softmax1"
    elif softmax_fn is sparsemax:
        normalizer = "sparsemax"
    elif softmax_fn is entmax15:
        normalizer = "entmax15"
    else:
        raise ValueError("paged_attention supports softmax1, sparsemax, or entmax15.")
    from .varlen import varlen_hopper_attention

    output = varlen_hopper_attention(
        packed_query,
        packed_key,
        packed_value,
        cu_seq_lens_q.to(torch.int32),
        cu_seq_lens_k.to(torch.int32),
        normalizer=normalizer,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        scale=getattr(module, "scaling", None),
        is_causal=True,
    )
    return output, None


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    softmax_fn: Callable[..., torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor]:
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = softmax_fn(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


CUSTOM_ATTENTION_FUNCTIONS: dict[str, Callable[..., tuple[torch.Tensor, Optional[torch.Tensor]]]] = {
    "eager": eager_attention_forward,
    "sdpa": sdpa_attention_forward,
    "flash_attention_2": flash_attention_softmax_n_forward,
    "sparsemax_triton": triton_sparsemax_attention_forward,
    "entmax15_triton": triton_entmax15_attention_forward,
    "softmaxn_eager": eager_attention_forward,
    "softmaxn_sdpa": sdpa_attention_forward,
    "softmaxn_flash_attention_2": flash_attention_softmax_n_forward,
    "flex_attention": flex_attention_normalizer_forward,
    "paged_attention": paged_normalizer_attention_forward,
    "sparsemax_flash_attention_3": hopper_sparsemax_attention_forward,
    "entmax15_flash_attention_3": hopper_entmax15_attention_forward,
    "softmax1_flash_attention_3": hopper_softmax1_attention_forward,
}


HF_ATTENTION_BACKENDS = (
    "eager",
    "sdpa",
    "flash_attention_2",
    "flash_attention_3",
    "flex_attention",
    "paged_attention",
    "paged|eager",
    "paged|sdpa",
    "paged|flash_attention_2",
    "paged|flash_attention_3",
)


def _make_softmax_attention_forward(
    base_backend: str,
    softmax_fn: Union[str, Callable[..., torch.Tensor]] = "softmax1",
    fallback_backend: str = "sdpa",
    mode: str = "strict",
) -> Callable[..., tuple[torch.Tensor, Optional[torch.Tensor]]]:
    resolved_softmax_fn = resolve_softmax_fn(softmax_fn)
    softmax_name = softmax_fn if isinstance(softmax_fn, str) else getattr(softmax_fn, "__name__", "callable")

    is_softmax_n = softmax_name in {"softmax1", "softmax_1", "softmax_n"}

    if base_backend == "eager":
        attention_forward = eager_attention_forward
    elif base_backend == "sdpa":
        attention_forward = sdpa_attention_forward
    elif base_backend == "flash_attention_2" and is_softmax_n:
        attention_forward = flash_attention_softmax_n_forward
    elif base_backend == "flex_attention":
        attention_forward = flex_attention_normalizer_forward
    elif base_backend == "paged_attention" and softmax_name in {"softmax1", "softmax_1", "sparsemax", "entmax15"}:
        attention_forward = paged_normalizer_attention_forward
    elif base_backend == "flash_attention_3" and softmax_name == "sparsemax":
        attention_forward = hopper_sparsemax_attention_forward
    elif base_backend == "flash_attention_3" and softmax_name == "entmax15":
        attention_forward = hopper_entmax15_attention_forward
    elif base_backend == "flash_attention_3" and is_softmax_n:
        attention_forward = hopper_softmax1_attention_forward
    elif mode == "fallback":
        attention_forward = CUSTOM_ATTENTION_FUNCTIONS[fallback_backend]
    elif mode == "strict":

        def unsupported_attention_forward(*args, **kwargs):
            raise NotImplementedError(
                f"{softmax_name}_{base_backend} is registered for HuggingFace compatibility, "
                "but this backend does not have a custom-softmax kernel yet. "
                f"Use mode='fallback' or choose {softmax_name}_eager/{softmax_name}_sdpa."
            )

        return unsupported_attention_forward
    else:
        raise ValueError("mode must be 'strict' or 'fallback'")

    def softmax_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        kwargs.pop("softmax_fn", None)
        return attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            softmax_fn=resolved_softmax_fn,
            softmax_n_param=1,
            **kwargs,
        )

    return softmax_attention_forward


def register_softmax_attention_backends(
    softmax_fn: Union[str, Callable[..., torch.Tensor]] = "softmax1",
    prefix: Optional[str] = None,
    mode: str = "strict",
    fallback_backend: str = "sdpa",
    include_unsupported: bool = True,
) -> tuple[str, ...]:
    """Register HuggingFace AttentionInterface backends for a custom softmax.

    In strict mode, eager/sdpa aliases run normally and unsupported fused
    backends raise if selected. In fallback mode, unsupported fused backends
    run through fallback_backend to preserve the custom softmax math.
    """
    if mode not in {"strict", "fallback"}:
        raise ValueError("mode must be 'strict' or 'fallback'")
    if fallback_backend not in CUSTOM_ATTENTION_FUNCTIONS:
        valid = ", ".join(sorted(CUSTOM_ATTENTION_FUNCTIONS))
        raise ValueError(f"fallback_backend must be one of: {valid}")

    softmax_name = prefix or (softmax_fn if isinstance(softmax_fn, str) else getattr(softmax_fn, "__name__", "custom"))
    registered = []

    for backend in HF_ATTENTION_BACKENDS:
        softmax_name_for_support = (
            softmax_fn if isinstance(softmax_fn, str) else getattr(softmax_fn, "__name__", "custom")
        )
        supports_softmax_n_kernel = softmax_name_for_support in {"softmax1", "softmax_1", "softmax_n"}
        supports_hopper_kernel = (
            backend == "flash_attention_3"
            and softmax_name_for_support in {"softmax1", "softmax_1", "softmax_n", "sparsemax", "entmax15"}
        )
        supported = backend in {"eager", "sdpa", "flex_attention", "paged_attention"} or supports_hopper_kernel or (
            backend == "flash_attention_2" and supports_softmax_n_kernel
        )
        if not supported and not include_unsupported:
            continue

        name = f"{softmax_name}_{backend}"
        attention_forward = _make_softmax_attention_forward(
            backend,
            softmax_fn=softmax_fn,
            fallback_backend=fallback_backend,
            mode=mode,
        )
        if hasattr(ALL_ATTENTION_FUNCTIONS, "register"):
            ALL_ATTENTION_FUNCTIONS.register(name, attention_forward)
        else:
            ALL_ATTENTION_FUNCTIONS[name] = attention_forward
        registered.append(name)

        try:
            from transformers import AttentionMaskInterface
            from transformers.masking_utils import flex_attention_mask, sdpa_mask

            mask_function = flex_attention_mask if backend == "flex_attention" else sdpa_mask
            AttentionMaskInterface.register(name, mask_function)
        except (ImportError, AttributeError):
            pass

    return tuple(registered)


def _register_attention_backend(name: str, attention_forward: Callable[..., tuple[torch.Tensor, Optional[torch.Tensor]]]) -> None:
    if hasattr(ALL_ATTENTION_FUNCTIONS, "register"):
        ALL_ATTENTION_FUNCTIONS.register(name, attention_forward)
    else:
        ALL_ATTENTION_FUNCTIONS[name] = attention_forward

    try:
        from transformers import AttentionMaskInterface
        from transformers.masking_utils import flex_attention_mask, sdpa_mask

        mask_function = flex_attention_mask if "flex_attention" in name else sdpa_mask
        AttentionMaskInterface.register(name, mask_function)
    except (ImportError, AttributeError):
        pass


def register_softmax1_attention_backends(
    mode: str = "strict",
    fallback_backend: str = "sdpa",
    include_unsupported: bool = True,
) -> tuple[str, ...]:
    return register_softmax_attention_backends(
        softmax_fn="softmax1",
        mode=mode,
        fallback_backend=fallback_backend,
        include_unsupported=include_unsupported,
    )


def register_sparsemax_attention_backends(
    mode: str = "strict",
    fallback_backend: str = "sdpa",
    include_unsupported: bool = True,
    include_triton: bool = True,
) -> tuple[str, ...]:
    registered = list(register_softmax_attention_backends(
        softmax_fn="sparsemax",
        mode=mode,
        fallback_backend=fallback_backend,
        include_unsupported=include_unsupported,
    ))
    if include_triton:
        _register_attention_backend("sparsemax_triton", triton_sparsemax_attention_forward)
        registered.append("sparsemax_triton")
    return tuple(registered)


def register_entmax15_attention_backends(
    mode: str = "strict",
    fallback_backend: str = "sdpa",
    include_unsupported: bool = True,
    include_triton: bool = True,
) -> tuple[str, ...]:
    registered = list(register_softmax_attention_backends(
        softmax_fn="entmax15",
        mode=mode,
        fallback_backend=fallback_backend,
        include_unsupported=include_unsupported,
    ))
    if include_triton:
        _register_attention_backend("entmax15_triton", triton_entmax15_attention_forward)
        registered.append("entmax15_triton")
    return tuple(registered)


def softmax_attention_backend_name(
    base_backend: str = "sdpa",
    softmax_fn: Union[str, Callable[..., torch.Tensor]] = "softmax1",
    prefix: Optional[str] = None,
) -> str:
    softmax_name = prefix or (softmax_fn if isinstance(softmax_fn, str) else getattr(softmax_fn, "__name__", "custom"))
    if base_backend == "triton" and softmax_name in {"sparsemax", "entmax15"}:
        return f"{softmax_name}_triton"
    return f"{softmax_name}_{base_backend}"


def set_softmax_attention_backend(
    model: nn.Module,
    base_backend: str = "sdpa",
    softmax_fn: Union[str, Callable[..., torch.Tensor]] = "softmax1",
    mode: str = "strict",
    fallback_backend: str = "sdpa",
) -> nn.Module:
    if softmax_fn == "sparsemax":
        register_sparsemax_attention_backends(mode=mode, fallback_backend=fallback_backend)
    elif softmax_fn == "entmax15":
        register_entmax15_attention_backends(mode=mode, fallback_backend=fallback_backend)
    elif softmax_fn in {"softmax1", "softmax_1"}:
        register_softmax1_attention_backends(mode=mode, fallback_backend=fallback_backend)
    else:
        register_softmax_attention_backends(
            softmax_fn=softmax_fn,
            mode=mode,
            fallback_backend=fallback_backend,
        )
    attn_implementation = softmax_attention_backend_name(base_backend, softmax_fn)

    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(attn_implementation)
    elif hasattr(model, "config"):
        model.config._attn_implementation = attn_implementation
    else:
        raise ValueError("model has neither set_attn_implementation nor config._attn_implementation")

    return model


@dataclass(frozen=True)
class AttentionReplacementPolicy:
    name: str
    matches: Callable[[nn.Module], bool]
    build: Callable[[nn.Module, Union[str, Callable[..., torch.Tensor]], str], nn.Module]


def get_attention_forward(attn_implementation: Optional[str]) -> Callable[..., tuple[torch.Tensor, Optional[torch.Tensor]]]:
    attn_implementation = attn_implementation or "eager"
    if attn_implementation in CUSTOM_ATTENTION_FUNCTIONS:
        return CUSTOM_ATTENTION_FUNCTIONS[attn_implementation]

    if hasattr(ALL_ATTENTION_FUNCTIONS, "get_interface"):
        return ALL_ATTENTION_FUNCTIONS.get_interface(attn_implementation, eager_attention_forward)

    return ALL_ATTENTION_FUNCTIONS.get(attn_implementation, eager_attention_forward)


def _uses_custom_softmax(softmax_name: str) -> bool:
    return softmax_name not in {"vanilla", "softmax"}


def _is_allowed_custom_softmax_backend(softmax_name: str, attn_implementation: str) -> bool:
    return (
        attn_implementation in CUSTOM_ATTENTION_FUNCTIONS
        or attn_implementation.startswith(f"{softmax_name}_")
    )


class Qwen3AttentionExtrea(nn.Module):
    """Qwen3 attention with pluggable attention backend and pluggable softmax."""

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        softmax_fn: Union[str, Callable[..., torch.Tensor]] = "softmax1",
        attn_implementation: Optional[str] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.softmax_fn = resolve_softmax_fn(softmax_fn)
        self.softmax_name = softmax_fn if isinstance(softmax_fn, str) else getattr(softmax_fn, "__name__", "callable")
        self.attn_implementation = attn_implementation

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        dtype, device = hidden_states.dtype, hidden_states.device
        self.to(device=device, dtype=dtype)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_implementation = (
            kwargs.pop("attn_implementation", None)
            or self.attn_implementation
            or getattr(self.config, "_attn_implementation", "eager")
        )
        if _uses_custom_softmax(self.softmax_name) and not _is_allowed_custom_softmax_backend(
            self.softmax_name,
            attn_implementation,
        ):
            valid = ", ".join(sorted(CUSTOM_ATTENTION_FUNCTIONS))
            raise ValueError(
                f"{self.softmax_name} requires a custom-softmax attention backend. "
                f"Use one of: {valid}. Got: {attn_implementation}"
            )
        attention_interface = get_attention_forward(attn_implementation)

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            softmax_fn=self.softmax_fn,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


Qwen3AttentionExtra = Qwen3AttentionExtrea


def _matches_qwen3_attention(module: nn.Module) -> bool:
    child_class = module.__class__.__name__
    is_qwen3_attn = child_class in {"Qwen3Attention", "Qwen3AttentionExtrea", "Qwen3AttentionExtra"}
    has_qwen3_attn_shape = all(hasattr(module, attr) for attr in ("q_proj", "k_proj", "v_proj", "o_proj"))
    has_required_config = hasattr(module, "config") and hasattr(module, "layer_idx")
    return is_qwen3_attn and has_qwen3_attn_shape and has_required_config


def _build_qwen3_attention_replacement(
    module: nn.Module,
    softmax_fn: Union[str, Callable[..., torch.Tensor]] = "softmax1",
    attn_implementation: str = "sdpa",
) -> nn.Module:
    replacement = Qwen3AttentionExtrea(
        module.config,
        module.layer_idx,
        softmax_fn=softmax_fn,
        attn_implementation=attn_implementation,
    )
    replacement.load_state_dict(module.state_dict(), strict=False)
    first_param = next(module.parameters())
    replacement.to(device=first_param.device, dtype=first_param.dtype)
    replacement.train(module.training)
    return replacement


ATTENTION_REPLACEMENT_POLICIES: dict[str, AttentionReplacementPolicy] = {
    "qwen3": AttentionReplacementPolicy(
        name="qwen3",
        matches=_matches_qwen3_attention,
        build=_build_qwen3_attention_replacement,
    ),
}


def register_attention_replacement_policy(policy: AttentionReplacementPolicy) -> None:
    ATTENTION_REPLACEMENT_POLICIES[policy.name] = policy


def supported_attention_policies() -> tuple[str, ...]:
    return tuple(sorted(ATTENTION_REPLACEMENT_POLICIES))


def replace_attention_modules(
    model: nn.Module,
    softmax_fn: Union[str, Callable[..., torch.Tensor]] = "softmax1",
    attn_implementation: str = "sdpa",
    policy_names: Optional[tuple[str, ...]] = None,
) -> int:
    """Replace supported self-attention modules in a loaded HuggingFace model.

    Returns the number of replaced attention modules.
    """
    policies = ATTENTION_REPLACEMENT_POLICIES
    if policy_names is not None:
        policies = {name: ATTENTION_REPLACEMENT_POLICIES[name] for name in policy_names}

    replaced = 0

    for name, child in list(model.named_children()):
        replacement = None
        for policy in policies.values():
            if policy.matches(child):
                replacement = policy.build(child, softmax_fn, attn_implementation)
                break

        if replacement is not None:
            setattr(model, name, replacement)
            replaced += 1
        else:
            replaced += replace_attention_modules(child, softmax_fn, attn_implementation, policy_names)

    return replaced


def replace_qwen3_attention_modules(
    model: nn.Module,
    softmax_fn: Union[str, Callable[..., torch.Tensor]] = "softmax1",
    attn_implementation: str = "sdpa",
) -> int:
    """Replace every Qwen3 self-attention module in a loaded model."""
    return replace_attention_modules(
        model,
        softmax_fn=softmax_fn,
        attn_implementation=attn_implementation,
        policy_names=("qwen3",),
    )


def apply_softmax1_attention(model: nn.Module, attn_implementation: str = "sdpa") -> nn.Module:
    """Make a loaded Qwen3 model use softmax1 in every attention layer."""
    replaced = replace_attention_modules(
        model,
        softmax_fn="softmax1",
        attn_implementation=attn_implementation,
    )
    if replaced == 0:
        supported = ", ".join(sorted(ATTENTION_REPLACEMENT_POLICIES))
        raise ValueError(f"No supported attention modules found. Registered policies: {supported}")
    if hasattr(model, "config"):
        model.config._attn_implementation = attn_implementation
    return model


def apply_softmax_attention(
    model: nn.Module,
    softmax_fn: Union[str, Callable[..., torch.Tensor]] = "softmax1",
    attn_implementation: str = "sdpa",
) -> nn.Module:
    """Make a loaded HuggingFace model use a custom softmax where a policy supports it."""
    replaced = replace_attention_modules(
        model,
        softmax_fn=softmax_fn,
        attn_implementation=attn_implementation,
    )
    if replaced == 0:
        supported = ", ".join(sorted(ATTENTION_REPLACEMENT_POLICIES))
        raise ValueError(f"No supported attention modules found. Registered policies: {supported}")
    if hasattr(model, "config"):
        model.config._attn_implementation = attn_implementation
    return model


def from_pretrained_with_softmax_attention(
    model_name_or_path: str,
    softmax_fn: Union[str, Callable[..., torch.Tensor]] = "softmax1",
    attn_implementation: str = "sdpa",
    auto_model_cls=None,
    **from_pretrained_kwargs,
) -> nn.Module:
    """Load a HuggingFace model, then patch every supported attention layer."""
    if auto_model_cls is None:
        from transformers import AutoModelForCausalLM

        auto_model_cls = AutoModelForCausalLM

    model = auto_model_cls.from_pretrained(model_name_or_path, **from_pretrained_kwargs)
    return apply_softmax_attention(
        model,
        softmax_fn=softmax_fn,
        attn_implementation=attn_implementation,
    )


def from_pretrained_with_registered_softmax_attention(
    model_name_or_path: str,
    softmax_fn: Union[str, Callable[..., torch.Tensor]] = "softmax1",
    base_backend: str = "sdpa",
    mode: str = "strict",
    fallback_backend: str = "sdpa",
    auto_model_cls=None,
    **from_pretrained_kwargs,
) -> nn.Module:
    """Load through HuggingFace's native attn_implementation registry."""
    if softmax_fn == "sparsemax":
        register_sparsemax_attention_backends(mode=mode, fallback_backend=fallback_backend)
    elif softmax_fn == "entmax15":
        register_entmax15_attention_backends(mode=mode, fallback_backend=fallback_backend)
    elif softmax_fn in {"softmax1", "softmax_1"}:
        register_softmax1_attention_backends(mode=mode, fallback_backend=fallback_backend)
    else:
        register_softmax_attention_backends(
            softmax_fn=softmax_fn,
            mode=mode,
            fallback_backend=fallback_backend,
        )
    from_pretrained_kwargs.setdefault(
        "attn_implementation",
        softmax_attention_backend_name(base_backend, softmax_fn),
    )

    if auto_model_cls is None:
        from transformers import AutoModelForCausalLM

        auto_model_cls = AutoModelForCausalLM

    return auto_model_cls.from_pretrained(model_name_or_path, **from_pretrained_kwargs)
