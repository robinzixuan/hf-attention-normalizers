from typing import Optional

import torch



def _validate_cumulative_lengths(
    cumulative_lengths: torch.Tensor,
    total_tokens: int,
    name: str,
) -> None:
    if cumulative_lengths.ndim != 1 or cumulative_lengths.numel() < 2:
        raise ValueError(f"{name} must have shape [batch + 1].")
    if cumulative_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must use int32 or int64.")
    if int(cumulative_lengths[0].item()) != 0:
        raise ValueError(f"{name} must start at zero.")
    if int(cumulative_lengths[-1].item()) != total_tokens:
        raise ValueError(f"{name} must end at the total token count.")
    if bool((cumulative_lengths[1:] < cumulative_lengths[:-1]).any()):
        raise ValueError(f"{name} must be nondecreasing.")


def varlen_hopper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    normalizer: str,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    scale: Optional[float] = None,
    is_causal: bool = True,
    block_n: int = 128,
    bisection_steps: int = 24,
) -> torch.Tensor:
    """Packed variable-length Hopper attention with full backward support.

    Q/K/V use FlashAttention's packed layout: ``[total_tokens, heads, dim]``.
    Each request is sliced from the cumulative sequence-length arrays and sent
    directly to the fused Hopper kernel without padding.
    """
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("Packed Q/K/V must have shape [total_tokens, heads, dim].")
    if key.shape != value.shape:
        raise ValueError("Packed K and V shapes must match.")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("Q/K/V head dimensions must match.")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("Query heads must be divisible by KV heads.")
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("Varlen Hopper attention requires CUDA tensors.")
    _validate_cumulative_lengths(cu_seqlens_q, query.shape[0], "cu_seqlens_q")
    _validate_cumulative_lengths(cu_seqlens_k, key.shape[0], "cu_seqlens_k")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("Q and K cumulative lengths must describe the same batch.")

    q_lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    k_lengths = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    if bool((q_lengths <= 0).any()) or bool((k_lengths <= 0).any()):
        raise ValueError("Empty packed sequences are not supported.")
    actual_max_q = int(q_lengths.max().item())
    actual_max_k = int(k_lengths.max().item())
    if max_seqlen_q is not None and max_seqlen_q < actual_max_q:
        raise ValueError("max_seqlen_q is smaller than an actual Q sequence.")
    if max_seqlen_k is not None and max_seqlen_k < actual_max_k:
        raise ValueError("max_seqlen_k is smaller than an actual K sequence.")

    if normalizer in {"softmax1", "softmax_1"}:
        # Packed kernels launch a single grid across every sequence and head.
        from .kernels.varlen_softmax1_hopper import varlen_hopper_softmax1_attention

        return varlen_hopper_softmax1_attention(
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=actual_max_q if max_seqlen_q is None else max_seqlen_q,
            max_seqlen_k=actual_max_k if max_seqlen_k is None else max_seqlen_k,
            scale=scale,
            is_causal=is_causal,
            block_n=block_n,
        )
    if normalizer not in {"sparsemax", "entmax15"}:
        raise ValueError("normalizer must be softmax1, sparsemax, or entmax15.")
    from .kernels.varlen_sparse_entmax_hopper import varlen_hopper_sparse_attention

    return varlen_hopper_sparse_attention(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        normalizer=normalizer,
        max_seqlen_q=actual_max_q if max_seqlen_q is None else max_seqlen_q,
        max_seqlen_k=actual_max_k if max_seqlen_k is None else max_seqlen_k,
        scale=scale,
        is_causal=is_causal,
        block_n=block_n,
        bisection_steps=bisection_steps,
    )
