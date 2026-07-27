from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def _require_triton() -> None:
    if triton is None:
        raise ImportError("This backend requires Triton. Install it with: pip install triton")


if triton is not None:

    @triton.jit
    def _sparse_entmax_attention_kernel(
        query,
        key,
        value,
        output,
        stride_qb: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_ql: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_ks: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vs: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_ol: tl.constexpr,
        stride_od: tl.constexpr,
        query_length: tl.constexpr,
        key_length: tl.constexpr,
        scale: tl.constexpr,
        is_causal: tl.constexpr,
        mode: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_l = tl.program_id(2)

        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        valid_n = offs_n < key_length

        q = tl.load(
            query + pid_b * stride_qb + pid_h * stride_qh + pid_l * stride_ql + offs_d * stride_qd,
            mask=offs_d < BLOCK_D,
            other=0.0,
        )
        k = tl.load(
            key
            + pid_b * stride_kb
            + pid_h * stride_kh
            + offs_n[:, None] * stride_ks
            + offs_d[None, :] * stride_kd,
            mask=valid_n[:, None],
            other=0.0,
        )

        scores = tl.sum(k * q[None, :], axis=1) * scale
        if is_causal:
            causal_limit = pid_l + key_length - query_length
            valid_n = valid_n & (offs_n <= causal_limit)
        scores = tl.where(valid_n, scores, -3.4028234663852886e38)

        if mode == 0:
            z = scores - tl.max(scores, axis=0)
            zs = tl.sort(z, descending=True)
            rho = tl.arange(1, BLOCK_N + 1)
            cssv = tl.cumsum(zs, axis=0)
            support = (1.0 + rho.to(tl.float32) * zs) > cssv
            support_f = support.to(tl.float32)
            support_size = tl.sum(support_f, axis=0)
            support_sum = tl.sum(tl.where(support, zs, 0.0), axis=0)
            tau = (support_sum - 1.0) / support_size
            probs = tl.maximum(z - tau, 0.0)
        else:
            z = (scores - tl.max(scores, axis=0)) * 0.5
            zs = tl.sort(z, descending=True)
            rho = tl.arange(1, BLOCK_N + 1).to(tl.float32)
            cssv = tl.cumsum(zs, axis=0)
            cssv_sq = tl.cumsum(zs * zs, axis=0)
            mean = cssv / rho
            mean_sq = cssv_sq / rho
            ss = rho * (mean_sq - mean * mean)
            delta = tl.maximum((1.0 - ss) / rho, 0.0)
            tau_candidates = mean - tl.sqrt(delta)
            support = tau_candidates <= zs
            support_f = support.to(tl.float32)
            support_size = tl.sum(support_f, axis=0)
            support_sum = tl.sum(tl.where(support, zs, 0.0), axis=0)
            support_sum_sq = tl.sum(tl.where(support, zs * zs, 0.0), axis=0)
            mean_star = support_sum / support_size
            mean_sq_star = support_sum_sq / support_size
            ss_star = support_size * (mean_sq_star - mean_star * mean_star)
            tau = mean_star - tl.sqrt(tl.maximum((1.0 - ss_star) / support_size, 0.0))
            probs = tl.maximum(z - tau, 0.0)
            probs = probs * probs

        v = tl.load(
            value
            + pid_b * stride_vb
            + pid_h * stride_vh
            + offs_n[:, None] * stride_vs
            + offs_d[None, :] * stride_vd,
            mask=valid_n[:, None],
            other=0.0,
        )
        out = tl.sum(probs[:, None] * v, axis=0)
        tl.store(
            output + pid_b * stride_ob + pid_h * stride_oh + pid_l * stride_ol + offs_d * stride_od,
            out,
            mask=offs_d < BLOCK_D,
        )


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _triton_sparse_entmax_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float],
    is_causal: bool,
    mode: int,
    max_block_n: int,
) -> torch.Tensor:
    _require_triton()
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("Triton sparse/entmax attention requires CUDA tensors.")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Expected query/key/value with shape [batch, heads, sequence, dim].")
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError("query/key/value batch sizes must match.")
    if query.shape[1] != key.shape[1] or query.shape[1] != value.shape[1]:
        raise ValueError("query/key/value head counts must match before calling the Triton kernel.")
    if query.shape[-1] != key.shape[-1] or query.shape[-1] != value.shape[-1]:
        raise ValueError("This experimental kernel currently requires query/key/value head dims to match.")

    batch, heads, query_length, head_dim = query.shape
    key_length = key.shape[-2]
    block_n = _next_power_of_2(key_length)
    block_d = _next_power_of_2(head_dim)
    if block_n > max_block_n:
        raise ValueError(f"key_length={key_length} exceeds max_block_n={max_block_n}.")
    if block_d > 256:
        raise ValueError("head_dim larger than 256 is not supported by this experimental kernel.")

    output = torch.empty_like(query)
    actual_scale = scale if scale is not None else head_dim**-0.5
    grid = (batch, heads, query_length)
    _sparse_entmax_attention_kernel[grid](
        query,
        key,
        value,
        output,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        query_length,
        key_length,
        actual_scale,
        is_causal,
        mode,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )
    return output


def triton_sparsemax_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
    is_causal: bool = False,
    max_block_n: int = 4096,
) -> torch.Tensor:
    return _triton_sparse_entmax_attention(query, key, value, scale, is_causal, mode=0, max_block_n=max_block_n)


def triton_entmax15_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
    is_causal: bool = False,
    max_block_n: int = 4096,
) -> torch.Tensor:
    return _triton_sparse_entmax_attention(query, key, value, scale, is_causal, mode=1, max_block_n=max_block_n)
