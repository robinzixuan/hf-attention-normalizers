"""Packed (varlen) Softmax1 Hopper attention.

Unlike the compatibility implementation, this launches one Triton grid across
all packed sequences.  The cumulative-length arrays are read in-kernel, so no
padded Q/K/V staging tensor is needed.
"""

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


if triton is not None:

    @triton.jit
    def _forward(
        query, key, value, cu_q, cu_k, output, row_max, row_denominator,
        stride_qt: tl.constexpr, stride_qh: tl.constexpr, stride_qd: tl.constexpr,
        stride_kt: tl.constexpr, stride_kh: tl.constexpr, stride_kd: tl.constexpr,
        stride_vt: tl.constexpr, stride_vh: tl.constexpr, stride_vd: tl.constexpr,
        stride_ot: tl.constexpr, stride_oh: tl.constexpr, stride_od: tl.constexpr,
        stride_mt: tl.constexpr, stride_mh: tl.constexpr,
        stride_lt: tl.constexpr, stride_lh: tl.constexpr,
        scale: tl.constexpr, is_causal: tl.constexpr, groups: tl.constexpr,
        max_q: tl.constexpr, max_k: tl.constexpr, head_dim: tl.constexpr,
        block_n: tl.constexpr, block_d: tl.constexpr,
    ):
        batch = tl.program_id(0)
        head = tl.program_id(1)
        q_local = tl.program_id(2)
        q_start = tl.load(cu_q + batch)
        q_end = tl.load(cu_q + batch + 1)
        k_start = tl.load(cu_k + batch)
        k_end = tl.load(cu_k + batch + 1)
        q_len = q_end - q_start
        k_len = k_end - k_start
        valid_q = q_local < q_len
        kv_head = head // groups
        offs_d = tl.arange(0, block_d)
        valid_d = offs_d < head_dim
        q = tl.load(
            query + (q_start + q_local) * stride_qt + head * stride_qh + offs_d * stride_qd,
            mask=valid_q & valid_d,
            other=0.0,
        ).to(tl.float32)
        running_max = 0.0
        denominator = 1.0
        accumulator = tl.zeros((block_d,), tl.float32)
        causal_limit = q_local + k_len - q_len
        for start in tl.range(0, max_k, block_n):
            offs_n = start + tl.arange(0, block_n)
            valid_n = offs_n < k_len
            if is_causal:
                valid_n = valid_n & (offs_n <= causal_limit)
            keys = tl.load(
                key + (k_start + offs_n[:, None]) * stride_kt + kv_head * stride_kh + offs_d[None, :] * stride_kd,
                mask=valid_n[:, None] & valid_d[None, :] & valid_q,
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(keys * q[None, :], axis=1) * scale
            block_max = tl.max(tl.where(valid_n, scores, -3.4028234663852886e38), axis=0)
            new_max = tl.maximum(running_max, block_max)
            previous_scale = tl.exp(running_max - new_max)
            exponentials = tl.where(valid_n, tl.exp(scores - new_max), 0.0)
            values = tl.load(
                value + (k_start + offs_n[:, None]) * stride_vt + kv_head * stride_vh + offs_d[None, :] * stride_vd,
                mask=valid_n[:, None] & valid_d[None, :] & valid_q,
                other=0.0,
            ).to(tl.float32)
            accumulator = accumulator * previous_scale + tl.sum(exponentials[:, None] * values, axis=0)
            denominator = denominator * previous_scale + tl.sum(exponentials, axis=0)
            running_max = new_max
        tl.store(output + (q_start + q_local) * stride_ot + head * stride_oh + offs_d * stride_od, accumulator / denominator, mask=valid_q & valid_d)
        tl.store(row_max + (q_start + q_local) * stride_mt + head * stride_mh, running_max, mask=valid_q)
        tl.store(row_denominator + (q_start + q_local) * stride_lt + head * stride_lh, denominator, mask=valid_q)


    @triton.jit
    def _backward(
        query, key, value, grad_output, cu_q, cu_k, row_max, row_denominator,
        grad_query, grad_key, grad_value,
        stride_qt: tl.constexpr, stride_qh: tl.constexpr, stride_qd: tl.constexpr,
        stride_kt: tl.constexpr, stride_kh: tl.constexpr, stride_kd: tl.constexpr,
        stride_vt: tl.constexpr, stride_vh: tl.constexpr, stride_vd: tl.constexpr,
        stride_dot: tl.constexpr, stride_doh: tl.constexpr, stride_dod: tl.constexpr,
        stride_mt: tl.constexpr, stride_mh: tl.constexpr, stride_lt: tl.constexpr, stride_lh: tl.constexpr,
        stride_dqt: tl.constexpr, stride_dqh: tl.constexpr, stride_dqd: tl.constexpr,
        stride_dkt: tl.constexpr, stride_dkh: tl.constexpr, stride_dkd: tl.constexpr,
        stride_dvt: tl.constexpr, stride_dvh: tl.constexpr, stride_dvd: tl.constexpr,
        scale: tl.constexpr, is_causal: tl.constexpr, groups: tl.constexpr,
        max_q: tl.constexpr, max_k: tl.constexpr, head_dim: tl.constexpr,
        block_n: tl.constexpr, block_d: tl.constexpr,
    ):
        batch = tl.program_id(0)
        head = tl.program_id(1)
        q_local = tl.program_id(2)
        q_start = tl.load(cu_q + batch)
        q_end = tl.load(cu_q + batch + 1)
        k_start = tl.load(cu_k + batch)
        k_end = tl.load(cu_k + batch + 1)
        q_len = q_end - q_start
        k_len = k_end - k_start
        valid_q = q_local < q_len
        kv_head = head // groups
        offs_d = tl.arange(0, block_d)
        valid_d = offs_d < head_dim
        q = tl.load(query + (q_start + q_local) * stride_qt + head * stride_qh + offs_d * stride_qd, mask=valid_q & valid_d, other=0.0).to(tl.float32)
        do = tl.load(grad_output + (q_start + q_local) * stride_dot + head * stride_doh + offs_d * stride_dod, mask=valid_q & valid_d, other=0.0).to(tl.float32)
        maximum = tl.load(row_max + (q_start + q_local) * stride_mt + head * stride_mh, mask=valid_q, other=0.0)
        denominator = tl.load(row_denominator + (q_start + q_local) * stride_lt + head * stride_lh, mask=valid_q, other=1.0)
        causal_limit = q_local + k_len - q_len
        correction = 0.0
        for start in tl.range(0, max_k, block_n):
            offs_n = start + tl.arange(0, block_n)
            valid_n = offs_n < k_len
            if is_causal:
                valid_n = valid_n & (offs_n <= causal_limit)
            keys = tl.load(key + (k_start + offs_n[:, None]) * stride_kt + kv_head * stride_kh + offs_d[None, :] * stride_kd, mask=valid_n[:, None] & valid_d[None, :] & valid_q, other=0.0).to(tl.float32)
            values = tl.load(value + (k_start + offs_n[:, None]) * stride_vt + kv_head * stride_vh + offs_d[None, :] * stride_vd, mask=valid_n[:, None] & valid_d[None, :] & valid_q, other=0.0).to(tl.float32)
            scores = tl.sum(keys * q[None, :], axis=1) * scale
            probabilities = tl.where(valid_n, tl.exp(scores - maximum) / denominator, 0.0)
            correction += tl.sum(probabilities * tl.sum(values * do[None, :], axis=1), axis=0)
        grad_q = tl.zeros((block_d,), tl.float32)
        for start in tl.range(0, max_k, block_n):
            offs_n = start + tl.arange(0, block_n)
            valid_n = offs_n < k_len
            if is_causal:
                valid_n = valid_n & (offs_n <= causal_limit)
            keys = tl.load(key + (k_start + offs_n[:, None]) * stride_kt + kv_head * stride_kh + offs_d[None, :] * stride_kd, mask=valid_n[:, None] & valid_d[None, :] & valid_q, other=0.0).to(tl.float32)
            values = tl.load(value + (k_start + offs_n[:, None]) * stride_vt + kv_head * stride_vh + offs_d[None, :] * stride_vd, mask=valid_n[:, None] & valid_d[None, :] & valid_q, other=0.0).to(tl.float32)
            scores = tl.sum(keys * q[None, :], axis=1) * scale
            probabilities = tl.where(valid_n, tl.exp(scores - maximum) / denominator, 0.0)
            grad_probabilities = tl.sum(values * do[None, :], axis=1)
            grad_scores = probabilities * (grad_probabilities - correction)
            grad_q += tl.sum(grad_scores[:, None] * keys, axis=0) * scale
            tl.atomic_add(grad_key + (k_start + offs_n[:, None]) * stride_dkt + kv_head * stride_dkh + offs_d[None, :] * stride_dkd, grad_scores[:, None] * q[None, :] * scale, mask=valid_n[:, None] & valid_d[None, :] & valid_q)
            tl.atomic_add(grad_value + (k_start + offs_n[:, None]) * stride_dvt + kv_head * stride_dvh + offs_d[None, :] * stride_dvd, probabilities[:, None] * do[None, :], mask=valid_n[:, None] & valid_d[None, :] & valid_q)
        tl.store(grad_query + (q_start + q_local) * stride_dqt + head * stride_dqh + offs_d * stride_dqd, grad_q, mask=valid_q & valid_d)


class _VarlenSoftmax1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, cu_q, cu_k, scale, is_causal, max_q, max_k, block_n):
        total_q, heads, head_dim = query.shape
        block_d = _next_power_of_2(head_dim)
        output = torch.empty_like(query)
        row_max = torch.empty((total_q, heads), device=query.device, dtype=torch.float32)
        row_denominator = torch.empty_like(row_max)
        groups = heads // key.shape[1]
        _forward[(cu_q.numel() - 1, heads, max_q)](
            query, key, value, cu_q, cu_k, output, row_max, row_denominator,
            *query.stride(), *key.stride(), *value.stride(), *output.stride(),
            *row_max.stride(), *row_denominator.stride(), scale, is_causal, groups,
            max_q, max_k, head_dim, block_n, block_d,
        )
        ctx.save_for_backward(query, key, value, cu_q, cu_k, row_max, row_denominator)
        ctx.scale, ctx.is_causal, ctx.max_q, ctx.max_k, ctx.block_n = scale, is_causal, max_q, max_k, block_n
        return output

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, cu_q, cu_k, row_max, row_denominator = ctx.saved_tensors
        total_q, heads, head_dim = query.shape
        block_d = _next_power_of_2(head_dim)
        grad_query = torch.empty_like(query, dtype=torch.float32)
        grad_key = torch.zeros_like(key, dtype=torch.float32)
        grad_value = torch.zeros_like(value, dtype=torch.float32)
        groups = heads // key.shape[1]
        _backward[(cu_q.numel() - 1, heads, ctx.max_q)](
            query, key, value, grad_output, cu_q, cu_k, row_max, row_denominator,
            grad_query, grad_key, grad_value,
            *query.stride(), *key.stride(), *value.stride(), *grad_output.stride(),
            *row_max.stride(), *row_denominator.stride(), *grad_query.stride(), *grad_key.stride(), *grad_value.stride(),
            ctx.scale, ctx.is_causal, groups, ctx.max_q, ctx.max_k, head_dim, ctx.block_n, block_d,
        )
        return grad_query, grad_key, grad_value, None, None, None, None, None, None, None


def varlen_hopper_softmax1_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int, max_seqlen_k: int, scale: Optional[float] = None,
    is_causal: bool = True, block_n: int = 128,
) -> torch.Tensor:
    if triton is None:
        raise ImportError("Varlen Hopper attention requires Triton.")
    if _next_power_of_2(query.shape[-1]) > 256:
        raise ValueError("head_dim larger than 256 is not supported.")
    actual_scale = query.shape[-1] ** -0.5 if scale is None else scale
    return _VarlenSoftmax1.apply(query, key, value, cu_seqlens_q, cu_seqlens_k, actual_scale, is_causal, max_seqlen_q, max_seqlen_k, block_n)
