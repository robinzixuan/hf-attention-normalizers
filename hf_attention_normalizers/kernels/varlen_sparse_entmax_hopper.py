"""Single-grid packed sparsemax/entmax15 Hopper attention."""

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
        query, key, value, cu_q, cu_k, output, thresholds,
        sqt: tl.constexpr, sqh: tl.constexpr, sqd: tl.constexpr,
        skt: tl.constexpr, skh: tl.constexpr, skd: tl.constexpr,
        svt: tl.constexpr, svh: tl.constexpr, svd: tl.constexpr,
        sot: tl.constexpr, soh: tl.constexpr, sod: tl.constexpr,
        stt: tl.constexpr, sth: tl.constexpr,
        scale: tl.constexpr, is_causal: tl.constexpr, groups: tl.constexpr, mode: tl.constexpr,
        max_q: tl.constexpr, max_k: tl.constexpr, head_dim: tl.constexpr,
        block_n: tl.constexpr, block_d: tl.constexpr, bisection_steps: tl.constexpr,
    ):
        batch, head, q_local = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        q_start, q_end = tl.load(cu_q + batch), tl.load(cu_q + batch + 1)
        k_start, k_end = tl.load(cu_k + batch), tl.load(cu_k + batch + 1)
        q_len, k_len, valid_q, kv_head = q_end - q_start, k_end - k_start, q_local < (q_end - q_start), head // groups
        offs_d = tl.arange(0, block_d)
        valid_d = offs_d < head_dim
        q = tl.load(query + (q_start + q_local) * sqt + head * sqh + offs_d * sqd, mask=valid_q & valid_d, other=0.).to(tl.float32)
        causal_limit = q_local + k_len - q_len
        lower, upper = 3.4028234663852886e38, -3.4028234663852886e38
        for start in tl.range(0, max_k, block_n):
            n = start + tl.arange(0, block_n)
            valid = n < k_len
            if is_causal: valid = valid & (n <= causal_limit)
            keys = tl.load(key + (k_start + n[:, None]) * skt + kv_head * skh + offs_d[None, :] * skd, mask=valid[:, None] & valid_d[None, :] & valid_q, other=0.).to(tl.float32)
            z = tl.sum(keys * q[None, :], axis=1) * scale
            z = z if mode == 0 else z * .5
            lower = tl.minimum(lower, tl.min(tl.where(valid, z, 3.4028234663852886e38), axis=0))
            upper = tl.maximum(upper, tl.max(tl.where(valid, z, -3.4028234663852886e38), axis=0))
        lower -= 1.
        for _ in tl.static_range(bisection_steps):
            midpoint, mass = (lower + upper) * .5, 0.
            for start in tl.range(0, max_k, block_n):
                n = start + tl.arange(0, block_n)
                valid = n < k_len
                if is_causal: valid = valid & (n <= causal_limit)
                keys = tl.load(key + (k_start + n[:, None]) * skt + kv_head * skh + offs_d[None, :] * skd, mask=valid[:, None] & valid_d[None, :] & valid_q, other=0.).to(tl.float32)
                z = tl.sum(keys * q[None, :], axis=1) * scale
                z = z if mode == 0 else z * .5
                positive = tl.maximum(z - midpoint, 0.)
                mass += tl.sum(tl.where(valid, positive if mode == 0 else positive * positive, 0.), axis=0)
            if mass > 1.: lower = midpoint
            else: upper = midpoint
        threshold, acc = (lower + upper) * .5, tl.zeros((block_d,), tl.float32)
        for start in tl.range(0, max_k, block_n):
            n = start + tl.arange(0, block_n)
            valid = n < k_len
            if is_causal: valid = valid & (n <= causal_limit)
            keys = tl.load(key + (k_start + n[:, None]) * skt + kv_head * skh + offs_d[None, :] * skd, mask=valid[:, None] & valid_d[None, :] & valid_q, other=0.).to(tl.float32)
            z = tl.sum(keys * q[None, :], axis=1) * scale
            z = z if mode == 0 else z * .5
            pos = tl.maximum(z - threshold, 0.)
            p = tl.where(valid, pos if mode == 0 else pos * pos, 0.)
            vals = tl.load(value + (k_start + n[:, None]) * svt + kv_head * svh + offs_d[None, :] * svd, mask=valid[:, None] & valid_d[None, :] & valid_q, other=0.).to(tl.float32)
            acc += tl.sum(p[:, None] * vals, axis=0)
        tl.store(output + (q_start + q_local) * sot + head * soh + offs_d * sod, acc, mask=valid_q & valid_d)
        tl.store(thresholds + (q_start + q_local) * stt + head * sth, threshold, mask=valid_q)


    @triton.jit
    def _backward(
        query, key, value, grad_output, cu_q, cu_k, thresholds, grad_query, grad_key, grad_value,
        sqt: tl.constexpr, sqh: tl.constexpr, sqd: tl.constexpr,
        skt: tl.constexpr, skh: tl.constexpr, skd: tl.constexpr,
        svt: tl.constexpr, svh: tl.constexpr, svd: tl.constexpr,
        sdot: tl.constexpr, sdoh: tl.constexpr, sdod: tl.constexpr,
        stt: tl.constexpr, sth: tl.constexpr,
        sdqt: tl.constexpr, sdqh: tl.constexpr, sdqd: tl.constexpr,
        sdkt: tl.constexpr, sdkh: tl.constexpr, sdkd: tl.constexpr,
        sdvt: tl.constexpr, sdvh: tl.constexpr, sdvd: tl.constexpr,
        scale: tl.constexpr, is_causal: tl.constexpr, groups: tl.constexpr, mode: tl.constexpr,
        max_q: tl.constexpr, max_k: tl.constexpr, head_dim: tl.constexpr, block_n: tl.constexpr, block_d: tl.constexpr,
    ):
        batch, head, q_local = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        q_start, q_end = tl.load(cu_q + batch), tl.load(cu_q + batch + 1)
        k_start, k_end = tl.load(cu_k + batch), tl.load(cu_k + batch + 1)
        q_len, k_len, valid_q, kv_head = q_end - q_start, k_end - k_start, q_local < (q_end - q_start), head // groups
        d = tl.arange(0, block_d); valid_d = d < head_dim
        q = tl.load(query + (q_start + q_local) * sqt + head * sqh + d * sqd, mask=valid_q & valid_d, other=0.).to(tl.float32)
        do = tl.load(grad_output + (q_start + q_local) * sdot + head * sdoh + d * sdod, mask=valid_q & valid_d, other=0.).to(tl.float32)
        threshold = tl.load(thresholds + (q_start + q_local) * stt + head * sth, mask=valid_q, other=0.)
        causal_limit = q_local + k_len - q_len
        numerator, denominator = 0., 0.
        for start in tl.range(0, max_k, block_n):
            n = start + tl.arange(0, block_n); valid = n < k_len
            if is_causal: valid = valid & (n <= causal_limit)
            keys = tl.load(key + (k_start + n[:, None]) * skt + kv_head * skh + d[None, :] * skd, mask=valid[:, None] & valid_d[None, :] & valid_q, other=0.).to(tl.float32)
            vals = tl.load(value + (k_start + n[:, None]) * svt + kv_head * svh + d[None, :] * svd, mask=valid[:, None] & valid_d[None, :] & valid_q, other=0.).to(tl.float32)
            z = tl.sum(keys * q[None, :], axis=1) * scale; z = z if mode == 0 else z * .5
            pos = tl.maximum(z - threshold, 0.); inv = (pos > 0.).to(tl.float32) if mode == 0 else pos; inv = tl.where(valid, inv, 0.)
            numerator += tl.sum(tl.sum(vals * do[None, :], axis=1) * inv, axis=0); denominator += tl.sum(inv, axis=0)
        correction, dq = numerator / denominator, tl.zeros((block_d,), tl.float32)
        for start in tl.range(0, max_k, block_n):
            n = start + tl.arange(0, block_n); valid = n < k_len
            if is_causal: valid = valid & (n <= causal_limit)
            keys = tl.load(key + (k_start + n[:, None]) * skt + kv_head * skh + d[None, :] * skd, mask=valid[:, None] & valid_d[None, :] & valid_q, other=0.).to(tl.float32)
            vals = tl.load(value + (k_start + n[:, None]) * svt + kv_head * svh + d[None, :] * svd, mask=valid[:, None] & valid_d[None, :] & valid_q, other=0.).to(tl.float32)
            z = tl.sum(keys * q[None, :], axis=1) * scale; z = z if mode == 0 else z * .5
            pos = tl.maximum(z - threshold, 0.); p = tl.where(valid, pos if mode == 0 else pos * pos, 0.); inv = (pos > 0.).to(tl.float32) if mode == 0 else pos; inv = tl.where(valid, inv, 0.)
            ds = inv * (tl.sum(vals * do[None, :], axis=1) - correction); ds = tl.where(valid, ds, 0.)
            dq += tl.sum(ds[:, None] * keys, axis=0) * scale
            tl.atomic_add(grad_key + (k_start + n[:, None]) * sdkt + kv_head * sdkh + d[None, :] * sdkd, ds[:, None] * q[None, :] * scale, mask=valid[:, None] & valid_d[None, :] & valid_q)
            tl.atomic_add(grad_value + (k_start + n[:, None]) * sdvt + kv_head * sdvh + d[None, :] * sdvd, p[:, None] * do[None, :], mask=valid[:, None] & valid_d[None, :] & valid_q)
        tl.store(grad_query + (q_start + q_local) * sdqt + head * sdqh + d * sdqd, dq, mask=valid_q & valid_d)


class _VarlenSparse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, cu_q, cu_k, scale, is_causal, max_q, max_k, mode, block_n, steps):
        total_q, heads, dim = query.shape; bd = _next_power_of_2(dim); output = torch.empty_like(query); thresholds = torch.empty((total_q, heads), device=query.device, dtype=torch.float32)
        _forward[(cu_q.numel() - 1, heads, max_q)](query, key, value, cu_q, cu_k, output, thresholds, *query.stride(), *key.stride(), *value.stride(), *output.stride(), *thresholds.stride(), scale, is_causal, heads // key.shape[1], mode, max_q, max_k, dim, block_n, bd, bisection_steps=steps)
        ctx.save_for_backward(query, key, value, cu_q, cu_k, thresholds); ctx.args = (scale, is_causal, max_q, max_k, mode, block_n)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, cu_q, cu_k, thresholds = ctx.saved_tensors; scale, causal, max_q, max_k, mode, bn = ctx.args; total_q, heads, dim = query.shape; bd = _next_power_of_2(dim)
        dq = torch.empty_like(query, dtype=torch.float32); dk = torch.zeros_like(key, dtype=torch.float32); dv = torch.zeros_like(value, dtype=torch.float32)
        _backward[(cu_q.numel() - 1, heads, max_q)](query, key, value, grad_output, cu_q, cu_k, thresholds, dq, dk, dv, *query.stride(), *key.stride(), *value.stride(), *grad_output.stride(), *thresholds.stride(), *dq.stride(), *dk.stride(), *dv.stride(), scale, causal, heads // key.shape[1], mode, max_q, max_k, dim, bn, bd)
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


def varlen_hopper_sparse_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, normalizer: str, max_seqlen_q: int, max_seqlen_k: int, scale: Optional[float] = None, is_causal: bool = True, block_n: int = 128, bisection_steps: int = 24) -> torch.Tensor:
    if triton is None: raise ImportError("Varlen Hopper attention requires Triton.")
    if _next_power_of_2(query.shape[-1]) > 256: raise ValueError("head_dim larger than 256 is not supported.")
    if normalizer not in {"sparsemax", "entmax15"}: raise ValueError("normalizer must be sparsemax or entmax15.")
    return _VarlenSparse.apply(query, key, value, cu_seqlens_q, cu_seqlens_k, query.shape[-1] ** -.5 if scale is None else scale, is_causal, max_seqlen_q, max_seqlen_k, 0 if normalizer == "sparsemax" else 1, block_n, bisection_steps)
