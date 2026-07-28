from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _paged_kv_tile(
        cache,
        block_table,
        batch_index,
        kv_head,
        offsets_n,
        offsets_d,
        valid_n,
        valid_d,
        stride_cache_token: tl.constexpr,
        stride_cache_head: tl.constexpr,
        stride_cache_dim: tl.constexpr,
        stride_table_batch: tl.constexpr,
        stride_table_block: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
    ):
        logical_blocks = offsets_n // PAGE_SIZE
        physical_blocks = tl.load(
            block_table
            + batch_index * stride_table_batch
            + logical_blocks * stride_table_block,
            mask=valid_n,
            other=0,
        )
        physical_tokens = physical_blocks * PAGE_SIZE + offsets_n % PAGE_SIZE
        values = tl.load(
            cache
            + physical_tokens[:, None] * stride_cache_token
            + kv_head * stride_cache_head
            + offsets_d[None, :] * stride_cache_dim,
            mask=valid_n[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)
        return values, physical_tokens


    @triton.jit
    def _paged_sparse_forward(
        query,
        key_cache,
        value_cache,
        block_table,
        sequence_lengths,
        output,
        thresholds,
        stride_qb: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_ql: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kt: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vt: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_btb: tl.constexpr,
        stride_btn: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_ol: tl.constexpr,
        stride_od: tl.constexpr,
        stride_tb: tl.constexpr,
        stride_th: tl.constexpr,
        stride_tl: tl.constexpr,
        query_length: tl.constexpr,
        max_key_length: tl.constexpr,
        head_dim: tl.constexpr,
        num_query_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        scale: tl.constexpr,
        is_causal: tl.constexpr,
        mode: tl.constexpr,
        BISECTION_STEPS: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        query_head = tl.program_id(1)
        query_index = tl.program_id(2)
        kv_head = query_head // (num_query_heads // num_kv_heads)
        key_length = tl.load(sequence_lengths + batch_index)
        causal_limit = query_index + key_length - query_length
        offsets_d = tl.arange(0, BLOCK_D)
        valid_d = offsets_d < head_dim
        q = tl.load(
            query
            + batch_index * stride_qb
            + query_head * stride_qh
            + query_index * stride_ql
            + offsets_d * stride_qd,
            mask=valid_d,
            other=0.0,
        ).to(tl.float32)

        lower = 3.4028234663852886e38
        upper = -3.4028234663852886e38
        for start_n in tl.range(0, max_key_length, BLOCK_N):
            offsets_n = start_n + tl.arange(0, BLOCK_N)
            valid_n = offsets_n < key_length
            if is_causal:
                valid_n = valid_n & (offsets_n <= causal_limit)
            keys, physical_tokens = _paged_kv_tile(
                key_cache, block_table, batch_index, kv_head, offsets_n, offsets_d,
                valid_n, valid_d, stride_kt, stride_kh, stride_kd,
                stride_btb, stride_btn, PAGE_SIZE,
            )
            scores = tl.sum(keys * q[None, :], axis=1) * scale
            z = scores if mode == 0 else scores * 0.5
            lower = tl.minimum(lower, tl.min(tl.where(valid_n, z, 3.4028234663852886e38), axis=0))
            upper = tl.maximum(upper, tl.max(tl.where(valid_n, z, -3.4028234663852886e38), axis=0))

        lower -= 1.0
        for bisection_index in tl.static_range(BISECTION_STEPS):
            midpoint = (lower + upper) * 0.5
            mass = tl.full((), 0.0, tl.float32)
            for start_n in tl.range(0, max_key_length, BLOCK_N):
                offsets_n = start_n + tl.arange(0, BLOCK_N)
                valid_n = offsets_n < key_length
                if is_causal:
                    valid_n = valid_n & (offsets_n <= causal_limit)
                keys, physical_tokens = _paged_kv_tile(
                    key_cache, block_table, batch_index, kv_head, offsets_n, offsets_d,
                    valid_n, valid_d, stride_kt, stride_kh, stride_kd,
                    stride_btb, stride_btn, PAGE_SIZE,
                )
                scores = tl.sum(keys * q[None, :], axis=1) * scale
                positive = tl.maximum((scores if mode == 0 else scores * 0.5) - midpoint, 0.0)
                probabilities = positive if mode == 0 else positive * positive
                mass += tl.sum(tl.where(valid_n, probabilities, 0.0), axis=0)
            if mass > 1.0:
                lower = midpoint
            else:
                upper = midpoint
        threshold = (lower + upper) * 0.5
        tl.store(
            thresholds + batch_index * stride_tb + query_head * stride_th + query_index * stride_tl,
            threshold,
        )

        accumulator = tl.zeros((BLOCK_D,), tl.float32)
        for start_n in tl.range(0, max_key_length, BLOCK_N):
            offsets_n = start_n + tl.arange(0, BLOCK_N)
            valid_n = offsets_n < key_length
            if is_causal:
                valid_n = valid_n & (offsets_n <= causal_limit)
            keys, physical_tokens = _paged_kv_tile(
                key_cache, block_table, batch_index, kv_head, offsets_n, offsets_d,
                valid_n, valid_d, stride_kt, stride_kh, stride_kd,
                stride_btb, stride_btn, PAGE_SIZE,
            )
            values, physical_tokens = _paged_kv_tile(
                value_cache, block_table, batch_index, kv_head, offsets_n, offsets_d,
                valid_n, valid_d, stride_vt, stride_vh, stride_vd,
                stride_btb, stride_btn, PAGE_SIZE,
            )
            scores = tl.sum(keys * q[None, :], axis=1) * scale
            positive = tl.maximum((scores if mode == 0 else scores * 0.5) - threshold, 0.0)
            probabilities = positive if mode == 0 else positive * positive
            probabilities = tl.where(valid_n, probabilities, 0.0)
            accumulator += tl.sum(probabilities[:, None] * values, axis=0)
        tl.store(
            output
            + batch_index * stride_ob
            + query_head * stride_oh
            + query_index * stride_ol
            + offsets_d * stride_od,
            accumulator,
            mask=valid_d,
        )


    @triton.jit
    def _paged_sparse_backward(
        query, key_cache, value_cache, block_table, sequence_lengths, grad_output,
        thresholds, grad_query, grad_key_cache, grad_value_cache,
        stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_ql: tl.constexpr, stride_qd: tl.constexpr,
        stride_kt: tl.constexpr, stride_kh: tl.constexpr, stride_kd: tl.constexpr,
        stride_vt: tl.constexpr, stride_vh: tl.constexpr, stride_vd: tl.constexpr,
        stride_btb: tl.constexpr, stride_btn: tl.constexpr,
        stride_dob: tl.constexpr, stride_doh: tl.constexpr, stride_dol: tl.constexpr, stride_dod: tl.constexpr,
        stride_tb: tl.constexpr, stride_th: tl.constexpr, stride_tl: tl.constexpr,
        stride_dqb: tl.constexpr, stride_dqh: tl.constexpr, stride_dql: tl.constexpr, stride_dqd: tl.constexpr,
        stride_dkt: tl.constexpr, stride_dkh: tl.constexpr, stride_dkd: tl.constexpr,
        stride_dvt: tl.constexpr, stride_dvh: tl.constexpr, stride_dvd: tl.constexpr,
        query_length: tl.constexpr, max_key_length: tl.constexpr, head_dim: tl.constexpr,
        num_query_heads: tl.constexpr, num_kv_heads: tl.constexpr, scale: tl.constexpr,
        is_causal: tl.constexpr, mode: tl.constexpr, PAGE_SIZE: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        query_head = tl.program_id(1)
        query_index = tl.program_id(2)
        kv_head = query_head // (num_query_heads // num_kv_heads)
        key_length = tl.load(sequence_lengths + batch_index)
        causal_limit = query_index + key_length - query_length
        offsets_d = tl.arange(0, BLOCK_D)
        valid_d = offsets_d < head_dim
        q = tl.load(
            query + batch_index * stride_qb + query_head * stride_qh
            + query_index * stride_ql + offsets_d * stride_qd,
            mask=valid_d, other=0.0,
        ).to(tl.float32)
        grad_out = tl.load(
            grad_output + batch_index * stride_dob + query_head * stride_doh
            + query_index * stride_dol + offsets_d * stride_dod,
            mask=valid_d, other=0.0,
        ).to(tl.float32)
        threshold = tl.load(
            thresholds + batch_index * stride_tb + query_head * stride_th + query_index * stride_tl
        )

        numerator = tl.full((), 0.0, tl.float32)
        denominator = tl.full((), 0.0, tl.float32)
        for start_n in tl.range(0, max_key_length, BLOCK_N):
            offsets_n = start_n + tl.arange(0, BLOCK_N)
            valid_n = offsets_n < key_length
            if is_causal:
                valid_n = valid_n & (offsets_n <= causal_limit)
            keys, physical_tokens = _paged_kv_tile(
                key_cache, block_table, batch_index, kv_head, offsets_n, offsets_d,
                valid_n, valid_d, stride_kt, stride_kh, stride_kd,
                stride_btb, stride_btn, PAGE_SIZE,
            )
            values, physical_tokens = _paged_kv_tile(
                value_cache, block_table, batch_index, kv_head, offsets_n, offsets_d,
                valid_n, valid_d, stride_vt, stride_vh, stride_vd,
                stride_btb, stride_btn, PAGE_SIZE,
            )
            scores = tl.sum(keys * q[None, :], axis=1) * scale
            positive = tl.maximum((scores if mode == 0 else scores * 0.5) - threshold, 0.0)
            inverse_hessian = (positive > 0).to(tl.float32) if mode == 0 else positive
            inverse_hessian = tl.where(valid_n, inverse_hessian, 0.0)
            grad_probabilities = tl.sum(values * grad_out[None, :], axis=1)
            numerator += tl.sum(inverse_hessian * grad_probabilities, axis=0)
            denominator += tl.sum(inverse_hessian, axis=0)
        correction = numerator / denominator

        grad_q = tl.zeros((BLOCK_D,), tl.float32)
        for start_n in tl.range(0, max_key_length, BLOCK_N):
            offsets_n = start_n + tl.arange(0, BLOCK_N)
            valid_n = offsets_n < key_length
            if is_causal:
                valid_n = valid_n & (offsets_n <= causal_limit)
            keys, physical_tokens = _paged_kv_tile(
                key_cache, block_table, batch_index, kv_head, offsets_n, offsets_d,
                valid_n, valid_d, stride_kt, stride_kh, stride_kd,
                stride_btb, stride_btn, PAGE_SIZE,
            )
            values, physical_tokens = _paged_kv_tile(
                value_cache, block_table, batch_index, kv_head, offsets_n, offsets_d,
                valid_n, valid_d, stride_vt, stride_vh, stride_vd,
                stride_btb, stride_btn, PAGE_SIZE,
            )
            scores = tl.sum(keys * q[None, :], axis=1) * scale
            positive = tl.maximum((scores if mode == 0 else scores * 0.5) - threshold, 0.0)
            probabilities = positive if mode == 0 else positive * positive
            probabilities = tl.where(valid_n, probabilities, 0.0)
            inverse_hessian = (positive > 0).to(tl.float32) if mode == 0 else positive
            grad_probabilities = tl.sum(values * grad_out[None, :], axis=1)
            grad_scores = tl.where(valid_n, inverse_hessian * (grad_probabilities - correction), 0.0)
            grad_q += tl.sum(grad_scores[:, None] * keys, axis=0) * scale
            tl.atomic_add(
                grad_key_cache + physical_tokens[:, None] * stride_dkt
                + kv_head * stride_dkh + offsets_d[None, :] * stride_dkd,
                grad_scores[:, None] * q[None, :] * scale,
                mask=valid_n[:, None] & valid_d[None, :],
            )
            tl.atomic_add(
                grad_value_cache + physical_tokens[:, None] * stride_dvt
                + kv_head * stride_dvh + offsets_d[None, :] * stride_dvd,
                probabilities[:, None] * grad_out[None, :],
                mask=valid_n[:, None] & valid_d[None, :],
            )
        tl.store(
            grad_query + batch_index * stride_dqb + query_head * stride_dqh
            + query_index * stride_dql + offsets_d * stride_dqd,
            grad_q, mask=valid_d,
        )


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


class _PagedSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, query, key_cache, value_cache, block_table, sequence_lengths,
        page_size, scale, is_causal, mode, block_n, bisection_steps,
    ):
        batch, query_heads, query_length, head_dim = query.shape
        kv_heads = key_cache.shape[1]
        max_key_length = int(sequence_lengths.max().item())
        block_d = _next_power_of_2(head_dim)
        output = torch.empty_like(query)
        thresholds = torch.empty(
            batch, query_heads, query_length, device=query.device, dtype=torch.float32
        )
        grid = (batch, query_heads, query_length)
        _paged_sparse_forward[grid](
            query, key_cache, value_cache, block_table, sequence_lengths, output, thresholds,
            *query.stride(), *key_cache.stride(), *value_cache.stride(), *block_table.stride(),
            *output.stride(), *thresholds.stride(), query_length, max_key_length, head_dim,
            query_heads, kv_heads, scale, is_causal, mode,
            BISECTION_STEPS=bisection_steps, PAGE_SIZE=page_size,
            BLOCK_N=block_n, BLOCK_D=block_d,
        )
        ctx.save_for_backward(
            query, key_cache, value_cache, block_table, sequence_lengths, thresholds
        )
        ctx.args = (page_size, scale, is_causal, mode, block_n)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        query, key_cache, value_cache, block_table, sequence_lengths, thresholds = ctx.saved_tensors
        page_size, scale, is_causal, mode, block_n = ctx.args
        batch, query_heads, query_length, head_dim = query.shape
        kv_heads = key_cache.shape[1]
        max_key_length = int(sequence_lengths.max().item())
        block_d = _next_power_of_2(head_dim)
        grad_query = torch.empty_like(query, dtype=torch.float32)
        grad_key = torch.zeros_like(key_cache, dtype=torch.float32)
        grad_value = torch.zeros_like(value_cache, dtype=torch.float32)
        grid = (batch, query_heads, query_length)
        _paged_sparse_backward[grid](
            query, key_cache, value_cache, block_table, sequence_lengths, grad_output,
            thresholds, grad_query, grad_key, grad_value,
            *query.stride(), *key_cache.stride(), *value_cache.stride(), *block_table.stride(),
            *grad_output.stride(), *thresholds.stride(), *grad_query.stride(),
            *grad_key.stride(), *grad_value.stride(), query_length, max_key_length, head_dim,
            query_heads, kv_heads, scale, is_causal, mode, PAGE_SIZE=page_size,
            BLOCK_N=block_n, BLOCK_D=block_d,
        )
        return grad_query, grad_key, grad_value, None, None, None, None, None, None, None, None


def paged_hopper_sparse_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    sequence_lengths: torch.Tensor,
    page_size: int,
    normalizer: str,
    scale: Optional[float] = None,
    is_causal: bool = True,
    block_n: int = 128,
    bisection_steps: int = 24,
) -> torch.Tensor:
    if triton is None:
        raise ImportError("Paged Hopper attention requires Triton.")
    if normalizer not in {"sparsemax", "entmax15"}:
        raise ValueError("normalizer must be 'sparsemax' or 'entmax15'.")
    if not query.is_cuda or not key_cache.is_cuda or not value_cache.is_cuda:
        raise ValueError("Paged Hopper attention requires CUDA tensors.")
    if key_cache.shape != value_cache.shape or key_cache.ndim != 3:
        raise ValueError("K/V caches must have matching [physical_tokens, kv_heads, dim] shapes.")
    if query.shape[0] != block_table.shape[0] or query.shape[0] != sequence_lengths.shape[0]:
        raise ValueError("Batch dimensions must match.")
    if query.shape[1] % key_cache.shape[1] != 0:
        raise ValueError("Query heads must be divisible by KV heads.")
    actual_scale = query.shape[-1] ** -0.5 if scale is None else scale
    mode = 0 if normalizer == "sparsemax" else 1
    return _PagedSparseAttention.apply(
        query, key_cache, value_cache, block_table, sequence_lengths,
        page_size, actual_scale, is_causal, mode, block_n, bisection_steps,
    )
