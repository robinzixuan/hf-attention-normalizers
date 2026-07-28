from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from .paged_sparse_entmax_hopper import _paged_kv_tile


if triton is not None:

    @triton.jit
    def _paged_softmax1_forward(
        query, key_cache, value_cache, block_table, sequence_lengths,
        output, row_max, row_denominator,
        stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_ql: tl.constexpr, stride_qd: tl.constexpr,
        stride_kt: tl.constexpr, stride_kh: tl.constexpr, stride_kd: tl.constexpr,
        stride_vt: tl.constexpr, stride_vh: tl.constexpr, stride_vd: tl.constexpr,
        stride_btb: tl.constexpr, stride_btn: tl.constexpr,
        stride_ob: tl.constexpr, stride_oh: tl.constexpr, stride_ol: tl.constexpr, stride_od: tl.constexpr,
        stride_mb: tl.constexpr, stride_mh: tl.constexpr, stride_ml: tl.constexpr,
        stride_lb: tl.constexpr, stride_lh: tl.constexpr, stride_ll: tl.constexpr,
        query_length: tl.constexpr, max_key_length: tl.constexpr, head_dim: tl.constexpr,
        num_query_heads: tl.constexpr, num_kv_heads: tl.constexpr, scale: tl.constexpr,
        is_causal: tl.constexpr, PAGE_SIZE: tl.constexpr,
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

        running_max = 0.0
        denominator = 1.0
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
            block_max = tl.max(tl.where(valid_n, scores, -3.4028234663852886e38), axis=0)
            new_max = tl.maximum(running_max, block_max)
            old_scale = tl.exp(running_max - new_max)
            exponentials = tl.where(valid_n, tl.exp(scores - new_max), 0.0)
            accumulator = accumulator * old_scale + tl.sum(exponentials[:, None] * values, axis=0)
            denominator = denominator * old_scale + tl.sum(exponentials, axis=0)
            running_max = new_max

        tl.store(
            output + batch_index * stride_ob + query_head * stride_oh
            + query_index * stride_ol + offsets_d * stride_od,
            accumulator / denominator, mask=valid_d,
        )
        tl.store(row_max + batch_index * stride_mb + query_head * stride_mh + query_index * stride_ml, running_max)
        tl.store(
            row_denominator + batch_index * stride_lb + query_head * stride_lh + query_index * stride_ll,
            denominator,
        )


    @triton.jit
    def _paged_softmax1_backward(
        query, key_cache, value_cache, block_table, sequence_lengths, grad_output,
        row_max, row_denominator, grad_query, grad_key_cache, grad_value_cache,
        stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_ql: tl.constexpr, stride_qd: tl.constexpr,
        stride_kt: tl.constexpr, stride_kh: tl.constexpr, stride_kd: tl.constexpr,
        stride_vt: tl.constexpr, stride_vh: tl.constexpr, stride_vd: tl.constexpr,
        stride_btb: tl.constexpr, stride_btn: tl.constexpr,
        stride_dob: tl.constexpr, stride_doh: tl.constexpr, stride_dol: tl.constexpr, stride_dod: tl.constexpr,
        stride_mb: tl.constexpr, stride_mh: tl.constexpr, stride_ml: tl.constexpr,
        stride_lb: tl.constexpr, stride_lh: tl.constexpr, stride_ll: tl.constexpr,
        stride_dqb: tl.constexpr, stride_dqh: tl.constexpr, stride_dql: tl.constexpr, stride_dqd: tl.constexpr,
        stride_dkt: tl.constexpr, stride_dkh: tl.constexpr, stride_dkd: tl.constexpr,
        stride_dvt: tl.constexpr, stride_dvh: tl.constexpr, stride_dvd: tl.constexpr,
        query_length: tl.constexpr, max_key_length: tl.constexpr, head_dim: tl.constexpr,
        num_query_heads: tl.constexpr, num_kv_heads: tl.constexpr, scale: tl.constexpr,
        is_causal: tl.constexpr, PAGE_SIZE: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, WRITE_KV: tl.constexpr,
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
        maximum = tl.load(row_max + batch_index * stride_mb + query_head * stride_mh + query_index * stride_ml)
        denominator = tl.load(
            row_denominator + batch_index * stride_lb + query_head * stride_lh + query_index * stride_ll
        )

        correction = tl.full((), 0.0, tl.float32)
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
            probabilities = tl.where(valid_n, tl.exp(scores - maximum) / denominator, 0.0)
            correction += tl.sum(probabilities * tl.sum(values * grad_out[None, :], axis=1), axis=0)

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
            probabilities = tl.where(valid_n, tl.exp(scores - maximum) / denominator, 0.0)
            grad_probabilities = tl.sum(values * grad_out[None, :], axis=1)
            grad_scores = probabilities * (grad_probabilities - correction)
            grad_q += tl.sum(grad_scores[:, None] * keys, axis=0) * scale
            if WRITE_KV:
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


    @triton.jit
    def _paged_softmax1_kv_backward(
        query, key_cache, value_cache, block_table, sequence_lengths, grad_output, output,
        row_max, row_denominator, grad_key_cache, grad_value_cache,
        stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_ql: tl.constexpr, stride_qd: tl.constexpr,
        stride_kt: tl.constexpr, stride_kh: tl.constexpr, stride_kd: tl.constexpr,
        stride_vt: tl.constexpr, stride_vh: tl.constexpr, stride_vd: tl.constexpr,
        stride_btb: tl.constexpr, stride_btn: tl.constexpr,
        stride_dob: tl.constexpr, stride_doh: tl.constexpr, stride_dol: tl.constexpr, stride_dod: tl.constexpr,
        stride_ob: tl.constexpr, stride_oh: tl.constexpr, stride_ol: tl.constexpr, stride_od: tl.constexpr,
        stride_mb: tl.constexpr, stride_mh: tl.constexpr, stride_ml: tl.constexpr,
        stride_lb: tl.constexpr, stride_lh: tl.constexpr, stride_ll: tl.constexpr,
        stride_dkt: tl.constexpr, stride_dkh: tl.constexpr, stride_dkd: tl.constexpr,
        stride_dvt: tl.constexpr, stride_dvh: tl.constexpr, stride_dvd: tl.constexpr,
        query_length: tl.constexpr, max_key_length: tl.constexpr, head_dim: tl.constexpr,
        num_query_heads: tl.constexpr, num_kv_heads: tl.constexpr, scale: tl.constexpr,
        is_causal: tl.constexpr, PAGE_SIZE: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        batch_index = tl.program_id(0)
        kv_head = tl.program_id(1)
        tile_index = tl.program_id(2)
        offsets_n = tile_index * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_d = tl.arange(0, BLOCK_D)
        key_length = tl.load(sequence_lengths + batch_index)
        valid_n = offsets_n < key_length
        valid_d = offsets_d < head_dim
        logical_blocks = offsets_n // PAGE_SIZE
        physical_blocks = tl.load(
            block_table + batch_index * stride_btb + logical_blocks * stride_btn,
            mask=valid_n,
            other=0,
        )
        physical_tokens = physical_blocks * PAGE_SIZE + offsets_n % PAGE_SIZE
        grad_key = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
        grad_value = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
        for group_index in tl.static_range(0, num_query_heads // num_kv_heads):
            query_head = kv_head * (num_query_heads // num_kv_heads) + group_index
            for query_index in tl.range(0, query_length):
                causal_limit = query_index + key_length - query_length
                row_valid = valid_n
                if is_causal:
                    row_valid = row_valid & (offsets_n <= causal_limit)
                q = tl.load(
                    query + batch_index * stride_qb + query_head * stride_qh
                    + query_index * stride_ql + offsets_d * stride_qd,
                    mask=valid_d,
                    other=0.0,
                ).to(tl.float32)
                grad_out = tl.load(
                    grad_output + batch_index * stride_dob + query_head * stride_doh
                    + query_index * stride_dol + offsets_d * stride_dod,
                    mask=valid_d,
                    other=0.0,
                ).to(tl.float32)
                attention_output = tl.load(
                    output + batch_index * stride_ob + query_head * stride_oh
                    + query_index * stride_ol + offsets_d * stride_od,
                    mask=valid_d,
                    other=0.0,
                ).to(tl.float32)
                maximum = tl.load(row_max + batch_index * stride_mb + query_head * stride_mh + query_index * stride_ml)
                denominator = tl.load(row_denominator + batch_index * stride_lb + query_head * stride_lh + query_index * stride_ll)
                keys = tl.load(
                    key_cache + physical_tokens[:, None] * stride_kt + kv_head * stride_kh + offsets_d[None, :] * stride_kd,
                    mask=row_valid[:, None] & valid_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                values = tl.load(
                    value_cache + physical_tokens[:, None] * stride_vt + kv_head * stride_vh + offsets_d[None, :] * stride_vd,
                    mask=row_valid[:, None] & valid_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                scores = tl.sum(keys * q[None, :], axis=1) * scale
                probabilities = tl.where(row_valid, tl.exp(scores - maximum) / denominator, 0.0)
                grad_probabilities = tl.sum(values * grad_out[None, :], axis=1)
                correction = tl.sum(attention_output * grad_out, axis=0)
                grad_scores = probabilities * (grad_probabilities - correction)
                grad_key += grad_scores[:, None] * q[None, :] * scale
                grad_value += probabilities[:, None] * grad_out[None, :]
        tl.store(
            grad_key_cache + physical_tokens[:, None] * stride_dkt + kv_head * stride_dkh + offsets_d[None, :] * stride_dkd,
            grad_key,
            mask=valid_n[:, None] & valid_d[None, :],
        )
        tl.store(
            grad_value_cache + physical_tokens[:, None] * stride_dvt + kv_head * stride_dvh + offsets_d[None, :] * stride_dvd,
            grad_value,
            mask=valid_n[:, None] & valid_d[None, :],
        )


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


class _PagedSoftmax1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key_cache, value_cache, block_table, sequence_lengths, page_size, scale, is_causal, block_n, max_key_length, assume_unique_pages):
        batch, query_heads, query_length, head_dim = query.shape
        kv_heads = key_cache.shape[1]
        if max_key_length is None:
            max_key_length = int(sequence_lengths.max().item())
        block_d = _next_power_of_2(head_dim)
        output = torch.empty_like(query)
        row_max = torch.empty(batch, query_heads, query_length, device=query.device, dtype=torch.float32)
        row_denominator = torch.empty_like(row_max)
        grid = (batch, query_heads, query_length)
        _paged_softmax1_forward[grid](
            query, key_cache, value_cache, block_table, sequence_lengths,
            output, row_max, row_denominator,
            *query.stride(), *key_cache.stride(), *value_cache.stride(), *block_table.stride(),
            *output.stride(), *row_max.stride(), *row_denominator.stride(),
            query_length, max_key_length, head_dim, query_heads, kv_heads, scale, is_causal,
            PAGE_SIZE=page_size, BLOCK_N=block_n, BLOCK_D=block_d,
        )
        ctx.save_for_backward(query, key_cache, value_cache, block_table, sequence_lengths, output, row_max, row_denominator)
        ctx.args = (page_size, scale, is_causal, block_n, max_key_length, assume_unique_pages)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        query, key_cache, value_cache, block_table, sequence_lengths, output, row_max, row_denominator = ctx.saved_tensors
        page_size, scale, is_causal, block_n, max_key_length, assume_unique_pages = ctx.args
        batch, query_heads, query_length, head_dim = query.shape
        kv_heads = key_cache.shape[1]
        block_d = _next_power_of_2(head_dim)
        grad_query = torch.empty_like(query, dtype=torch.float32)
        grad_key = torch.zeros_like(key_cache, dtype=torch.float32)
        grad_value = torch.zeros_like(value_cache, dtype=torch.float32)
        # When each logical row owns distinct physical blocks, accumulate all
        # Q-head/Q-token contributions per KV tile and store once. Shared pages
        # retain the atomic path to preserve cross-request accumulation.
        no_shared_pages = (
            assume_unique_pages
            if torch.compiler.is_compiling()
            else assume_unique_pages or bool(torch.unique(block_table).numel() == block_table.numel())
        )
        grid = (batch, query_heads, query_length)
        _paged_softmax1_backward[grid](
            query, key_cache, value_cache, block_table, sequence_lengths, grad_output,
            row_max, row_denominator, grad_query, grad_key, grad_value,
            *query.stride(), *key_cache.stride(), *value_cache.stride(), *block_table.stride(),
            *grad_output.stride(), *row_max.stride(), *row_denominator.stride(),
            *grad_query.stride(), *grad_key.stride(), *grad_value.stride(),
            query_length, max_key_length, head_dim, query_heads, kv_heads, scale, is_causal,
            PAGE_SIZE=page_size, BLOCK_N=block_n, BLOCK_D=block_d, WRITE_KV=not no_shared_pages,
        )
        if no_shared_pages:
            reduction_block_n = min(16, block_n)
            reduction_grid = (batch, kv_heads, triton.cdiv(max_key_length, reduction_block_n))
            _paged_softmax1_kv_backward[reduction_grid](
                query, key_cache, value_cache, block_table, sequence_lengths, grad_output, output,
                row_max, row_denominator, grad_key, grad_value,
                *query.stride(), *key_cache.stride(), *value_cache.stride(), *block_table.stride(),
                *grad_output.stride(), *output.stride(), *row_max.stride(), *row_denominator.stride(),
                *grad_key.stride(), *grad_value.stride(),
                query_length, max_key_length, head_dim, query_heads, kv_heads, scale, is_causal,
                PAGE_SIZE=page_size, BLOCK_N=reduction_block_n, BLOCK_D=block_d,
            )
        return grad_query, grad_key, grad_value, None, None, None, None, None, None, None, None


def paged_hopper_softmax1_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    sequence_lengths: torch.Tensor,
    page_size: int,
    scale: Optional[float] = None,
    is_causal: bool = True,
    block_n: int = 128,
    max_key_length: Optional[int] = None,
    assume_unique_pages: bool = False,
) -> torch.Tensor:
    if triton is None:
        raise ImportError("Paged Softmax1 attention requires Triton.")
    if not query.is_cuda or not key_cache.is_cuda or not value_cache.is_cuda:
        raise ValueError("Paged Softmax1 attention requires CUDA tensors.")
    if key_cache.shape != value_cache.shape or query.shape[1] % key_cache.shape[1] != 0:
        raise ValueError("Incompatible Q/K/V head shapes.")
    actual_scale = query.shape[-1] ** -0.5 if scale is None else scale
    return _PagedSoftmax1.apply(
        query, key_cache, value_cache, block_table, sequence_lengths,
        page_size, actual_scale, is_causal, block_n, max_key_length, assume_unique_pages,
    )
