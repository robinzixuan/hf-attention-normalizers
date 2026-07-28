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
        raise ImportError("Hopper sparse attention requires Triton.")


if triton is not None:

    @triton.jit
    def _score_tile(q, key_ptr, offs_n, offs_d, valid_n, valid_d, stride_ks, stride_kd, scale):
        key = tl.load(
            key_ptr + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
            mask=valid_n[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)
        return tl.sum(key * q[None, :], axis=1) * scale, key


    @triton.jit
    def _multi_block_forward_kernel(
        query,
        key,
        value,
        output,
        thresholds,
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
        stride_tb: tl.constexpr,
        stride_th: tl.constexpr,
        stride_tl: tl.constexpr,
        query_length: tl.constexpr,
        key_length: tl.constexpr,
        head_dim: tl.constexpr,
        scale: tl.constexpr,
        is_causal: tl.constexpr,
        mode: tl.constexpr,
        BISECTION_STEPS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_l = tl.program_id(2)
        offs_d = tl.arange(0, BLOCK_D)
        valid_d = offs_d < head_dim
        q_ptr = query + pid_b * stride_qb + pid_h * stride_qh + pid_l * stride_ql
        k_ptr = key + pid_b * stride_kb + pid_h * stride_kh
        v_ptr = value + pid_b * stride_vb + pid_h * stride_vh
        q = tl.load(q_ptr + offs_d * stride_qd, mask=valid_d, other=0.0).to(tl.float32)
        causal_limit = pid_l + key_length - query_length

        lower = 3.4028234663852886e38
        upper = -3.4028234663852886e38
        for start_n in tl.range(0, key_length, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid_n = offs_n < key_length
            if is_causal:
                valid_n = valid_n & (offs_n <= causal_limit)
            scores, ignored_keys = _score_tile(
                q, k_ptr, offs_n, offs_d, valid_n, valid_d, stride_ks, stride_kd, scale
            )
            z = scores if mode == 0 else scores * 0.5
            lower = tl.minimum(lower, tl.min(tl.where(valid_n, z, 3.4028234663852886e38), axis=0))
            upper = tl.maximum(upper, tl.max(tl.where(valid_n, z, -3.4028234663852886e38), axis=0))

        # Both transforms have a unique threshold satisfying sum(phi(z-tau))=1.
        lower -= 1.0
        for bisection_index in tl.static_range(BISECTION_STEPS):
            midpoint = (lower + upper) * 0.5
            mass = tl.full((), 0.0, tl.float32)
            for start_n in tl.range(0, key_length, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                valid_n = offs_n < key_length
                if is_causal:
                    valid_n = valid_n & (offs_n <= causal_limit)
                scores, ignored_keys = _score_tile(
                    q, k_ptr, offs_n, offs_d, valid_n, valid_d, stride_ks, stride_kd, scale
                )
                z = scores if mode == 0 else scores * 0.5
                positive = tl.maximum(z - midpoint, 0.0)
                probabilities = positive if mode == 0 else positive * positive
                mass += tl.sum(tl.where(valid_n, probabilities, 0.0), axis=0)
            if mass > 1.0:
                lower = midpoint
            else:
                upper = midpoint
        threshold = (lower + upper) * 0.5
        tl.store(
            thresholds + pid_b * stride_tb + pid_h * stride_th + pid_l * stride_tl,
            threshold,
        )

        accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for start_n in tl.range(0, key_length, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid_n = offs_n < key_length
            if is_causal:
                valid_n = valid_n & (offs_n <= causal_limit)
            scores, ignored_keys = _score_tile(
                q, k_ptr, offs_n, offs_d, valid_n, valid_d, stride_ks, stride_kd, scale
            )
            z = scores if mode == 0 else scores * 0.5
            positive = tl.maximum(z - threshold, 0.0)
            probabilities = positive if mode == 0 else positive * positive
            probabilities = tl.where(valid_n, probabilities, 0.0)
            values = tl.load(
                v_ptr + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                mask=valid_n[:, None] & valid_d[None, :],
                other=0.0,
            ).to(tl.float32)
            accumulator += tl.sum(probabilities[:, None] * values, axis=0)
        tl.store(
            output
            + pid_b * stride_ob
            + pid_h * stride_oh
            + pid_l * stride_ol
            + offs_d * stride_od,
            accumulator,
            mask=valid_d,
        )


    @triton.jit
    def _multi_block_backward_kernel(
        query,
        key,
        value,
        grad_output,
        thresholds,
        grad_query,
        grad_key,
        grad_value,
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
        stride_dob: tl.constexpr,
        stride_doh: tl.constexpr,
        stride_dol: tl.constexpr,
        stride_dod: tl.constexpr,
        stride_tb: tl.constexpr,
        stride_th: tl.constexpr,
        stride_tl: tl.constexpr,
        stride_dqb: tl.constexpr,
        stride_dqh: tl.constexpr,
        stride_dql: tl.constexpr,
        stride_dqd: tl.constexpr,
        stride_dkb: tl.constexpr,
        stride_dkh: tl.constexpr,
        stride_dks: tl.constexpr,
        stride_dkd: tl.constexpr,
        stride_dvb: tl.constexpr,
        stride_dvh: tl.constexpr,
        stride_dvs: tl.constexpr,
        stride_dvd: tl.constexpr,
        query_length: tl.constexpr,
        key_length: tl.constexpr,
        head_dim: tl.constexpr,
        scale: tl.constexpr,
        is_causal: tl.constexpr,
        mode: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_l = tl.program_id(2)
        offs_d = tl.arange(0, BLOCK_D)
        valid_d = offs_d < head_dim
        q_ptr = query + pid_b * stride_qb + pid_h * stride_qh + pid_l * stride_ql
        k_ptr = key + pid_b * stride_kb + pid_h * stride_kh
        v_ptr = value + pid_b * stride_vb + pid_h * stride_vh
        q = tl.load(q_ptr + offs_d * stride_qd, mask=valid_d, other=0.0).to(tl.float32)
        do = tl.load(
            grad_output
            + pid_b * stride_dob
            + pid_h * stride_doh
            + pid_l * stride_dol
            + offs_d * stride_dod,
            mask=valid_d,
            other=0.0,
        ).to(tl.float32)
        threshold = tl.load(
            thresholds + pid_b * stride_tb + pid_h * stride_th + pid_l * stride_tl
        )
        causal_limit = pid_l + key_length - query_length

        correction_numerator = tl.full((), 0.0, tl.float32)
        correction_denominator = tl.full((), 0.0, tl.float32)
        for start_n in tl.range(0, key_length, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid_n = offs_n < key_length
            if is_causal:
                valid_n = valid_n & (offs_n <= causal_limit)
            scores, ignored_keys = _score_tile(
                q, k_ptr, offs_n, offs_d, valid_n, valid_d, stride_ks, stride_kd, scale
            )
            z = scores if mode == 0 else scores * 0.5
            positive = tl.maximum(z - threshold, 0.0)
            values = tl.load(
                v_ptr + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                mask=valid_n[:, None] & valid_d[None, :],
                other=0.0,
            ).to(tl.float32)
            grad_probabilities = tl.sum(values * do[None, :], axis=1)
            inverse_hessian = (positive > 0.0).to(tl.float32) if mode == 0 else positive
            inverse_hessian = tl.where(valid_n, inverse_hessian, 0.0)
            correction_numerator += tl.sum(grad_probabilities * inverse_hessian, axis=0)
            correction_denominator += tl.sum(inverse_hessian, axis=0)
        correction = correction_numerator / correction_denominator

        dq = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for start_n in tl.range(0, key_length, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid_n = offs_n < key_length
            if is_causal:
                valid_n = valid_n & (offs_n <= causal_limit)
            scores, keys = _score_tile(
                q, k_ptr, offs_n, offs_d, valid_n, valid_d, stride_ks, stride_kd, scale
            )
            z = scores if mode == 0 else scores * 0.5
            positive = tl.maximum(z - threshold, 0.0)
            probabilities = positive if mode == 0 else positive * positive
            probabilities = tl.where(valid_n, probabilities, 0.0)
            values = tl.load(
                v_ptr + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                mask=valid_n[:, None] & valid_d[None, :],
                other=0.0,
            ).to(tl.float32)
            grad_probabilities = tl.sum(values * do[None, :], axis=1)
            inverse_hessian = (positive > 0.0).to(tl.float32) if mode == 0 else positive
            grad_scores = inverse_hessian * (grad_probabilities - correction)
            grad_scores = tl.where(valid_n, grad_scores, 0.0)
            dq += tl.sum(grad_scores[:, None] * keys, axis=0) * scale
            tl.atomic_add(
                grad_key
                + pid_b * stride_dkb
                + pid_h * stride_dkh
                + offs_n[:, None] * stride_dks
                + offs_d[None, :] * stride_dkd,
                grad_scores[:, None] * q[None, :] * scale,
                mask=valid_n[:, None] & valid_d[None, :],
            )
            tl.atomic_add(
                grad_value
                + pid_b * stride_dvb
                + pid_h * stride_dvh
                + offs_n[:, None] * stride_dvs
                + offs_d[None, :] * stride_dvd,
                probabilities[:, None] * do[None, :],
                mask=valid_n[:, None] & valid_d[None, :],
            )
        tl.store(
            grad_query
            + pid_b * stride_dqb
            + pid_h * stride_dqh
            + pid_l * stride_dql
            + offs_d * stride_dqd,
            dq,
            mask=valid_d,
        )


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


class _MultiBlockSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, scale, is_causal, mode, block_n, bisection_steps):
        batch, heads, query_length, head_dim = query.shape
        key_length = key.shape[-2]
        block_d = _next_power_of_2(head_dim)
        output = torch.empty_like(query)
        thresholds = torch.empty(
            batch,
            heads,
            query_length,
            dtype=torch.float32,
            device=query.device,
        )
        grid = (batch, heads, query_length)
        _multi_block_forward_kernel[grid](
            query,
            key,
            value,
            output,
            thresholds,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *output.stride(),
            *thresholds.stride(),
            query_length,
            key_length,
            head_dim,
            scale,
            is_causal,
            mode,
            BISECTION_STEPS=bisection_steps,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
        )
        ctx.save_for_backward(query, key, value, thresholds)
        ctx.scale = scale
        ctx.is_causal = is_causal
        ctx.mode = mode
        ctx.block_n = block_n
        return output

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, thresholds = ctx.saved_tensors
        batch, heads, query_length, head_dim = query.shape
        key_length = key.shape[-2]
        block_d = _next_power_of_2(head_dim)
        grad_query = torch.empty_like(query, dtype=torch.float32)
        grad_key = torch.zeros_like(key, dtype=torch.float32)
        grad_value = torch.zeros_like(value, dtype=torch.float32)
        grid = (batch, heads, query_length)
        _multi_block_backward_kernel[grid](
            query,
            key,
            value,
            grad_output,
            thresholds,
            grad_query,
            grad_key,
            grad_value,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *grad_output.stride(),
            *thresholds.stride(),
            *grad_query.stride(),
            *grad_key.stride(),
            *grad_value.stride(),
            query_length,
            key_length,
            head_dim,
            ctx.scale,
            ctx.is_causal,
            ctx.mode,
            BLOCK_N=ctx.block_n,
            BLOCK_D=block_d,
        )
        return grad_query, grad_key, grad_value, None, None, None, None, None


def _multi_block_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float],
    is_causal: bool,
    mode: int,
    block_n: int,
    bisection_steps: int,
) -> torch.Tensor:
    _require_triton()
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("Hopper sparse attention requires CUDA tensors.")
    if query.ndim != 4 or query.shape[:2] != key.shape[:2] or key.shape != value.shape:
        raise ValueError("Expected compatible Q/K/V tensors shaped [batch, heads, sequence, dim].")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("Q/K/V head dimensions must match.")
    if _next_power_of_2(query.shape[-1]) > 256:
        raise ValueError("head_dim larger than 256 is not supported.")
    actual_scale = query.shape[-1] ** -0.5 if scale is None else scale
    return _MultiBlockSparseAttention.apply(
        query,
        key,
        value,
        actual_scale,
        is_causal,
        mode,
        block_n,
        bisection_steps,
    )


def hopper_sparsemax_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
    is_causal: bool = False,
    block_n: int = 128,
    bisection_steps: int = 24,
) -> torch.Tensor:
    return _multi_block_attention(
        query, key, value, scale, is_causal, 0, block_n, bisection_steps
    )


def hopper_entmax15_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
    is_causal: bool = False,
    block_n: int = 128,
    bisection_steps: int = 24,
) -> torch.Tensor:
    return _multi_block_attention(
        query, key, value, scale, is_causal, 1, block_n, bisection_steps
    )
