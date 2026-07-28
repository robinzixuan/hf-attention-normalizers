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
        raise ImportError("Softmax1 Hopper attention requires Triton.")


if triton is not None:

    @triton.jit
    def _softmax1_forward_kernel(
        query,
        key,
        value,
        output,
        row_max,
        row_denominator,
        seed,
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
        stride_mb: tl.constexpr,
        stride_mh: tl.constexpr,
        stride_ml: tl.constexpr,
        stride_lb: tl.constexpr,
        stride_lh: tl.constexpr,
        stride_ll: tl.constexpr,
        query_length: tl.constexpr,
        key_length: tl.constexpr,
        head_dim: tl.constexpr,
        scale: tl.constexpr,
        is_causal: tl.constexpr,
        window_left: tl.constexpr,
        num_heads: tl.constexpr,
        dropout_p: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_l = tl.program_id(2)
        offs_d = tl.arange(0, BLOCK_D)
        valid_d = offs_d < head_dim
        q = tl.load(
            query
            + pid_b * stride_qb
            + pid_h * stride_qh
            + pid_l * stride_ql
            + offs_d * stride_qd,
            mask=valid_d,
            other=0.0,
        ).to(tl.float32)
        key_base = key + pid_b * stride_kb + pid_h * stride_kh
        value_base = value + pid_b * stride_vb + pid_h * stride_vh
        causal_limit = pid_l + key_length - query_length

        # The +1 in Softmax1 is a virtual logit=0, value=0 attention sink.
        running_max = 0.0
        denominator = 1.0
        accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for start_n in tl.range(0, key_length, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid_n = offs_n < key_length
            if is_causal:
                valid_n = valid_n & (offs_n <= causal_limit)
            if window_left > 0:
                valid_n = valid_n & (offs_n > causal_limit - window_left)
            keys = tl.load(
                key_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                mask=valid_n[:, None] & valid_d[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(keys * q[None, :], axis=1) * scale
            block_max = tl.max(
                tl.where(valid_n, scores, -3.4028234663852886e38),
                axis=0,
            )
            new_max = tl.maximum(running_max, block_max)
            previous_scale = tl.exp(running_max - new_max)
            exponentials = tl.exp(scores - new_max)
            exponentials = tl.where(valid_n, exponentials, 0.0)
            rng_offsets = (
                ((pid_b * num_heads + pid_h) * query_length + pid_l) * key_length + offs_n
            )
            keep = tl.rand(tl.load(seed), rng_offsets) >= dropout_p
            dropout_scale = 1.0 / (1.0 - dropout_p)
            dropped_exponentials = exponentials * keep.to(tl.float32) * dropout_scale
            values = tl.load(
                value_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                mask=valid_n[:, None] & valid_d[None, :],
                other=0.0,
            ).to(tl.float32)
            accumulator = accumulator * previous_scale + tl.sum(
                dropped_exponentials[:, None] * values,
                axis=0,
            )
            denominator = denominator * previous_scale + tl.sum(exponentials, axis=0)
            running_max = new_max

        output_base = output + pid_b * stride_ob + pid_h * stride_oh + pid_l * stride_ol
        tl.store(
            output_base + offs_d * stride_od,
            accumulator / denominator,
            mask=valid_d,
        )
        tl.store(
            row_max + pid_b * stride_mb + pid_h * stride_mh + pid_l * stride_ml,
            running_max,
        )
        tl.store(
            row_denominator + pid_b * stride_lb + pid_h * stride_lh + pid_l * stride_ll,
            denominator,
        )


    @triton.jit
    def _softmax1_backward_kernel(
        query,
        key,
        value,
        grad_output,
        row_max,
        row_denominator,
        grad_query,
        grad_key,
        grad_value,
        seed,
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
        stride_mb: tl.constexpr,
        stride_mh: tl.constexpr,
        stride_ml: tl.constexpr,
        stride_lb: tl.constexpr,
        stride_lh: tl.constexpr,
        stride_ll: tl.constexpr,
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
        window_left: tl.constexpr,
        num_heads: tl.constexpr,
        dropout_p: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_l = tl.program_id(2)
        offs_d = tl.arange(0, BLOCK_D)
        valid_d = offs_d < head_dim
        q = tl.load(
            query
            + pid_b * stride_qb
            + pid_h * stride_qh
            + pid_l * stride_ql
            + offs_d * stride_qd,
            mask=valid_d,
            other=0.0,
        ).to(tl.float32)
        grad_out = tl.load(
            grad_output
            + pid_b * stride_dob
            + pid_h * stride_doh
            + pid_l * stride_dol
            + offs_d * stride_dod,
            mask=valid_d,
            other=0.0,
        ).to(tl.float32)
        maximum = tl.load(
            row_max + pid_b * stride_mb + pid_h * stride_mh + pid_l * stride_ml
        )
        denominator = tl.load(
            row_denominator + pid_b * stride_lb + pid_h * stride_lh + pid_l * stride_ll
        )
        key_base = key + pid_b * stride_kb + pid_h * stride_kh
        value_base = value + pid_b * stride_vb + pid_h * stride_vh
        causal_limit = pid_l + key_length - query_length

        correction = tl.full((), 0.0, tl.float32)
        for start_n in tl.range(0, key_length, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid_n = offs_n < key_length
            if is_causal:
                valid_n = valid_n & (offs_n <= causal_limit)
            if window_left > 0:
                valid_n = valid_n & (offs_n > causal_limit - window_left)
            keys = tl.load(
                key_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                mask=valid_n[:, None] & valid_d[None, :],
                other=0.0,
            ).to(tl.float32)
            values = tl.load(
                value_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                mask=valid_n[:, None] & valid_d[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(keys * q[None, :], axis=1) * scale
            probabilities = tl.exp(scores - maximum) / denominator
            probabilities = tl.where(valid_n, probabilities, 0.0)
            rng_offsets = (
                ((pid_b * num_heads + pid_h) * query_length + pid_l) * key_length + offs_n
            )
            keep = tl.rand(tl.load(seed), rng_offsets) >= dropout_p
            dropout_scale = 1.0 / (1.0 - dropout_p)
            grad_probabilities = tl.sum(values * grad_out[None, :], axis=1)
            grad_probabilities *= keep.to(tl.float32) * dropout_scale
            correction += tl.sum(probabilities * grad_probabilities, axis=0)

        grad_q = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for start_n in tl.range(0, key_length, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            valid_n = offs_n < key_length
            if is_causal:
                valid_n = valid_n & (offs_n <= causal_limit)
            if window_left > 0:
                valid_n = valid_n & (offs_n > causal_limit - window_left)
            keys = tl.load(
                key_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                mask=valid_n[:, None] & valid_d[None, :],
                other=0.0,
            ).to(tl.float32)
            values = tl.load(
                value_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                mask=valid_n[:, None] & valid_d[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(keys * q[None, :], axis=1) * scale
            probabilities = tl.exp(scores - maximum) / denominator
            probabilities = tl.where(valid_n, probabilities, 0.0)
            rng_offsets = (
                ((pid_b * num_heads + pid_h) * query_length + pid_l) * key_length + offs_n
            )
            keep = tl.rand(tl.load(seed), rng_offsets) >= dropout_p
            dropout_scale = 1.0 / (1.0 - dropout_p)
            grad_probabilities = tl.sum(values * grad_out[None, :], axis=1)
            grad_probabilities *= keep.to(tl.float32) * dropout_scale
            grad_scores = probabilities * (grad_probabilities - correction)
            grad_q += tl.sum(grad_scores[:, None] * keys, axis=0) * scale
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
                probabilities[:, None] * keep[:, None].to(tl.float32) * dropout_scale * grad_out[None, :],
                mask=valid_n[:, None] & valid_d[None, :],
            )
        tl.store(
            grad_query
            + pid_b * stride_dqb
            + pid_h * stride_dqh
            + pid_l * stride_dql
            + offs_d * stride_dqd,
            grad_q,
            mask=valid_d,
        )


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


class _Softmax1HopperAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, scale, is_causal, window_left, block_n, dropout_p, seed):
        batch, heads, query_length, head_dim = query.shape
        key_length = key.shape[-2]
        block_d = _next_power_of_2(head_dim)
        output = torch.empty_like(query)
        row_max = torch.empty(
            batch, heads, query_length, device=query.device, dtype=torch.float32
        )
        row_denominator = torch.empty_like(row_max)
        grid = (batch, heads, query_length)
        _softmax1_forward_kernel[grid](
            query,
            key,
            value,
            output,
            row_max,
            row_denominator,
            seed,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *output.stride(),
            *row_max.stride(),
            *row_denominator.stride(),
            query_length,
            key_length,
            head_dim,
            scale,
            is_causal,
            window_left,
            heads,
            dropout_p,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
        )
        ctx.save_for_backward(query, key, value, row_max, row_denominator)
        ctx.scale = scale
        ctx.is_causal = is_causal
        ctx.window_left = window_left
        ctx.block_n = block_n
        ctx.dropout_p = dropout_p
        ctx.seed = seed
        return output

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, row_max, row_denominator = ctx.saved_tensors
        batch, heads, query_length, head_dim = query.shape
        key_length = key.shape[-2]
        block_d = _next_power_of_2(head_dim)
        grad_query = torch.empty_like(query, dtype=torch.float32)
        grad_key = torch.zeros_like(key, dtype=torch.float32)
        grad_value = torch.zeros_like(value, dtype=torch.float32)
        grid = (batch, heads, query_length)
        _softmax1_backward_kernel[grid](
            query,
            key,
            value,
            grad_output,
            row_max,
            row_denominator,
            grad_query,
            grad_key,
            grad_value,
            ctx.seed,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *grad_output.stride(),
            *row_max.stride(),
            *row_denominator.stride(),
            *grad_query.stride(),
            *grad_key.stride(),
            *grad_value.stride(),
            query_length,
            key_length,
            head_dim,
            ctx.scale,
            ctx.is_causal,
            ctx.window_left,
            heads,
            ctx.dropout_p,
            BLOCK_N=ctx.block_n,
            BLOCK_D=block_d,
        )
        return grad_query, grad_key, grad_value, None, None, None, None, None, None


def hopper_softmax1_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
    is_causal: bool = False,
    window_left: Optional[int] = None,
    block_n: int = 128,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    _require_triton()
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("Softmax1 Hopper attention requires CUDA tensors.")
    if query.ndim != 4 or query.shape[:2] != key.shape[:2] or key.shape != value.shape:
        raise ValueError("Expected compatible Q/K/V tensors shaped [batch, heads, sequence, dim].")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("Q/K/V head dimensions must match.")
    if _next_power_of_2(query.shape[-1]) > 256:
        raise ValueError("head_dim larger than 256 is not supported.")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0, 1).")
    actual_scale = query.shape[-1] ** -0.5 if scale is None else scale
    # This consumes one value from PyTorch's CUDA RNG. Re-seeding PyTorch before
    # a call therefore reproduces the fused mask; backward recomputes it from
    # this saved seed instead of retaining a dense [B,H,Q,K] mask.
    seed = torch.empty((), device=query.device, dtype=torch.int64)
    if dropout_p:
        seed.random_(2**31 - 1)
    else:
        seed.zero_()
    return _Softmax1HopperAttention.apply(
        query,
        key,
        value,
        actual_scale,
        is_causal,
        -1 if window_left is None else window_left,
        block_n,
        dropout_p,
        seed,
    )
