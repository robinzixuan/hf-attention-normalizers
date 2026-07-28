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

        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        valid_n = offs_n < key_length

        q = tl.load(
            query + pid_b * stride_qb + pid_h * stride_qh + pid_l * stride_ql + offs_d * stride_qd,
            mask=offs_d < head_dim,
            other=0.0,
        )
        k = tl.load(
            key
            + pid_b * stride_kb
            + pid_h * stride_kh
            + offs_n[:, None] * stride_ks
            + offs_d[None, :] * stride_kd,
            mask=valid_n[:, None] & (offs_d[None, :] < head_dim),
            other=0.0,
        )

        scores = tl.sum(k * q[None, :], axis=1) * scale
        if is_causal:
            causal_limit = pid_l + key_length - query_length
            valid_n = valid_n & (offs_n <= causal_limit)
            valid_count = tl.maximum(causal_limit + 1, 0)
        else:
            valid_count = key_length
        scores = tl.where(valid_n, scores, -3.4028234663852886e38)

        if mode == 0:
            z = scores - tl.max(scores, axis=0)
            zs = tl.sort(z, descending=True)
            rho = tl.arange(1, BLOCK_N + 1)
            sorted_valid = rho <= valid_count
            zs_for_stats = tl.where(sorted_valid, zs, 0.0)
            cssv = tl.cumsum(zs_for_stats, axis=0)
            support = sorted_valid & ((1.0 + rho.to(tl.float32) * zs) > cssv)
            support_f = support.to(tl.float32)
            support_size = tl.sum(support_f, axis=0)
            support_sum = tl.sum(tl.where(support, zs_for_stats, 0.0), axis=0)
            tau = (support_sum - 1.0) / support_size
            probs = tl.maximum(z - tau, 0.0)
        else:
            z = (scores - tl.max(scores, axis=0)) * 0.5
            zs = tl.sort(z, descending=True)
            rho = tl.arange(1, BLOCK_N + 1).to(tl.float32)
            sorted_valid = rho <= valid_count
            zs_for_stats = tl.where(sorted_valid, zs, 0.0)
            cssv = tl.cumsum(zs_for_stats, axis=0)
            cssv_sq = tl.cumsum(zs_for_stats * zs_for_stats, axis=0)
            mean = cssv / rho
            mean_sq = cssv_sq / rho
            ss = rho * (mean_sq - mean * mean)
            delta = tl.maximum((1.0 - ss) / rho, 0.0)
            tau_candidates = mean - tl.sqrt(delta)
            support = sorted_valid & (tau_candidates <= zs)
            support_f = support.to(tl.float32)
            support_size = tl.sum(support_f, axis=0)
            support_sum = tl.sum(tl.where(support, zs_for_stats, 0.0), axis=0)
            support_sum_sq = tl.sum(tl.where(support, zs_for_stats * zs_for_stats, 0.0), axis=0)
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
            mask=valid_n[:, None] & (offs_d[None, :] < head_dim),
            other=0.0,
        )
        out = tl.sum(probs[:, None] * v, axis=0)
        tl.store(
            output + pid_b * stride_ob + pid_h * stride_oh + pid_l * stride_ol + offs_d * stride_od,
            out,
            mask=offs_d < head_dim,
        )

    @triton.jit
    def _sparse_entmax_attention_backward_kernel(
        query,
        key,
        value,
        grad_output,
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

        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        valid_n = offs_n < key_length
        valid_d = offs_d < head_dim

        q = tl.load(
            query + pid_b * stride_qb + pid_h * stride_qh + pid_l * stride_ql + offs_d * stride_qd,
            mask=valid_d,
            other=0.0,
        ).to(tl.float32)
        do = tl.load(
            grad_output
            + pid_b * stride_dob
            + pid_h * stride_doh
            + pid_l * stride_dol
            + offs_d * stride_dod,
            mask=valid_d,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            key
            + pid_b * stride_kb
            + pid_h * stride_kh
            + offs_n[:, None] * stride_ks
            + offs_d[None, :] * stride_kd,
            mask=valid_n[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            value
            + pid_b * stride_vb
            + pid_h * stride_vh
            + offs_n[:, None] * stride_vs
            + offs_d[None, :] * stride_vd,
            mask=valid_n[:, None] & valid_d[None, :],
            other=0.0,
        ).to(tl.float32)

        scores = tl.sum(k * q[None, :], axis=1) * scale
        if is_causal:
            causal_limit = pid_l + key_length - query_length
            valid_n = valid_n & (offs_n <= causal_limit)
            valid_count = tl.maximum(causal_limit + 1, 0)
        else:
            valid_count = key_length
        scores = tl.where(valid_n, scores, -3.4028234663852886e38)

        if mode == 0:
            z = scores - tl.max(scores, axis=0)
            zs = tl.sort(z, descending=True)
            rho = tl.arange(1, BLOCK_N + 1)
            sorted_valid = rho <= valid_count
            zs_for_stats = tl.where(sorted_valid, zs, 0.0)
            cssv = tl.cumsum(zs_for_stats, axis=0)
            support = sorted_valid & ((1.0 + rho.to(tl.float32) * zs) > cssv)
            support_size = tl.sum(support.to(tl.float32), axis=0)
            support_sum = tl.sum(tl.where(support, zs_for_stats, 0.0), axis=0)
            tau = (support_sum - 1.0) / support_size
            probs = tl.maximum(z - tau, 0.0)
        else:
            z = (scores - tl.max(scores, axis=0)) * 0.5
            zs = tl.sort(z, descending=True)
            rho = tl.arange(1, BLOCK_N + 1).to(tl.float32)
            sorted_valid = rho <= valid_count
            zs_for_stats = tl.where(sorted_valid, zs, 0.0)
            cssv = tl.cumsum(zs_for_stats, axis=0)
            cssv_sq = tl.cumsum(zs_for_stats * zs_for_stats, axis=0)
            mean = cssv / rho
            mean_sq = cssv_sq / rho
            ss = rho * (mean_sq - mean * mean)
            delta = tl.maximum((1.0 - ss) / rho, 0.0)
            tau_candidates = mean - tl.sqrt(delta)
            support = sorted_valid & (tau_candidates <= zs)
            support_size = tl.sum(support.to(tl.float32), axis=0)
            support_sum = tl.sum(tl.where(support, zs_for_stats, 0.0), axis=0)
            support_sum_sq = tl.sum(tl.where(support, zs_for_stats * zs_for_stats, 0.0), axis=0)
            mean_star = support_sum / support_size
            mean_sq_star = support_sum_sq / support_size
            ss_star = support_size * (mean_sq_star - mean_star * mean_star)
            tau = mean_star - tl.sqrt(tl.maximum((1.0 - ss_star) / support_size, 0.0))
            probs = tl.maximum(z - tau, 0.0)
            probs = probs * probs

        probs = tl.where(valid_n, probs, 0.0)
        grad_probs = tl.sum(v * do[None, :], axis=1)
        if mode == 0:
            probability_support = probs > 0.0
            support_size = tl.sum(probability_support.to(tl.float32), axis=0)
            correction = tl.sum(tl.where(probability_support, grad_probs, 0.0), axis=0) / support_size
            grad_scores = tl.where(probability_support, grad_probs - correction, 0.0)
        else:
            inverse_hessian = tl.sqrt(probs)
            correction = tl.sum(grad_probs * inverse_hessian, axis=0) / tl.sum(inverse_hessian, axis=0)
            grad_scores = inverse_hessian * (grad_probs - correction)

        grad_scores = tl.where(valid_n, grad_scores, 0.0)
        dq = tl.sum(grad_scores[:, None] * k, axis=0) * scale
        tl.store(
            grad_query
            + pid_b * stride_dqb
            + pid_h * stride_dqh
            + pid_l * stride_dql
            + offs_d * stride_dqd,
            dq,
            mask=valid_d,
        )
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
            probs[:, None] * do[None, :],
            mask=valid_n[:, None] & valid_d[None, :],
        )


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _launch_triton_sparse_entmax_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
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
        head_dim,
        scale,
        is_causal,
        mode,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )
    return output


def _launch_triton_sparse_entmax_attention_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_output: torch.Tensor,
    scale: float,
    is_causal: bool,
    mode: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, query_length, head_dim = query.shape
    key_length = key.shape[-2]
    block_n = _next_power_of_2(key_length)
    block_d = _next_power_of_2(head_dim)
    grad_query = torch.empty_like(query, dtype=torch.float32)
    grad_key = torch.zeros_like(key, dtype=torch.float32)
    grad_value = torch.zeros_like(value, dtype=torch.float32)
    grid = (batch, heads, query_length)
    _sparse_entmax_attention_backward_kernel[grid](
        query,
        key,
        value,
        grad_output,
        grad_query,
        grad_key,
        grad_value,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *grad_output.stride(),
        *grad_query.stride(),
        *grad_key.stride(),
        *grad_value.stride(),
        query_length,
        key_length,
        head_dim,
        scale,
        is_causal,
        mode,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )
    return grad_query, grad_key, grad_value


class _SparseEntmaxAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        is_causal: bool,
        mode: int,
        max_block_n: int,
    ) -> torch.Tensor:
        output = _launch_triton_sparse_entmax_attention(
            query,
            key,
            value,
            scale,
            is_causal,
            mode,
            max_block_n,
        )
        ctx.save_for_backward(query, key, value)
        ctx.scale = scale
        ctx.is_causal = is_causal
        ctx.mode = mode
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        query, key, value = ctx.saved_tensors
        grad_query, grad_key, grad_value = _launch_triton_sparse_entmax_attention_backward(
            query,
            key,
            value,
            grad_output,
            ctx.scale,
            ctx.is_causal,
            ctx.mode,
        )
        return grad_query, grad_key, grad_value, None, None, None, None


def _triton_sparse_entmax_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float],
    is_causal: bool,
    mode: int,
    max_block_n: int,
) -> torch.Tensor:
    actual_scale = scale if scale is not None else query.shape[-1] ** -0.5
    return _SparseEntmaxAttention.apply(
        query,
        key,
        value,
        actual_scale,
        is_causal,
        mode,
        max_block_n,
    )


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
