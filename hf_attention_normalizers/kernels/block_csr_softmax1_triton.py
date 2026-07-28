"""Full-block CSR Softmax1 attention forward kernel for Flex BlockMask."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _reference(query, key, value, num_blocks, block_indices, q_block, kv_block, scale):
    """Differentiable CSR reference used by the transitional backward path."""
    output = torch.empty_like(query)
    for batch in range(query.shape[0]):
        for head in range(query.shape[1]):
            for q_block_index in range(num_blocks.shape[2]):
                q_start = q_block_index * q_block
                q_end = min(q_start + q_block, query.shape[-2])
                if q_start == q_end:
                    continue
                count = int(num_blocks[batch, head, q_block_index].item())
                if count == 0:
                    output[batch, head, q_start:q_end] = 0
                    continue
                blocks = block_indices[batch, head, q_block_index, :count]
                offsets = torch.arange(kv_block, device=query.device)
                positions = (blocks[:, None] * kv_block + offsets).reshape(-1)
                positions = positions[positions < key.shape[-2]]
                scores = query[batch, head, q_start:q_end] @ key[batch, head, positions].transpose(-2, -1)
                probabilities = torch.softmax(torch.cat((scores * scale, torch.zeros_like(scores[..., :1])), dim=-1), dim=-1)[..., :-1]
                output[batch, head, q_start:q_end] = probabilities @ value[batch, head, positions]
    return output


if triton is not None:

    @triton.jit
    def _forward(
        query, key, value, num_blocks, block_indices, output,
        sqb: tl.constexpr, sqh: tl.constexpr, sql: tl.constexpr, sqd: tl.constexpr,
        skb: tl.constexpr, skh: tl.constexpr, skl: tl.constexpr, skd: tl.constexpr,
        svb: tl.constexpr, svh: tl.constexpr, svl: tl.constexpr, svd: tl.constexpr,
        snb: tl.constexpr, snh: tl.constexpr, snq: tl.constexpr,
        sib: tl.constexpr, sih: tl.constexpr, siq: tl.constexpr, sis: tl.constexpr,
        sob: tl.constexpr, soh: tl.constexpr, sol: tl.constexpr, sod: tl.constexpr,
        query_length: tl.constexpr, key_length: tl.constexpr, head_dim: tl.constexpr,
        scale: tl.constexpr, Q_BLOCK: tl.constexpr, KV_BLOCK: tl.constexpr,
        MAX_BLOCKS: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        batch, head, query_index = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        q_block_index = query_index // Q_BLOCK
        count = tl.load(num_blocks + batch * snb + head * snh + q_block_index * snq)
        d = tl.arange(0, BLOCK_D)
        valid_d = d < head_dim
        q = tl.load(query + batch * sqb + head * sqh + query_index * sql + d * sqd, mask=valid_d, other=0.).to(tl.float32)
        running_max, denominator = 0., 1.
        acc = tl.zeros((BLOCK_D,), tl.float32)
        for slot in tl.static_range(MAX_BLOCKS):
            block = tl.load(block_indices + batch * sib + head * sih + q_block_index * siq + slot * sis, mask=slot < count, other=0)
            n = block * KV_BLOCK + tl.arange(0, KV_BLOCK)
            valid_n = (slot < count) & (n < key_length)
            keys = tl.load(key + batch * skb + head * skh + n[:, None] * skl + d[None, :] * skd, mask=valid_n[:, None] & valid_d[None, :], other=0.).to(tl.float32)
            scores = tl.sum(keys * q[None, :], axis=1) * scale
            block_max = tl.max(tl.where(valid_n, scores, -3.4028234663852886e38), axis=0)
            new_max = tl.maximum(running_max, block_max)
            old = tl.exp(running_max - new_max)
            weights = tl.where(valid_n, tl.exp(scores - new_max), 0.)
            vals = tl.load(value + batch * svb + head * svh + n[:, None] * svl + d[None, :] * svd, mask=valid_n[:, None] & valid_d[None, :], other=0.).to(tl.float32)
            acc = acc * old + tl.sum(weights[:, None] * vals, axis=0)
            denominator = denominator * old + tl.sum(weights, axis=0)
            running_max = new_max
        tl.store(output + batch * sob + head * soh + query_index * sol + d * sod, acc / denominator, mask=valid_d)


class _BlockCSRSoftmax1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, num_blocks, block_indices, q_block, kv_block, scale):
        if triton is None:
            raise ImportError("Block CSR Softmax1 requires Triton.")
        output = torch.empty_like(query)
        block_d = _next_power_of_2(query.shape[-1])
        _forward[(query.shape[0], query.shape[1], query.shape[-2])](
            query, key, value, num_blocks, block_indices, output,
            *query.stride(), *key.stride(), *value.stride(), *num_blocks.stride(), *block_indices.stride(), *output.stride(),
            query.shape[-2], key.shape[-2], query.shape[-1], scale, q_block, kv_block,
            MAX_BLOCKS=block_indices.shape[-1], BLOCK_D=block_d,
        )
        ctx.save_for_backward(query, key, value, num_blocks, block_indices)
        ctx.args = (q_block, kv_block, scale)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, num_blocks, block_indices = ctx.saved_tensors
        q_block, kv_block, scale = ctx.args
        with torch.enable_grad():
            q = query.detach().requires_grad_(True)
            k = key.detach().requires_grad_(True)
            v = value.detach().requires_grad_(True)
            output = _reference(q, k, v, num_blocks, block_indices, q_block, kv_block, scale)
            dq, dk, dv = torch.autograd.grad(output, (q, k, v), grad_output)
        return dq, dk, dv, None, None, None, None, None


def block_csr_softmax1_attention(query, key, value, num_blocks, block_indices, q_block, kv_block, scale=None):
    actual_scale = query.shape[-1] ** -0.5 if scale is None else scale
    return _BlockCSRSoftmax1.apply(query, key, value, num_blocks, block_indices, q_block, kv_block, actual_scale)
