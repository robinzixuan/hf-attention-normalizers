import unittest

import torch

from hf_attention_normalizers import varlen_hopper_attention
from hf_attention_normalizers.backends import resolve_softmax_fn


def _packed_reference(query, key, value, cu_q, cu_k, normalizer):
    outputs = []
    groups = query.shape[1] // key.shape[1]
    scale = query.shape[-1] ** -0.5
    for index in range(cu_q.numel() - 1):
        q = query[cu_q[index] : cu_q[index + 1]].transpose(0, 1)
        k = key[cu_k[index] : cu_k[index + 1]].transpose(0, 1)
        v = value[cu_k[index] : cu_k[index + 1]].transpose(0, 1)
        k = k.repeat_interleave(groups, dim=0)
        v = v.repeat_interleave(groups, dim=0)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        q_length, k_length = scores.shape[-2:]
        q_positions = torch.arange(q_length, device=query.device)[:, None]
        k_positions = torch.arange(k_length, device=query.device)[None, :]
        mask = k_positions <= q_positions + k_length - q_length
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probabilities = resolve_softmax_fn(normalizer)(scores, dim=-1)
        outputs.append(torch.matmul(probabilities, v).transpose(0, 1))
    return torch.cat(outputs)


@unittest.skipUnless(torch.cuda.is_available(), "Varlen attention tests require CUDA")
class VarlenHopperAttentionTest(unittest.TestCase):
    def test_packed_gqa_forward_and_backward(self):
        cu_q = torch.tensor([0, 3, 8], device="cuda", dtype=torch.int32)
        cu_k = torch.tensor([0, 7, 12], device="cuda", dtype=torch.int32)
        for normalizer in ("softmax1", "sparsemax", "entmax15"):
            with self.subTest(normalizer=normalizer):
                torch.manual_seed(59)
                query = torch.randn(
                    8, 4, 32, device="cuda", dtype=torch.float16, requires_grad=True
                )
                key = torch.randn(
                    12, 2, 32, device="cuda", dtype=torch.float16, requires_grad=True
                )
                value = torch.randn_like(key, requires_grad=True)
                ref_query = query.detach().float().requires_grad_(True)
                ref_key = key.detach().float().requires_grad_(True)
                ref_value = value.detach().float().requires_grad_(True)

                output = varlen_hopper_attention(
                    query,
                    key,
                    value,
                    cu_q,
                    cu_k,
                    normalizer,
                    # Exercise inactive programs/tiles in the shared varlen grid.
                    max_seqlen_q=8,
                    max_seqlen_k=10,
                )
                reference = _packed_reference(
                    ref_query,
                    ref_key,
                    ref_value,
                    cu_q,
                    cu_k,
                    normalizer,
                )
                upstream = torch.randn_like(reference)
                output.backward(upstream.to(output.dtype))
                reference.backward(upstream)

                self.assertLess((output.float() - reference).abs().mean().item(), 0.001)
                for actual, expected in zip(
                    (query.grad, key.grad, value.grad),
                    (ref_query.grad, ref_key.grad, ref_value.grad),
                ):
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertLess(
                        (actual.float() - expected).abs().mean().item(),
                        0.001,
                    )


if __name__ == "__main__":
    unittest.main()
