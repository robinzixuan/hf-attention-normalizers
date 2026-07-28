import unittest

import torch

from hf_attention_normalizers.kernels.sparse_entmax_triton import (
    triton_entmax15_attention,
    triton_sparsemax_attention,
)
from hf_attention_normalizers.vutils import entmax15, sparsemax


@unittest.skipUnless(torch.cuda.is_available(), "Triton attention tests require CUDA")
class SparseEntmaxTritonTest(unittest.TestCase):
    def _check_forward_and_backward(self, kernel, normalizer, is_causal):
        torch.manual_seed(7)
        shape = (1, 2, 17, 48)
        scale = shape[-1] ** -0.5
        inputs = [
            torch.randn(shape, device="cuda", dtype=torch.float16)
            for _ in range(3)
        ]
        query, key, value = [
            tensor.detach().clone().requires_grad_(True) for tensor in inputs
        ]
        ref_query, ref_key, ref_value = [
            tensor.float().detach().requires_grad_(True) for tensor in inputs
        ]

        output = kernel(query, key, value, is_causal=is_causal)
        scores = torch.matmul(ref_query, ref_key.transpose(-2, -1)) * scale
        if is_causal:
            mask = torch.ones(17, 17, device="cuda", dtype=torch.bool).tril()
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        reference = torch.matmul(normalizer(scores, dim=-1), ref_value)

        upstream = torch.randn_like(reference)
        output.backward(upstream.to(output.dtype))
        reference.backward(upstream)

        self.assertTrue(torch.isfinite(output).all())
        self.assertLess((output.float() - reference).abs().mean().item(), 0.01)
        for actual, expected in zip(
            (query.grad, key.grad, value.grad),
            (ref_query.grad, ref_key.grad, ref_value.grad),
        ):
            self.assertTrue(torch.isfinite(actual).all())
            self.assertLess(
                (actual.float() - expected).abs().mean().item(),
                0.001,
            )

    def test_sparsemax_forward_and_backward(self):
        for is_causal in (False, True):
            with self.subTest(is_causal=is_causal):
                self._check_forward_and_backward(
                    triton_sparsemax_attention,
                    sparsemax,
                    is_causal,
                )

    def test_entmax15_forward_and_backward(self):
        for is_causal in (False, True):
            with self.subTest(is_causal=is_causal):
                self._check_forward_and_backward(
                    triton_entmax15_attention,
                    entmax15,
                    is_causal,
                )


if __name__ == "__main__":
    unittest.main()
