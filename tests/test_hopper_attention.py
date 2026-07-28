import unittest

import torch

from hf_attention_normalizers.kernels import (
    hopper_entmax15_attention,
    hopper_softmax1_attention,
    hopper_sparsemax_attention,
)
from hf_attention_normalizers.vutils import entmax15, softmax_1, sparsemax


@unittest.skipUnless(torch.cuda.is_available(), "Hopper attention tests require CUDA")
class HopperSparseAttentionTest(unittest.TestCase):
    def test_forward_and_backward_match_reference(self):
        for kernel, normalizer in (
            (hopper_softmax1_attention, softmax_1),
            (hopper_sparsemax_attention, sparsemax),
            (hopper_entmax15_attention, entmax15),
        ):
            with self.subTest(kernel=kernel.__name__):
                torch.manual_seed(31)
                shape = (1, 2, 129, 32)
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
                output = kernel(query, key, value, is_causal=True, block_n=64)
                scores = torch.matmul(ref_query, ref_key.transpose(-2, -1)) * 32**-0.5
                mask = torch.ones(129, 129, device="cuda", dtype=torch.bool).tril()
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
                reference = torch.matmul(normalizer(scores, dim=-1), ref_value)
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

    def test_sequence_longer_than_old_single_block_limit(self):
        for kernel in (
            hopper_softmax1_attention,
            hopper_sparsemax_attention,
            hopper_entmax15_attention,
        ):
            with self.subTest(kernel=kernel.__name__):
                query = torch.randn(
                    1, 1, 2, 64, device="cuda", dtype=torch.float16, requires_grad=True
                )
                key = torch.randn(
                    1, 1, 8192, 64, device="cuda", dtype=torch.float16, requires_grad=True
                )
                value = torch.randn_like(key, requires_grad=True)
                output = kernel(query, key, value, is_causal=True)
                output.float().square().mean().backward()
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.isfinite(query.grad).all())
                self.assertTrue(torch.isfinite(key.grad).all())
                self.assertTrue(torch.isfinite(value.grad).all())

    def test_softmax1_huggingface_backend(self):
        from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM

        from hf_attention_normalizers import set_softmax_attention_backend

        config = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=64,
        )
        model = Qwen3ForCausalLM(config).to(
            device="cuda",
            dtype=torch.float16,
        ).train()
        set_softmax_attention_backend(
            model,
            base_backend="flash_attention_3",
            softmax_fn="softmax1",
            mode="strict",
        )
        logits = model(
            torch.randint(0, config.vocab_size, (2, 16), device="cuda")
        ).logits
        logits.float().square().mean().backward()
        self.assertEqual(
            model.config._attn_implementation,
            "softmax1_flash_attention_3",
        )
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
