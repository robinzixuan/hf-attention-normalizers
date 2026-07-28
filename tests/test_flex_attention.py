import unittest
from unittest.mock import patch

import torch

from hf_attention_normalizers import set_softmax_attention_backend
from hf_attention_normalizers.backends import (
    flex_attention_normalizer_forward,
    resolve_softmax_fn,
)


@unittest.skipUnless(torch.cuda.is_available(), "FlexAttention tests require CUDA")
class FlexAttentionNormalizerTest(unittest.TestCase):
    def test_pure_causal_block_mask_uses_native_hopper_path(self):
        from torch import nn
        from transformers.masking_utils import causal_mask_function, flex_attention_mask

        class Module(nn.Module):
            is_causal = True
            num_key_value_groups = 1

        block_mask = flex_attention_mask(
            batch_size=1,
            cache_position=torch.arange(16, device="cuda"),
            kv_length=16,
            mask_function=causal_mask_function,
            attention_mask=None,
        )
        query = torch.randn(1, 2, 16, 32, device="cuda", dtype=torch.float16)
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        for normalizer in ("softmax1", "sparsemax", "entmax15"):
            with self.subTest(normalizer=normalizer):
                # A pure causal mask must not reach the dense compatibility helper.
                with patch(
                    "hf_attention_normalizers.backends._dense_mask_from_flex_block_mask",
                    side_effect=AssertionError("dense BlockMask fallback was used"),
                ):
                    output, _ = flex_attention_normalizer_forward(
                        Module(),
                        query,
                        key,
                        value,
                        block_mask,
                        softmax_fn=resolve_softmax_fn(normalizer),
                        scaling=32**-0.5,
                    )
                self.assertEqual(output.shape, (1, 16, 2, 32))
                self.assertTrue(torch.isfinite(output).all())

    def test_qwen3_forward_and_backward(self):
        from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM

        for normalizer in ("softmax1", "sparsemax", "entmax15"):
            with self.subTest(normalizer=normalizer):
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
                    base_backend="flex_attention",
                    softmax_fn=normalizer,
                    mode="strict",
                )
                logits = model(
                    torch.randint(0, config.vocab_size, (2, 16), device="cuda")
                ).logits
                logits.float().square().mean().backward()

                self.assertEqual(
                    model.config._attn_implementation,
                    f"{normalizer}_flex_attention",
                )
                self.assertTrue(torch.isfinite(logits).all())
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                self.assertTrue(gradients)
                self.assertTrue(
                    all(torch.isfinite(gradient).all() for gradient in gradients)
                )


if __name__ == "__main__":
    unittest.main()
