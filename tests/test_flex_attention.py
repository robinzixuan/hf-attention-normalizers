import unittest

import torch

from hf_attention_normalizers import set_softmax_attention_backend


@unittest.skipUnless(torch.cuda.is_available(), "FlexAttention tests require CUDA")
class FlexAttentionNormalizerTest(unittest.TestCase):
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
