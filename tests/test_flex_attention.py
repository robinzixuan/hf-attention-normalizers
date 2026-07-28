import unittest
from unittest.mock import patch

import torch

from hf_attention_normalizers import set_softmax_attention_backend
from hf_attention_normalizers.backends import (
    flex_attention_normalizer_forward,
    resolve_softmax_fn,
    sdpa_attention_forward,
)


@unittest.skipUnless(torch.cuda.is_available(), "FlexAttention tests require CUDA")
class FlexAttentionNormalizerTest(unittest.TestCase):
    def test_full_block_csr_mask_avoids_dense_expansion(self):
        from torch import nn
        from torch.nn.attention.flex_attention import BlockMask

        class Module(nn.Module):
            is_causal = False
            num_key_value_groups = 1

        # Q block 0 attends KV block 0; Q block 1 attends KV blocks 0 and 1.
        counts = torch.tensor([[[1, 2]]], device="cuda", dtype=torch.int32)
        indices = torch.tensor([[[[0, 0], [0, 1]]]], device="cuda", dtype=torch.int32)
        block_mask = BlockMask.from_kv_blocks(
            counts, indices, BLOCK_SIZE=(4, 4), seq_lengths=(8, 8)
        )
        base = [
            torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.float16)
            for _ in range(3)
        ]
        dense_mask = torch.zeros(1, 1, 8, 8, device="cuda", dtype=torch.bool)
        dense_mask[..., :4, :4] = True
        dense_mask[..., 4:, :] = True
        for normalizer in ("softmax1", "sparsemax", "entmax15"):
            with self.subTest(normalizer=normalizer):
                fn = resolve_softmax_fn(normalizer)
                query, key, value = [x.detach().clone().requires_grad_(True) for x in base]
                ref_query, ref_key, ref_value = [x.detach().clone().requires_grad_(True) for x in base]
                with patch(
                    "hf_attention_normalizers.backends._dense_mask_from_flex_block_mask",
                    side_effect=AssertionError("dense BlockMask fallback was used"),
                ):
                    output, _ = flex_attention_normalizer_forward(
                        Module(), query, key, value, block_mask, softmax_fn=fn,
                        scaling=16**-0.5,
                    )
                reference, _ = sdpa_attention_forward(
                    Module(), ref_query, ref_key, ref_value, dense_mask,
                    softmax_fn=fn, scaling=16**-0.5, is_causal=False,
                )
                upstream = torch.randn_like(output)
                output.backward(upstream)
                reference.backward(upstream)
                torch.testing.assert_close(output, reference, atol=0.004, rtol=0.004)
                for actual, expected in zip(
                    (query.grad, key.grad, value.grad),
                    (ref_query.grad, ref_key.grad, ref_value.grad),
                ):
                    torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.008)

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

    def test_sliding_causal_block_mask_uses_native_hopper_path(self):
        from torch import nn
        from transformers.masking_utils import (
            flex_attention_mask,
            sliding_window_causal_mask_function,
        )

        class Module(nn.Module):
            is_causal = True
            num_key_value_groups = 1

        sequence_length = 16
        window = 4
        block_mask = flex_attention_mask(
            batch_size=1,
            cache_position=torch.arange(sequence_length, device="cuda"),
            kv_length=sequence_length,
            mask_function=sliding_window_causal_mask_function(window),
            attention_mask=None,
        )
        base = [
            torch.randn(1, 2, sequence_length, 32, device="cuda", dtype=torch.float16)
            for _ in range(3)
        ]
        positions = torch.arange(sequence_length, device="cuda")
        dense_mask = (
            (positions[:, None] >= positions[None, :])
            & (positions[:, None] - positions[None, :] < window)
        )[None, None]

        for normalizer in ("softmax1", "sparsemax", "entmax15"):
            with self.subTest(normalizer=normalizer):
                normalizer_fn = resolve_softmax_fn(normalizer)
                query, key, value = [
                    tensor.detach().clone().requires_grad_(True) for tensor in base
                ]
                ref_query, ref_key, ref_value = [
                    tensor.detach().clone().requires_grad_(True) for tensor in base
                ]
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
                        softmax_fn=normalizer_fn,
                        scaling=32**-0.5,
                    )
                reference, _ = sdpa_attention_forward(
                    Module(),
                    ref_query,
                    ref_key,
                    ref_value,
                    dense_mask,
                    softmax_fn=normalizer_fn,
                    scaling=32**-0.5,
                    is_causal=False,
                )
                upstream = torch.randn_like(output)
                output.backward(upstream)
                reference.backward(upstream)
                torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)
                for actual, expected in zip(
                    (query.grad, key.grad, value.grad),
                    (ref_query.grad, ref_key.grad, ref_value.grad),
                ):
                    torch.testing.assert_close(actual, expected, atol=0.012, rtol=0.012)

    def test_right_padded_causal_block_mask_uses_native_hopper_path(self):
        from torch import nn
        from transformers.masking_utils import causal_mask_function, flex_attention_mask

        class Module(nn.Module):
            is_causal = True
            num_key_value_groups = 1

        sequence_length = 16
        padding = torch.tensor(
            [[True] * 16, [True] * 11 + [False] * 5], device="cuda"
        )
        block_mask = flex_attention_mask(
            batch_size=2,
            cache_position=torch.arange(sequence_length, device="cuda"),
            kv_length=sequence_length,
            mask_function=causal_mask_function,
            attention_mask=padding,
        )
        base = [
            torch.randn(2, 2, sequence_length, 32, device="cuda", dtype=torch.float16)
            for _ in range(3)
        ]
        causal = torch.arange(sequence_length, device="cuda")[:, None] >= torch.arange(
            sequence_length, device="cuda"
        )[None, :]
        dense_mask = causal[None, None] & padding[:, None, None]

        for normalizer in ("softmax1", "sparsemax", "entmax15"):
            with self.subTest(normalizer=normalizer):
                normalizer_fn = resolve_softmax_fn(normalizer)
                query, key, value = [
                    tensor.detach().clone().requires_grad_(True) for tensor in base
                ]
                ref_query, ref_key, ref_value = [
                    tensor.detach().clone().requires_grad_(True) for tensor in base
                ]
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
                        softmax_fn=normalizer_fn,
                        scaling=32**-0.5,
                    )
                reference, _ = sdpa_attention_forward(
                    Module(),
                    ref_query,
                    ref_key,
                    ref_value,
                    dense_mask,
                    softmax_fn=normalizer_fn,
                    scaling=32**-0.5,
                    is_causal=False,
                )
                upstream = torch.randn_like(output)
                output.backward(upstream)
                reference.backward(upstream)
                torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)
                for actual, expected in zip(
                    (query.grad, key.grad, value.grad),
                    (ref_query.grad, ref_key.grad, ref_value.grad),
                ):
                    torch.testing.assert_close(actual, expected, atol=0.012, rtol=0.012)

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
                    attention_dropout=0.2,
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
