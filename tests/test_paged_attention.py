import unittest

import torch

from hf_attention_normalizers import (
    DifferentiablePagedCache,
    paged_attention,
    paged_triton_attention,
    set_softmax_attention_backend,
)
from hf_attention_normalizers.backends import resolve_softmax_fn


def _dense_reference(
    query,
    key_cache,
    value_cache,
    block_table,
    sequence_lengths,
    block_size,
    normalizer,
):
    outputs = []
    groups = query.shape[1] // key_cache.shape[1]
    scale = query.shape[-1] ** -0.5
    for batch_index, sequence_length in enumerate(sequence_lengths.tolist()):
        logical_positions = torch.arange(sequence_length, device=query.device)
        physical = (
            block_table[batch_index, logical_positions // block_size] * block_size
            + logical_positions % block_size
        )
        key = key_cache[physical].transpose(0, 1).repeat_interleave(groups, dim=0)
        value = value_cache[physical].transpose(0, 1).repeat_interleave(groups, dim=0)
        scores = torch.matmul(query[batch_index], key.transpose(-2, -1)) * scale
        causal = torch.arange(sequence_length, device=query.device)[None, :] <= (
            torch.arange(query.shape[-2], device=query.device)[:, None]
            + sequence_length
            - query.shape[-2]
        )
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        probabilities = resolve_softmax_fn(normalizer)(scores, dim=-1)
        outputs.append(torch.matmul(probabilities, value))
    return torch.stack(outputs)


class PagedAttentionTest(unittest.TestCase):
    def test_forward_and_full_cache_backward(self):
        for normalizer in ("softmax1", "sparsemax", "entmax15"):
            with self.subTest(normalizer=normalizer):
                torch.manual_seed(11)
                query = torch.randn(2, 4, 2, 8, dtype=torch.float64, requires_grad=True)
                key_cache = torch.randn(16, 2, 8, dtype=torch.float64, requires_grad=True)
                value_cache = torch.randn(16, 2, 8, dtype=torch.float64, requires_grad=True)
                ref_query = query.detach().clone().requires_grad_(True)
                ref_key = key_cache.detach().clone().requires_grad_(True)
                ref_value = value_cache.detach().clone().requires_grad_(True)
                block_table = torch.tensor([[2, 0], [1, 3]])
                sequence_lengths = torch.tensor([5, 7])

                output = paged_attention(
                    query,
                    key_cache,
                    value_cache,
                    block_table,
                    sequence_lengths,
                    block_size=4,
                    normalizer=normalizer,
                )
                reference = _dense_reference(
                    ref_query,
                    ref_key,
                    ref_value,
                    block_table,
                    sequence_lengths,
                    4,
                    normalizer,
                )
                upstream = torch.randn_like(output)
                output.backward(upstream)
                reference.backward(upstream)

                torch.testing.assert_close(output, reference)
                torch.testing.assert_close(query.grad, ref_query.grad)
                torch.testing.assert_close(key_cache.grad, ref_key.grad)
                torch.testing.assert_close(value_cache.grad, ref_value.grad)


@unittest.skipUnless(torch.cuda.is_available(), "Paged Triton tests require CUDA")
class PagedTritonAttentionTest(unittest.TestCase):
    def test_variable_length_forward_and_cache_scatter_backward(self):
        for normalizer in ("softmax1", "sparsemax", "entmax15"):
            with self.subTest(normalizer=normalizer):
                torch.manual_seed(23)
                query = torch.randn(
                    2, 4, 2, 32, device="cuda", dtype=torch.float16, requires_grad=True
                )
                key_cache = torch.randn(
                    16, 2, 32, device="cuda", dtype=torch.float16, requires_grad=True
                )
                value_cache = torch.randn(
                    16, 2, 32, device="cuda", dtype=torch.float16, requires_grad=True
                )
                ref_query = query.detach().float().requires_grad_(True)
                ref_key = key_cache.detach().float().requires_grad_(True)
                ref_value = value_cache.detach().float().requires_grad_(True)
                # Both requests share physical block 2, exercising gradient accumulation.
                block_table = torch.tensor([[2, 0], [2, 3]], device="cuda")
                sequence_lengths = torch.tensor([5, 7], device="cuda")

                output = paged_triton_attention(
                    query,
                    key_cache,
                    value_cache,
                    block_table,
                    sequence_lengths,
                    block_size=4,
                    normalizer=normalizer,
                )
                reference = paged_attention(
                    ref_query,
                    ref_key,
                    ref_value,
                    block_table,
                    sequence_lengths,
                    block_size=4,
                    normalizer=normalizer,
                )
                upstream = torch.randn_like(reference)
                output.backward(upstream.to(output.dtype))
                reference.backward(upstream)

                self.assertLess((output.float() - reference).abs().mean().item(), 0.01)
                for actual, expected in zip(
                    (query.grad, key_cache.grad, value_cache.grad),
                    (ref_query.grad, ref_key.grad, ref_value.grad),
                ):
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertLess(
                        (actual.float() - expected).abs().mean().item(),
                        0.001,
                    )

    def test_qwen3_incremental_training_with_functional_paged_cache(self):
        from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM

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
        model = Qwen3ForCausalLM(config).to(device="cuda", dtype=torch.float16).train()
        set_softmax_attention_backend(model, base_backend="triton", softmax_fn="sparsemax")
        cache = DifferentiablePagedCache(
            num_hidden_layers=1,
            block_table=torch.tensor([[0, 1], [2, 3]], device="cuda"),
            block_size=4,
            num_key_value_heads=2,
            head_dim=16,
            dtype=torch.float16,
            device=torch.device("cuda"),
        )

        model(
            torch.randint(0, config.vocab_size, (2, 4), device="cuda"),
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(4, device="cuda"),
        )
        output = model(
            torch.randint(0, config.vocab_size, (2, 4), device="cuda"),
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(4, 8, device="cuda"),
        ).logits
        cache.layers[0].keys.retain_grad()
        cache.layers[0].values.retain_grad()
        output.float().square().mean().backward()

        self.assertEqual(cache.get_seq_length(), 8)
        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNotNone(cache.layers[0].keys.grad)
        self.assertIsNotNone(cache.layers[0].values.grad)
        self.assertTrue(torch.isfinite(cache.layers[0].keys.grad).all())
        self.assertTrue(torch.isfinite(cache.layers[0].values.grad).all())
        parameter_gradients = [
            parameter.grad for parameter in model.parameters() if parameter.grad is not None
        ]
        self.assertTrue(parameter_gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in parameter_gradients))


if __name__ == "__main__":
    unittest.main()
