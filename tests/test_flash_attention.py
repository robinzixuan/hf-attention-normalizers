import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hf_attention_normalizers.backends import flash_attention_softmax_n_forward


class _AttentionModule(nn.Module):
    is_causal = True


class FlashAttentionSoftmaxNForwardTest(unittest.TestCase):
    def test_expands_grouped_key_value_heads(self):
        captured = {}

        def fake_flash_attention_n(*, query, key, value, **kwargs):
            captured["shapes"] = (query.shape, key.shape, value.shape)
            return torch.zeros_like(query)

        fake_module = types.SimpleNamespace(flash_attention_n=fake_flash_attention_n)
        query = torch.randn(2, 4, 8, 16)
        key = torch.randn(2, 2, 8, 16)
        value = torch.randn(2, 2, 8, 16)

        with patch.dict(sys.modules, {"flash_attention_softmax_n": fake_module}):
            output, weights = flash_attention_softmax_n_forward(
                _AttentionModule(),
                query,
                key,
                value,
                attention_mask=None,
                is_causal=True,
            )

        self.assertEqual(
            captured["shapes"],
            (torch.Size((2, 4, 8, 16)),) * 3,
        )
        self.assertEqual(output.shape, (2, 8, 4, 16))
        self.assertIsNone(weights)

    def test_rejects_non_divisible_head_counts(self):
        fake_module = types.SimpleNamespace(flash_attention_n=lambda **kwargs: None)
        query = torch.randn(1, 3, 8, 16)
        key = torch.randn(1, 2, 8, 16)
        value = torch.randn(1, 2, 8, 16)

        with patch.dict(sys.modules, {"flash_attention_softmax_n": fake_module}):
            with self.assertRaisesRegex(ValueError, "3 query heads and 2 key/value heads"):
                flash_attention_softmax_n_forward(
                    _AttentionModule(),
                    query,
                    key,
                    value,
                    attention_mask=None,
                )


if __name__ == "__main__":
    unittest.main()
