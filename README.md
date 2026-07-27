# HF Attention Normalizers

Utilities for experimenting with alternative attention normalizers in HuggingFace Transformers.

Package name:

```bash
pip install hf-attention-normalizers
```

Python import name:

```python
import hf_attention_normalizers
```

This project has two integration paths:

1. Register HuggingFace-style attention backend names such as `softmax1_sdpa`.
2. Patch supported model modules directly with a surgery policy. Currently the built-in policy covers Qwen3.

## Supported Normalizers

| Normalizer | Function | Notes |
| --- | --- | --- |
| `softmax1` | `exp(x_i) / (1 + sum_j exp(x_j))` | Also called Softmax-N with `n=1`. |
| `sparsemax` | sparse probability transform | Can produce exact zeros in attention weights. |
| `entmax15` | 1.5-entmax transform | Between softmax and sparsemax; also sparse. |
| `vanilla` / `softmax` | PyTorch softmax | Baseline. |

## Backend Support Matrix

| Backend name | `softmax1` | `sparsemax` | `entmax15` |
| --- | --- | --- | --- |
| `*_eager` | supported | supported | supported |
| `*_sdpa` | supported via custom SDPA-like path | supported via custom SDPA-like path | supported via custom SDPA-like path |
| `softmax1_flash_attention_2` | supported if `flash-attention-softmax-n` is installed | not supported | not supported |
| `*_triton` | not used | experimental forward kernel | experimental forward kernel |
| `*_flash_attention_3` | registered only; strict mode raises | registered only; strict mode raises | registered only; strict mode raises |
| `*_flex_attention` | registered only; strict mode raises | registered only; strict mode raises | registered only; strict mode raises |
| `*_paged|*` | registered only; strict mode raises | registered only; strict mode raises | registered only; strict mode raises |

`flash-attention-softmax-n` implements Softmax-N kernels. It cannot be reused for `sparsemax` or `entmax15`, because those transforms have different normalization math and would need their own kernels.

## Install

Minimum runtime:

```bash
pip install torch transformers
```

Editable local install:

```bash
pip install -e .
```

Optional Softmax-N FlashAttention backend:

```bash
pip install "hf-attention-normalizers[flash-softmax-n]"
```

Optional experimental sparsemax/entmax Triton kernels:

```bash
pip install "hf-attention-normalizers[triton]"
```

Install every optional backend:

```bash
pip install "hf-attention-normalizers[all]"
```

## HuggingFace-Native Usage

Register backends, then use `attn_implementation` just like native HuggingFace backends:

```python
from transformers import AutoModelForCausalLM
from hf_attention_normalizers import register_softmax1_attention_backends

register_softmax1_attention_backends(mode="strict")

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-...",
    attn_implementation="softmax1_sdpa",
)
```

For sparsemax:

```python
from hf_attention_normalizers import register_sparsemax_attention_backends

register_sparsemax_attention_backends(mode="strict")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="sparsemax_sdpa",
)
```

Experimental sparsemax Triton backend:

```python
from hf_attention_normalizers import register_sparsemax_attention_backends

register_sparsemax_attention_backends(mode="strict")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="sparsemax_triton",
)
```

For entmax:

```python
from hf_attention_normalizers import register_entmax15_attention_backends

register_entmax15_attention_backends(mode="strict")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="entmax15_sdpa",
)
```

Experimental entmax15 Triton backend:

```python
from hf_attention_normalizers import register_entmax15_attention_backends

register_entmax15_attention_backends(mode="strict")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="entmax15_triton",
)
```

## Softmax1 FlashAttention 2

If `flash-attention-softmax-n` is installed, you can use:

```python
from hf_attention_normalizers import register_softmax1_attention_backends

register_softmax1_attention_backends(mode="strict")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="softmax1_flash_attention_2",
)
```

This path calls `flash_attention_softmax_n.flash_attention_n` with `softmax_n_param=1`.

## Strict vs Fallback Mode

Strict mode is safest:

```python
register_softmax1_attention_backends(mode="strict")
```

Unsupported fused backends such as `softmax1_flash_attention_3` raise a clear error.

Fallback mode preserves math but may not preserve the requested backend performance:

```python
register_softmax1_attention_backends(mode="fallback")
```

For unsupported fused or paged names, fallback mode routes to the configured fallback backend, defaulting to custom `sdpa`.

## Already Loaded Models

You can switch an existing model:

```python
from hf_attention_normalizers import set_softmax_attention_backend

set_softmax_attention_backend(
    model,
    base_backend="sdpa",
    softmax_fn="softmax1",
)
```

For sparsemax:

```python
set_softmax_attention_backend(model, base_backend="sdpa", softmax_fn="sparsemax")
```

For entmax:

```python
set_softmax_attention_backend(model, base_backend="sdpa", softmax_fn="entmax15")
```

For the experimental Triton kernels:

```python
set_softmax_attention_backend(model, base_backend="triton", softmax_fn="sparsemax")
set_softmax_attention_backend(model, base_backend="triton", softmax_fn="entmax15")
```

## Qwen3 Surgery Path

For models that do not use HuggingFace `AttentionInterface`, use the surgery path:

```python
from transformers import AutoModelForCausalLM
from hf_attention_normalizers import apply_softmax_attention

model = AutoModelForCausalLM.from_pretrained(model_id)
model = apply_softmax_attention(
    model,
    softmax_fn="softmax1",
    attn_implementation="sdpa",
)
```

Current built-in surgery policies:

```python
from hf_attention_normalizers import supported_attention_policies

print(supported_attention_policies())
# ("qwen3",)
```

Additional model families can be added by registering an `AttentionReplacementPolicy`.

The old `Qwen_attention.py` module is kept as a compatibility shim, but new code should import from `hf_attention_normalizers`.

## Important Limitations

- Native PyTorch SDPA and native FlashAttention kernels do not expose a `softmax_fn` argument.
- `*_sdpa` in this project is a custom SDPA-like implementation that materializes attention weights so it can apply another normalizer.
- `softmax1_flash_attention_2` uses `flash-attention-softmax-n`, not HuggingFace's native FlashAttention 2 kernel.
- `sparsemax_triton` and `entmax15_triton` are experimental forward-only Triton kernels, not full FlashAttention replacements.
- The experimental Triton kernels currently require CUDA tensors, no dropout, no external padding mask on the fast path, matching Q/K/V head dimensions, and `key_length <= max_block_n` where the default is 4096.
- If a mask/dropout/CPU path is encountered through the HuggingFace wrapper, the Triton wrapper falls back to the custom SDPA implementation to preserve math.
- Sparsemax and entmax FlashAttention-quality kernels would still need more work: efficient multi-block reductions, backward kernels, variable-length support, and paged KV-cache support.
- Paged attention needs paged KV-cache-aware kernels and is not implemented yet.
