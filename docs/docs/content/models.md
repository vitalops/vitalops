# Models

The planner needs a few shapes from a model: how many parameters it has (and how many a token passes through, for a mixture of experts), how many layers, the KV heads and head dimension that set the KV cache per token, and the hidden size that sets the tensor-parallel traffic. They come from the built-in catalogue or from the Hub.

## The built-in catalogue

`vitalops models` lists it. These need no network and are the names `--offline` accepts; the id, the name and the short aliases (`llama-70b`, `qwen3-32b`, `deepseek-v3`, …) all work.

| Id | Name | Parameters | Layers · KV heads × head dim |
|---|---|---|---|
| `meta-llama/Llama-3.2-1B-Instruct` | Llama-3.2-1B | 1.2B | 16 · 8 × 64 |
| `meta-llama/Llama-3.2-3B-Instruct` | Llama-3.2-3B | 3.2B | 28 · 8 × 128 |
| `meta-llama/Llama-3.1-8B-Instruct` | Llama-3.1-8B | 8.0B | 32 · 8 × 128 |
| `meta-llama/Llama-3.3-70B-Instruct` | Llama-3.3-70B | 70.6B | 80 · 8 × 128 |
| `Qwen/Qwen3-4B` | Qwen3-4B | 4.0B | 36 · 8 × 128 |
| `Qwen/Qwen3-8B` | Qwen3-8B | 8.2B | 36 · 8 × 128 |
| `Qwen/Qwen3-14B` | Qwen3-14B | 14.8B | 40 · 8 × 128 |
| `Qwen/Qwen3-32B` | Qwen3-32B | 32.8B | 64 · 8 × 128 |
| `Qwen/Qwen3-30B-A3B` | Qwen3-30B-A3B (MoE) | 30.5B, 3.3B active | 48 · 4 × 128 |
| `Qwen/Qwen3-235B-A22B` | Qwen3-235B-A22B (MoE) | 235B, 22B active | 94 · 4 × 128 |
| `Qwen/Qwen2.5-7B-Instruct` | Qwen2.5-7B | 7.6B | 28 · 4 × 128 |
| `Qwen/Qwen2.5-72B-Instruct` | Qwen2.5-72B | 72.7B | 80 · 8 × 128 |
| `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | Mistral-Small-24B | 23.6B | 40 · 8 × 128 |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | Mixtral-8x7B (MoE) | 46.7B, 12.9B active | 32 · 8 × 128 |
| `google/gemma-3-12b-it` | Gemma-3-12B | 12.2B | 48 · 8 × 256 |
| `google/gemma-3-27b-it` | Gemma-3-27B | 27.4B | 62 · 16 × 128 |
| `microsoft/phi-4` | Phi-4 (14B) | 14.7B | 40 · 10 × 128 |
| `deepseek-ai/DeepSeek-V3` | DeepSeek-V3 (MoE, MLA) | 671B, 37B active | 61 · latent 576 per layer |
| `deepseek-ai/DeepSeek-R1` | DeepSeek-R1 (MoE, MLA) | 671B, 37B active | 61 · latent 576 per layer |
| `openai/gpt-oss-20b` | gpt-oss-20b (MoE) | 20.9B, 3.6B active | 24 · 8 × 64 |
| `openai/gpt-oss-120b` | gpt-oss-120b (MoE) | 117B, 5.1B active | 36 · 8 × 64 |

Parameter counts are from the published checkpoints or model cards; shapes from each repository's `config.json`. The catalogue is plain Python in `vitalops/models.py`.

## Any repository on the Hub

A repository id that is not in the catalogue is read from the Hub:

1. `config.json` gives the shapes: `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads` (the attention heads when absent), `head_dim` (hidden ÷ heads when absent), `intermediate_size`, `vocab_size`, `tie_word_embeddings`. A multimodal wrapper with a `text_config` is unwrapped to its language model.
2. `model.safetensors.index.json` gives the checkpoint's total bytes; divided by the bytes per value of `torch_dtype`, that is the exact parameter count. A single-file checkpoint is sized with a HEAD request on `model.safetensors` instead. A quantised checkpoint (one with `quantization_config`) is not sized this way, because its bytes do not reflect its parameters; the shapes are used.
3. When the checkpoint size and the shapes disagree by more than a third, the shapes win and the report carries a note.

Fetched data is cached for a week under `~/.cache/vitalops`. Gated repositories need `HF_TOKEN`. `--offline` disables all of this.

```
$ vitalops plan Qwen/Qwen2.5-14B-Instruct --users 8 -q
vitalops plan · Qwen2.5-14B-Instruct on H100 SXM
────────────────────────────────────────────────
  model      14.8B params [hub] · 48 layers · KV 96 KB/token (fp8)
  ...
```

## Mixtures of experts

For a mixture of experts the config's `num_local_experts`, `num_experts` or `n_routed_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `n_shared_experts` and `first_k_dense_replace` are read. The **total** count includes every expert; the **active** count is what one token passes through: attention, the top-k routed experts, the shared experts and the dense layers.

Decode reads only the experts a batch touches. With `p = active / total`, a batch of `b` tokens reads `total × (1 − (1 − p)^b)` of the weights: at batch 1 roughly the active parameters, at a large batch nearly all of them. That is why a mixture of experts is fast for a single stream and converges towards a dense model's cost per step as the batch grows. The matmul FLOPs use the active count.

## Latent attention

DeepSeek-V2 and V3 compress keys and values into a latent vector per token; the KV cache holds `kv_lora_rank + qk_rope_head_dim` values per layer (576 for V3) rather than `2 × KV heads × head dim`. The loader reads those fields and sets `kv_per_layer`; the catalogue entries carry it too. The report shows the result as a much smaller KV cache per token (34 KB at fp8 for DeepSeek-V3, against 160 KB for Llama-3.3-70B).

## Caveats

- **Sliding-window attention**: Gemma 3 and gpt-oss alternate full-attention layers with local ones that cache only a window. The estimate assumes full attention on every layer, so it overstates their KV cache; the report says so in a note.
- **Shapes the loader does not understand**: a config without `hidden_size`, `num_hidden_layers` and `num_attention_heads` (an encoder, a diffusion model, a vision model) is refused with a clear message.
- **Fine-tunes** share their base model's shapes; passing the base's catalogue name is the same as fetching the fine-tune.
- **Your own numbers**: a `Model` constructed directly is accepted everywhere a name is; see [Python API](#python-api).
