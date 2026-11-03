# The physics

vitalops plan is a first-order model of transformer inference. This page is the whole of it: the formulas, the constants, what is left out, and how to compare an estimate with a measurement. The code is `vitalops/physics.py`, about 150 lines.

Throughout: `b` is the batch (requests on one replica), `tp` the tensor-parallel degree (GPUs in one replica), `P` the prompt tokens and `O` the output tokens per request.

## Memory

**Weights.** `weight_bytes = total_params × bytes_per_param`, with 2 for bf16, 1 for fp8 and 0.5625 for int4 (4 bits plus the group scales). Split evenly across the `tp` GPUs of a replica.

**KV cache per token.** `layers × 2 × kv_heads × head_dim × bytes_per_value`, with 2 for bf16 and 1 for fp8. A model with latent attention uses `layers × kv_per_layer × bytes_per_value` instead (576 per layer for DeepSeek-V3). Each request reserves `P + O` tokens for its whole life, as a serving engine does when it admits it.

**Runtime reserve.** `0.08 × memory + 1.5 GB` on each GPU for activations, CUDA graphs and allocator slack.

**KV capacity.** `(memory − reserve − weights / tp) × tp` across the group; `max_batch = min(512, kv_capacity / kv_per_request)`.

## The decode step

One step generates one token for every request in the batch. Its time is

```
step = max(t_memory, t_compute) + t_comm
```

**Memory time.** The weights the step reads plus the batch's KV cache, at 75% of the group's aggregate peak bandwidth:

```
t_memory = (weights_read + b × (P + O/2) × kv_per_token) / (0.75 × bw × tp)
```

`P + O/2` is the average context over the life of a request. For a dense model `weights_read = weight_bytes`. For a mixture of experts the step reads only the experts the batch touches: with `p = active / total`,

```
weights_read = weight_bytes × (1 − (1 − p)^b)
```

which is about the active parameters at `b = 1` and nearly all of them at a large batch.

**Compute time.** The matmuls, 2 FLOPs per active parameter per token, at 50% of the dense tensor-core rate for the format the GPU runs:

```
t_compute = 2 × active_params × b / (0.50 × rate × tp)
```

The rate is bf16 everywhere; fp8 where the GPU has fp8 tensor cores; fp4 for int4 weights on Blackwell. The step is reported as memory-bound when `t_memory ≥ t_compute`, which is the usual case below a batch of a few hundred.

**Communication.** Tensor parallelism needs two all-reduces per layer per step. A ring all-reduce passes `2(tp − 1)/tp` of the tensor through each GPU's link, and each all-reduce pays a latency:

```
t_comm = layers × 2 × (b × hidden × 2 bytes × 2(tp − 1)/tp / link_bandwidth + latency)
```

with 8 µs on NVLink and Infinity Fabric and 25 µs on PCIe. At `tp = 1` it is zero.

## Prefill and time to first token

Prefill is compute bound at 60% of the dense rate, plus its own all-reduces over the prompt tokens:

```
t_prefill = P × 2 × active_params / (0.60 × rate × tp) + t_comm(P tokens)
ttft = t_prefill + step
```

No queueing is modelled: a request is assumed to start prefill when it arrives.

## Tokens per second

The decode rate of one stream is `1 / step`. Prefill shares the GPU with decode, so its time is amortised over the batch's output tokens: with `k = b × t_prefill / O`,

```
user_tps = (1 / step) / (1 + k / step)
agg_tps  = user_tps × b × replicas
```

This is why a long prompt lowers the per-stream decode rate even though the step itself is unchanged.

## The layout search

For each `tp` in 1, 2, 4, 8:

1. Skip it if the weights and the reserve do not leave room for one request's KV cache.
2. `max_batch` from the KV capacity; `replicas = ceil(users / max_batch)`; `b = ceil(users / replicas)`.
3. Evaluate the step, the prefill and TTFT at that `b`.

Among the layouts that meet the TTFT target, the fewest GPUs win (and the lower `tp` on a tie); if none meets it, the fastest TTFT is shown with a note. `--tp` restricts the search to one degree.

## Cost

```
cost_per_1M_output = gpus × price_per_hour / (agg_tps × 3600 / 1e6)
cost_per_month     = gpus × price_per_hour × 24 × 30.4
```

## The constants

| Constant | Value | Why |
|---|---|---|
| Bandwidth efficiency | 0.75 | what well-tuned decode kernels reach of peak HBM bandwidth |
| Decode compute efficiency | 0.50 | small-batch matmuls do not reach the dense rate |
| Prefill compute efficiency | 0.60 | long sequences do better |
| Runtime reserve | 8% + 1.5 GB | activations, CUDA graphs, allocator slack |
| All-reduce latency | 8 µs NVLink / IF, 25 µs PCIe | per collective, from microbenchmarks of the two fabrics |
| Batch cap | 512 | serving engines' usual maximum concurrency |
| int4 bytes per parameter | 0.5625 | 4 bits + fp16 scale per group of 128 |

They live at the top of `vitalops/physics.py`. Change them there, or build a `Scenario` and call `perf` with your own, when you have measurements to calibrate against.

## Not modelled

- Queueing and arrival patterns: every request is assumed to be running or prefilling, never waiting.
- Speculative decoding, which raises the per-stream rate by accepting several tokens per step.
- Prefix caching and chunked prefill, which change TTFT for shared or long prompts.
- Kernel efficiency at very small batches (below 4) and very large ones, CUDA-graph effects, attention-kernel differences.
- Pipeline and expert parallelism; only tensor parallelism and replicas.
- Sliding-window attention, which caches less than the estimate for Gemma 3 and gpt-oss.
- Memory fragmentation in a paged KV cache.

These are the reasons a measured figure differs from an estimate, usually by tens of percent, not by multiples.

## Calibrating against a measurement

Run the model on the hardware with a load generator at the scenario's concurrency and prompt and output lengths, and compare:

- **Aggregate tokens per second** against `agg_tps`. If the measurement is lower across the board, the bandwidth efficiency is the first constant to lower; if it is lower only at large batches, the compute efficiency.
- **TTFT at low load** against `ttft_ms`. A gap here is prefill efficiency, or chunked prefill in the engine.
- **The fit**: if the engine refuses the concurrency the plan claims, the runtime reserve is too small for that engine and model; raise it.

`vitalops plan MODEL --json` gives every intermediate figure, so a comparison can be scripted. The constants are the only knobs; the shapes and the datasheet figures are facts.
