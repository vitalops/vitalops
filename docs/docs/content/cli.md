# Command line

```
vitalops plan MODEL [options]           plan a deployment (the default command)
vitalops plan MODEL [options]      the same, spelled out
vitalops gpus [--json]             the GPU catalogue
vitalops models [--json]           the built-in model catalogue
vitalops ui [--port N] [--no-browser]   the interactive planner, served locally
vitalops --version
```

`MODEL` is a Hugging Face repository id such as `Qwen/Qwen3-32B` or `meta-llama/Llama-3.3-70B-Instruct`, or a catalogue name such as `llama-70b`, `qwen3-32b`, `deepseek-v3`. Names are case-insensitive. `vitalops models` lists the catalogue.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--gpu KEY` | `h100` | The GPU, by catalogue key: `h100`, `h200`, `b200`, `a100`, `l40s`, `rtx5090`, `mi300x`, `agx_orin`. Aliases such as `5090` and `mi300` work. |
| `--all-gpus` | off | One row per GPU in the catalogue instead of a single report. |
| `--weights FMT` | `fp8` | Weight format: `bf16` (2 bytes per parameter), `fp8` (1), `int4` (0.5625, including group scales). Sets the tensor-core rate the matmuls run at. |
| `--kv FMT` | `fp8` | KV-cache format: `bf16` or `fp8`. |
| `--users N` | `64` | Concurrent requests across the whole deployment. |
| `--prompt N` | `2048` | Prompt tokens per request. |
| `--output N` | `300` | Output tokens per request. |
| `--slo MS` | `500` | Time-to-first-token target, in milliseconds. The layout search prefers the fewest GPUs that meet it. |
| `--price $` | the catalogue's | Dollars per GPU-hour, for the cost lines. |
| `--tp N` | search | Force a tensor-parallel degree of 1, 2, 4 or 8 instead of searching. |
| `--revision REV` | `main` | The Hub revision to read `config.json` from. |
| `--offline` | off | Never contact the Hub; catalogue names only. |
| `--json` | off | Machine-readable output, see below. |
| `-q`, `--quiet` | off | Omit the assumptions footer. |

## The report

```
$ vitalops plan meta-llama/Llama-3.3-70B-Instruct --gpu h100 --users 64
vitalops plan · Llama-3.3-70B on H100 SXM
─────────────────────────────────────────
  model      70.6B params · 80 layers · KV 160 KB/token (fp8)
  weights    fp8 65.8 GiB · matmuls at 1,979 TFLOPS fp8
  traffic    64 concurrent requests × (2,048 in + 300 out) · TTFT target 500 ms

  layout     TP2 × 1 replica = 2 GPUs · batch 64 per replica (max 218)
  decode     20.4 tok/s per stream · 1,304 tok/s aggregate · 4.35 requests/s
  TTFT       155 ms = prefill 135 ms + one step 20.3 ms · target met
  step       20.3 ms = weights 14.0 ms + KV 4.59 ms + all-reduces 1.65 ms (8%) · memory-bandwidth bound (compute floor 4.57 ms)
  memory/GPU weights 32.9 GiB · KV 11.5 GiB · runtime 7.90 GiB · free 27.8 GiB of 80.0 GiB
  cost       $1.07 per 1M output tokens · $3,648 / month at $2.50 per GPU-hour

  Assumptions: decode reads the weights (the experts a batch touches, for a mixture of experts) plus every running request's KV cache at 75% of peak bandwidth,
  with a compute floor at 50% of the dense tensor-core rate; prefill at 60% of that rate; 8% of memory + 1.5 GB reserved for the runtime;
  two all-reduces per layer for tensor parallelism over the GPU's link with 8 µs latency on NVLink or Infinity Fabric and 25 µs on PCIe; no queueing;
  the KV cache is reserved for prompt + output tokens for the whole request. Estimates for planning, not measurements.
```

Line by line:

- **model**: parameters (total, and active per token for a mixture of experts), layers, and the KV cache per token at the chosen format. `[hub]` after the parameter count means the shapes were read from the Hub rather than the catalogue.
- **weights**: the weight bytes at the chosen format and the dense tensor-core rate the matmuls run at. An A100 has no fp8 units, so fp8 weights run at the bf16 rate; Blackwell runs int4 as NVFP4.
- **layout**: tensor-parallel degree × replicas, the batch per replica and the most requests the replica's KV cache would hold.
- **decode**: tokens per second per stream with prefill amortised, the aggregate across replicas, and the requests per second that implies.
- **TTFT**: one prompt's prefill plus one decode step, against the target.
- **step**: the decode step split into the weight read, the KV read, the all-reduces, and whether the step is bound by memory or compute.
- **memory/GPU**: weights, KV cache for this batch, the runtime reserve and what is left on each GPU.
- **cost**: dollars per million output tokens and per month, at the price per GPU-hour.
- **note** lines appear when the model has a caveat (sliding-window layers, a disagreement between the checkpoint size and the config), when the GPU has one (the A100's fp8), or when no layout meets the target.

When the model does not fit a tensor-parallel group of up to eight GPUs the report says so and why:

```
$ vitalops plan deepseek-v3 --gpu h100 -q
vitalops plan · DeepSeek-V3 (MoE, MLA) on H100 SXM
──────────────────────────────────────────────────
  model      671.0B (37.0B active) params · 61 layers · KV 34 KB/token (fp8)
  weights    fp8 625 GiB · matmuls at 1,979 TFLOPS fp8
  traffic    64 concurrent requests × (2,048 in + 300 out) · TTFT target 500 ms

  does not fit: fp8 weights are 624.9 GiB; with the runtime reserve and one request's KV cache (632.9 GiB) that exceeds 8 × 80 GB.
```

## Comparing GPUs

```
$ vitalops plan qwen3-32b --all-gpus -q
vitalops plan · Qwen3-32B · fp8 weights, fp8 KV · 64 × (2,048 in + 300 out) · TTFT target 500 ms
  GPU             GPUs  TP  tok/s/user  aggregate     TTFT  $/1M out
  ────────────────────────────────────────────────────────────────────
  H100 SXM           1   1        22.5      1,437   134 ms     $0.48
  H200 SXM           1   1        26.1      1,668   127 ms     $0.58
  B200               1   1        52.2      3,341  58.3 ms     $0.46
  A100 SXM 80 GB     2   2        10.3        659   387 ms     $1.35
  L40S               2   1        10.3        657   370 ms     $0.85
  RTX 5090           2   2        10.7        687   348 ms     $0.57
  MI300X             1   1        32.1      2,054  98.5 ms     $0.34
```

The last column is blank when the layout meets the TTFT target, and says `misses TTFT target` or `does not fit up to TP8` otherwise. The Jetson is left out of this table.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | The model fits (or, for `--all-gpus`, `gpus`, `models`: the command ran). |
| 1 | The model does not fit a group of up to eight of this GPU. |
| 2 | A bad argument: unknown model or GPU, or the Hub could not be read. |

That makes a plan usable as a check in a script or in CI:

```
vitalops plan Qwen/Qwen3-32B --gpu l40s --users 32 --slo 400 -q || echo "needs a different layout"
```

## JSON

`--json` prints the whole `Plan` as JSON, with the model's shapes under `model_shapes`. Times are in seconds, sizes in bytes.

```
$ vitalops plan qwen3-32b --json
{
  "fits": true,
  "model": "Qwen3-32B",
  "gpu": "H100 SXM",
  "weights": "fp8",
  "kv": "fp8",
  "rate_tflops": 1979,
  "rate_dtype": "fp8",
  "users": 64,
  "prompt": 2048,
  "output": 300,
  "slo_ms": 500,
  "weight_bytes": 32800000000.0,
  "kv_bytes_per_token": 131072.0,
  "kv_bytes_per_request": 307757056.0,
  "mem_bytes": 85899345920,
  "layout": {
    "tp": 1, "replicas": 1, "gpus": 1, "batch": 64, "max_batch": 144,
    "weights_per_gpu": 32800000000.0, "kv_per_gpu": 19696451584.0, "runtime_per_gpu": 8482560409.6, "kv_cap": 44616785510.4,
    "perf": {
      "step_s": 0.020393, "t_weights_s": 0.013055, "t_kv_s": 0.007339, "t_compute_s": 0.004243, "t_comm_s": 0.0,
      "prefill_s": 0.113145, "ttft_s": 0.133539, "user_tps": 22.456, "agg_tps": 1437.2, "memory_bound": true
    },
    "slo_ok": true
  },
  "options": [ ...one entry per tensor-parallel degree that fits, same shape as layout... ],
  "agg_tps": 1437.2,
  "user_tps": 22.456,
  "ttft_ms": 133.54,
  "requests_per_s": 4.79,
  "cost_per_1m_output": 0.483,
  "cost_per_month": 1824.0,
  "price_per_gpu_hour": 2.5,
  "reason": "",
  "model_shapes": {
    "id": "Qwen/Qwen3-32B", "name": "Qwen3-32B", "total": 32800000000.0, "active": 32800000000.0,
    "layers": 64, "kv_heads": 8, "head_dim": 128, "hidden": 5120, "kv_per_layer": null,
    "experts": 0, "top_k": 0, "source": "catalogue", "note": ""
  }
}
```

With `--all-gpus --json` the output is an object keyed by GPU, each value a `Plan`. `vitalops gpus --json` and `vitalops models --json` dump the catalogues.

## The Hub

A repository id that is not in the catalogue is read from the Hub: `config.json` for the shapes, and `model.safetensors.index.json` (or the size of `model.safetensors`) for the exact parameter count. Fetched configs are cached for a week under `~/.cache/vitalops`. Gated repositories need `HF_TOKEN`:

```
HF_TOKEN=hf_... vitalops meta-llama/Meta-Llama-3-8B-Instruct
```

Without a token the command exits with code 2 and says the repository is gated. [Models](#models) describes what is read and how it is interpreted.
