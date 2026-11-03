# Introduction

tokplan answers four questions about running an open-weight model: will it fit, how fast will it run, how many GPUs does it take, and what will it cost. It is a command-line tool and a Python library, pure Python with no dependencies, for Python 3.9 or newer.

```
$ tokplan Qwen/Qwen3-32B --gpu h100 --users 64
tokplan · Qwen3-32B on H100 SXM
───────────────────────────────
  model      32.8B params · 64 layers · KV 128 KB/token (fp8)
  weights    fp8 30.5 GiB · matmuls at 1,979 TFLOPS fp8
  traffic    64 concurrent requests × (2,048 in + 300 out) · TTFT target 500 ms

  layout     TP1 × 1 replica = 1 GPU · batch 64 per replica (max 144)
  decode     22.5 tok/s per stream · 1,437 tok/s aggregate · 4.79 requests/s
  TTFT       134 ms = prefill 113 ms + one step 20.4 ms · target met
  step       20.4 ms = weights 13.1 ms + KV 7.34 ms · memory-bandwidth bound (compute floor 4.24 ms)
  memory/GPU weights 30.5 GiB · KV 18.3 GiB · runtime 7.90 GiB · free 23.2 GiB of 80.0 GiB
  cost       $0.48 per 1M output tokens · $1,824 / month at $2.50 per GPU-hour
```

## What it does

1. Reads the model's shapes: parameter count, layers, KV heads, head dimension, hidden size, and the expert layout for a mixture of experts. For a Hugging Face repository id it fetches `config.json` (and the checkpoint size) from the Hub; for the models in the built-in catalogue it needs no network. See [Models](#models).
2. Takes the GPU from a catalogue of datasheet figures: memory, bandwidth, dense tensor-core rates per format, and the interconnect. See [GPUs](#gpus).
3. Applies a first-order model of inference: decode is bound by memory bandwidth, prefill by compute, tensor parallelism costs all-reduces over the real link. See [The physics](#physics).
4. Searches tensor-parallel degrees 1, 2, 4 and 8, works out how many requests each replica's KV cache holds, and picks the fewest GPUs that meet the time-to-first-token target.
5. Prints a report, or JSON, or one row per GPU in the catalogue.

## What it is not

The figures are estimates for planning. They come from public specifications and a handful of stated efficiency assumptions, and they leave out queueing, speculative decoding, prefix caching, chunked prefill and the efficiency of real kernels at small batch sizes. A measured number on your hardware typically differs by tens of percent, not by multiples; [the physics page](#physics) says how to compare the two.

## Where to go next

- [Install](#install) and run it.
- [Command line](#cli): every flag, the exit codes, the JSON output.
- [Python API](#python-api): `tokplan.plan`, `tokplan.fit`, the `Plan` dataclass.
- [The interactive planner](#ui): the same estimate in a browser, with `tokplan ui` or on [vitalops.ai](https://vitalops.ai/lab/).
