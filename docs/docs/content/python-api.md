# Python API

```python
import vitalops

p = vitalops.plan("Qwen/Qwen3-235B-A22B", gpu="h200", weights="fp8", users=128, prompt=4096, output=512)
if p.fits:
    print(p.layout.gpus, "GPUs at TP", p.layout.tp)
    print(round(p.user_tps, 1), "tok/s per stream,", round(p.ttft_ms), "ms TTFT")
    print("$%.2f per 1M output tokens" % p.cost_per_1m_output)
else:
    print(p.reason)
```

Everything the command line prints comes from these functions and dataclasses.

## `vitalops.plan(model, gpu="h100", **kw) -> Plan`

Plans a deployment. `model` is a catalogue name, a Hugging Face repository id, or a `Model`; `gpu` is a catalogue key or a `GPU`. Keyword arguments, with the command-line defaults:

| Argument | Default | Meaning |
|---|---|---|
| `weights` | `"fp8"` | `"bf16"`, `"fp8"` or `"int4"` |
| `kv` | `"fp8"` | `"bf16"` or `"fp8"` |
| `users` | `64` | concurrent requests |
| `prompt` | `2048` | prompt tokens per request |
| `output` | `300` | output tokens per request |
| `slo_ms` | `500.0` | time-to-first-token target |
| `price` | `None` | dollars per GPU-hour; the catalogue's figure when `None` |
| `tp` | `None` | force a tensor-parallel degree; search 1, 2, 4, 8 when `None` |
| `offline` | `False` | never contact the Hub |

Raises `LookupError` for an unknown name, a gated repository without `HF_TOKEN`, or a config this planner does not understand; network errors from `urllib` propagate.

## `vitalops.fit(model, gpu="h100", **kw) -> bool`

`plan(...).fits`: whether the model, with one request's KV cache, fits a tensor-parallel group of up to eight of this GPU.

## `Plan`

A dataclass; `.to_dict()` gives the JSON the command line prints.

| Field | Type | Meaning |
|---|---|---|
| `fits` | bool | whether any layout fits |
| `model`, `gpu` | str | display names |
| `weights`, `kv` | str | the formats used |
| `rate_tflops`, `rate_dtype` | float, str | the dense tensor-core rate the matmuls run at, and in which format |
| `users`, `prompt`, `output`, `slo_ms` | | the scenario |
| `weight_bytes` | float | parameters × bytes per parameter |
| `kv_bytes_per_token`, `kv_bytes_per_request` | float | the KV cache per token and per request (prompt + output) |
| `mem_bytes` | float | one GPU's memory |
| `layout` | `Layout` or `None` | the chosen layout |
| `options` | list of `Layout` | every tensor-parallel degree that fits |
| `agg_tps` | float | tokens per second across all replicas |
| `user_tps` | float | tokens per second per stream, prefill amortised |
| `ttft_ms` | float | time to first token |
| `requests_per_s` | float | `agg_tps / output` |
| `cost_per_1m_output`, `cost_per_month`, `price_per_gpu_hour` | float | at the price used |
| `reason` | str | why it does not fit, or why the target was missed; empty otherwise |

## `Layout`

| Field | Meaning |
|---|---|
| `tp`, `replicas`, `gpus` | tensor-parallel degree, replica count, `tp × replicas` |
| `batch`, `max_batch` | requests per replica, and the most its KV cache would hold (capped at 512) |
| `weights_per_gpu`, `kv_per_gpu`, `runtime_per_gpu` | bytes on each GPU |
| `kv_cap` | bytes of KV cache available across the group |
| `perf` | a `Perf` |
| `slo_ok` | whether `perf.ttft_s` meets the target |

## `Perf`

One operating point: a batch on a tensor-parallel group. All times in seconds.

| Field | Meaning |
|---|---|
| `step_s` | one decode step: `max(t_weights_s + t_kv_s, t_compute_s) + t_comm_s` |
| `t_weights_s`, `t_kv_s` | the weight read and the KV read at 75% of peak bandwidth |
| `t_compute_s` | the matmuls at 50% of the dense rate |
| `t_comm_s` | the tensor-parallel all-reduces |
| `prefill_s` | one prompt, including its all-reduces |
| `ttft_s` | `prefill_s + step_s` |
| `user_tps` | `1 / step_s`, reduced by the prefill share |
| `agg_tps` | `user_tps × batch`, for this replica |
| `memory_bound` | whether the memory time exceeds the compute time |

`vitalops.perf(scenario, tp, batch)` evaluates one point directly:

```python
from vitalops import Scenario, perf, load_model, get_gpu

s = Scenario(model=load_model("llama-70b"), gpu=get_gpu("h100"), weights="fp8", kv="fp8", users=64, prompt=2048, output=300)
for b in (1, 8, 64, 256):
    pf = perf(s, tp=2, batch=b)
    print(b, round(1 / pf.step_s, 1), "tok/s per stream decode only,", round(pf.agg_tps), "aggregate")
```

## `Scenario`

The input dataclass: `model`, `gpu`, `weights`, `kv`, `users`, `prompt`, `output`, `slo_ms`, `price`, `tp`. Properties `wb` and `kvb` give the bytes per weight and per KV value, `gpu_price` the price in use.

## The catalogues

```python
vitalops.load_model("qwen3-32b")                 # catalogue name
vitalops.load_model("Qwen/Qwen2.5-14B-Instruct")  # the Hub, cached for a week
vitalops.from_hub("Qwen/Qwen2.5-14B-Instruct", revision="main")
vitalops.catalogue()                              # list of Model
vitalops.get_gpu("h100")                          # GPU
vitalops.GPUS                                     # dict of key -> GPU
```

`Model` has `id`, `name`, `total`, `active`, `layers`, `kv_heads`, `head_dim`, `hidden`, `kv_per_layer`, `experts`, `top_k`, `source` and `note`, plus `moe` (whether `active < total`), `kv_width()` and `kv_bytes_per_token(kv_bytes)`. `GPU` has `key`, `name`, `vendor`, `arch`, `sms`, `mem`, `mem_type`, `stacks`, `stack_gb`, `bw`, `flops`, `link_name`, `link_gbps`, `tdp`, `price`, `latency_ns`, `note`, plus `pcie` and `rate(weights)`.

Both dataclasses are plain Python: a `Model` or `GPU` you construct yourself is accepted anywhere a name is, which is how to plan for hardware or a model not in the catalogues.

```python
from vitalops import Model, GPU, plan

my_gpu = GPU(key="x", name="My accelerator", vendor="x", arch="x", sms=100, mem=96, mem_type="HBM3", stacks=6, stack_gb=16,
             bw=4.0, flops={"bf16": 1000, "fp8": 2000}, link_name="NVLink", link_gbps=400, tdp=700, price=3.0)
print(plan("qwen3-32b", my_gpu, offline=True).user_tps)
```

## `vitalops.bench.bench(url, **kw) -> Run`

Measures an OpenAI-compatible endpoint; the Python side of [vitalops bench](#bench).

```python
from vitalops.bench import bench

run = bench("http://localhost:8000", users=(1, 8, 32), prompt=1024, output=128, slo_ms=400,
            plan_gpu="h100", tp=1, scenario_kw={"weights": "fp8", "kv": "fp8"})
run.capacity                  # the most tested concurrency whose p95 TTFT met slo_ms, or None
for lv in run.levels:         # one Level per concurrency
    lv.ttft_ms_p95, lv.itl_ms_p50, lv.output_tps, lv.user_tps, lv.predicted
run.to_dict()                 # the JSON the command prints
```

Keyword arguments: `users`, `prompt`, `output`, `shared_prefix`, `requests`, `slo_ms`, `model`, `api` (`"completions"` or `"chat"`), `api_key` (default `OPENAI_API_KEY`), `seed`, `plan_gpu`, `plan_model`, `tp`, `scenario_kw`, `progress` (a function that takes a status line). It raises `vitalops.bench.BenchError` when the server cannot be reached or accepts no request; failed requests during a run are counted in `Level.errors`.
