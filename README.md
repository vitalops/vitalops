<div align="center">

# vitalops

Open-source tools for running LLMs in production.

[Live planner, running this package in the browser](https://vitalops.ai/oss/) · [Docs](https://vitalops.ai/oss/docs/)

</div>

```
vitalops plan  MODEL --gpu GPU    will it fit, how fast, how many GPUs, what will it cost
vitalops bench URL --plan GPU     measure an OpenAI-compatible endpoint, next to the plan
vitalops ui                       the planner in a browser
```

## vitalops plan

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
```

vitalops plan reads a model's shapes from its `config.json` on the Hugging Face Hub (or from a built-in
catalogue of popular models, offline), takes the GPU from a catalogue of public datasheet figures,
and applies a first-order model of inference: decode is bound by memory bandwidth, prefill by
compute, tensor parallelism costs two all-reduces per layer over the real link. It searches the
tensor-parallel degree and the replica count for the fewest GPUs that meet your time-to-first-token
target, and prices the result.

The numbers are estimates for planning, not measurements: every assumption is printed under the
report, and `vitalops bench` measures the real figure.

## vitalops bench

```
vitalops bench http://localhost:8000 --users 1,8,32,64 --prompt 1024 --output 256 --slo 500 --plan h100
```

Points concurrent streaming load at any OpenAI-compatible server (vLLM, SGLang, TGI, llama.cpp, a
hosted API) and reports, per concurrency level, time to first token and time between tokens (p50
and p95), tokens per second per stream and in total, and requests per second. It finds the most
concurrent requests that still meet the TTFT target. With `--plan GPU` the planner's estimate for
the same traffic is printed under each measured row, and large gaps are named: queueing, small-batch
overheads, prefix caching. `--require 32` exits 1 when the target is missed at 32 requests, for CI.

Prompts are calibrated to the server's own token counts, every request is distinct unless
`--shared-prefix` asks otherwise, and the first request of each worker is a warm-up.
[The bench docs](https://vitalops.ai/oss/docs/#bench) cover the method.

Everything is pure Python with no dependencies, the load generator included.

## Install

```
pip install vitalops
```

or download one file and run it directly:

```
wget https://github.com/vitalops/vitalops/releases/download/onefile/vitalops.py
python vitalops.py plan qwen3-32b
```

Requires Python 3.9 or newer. Nothing else.

## Usage

```
vitalops bench URL [--users 1,4,16,64] [--prompt 1024] [--output 256] [--shared-prefix N] [--slo 500] [--plan GPU] [--require N] [--json]
vitalops plan MODEL [--gpu h100] [--users 64] [--prompt 2048] [--output 300] [--weights fp8] [--kv fp8] [--slo 500] [--price 2.50] [--tp N]
vitalops plan MODEL --all-gpus          # one row per GPU in the catalogue
vitalops plan MODEL --json              # everything, machine-readable
vitalops gpus                           # the GPU catalogue and its figures
vitalops models                         # the built-in model catalogue
vitalops ui                        # the interactive planner, served locally
```

`MODEL` is a Hugging Face repository id such as `Qwen/Qwen3-32B`, or a catalogue name such as
`llama-70b`. For a gated repository set `HF_TOKEN`. Fetched configs are cached for a week under
`~/.cache/vitalops`; `--offline` uses the catalogue only.

The exit code is 0 when the model fits, 1 when it does not, 2 on a bad argument, so it works in a
script or a CI check:

```
vitalops plan Qwen/Qwen3-32B --gpu l40s --users 32 --slo 400 -q || echo "needs a different layout"
```

### From Python

```python
import vitalops

p = vitalops.plan("Qwen/Qwen3-235B-A22B", gpu="h200", weights="fp8", users=128, prompt=4096, output=512)
if p.fits:
    print(p.layout.gpus, "GPUs at TP", p.layout.tp)
    print(round(p.user_tps, 1), "tok/s per stream,", round(p.ttft_ms), "ms TTFT")
    print("$%.2f per 1M output tokens" % p.cost_per_1m_output)
else:
    print(p.reason)

for layout in p.options:          # every tensor-parallel degree that fits, with its figures
    print(layout.tp, layout.gpus, layout.perf.step_s)
```

`vitalops.plan` returns a `Plan` dataclass; `.to_dict()` gives the JSON the CLI prints.
`vitalops.load_model(id)` and `vitalops.get_gpu(key)` expose the catalogues, and `vitalops.perf(scenario, tp, batch)`
evaluates one operating point.

## What it models

For a batch of `b` requests on a tensor-parallel group of `tp` GPUs:

- **Weight bytes** = parameters × bytes per parameter (bf16 2, fp8 1, int4 0.5625 with group scales),
  split across the group. A mixture of experts reads only the experts the batch touches:
  total × (1 − (1 − active/total)^b).
- **KV cache** = 2 × layers × KV heads × head dim × bytes per value, per token; models with latent
  attention (DeepSeek) use their latent width instead. Each request reserves prompt + output tokens.
- **Decode step** = max(memory time, compute time) + all-reduce time, where memory time is the
  weights plus the batch's KV cache at 75% of peak bandwidth, and compute time is 2 × active
  parameters × b at 50% of the dense tensor-core rate for the weight format the GPU has
  (an A100 has no fp8 units, so fp8 weights run at the bf16 rate; Blackwell runs int4 as NVFP4).
- **Prefill** is compute bound at 60% of that rate; its time is amortised over the output tokens
  of the batch, which is what lowers per-stream decode speed with long prompts.
- **Tensor parallelism** costs two all-reduces per layer; a ring all-reduce moves 2(tp−1)/tp of the
  activation tensor through each GPU's link, with 8 µs latency on NVLink or Infinity Fabric and
  25 µs on PCIe. That is why TP over PCIe hurts.
- **Memory**: 8% of the GPU's memory plus 1.5 GB is reserved for activations, CUDA graphs and
  allocator slack; what is left after the weights is KV cache.
- **Layout search**: for each tensor-parallel degree 1, 2, 4, 8 that fits, the most requests the
  KV cache allows gives the replica count; the fewest GPUs that meet the TTFT target win, else
  the fastest.
- **Cost** = GPUs × price per hour ÷ output tokens per hour.

Not modelled: queueing, speculative decoding, prefix caching, chunked prefill, CUDA-graph and
kernel efficiency at small batches, pipeline or expert parallelism, sliding-window attention.
These are the reasons a measured number differs from the estimate, usually by tens of percent,
not by multiples.

## The catalogues

`vitalops gpus` lists the hardware: H100, H200, B200, A100, L40S, RTX 5090, MI300X and Jetson AGX
Orin, with memory, bandwidth, dense TFLOPS per format, interconnect and a default price per hour.
`vitalops models` lists the built-in models. Both are plain Python in `vitalops/hw.py` and
`vitalops/models.py`; corrections and additions are welcome as pull requests with a datasheet link.

## Development

```
git clone https://github.com/vitalops/vitalops && cd vitalops
pip install -e .[dev]
pytest
python tools/onefile.py        # writes dist/vitalops.py, the single-file build
```

The web demo in `docs/` runs this package in the browser through Pyodide, and `vitalops ui` calls it over
localhost; both go through `vitalops/web.py`. `python tools/site.py` copies the package into `docs/py/`
after a change (the tests fail until it has). [vitalops.ai/lab](https://vitalops.ai/lab/) uses a JavaScript
port of the same formulas; the tests pin one scenario to the figures it shows.

## Licence

MIT. Built by [VitalOps](https://vitalops.ai).
