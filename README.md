<div align="center">

# tokplan

Will the model fit, how fast will it run, how many GPUs, what will it cost.

[Live demo](https://vitalops.ai/tokplan/) · [Docs](https://vitalops.ai/tokplan/docs/)

</div>

```
$ tokplan meta-llama/Llama-3.3-70B-Instruct --gpu h100 --users 64
tokplan · Llama-3.3-70B on H100 SXM
───────────────────────────────────
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

tokplan reads a model's shapes from its `config.json` on the Hugging Face Hub (or from a built-in
catalogue of popular models, offline), takes the GPU from a catalogue of public datasheet figures,
and applies a first-order model of inference: decode is bound by memory bandwidth, prefill by
compute, tensor parallelism costs two all-reduces per layer over the real link. It searches the
tensor-parallel degree and the replica count for the fewest GPUs that meet your time-to-first-token
target, and prices the result.

It is pure Python with no dependencies. The numbers are estimates for planning, not measurements:
every assumption is printed under the report, and the real figure for your model on your hardware
is what you measure afterwards.

## Install

```
pip install tokplan
```

or download one file and run it directly:

```
wget https://github.com/vitalops/tokplan/releases/download/onefile/tokplan.py
python tokplan.py qwen3-32b
```

Requires Python 3.9 or newer. Nothing else.

## Usage

```
tokplan MODEL [--gpu h100] [--users 64] [--prompt 2048] [--output 300] [--weights fp8] [--kv fp8] [--slo 500] [--price 2.50] [--tp N]
tokplan MODEL --all-gpus          # one row per GPU in the catalogue
tokplan MODEL --json              # everything, machine-readable
tokplan gpus                      # the GPU catalogue and its figures
tokplan models                    # the built-in model catalogue
tokplan ui                        # the interactive planner, served locally
```

`MODEL` is a Hugging Face repository id such as `Qwen/Qwen3-32B`, or a catalogue name such as
`llama-70b`. For a gated repository set `HF_TOKEN`. Fetched configs are cached for a week under
`~/.cache/tokplan`; `--offline` uses the catalogue only.

The exit code is 0 when the model fits, 1 when it does not, 2 on a bad argument, so it works in a
script or a CI check:

```
tokplan Qwen/Qwen3-32B --gpu l40s --users 32 --slo 400 -q || echo "needs a different layout"
```

### From Python

```python
import tokplan

p = tokplan.plan("Qwen/Qwen3-235B-A22B", gpu="h200", weights="fp8", users=128, prompt=4096, output=512)
if p.fits:
    print(p.layout.gpus, "GPUs at TP", p.layout.tp)
    print(round(p.user_tps, 1), "tok/s per stream,", round(p.ttft_ms), "ms TTFT")
    print("$%.2f per 1M output tokens" % p.cost_per_1m_output)
else:
    print(p.reason)

for layout in p.options:          # every tensor-parallel degree that fits, with its figures
    print(layout.tp, layout.gpus, layout.perf.step_s)
```

`tokplan.plan` returns a `Plan` dataclass; `.to_dict()` gives the JSON the CLI prints.
`tokplan.load_model(id)` and `tokplan.get_gpu(key)` expose the catalogues, and `tokplan.perf(scenario, tp, batch)`
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

`tokplan gpus` lists the hardware: H100, H200, B200, A100, L40S, RTX 5090, MI300X and Jetson AGX
Orin, with memory, bandwidth, dense TFLOPS per format, interconnect and a default price per hour.
`tokplan models` lists the built-in models. Both are plain Python in `tokplan/hw.py` and
`tokplan/models.py`; corrections and additions are welcome as pull requests with a datasheet link.

## Development

```
git clone https://github.com/vitalops/tokplan && cd tokplan
pip install -e .[dev]
pytest
python tools/onefile.py        # writes dist/tokplan.py, the single-file build
```

The web demo in `docs/` and `tokplan ui` share their JavaScript with [vitalops.ai](https://vitalops.ai/lab/);
the CLI's figures and the page's agree, and the tests check one scenario against the other.

## Licence

MIT. Built by [VitalOps](https://vitalops.ai).
