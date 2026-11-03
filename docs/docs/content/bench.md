# vitalops bench

`vitalops bench` measures an inference server that speaks the OpenAI API (vLLM, SGLang, TGI, llama.cpp's server, or a hosted API) at several concurrency levels and reports what users see: time to first token, time between tokens, tokens per second per stream and in total. With `--plan GPU` it prints the planner's estimate for the same traffic under each measured row, so a gap between the estimate and the server shows up as a number.

```
vitalops bench URL [--users 1,4,16,64] [--prompt 1024] [--output 256] [--shared-prefix 0] [--requests 16]
                   [--slo 500] [--model ID] [--api completions|chat]
                   [--plan GPU] [--plan-model HF_ID] [--tp 1] [--weights fp8] [--kv fp8]
                   [--require USERS] [--save FILE] [--json] [-q]
```

## An example

This run is against `tests/mock_server.py`, a stand-in server with a made-up timing law, so the measured figures are not a GPU's; the shape of the report is the point. The plan rows are what `vitalops plan` estimates for Llama-3.1-8B on one H100.

```
$ python tests/mock_server.py 8000 &
$ vitalops bench http://localhost:8000 --users 1,8,32 --prompt 1024 --output 128 --slo 80 --plan h100 --plan-model llama-8b
vitalops bench · mock/llama-tiny at http://localhost:8000
─────────────────────────────────────────────────────────
  traffic    1,024 in + 128 out per request · /v1/completions · TTFT target 80 ms (p95)

  users  requests  req/s  tok/s all  tok/s/user  TTFT p50  TTFT p95  ITL p50  ITL p95
  ────────────────────────────────────────────────────────────────────────────────────────────────
      1  16         1.69        216       227.0     33 ms     34 ms   4.4 ms   5.1 ms
         planned    2.34        300       300.1     17 ms             3.2 ms
      8  16         9.48      1,213       162.4     47 ms     48 ms   6.1 ms   6.3 ms
         planned   14.57      1,865       233.2     17 ms             3.4 ms
     32  64        15.47      1,980        91.3     95 ms     96 ms  12.1 ms  12.3 ms  over target
         planned   33.04      4,229       132.2     18 ms             4.1 ms

  capacity   8 concurrent requests meet the 80 ms p95 TTFT target (the most tested that did)
  planned    one TP1 replica on H100 SXM, from `vitalops plan`; no queueing or prefix caching in the estimate
  gap        At 32 requests TTFT is 5.3× the estimate: requests wait behind other prefills. The planner's TTFT has no queueing.
```

## How it measures

- **The model and the API.** The model is the first one `/v1/models` lists, unless `--model` names one. Requests go to `/v1/completions`, or to `/v1/chat/completions` when the server has no completions endpoint (or with `--api chat`). Every request streams, with `temperature 0`.
- **Output length.** Each request asks for exactly `--output` tokens with `max_tokens`, `min_tokens` and `ignore_eos`, which vLLM honours; other servers vary. A server that rejects those fields is asked again without them, and the report notes that outputs can be shorter than asked.
- **Prompts.** Prompts are random common English words. Before the run, two probe requests of different lengths read the server's own `prompt_tokens` count, which gives tokens per word and the chat template's overhead, so a 1,024-token prompt is 1,024 of the server's tokens within a few. Every prompt is different, so nothing is served from a prefix cache, unless `--shared-prefix N` makes the first N tokens identical across requests, like a system prompt.
- **Concurrency.** At each level, that many workers send requests back to back over keep-alive connections (a closed loop). The first request of each worker is a warm-up and is not counted, so the burst of simultaneous first prefills does not inflate TTFT. Then `max(--requests, 2 × users)` requests are measured.
- **What is recorded.** TTFT is from sending the request to the first streamed text. Time between tokens (ITL) is the gap between streamed chunks. Per-stream tokens per second is a request's output tokens after the first, over the time after its first token, averaged over requests. Aggregate tokens per second is all output tokens over the level's wall time. Output token counts come from the server's `usage`; a server that sends none is counted one token per chunk.
- **Capacity** is the highest tested concurrency whose p95 TTFT meets `--slo`.

## Against the plan

`--plan GPU` runs the planner's step model for one replica on that GPU (`--tp`, `--weights` and `--kv` should match the server) at each measured concurrency, with the same prompt and output lengths, and prints it under the measured row. The model for the plan is the served id, which is usually the Hugging Face id under vLLM; when it is not, give the Hugging Face id with `--plan-model`.

The estimate leaves things out, and the report names the likely one when the gap is large:

- TTFT well above the estimate as concurrency rises: requests waiting behind other prefills. The planner's TTFT is one prompt on an idle replica.
- Per-stream speed well below the estimate at one request: kernel launch and scheduling overheads at small batches.
- TTFT below the estimate with a shared prefix: the prefix cache.

A gap of tens of percent elsewhere is the planner's efficiency assumptions (75% of peak bandwidth, 50% of the tensor-core rate) against this server's kernels. A gap of several times usually means the server is not configured the way the plan assumes: a different weight format, a different tensor-parallel degree, or CUDA graphs disabled.

## In CI

`--require USERS` exits 1 unless the p95 TTFT target is met at that concurrency, which must be one of `--users`:

```
vitalops bench $ENDPOINT --users 32 --slo 400 --require 32 -q || exit 1
```

Exit codes: 0 when the run completes (and `--require` is met), 1 when `--require` is missed, 2 when the server cannot be reached or an argument is wrong. Failed requests during a run are counted per level and reported, not fatal.

## JSON

`--json` prints the run; `--save FILE` writes the same to a file. The fields are `url`, `model`, `api`, `prompt`, `output`, `shared_prefix`, `slo_ms`, `capacity`, `exact_output`, `notes`, `plan`, and `levels`: one object per concurrency with `users`, `requests`, `errors`, `duration_s`, `requests_per_s`, `output_tps`, `user_tps`, `ttft_ms_p50`, `ttft_ms_p95`, `itl_ms_p50`, `itl_ms_p95`, `prompt_tokens`, `output_tokens`, `slo_ok` and, with `--plan`, `predicted`.

## From Python

```python
from vitalops.bench import bench

run = bench("http://localhost:8000", users=(1, 8, 32), prompt=1024, output=128, slo_ms=400, plan_gpu="h100")
print(run.capacity)
for lv in run.levels:
    print(lv.users, round(lv.ttft_ms_p95), round(lv.output_tps), lv.predicted and round(lv.predicted["output_tps"]))
```

## Testing without a GPU

`tests/mock_server.py` is a small OpenAI-compatible server for trying the command and for the test suite. It streams placeholder words on a timing law that slows as more requests are in flight; `python tests/mock_server.py 8000` runs it. Its figures say nothing about any real hardware.
