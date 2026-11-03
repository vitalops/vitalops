# The interactive planner

The same estimate as the command line, in a browser. It is on [this site's front page](../#demo), on [vitalops.ai/lab](https://vitalops.ai/lab/), and in the package:

```
tokplan ui
tokplan ui --port 9000 --no-browser
```

`tokplan ui` serves the page from the package on `127.0.0.1:8765` and opens it in the default browser. It works offline except for the hardware photographs, which are loaded from vitalops.ai when there is a connection and left out otherwise. Ctrl-C stops it. The single-file build embeds the page too, so `python tokplan.py ui` works.

## What is on the panel

- The hardware carousel at the top selects the GPU; it is the only GPU control.
- Model, weight and KV formats, price per GPU-hour, concurrent requests, prompt and output tokens, and the TTFT target on the left; the model list is the website's shorter list of seven, including the two mixtures of experts.
- On the right: GPUs needed, per-stream and aggregate tokens per second, TTFT and cost per million output tokens; the GPU's card with its datasheet figures and its memory drawn as its stacks or channels, filled with weights, KV cache and the runtime reserve; the time per output token broken into the weight read, the KV read, the all-reduces and the amortised prefill; and throughput against batch size for one replica, with the planned operating point marked.
- Every control and readout has a short note on hover or keyboard focus that names the formula or the source.
- "How this is calculated" at the bottom lists every assumption.

## The same physics, twice

The page is JavaScript (`hw.js` for the catalogue, `lab-planner.js` for the model) and the command line is Python; they implement the same formulas with the same constants. The test suite checks one scenario against the other: Qwen3-32B in fp8 on an H100 with 64 requests of 2,048 in and 300 out gives 22.5 tokens per second per stream, 1,437 aggregate, 134 ms TTFT and $0.48 per million output tokens in both. When one changes, the other does.

Differences to know about: the page's model list is shorter than the catalogue and has no Hub lookup; the command line has `--tp`, `--all-gpus` and JSON output. For anything the page cannot express, use the command line or the [Python API](#python-api).
