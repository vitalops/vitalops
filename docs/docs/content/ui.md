# The interactive planner

The same estimate as the command line, in a browser. It is on [this site's front page](../#demo), on [vitalops.ai/lab](https://vitalops.ai/lab/), and in the package:

```
vitalops ui
vitalops ui --port 9000 --no-browser
```

`vitalops ui` serves the page from the package on `127.0.0.1:8765` and opens it in the default browser. The page asks the installed package for every figure over a small JSON API on that port, so Hub models load with your `HF_TOKEN`, gated ones included. It works offline for catalogue models; the hardware photographs are loaded from vitalops.ai when there is a connection and left out otherwise. Ctrl-C stops it. The single-file build embeds the page too, so `python vitalops.py ui` works.

## What is on the panel

- The hardware carousel at the top selects the GPU; it is the only GPU control.
- Model, weight and KV formats, price per GPU-hour, concurrent requests, prompt and output tokens, and the TTFT target on the left; the model list is the full catalogue, and "Another Hugging Face model…" at its end loads any repository id.
- On the right: GPUs needed, per-stream and aggregate tokens per second, TTFT and cost per million output tokens; the GPU's card with its datasheet figures and its memory drawn as its stacks or channels, filled with weights, KV cache and the runtime reserve; the time per output token broken into the weight read, the KV read, the all-reduces and the amortised prefill; and throughput against batch size for one replica, with the planned operating point marked.
- Every control and readout has a short note on hover or keyboard focus that names the formula or the source.
- "Command line" shows the `vitalops` command that prints the same report, and its output.
- "How this is calculated" at the bottom lists every assumption.

## Where the figures come from

On this site the page loads vitalops's own Python source into [Pyodide](https://pyodide.org), CPython compiled to WebAssembly: `docs/engine.js` fetches the package files from `docs/py/vitalops/` (a copy that `tools/site.py` keeps in sync, checked by the tests) and calls `vitalops.web.evaluate` on every change. The first visit downloads about 7 MB; after that the browser cache serves it. Pyodide has no sockets, so for a Hub model the page fetches `config.json` and the checkpoint size itself and hands them to vitalops's loader through its cache. Gated repositories need the command line with `HF_TOKEN`.

Under `vitalops ui` the same `vitalops.web.evaluate` runs in the local server instead. Either way the numbers are the command line's, not a port of them.

vitalops.ai/lab runs a JavaScript port of the same formulas, to avoid loading Pyodide on the company site; the test suite pins one scenario to the figures it shows (Qwen3-32B in fp8 on an H100 with 64 requests of 2,048 in and 300 out: 22.5 tokens per second per stream, 1,437 aggregate, 134 ms TTFT and $0.48 per million output tokens). If Pyodide fails to load, this page falls back to that port and says so.

The command line still has more: `--tp`, `--all-gpus` and JSON output. For anything the page cannot express, use the command line or the [Python API](#python-api).
