# Contributing

The repository is [github.com/vitalops/vitalops](https://github.com/vitalops/vitalops), MIT licensed. Corrections to a catalogue figure, new hardware, new models and fixes to the model of inference are welcome as pull requests.

## Layout

| Path | What it is |
|---|---|
| `vitalops/hw.py` | the GPU catalogue and the `GPU` dataclass |
| `vitalops/models.py` | the model catalogue, the Hub loader, the `Model` dataclass |
| `vitalops/physics.py` | the model of inference: `Scenario`, `perf`, `plan`, the constants |
| `vitalops/cli.py` | the command line |
| `vitalops/web.py` | what the interactive page calls: `evaluate(state)` returns the plan, the report and the chart |
| `vitalops/ui.py`, `vitalops/ui/` | `vitalops ui`: a local server with a JSON API over `web.py`, and the page it serves |
| `tests/` | pytest |
| `tools/onefile.py` | the single-file build |
| `tools/site.py` | copies the package into `docs/py/` and the page's scripts into `docs/`; run it after a change |
| `docs/` | this site, served by GitHub Pages |

## Running the tests

```
git clone https://github.com/vitalops/vitalops && cd vitalops
pip install -e .[dev]
pytest
```

The tests cover the physics against the website's figures, the layout search, the fit check, the MoE weight read, the parameter count from config shapes, and the command line's JSON and exit codes. They run offline.

## Adding a GPU

Add an entry to `GPUS` in `vitalops/hw.py`:

```python
"h100": _g(key="h100", name="H100 SXM", vendor="nvidia", arch="sm_90", sms=132, mem=80, mem_type="HBM3", stacks=5, stack_gb=16, bw=3.35,
           flops={"fp32": 67, "bf16": 989, "fp8": 1979, "int8": 1979}, link_name="NVLink 4", link_gbps=450, tdp=700, price=2.50),
```

Every figure except `price` and `latency_ns` must come from the vendor's datasheet; put the link in the pull request. `flops` holds dense rates (not the sparse ones datasheets headline), per format the tensor cores support; leave out a format the GPU does not have, because the planner uses the list to decide which rate the matmuls run at. `link_gbps` is per direction. `mem` is the vendor's usable figure. Name PCIe links with the word `PCIe`, because the all-reduce latency is chosen from the name. Add an alias to `ALIASES` if people call it something shorter. The same figures go into `hw.js` for the web page; keep the two in step.

## Adding a model

Add an entry to `CATALOGUE` in `vitalops/models.py`:

```python
_m("Qwen/Qwen3-32B", "Qwen3-32B", 32.8e9, 32.8e9, 64, 8, 128, 5120),
_m("Qwen/Qwen3-235B-A22B", "Qwen3-235B-A22B (MoE)", 235e9, 22e9, 94, 4, 128, 4096, experts=128, top_k=8),
```

The arguments are id, name, total parameters, active parameters, layers, KV heads, head dimension, hidden size, then `kv_per_layer` for latent attention, `experts` and `top_k` for a mixture of experts, and a `note` for a caveat such as sliding-window layers. Take the shapes from the repository's `config.json` and the parameter count from the checkpoint size; a model that is not in the catalogue already works through the Hub loader, so the catalogue is for offline use and for the names people type. Add aliases to `_ALIASES`.

## Changing the physics

The constants are at the top of `vitalops/physics.py`. A change to them or to a formula should come with the measurement that motivates it, and the tests that pin the website's figures will need the same change in the JavaScript port in `lab-planner.js` that vitalops.ai uses. This site and `vitalops ui` call the package itself, so they follow automatically; run `python tools/site.py` to refresh `docs/py/`.

## The single-file build

```
python tools/onefile.py
python dist/vitalops.py qwen3-32b --offline
```

`tools/onefile.py` concatenates the modules in dependency order, strips the relative imports and embeds the UI page, and writes `dist/vitalops.py`. CI runs it on every push; a release attaches it as `vitalops.py`.

## Style

Plain Python 3.9, standard library only, no dependencies at runtime. Dataclasses for data, functions for the maths, no classes with behaviour. British English in text. Numbers on screen are estimates and the wording should say so where it matters.
