# Install

vitalops is pure Python with no dependencies. It needs Python 3.9 or newer and nothing else.

## pip

```
pip install vitalops
vitalops --version
```

This installs the `vitalops` command and the `vitalops` package.

## One file

The whole package is also published as a single file that runs as a script or imports as a module:

```
wget https://github.com/vitalops/vitalops/releases/download/onefile/vitalops.py
python vitalops.py plan qwen3-32b
```

Put `vitalops.py` on `sys.path` and `import vitalops` works the same as the package. The interactive planner is embedded too, so `python vitalops.py ui` works from the one file.

## Check it works

```
$ vitalops plan qwen3-32b --offline -q
vitalops plan · Qwen3-32B on H100 SXM
─────────────────────────────────────
  model      32.8B params · 64 layers · KV 128 KB/token (fp8)
  ...
```

`--offline` uses the built-in catalogue and never contacts the Hub, so it works without a network. Without the flag, a Hugging Face repository id is looked up on the Hub; for a gated repository set `HF_TOKEN` to a token that can read it.

## From source

```
git clone https://github.com/vitalops/vitalops && cd vitalops
pip install -e .[dev]
pytest
```

See [Contributing](#contributing) for the layout of the package and the single-file build.
