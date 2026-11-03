# Install

tokplan is pure Python with no dependencies. It needs Python 3.9 or newer and nothing else.

## pip

```
pip install tokplan
tokplan --version
```

This installs the `tokplan` command and the `tokplan` package.

## One file

The whole package is also published as a single file that runs as a script or imports as a module:

```
wget https://github.com/vitalops/tokplan/releases/download/onefile/tokplan.py
python tokplan.py qwen3-32b
```

Put `tokplan.py` on `sys.path` and `import tokplan` works the same as the package. The interactive planner is embedded too, so `python tokplan.py ui` works from the one file.

## Check it works

```
$ tokplan qwen3-32b --offline -q
tokplan · Qwen3-32B on H100 SXM
───────────────────────────────
  model      32.8B params · 64 layers · KV 128 KB/token (fp8)
  ...
```

`--offline` uses the built-in catalogue and never contacts the Hub, so it works without a network. Without the flag, a Hugging Face repository id is looked up on the Hub; for a gated repository set `HF_TOKEN` to a token that can read it.

## From source

```
git clone https://github.com/vitalops/tokplan && cd tokplan
pip install -e .[dev]
pytest
```

See [Contributing](#contributing) for the layout of the package and the single-file build.
