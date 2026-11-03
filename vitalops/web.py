"""The bridge the interactive planner calls: the docs site loads this package into Pyodide and calls
it directly; `vitalops ui` calls it over HTTP on localhost. State in, plain dicts (or JSON) out.
"""
from __future__ import annotations

import json
import math
from typing import Optional

from . import __version__
from .cli import report
from .hw import GiB, gpu as get_gpu
from .models import catalogue, load as load_model, _store
from .physics import RUNTIME_FIXED, RUNTIME_FRAC, Scenario, perf, plan

DEFAULTS = {"weights": "fp8", "kv": "fp8", "users": 64, "prompt": 2048, "output": 300, "slo": 500}
CHART_POINTS = 48


def model_list() -> list:
    return [{"id": m.id, "name": m.name, "moe": m.moe} for m in catalogue()]


def model_info(name: str, offline: bool = False) -> dict:
    m = load_model(name, offline=offline)
    d = m.to_dict()
    d["moe"] = m.moe
    return d


def preload(repo: str, config: dict, total_bytes: Optional[int] = None) -> None:
    """Seeds the Hub cache with a config fetched elsewhere (the browser fetches it, since Pyodide has no sockets)."""
    _store(repo, {"repo": repo, "config": config, "total_bytes": total_bytes})


def command(st: dict) -> str:
    """The command line that prints the same report."""
    parts = ["vitalops", "plan", st["model"], "--gpu", st["gpu"]]
    for k in ("weights", "kv", "users", "prompt", "output", "slo"):
        v = st.get(k, DEFAULTS[k])
        if v != DEFAULTS[k]:
            parts += ["--" + k, "%g" % v if isinstance(v, float) else str(v)]
    price = st.get("price")
    if price is not None and abs(price - get_gpu(st["gpu"]).price) > 1e-9:
        parts += ["--price", "%.2f" % price]
    return " ".join(parts)


def evaluate(st: dict) -> dict:
    """Plan one scenario: the plan, the CLI report, the equivalent command and the throughput curve of the chosen layout."""
    try:
        m = load_model(st["model"], offline=bool(st.get("offline")))
        g = get_gpu(st["gpu"])
    except (LookupError, KeyError) as e:
        return {"error": str(e.args[0] if e.args else e)}
    s = Scenario(model=m, gpu=g, weights=st.get("weights", "fp8"), kv=st.get("kv", "fp8"), users=int(st.get("users", 64)),
                 prompt=int(st.get("prompt", 2048)), output=int(st.get("output", 300)), slo_ms=float(st.get("slo", 500)), price=st.get("price"))
    p = plan(s)
    out = {"version": __version__, "plan": p.to_dict(), "report": report(p, s), "command": command(st), "model": dict(m.to_dict(), moe=m.moe),
           "runtime_bytes": RUNTIME_FRAC * g.mem * GiB + RUNTIME_FIXED, "chart": []}
    if p.fits:
        L = p.layout
        top = max(2, L.max_batch)
        prev = 0
        for i in range(CHART_POINTS + 1):
            b = max(1, int(math.floor(math.exp(math.log(top) * i / CHART_POINTS) + 0.5)))
            if b != prev:
                pf = perf(s, L.tp, b)
                out["chart"].append({"b": b, "agg": pf.agg_tps, "user": pf.user_tps})
                prev = b
    return out


# JSON in, JSON out: the form Pyodide and the local server pass across
def evaluate_json(state: str) -> str:
    return json.dumps(evaluate(json.loads(state)))


def model_list_json() -> str:
    return json.dumps({"version": __version__, "models": model_list()})


def model_info_json(name: str) -> str:
    try:
        return json.dumps(model_info(name))
    except Exception as e:
        return json.dumps({"error": str(e.args[0] if e.args else e)})


def preload_json(repo: str, payload: str) -> None:
    d = json.loads(payload)
    preload(repo, d["config"], d.get("total_bytes"))
