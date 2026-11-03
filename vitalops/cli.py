"""The vitalops command line.

    vitalops plan meta-llama/Llama-3.3-70B-Instruct --gpu h100 --users 64
    vitalops plan Qwen3-32B --all-gpus
    vitalops bench http://localhost:8000 --users 1,8,32,64 --slo 500 --plan h100
    vitalops gpus
    vitalops models
    vitalops ui
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from . import __version__
from .hw import GPUS, GiB, gpu as get_gpu
from .models import catalogue, load as load_model
from .physics import BW_EFF, FLOP_EFF, PREFILL_EFF, RUNTIME_FRAC, RUNTIME_FIXED, LAT_FAST, LAT_PCIE, Scenario, plan



def gib(b: float) -> str:
    g = b / GiB
    return ("%.0f GiB" if g >= 100 else "%.1f GiB" if g >= 10 else "%.2f GiB" if g >= 1 else "%.0f MiB") % (g if g >= 1 else b / 1048576)


def n(x: float, d: int = 0) -> str:
    return format(x, ",.%df" % d)


def ms(s: float) -> str:
    v = s * 1000
    return ("%.0f ms" if v >= 100 else "%.1f ms" if v >= 10 else "%.2f ms") % v


def tok(x: float) -> str:
    return ("%.1fk" % (x / 1000)) if x >= 1000 else "%d" % x


def report(p, s: Scenario) -> str:
    m, g = s.model, s.gpu
    out = []
    hdr = "vitalops plan · %s on %s" % (m.name, g.name)
    out.append(hdr); out.append("─" * len(hdr))
    out.append("  model      %s params%s · %d layers · KV %s/token (%s)" % (
        ("%.1fB" % (m.total / 1e9)) + (" (%.1fB active)" % (m.active / 1e9) if m.moe else ""), " [" + m.source + "]" if m.source != "catalogue" else "",
        m.layers, (("%.0f KB" % (p.kv_bytes_per_token / 1024)) if p.kv_bytes_per_token >= 1024 else "%d B" % p.kv_bytes_per_token), s.kv))
    out.append("  weights    %s %s · matmuls at %s TFLOPS %s" % (s.weights, gib(p.weight_bytes), n(p.rate_tflops), p.rate_dtype))
    out.append("  traffic    %d concurrent requests × (%s in + %s out) · TTFT target %.0f ms" % (s.users, n(s.prompt), n(s.output), s.slo_ms))
    if not p.fits:
        out.append("")
        out.append("  does not fit: %s." % p.reason)
        return "\n".join(out)
    L, pf = p.layout, p.layout.perf
    out.append("")
    out.append("  layout     TP%d × %d replica%s = %d GPU%s · batch %d per replica (max %d)" % (L.tp, L.replicas, "s" if L.replicas != 1 else "", L.gpus, "s" if L.gpus != 1 else "", L.batch, L.max_batch))
    out.append("  decode     %.1f tok/s per stream · %s tok/s aggregate · %.2f requests/s" % (p.user_tps, n(p.agg_tps), p.requests_per_s))
    out.append("  TTFT       %s = prefill %s + one step %s · target %s" % (ms(pf.ttft_s), ms(pf.prefill_s), ms(pf.step_s), "met" if L.slo_ok else "MISSED"))
    comm = (" + all-reduces %s (%.0f%%)" % (ms(pf.t_comm_s), 100 * pf.t_comm_s / pf.step_s)) if pf.t_comm_s > 0 else ""
    bound = "memory-bandwidth bound" if pf.memory_bound else "compute bound"
    out.append("  step       %s = weights %s + KV %s%s · %s (compute floor %s)" % (ms(pf.step_s), ms(pf.t_weights_s), ms(pf.t_kv_s), comm, bound, ms(pf.t_compute_s)))
    free = p.mem_bytes - L.weights_per_gpu - L.kv_per_gpu - L.runtime_per_gpu
    out.append("  memory/GPU weights %s · KV %s · runtime %s · free %s of %s" % (gib(L.weights_per_gpu), gib(L.kv_per_gpu), gib(L.runtime_per_gpu), gib(free), gib(p.mem_bytes)))
    if s.gpu_price > 0:
        out.append("  cost       $%.2f per 1M output tokens · $%s / month at $%.2f per GPU-hour" % (p.cost_per_1m_output, n(p.cost_per_month), s.gpu_price))
    if p.reason:
        out.append("  note       " + p.reason)
    if m.note:
        out.append("  note       " + m.note)
    if g.note:
        out.append("  note       " + g.note)
    return "\n".join(out)


def table_all(m, s_kw: dict) -> str:
    rows = [("GPU", "GPUs", "TP", "tok/s/user", "aggregate", "TTFT", "$/1M out", "")]
    for key, g in GPUS.items():
        if key == "agx_orin":
            continue
        p = plan(Scenario(model=m, gpu=g, **s_kw))
        if p.fits:
            L = p.layout
            rows.append((g.name, str(L.gpus), str(L.tp), "%.1f" % p.user_tps, n(p.agg_tps), ms(L.perf.ttft_s), ("$%.2f" % p.cost_per_1m_output) if s_kw.get("price") is None or s_kw["price"] > 0 else "–", "" if L.slo_ok else "misses TTFT target"))
        else:
            rows.append((g.name, "–", "–", "–", "–", "–", "–", "does not fit up to TP8"))
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = []
    for k, r in enumerate(rows):
        lines.append("  " + "  ".join((r[i].ljust(w[i]) if i in (0, 7) else r[i].rjust(w[i])) for i in range(len(r))).rstrip())
        if k == 0:
            lines.append("  " + "─" * (sum(w) + 2 * (len(w) - 1)))
    return "\n".join(lines)


def assumptions() -> str:
    return "\n".join([
        "  Assumptions: decode reads the weights (the experts a batch touches, for a mixture of experts) plus every running request's KV cache at %d%% of peak bandwidth," % round(BW_EFF * 100),
        "  with a compute floor at %d%% of the dense tensor-core rate; prefill at %d%% of that rate; %d%% of memory + %.1f GB reserved for the runtime;" % (round(FLOP_EFF * 100), round(PREFILL_EFF * 100), round(RUNTIME_FRAC * 100), RUNTIME_FIXED / GiB),
        "  two all-reduces per layer for tensor parallelism over the GPU's link with %d µs latency on NVLink or Infinity Fabric and %d µs on PCIe; no queueing;" % (LAT_FAST * 1e6, LAT_PCIE * 1e6),
        "  the KV cache is reserved for prompt + output tokens for the whole request. Estimates for planning, not measurements.",
    ])


def cmd_plan(a) -> int:
    try:
        m = load_model(a.model, offline=a.offline, revision=a.revision)
    except LookupError as e:
        print("vitalops: %s" % e, file=sys.stderr); return 2
    except Exception as e:  # network
        print("vitalops: could not read %s from the Hub (%s); try --offline with a catalogue name" % (a.model, e), file=sys.stderr); return 2
    s_kw = dict(weights=a.weights, kv=a.kv, users=a.users, prompt=a.prompt, output=a.output, slo_ms=a.slo, price=a.price, tp=a.tp)
    if a.all_gpus:
        if a.json:
            out = {k: plan(Scenario(model=m, gpu=g, **s_kw)).to_dict() for k, g in GPUS.items() if k != "agx_orin"}
            print(json.dumps(out, indent=2)); return 0
        print("vitalops plan · %s · %s weights, %s KV · %d × (%s in + %s out) · TTFT target %.0f ms" % (m.name, a.weights, a.kv, a.users, n(a.prompt), n(a.output), a.slo))
        print(table_all(m, s_kw))
        if not a.quiet:
            print(); print(assumptions())
        return 0
    try:
        g = get_gpu(a.gpu)
    except KeyError as e:
        print("vitalops: %s" % e.args[0], file=sys.stderr); return 2
    s = Scenario(model=m, gpu=g, **s_kw)
    p = plan(s)
    if a.json:
        d = p.to_dict(); d["model_shapes"] = m.to_dict(); print(json.dumps(d, indent=2)); return 0
    print(report(p, s))
    if not a.quiet:
        print(); print(assumptions())
    return 0 if p.fits else 1


def cmd_gpus(a) -> int:
    if a.json:
        print(json.dumps({k: asdict(g) for k, g in GPUS.items()}, indent=2)); return 0
    print("  %-8s %-22s %-13s %-8s %-30s %-21s %s" % ("key", "name", "memory", "bw TB/s", "dense TFLOPS", "link", "$/h"))
    for k, g in GPUS.items():
        fl = " ".join("%s %s" % (d, n(v)) for d, v in g.flops.items() if d in ("bf16", "fp8", "fp4"))
        print("  %-8s %-22s %-13s %-8s %-30s %-21s %.2f" % (k, g.name, "%d GB %s" % (g.mem, g.mem_type), "%g" % g.bw, fl, "%s %d GB/s" % (g.link_name, g.link_gbps), g.price))
    return 0


def cmd_models(a) -> int:
    if a.json:
        print(json.dumps([m.to_dict() for m in catalogue()], indent=2)); return 0
    print("  %-46s %-26s %-16s %s" % ("id", "name", "params", "layers · KV heads × head dim"))
    for m in catalogue():
        pr = "%.1fB" % (m.total / 1e9) + (" / %.1fB act" % (m.active / 1e9) if m.moe else "")
        print("  %-46s %-26s %-16s %d · %d × %d%s" % (m.id, m.name, pr, m.layers, m.kv_heads, m.head_dim, " (MLA %d)" % m.kv_per_layer if m.kv_per_layer else ""))
    print("\n  Any other Hugging Face repository id works too; its config.json is read from the Hub (set HF_TOKEN for gated models).")
    return 0


def _f(x: float, d: int = 1) -> str:
    return "–" if x != x else n(x, d)


def bench_report(run) -> str:
    from .bench import explain
    hdr = "vitalops bench · %s at %s" % (run.model, run.url)
    out = [hdr, "─" * len(hdr),
           "  traffic    %s in + %s out per request%s · /v1/%s · TTFT target %.0f ms (p95)" % (
               n(run.prompt), n(run.output), " · first %s tokens shared" % n(run.shared_prefix) if run.shared_prefix else "", run.api, run.slo_ms), ""]
    planned = any(lv.predicted for lv in run.levels)
    rows = [("users", "requests", "req/s", "tok/s all", "tok/s/user", "TTFT p50", "TTFT p95", "ITL p50", "ITL p95", "")]
    for lv in run.levels:
        rows.append((str(lv.users), str(lv.requests) + (" (%d failed)" % lv.errors if lv.errors else ""), _f(lv.requests_per_s, 2), _f(lv.output_tps, 0),
                     _f(lv.user_tps), _f(lv.ttft_ms_p50, 0) + " ms", _f(lv.ttft_ms_p95, 0) + " ms", _f(lv.itl_ms_p50) + " ms", _f(lv.itl_ms_p95) + " ms",
                     "" if lv.slo_ok else "over target"))
        p = lv.predicted
        if planned:
            if p and p.get("fits"):
                rows.append(("", "planned", _f(p["requests_per_s"], 2), _f(p["output_tps"], 0), _f(p["user_tps"]), _f(p["ttft_ms"], 0) + " ms", "", _f(p["step_ms"]) + " ms", "", ""))
            elif p:
                rows.append(("", "planned", "", "", "", "", "", "", "", p.get("reason", "")))
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for k, r in enumerate(rows):
        out.append("  " + "  ".join((r[i].ljust(w[i]) if i in (1, 9) else r[i].rjust(w[i])) for i in range(len(r))).rstrip())
        if k == 0:
            out.append("  " + "─" * (sum(w) + 2 * (len(w) - 1)))
    out.append("")
    if run.capacity:
        out.append("  capacity   %d concurrent requests meet the %.0f ms p95 TTFT target (the most tested that did)" % (run.capacity, run.slo_ms))
    else:
        out.append("  capacity   no tested concurrency met the %.0f ms p95 TTFT target" % run.slo_ms)
    if run.plan:
        out.append("  planned    one TP%d replica on %s, from `vitalops plan`; no queueing or prefix caching in the estimate" % (run.plan["tp"], get_gpu(run.plan["gpu"]).name))
    for line in explain(run):
        out.append("  gap        " + line)
    for line in run.notes:
        out.append("  note       " + line)
    return "\n".join(out)


def cmd_bench(a) -> int:
    from .bench import BenchError, bench
    try:
        users = [int(x) for x in a.users.split(",") if x.strip()]
        if not users or min(users) < 1:
            raise ValueError
    except ValueError:
        print("vitalops: --users takes a comma-separated list of positive integers, e.g. 1,8,32", file=sys.stderr); return 2
    if a.plan:
        try:
            get_gpu(a.plan)
        except KeyError as e:
            print("vitalops: %s" % e.args[0], file=sys.stderr); return 2
    say = (lambda msg: None) if a.json or a.quiet else (lambda msg: print("  … " + msg, file=sys.stderr))
    try:
        run = bench(a.url, users=users, prompt=a.prompt, output=a.output, shared_prefix=a.shared_prefix, requests=a.requests, slo_ms=a.slo,
                    model=a.model, api=a.api, seed=a.seed, plan_gpu=a.plan, plan_model=a.plan_model, tp=a.tp,
                    scenario_kw=dict(weights=a.weights, kv=a.kv), progress=say)
    except BenchError as e:
        print("vitalops: %s" % e, file=sys.stderr); return 2
    d = run.to_dict()
    if a.save:
        with open(a.save, "w") as f:
            json.dump(d, f, indent=2)
    if a.json:
        print(json.dumps(d, indent=2))
    else:
        print(bench_report(run))
    if a.require is not None:
        lv = [x for x in run.levels if x.users == a.require]
        if not lv or not lv[0].slo_ok:
            return 1
    return 0


def cmd_ui(a) -> int:
    from .ui import serve
    return serve(a.port, a.no_browser)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vitalops", description="Open-source tools for running LLMs in production.")
    ap.add_argument("--version", action="version", version="vitalops " + __version__)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("plan", help="will the model fit, how fast, how many GPUs, what will it cost: first-order estimates from public specifications")
    p.add_argument("model", help="Hugging Face repository id (meta-llama/Llama-3.1-8B-Instruct) or a catalogue name (llama-70b, qwen3-32b, ...)")
    p.add_argument("--gpu", default="h100", help="GPU key, see `vitalops gpus` (default h100)")
    p.add_argument("--all-gpus", action="store_true", help="compare every GPU in the catalogue")
    p.add_argument("--weights", choices=("bf16", "fp8", "int4"), default="fp8", help="weight format (default fp8)")
    p.add_argument("--kv", choices=("bf16", "fp8"), default="fp8", help="KV cache format (default fp8)")
    p.add_argument("--users", type=int, default=64, help="concurrent requests (default 64)")
    p.add_argument("--prompt", type=int, default=2048, help="prompt tokens per request (default 2048)")
    p.add_argument("--output", type=int, default=300, help="output tokens per request (default 300)")
    p.add_argument("--slo", type=float, default=500, help="time-to-first-token target in ms (default 500)")
    p.add_argument("--price", type=float, default=None, help="dollars per GPU-hour (default: the catalogue's figure)")
    p.add_argument("--tp", type=int, choices=(1, 2, 4, 8), default=None, help="force a tensor-parallel degree (default: search 1-8)")
    p.add_argument("--revision", default="main", help="Hub revision (default main)")
    p.add_argument("--offline", action="store_true", help="never contact the Hub; catalogue names only")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("-q", "--quiet", action="store_true", help="omit the assumptions footer")
    p.set_defaults(fn=cmd_plan)
    g = sub.add_parser("gpus", help="list the GPU catalogue"); g.add_argument("--json", action="store_true"); g.set_defaults(fn=cmd_gpus)
    mo = sub.add_parser("models", help="list the built-in model catalogue"); mo.add_argument("--json", action="store_true"); mo.set_defaults(fn=cmd_models)
    b = sub.add_parser("bench", help="measure an OpenAI-compatible endpoint across concurrency levels, optionally against the plan")
    b.add_argument("url", help="server base URL, e.g. http://localhost:8000 (the /v1 is added when missing)")
    b.add_argument("--users", default="1,4,16,64", help="concurrency levels, comma-separated (default 1,4,16,64)")
    b.add_argument("--prompt", type=int, default=1024, help="prompt tokens per request (default 1024)")
    b.add_argument("--output", type=int, default=256, help="output tokens per request (default 256)")
    b.add_argument("--shared-prefix", type=int, default=0, help="tokens at the start of every prompt that are the same across requests, like a system prompt (default 0)")
    b.add_argument("--requests", type=int, default=16, help="requests measured per level, at least twice the concurrency (default 16)")
    b.add_argument("--slo", type=float, default=500, help="p95 time-to-first-token target in ms (default 500)")
    b.add_argument("--model", default=None, help="model id to request (default: the first one /v1/models lists)")
    b.add_argument("--api", choices=("completions", "chat"), default=None, help="endpoint to use (default: completions, else chat)")
    b.add_argument("--plan", metavar="GPU", default=None, help="also run `vitalops plan` for one replica on this GPU and print it under each level")
    b.add_argument("--plan-model", default=None, help="Hugging Face id for the plan when the served name is not one (default: the served model id)")
    b.add_argument("--tp", type=int, choices=(1, 2, 4, 8), default=1, help="tensor-parallel degree of the server, for the plan (default 1)")
    b.add_argument("--weights", choices=("bf16", "fp8", "int4"), default="fp8", help="weight format of the server, for the plan (default fp8)")
    b.add_argument("--kv", choices=("bf16", "fp8"), default="fp8", help="KV cache format of the server, for the plan (default fp8)")
    b.add_argument("--require", type=int, default=None, metavar="USERS", help="exit 1 unless the p95 TTFT target is met at this concurrency (one of --users)")
    b.add_argument("--seed", type=int, default=0, help="random seed for the prompts (default 0)")
    b.add_argument("--save", metavar="FILE", default=None, help="write the run as JSON to FILE")
    b.add_argument("--json", action="store_true", help="machine-readable output")
    b.add_argument("-q", "--quiet", action="store_true", help="no progress lines")
    b.set_defaults(fn=cmd_bench)
    u = sub.add_parser("ui", help="open the interactive planner in a browser (served locally)")
    u.add_argument("--port", type=int, default=8765); u.add_argument("--no-browser", action="store_true"); u.set_defaults(fn=cmd_ui)
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_parser()
    a = ap.parse_args(argv)
    if not getattr(a, "fn", None):
        ap.print_help(); return 0
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
