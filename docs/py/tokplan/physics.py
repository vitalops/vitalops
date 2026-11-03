"""The first-order model of inference on one GPU, or a tensor-parallel group of them.

Decode is bound by memory bandwidth: a step reads the weights (for a mixture of experts, the experts
the batch touches) plus the KV cache of every running request, with a compute floor from the
matmuls. Prefill is compute bound. Tensor parallelism adds two all-reduces per layer over the real
link. Everything else is an explicit assumption below, and the report says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

from .hw import GPU, GiB
from .models import Model

BW_EFF = 0.75          # share of peak memory bandwidth a decode step achieves
FLOP_EFF = 0.50        # share of the dense tensor-core rate the decode matmuls achieve
PREFILL_EFF = 0.60     # share of the dense rate prefill achieves
RUNTIME_FRAC = 0.08    # memory reserved for activations, CUDA graphs and allocator slack ...
RUNTIME_FIXED = 1.5 * GiB   # ... plus this much
LAT_FAST = 8e-6        # all-reduce latency on NVLink / Infinity Fabric, s
LAT_PCIE = 25e-6       # all-reduce latency on PCIe, s
MAX_BATCH = 512
TPS = (1, 2, 4, 8)
WEIGHT_BYTES = {"bf16": 2.0, "fp8": 1.0, "int4": 0.5625}   # int4 includes group scales
KV_BYTES = {"bf16": 2.0, "fp8": 1.0}


@dataclass
class Scenario:
    model: Model
    gpu: GPU
    weights: str = "fp8"
    kv: str = "fp8"
    users: int = 64
    prompt: int = 2048
    output: int = 300
    slo_ms: float = 500.0
    price: Optional[float] = None      # $ per GPU-hour; the GPU's default when None
    tp: Optional[int] = None           # force a tensor-parallel degree; search when None

    @property
    def wb(self) -> float:
        return WEIGHT_BYTES[self.weights]

    @property
    def kvb(self) -> float:
        return KV_BYTES[self.kv]

    @property
    def gpu_price(self) -> float:
        return self.gpu.price if self.price is None else self.price


@dataclass
class Perf:
    step_s: float           # one decode step at the operating point
    t_weights_s: float
    t_kv_s: float
    t_compute_s: float
    t_comm_s: float
    prefill_s: float        # one prompt, including its all-reduces
    ttft_s: float
    user_tps: float         # tokens per second per stream, prefill amortised
    agg_tps: float          # per replica
    memory_bound: bool


@dataclass
class Layout:
    tp: int
    replicas: int
    gpus: int
    batch: int              # requests per replica
    max_batch: int          # the most the KV cache allows at this layout
    weights_per_gpu: float  # bytes
    kv_per_gpu: float       # bytes, for this batch
    runtime_per_gpu: float  # bytes
    kv_cap: float           # bytes of KV cache available across the TP group
    perf: Perf
    slo_ok: bool


@dataclass
class Plan:
    fits: bool
    model: str
    gpu: str
    weights: str
    kv: str
    rate_tflops: float
    rate_dtype: str
    users: int
    prompt: int
    output: int
    slo_ms: float
    weight_bytes: float
    kv_bytes_per_token: float
    kv_bytes_per_request: float
    mem_bytes: float
    layout: Optional[Layout]
    options: List[Layout]
    agg_tps: float = 0.0
    user_tps: float = 0.0
    ttft_ms: float = 0.0
    requests_per_s: float = 0.0
    cost_per_1m_output: float = 0.0
    cost_per_month: float = 0.0
    price_per_gpu_hour: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def kv_bytes_per_token(m: Model, s: Scenario) -> float:
    return m.kv_bytes_per_token(s.kvb)


def weights_read(m: Model, wb: float, batch: int) -> float:
    """Weight bytes a decode step reads: all of them for a dense model; for a mixture of experts, the experts a batch of that size touches."""
    if m.active >= m.total:
        return m.total * wb
    p = m.active / m.total
    return m.total * wb * (1 - (1 - p) ** batch)


def comm_time(m: Model, g: GPU, tp: int, tokens: int) -> float:
    """Tensor-parallel all-reduces: two per layer; a ring all-reduce moves 2(TP-1)/TP of the activation tensor through each GPU's link."""
    if tp < 2:
        return 0.0
    bytes_ = tokens * m.hidden * 2 * 2 * (tp - 1) / tp
    lat = LAT_PCIE if g.pcie else LAT_FAST
    return m.layers * 2 * (bytes_ / (g.link_gbps * 1e9) + lat)


def perf(s: Scenario, tp: int, batch: int) -> Perf:
    m, g = s.model, s.gpu
    tf, _ = g.rate(s.weights)
    kv_tok = kv_bytes_per_token(m, s)
    ctx_avg = s.prompt + s.output / 2
    bw = g.bw * 1e12 * BW_EFF * tp
    F = tf * 1e12 * tp
    t_w = weights_read(m, s.wb, batch) / bw
    t_kv = batch * ctx_avg * kv_tok / bw
    t_mem = t_w + t_kv
    t_cmp = 2 * m.active * batch / (F * FLOP_EFF)
    t_comm = comm_time(m, g, tp, batch)
    step = max(t_mem, t_cmp) + t_comm
    r = 1 / step
    t_prompt = s.prompt * 2 * m.active / (F * PREFILL_EFF) + comm_time(m, g, tp, s.prompt)
    k = batch * t_prompt / s.output
    r_eff = r / (1 + k * r)
    return Perf(step_s=step, t_weights_s=t_w, t_kv_s=t_kv, t_compute_s=t_cmp, t_comm_s=t_comm, prefill_s=t_prompt,
                ttft_s=t_prompt + step, user_tps=r_eff, agg_tps=r_eff * batch, memory_bound=t_mem >= t_cmp)


def plan(s: Scenario) -> Plan:
    m, g = s.model, s.gpu
    tf, dtype = g.rate(s.weights)
    mem = g.mem * GiB
    runtime = RUNTIME_FRAC * mem + RUNTIME_FIXED
    weights = m.total * s.wb
    kv_tok = kv_bytes_per_token(m, s)
    per_req = kv_tok * (s.prompt + s.output)
    base = dict(model=m.name, gpu=g.name, weights=s.weights, kv=s.kv, rate_tflops=tf, rate_dtype=dtype, users=s.users, prompt=s.prompt, output=s.output,
                slo_ms=s.slo_ms, weight_bytes=weights, kv_bytes_per_token=kv_tok, kv_bytes_per_request=per_req, mem_bytes=mem, price_per_gpu_hour=s.gpu_price)
    options: List[Layout] = []
    for tp in (TPS if s.tp is None else (s.tp,)):
        w_per = weights / tp
        kv_cap = (mem - runtime - w_per) * tp
        if kv_cap <= 0:
            continue
        max_b = min(MAX_BATCH, int(kv_cap // per_req))
        if max_b < 1:
            continue
        replicas = -(-s.users // max_b)
        b = -(-s.users // replicas)
        p = perf(s, tp, b)
        options.append(Layout(tp=tp, replicas=replicas, gpus=tp * replicas, batch=b, max_batch=max_b, weights_per_gpu=w_per, kv_per_gpu=b * per_req / tp,
                              runtime_per_gpu=runtime, kv_cap=kv_cap, perf=p, slo_ok=p.ttft_s * 1000 <= s.slo_ms))
    if not options:
        need = weights + runtime + per_req
        tps_max = max(TPS) if s.tp is None else s.tp
        reason = ("%s weights are %.1f GiB; with the runtime reserve and one request's KV cache (%.1f GiB) that exceeds %d × %.0f GB"
                  % (s.weights, weights / GiB, need / GiB, tps_max, g.mem))
        return Plan(fits=False, layout=None, options=[], reason=reason, **base)
    ok = [o for o in options if o.slo_ok]
    if ok:
        best = min(ok, key=lambda o: (o.gpus, o.tp))
    else:
        best = min(options, key=lambda o: o.perf.ttft_s)
    agg = best.perf.agg_tps * best.replicas
    hours = best.gpus * s.gpu_price
    cost_1m = hours / (agg * 3600 / 1e6) if agg > 0 else 0.0
    return Plan(fits=True, layout=best, options=options, agg_tps=agg, user_tps=best.perf.user_tps, ttft_ms=best.perf.ttft_s * 1000,
                requests_per_s=agg / s.output, cost_per_1m_output=cost_1m, cost_per_month=hours * 24 * 30.4,
                reason="" if best.slo_ok else "no layout up to TP%d meets the TTFT target; the fastest is shown" % max(o.tp for o in options), **base)
