"""vitalops: open-source tools for running LLMs in production.

    import vitalops
    p = vitalops.plan("meta-llama/Llama-3.3-70B-Instruct", gpu="h100", users=64)
    print(p.layout.gpus, p.user_tps, p.cost_per_1m_output)

A first-order model from public specifications: decode bound by memory bandwidth, prefill by
compute, tensor parallelism over the real link. The numbers are estimates for planning, not
measurements; see `vitalops plan --help` and the docs for every assumption.
"""
from __future__ import annotations

from typing import Optional, Union

from .hw import GPU, GPUS, gpu as get_gpu
from .models import Model, catalogue, load as load_model, from_hub
from .physics import Scenario, Plan, Layout, Perf, plan as _plan, perf

__version__ = "0.1.0"
__all__ = ["plan", "fit", "Scenario", "Plan", "Layout", "Perf", "GPU", "GPUS", "Model", "get_gpu", "load_model", "from_hub", "catalogue", "perf"]


def _scenario(model: Union[str, Model], gpu: Union[str, GPU], **kw) -> Scenario:
    m = load_model(model, offline=kw.pop("offline", False)) if isinstance(model, str) else model
    g = get_gpu(gpu) if isinstance(gpu, str) else gpu
    return Scenario(model=m, gpu=g, **kw)


def plan(model: Union[str, Model], gpu: Union[str, GPU] = "h100", **kw) -> Plan:
    """Plan a deployment. Keyword arguments: weights ('bf16'|'fp8'|'int4'), kv ('bf16'|'fp8'), users, prompt, output, slo_ms, price, tp, offline."""
    return _plan(_scenario(model, gpu, **kw))


def fit(model: Union[str, Model], gpu: Union[str, GPU] = "h100", **kw) -> bool:
    """Whether the model fits a tensor-parallel group of up to eight of this GPU with one request's KV cache."""
    return plan(model, gpu, **kw).fits
