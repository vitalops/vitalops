"""Model shapes: a built-in catalogue of popular open-weight models, and a loader that reads any
Hugging Face repository's config.json (and the safetensors index for the exact parameter count).

Only the shapes the planner needs are kept: parameter counts (total and active per token), layers,
KV heads, head dimension, hidden size, and the per-layer KV width for models with latent attention.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

HUB = "https://huggingface.co"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "vitalops")
CACHE_TTL = 7 * 86400
_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "float8_e4m3fn": 1, "float8_e5m2": 1}


@dataclass
class Model:
    id: str                     # Hugging Face repository id, or a catalogue name
    name: str
    total: float                # parameters, all of them
    active: float               # parameters a token passes through (equal to total for dense models)
    layers: int
    kv_heads: int
    head_dim: int
    hidden: int
    kv_per_layer: Optional[int] = None   # KV cache width per token per layer when it is not 2 × kv_heads × head_dim (MLA)
    experts: int = 0
    top_k: int = 0
    source: str = "catalogue"
    note: str = ""

    @property
    def moe(self) -> bool:
        return self.active < self.total

    def kv_width(self) -> int:
        """KV cache elements per token per layer."""
        return self.kv_per_layer if self.kv_per_layer else 2 * self.kv_heads * self.head_dim

    def kv_bytes_per_token(self, kv_bytes: float) -> float:
        return self.layers * self.kv_width() * kv_bytes

    def to_dict(self) -> dict:
        return asdict(self)


def _m(id, name, total, active, layers, kv_heads, head_dim, hidden, kv_per_layer=None, experts=0, top_k=0, note="") -> Model:
    return Model(id=id, name=name, total=total, active=active, layers=layers, kv_heads=kv_heads, head_dim=head_dim, hidden=hidden,
                 kv_per_layer=kv_per_layer, experts=experts, top_k=top_k, note=note)


# Parameter counts are from the published checkpoints (safetensors sizes) or the model cards; shapes from config.json.
CATALOGUE: List[Model] = [
    _m("meta-llama/Llama-3.2-1B-Instruct", "Llama-3.2-1B", 1.24e9, 1.24e9, 16, 8, 64, 2048),
    _m("meta-llama/Llama-3.2-3B-Instruct", "Llama-3.2-3B", 3.21e9, 3.21e9, 28, 8, 128, 3072),
    _m("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B", 8.03e9, 8.03e9, 32, 8, 128, 4096),
    _m("meta-llama/Llama-3.3-70B-Instruct", "Llama-3.3-70B", 70.6e9, 70.6e9, 80, 8, 128, 8192),
    _m("Qwen/Qwen3-4B", "Qwen3-4B", 4.02e9, 4.02e9, 36, 8, 128, 2560),
    _m("Qwen/Qwen3-8B", "Qwen3-8B", 8.19e9, 8.19e9, 36, 8, 128, 4096),
    _m("Qwen/Qwen3-14B", "Qwen3-14B", 14.8e9, 14.8e9, 40, 8, 128, 5120),
    _m("Qwen/Qwen3-32B", "Qwen3-32B", 32.8e9, 32.8e9, 64, 8, 128, 5120),
    _m("Qwen/Qwen3-30B-A3B", "Qwen3-30B-A3B (MoE)", 30.5e9, 3.3e9, 48, 4, 128, 2048, experts=128, top_k=8),
    _m("Qwen/Qwen3-235B-A22B", "Qwen3-235B-A22B (MoE)", 235e9, 22e9, 94, 4, 128, 4096, experts=128, top_k=8),
    _m("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B", 7.62e9, 7.62e9, 28, 4, 128, 3584),
    _m("Qwen/Qwen2.5-72B-Instruct", "Qwen2.5-72B", 72.7e9, 72.7e9, 80, 8, 128, 8192),
    _m("mistralai/Mistral-Small-3.1-24B-Instruct-2503", "Mistral-Small-24B", 23.6e9, 23.6e9, 40, 8, 128, 5120),
    _m("mistralai/Mixtral-8x7B-Instruct-v0.1", "Mixtral-8x7B (MoE)", 46.7e9, 12.9e9, 32, 8, 128, 4096, experts=8, top_k=2),
    _m("google/gemma-3-12b-it", "Gemma-3-12B", 12.2e9, 12.2e9, 48, 8, 256, 3840, note="sliding-window layers cache less than this full-attention estimate"),
    _m("google/gemma-3-27b-it", "Gemma-3-27B", 27.4e9, 27.4e9, 62, 16, 128, 5376, note="sliding-window layers cache less than this full-attention estimate"),
    _m("microsoft/phi-4", "Phi-4 (14B)", 14.7e9, 14.7e9, 40, 10, 128, 5120),
    _m("deepseek-ai/DeepSeek-V3", "DeepSeek-V3 (MoE, MLA)", 671e9, 37e9, 61, 128, 128, 7168, kv_per_layer=576, experts=256, top_k=8),
    _m("deepseek-ai/DeepSeek-R1", "DeepSeek-R1 (MoE, MLA)", 671e9, 37e9, 61, 128, 128, 7168, kv_per_layer=576, experts=256, top_k=8),
    _m("openai/gpt-oss-20b", "gpt-oss-20b (MoE)", 20.9e9, 3.6e9, 24, 8, 64, 2880, experts=32, top_k=4, note="published in MXFP4; sliding-window layers cache less"),
    _m("openai/gpt-oss-120b", "gpt-oss-120b (MoE)", 117e9, 5.1e9, 36, 8, 64, 2880, experts=128, top_k=4, note="published in MXFP4; sliding-window layers cache less"),
]
_BY_ID = {m.id.lower(): m for m in CATALOGUE}
_ALIASES = {
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct", "llama8b": "meta-llama/Llama-3.1-8B-Instruct", "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct", "llama70b": "meta-llama/Llama-3.3-70B-Instruct", "llama-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen3-32b": "Qwen/Qwen3-32B", "qwen3-8b": "Qwen/Qwen3-8B", "qwen3-14b": "Qwen/Qwen3-14B", "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-235b": "Qwen/Qwen3-235B-A22B", "qwen3-235b-a22b": "Qwen/Qwen3-235B-A22B", "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
    "mistral-small-24b": "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "gemma-3-27b": "google/gemma-3-27b-it", "gemma-3-12b": "google/gemma-3-12b-it", "phi-4": "microsoft/phi-4",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3", "deepseek-r1": "deepseek-ai/DeepSeek-R1", "gpt-oss-120b": "openai/gpt-oss-120b", "gpt-oss-20b": "openai/gpt-oss-20b",
}


def catalogue() -> List[Model]:
    return list(CATALOGUE)


# ---------------------------------------------------------------- Hugging Face Hub
def _get(url: str, timeout: float = 20.0, head: bool = False):
    req = urllib.request.Request(url, method="HEAD" if head else "GET", headers={"User-Agent": "vitalops/0.1"})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", "Bearer " + tok)
    return urllib.request.urlopen(req, timeout=timeout)


def _cache_path(repo: str) -> str:
    return os.path.join(CACHE_DIR, re.sub(r"[^A-Za-z0-9_.-]", "__", repo) + ".json")


def _cached(repo: str) -> Optional[dict]:
    p = _cache_path(repo)
    try:
        if time.time() - os.path.getmtime(p) < CACHE_TTL:
            with open(p) as f:
                return json.load(f)
    except OSError:
        pass
    return None


def _store(repo: str, data: dict) -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(_cache_path(repo), "w") as f:
            json.dump(data, f)
    except OSError:
        pass


def fetch_hub(repo: str, revision: str = "main") -> dict:
    """config.json plus, when available, the checkpoint's total byte size from the safetensors index."""
    c = _cached(repo)
    if c:
        return c
    base = "%s/%s/resolve/%s/" % (HUB, repo, revision)
    try:
        with _get(base + "config.json") as r:
            cfg = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise LookupError("%s is gated or private; set HF_TOKEN to a token that can read it" % repo)
        if e.code == 404:
            raise LookupError("no config.json at %s" % (base + "config.json"))
        raise
    total_bytes = None
    try:
        with _get(base + "model.safetensors.index.json") as r:
            idx = json.loads(r.read().decode("utf-8"))
        total_bytes = int(idx.get("metadata", {}).get("total_size") or 0) or None
    except Exception:
        try:
            with _get(base + "model.safetensors", head=True) as r:
                total_bytes = int(r.headers.get("Content-Length") or 0) or None
        except Exception:
            total_bytes = None
    data = {"repo": repo, "config": cfg, "total_bytes": total_bytes}
    _store(repo, data)
    return data


def _shapes_params(cfg: dict) -> tuple:
    """(total, active) parameters from the shapes in a config, for the common decoder layouts."""
    h = cfg["hidden_size"]; L = cfg["num_hidden_layers"]; nh = cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads") or nh; hd = cfg.get("head_dim") or h // nh
    vocab = cfg.get("vocab_size", 0); inter = cfg.get("intermediate_size", 4 * h)
    tied = cfg.get("tie_word_embeddings", False)
    experts = cfg.get("num_local_experts") or cfg.get("num_experts") or cfg.get("n_routed_experts") or 0
    top_k = cfg.get("num_experts_per_tok") or 0
    moe_inter = cfg.get("moe_intermediate_size") or inter
    shared = cfg.get("n_shared_experts") or 0
    dense_first = cfg.get("first_k_dense_replace") or 0
    if cfg.get("kv_lora_rank"):     # multi-head latent attention (DeepSeek-V2/V3)
        q_rank = cfg.get("q_lora_rank") or 0
        attn = (h * q_rank + q_rank * nh * (cfg.get("qk_nope_head_dim", hd) + cfg.get("qk_rope_head_dim", 0))) if q_rank else h * nh * (cfg.get("qk_nope_head_dim", hd) + cfg.get("qk_rope_head_dim", 0))
        attn += h * (cfg["kv_lora_rank"] + cfg.get("qk_rope_head_dim", 0)) + cfg["kv_lora_rank"] * nh * (cfg.get("qk_nope_head_dim", hd) + cfg.get("v_head_dim", hd)) + nh * cfg.get("v_head_dim", hd) * h
    else:
        attn = h * nh * hd + 2 * h * nkv * hd + nh * hd * h
    dense_mlp = 3 * h * inter
    total = active = 0.0
    for i in range(L):
        if experts and i >= dense_first:
            total += attn + experts * 3 * h * moe_inter + shared * 3 * h * moe_inter + h * experts
            active += attn + top_k * 3 * h * moe_inter + shared * 3 * h * moe_inter + h * experts
        else:
            total += attn + dense_mlp
            active += attn + dense_mlp
    head = vocab * h
    total += head + (0 if tied else head)
    active += head + (0 if tied else head)   # embeddings count in both, so a dense model's active equals its total
    return total, active


def from_hub(repo: str, revision: str = "main") -> Model:
    """A Model from a Hugging Face repository: shapes from config.json, parameter count from the checkpoint size when possible."""
    data = fetch_hub(repo, revision)
    cfg = data["config"]
    if "text_config" in cfg and "hidden_size" not in cfg:   # multimodal wrappers: the language model is inside
        cfg = dict(cfg["text_config"])
    for k in ("hidden_size", "num_hidden_layers", "num_attention_heads"):
        if k not in cfg:
            raise LookupError("%s: config.json has no %s; not a decoder-only language model this planner understands" % (repo, k))
    h = cfg["hidden_size"]; nh = cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads") or nh; hd = cfg.get("head_dim") or h // nh
    total_shapes, active = _shapes_params(cfg)
    total = total_shapes
    note = ""
    dtype = str(cfg.get("torch_dtype", "bfloat16"))
    if data.get("total_bytes") and not cfg.get("quantization_config") and dtype in _DTYPE_BYTES:
        total = data["total_bytes"] / _DTYPE_BYTES[dtype]
        if total_shapes and abs(total - total_shapes) / total_shapes > 0.35:
            note = "checkpoint size and config shapes disagree by more than a third; the config shapes are used"
            total = total_shapes
        elif total_shapes:
            active = active * total / total_shapes if active < total_shapes else total
    experts = cfg.get("num_local_experts") or cfg.get("num_experts") or cfg.get("n_routed_experts") or 0
    kv_per_layer = (cfg["kv_lora_rank"] + cfg.get("qk_rope_head_dim", 0)) if cfg.get("kv_lora_rank") else None
    if cfg.get("sliding_window") and cfg.get("use_sliding_window", True) and not note:
        note = "some layers use a sliding window; the KV estimate assumes full attention on every layer"
    name = repo.split("/")[-1]
    return Model(id=repo, name=name, total=float(total), active=float(min(active, total)), layers=cfg["num_hidden_layers"], kv_heads=nkv, head_dim=hd, hidden=h,
                 kv_per_layer=kv_per_layer, experts=experts, top_k=cfg.get("num_experts_per_tok") or 0, source="hub", note=note)


def load(name: str, offline: bool = False, revision: str = "main") -> Model:
    """A model by catalogue alias, catalogue id, or Hugging Face repository id."""
    k = name.strip().lower()
    if k in _ALIASES:
        return _BY_ID[_ALIASES[k].lower()]
    if k in _BY_ID:
        return _BY_ID[k]
    for m in CATALOGUE:
        if m.name.lower() == k:
            return m
    if "/" in name:
        if offline:
            raise LookupError("%s is not in the built-in catalogue and --offline is set" % name)
        return from_hub(name, revision)
    raise LookupError("unknown model %r; use a Hugging Face id like meta-llama/Llama-3.1-8B-Instruct, or one of: %s" % (name, ", ".join(sorted(_ALIASES))))
