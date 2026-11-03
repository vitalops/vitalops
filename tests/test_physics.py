import json
import subprocess
import sys

import tokplan
from tokplan import models, physics
from tokplan.hw import GiB


def test_catalogue_defaults_match_the_website_example():
    # Qwen3-32B, fp8 weights and KV, H100, 64 users, 2k in / 300 out: the figures the website's planner shows
    p = tokplan.plan("qwen3-32b", "h100", users=64, prompt=2048, output=300, offline=True)
    assert p.fits and p.layout.tp == 1 and p.layout.gpus == 1
    assert abs(p.user_tps - 22.5) < 0.3
    assert abs(p.agg_tps - 1437) < 20
    assert abs(p.ttft_ms - 134) < 3
    assert abs(p.cost_per_1m_output - 0.48) < 0.02


def test_llama8b_batch_one_decode_rate():
    # bf16 on one H100 at batch 1: weights 16 GB at 75% of 3.35 TB/s → about 157 tok/s before prefill amortisation
    m = models.load("llama-8b")
    s = physics.Scenario(model=m, gpu=tokplan.get_gpu("h100"), weights="bf16", kv="bf16", users=1, prompt=256, output=256)
    pf = physics.perf(s, 1, 1)
    assert 140 < 1 / pf.step_s < 170


def test_70b_needs_two_h100_and_comms_are_small_on_nvlink():
    p = tokplan.plan("llama-70b", "h100", users=64, offline=True)
    assert p.layout.tp == 2 and p.layout.gpus == 2
    pf = p.layout.perf
    assert pf.t_comm_s < 0.002 and pf.t_comm_s / pf.step_s < 0.12


def test_pcie_comms_are_expensive():
    p = tokplan.plan("llama-70b", "rtx5090", weights="int4", users=64, offline=True)
    assert p.fits
    pf = p.layout.perf
    assert pf.t_comm_s / pf.step_s > 0.15


def test_does_not_fit():
    p = tokplan.plan("deepseek-v3", "h100", weights="fp8", users=8, offline=True)
    assert not p.fits and "exceeds" in p.reason
    assert tokplan.fit("deepseek-v3", "h200", weights="fp8", users=8, offline=True)


def test_moe_reads_only_touched_experts_at_batch_one():
    m = models.load("qwen3-235b")
    b1 = physics.weights_read(m, 1.0, 1)
    b256 = physics.weights_read(m, 1.0, 256)
    assert abs(b1 - m.active) / m.active < 0.01
    assert b256 > 0.9 * m.total


def test_shapes_param_count_for_llama_like_config():
    cfg = {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 128256, "tie_word_embeddings": False}
    total, active = models._shapes_params(cfg)
    assert abs(total - 8.03e9) / 8.03e9 < 0.01 and active == total


def test_shapes_param_count_for_moe_config():
    cfg = {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 32000,
           "num_local_experts": 8, "num_experts_per_tok": 2, "tie_word_embeddings": False}
    total, active = models._shapes_params(cfg)
    assert abs(total - 46.7e9) / 46.7e9 < 0.02 and abs(active - 12.9e9) / 12.9e9 < 0.02


def test_cli_json_and_exit_codes():
    out = subprocess.run([sys.executable, "-m", "tokplan.cli", "qwen3-32b", "--offline", "--json"], capture_output=True, text=True)
    assert out.returncode == 0
    d = json.loads(out.stdout)
    assert d["fits"] and d["layout"]["gpus"] == 1 and d["model_shapes"]["layers"] == 64
    out = subprocess.run([sys.executable, "-m", "tokplan.cli", "deepseek-v3", "--gpu", "h100", "--offline", "-q"], capture_output=True, text=True)
    assert out.returncode == 1 and "does not fit" in out.stdout
    out = subprocess.run([sys.executable, "-m", "tokplan.cli", "nonsense-model", "--offline"], capture_output=True, text=True)
    assert out.returncode == 2
