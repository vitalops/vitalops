"""The bridge the interactive planner calls, the local server's API, and the docs site's copy of the source."""
import json
import os
import socket
import threading
import urllib.request

from vitalops import web

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_evaluate_matches_the_report():
    r = web.evaluate({"model": "Qwen/Qwen3-32B", "gpu": "h100"})
    p = r["plan"]
    assert p["fits"] and p["layout"]["gpus"] == 1
    assert round(p["user_tps"], 1) == 22.5 and round(p["agg_tps"]) == 1437 and round(p["ttft_ms"]) == 134
    assert "22.5 tok/s per stream" in r["report"] and r["command"] == "vitalops plan Qwen/Qwen3-32B --gpu h100"
    assert r["chart"][0]["b"] == 1 and r["chart"][-1]["b"] == p["layout"]["max_batch"]


def test_evaluate_reports_errors_and_misfits():
    assert "error" in web.evaluate({"model": "not-a-model", "gpu": "h100", "offline": True})
    r = web.evaluate({"model": "deepseek-v3", "gpu": "h100"})
    assert not r["plan"]["fits"] and r["chart"] == [] and r["runtime_bytes"] > 0


def test_command_lists_only_non_defaults():
    st = {"model": "llama-70b", "gpu": "l40s", "users": 32, "slo": 400, "weights": "fp8", "price": 1.0}
    assert web.command(st) == "vitalops plan llama-70b --gpu l40s --users 32 --slo 400"
    st["price"] = 0.8
    assert web.command(st).endswith("--price 0.80")


def test_preload_feeds_the_hub_loader(tmp_path, monkeypatch):
    from vitalops import models
    monkeypatch.setattr(models, "CACHE_DIR", str(tmp_path))
    cfg = {"hidden_size": 2048, "num_hidden_layers": 24, "num_attention_heads": 16, "num_key_value_heads": 2, "intermediate_size": 8192, "vocab_size": 50000}
    web.preload("someone/tiny-model", cfg, None)
    mi = json.loads(web.model_info_json("someone/tiny-model"))
    assert mi["layers"] == 24 and mi["kv_heads"] == 2 and mi["source"] == "hub"
    assert "error" not in web.evaluate({"model": "someone/tiny-model", "gpu": "rtx5090"})


def test_ui_server_api():
    from vitalops import ui
    import socketserver
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    httpd = socketserver.TCPServer(("127.0.0.1", port), ui._Quiet)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = "http://127.0.0.1:%d/" % port
        d = json.load(urllib.request.urlopen(base + "api/models"))
        assert any(m["id"] == "Qwen/Qwen3-32B" for m in d["models"])
        req = urllib.request.Request(base + "api/evaluate", data=json.dumps({"model": "Qwen/Qwen3-32B", "gpu": "h100"}).encode(), method="POST")
        r = json.load(urllib.request.urlopen(req))
        assert round(r["plan"]["user_tps"], 1) == 22.5
        assert b"VO_PLANNER_ENGINE" in urllib.request.urlopen(base + "engine.js").read()
    finally:
        httpd.shutdown(); httpd.server_close()


def test_docs_site_copy_is_in_sync():
    import importlib.util
    spec = importlib.util.spec_from_file_location("site", os.path.join(ROOT, "tools", "site.py"))
    site = importlib.util.module_from_spec(spec); spec.loader.exec_module(site)
    stale = [os.path.relpath(d, ROOT) for s, d in site.pairs() if not os.path.exists(d) or open(s, "rb").read() != open(d, "rb").read()]
    assert not stale, "run python tools/site.py: %s" % stale
