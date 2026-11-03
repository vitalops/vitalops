"""`vitalops bench` against the mock server in tests/mock_server.py."""
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mock_server  # noqa: E402

from vitalops.bench import Prompts, bench, pct  # noqa: E402


@pytest.fixture(scope="module")
def server():
    s = mock_server.serve()
    yield "http://127.0.0.1:%d" % s.server_address[1]
    s.shutdown()


@pytest.fixture(scope="module")
def strict():
    s = mock_server.serve(strict=True)
    yield "http://127.0.0.1:%d" % s.server_address[1]
    s.shutdown()


def test_measures_each_level_and_finds_capacity(server):
    run = bench(server, users=(1, 8), prompt=200, output=12, requests=8)
    assert run.model == mock_server.MODEL and run.api == "completions" and run.exact_output
    a, b = run.levels
    assert a.requests == 8 and b.requests == 16 and a.errors == b.errors == 0
    assert a.output_tokens == 12                       # min_tokens honoured
    assert abs(a.prompt_tokens - 200) <= 6              # prompts calibrated with the server's own counts
    assert b.output_tps > a.output_tps * 2              # concurrency raises throughput ...
    assert b.itl_ms_p50 > a.itl_ms_p50                  # ... and slows each stream, as the mock's timing law says
    assert run.capacity == 8 and not run.notes


def test_strict_server_falls_back_and_says_so(strict):
    run = bench(strict, users=(2,), prompt=64, output=8, requests=4, api="chat")
    assert run.api == "chat" and not run.exact_output
    assert run.levels[0].errors == 0 and run.levels[0].output_tokens < 8
    assert any("min_tokens" in n for n in run.notes)


def test_plan_is_attached_per_level(server):
    run = bench(server, users=(1, 4), prompt=128, output=8, requests=4, plan_gpu="h100", plan_model="llama-8b")
    p = run.levels[0].predicted
    assert p["fits"] and p["user_tps"] > 0 and p["ttft_ms"] > 0
    assert run.plan == {"gpu": "h100", "tp": 1, "model": "llama-8b"}


def test_cli_exit_codes_and_json(server, tmp_path):
    cli = [sys.executable, "-m", "vitalops.cli", "bench", server, "--users", "1,4", "--prompt", "64", "--output", "8", "--requests", "4", "-q"]
    out = subprocess.run(cli + ["--json", "--save", str(tmp_path / "run.json")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["levels"][1]["users"] == 4 and (tmp_path / "run.json").exists()
    assert subprocess.run(cli + ["--slo", "1", "--require", "4"], capture_output=True, text=True).returncode == 1
    assert subprocess.run(cli + ["--users", "x"], capture_output=True, text=True).returncode == 2
    import socket
    with socket.socket() as so:                         # a port nothing listens on
        so.bind(("127.0.0.1", 0)); closed = so.getsockname()[1]
    assert subprocess.run([sys.executable, "-m", "vitalops.cli", "bench", "http://127.0.0.1:%d" % closed, "-q"], capture_output=True, text=True).returncode == 2


def test_helpers():
    assert pct([1, 2, 3, 4], 0.5) == 2.5 and pct([5], 0.95) == 5
    p = Prompts()
    p.calibrate(lambda text: int(len(text.split()) * 1.3) + 7)
    assert abs(p.overhead - 7) <= 1 and abs(1 / p.words_per_token - 1.3) < 0.02
