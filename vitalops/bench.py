"""Measure an OpenAI-compatible inference endpoint (vLLM, SGLang, TGI, llama.cpp, a hosted API).

For each concurrency level, that many workers send streaming requests back to back (a closed loop)
and every request's time to first token, time between tokens and output length are recorded. The
first request of each worker is a warm-up and is not counted. Prompts are made of random common
words, calibrated to a token count with the server's own usage figures, so every request is distinct
unless a shared prefix is asked for. Only the standard library is used.
"""
from __future__ import annotations

import http.client
import json
import os
import random
import ssl
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
from urllib.parse import urlparse

from .hw import gpu as get_gpu
from .models import load as load_model
from .physics import Scenario, perf, plan

WORDS = ("time year people way day man thing woman life child world school state family student group country problem hand part place case week company system "
         "program question work government number night point home water room mother area money story fact month lot right study book eye job word business issue "
         "side kind head house service friend father power hour game line end member law car city community name president team minute idea kid body information "
         "back parent face others level office door health person art war history party result change morning reason research girl guy moment air teacher force "
         "education foot boy age policy process music market sense nation plan college interest death experience effect class control care field development role "
         "effort rate heart drug show leader light voice wife police mind price report decision son view relationship town road arm difference value building action "
         "model season society tax director position player record paper space ground form event official matter center couple site project activity star table need "
         "court oil situation cost industry figure street image phone data picture practice piece land product doctor wall patient worker news test movie north love "
         "support technology step baby computer type attention film tree source organization hair window evidence population site truth river green blue small large").split()


class BenchError(Exception):
    pass


@dataclass
class Request:
    start: float
    ttft: Optional[float] = None      # s
    end: Optional[float] = None
    out_tokens: int = 0
    prompt_tokens: int = 0
    gaps: List[float] = field(default_factory=list)   # s between streamed chunks after the first
    error: str = ""


@dataclass
class Level:
    users: int
    requests: int
    errors: int
    duration_s: float
    requests_per_s: float
    output_tps: float                 # output tokens per second, all streams
    user_tps: float                   # output tokens per second one stream sees after its first token, mean
    ttft_ms_p50: float
    ttft_ms_p95: float
    itl_ms_p50: float                 # time between tokens
    itl_ms_p95: float
    prompt_tokens: float
    output_tokens: float
    slo_ok: Optional[bool] = None
    predicted: Optional[dict] = None


@dataclass
class Run:
    url: str
    model: str
    api: str
    prompt: int
    output: int
    shared_prefix: int
    levels: List[Level]
    capacity: Optional[int]           # the most concurrent requests whose p95 TTFT met the target
    slo_ms: float
    exact_output: bool                # whether the server honoured min_tokens / ignore_eos
    notes: List[str] = field(default_factory=list)
    plan: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


def pct(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    i = int(k)
    return xs[i] if i + 1 >= len(xs) else xs[i] + (xs[i + 1] - xs[i]) * (k - i)


class Client:
    """One keep-alive connection to the server; each worker thread has its own."""

    def __init__(self, url: str, api_key: Optional[str], timeout: float = 600.0):
        u = urlparse(url if "://" in url else "http://" + url)
        self.https = u.scheme == "https"
        self.host = u.hostname
        self.port = u.port or (443 if self.https else 80)
        base = u.path.rstrip("/")
        self.base = base if base.endswith("/v1") else base + "/v1"
        self.headers = {"Content-Type": "application/json", "User-Agent": "vitalops-bench"}
        if api_key:
            self.headers["Authorization"] = "Bearer " + api_key
        self.timeout = timeout
        self.conn = None

    def _connect(self):
        if self.conn is None:
            if self.https:
                self.conn = http.client.HTTPSConnection(self.host, self.port, timeout=self.timeout, context=ssl.create_default_context())
            else:
                self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        return self.conn

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def request(self, method: str, path: str, body: Optional[dict] = None):
        for attempt in (0, 1):
            c = self._connect()
            try:
                c.request(method, self.base + path, body=json.dumps(body) if body is not None else None, headers=self.headers)
                return c.getresponse()
            except (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError, http.client.CannotSendRequest):
                self.close()
                if attempt:
                    raise

    def get_json(self, path: str) -> dict:
        r = self.request("GET", path)
        data = r.read()
        if r.status != 200:
            raise BenchError("GET %s answered %d: %s" % (self.base + path, r.status, data[:200].decode("utf-8", "replace")))
        return json.loads(data)

    def stream(self, api: str, body: dict, rq: Request) -> None:
        """Sends one streaming request and fills in its timings."""
        r = self.request("POST", "/completions" if api == "completions" else "/chat/completions", body)
        if r.status != 200:
            data = r.read()
            raise BenchError("%d %s" % (r.status, data[:300].decode("utf-8", "replace")))
        last = None
        chunks = 0
        while True:
            line = r.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                r.read()                    # the rest of the body, so the connection can be reused
                break
            d = json.loads(payload)
            if d.get("usage"):
                rq.out_tokens = d["usage"].get("completion_tokens") or rq.out_tokens
                rq.prompt_tokens = d["usage"].get("prompt_tokens") or rq.prompt_tokens
            ch = d.get("choices") or []
            if not ch:
                continue
            text = ch[0].get("text") if api == "completions" else (ch[0].get("delta") or {}).get("content")
            if not text:
                continue
            now = time.perf_counter()
            if rq.ttft is None:
                rq.ttft = now - rq.start
            elif last is not None:
                rq.gaps.append(now - last)
            last = now
            chunks += 1
        rq.end = time.perf_counter()
        if not rq.out_tokens:
            rq.out_tokens = chunks          # servers that send no usage: one chunk is about one token


class Prompts:
    """Prompts of random common words, sized in the server's own tokens."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.words_per_token = 1.0
        self.overhead = 0

    def text(self, tokens: int, rng: Optional[random.Random] = None) -> str:
        r = rng or self.rng
        n = max(1, int(round((tokens - self.overhead) * self.words_per_token)))
        return " ".join(r.choice(WORDS) for _ in range(n))

    def calibrate(self, measure) -> None:
        """measure(text) -> prompt tokens the server counted; two sizes give the slope and the template overhead."""
        a, b = 64, 512
        ta = measure(" ".join(self.rng.choice(WORDS) for _ in range(a)))
        tb = measure(" ".join(self.rng.choice(WORDS) for _ in range(b)))
        if ta and tb and tb > ta:
            slope = (tb - ta) / (b - a)            # tokens per word
            self.words_per_token = 1 / slope
            self.overhead = max(0, int(round(ta - a * slope)))


def _body(api: str, model: str, text: str, output: int, exact: bool) -> dict:
    b = {"model": model, "max_tokens": output, "stream": True, "stream_options": {"include_usage": True}, "temperature": 0}
    if exact:
        b.update({"ignore_eos": True, "min_tokens": output})
    if api == "completions":
        b["prompt"] = text
    else:
        b["messages"] = [{"role": "user", "content": text}]
    return b


def detect(url: str, api_key: Optional[str], model: Optional[str], api: Optional[str]) -> tuple:
    """(model id, api, exact output supported) from the server itself."""
    c = Client(url, api_key, timeout=15)
    try:
        if not model:
            try:
                ms = c.get_json("/models").get("data") or []
            except (OSError, BenchError) as e:
                raise BenchError("could not list models at %s (%s); pass --model" % (c.base + "/models", e))
            if not ms:
                raise BenchError("the server lists no models; pass --model")
            model = ms[0]["id"]
        apis = [api] if api else ["completions", "chat"]
        last_err = ""
        for a in apis:
            for exact in (True, False):
                rq = Request(start=time.perf_counter())
                try:
                    c.stream(a, _body(a, model, "hello there", 2, exact), rq)
                    return model, a, exact
                except BenchError as e:
                    last_err = str(e)
                    if not exact:
                        break
                except OSError as e:
                    raise BenchError("could not reach %s: %s" % (url, e))
        raise BenchError("neither /v1/completions nor /v1/chat/completions accepted a request: %s" % last_err)
    finally:
        c.close()


def run_level(url: str, api_key: Optional[str], model: str, api: str, exact: bool, prompts: Prompts, users: int, measured: int,
              prompt: int, output: int, shared_prefix: int, seed: int) -> List[Request]:
    """`users` workers in a closed loop; the first request of each is a warm-up. Returns the measured requests."""
    shared = prompts.text(shared_prefix, random.Random(seed)) if shared_prefix > 0 else ""
    lock = threading.Lock()
    started = [0]
    done: List[Request] = []
    target = users + measured

    def worker(wid: int):
        c = Client(url, api_key)
        rng = random.Random(seed * 1000003 + wid)
        first = True
        try:
            while True:
                with lock:
                    if started[0] >= target:
                        return
                    started[0] += 1
                uniq = prompts.text(max(1, prompt - shared_prefix), rng)
                text = (shared + " " + uniq) if shared else uniq
                rq = Request(start=time.perf_counter())
                try:
                    c.stream(api, _body(api, model, text, output, exact), rq)
                except Exception as e:      # recorded, the run continues
                    rq.error = str(e)[:200] or type(e).__name__
                    rq.end = time.perf_counter()
                    c.close()
                if not first:
                    with lock:
                        done.append(rq)
                first = False
        finally:
            c.close()

    ts = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(users)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return done


def summarise(users: int, rqs: List[Request], slo_ms: float) -> Level:
    ok = [r for r in rqs if not r.error and r.ttft is not None]
    errors = len(rqs) - len(ok)
    if not ok:
        return Level(users=users, requests=0, errors=errors, duration_s=0, requests_per_s=0, output_tps=0, user_tps=0, ttft_ms_p50=float("nan"),
                     ttft_ms_p95=float("nan"), itl_ms_p50=float("nan"), itl_ms_p95=float("nan"), prompt_tokens=0, output_tokens=0, slo_ok=False)
    t0 = min(r.start for r in ok)
    t1 = max(r.end for r in ok)
    span = max(1e-9, t1 - t0)
    out = sum(r.out_tokens for r in ok)
    ttfts = [r.ttft * 1000 for r in ok]
    gaps = [g * 1000 for r in ok for g in r.gaps]
    per_stream = [(r.out_tokens - 1) / (r.end - r.start - r.ttft) for r in ok if r.out_tokens > 1 and r.end - r.start - r.ttft > 0]
    p95 = pct(ttfts, 0.95)
    return Level(users=users, requests=len(ok), errors=errors, duration_s=span, requests_per_s=len(ok) / span, output_tps=out / span,
                 user_tps=sum(per_stream) / len(per_stream) if per_stream else float("nan"), ttft_ms_p50=pct(ttfts, 0.5), ttft_ms_p95=p95,
                 itl_ms_p50=pct(gaps, 0.5), itl_ms_p95=pct(gaps, 0.95), prompt_tokens=sum(r.prompt_tokens for r in ok) / len(ok),
                 output_tokens=out / len(ok), slo_ok=p95 <= slo_ms)


def predict(scenario_kw: dict, model_id: str, gpu: str, tp: int, users: int, prompt: int, output: int) -> Optional[dict]:
    """The planner's figures for one replica at this concurrency: what the measurement is compared with."""
    m = load_model(model_id)
    s = Scenario(model=m, gpu=get_gpu(gpu), users=users, prompt=prompt, output=output, tp=tp, **scenario_kw)
    p = plan(s)
    if not p.fits:
        return {"fits": False, "reason": p.reason}
    L = p.layout
    if users > L.max_batch:
        return {"fits": False, "reason": "%d requests do not fit one TP%d replica's KV cache (max %d)" % (users, tp, L.max_batch)}
    pf = perf(s, tp, users)
    return {"fits": True, "user_tps": pf.user_tps, "output_tps": pf.agg_tps, "ttft_ms": pf.ttft_s * 1000, "step_ms": pf.step_s * 1000,
            "requests_per_s": pf.agg_tps / output}


def bench(url: str, users=(1, 4, 16, 64), prompt: int = 1024, output: int = 256, shared_prefix: int = 0, requests: int = 16, slo_ms: float = 500.0,
          model: Optional[str] = None, api: Optional[str] = None, api_key: Optional[str] = None, seed: int = 0,
          plan_gpu: Optional[str] = None, plan_model: Optional[str] = None, tp: int = 1, scenario_kw: Optional[dict] = None, progress=None) -> Run:
    """Measures the endpoint at each concurrency in `users`. With plan_gpu, adds the planner's prediction for one replica on that GPU."""
    api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
    model, api, exact = detect(url, api_key, model, api)
    say = progress or (lambda msg: None)
    say("model %s · /v1/%s%s" % (model, api, "" if exact else " · the server ignores min_tokens, so outputs may end early"))
    prompts = Prompts(seed)
    c = Client(url, api_key, timeout=120)

    def measure(text):
        rq = Request(start=time.perf_counter())
        b = _body(api, model, text, 1, False)
        c.stream(api, b, rq)
        return rq.prompt_tokens

    try:
        prompts.calibrate(measure)
    except Exception:
        pass
    finally:
        c.close()
    notes = []
    if prompts.overhead == 0 and prompts.words_per_token == 1.0:
        notes.append("the server reports no prompt token counts; prompt sizes assume one token per word")
    if not exact:
        notes.append("the server ignored min_tokens/ignore_eos; outputs can be shorter than asked")
    levels = []
    for u in users:
        say("%d concurrent: %d requests after a warm-up of %d…" % (u, max(requests, 2 * u), u))
        rqs = run_level(url, api_key, model, api, exact, prompts, u, max(requests, 2 * u), prompt, output, shared_prefix, seed + u)
        lv = summarise(u, rqs, slo_ms)
        if plan_gpu:
            try:
                lv.predicted = predict(scenario_kw or {}, plan_model or model, plan_gpu, tp, u, prompt, output)
            except Exception as e:
                lv.predicted = {"fits": False, "reason": str(e.args[0] if e.args else e)}
        levels.append(lv)
        if lv.errors:
            errs = [r.error for r in rqs if r.error]
            notes.append("%d concurrent: %d of %d requests failed, e.g. %s" % (u, lv.errors, len(rqs), errs[0]))
    cap = None
    for lv in levels:
        if lv.slo_ok and lv.requests:
            cap = lv.users if cap is None or lv.users > cap else cap
    return Run(url=url, model=model, api=api, prompt=prompt, output=output, shared_prefix=shared_prefix, levels=levels, capacity=cap, slo_ms=slo_ms,
               exact_output=exact, notes=notes, plan={"gpu": plan_gpu, "tp": tp, "model": plan_model or model} if plan_gpu else None)


def explain(run: Run) -> List[str]:
    """Where the measurement and the planner part ways, in terms of what the planner leaves out."""
    out = []
    for lv in run.levels:
        p = lv.predicted
        if not p or not p.get("fits") or not lv.requests:
            continue
        r_tps = lv.user_tps / p["user_tps"] if p["user_tps"] else float("nan")
        r_ttft = lv.ttft_ms_p50 / p["ttft_ms"] if p["ttft_ms"] else float("nan")
        if lv.users == 1 and r_tps < 0.7:
            out.append("At 1 request the stream runs at %.0f%% of the estimate: kernel launch and scheduling overheads at small batches, which the planner does not model." % (100 * r_tps))
        if lv.users > 1 and r_ttft > 2 and lv.ttft_ms_p50 - p["ttft_ms"] > 50:
            out.append("At %d requests TTFT is %.1f× the estimate: requests wait behind other prefills. The planner's TTFT has no queueing." % (lv.users, r_ttft))
        if run.shared_prefix and r_ttft < 0.7 and p["ttft_ms"] - lv.ttft_ms_p50 > 20:
            out.append("At %d requests TTFT is %.0f%% of the estimate: the shared prefix is served from the prefix cache, which the planner does not model." % (lv.users, 100 * r_ttft))
    return out
