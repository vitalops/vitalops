/* Capacity planner: a first-order roofline estimate of what a model needs to serve
   a given load. Renders into every [data-lab="planner"]. No libraries, no network.
   GPU specifications and the hardware card come from assets/js/hw.js (window.VO_HW). */
(function () {
  'use strict';

  var MODELS = [
    // total / active params (B), layers, KV heads, head dim, hidden size; kvPerLayer overrides (MLA)
    { id: 'llama8',   name: 'Llama-3.1-8B',        total: 8.03,  active: 8.03,  layers: 32, kvh: 8,  hd: 128, h: 4096 },
    { id: 'mistral24',name: 'Mistral-Small-24B',   total: 23.6,  active: 23.6,  layers: 40, kvh: 8,  hd: 128, h: 5120 },
    { id: 'gemma27',  name: 'Gemma-3-27B',         total: 27.4,  active: 27.4,  layers: 62, kvh: 16, hd: 128, h: 5376 },
    { id: 'qwen32',   name: 'Qwen3-32B',           total: 32.8,  active: 32.8,  layers: 64, kvh: 8,  hd: 128, h: 5120 },
    { id: 'llama70',  name: 'Llama-3.3-70B',       total: 70.6,  active: 70.6,  layers: 80, kvh: 8,  hd: 128, h: 8192 },
    { id: 'qwen235',  name: 'Qwen3-235B-A22B (MoE)', total: 235, active: 22,    layers: 94, kvh: 4,  hd: 128, h: 4096 },
    { id: 'dsv3',     name: 'DeepSeek-V3 (MoE, MLA)', total: 671, active: 37,   layers: 61, kvPerLayer: 576, h: 7168 }
  ];
  var GPU_ORDER = ['h100', 'h200', 'b200', 'mi300x', 'a100', 'l40s', 'rtx5090'];
  var WPREC = [['bf16', 2], ['fp8', 1], ['int4', 0.5625]];
  var KVPREC = [['bf16', 2], ['fp8', 1]];
  var USERS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
  var PROMPT = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768];
  var OUTPUT = [100, 200, 300, 500, 1000, 2000, 4000];
  var SLO = [200, 300, 500, 800, 1200, 2000, 3000];
  var TPS = [1, 2, 4, 8];
  var BW_EFF = 0.75, FLOP_EFF = 0.5, PREFILL_EFF = 0.6, MAX_BATCH = 512, GiB = 1073741824;
  var LAT_FAST = 8e-6, LAT_PCIE = 25e-6;                 // all-reduce latency assumptions: NVLink / Infinity Fabric, PCIe
  var C_W = 'var(--brand)', C_KV = '#14a89a', C_ACT = 'var(--ink4)', C_COMM = 'var(--warn)', C_PRE = '#8b5cf6';

  var CSS = [
    '.lp{font-family:var(--font);color:var(--ink2);background:var(--white);border:1px solid var(--line);border-radius:12px;box-shadow:0 1px 2px rgba(16,24,40,.05);display:grid;grid-template-columns:280px minmax(0,1fr);overflow:hidden;text-align:left;line-height:1.45;font-size:13.5px}',
    '.vlab .lp{--ok:#34d399;--warn:#fbbf24;--bad:#f87171;box-shadow:none}',
    '.lp *{box-sizing:border-box}',
    '.lp-ctl{padding:14px;border-right:1px solid var(--line);background:var(--bg);display:flex;flex-direction:column;gap:10px;min-width:0}',
    '.lp-hd{display:flex;align-items:center;gap:8px;font-size:12px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--ink3)}',
    '.lp-hd i{width:7px;height:7px;border-radius:50%;background:var(--brand)}',
    '.lp-f{display:flex;flex-direction:column;gap:6px;min-width:0}',
    '.lp-f>label,.lp-f>.lp-lab{font-size:12.5px;font-weight:600;color:var(--ink);display:flex;justify-content:space-between;gap:8px}',
    '.lp-f>label output,.lp-f>.lp-lab output{font-family:var(--mono);font-weight:500;color:var(--brand);font-variant-numeric:tabular-nums}',
    '.lp-sel,.lp-num{width:100%;font:inherit;font-size:13.5px;color:var(--ink);background:var(--white);border:1px solid var(--line);border-radius:8px;padding:8px 10px;min-width:0}',
    '.lp-sel{appearance:none;-webkit-appearance:none;background-image:linear-gradient(45deg,transparent 50%,var(--ink3) 50%),linear-gradient(135deg,var(--ink3) 50%,transparent 50%);background-position:calc(100% - 15px) 52%,calc(100% - 10px) 52%;background-size:5px 5px;background-repeat:no-repeat;padding-right:28px}',
    '.lp-inl{flex-direction:row;align-items:center;justify-content:space-between;gap:10px}',
    '.lp-inl>label{font-weight:500!important;color:var(--ink2)!important}',
    '.lp-usd{position:relative;width:104px;flex:none}.lp-usd::before{content:"$";position:absolute;left:10px;top:50%;transform:translateY(-50%);font-family:var(--mono);font-size:13px;color:var(--ink3)}.lp-usd .lp-num{padding-left:22px}',
    '.lp-num{font-family:var(--mono);font-variant-numeric:tabular-nums}',
    '.lp-sel:hover,.lp-num:hover{border-color:var(--brand200)}',
    '.lp-seg{display:flex;border:1px solid var(--line);border-radius:8px;background:var(--white);padding:2px;gap:2px}',
    '.lp-seg button{flex:1;font:inherit;font-size:12.5px;font-weight:600;font-family:var(--mono);color:var(--ink3);background:none;border:0;border-radius:6px;padding:6px 4px;cursor:pointer}',
    '.lp-seg button:hover{color:var(--ink)}',
    '.lp-seg button[aria-pressed="true"]{background:var(--brand);color:#fff}',
    '.lp-row2{display:grid;grid-template-columns:1fr 1fr;gap:10px}',
    '.lp input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:20px;background:none;margin:0;cursor:pointer}',
    '.lp input[type=range]::-webkit-slider-runnable-track{height:4px;border-radius:2px;background:linear-gradient(var(--brand),var(--brand)) 0/var(--p,50%) 100% no-repeat,var(--line)}',
    '.lp input[type=range]::-moz-range-track{height:4px;border-radius:2px;background:var(--line)}',
    '.lp input[type=range]::-moz-range-progress{height:4px;border-radius:2px;background:var(--brand)}',
    '.lp input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:var(--white);border:2px solid var(--brand);margin-top:-6px;box-shadow:0 1px 2px rgba(16,24,40,.15)}',
    '.lp input[type=range]::-moz-range-thumb{width:12px;height:12px;border-radius:50%;background:var(--white);border:2px solid var(--brand)}',
    '.lp button:focus-visible,.lp select:focus-visible,.lp input:focus-visible,.lp summary:focus-visible{outline:2px solid var(--brand);outline-offset:2px}',
    '.lp-out{padding:12px 14px 14px;display:flex;flex-direction:column;gap:12px;min-width:0}',
    '.lp-kpis{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));border:1px solid var(--line);border-radius:10px;overflow:hidden}',
    '.lp-kpi{padding:8px 10px;border-left:1px solid var(--line);min-width:0}',
    '.lp-kpi:first-child{border-left:0;background:var(--brand50)}',
    '.lp-kpi small{display:block;font-size:11.5px;color:var(--ink3);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
    '.lp-kpi b{display:block;font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:18px;font-weight:500;color:var(--ink);letter-spacing:-.02em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
    '.lp-kpi:first-child b{color:var(--brand)}',
    '.lp-kpi span{display:block;font-size:11.5px;color:var(--ink3);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
    '.lp-kpi span.lp-ok{color:var(--ok)}.lp-kpi span.lp-bad{color:var(--bad)}',
    '.lp-alert{font-size:13px;border-radius:8px;padding:9px 12px;color:var(--warn);background:color-mix(in srgb,var(--warn) 12%,transparent);border:1px solid color-mix(in srgb,var(--warn) 35%,transparent)}',
    '.lp-alert.lp-err{color:var(--bad);background:color-mix(in srgb,var(--bad) 12%,transparent);border-color:color-mix(in srgb,var(--bad) 35%,transparent)}',
    '.lp-mid{display:grid;grid-template-columns:260px minmax(0,1fr);gap:12px;align-items:start}',
    '.lp-r{display:flex;flex-direction:column;gap:12px;min-width:0}',
    '.lp-card .hwc-kv{font-size:11.5px}',
    '.lp-sec{display:flex;justify-content:space-between;align-items:baseline;gap:10px;font-size:12.5px;font-weight:600;color:var(--ink);margin-bottom:8px}',
    '.lp-sec span{font-weight:400;color:var(--ink3);font-family:var(--mono);font-size:12px;font-variant-numeric:tabular-nums;text-align:right}',
    '.lp-stk{display:flex;height:22px;border-radius:6px;overflow:hidden;border:1px solid var(--line);background:var(--line2)}',
    '.lp-stk i{display:block;height:100%;transition:width .35s ease;min-width:0}',
    '.lp-leg{display:flex;flex-wrap:wrap;gap:4px 14px;margin-top:6px;font-size:11.5px;color:var(--ink3)}',
    '.lp-leg span{display:inline-flex;align-items:center;gap:6px}',
    '.lp-leg em{font-style:normal;font-family:var(--mono);color:var(--ink);font-variant-numeric:tabular-nums}',
    '.lp-leg i{width:10px;height:10px;border-radius:2px;display:inline-block}',
    '.lp-chart svg{display:block;width:100%;height:auto}',
    '.lp-chart .lp-grid{stroke:var(--line2)}',
    '.lp-chart text{font-family:var(--mono);font-size:10px;fill:var(--ink4)}',
    '.lp-chart .lp-agg{fill:none;stroke:var(--brand);stroke-width:2}',
    '.lp-chart .lp-usr{fill:none;stroke:var(--ink3);stroke-width:1.5;stroke-dasharray:4 3}',
    '.lp-chart .lp-area{fill:var(--brand);opacity:.07}',
    '.lp-chart .lp-op{stroke:var(--brand);stroke-dasharray:2 3}',
    '.lp-chart .lp-dot{fill:var(--white);stroke:var(--brand);stroke-width:2}',
    '.lp-chart .lp-dot2{fill:var(--white);stroke:var(--ink3);stroke-width:1.5}',
    '.lp-chart .lp-oplab{fill:var(--brand);font-weight:500}',
    '.lp-tbl{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0;border-top:1px solid var(--line)}',
    '.lp-tbl div{padding:6px 10px 6px 0;border-bottom:1px solid var(--line2);min-width:0}',
    '.lp-tbl dt{font-size:11.5px;color:var(--ink3)}',
    '.lp-tbl dd{margin:0;font-family:var(--mono);font-size:13px;color:var(--ink);font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
    '.lp-how{font-size:13px;color:var(--ink2);border-top:1px solid var(--line);padding-top:10px}',
    '.lp-how summary{cursor:pointer;font-weight:600;color:var(--ink);font-size:13px}',
    '.lp-how ul{margin:10px 0 0;padding-left:18px;display:grid;gap:6px}',
    '.lp-how code{font-family:var(--mono);font-size:12px;color:var(--ink);background:var(--line2);padding:1px 4px;border-radius:4px;overflow-wrap:anywhere}',
    '.lp-note{font-size:12px;color:var(--ink3);margin-top:8px}',
    '@media (max-width:1100px){.lp-mid{grid-template-columns:minmax(0,1fr)}.lp-card .hwc{grid-template-columns:minmax(150px,220px) minmax(0,1fr)}.lp-card .hwc-ph{aspect-ratio:auto;min-height:100%}}',
    '@media (max-width:1040px){.lp-kpis{grid-template-columns:repeat(3,minmax(0,1fr))}.lp-kpi:nth-child(4){border-left:0}.lp-kpi:nth-child(n+4){border-top:1px solid var(--line)}.lp-tbl{grid-template-columns:repeat(3,minmax(0,1fr))}}',
    '@media (max-width:860px){.lp{grid-template-columns:minmax(0,1fr)}.lp-ctl{border-right:0;border-bottom:1px solid var(--line)}}',
    '@media (max-width:520px){.lp-kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.lp-kpi{border-left:1px solid var(--line)!important;border-top:1px solid var(--line)}.lp-kpi:nth-child(odd){border-left:0!important}.lp-kpi:nth-child(-n+2){border-top:0}.lp-kpi:first-child{grid-column:1/-1;border-left:0!important}.lp-kpi:nth-child(2){border-left:0!important;border-top:1px solid var(--line)}.lp-kpi:nth-child(3){border-top:1px solid var(--line)}.lp-tbl{grid-template-columns:repeat(2,minmax(0,1fr))}.lp-out{padding:16px}.lp-ctl{padding:16px}.lp-card .hwc{grid-template-columns:minmax(0,1fr)}.lp-card .hwc-ph{aspect-ratio:3/2;min-height:0}}',
    '@media (prefers-reduced-motion:reduce){.lp-stk i{transition:none}}'
  ].join('\n');

  function injectCSS() {
    if (document.getElementById('lab-planner-css')) return;
    var s = document.createElement('style'); s.id = 'lab-planner-css'; s.textContent = CSS;
    document.head.appendChild(s);
  }

  var HW = window.VO_HW;
  function fmtN(v) {
    if (!isFinite(v)) return '—';
    if (v >= 100) return Math.round(v).toLocaleString('en-GB');
    if (v >= 10) return v.toFixed(1);
    return v.toFixed(2);
  }
  function fmtTok(n) { return n >= 1024 ? (n / 1024) + 'k' : String(n); }
  var fmtGB = function (b) { return HW.gb(b); };
  function fmtMs(s) { return s >= 1 ? s.toFixed(2) + ' s' : Math.round(s * 1000) + ' ms'; }
  function ms(s) { var v = s * 1000; return (v >= 100 ? Math.round(v) : v >= 10 ? v.toFixed(1) : v.toFixed(2)) + ' ms'; }
  function fmtUSD(v) { return v >= 1000 ? '$' + Math.round(v).toLocaleString('en-GB') : v >= 10 ? '$' + v.toFixed(0) : '$' + v.toFixed(2); }
  function byId(arr, id) { for (var i = 0; i < arr.length; i++) if (arr[i].id === id) return arr[i]; return arr[0]; }
  function tipAttr(t) { return t ? ' data-tip="' + HW.esc(t) + '"' : ''; }
  function tipRows(card, tips) { var dts = card.querySelectorAll('.hwc-kv dt'); tips.forEach(function (t, i) { if (dts[i] && t) dts[i].setAttribute('data-tip', t); }); }
  var TIP = {
    model: 'Open-weight model preset. Its parameter count, layers, KV heads, head size and hidden size set the weight bytes, the KV cache per token and the all-reduce sizes.',
    w: 'Weight precision: bytes read per parameter in every decode step. bf16 2 B, fp8 1 B, int4 0.5625 B including group scales.',
    bf16: '16-bit brain float, the checkpoint precision: 2 bytes per parameter; matmuls at the bf16 tensor-core rate.',
    fp8: '8-bit floating point (E4M3): 1 byte per parameter; matmuls at the fp8 tensor-core rate where the GPU has one, otherwise dequantised to bf16.',
    int4: '4-bit integers with group scales: 0.5625 bytes per parameter; runs as NVFP4 on Blackwell, otherwise dequantised to fp8 or bf16 for the matmuls.',
    kv: 'KV-cache precision: bytes per cached key or value element. Sets the KV bytes read per step and the cache each request reserves.',
    kvbf16: '2 bytes per cached element.',
    kvfp8: '1 byte per cached element: half the KV bytes read per step, twice the requests in the same memory.',
    price: 'Hourly price of one GPU. The default is an assumption per GPU, editable; it only affects the cost figures.',
    users: 'Requests in flight at once across all replicas. Sets the batch per replica and the KV cache that must fit.',
    inp: 'Average prompt length in tokens. Sets prefill work, time to first token and the KV cache per request.',
    out: 'Average answer length in tokens. Sets the KV cache per request and how much prefill is amortised over each answer.',
    slo: 'Time-to-first-token target. The planner picks the fewest GPUs among TP 1, 2, 4 and 8 whose TTFT meets it.',
    gpus: 'Tensor-parallel degree × replicas: the fewest GPUs whose memory holds the weights and KV cache and whose TTFT meets the target.',
    user: 'Tokens per second one stream sees: 1 ÷ (decode step + prefill share amortised per output token).',
    agg: 'Per-user rate × batch per replica × replicas.',
    ttft: 'Time to first token: one prompt prefilled at 60% of the matmul rate, plus its all-reduces, plus one decode step. No queueing is modelled.',
    cpm: 'GPUs × hourly price ÷ (aggregate tokens per second × 3600), per million output tokens. The monthly figure is the hourly cost × 730.',
    stk: 'How one output token\'s time splits at the operating point: weights read, KV cache read, tensor-parallel all-reduces, and the prefill share. The total is 1 ÷ the per-user rate.',
    l1: 'Weight bytes read per step ÷ (bandwidth × 0.75), the bytes split across the tensor-parallel group. When compute bound, the matmul time replaces it.',
    l2: 'Batch × average context × KV bytes per token ÷ (bandwidth × 0.75).',
    l3: 'Two ring all-reduces per layer per step over the GPU link, plus an assumed 8 µs latency each over NVLink or Infinity Fabric, 25 µs over PCIe.',
    l4: 'Prefill time of the batch\'s prompts spread over their output tokens.',
    chart: 'Aggregate and per-user tokens per second as the batch grows, one replica, from the same step model. The marked batch is the planned operating point.',
    cagg: 'Batch × per-user rate.',
    cusr: '1 ÷ (step time + amortised prefill): falls as the batch\'s KV reads and prefill grow.',
    tp: 'Tensor-parallel degree per replica × number of replicas.',
    batch: 'Requests per replica, and the most that fit its KV cache at this layout (capped at 512).',
    rps: 'Aggregate tokens per second ÷ output tokens per request.',
    rate: 'Dense tensor-core rate the matmuls run at for this weight format, from the datasheet.',
    kvreq: 'KV bytes per token × (prompt + output tokens), reserved for the whole request.',
    kvall: 'KV cache per request × concurrent requests.',
    step: 'One decode step at the operating point: max(memory time, compute time) + communication.',
    comm: 'Tensor-parallel all-reduce time per step and its share of the step. None at TP 1.',
    mem: 'Memory of one GPU drawn as its stacks or channels; memory is interleaved across them, so each shows the same fill.',
    regime: 'Memory-bandwidth bound when the bytes read per step take longer than the matmuls at the rate used; compute bound otherwise.'
  };
  var CARD_TIPS = ['Datasheet: HBM stacks or GDDR channels and the capacity of each.', 'Datasheet peak memory bandwidth; the estimate uses 75% of it.', 'Datasheet dense tensor-core rate for the format the matmuls run in.', 'Streaming multiprocessors (NVIDIA) or compute units (AMD) from the datasheet.', 'GPU-to-GPU link and its usable bandwidth per direction, from the datasheet; it carries the tensor-parallel all-reduces.', 'Datasheet board power.'];
  var MEM_TIPS = ['Weight bytes held by this GPU: params × bytes per parameter ÷ TP.', 'KV cache reserved for the requests on this replica, split across the TP group.', 'Assumed 8% of memory + 1.5 GB for activations, CUDA graphs and allocator slack.', 'Memory left after weights, KV cache and the runtime reserve.'];
  function isPCIe(g) { return /PCIe/i.test(g.link.name); }

  /* ---------- the model ---------- */
  function kvBytesPerToken(m, kvb) {
    var perLayer = m.kvPerLayer ? m.kvPerLayer : 2 * m.kvh * m.hd;
    return m.layers * perLayer * kvb;
  }
  // weight bytes a decode step reads: dense = all; MoE = the experts the batch touches
  function weightsRead(m, wb, b) {
    if (m.active >= m.total) return m.total * 1e9 * wb;
    var p = m.active / m.total;
    return m.total * 1e9 * wb * (1 - Math.pow(1 - p, b));
  }
  // the dense tensor-core rate the matmuls run at for this weight format
  function rate(g, wname) {
    var f = g.flops;
    if (wname === 'bf16') return { tf: f.bf16, as: 'bf16' };
    if (wname === 'int4' && f.fp4) return { tf: f.fp4, as: 'fp4' };
    if (f.fp8) return { tf: f.fp8, as: 'fp8' };
    return { tf: f.bf16, as: 'bf16' };
  }
  // tensor-parallel all-reduces: two per layer; a ring all-reduce moves 2(TP-1)/TP of the tensor through each GPU's link
  function commTime(m, g, tp, tokens) {
    if (tp < 2) return 0;
    var bytes = tokens * m.h * 2 * 2 * (tp - 1) / tp;
    var lat = isPCIe(g) ? LAT_PCIE : LAT_FAST;
    return m.layers * 2 * (bytes / (g.link.gbps * 1e9) + lat);
  }

  // per-replica figures at batch b, tensor-parallel tp
  function perf(s, tp, b) {
    var m = s.m, g = s.g, wb = s.wb, kvTok = s.kvTok;
    var ctxAvg = s.inTok + s.outTok / 2;
    var bw = g.bw * 1e12 * BW_EFF * tp;
    var F = s.rate.tf * 1e12 * tp;
    var tW = weightsRead(m, wb, b) / bw, tKV = b * ctxAvg * kvTok / bw;
    var tMem = tW + tKV;
    var tCmp = 2 * m.active * 1e9 * b / (F * FLOP_EFF);
    var tComm = commTime(m, g, tp, b);
    var tStep = Math.max(tMem, tCmp) + tComm;
    var r = 1 / tStep;                                                   // tok/s per stream, decode only
    var tPrompt = s.inTok * 2 * m.active * 1e9 / (F * PREFILL_EFF) + commTime(m, g, tp, s.inTok); // s per prompt
    var k = b * tPrompt / s.outTok;                                       // prefill share, amortised per decode step
    var rEff = r / (1 + k * r);
    var ttft = tPrompt + tStep;
    return { tStep: tStep, tW: tW, tKV: tKV, tCmp: tCmp, tComm: tComm, tPre: k, user: rEff, agg: rEff * b, ttft: ttft, memBound: tMem >= tCmp };
  }

  function plan(s) {
    var g = s.g, m = s.m;
    var memB = g.mem * GiB;
    var act = 0.08 * memB + 1.5 * GiB;                  // activations, CUDA graphs, allocator slack
    var weights = m.total * 1e9 * s.wb;
    var perReq = s.kvTok * (s.inTok + s.outTok);
    var opts = [];
    for (var i = 0; i < TPS.length; i++) {
      var tp = TPS[i];
      var wPer = weights / tp;
      var kvCap = (memB - act - wPer) * tp;
      if (kvCap <= 0) continue;
      var maxB = Math.min(MAX_BATCH, Math.floor(kvCap / perReq));
      if (maxB < 1) continue;
      var replicas = Math.ceil(s.users / maxB);
      var b = Math.ceil(s.users / replicas);
      var p = perf(s, tp, b);
      opts.push({ tp: tp, replicas: replicas, gpus: tp * replicas, b: b, maxB: maxB, p: p, wPer: wPer, act: act, kvPer: b * perReq / tp, kvCap: kvCap, sloOk: p.ttft * 1000 <= s.slo });
    }
    if (!opts.length) return { fail: true, weights: weights, act: act, memB: memB };
    var ok = opts.filter(function (o) { return o.sloOk; });
    var best;
    if (ok.length) {
      best = ok.reduce(function (a, o) { return o.gpus < a.gpus || (o.gpus === a.gpus && o.tp < a.tp) ? o : a; });
    } else {
      best = opts.reduce(function (a, o) { return o.p.ttft < a.p.ttft ? o : a; });
    }
    best.memB = memB; best.perReq = perReq; best.weights = weights;
    return best;
  }

  /* ---------- rendering ---------- */
  function seg(name, opts, sel, label, tip, optTips) {
    return '<div class="lp-f"><span class="lp-lab" id="' + name + '-l"' + tipAttr(tip) + '>' + label + '</span><div class="lp-seg" role="group" aria-labelledby="' + name + '-l" data-seg="' + name + '">' +
      opts.map(function (o) { return '<button type="button" data-v="' + o[0] + '" aria-pressed="' + (o[0] === sel) + '"' + tipAttr(optTips && optTips[o[0]]) + '>' + o[0] + '</button>'; }).join('') + '</div></div>';
  }
  function range(name, label, arr, idx, fmt, tip) {
    return '<div class="lp-f"><label for="' + name + '"' + tipAttr(tip) + '>' + label + ' <output data-o="' + name + '">' + fmt(arr[idx]) + '</output></label>' +
      '<input type="range" id="' + name + '" data-r="' + name + '" min="0" max="' + (arr.length - 1) + '" step="1" value="' + idx + '"></div>';
  }

  var uid = 0;
  function mount(root) {
    if (root.__lp) return; root.__lp = true;
    if (!HW) { root.innerHTML = '<p class="lp-note">The hardware catalogue (hw.js) did not load.</p>'; return; }
    var u = 'lp' + (++uid) + '-', L = window.VO_LAB;   // the demo shell, when there is one: it owns the GPU choice
    var st = { model: 'qwen32', w: 'fp8', kv: 'fp8', gpu: L ? L.gpu : 'h100', price: HW.gpus[L ? L.gpu : 'h100'].price, users: 6, inp: 3, out: 2, slo: 2 };
    root.innerHTML =
      '<div class="lp">' +
      '<div class="lp-ctl">' +
        '<div class="lp-hd"><i></i>Capacity planner</div>' +
        '<div class="lp-f"><label for="' + u + 'm"' + tipAttr(TIP.model) + '>Model</label><select class="lp-sel" id="' + u + 'm" data-k="model">' +
          MODELS.map(function (m) { return '<option value="' + m.id + '"' + (m.id === st.model ? ' selected' : '') + '>' + m.name + '</option>'; }).join('') + '</select></div>' +
        '<div class="lp-row2">' + seg(u + 'w', WPREC, st.w, 'Weights', TIP.w, { bf16: TIP.bf16, fp8: TIP.fp8, int4: TIP.int4 }) + seg(u + 'kv', KVPREC, st.kv, 'KV cache', TIP.kv, { bf16: TIP.kvbf16, fp8: TIP.kvfp8 }) + '</div>' +
        (L ? '' : '<div class="lp-f"><label for="' + u + 'g" data-tip="The GPU every replica runs on; its memory, bandwidth, tensor rates and link come from the catalogue.">GPU</label><select class="lp-sel" id="' + u + 'g" data-k="gpu">' +
          GPU_ORDER.map(function (k) { var g = HW.gpus[k]; return '<option value="' + k + '"' + (k === st.gpu ? ' selected' : '') + '>' + g.name + ' · ' + g.mem + ' GB</option>'; }).join('') + '</select></div>') +
        '<div class="lp-f lp-inl"><label for="' + u + 'p"' + tipAttr(TIP.price) + '>Price per GPU-hour</label><span class="lp-usd"><input class="lp-num" id="' + u + 'p" data-k="price" type="number" min="0" step="0.05" value="' + st.price.toFixed(2) + '" inputmode="decimal"></span></div>' +
        range(u + 'users', 'Concurrent requests', USERS, st.users, String, TIP.users) +
        range(u + 'inp', 'Prompt tokens', PROMPT, st.inp, fmtTok, TIP.inp) +
        range(u + 'out', 'Output tokens', OUTPUT, st.out, String, TIP.out) +
        range(u + 'slo', 'TTFT target', SLO, st.slo, function (v) { return v + ' ms'; }, TIP.slo) +
      '</div>' +
      '<div class="lp-out" aria-live="polite">' +
        '<div class="lp-kpis">' +
          '<div class="lp-kpi"' + tipAttr(TIP.gpus) + '><small>GPUs needed</small><b data-x="gpus">—</b><span data-x="layout">—</span></div>' +
          '<div class="lp-kpi"' + tipAttr(TIP.user) + '><small>Per-user decode</small><b data-x="user">—</b><span>tok/s per stream</span></div>' +
          '<div class="lp-kpi"' + tipAttr(TIP.agg) + '><small>Aggregate output</small><b data-x="agg">—</b><span>tok/s, all GPUs</span></div>' +
          '<div class="lp-kpi"' + tipAttr(TIP.ttft) + '><small>Time to first token</small><b data-x="ttft">—</b><span data-x="slo">—</span></div>' +
          '<div class="lp-kpi"' + tipAttr(TIP.cpm) + '><small>Per 1M output tokens</small><b data-x="cpm">—</b><span data-x="month">—</span></div>' +
        '</div>' +
        '<div data-x="alert"></div>' +
        '<div class="lp-mid"><div class="lp-card" data-x="card"></div><div class="lp-r">' +
          '<div><div class="lp-sec"><span' + tipAttr(TIP.stk) + '>Time per output token, one stream</span><span data-x="steplab">—</span></div>' +
            '<div class="lp-stk" role="img" data-x="stk"><i style="background:' + C_W + '"></i><i style="background:' + C_KV + '"></i><i style="background:' + C_COMM + '"></i><i style="background:' + C_PRE + '"></i></div>' +
            '<div class="lp-leg"><span' + tipAttr(TIP.l1) + '><i style="background:' + C_W + '"></i><span data-x="l1n">Weights read</span> <em data-x="l1">—</em></span><span data-x="l2s"' + tipAttr(TIP.l2) + '><i style="background:' + C_KV + '"></i>KV cache read <em data-x="l2">—</em></span><span' + tipAttr(TIP.l3) + '><i style="background:' + C_COMM + '"></i>TP all-reduces <em data-x="l3">—</em></span><span' + tipAttr(TIP.l4) + '><i style="background:' + C_PRE + '"></i>Prefill share <em data-x="l4">—</em></span></div></div>' +
          '<div class="lp-chart"><div class="lp-sec"><span' + tipAttr(TIP.chart) + '>Throughput against batch size, one replica</span><span data-x="regime"' + tipAttr(TIP.regime) + '>—</span></div><div data-x="chart"></div>' +
            '<div class="lp-leg"><span' + tipAttr(TIP.cagg) + '><i style="background:' + C_W + '"></i>Aggregate tok/s (left axis)</span><span' + tipAttr(TIP.cusr) + '><i style="background:none;border-top:2px dashed var(--ink3);height:0;width:14px"></i>Per-user tok/s (right axis)</span></div></div>' +
        '</div></div>' +
        '<dl class="lp-tbl" style="margin:0">' +
          '<div><dt' + tipAttr(TIP.tp) + '>Tensor parallel × replicas</dt><dd data-x="tp">—</dd></div>' +
          '<div><dt' + tipAttr(TIP.batch) + '>Batch per replica</dt><dd data-x="batch">—</dd></div>' +
          '<div><dt' + tipAttr(TIP.rps) + '>Requests per second</dt><dd data-x="rps">—</dd></div>' +
          '<div><dt' + tipAttr(TIP.rate) + '>Matmul rate used</dt><dd data-x="rate">—</dd></div>' +
          '<div><dt' + tipAttr(TIP.kvreq) + '>KV cache per request</dt><dd data-x="kvreq">—</dd></div>' +
          '<div><dt' + tipAttr(TIP.kvall) + '>KV cache, all requests</dt><dd data-x="kvall">—</dd></div>' +
          '<div><dt' + tipAttr(TIP.step) + '>Decode step</dt><dd data-x="step">—</dd></div>' +
          '<div><dt' + tipAttr(TIP.comm) + '>Of which communication</dt><dd data-x="comm">—</dd></div>' +
        '</dl>' +
        '<details class="lp-how"><summary>How this is calculated</summary><ul>' +
          '<li>Weights take <code>params × bytes</code> (bf16 2, fp8 1, int4 0.56 with group scales), split across the tensor-parallel group. Each GPU also holds <code>8% of memory + 1.5 GB</code> for activations, CUDA graphs and allocator slack; what is left holds the KV cache. GB means 2<sup>30</sup> bytes, as the driver reports it. The memory drawing shows the GPU\'s HBM stacks (or GDDR channels); memory is interleaved across them, so every stack holds the same share.</li>' +
          '<li>KV cache per token is <code>2 × layers × KV heads × head dim × bytes</code> (DeepSeek-V3 caches a 576-wide latent per layer). Each request reserves room for prompt plus output.</li>' +
          '<li>A decode step is memory bound: <code>(weight bytes read + batch × avg context × KV bytes) ÷ (bandwidth × 0.75)</code>, with a compute floor of <code>2 × active params × batch ÷ (FLOPS × 0.5)</code>. For MoE models, the weight bytes read grow with the experts the batch touches. The 0.75 and 0.5 are assumed efficiencies against the datasheet peak.</li>' +
          '<li>The matmul rate is the GPU\'s dense tensor-core figure for the weight format: bf16 at the bf16 rate; fp8 at the fp8 rate where the GPU has one, otherwise dequantised and run at bf16 (A100); int4 at the fp4 rate on Blackwell, otherwise dequantised to fp8 or bf16.</li>' +
          '<li>Tensor parallelism adds two all-reduces per layer per step over <code>batch × hidden × 2 bytes</code>. With a ring all-reduce each GPU sends and receives <code>2(TP−1)/TP</code> of that through its link, so communication per step is <code>layers × 2 × (bytes ÷ link bandwidth + latency)</code>, with an assumed all-reduce latency of 8 µs over NVLink or Infinity Fabric and 25 µs over PCIe. It is not overlapped with compute here. This is why TP over PCIe (L40S, RTX 5090) costs so much more than over NVLink.</li>' +
          '<li>Prefill is compute bound at 60% of the same matmul rate, plus its own all-reduces. Prefill time is taken out of decode time, so long prompts lower per-user speed; the stacked bar shows that share amortised per output token. TTFT is one prompt\'s prefill plus one decode step, with no queueing.</li>' +
          '<li>The planner tries TP 1, 2, 4 and 8 and picks the layout that meets the TTFT target with the fewest GPUs. Not modelled: prefix caching, speculative decoding, chunked-prefill scheduling, quantisation accuracy, queueing under bursty traffic. That is what an Optimize run measures.</li>' +
        '</ul></details>' +
        '<p class="lp-note" style="margin:0">A first-order estimate from public hardware specs. Optimize measures the real number on your model, traffic and hardware.</p>' +
      '</div></div>';

    var $ = function (sel) { return root.querySelector(sel); };
    var X = {}; [].forEach.call(root.querySelectorAll('[data-x]'), function (e) { X[e.getAttribute('data-x')] = e; });
    var stkI = X.stk.querySelectorAll('i');
    var cardGPU = null, memBar = null;

    root.addEventListener('click', function (e) {
      var b = e.target.closest('.lp-seg button'); if (!b || !root.contains(b)) return;
      var grp = b.parentNode, name = grp.getAttribute('data-seg');
      [].forEach.call(grp.children, function (c) { c.setAttribute('aria-pressed', c === b ? 'true' : 'false'); });
      if (name === u + 'w') st.w = b.getAttribute('data-v'); else st.kv = b.getAttribute('data-v');
      update();
    });
    root.addEventListener('input', function (e) {
      var t = e.target, k = t.getAttribute('data-k'), r = t.getAttribute('data-r');
      if (k === 'model') st.model = t.value;
      else if (k === 'gpu') { st.gpu = t.value; st.price = HW.gpus[st.gpu].price; $('#' + u + 'p').value = st.price.toFixed(2); }
      else if (k === 'price') { var v = parseFloat(t.value); st.price = isFinite(v) && v >= 0 ? v : 0; }
      else if (r) { st[r.slice(u.length)] = +t.value; }
      update();
    });

    // the hardware card: rebuilt when the GPU or the weight format changes; the memory stacks inside it update in place
    function buildCard(g, rt, wname) {
      var fallback = rt.as !== wname && !(wname === 'int4' && rt.as === 'fp4');
      var rows = [
        ['Memory', g.stacks + ' × ' + g.stackGB + ' GB ' + g.memType],
        ['Bandwidth', HW.fmt(g.bw, 2) + ' TB/s'],
        ['Dense ' + rt.as, HW.fmt(rt.tf) + ' TFLOPS'],
        [g.cus ? 'Compute units' : 'SMs', HW.fmt(g.cus || g.sms)],
        ['Interconnect', HW.esc(g.link.name) + ' · ' + g.link.gbps + ' GB/s'],
        ['Power', g.tdp + ' W']
      ];
      var note = fallback ? (wname === 'fp8' ? 'No fp8 tensor cores: fp8 weights are dequantised and the matmuls run at the bf16 rate.' : 'int4 weights are dequantised to ' + rt.as + ' for the matmuls.') : (wname === 'int4' && rt.as === 'fp4' ? 'int4 weights run as NVFP4 at the fp4 tensor-core rate.' : '');
      var c = HW.card(g, { photo: !L, rows: rows, sub: HW.arch[g.arch].name, note: note });
      tipRows(c, CARD_TIPS);
      X.card.innerHTML = ''; X.card.appendChild(c);
      memBar = null;
    }
    function setMem(g, w, kv, act, tp) {
      var parts = [{ label: 'Weights', bytes: w, color: C_W }, { label: 'KV cache', bytes: kv, color: C_KV }, { label: 'Activations and runtime', bytes: act, color: C_ACT }];
      var slot = X.card.querySelector('.hwc-slot');
      if (!memBar) {
        slot.innerHTML = '<div class="lp-sec" style="margin-bottom:6px"><span' + tipAttr(TIP.mem) + '>Memory per GPU</span><span data-x="memcap"></span></div>';
        memBar = HW.memoryBar(g, parts); slot.appendChild(memBar);
        [].forEach.call(memBar.querySelectorAll('.hwm-lg span'), function (sp, i) { if (MEM_TIPS[i]) sp.setAttribute('data-tip', MEM_TIPS[i]); });
      } else {
        var cap = g.mem * GiB, stacks = memBar.querySelectorAll('.hwm-st');
        [].forEach.call(stacks, function (s) {
          var x = 0, is = s.querySelectorAll('i');
          parts.forEach(function (p, i) { var wd = Math.max(0, p.bytes) / cap * 100; is[i].style.left = x + '%'; is[i].style.width = wd + '%'; x += wd; });
        });
        var lg = memBar.querySelectorAll('.hwm-lg span'), used = w + kv + act;
        parts.forEach(function (p, i) { lg[i].textContent = p.label + ' ' + fmtGB(p.bytes); });
        lg[3].textContent = 'free ' + fmtGB(Math.max(0, cap - used));
        memBar.setAttribute('aria-label', fmtGB(used) + ' of ' + g.mem + ' GB used');
      }
      slot.querySelector('[data-x="memcap"]').textContent = g.mem + ' GB' + (tp > 1 ? ' · TP' + tp : '');
    }

    function update() {
      [['users', USERS, String], ['inp', PROMPT, fmtTok], ['out', OUTPUT, String], ['slo', SLO, function (v) { return v + ' ms'; }]].forEach(function (a) {
        var inp = $('#' + u + a[0]); inp.style.setProperty('--p', (st[a[0]] / (a[1].length - 1) * 100) + '%');
        root.querySelector('[data-o="' + u + a[0] + '"]').textContent = a[2](a[1][st[a[0]]]);
      });
      var m = byId(MODELS, st.model), g = HW.gpus[st.gpu];
      var wb = WPREC.filter(function (x) { return x[0] === st.w; })[0][1];
      var kvb = KVPREC.filter(function (x) { return x[0] === st.kv; })[0][1];
      var rt = rate(g, st.w);
      if (cardGPU !== st.gpu + '/' + st.w) { cardGPU = st.gpu + '/' + st.w; buildCard(g, rt, st.w); }
      var s = { m: m, g: g, wb: wb, wname: st.w, rate: rt, kvTok: kvBytesPerToken(m, kvb), users: USERS[st.users], inTok: PROMPT[st.inp], outTok: OUTPUT[st.out], slo: SLO[st.slo] };
      var r = plan(s);
      X.rate.textContent = HW.fmt(rt.tf) + ' TFLOPS ' + rt.as;
      var alert = '';
      if (r.fail) {
        X.gpus.textContent = '> 8'; X.layout.textContent = 'does not fit one node';
        ['user', 'agg', 'ttft', 'cpm', 'tp', 'batch', 'rps', 'kvall', 'step', 'comm'].forEach(function (k) { X[k].textContent = '—'; });
        X.slo.textContent = ''; X.month.textContent = ''; X.slo.className = '';
        X.kvreq.textContent = fmtGB(s.kvTok * (s.inTok + s.outTok));
        X.alert.innerHTML = '<div class="lp-alert lp-err">' + fmtGB(r.weights) + ' of weights plus the KV cache don\'t fit on 8 × ' + g.name + '. Lower-precision weights, a larger GPU or shorter contexts would be needed.</div>';
        var wPer8 = r.weights / 8;
        setMem(g, Math.min(wPer8, r.memB), 0, Math.min(r.act, Math.max(0, r.memB - wPer8)), 8);
        setStack(null);
        X.chart.innerHTML = ''; X.regime.textContent = '';
        return;
      }
      var p = r.p;
      X.gpus.textContent = r.gpus;
      X.layout.textContent = r.replicas > 1 ? r.replicas + ' replicas × TP' + r.tp : 'TP' + r.tp + ', one replica';
      X.user.textContent = fmtN(p.user);
      X.agg.textContent = fmtN(p.agg * r.replicas);
      X.ttft.textContent = fmtMs(p.ttft);
      X.slo.textContent = r.sloOk ? 'within ' + s.slo + ' ms' : 'misses ' + s.slo + ' ms';
      X.slo.className = r.sloOk ? 'lp-ok' : 'lp-bad';
      var hourly = r.gpus * st.price;
      var cpm = hourly / (p.agg * r.replicas * 3600) * 1e6;
      X.cpm.textContent = st.price > 0 ? fmtUSD(cpm) : '—';
      X.month.textContent = st.price > 0 ? fmtUSD(hourly * 730) + ' a month' : 'set a GPU price';
      X.tp.textContent = 'TP' + r.tp + ' × ' + r.replicas;
      X.batch.textContent = r.b + ' (max ' + r.maxB + ')';
      X.rps.textContent = fmtN(p.agg * r.replicas / s.outTok);
      X.kvreq.textContent = fmtGB(r.perReq);
      X.kvall.textContent = fmtGB(r.perReq * s.users);
      X.step.textContent = ms(p.tStep);
      X.comm.textContent = r.tp > 1 ? ms(p.tComm) + ' (' + Math.round(p.tComm / p.tStep * 100) + '%)' : 'none, TP1';
      if (!r.sloOk) alert = '<div class="lp-alert">No layout up to TP8 prefills a ' + fmtTok(s.inTok) + '-token prompt within ' + s.slo + ' ms on ' + g.name + '. Shown: the fastest layout. Prefix caching or a faster GPU would be needed.</div>';
      X.alert.innerHTML = alert;
      setMem(g, r.wPer, r.kvPer, r.act, r.tp);
      setStack(p);
      X.regime.textContent = p.memBound ? 'memory-bandwidth bound' : 'compute bound';
      chart(s, r);
    }

    // stacked bar: how the time per output token splits at the operating point
    function setStack(p) {
      if (!p) { [].forEach.call(stkI, function (i) { i.style.width = '0%'; }); X.steplab.textContent = '—'; ['l1', 'l2', 'l3', 'l4'].forEach(function (k) { X[k].textContent = '—'; }); return; }
      var a = p.memBound ? p.tW : p.tCmp, b = p.memBound ? p.tKV : 0, total = a + b + p.tComm + p.tPre;
      var seg = [a, b, p.tComm, p.tPre];
      seg.forEach(function (v, i) { stkI[i].style.width = (v / total * 100).toFixed(2) + '%'; });
      X.l1n.textContent = p.memBound ? 'Weights read' : 'Matmuls (compute bound)';
      X.l2s.style.display = p.memBound ? '' : 'none';
      X.l1.textContent = ms(a); X.l2.textContent = ms(b); X.l3.textContent = ms(p.tComm); X.l4.textContent = ms(p.tPre);
      X.steplab.textContent = ms(total) + ' = ' + fmtN(1 / total) + ' tok/s';
      X.stk.setAttribute('aria-label', 'Per output token: ' + X.l1n.textContent + ' ' + ms(a) + (p.memBound ? ', KV read ' + ms(b) : '') + ', communication ' + ms(p.tComm) + ', prefill share ' + ms(p.tPre));
    }

    function nice(v) { var e = Math.pow(10, Math.floor(Math.log10(v))), f = v / e; return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * e; }
    function chart(s, r) {
      var W = Math.round(Math.max(300, Math.min(640, X.chart.clientWidth || 600))), H = Math.round(Math.max(180, W * 0.36)), L = 40, R = 34, T = 12, B = 26;
      var maxB = Math.max(2, r.maxB), n = 48, pts = [];
      for (var i = 0; i <= n; i++) { var b = Math.max(1, Math.round(Math.exp(Math.log(maxB) * i / n))); if (!pts.length || pts[pts.length - 1].b !== b) pts.push({ b: b, p: perf(s, r.tp, b) }); }
      var aMax = nice(Math.max.apply(null, pts.map(function (q) { return q.p.agg; })) * 1.08);
      var uMax = nice(Math.max.apply(null, pts.map(function (q) { return q.p.user; })) * 1.08);
      var x = function (b) { return L + Math.log(b) / Math.log(maxB) * (W - L - R); };
      var ya = function (v) { return T + (1 - v / aMax) * (H - T - B); };
      var yu = function (v) { return T + (1 - v / uMax) * (H - T - B); };
      var o = ['<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Aggregate and per-user tokens per second against batch size">'];
      for (var k = 0; k <= 4; k++) {
        var yy = T + k / 4 * (H - T - B);
        o.push('<line class="lp-grid" x1="' + L + '" x2="' + (W - R) + '" y1="' + yy.toFixed(1) + '" y2="' + yy.toFixed(1) + '"/>');
        o.push('<text x="' + (L - 6) + '" y="' + (yy + 3).toFixed(1) + '" text-anchor="end">' + short(aMax * (1 - k / 4)) + '</text>');
        o.push('<text x="' + (W - R + 6) + '" y="' + (yy + 3).toFixed(1) + '">' + short(uMax * (1 - k / 4)) + '</text>');
      }
      for (var t = 1; t <= maxB; t *= 4) o.push('<text x="' + x(t).toFixed(1) + '" y="' + (H - 8) + '" text-anchor="middle">' + t + '</text>');
      o.push('<text x="' + (W - R) + '" y="' + (H - 8) + '" text-anchor="end">batch</text>');
      var da = pts.map(function (q, j) { return (j ? 'L' : 'M') + x(q.b).toFixed(1) + ' ' + ya(q.p.agg).toFixed(1); }).join('');
      var du = pts.map(function (q, j) { return (j ? 'L' : 'M') + x(q.b).toFixed(1) + ' ' + yu(q.p.user).toFixed(1); }).join('');
      o.push('<path class="lp-area" d="' + da + 'L' + x(pts[pts.length - 1].b).toFixed(1) + ' ' + (H - B) + 'L' + x(1).toFixed(1) + ' ' + (H - B) + 'Z"/>');
      o.push('<path class="lp-agg" d="' + da + '"/><path class="lp-usr" d="' + du + '"/>');
      var ob = Math.min(r.b, maxB), ox = x(ob), op = r.p;
      o.push('<line class="lp-op" x1="' + ox.toFixed(1) + '" x2="' + ox.toFixed(1) + '" y1="' + T + '" y2="' + (H - B) + '"/>');
      o.push('<circle class="lp-dot" cx="' + ox.toFixed(1) + '" cy="' + ya(op.agg).toFixed(1) + '" r="4"/><circle class="lp-dot2" cx="' + ox.toFixed(1) + '" cy="' + yu(op.user).toFixed(1) + '" r="3.5"/>');
      var right = ox > W - R - 110;
      o.push('<text class="lp-oplab" x="' + (ox + (right ? -6 : 6)).toFixed(1) + '" y="' + (T + 10) + '" text-anchor="' + (right ? 'end' : 'start') + '">batch ' + r.b + '</text>');
      o.push('</svg>');
      X.chart.innerHTML = o.join('');
    }
    function short(v) { return v >= 1e4 ? Math.round(v / 1e3) + 'k' : v >= 1e3 ? (v / 1e3).toFixed(1).replace(/\.0$/, '') + 'k' : String(Math.round(v)); }

    if (L) L.subscribe(function (m) { if (m.kind === 'gpu' && m.gpu !== st.gpu) { st.gpu = m.gpu; st.price = HW.gpus[st.gpu].price; $('#' + u + 'p').value = st.price.toFixed(2); update(); } });
    update();
    if ('ResizeObserver' in window) {
      var lastW = 0;
      new ResizeObserver(function () { var w = X.chart.clientWidth; if (Math.abs(w - lastW) > 24) { lastW = w; update(); } }).observe(X.chart);
    }
  }

  function init() { injectCSS(); [].forEach.call(document.querySelectorAll('[data-lab="planner"]'), mount); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
