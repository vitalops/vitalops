/* Hardware catalogue shared by the live demos (assets/js/hw.js).
   Public specifications only; every figure below comes from the vendor's datasheet or the CUDA
   programming guide's compute-capability table, and the notes say where a figure is an assumption.
   Nothing here is a measurement of any VitalOps product.

   window.VO_HW = { gpus, devices, arch, photoURL, card(spec, opts), carousel(kind, key, onSelect), memoryBar(spec, parts), fmt, gb }
*/
(function () {
  'use strict';
  var ROOT = (function () {
    if (window.VO_HW_ROOT) return window.VO_HW_ROOT;      // a host page may point the photos elsewhere (vitalops ui does)
    var s = document.currentScript && document.currentScript.src;
    return s ? s.replace(/assets\/js\/hw\.js.*$/, '') : '';
  })();

  /* ── compute-capability rules used by the occupancy calculator (CUDA programming guide, table "Technical specifications per compute capability") ── */
  var ARCH = {
    sm_80:  { name: 'Ampere (sm_80)',    regsPerSM: 65536, regAllocUnit: 256, warpAllocGran: 4, maxRegsPerThread: 255, maxWarpsPerSM: 64, maxThreadsPerSM: 2048, maxBlocksPerSM: 32, smemPerSM: 164 * 1024, smemPerBlockMax: 163 * 1024, smemAllocUnit: 128, smemReservedPerBlock: 1024, warpSize: 32 },
    sm_87:  { name: 'Ampere (sm_87)',    regsPerSM: 65536, regAllocUnit: 256, warpAllocGran: 4, maxRegsPerThread: 255, maxWarpsPerSM: 48, maxThreadsPerSM: 1536, maxBlocksPerSM: 16, smemPerSM: 164 * 1024, smemPerBlockMax: 163 * 1024, smemAllocUnit: 128, smemReservedPerBlock: 1024, warpSize: 32 },
    sm_89:  { name: 'Ada (sm_89)',       regsPerSM: 65536, regAllocUnit: 256, warpAllocGran: 4, maxRegsPerThread: 255, maxWarpsPerSM: 48, maxThreadsPerSM: 1536, maxBlocksPerSM: 24, smemPerSM: 100 * 1024, smemPerBlockMax: 99 * 1024,  smemAllocUnit: 128, smemReservedPerBlock: 1024, warpSize: 32 },
    sm_90:  { name: 'Hopper (sm_90)',    regsPerSM: 65536, regAllocUnit: 256, warpAllocGran: 4, maxRegsPerThread: 255, maxWarpsPerSM: 64, maxThreadsPerSM: 2048, maxBlocksPerSM: 32, smemPerSM: 228 * 1024, smemPerBlockMax: 227 * 1024, smemAllocUnit: 128, smemReservedPerBlock: 1024, warpSize: 32 },
    sm_100: { name: 'Blackwell (sm_100)', regsPerSM: 65536, regAllocUnit: 256, warpAllocGran: 4, maxRegsPerThread: 255, maxWarpsPerSM: 64, maxThreadsPerSM: 2048, maxBlocksPerSM: 32, smemPerSM: 228 * 1024, smemPerBlockMax: 227 * 1024, smemAllocUnit: 128, smemReservedPerBlock: 1024, warpSize: 32 },
    sm_120: { name: 'Blackwell (sm_120)', regsPerSM: 65536, regAllocUnit: 256, warpAllocGran: 4, maxRegsPerThread: 255, maxWarpsPerSM: 48, maxThreadsPerSM: 1536, maxBlocksPerSM: 24, smemPerSM: 100 * 1024, smemPerBlockMax: 99 * 1024,  smemAllocUnit: 128, smemReservedPerBlock: 1024, warpSize: 32 },
    /* CDNA3: a compute unit has four SIMDs, each with a 512-register (128 KB) VGPR file and at most 8 waves; wave64. */
    cdna3:  { name: 'CDNA 3 (gfx942)', amd: true, vgprPerSIMD: 512, vgprAllocUnit: 8, maxVgprPerWave: 512, maxWavesPerSIMD: 8, simdsPerCU: 4, ldsPerCU: 64 * 1024, ldsAllocUnit: 512, maxWorkgroupsPerCU: 32, warpSize: 64 }
  };

  /* ── data-centre and workstation GPUs ──
     mem: GB usable. stacks: HBM stacks (or GDDR channels) and GB each, for the memory drawing. bw: TB/s peak. flops: dense TFLOPS by dtype (fp32 is the vector rate, the rest tensor-core rates).
     link: interconnect usable per direction, GB/s (NVLink figure is the per-GPU total split in two directions; PCIe is the Gen x16 figure).
     latency: assumed DRAM latency in ns for the Little's-law estimate (microbenchmarks put HBM3 at roughly 600-800 ns; GDDR a little lower). price: default $/GPU-hour, editable in the demos. */
  var GPUS = {
    h100:  { name: 'H100 SXM',  vendor: 'nvidia', arch: 'sm_90',  sms: 132, mem: 80,  memType: 'HBM3',   stacks: 5, stackGB: 16,  bw: 3.35, flops: { fp32: 67, bf16: 989,  fp8: 1979, int8: 1979 },           l2: 50, link: { name: 'NVLink 4', gbps: 450 }, tdp: 700, latency: 650, price: 2.50, photo: 'h100', photoNote: 'H100 PCIe card pictured (same GH100 die as the SXM module).' },
    h200:  { name: 'H200 SXM',  vendor: 'nvidia', arch: 'sm_90',  sms: 132, mem: 141, memType: 'HBM3e',  stacks: 6, stackGB: 24,  bw: 4.8,  flops: { fp32: 67, bf16: 989,  fp8: 1979, int8: 1979 },           l2: 50, link: { name: 'NVLink 4', gbps: 450 }, tdp: 700, latency: 650, price: 3.50, photo: 'h100', photoNote: 'H100 card pictured; no licensed H200 photo (same die, HBM3e).' },
    b200:  { name: 'B200',      vendor: 'nvidia', arch: 'sm_100', sms: 148, mem: 180, memType: 'HBM3e',  stacks: 8, stackGB: 24,  bw: 8.0,  flops: { fp32: 80, bf16: 2250, fp8: 4500, int8: 4500, fp4: 9000 }, l2: 126, link: { name: 'NVLink 5', gbps: 900 }, tdp: 1000, latency: 700, price: 5.50, photo: 'gb200', photoNote: 'GB200 superchip board: two Blackwell GPUs and a Grace CPU (Computex 2024).' },
    a100:  { name: 'A100 SXM 80 GB', vendor: 'nvidia', arch: 'sm_80', sms: 108, mem: 80, memType: 'HBM2e', stacks: 5, stackGB: 16, bw: 2.0, flops: { fp32: 19.5, bf16: 312, int8: 624 },                   l2: 40, link: { name: 'NVLink 3', gbps: 300 }, tdp: 400, latency: 550, price: 1.60, photo: 'a100', photoNote: 'A100 PCIe card pictured.' },
    l40s:  { name: 'L40S',      vendor: 'nvidia', arch: 'sm_89',  sms: 142, mem: 48,  memType: 'GDDR6',  stacks: 12, stackGB: 4,  bw: 0.864, flops: { fp32: 91.6, bf16: 362, fp8: 733, int8: 733 },             l2: 96, link: { name: 'PCIe 4.0 x16', gbps: 25 }, tdp: 350, latency: 550, price: 1.00, photo: 'l40s', photoNote: 'NVIDIA L40S (product image).' },
    rtx5090: { name: 'RTX 5090', vendor: 'nvidia', arch: 'sm_120', sms: 170, mem: 32, memType: 'GDDR7', stacks: 16, stackGB: 2, bw: 1.79, flops: { fp32: 104.8, bf16: 209, fp8: 419, int8: 419, fp4: 838 },      l2: 96, link: { name: 'PCIe 5.0 x16', gbps: 50 }, tdp: 575, latency: 550, price: 0.70, photo: 'rtx5090', photoNote: 'Palit GameRock board pictured; memory and compute are the same GB202 on every board.' },
    mi300x: { name: 'MI300X',   vendor: 'amd',    arch: 'cdna3',  cus: 304, mem: 192, memType: 'HBM3',   stacks: 8, stackGB: 24,  bw: 5.3,  flops: { fp32: 163.4, bf16: 1307, fp8: 2615, int8: 2615 },           l2: 256, link: { name: 'Infinity Fabric', gbps: 448 }, tdp: 750, latency: 850, price: 2.50, photo: 'mi300x', photoNote: 'AMD Instinct MI300-series OAM module.' },
    /* Jetson modules share LPDDR5 between CPU and GPU; "mem" is the whole module. */
    agx_orin: { name: 'Jetson AGX Orin 64 GB', vendor: 'nvidia', arch: 'sm_87', sms: 16, mem: 64, memType: 'LPDDR5', stacks: 8, stackGB: 8, bw: 0.2048, flops: { fp32: 5.3, bf16: 42, int8: 85 }, l2: 4, link: { name: 'PCIe 4.0 x8', gbps: 12 }, tdp: 60, latency: 450, price: 0, photo: 'agx-orin', photoNote: 'AGX Orin Developer Kit pictured.' }
  };

  /* ── edge devices for the fit demo. usable: GB one process can expect (assumption, see note). bw: GB/s. ── */
  var DEVICES = {
    orin_nano: { name: 'Jetson Orin Nano Super 8 GB', mem: 8,  usable: 6.2, bw: 102,  kind: 'Jetson, 1024 CUDA cores, sm_87', photo: 'orin-nx', photoNote: 'Orin NX module pictured; Orin Nano uses the same module format.', eff: 0.65, note: 'Usable memory assumes JetPack with the desktop disabled.' },
    orin_nx:   { name: 'Jetson Orin NX 16 GB',        mem: 16, usable: 13.5, bw: 102,  kind: 'Jetson, 1024 CUDA cores, sm_87', photo: 'orin-nx', photoNote: 'Orin NX module on a carrier board.', eff: 0.65 },
    agx_orin:  { name: 'Jetson AGX Orin 64 GB',       mem: 64, usable: 58,  bw: 204.8, kind: 'Jetson, 2048 CUDA cores, sm_87', photo: 'agx-orin', photoNote: 'AGX Orin Developer Kit.', eff: 0.65 },
    pi5:       { name: 'Raspberry Pi 5 8 GB',         mem: 8,  usable: 6.5, bw: 17,   kind: 'Arm Cortex-A76 ×4, LPDDR4X', photo: 'pi5', photoNote: 'Raspberry Pi 5 board.', eff: 0.6, note: 'CPU inference; a Hailo module does not run language models.' },
    iphone16:  { name: 'iPhone 16 Pro 8 GB',          mem: 8,  usable: 4.5, bw: 60,   kind: 'A18 Pro, LPDDR5X', photo: 'iphone16pro', photoNote: 'iPhone 16 Pro.', eff: 0.6, note: 'iOS limits one app to roughly 4-5 GB before it is killed.' },
    pixel9:    { name: 'Pixel 9 Pro 16 GB',           mem: 16, usable: 8,   bw: 68,   kind: 'Tensor G4, LPDDR5X', photo: 'pixel9pro', photoNote: 'Pixel 9 and 9 Pro.', eff: 0.6, note: 'Android keeps a large share for the system and other apps.' },
    s25u:      { name: 'Galaxy S25 Ultra 12 GB',      mem: 12, usable: 7,   bw: 77,   kind: 'Snapdragon 8 Elite, LPDDR5X', photo: 's25-ultra', photoNote: 'Galaxy S25 Ultra.', eff: 0.6 },
    mba:       { name: 'MacBook Air M3 16 GB',        mem: 16, usable: 12,  bw: 100,  kind: 'Apple M3, unified memory', photo: 'macbook-air', photoNote: 'MacBook Air (M2 model pictured; the M3 chassis is identical).', eff: 0.7 }
  };

  var PHOTOS = {
    h100: 'h100', a100: 'a100', b200: 'b200', gb200: 'gb200', mi300x: 'mi300x', l40s: 'l40s', rtx5090: 'rtx5090', 'agx-orin': 'agx-orin', 'orin-nx': 'orin-nx', pi5: 'pi5', iphone16pro: 'iphone16pro', pixel9pro: 'pixel9pro', 'macbook-air': 'macbook-air', 's25-ultra': 's25-ultra'
  };
  function photoURL(key) { return key && PHOTOS[key] ? ROOT + 'assets/photos/hw/' + PHOTOS[key] + '.webp' : null; }

  /* ── formatting ── */
  function fmt(n, d) { return Number(n).toLocaleString('en-GB', { maximumFractionDigits: d == null ? 0 : d, minimumFractionDigits: d == null ? 0 : d }); }
  function gb(bytes) { var g = bytes / 1073741824; return g >= 100 ? fmt(g, 0) + ' GB' : g >= 10 ? fmt(g, 1) + ' GB' : g >= 1 ? fmt(g, 2) + ' GB' : fmt(bytes / 1048576, 0) + ' MB'; }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }

  /* ── the hardware card: photo (or a labelled silhouette), name, spec rows, and a slot for a drawing ── */
  var CSS_ID = 'vo-hw-css';
  function css() {
    if (document.getElementById(CSS_ID)) return;
    var s = document.createElement('style'); s.id = CSS_ID;
    s.textContent = [
      '.hwc{display:grid;grid-template-columns:minmax(0,1fr);border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--white);font-family:var(--font)}',
      '.hwc-ph{position:relative;aspect-ratio:3/2;background:#0b1020;overflow:hidden}',
      '.hwc-ph img{width:100%;height:100%;object-fit:cover;display:block}',
      '.hwc-ph.none{display:grid;place-items:center;color:#94a3b8}',
      '.hwc-ph.none svg{width:60%;height:auto;opacity:.8}',
      /* carousel */
      '.hwcar{position:relative;display:grid;grid-template-columns:32px minmax(0,1fr) 32px;align-items:center;gap:6px}',
      '.hwcar-track{display:flex;gap:10px;overflow-x:auto;scroll-snap-type:x proximity;scrollbar-width:none;padding:6px 4px;scroll-padding:0 40px;cursor:grab;user-select:none;-webkit-user-select:none}',
      '.hwcar-track.dragging{cursor:grabbing;scroll-snap-type:none;scroll-behavior:auto}',
      '.hwcar-track::-webkit-scrollbar{display:none}',
      '.hwcar-it{flex:0 0 172px;scroll-snap-align:center;display:grid;gap:4px;padding:6px 6px 8px;border:1px solid var(--line);border-radius:10px;background:var(--white);color:var(--ink);font:inherit;text-align:left;cursor:pointer;opacity:.72;transition:opacity .15s,border-color .15s,transform .15s}',
      '.hwcar-it:hover{opacity:1;border-color:var(--brand200)}',
      '.hwcar-it.on{opacity:1;border-color:var(--brand);box-shadow:0 0 0 3px var(--brand100);transform:translateY(-2px)}',
      '.hwcar-it:focus-visible{outline:2px solid var(--brand);outline-offset:2px}',
      '.hwcar-it .hwc-ph{border-radius:6px;aspect-ratio:16/10}',
      '.hwcar-n{font-size:12.5px;font-weight:600;color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:4px}',
      '.hwcar-s{font:11px/1.3 var(--mono);color:var(--ink3);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}',
      '.hwcar-ar{width:32px;height:32px;border-radius:50%;border:1px solid var(--line);background:var(--white);color:var(--ink2);font-size:20px;line-height:1;cursor:pointer;display:grid;place-items:center}',
      '.hwcar-ar:hover{border-color:var(--brand200);color:var(--brand)}',
      '@media(max-width:600px){.hwcar{grid-template-columns:minmax(0,1fr)}.hwcar-ar{display:none}.hwcar-it{flex-basis:150px}}',
      '@media (prefers-reduced-motion:reduce){.hwcar-it{transition:none}}',
      '.hwc-b{padding:10px 12px 12px}',
      '.hwc-b b{display:block;font-size:14px;color:var(--ink);letter-spacing:-.01em}',
      '.hwc-b small{display:block;font-size:11.5px;color:var(--ink3);margin-top:1px}',
      '.hwc-kv{display:grid;grid-template-columns:auto minmax(0,1fr);column-gap:10px;row-gap:3px;margin-top:8px;font-size:11.5px}',
      '.hwc-kv dt{color:var(--ink3)}.hwc-kv dd{margin:0;font-family:var(--mono);color:var(--ink);text-align:right;font-variant-numeric:tabular-nums}',
      '.hwc-slot{margin-top:10px}',
      '.hwc-note{font-size:10.5px;color:var(--ink4);margin-top:8px;line-height:1.4}',
      /* memory drawing: one block per HBM stack or GDDR channel; fill is interleaved, so every stack shows the same share */
      '.hwm{display:grid;gap:4px}',
      '.hwm-row{display:grid;gap:3px}',
      '.hwm-st{position:relative;height:18px;border:1px solid var(--line);border-radius:3px;background:var(--line2);overflow:hidden}',
      '.hwm-st i{position:absolute;top:0;bottom:0;left:0;display:block;transition:width .35s ease}',
      '.hwm-lg{display:flex;flex-wrap:wrap;gap:4px 12px;font-size:11px;color:var(--ink3)}',
      '.hwm-lg span::before{content:"";display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;vertical-align:-1px;background:var(--c);box-shadow:inset 0 0 0 1px rgba(15,23,42,.18)}',
      '@media (prefers-reduced-motion:reduce){.hwm-st i{transition:none}}'
    ].join('\n');
    document.head.appendChild(s);
  }

  function silhouette(kind) {
    if (kind === 'oam') return '<svg viewBox="0 0 120 70" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="10" y="8" width="100" height="54" rx="3"/><rect x="38" y="22" width="44" height="26" rx="2"/><path d="M14 62v4M30 62v4M46 62v4M62 62v4M78 62v4M94 62v4M106 62v4"/></svg>';
    return '<svg viewBox="0 0 120 70" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M8 14h104v40H8z"/><path d="M18 54v8h60v-8"/><circle cx="36" cy="34" r="12"/><circle cx="74" cy="34" r="12"/><path d="M8 20h-4v28h4"/></svg>';
  }

  function photoHTML(spec) {
    var url = photoURL(spec.photo);
    return url ? '<div class="hwc-ph"><img src="' + url + '" alt="" loading="lazy" title="' + esc(spec.photoNote || spec.name) + '"></div>'
               : '<div class="hwc-ph none" title="' + esc(spec.photoNote || 'No licensed photo') + '">' + silhouette(spec.arch === 'cdna3' ? 'oam' : 'card') + '</div>';
  }

  /* ── the carousel: a row of hardware cards that is the selector. kind 'gpu' or 'device'. ──
     carousel(kind, selectedKey, onSelect, opts) -> element with .select(key, silent) and .setKind(kind, selectedKey) */
  function carousel(kind, selected, onSelect, opts) {
    css(); opts = opts || {};
    var el = document.createElement('div'); el.className = 'hwcar';
    el.innerHTML = '<button class="hwcar-ar l" type="button" aria-label="Previous">&#8249;</button><div class="hwcar-track" role="radiogroup"></div><button class="hwcar-ar r" type="button" aria-label="Next">&#8250;</button>';
    var track = el.querySelector('.hwcar-track'), cur = selected, curKind = kind;
    function list() { var src = curKind === 'device' ? DEVICES : GPUS; return Object.keys(src).filter(function (k) { return !(opts.exclude && opts.exclude.indexOf(k) >= 0); }); }
    function spec(k) { return (curKind === 'device' ? DEVICES : GPUS)[k]; }
    function specLine(k) {
      var g = spec(k);
      if (curKind === 'device') return esc(g.mem + ' GB · ' + g.bw + ' GB/s');
      return esc(g.mem + ' GB · ' + (g.bw >= 1 ? g.bw.toFixed(g.bw >= 10 ? 0 : 2).replace(/\.?0+$/, '') + ' TB/s' : Math.round(g.bw * 1000) + ' GB/s'));
    }
    function build() {
      track.innerHTML = list().map(function (k) {
        var g = spec(k);
        return '<button class="hwcar-it' + (k === cur ? ' on' : '') + '" type="button" role="radio" aria-checked="' + (k === cur) + '" data-key="' + k + '" tabindex="' + (k === cur ? 0 : -1) + '">' +
          photoHTML(g) + '<span class="hwcar-n">' + esc(g.name) + '</span><span class="hwcar-s">' + specLine(k) + '</span></button>';
      }).join('');
    }
    function reveal(k) { var b = track.querySelector('[data-key="' + k + '"]'); if (b) b.scrollIntoView({ block: 'nearest', inline: 'center', behavior: reduced() ? 'auto' : 'smooth' }); }
    function reduced() { return window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches; }
    function select(k, silent, focus) {
      if (!spec(k)) return;
      cur = k;
      [].forEach.call(track.children, function (b) { var on = b.getAttribute('data-key') === k; b.classList.toggle('on', on); b.setAttribute('aria-checked', on); b.tabIndex = on ? 0 : -1; if (on && focus) b.focus({ preventScroll: true }); });
      reveal(k);
      if (!silent && onSelect) onSelect(curKind, k, spec(k));
    }
    dragScroll(track);
    track.addEventListener('click', function (e) { var b = e.target.closest('.hwcar-it'); if (b) select(b.getAttribute('data-key')); });
    track.addEventListener('keydown', function (e) {
      var d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0; if (!d) return;
      e.preventDefault(); var ks = list(), i = ks.indexOf(cur); select(ks[(i + d + ks.length) % ks.length], false, true);
    });
    el.querySelector('.hwcar-ar.l').addEventListener('click', function () { var ks = list(), i = ks.indexOf(cur); select(ks[(i - 1 + ks.length) % ks.length]); });
    el.querySelector('.hwcar-ar.r').addEventListener('click', function () { var ks = list(), i = ks.indexOf(cur); select(ks[(i + 1) % ks.length]); });
    el.select = select;
    el.setKind = function (k, sel) { if (k !== curKind) { curKind = k; build(); } if (sel) select(sel, true); else reveal(cur); };
    el.current = function () { return { kind: curKind, key: cur, spec: spec(cur) }; };
    build(); setTimeout(function () { reveal(cur); }, 0);
    return el;
  }

  /* mouse drag on a horizontally scrolling row; touch already scrolls natively. A drag suppresses the click that follows it. */
  function dragScroll(el) {
    var x0 = 0, s0 = 0, moved = false, down = false, swallow = false;
    el.addEventListener('pointerdown', function (e) { if (e.pointerType !== 'mouse' || e.button) return; down = true; moved = false; x0 = e.clientX; s0 = el.scrollLeft; });
    el.addEventListener('pointermove', function (e) {
      if (!down) return; var dx = e.clientX - x0;
      if (!moved && Math.abs(dx) > 4) { moved = true; el.classList.add('dragging'); try { el.setPointerCapture(e.pointerId); } catch (x) {} }
      if (moved) { el.scrollLeft = s0 - dx; e.preventDefault(); }
    });
    /* the click that ends a drag is swallowed; the flag clears right after it, so the next plain click works */
    function up(e) { if (!down) return; down = false; el.classList.remove('dragging'); if (moved) { swallow = true; setTimeout(function () { swallow = false; }, 0); } moved = false; }
    el.addEventListener('pointerup', up); el.addEventListener('pointercancel', up); el.addEventListener('pointerleave', up);
    el.addEventListener('click', function (e) { if (swallow) { e.stopPropagation(); e.preventDefault(); } }, true);
    el.addEventListener('dragstart', function (e) { e.preventDefault(); });
    return el;
  }

  function card(spec, opts) {
    css(); opts = opts || {};
    var el = document.createElement('div'); el.className = 'hwc' + (opts.cls ? ' ' + opts.cls : '');
    var ph = opts.photo === false ? '' : photoHTML(spec);
    var rows = (opts.rows || []).map(function (r) { return '<dt>' + esc(r[0]) + '</dt><dd>' + r[1] + '</dd>'; }).join('');
    el.innerHTML = ph + '<div class="hwc-b"><b>' + esc(spec.name) + '</b>' + (opts.sub ? '<small>' + esc(opts.sub) + '</small>' : '') +
      (rows ? '<dl class="hwc-kv">' + rows + '</dl>' : '') + '<div class="hwc-slot"></div>' + (opts.note ? '<p class="hwc-note">' + esc(opts.note) + '</p>' : '') + '</div>';
    return el;
  }

  /* Draws the GPU's memory as its stacks. parts: [{label, bytes, color}] in order; the total capacity is spec.mem GB.
     Memory is interleaved across stacks, so each stack shows the same proportions; that is how the hardware places it. */
  function memoryBar(spec, parts) {
    css();
    var cap = spec.mem * 1073741824, n = spec.stacks || 1, perRow = n > 8 ? Math.ceil(n / 2) : n;
    var wrap = document.createElement('div'); wrap.className = 'hwm';
    var rows = Math.ceil(n / perRow), html = '';
    for (var r = 0; r < rows; r++) {
      html += '<div class="hwm-row" style="grid-template-columns:repeat(' + perRow + ',minmax(0,1fr))">';
      for (var k = 0; k < perRow && r * perRow + k < n; k++) {
        var x = 0, segs = '';
        parts.forEach(function (p) { var w = Math.max(0, p.bytes) / cap * 100; segs += '<i style="left:' + x + '%;width:' + w + '%;background:' + p.color + '"></i>'; x += w; });
        html += '<div class="hwm-st" title="' + esc((spec.memType || 'memory') + ' ' + (spec.stackGB || '') + ' GB') + '">' + segs + '</div>';
      }
      html += '</div>';
    }
    var used = parts.reduce(function (a, p) { return a + p.bytes; }, 0);
    html += '<div class="hwm-lg">' + parts.map(function (p) { return '<span style="--c:' + p.color + '">' + esc(p.label) + ' ' + gb(p.bytes) + '</span>'; }).join('') +
      '<span style="--c:var(--line2)">free ' + gb(Math.max(0, cap - used)) + '</span></div>';
    wrap.innerHTML = html;
    wrap.setAttribute('aria-label', n + ' × ' + (spec.stackGB || '') + ' GB ' + (spec.memType || '') + ', ' + gb(used) + ' of ' + spec.mem + ' GB used');
    return wrap;
  }

  window.VO_HW = { gpus: GPUS, devices: DEVICES, arch: ARCH, photoURL: photoURL, photoHTML: photoHTML, card: card, carousel: carousel, dragScroll: dragScroll, memoryBar: memoryBar, fmt: fmt, gb: gb, esc: esc };
})();
