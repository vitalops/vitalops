/* The demo shell (assets/js/lab-shell.js): a dark panel with the hardware carousel as the selector and one tab per demo.

   <div class="lab" data-lab-shell data-tabs="planner,batching,kernel" data-gpu="h100" data-device="orin_nano"></div>

   It builds a mount <div data-lab="name"> for each tab; the demo scripts (loaded after this one) render into them.
   Demos read the selection from window.VO_LAB and subscribe to changes; a demo running without a shell keeps its own selector.

   window.VO_LAB = { gpu, device, select(kind, key), subscribe(fn) -> unsubscribe }   fn({ kind, gpu, device, spec })
*/
(function () {
  'use strict';
  var HW = window.VO_HW; if (!HW) return;
  /* icons: Lucide (ISC) */
  var ICON = {
    gauge: '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    layers: '<path d="M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83z"/><path d="M2 12a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 12"/><path d="M2 17a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 17"/>',
    cpu: '<path d="M12 20v2M12 2v2M17 20v2M17 2v2M2 12h2M2 17h2M2 7h2M20 12h2M20 17h2M20 7h2M7 20v2M7 2v2"/><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="8" y="8" width="8" height="8" rx="1"/>',
    boxes: '<path d="M2.97 12.92A2 2 0 0 0 2 14.63v3.24a2 2 0 0 0 .97 1.71l3 1.8a2 2 0 0 0 2.06 0L12 19v-5.5l-5-3-4.03 2.42Z"/><path d="m7 16.5-4.74-2.85M7 16.5l5-3M7 16.5v5.17"/><path d="M12 13.5V19l3.97 2.38a2 2 0 0 0 2.06 0l3-1.8a2 2 0 0 0 .97-1.71v-3.24a2 2 0 0 0-.97-1.71L17 10.5l-5 3Z"/><path d="m17 16.5-5-3M17 16.5l4.74-2.85M17 16.5v5.17"/><path d="M7.97 4.42A2 2 0 0 0 7 6.13v4.37l5 3 5-3V6.13a2 2 0 0 0-.97-1.71l-3-1.8a2 2 0 0 0-2.06 0l-3 1.8Z"/><path d="M12 8 7.26 5.15M12 8l4.74-2.85M12 13.5V8"/>',
    zap: '<path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/>',
    smartphone: '<rect width="14" height="20" x="5" y="2" rx="2" ry="2"/><path d="M12 18h.01"/>'
  };
  function icon(n) { return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + ICON[n] + '</svg>'; }
  var TABS = {
    planner:  { label: 'Capacity planner', kind: 'gpu', icon: 'gauge' },
    batching: { label: 'Batching simulator', kind: 'gpu', icon: 'layers' },
    kernel:   { label: 'Kernel lab', kind: 'gpu', icon: 'cpu' },
    quant:    { label: 'Quantisation lab', kind: 'gpu', icon: 'boxes' },
    specdec:  { label: 'Speculative decoding', kind: 'gpu', icon: 'zap' },
    edgefit:  { label: 'Edge fit', kind: 'device', icon: 'smartphone' }
  };
  var state = { gpu: 'h100', device: 'orin_nano' }, subs = [];
  function notify(kind) {
    var msg = { kind: kind, gpu: state.gpu, device: state.device, spec: kind === 'device' ? HW.devices[state.device] : HW.gpus[state.gpu] };
    subs.slice().forEach(function (f) { try { f(msg); } catch (e) { if (window.console) console.error(e); } });
    document.dispatchEvent(new CustomEvent('vo:hw', { detail: msg }));
  }
  window.VO_LAB = {
    get gpu() { return state.gpu; }, get device() { return state.device; },
    select: function (kind, key) {
      if (kind === 'device' ? !HW.devices[key] : !HW.gpus[key]) return;
      state[kind] = key; shells.forEach(function (sh) { if (sh.car.current().kind === kind) sh.car.select(key, true); }); notify(kind);
    },
    subscribe: function (f) { subs.push(f); return function () { var i = subs.indexOf(f); if (i >= 0) subs.splice(i, 1); }; }
  };

  var css = document.createElement('style'); css.id = 'lab-shell-css';
  css.textContent = [
    /* the dark scope: the demos read these variables, so they recolour without their own dark styles */
    '.vlab{--bg:#0b1020;--white:#111a30;--ink:#e8ecf5;--ink2:#c3cbdc;--ink3:#8e9ab3;--ink4:#5f6b85;--line:#263252;--line2:#1b2440;--brand50:#1a2452;--brand100:#243576;--brand200:#3b50a8;--brandD:#8fa3ff;color-scheme:dark;',
    '  background:#0b1020;color:var(--ink2);border:1px solid #1f2a44;border-radius:16px;overflow:hidden;font-family:var(--font);scroll-margin-top:74px}',
    '.vlab-head{padding:6px 10px 4px;border-bottom:1px solid #1f2a44;background:#0d1428}',
    '.vlab-head .hwcar-it{background:#0f1730;border-color:#263252;flex-basis:136px;padding:4px 4px 5px;gap:2px}',
    '.vlab-head .hwcar-it .hwc-ph{aspect-ratio:12/5}',
    '.vlab-head .hwcar-n{font-size:12px;margin-top:3px}.vlab-head .hwcar-s{font-size:10.5px}',
    '.vlab-head .hwcar-track{padding:4px 4px}',
    '.vlab-head .hwcar-ar{background:#0f1730;border-color:#263252;color:#c3cbdc}',
    '.vlab-wrap{scroll-margin-top:74px}',
    '.vlab-tabwrap{position:relative;margin:0 0 10px}',
    '.vlab-tabs{display:flex;flex-wrap:nowrap;gap:8px;margin:0;overflow-x:auto;scrollbar-width:none;padding:2px 2px;cursor:grab;user-select:none;-webkit-user-select:none;min-height:38px;align-items:center;touch-action:pan-x pan-y;-webkit-overflow-scrolling:touch;overscroll-behavior-x:contain}',
    '.vlab-tabs.dragging{cursor:grabbing}',
    /* edge hints: a fade and a chevron on whichever side still has tabs out of view */
    '.vlab-tabwrap::before,.vlab-tabwrap::after{content:"";position:absolute;top:0;bottom:0;width:84px;pointer-events:none;opacity:0;transition:opacity .25s;z-index:2}',
    '.vlab-tabwrap::before{left:0;background:linear-gradient(90deg,var(--bg) 34%,transparent)}',
    '.vlab-tabwrap::after{right:0;background:linear-gradient(270deg,var(--bg) 58%,transparent)}',
    '.vlab-tabwrap.more-l::before,.vlab-tabwrap.more-r::after{opacity:1}',
    '.vlab-tabhint{position:absolute;top:50%;transform:translateY(-50%);z-index:3;width:22px;height:22px;border-radius:50%;background:var(--brand);color:#fff;display:grid;place-items:center;pointer-events:none;opacity:0;transition:opacity .25s;box-shadow:0 2px 8px rgba(15,23,42,.25)}',
    '.vlab-tabhint svg{width:13px;height:13px}',
    '.vlab-tabhint.l{left:4px}.vlab-tabhint.r{right:46px}',
    '.vlab-tabwrap.more-l .vlab-tabhint.l,.vlab-tabwrap.more-r .vlab-tabhint.r{opacity:1;animation:vlab-nudge 1.6s ease-in-out 2}',
    '@keyframes vlab-nudge{0%,100%{transform:translateY(-50%)}50%{transform:translateY(-50%) translateX(var(--nx,3px))}}',
    '.vlab-tabhint.l{--nx:-3px}',
    '.vlab-full .vlab-tabwrap{margin:0;background:#0d1428;border-bottom:1px solid #1f2a44}',
    '.vlab-full .vlab-tabwrap::before{background:linear-gradient(90deg,#0d1428 34%,transparent)}.vlab-full .vlab-tabwrap::after{background:linear-gradient(270deg,#0d1428 58%,transparent)}',
    '@media (prefers-reduced-motion:reduce){.vlab-tabhint{animation:none!important}}',
    '.vlab-tabs::-webkit-scrollbar{display:none}',
    '.vlab-tab{scroll-snap-align:center;flex:none}',
    '.vlab-tab{appearance:none;display:inline-flex;align-items:center;gap:8px;padding:9px 14px;border:1px solid var(--line);border-radius:999px;background:var(--white);font:600 14px/1 var(--font);color:var(--ink2);cursor:pointer;white-space:nowrap;transition:border-color .15s,color .15s,background .15s}',
    '.vlab-tab svg{width:15px;height:15px;color:var(--brand)}',
    '.vlab-tab:hover{border-color:var(--brand200);color:var(--brand)}',
    '.vlab-tab[aria-selected=true]{background:var(--brand);border-color:var(--brand);color:#fff}.vlab-tab[aria-selected=true] svg{color:#fff}',
    '.vlab-tab:focus-visible{outline:2px solid var(--brand);outline-offset:2px}',
    '@media(max-width:600px){.vlab-tabs{gap:6px}.vlab-tab{padding:8px 11px;font-size:13px}}',
    /* full screen: the wrapper covers the window, the tabs turn dark and join the panel */
    '.vlab-wrap.vlab-full{position:fixed;inset:0;z-index:120;margin:0;display:flex;flex-direction:column;background:#0b1020}',

    '.vlab-full .vlab-tab{background:#0f1730;border-color:#263252;color:#c3cbdc}.vlab-full .vlab-tab:hover{color:#fff;border-color:#3b50a8}',
    '.vlab-full .vlab-tab[aria-selected=true]{background:var(--brand);border-color:var(--brand);color:#fff}',
    '.vlab-full .vlab{flex:1;min-height:0;border:0;border-radius:0;display:flex;flex-direction:column}',
    '.vlab-full .vlab-body{flex:1;min-height:0;height:auto}',
    'body.vlab-lock{overflow:hidden}',
    '.vlab-wrap{position:relative}',
    '.vlab-fs{position:absolute;top:0;right:0;z-index:5;width:38px;height:38px;border-radius:999px;border:1px solid var(--line);background:var(--white);color:var(--ink2);cursor:pointer;display:grid;place-items:center;box-shadow:0 1px 2px rgba(15,23,42,.05)}',
    '.vlab-fs:hover{border-color:var(--brand200);color:var(--brand)}.vlab-fs svg{width:17px;height:17px}',
    '.vlab-fs:focus-visible{outline:2px solid var(--brand);outline-offset:2px}',
    '.vlab-full .vlab-fs{position:fixed;top:10px;right:10px;background:#0f1730;border-color:#263252;color:#c3cbdc}.vlab-full .vlab-fs:hover{border-color:#3b50a8;color:#fff}',
    /* the tab row fades out under the floating button so a scrolled row never collides with it */
    '.vlab-tabs{padding-right:48px}',
    '.vlab-full .vlab-tabs{padding:8px 56px 8px 10px}',

    '.vlab-body{position:relative;height:var(--lab-h,640px);overflow:auto;overscroll-behavior:contain;scrollbar-width:thin;scrollbar-color:#2a365a #0b1020}',
    '.vlab-panel{padding:10px}',
    '.vlab-panel[hidden]{display:none}',
    '.vlab-note{padding:4px 12px 6px;font-size:11px;color:#5f6b85;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;border-top:1px solid #1f2a44}',
    '.vlab-note b{font-weight:500;color:#8e9ab3}',
    '.vlab .hwc{background:#111a30}',
    '.vlab-tip{position:fixed;z-index:200;max-width:300px;padding:8px 10px;border-radius:8px;background:#1b2440;color:#e8ecf5;border:1px solid #34436b;box-shadow:0 10px 30px -10px rgba(0,0,0,.6);font:400 12px/1.45 var(--font);pointer-events:none;display:none}',
    '.vlab [data-tip]{cursor:help}.vlab button[data-tip],.vlab input[data-tip],.vlab select[data-tip],.vlab label[data-tip]{cursor:pointer}',
    /* a dotted underline marks inline text that carries a tip; tiles, rows and controls show the tip without it */
    '.vlab :is(span,small,dt,dd,h3,h4,b,em,i,p)[data-tip]{text-decoration:underline dotted var(--ink4);text-underline-offset:3px}',
    '.vlab :is(div,li,section,article)[data-tip]{cursor:help}',
    '.vlab input,.vlab select,.vlab button{color-scheme:dark}',
    '@media(max-width:600px){.vlab-head{padding:8px 8px 4px}.vlab-panel{padding:8px}.vlab-note span+span{display:none}}'
  ].join('\n');
  document.head.appendChild(css);

  /* tooltips: elements with data-tip="…" show it on hover and on keyboard focus; touch gets it on tap */
  var tip = null, tipFor = null, tipTimer = null;
  function showTip(el) {
    var text = el.getAttribute('data-tip'); if (!text) return;
    if (!tip) { tip = document.createElement('div'); tip.className = 'vlab-tip'; tip.setAttribute('role', 'tooltip'); }
    /* in the browser's full-screen mode only the full-screen element is drawn, so the tip has to live inside it */
    var host = document.fullscreenElement || document.webkitFullscreenElement || document.body;
    if (tip.parentNode !== host) host.appendChild(tip);
    tip.textContent = text; tipFor = el; tip.style.display = 'block';
    var r = el.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
    var x = Math.min(Math.max(8, r.left + r.width / 2 - w / 2), innerWidth - w - 8), above = r.top - h - 8 >= 8;
    tip.style.left = x + 'px'; tip.style.top = (above ? r.top - h - 8 : r.bottom + 8) + 'px';
    tip.classList.toggle('below', !above);
  }
  function hideTip() { if (tip) tip.style.display = 'none'; tipFor = null; }
  document.addEventListener('mouseover', function (e) { var el = e.target.closest && e.target.closest('[data-tip]'); clearTimeout(tipTimer); if (!el) { if (tipFor && !(e.target.closest && e.target.closest('.vlab-tip'))) hideTip(); return; } if (el !== tipFor) tipTimer = setTimeout(function () { showTip(el); }, 120); });
  document.addEventListener('mouseout', function (e) { var el = e.target.closest && e.target.closest('[data-tip]'); if (el && (!e.relatedTarget || !el.contains(e.relatedTarget))) { clearTimeout(tipTimer); hideTip(); } });
  document.addEventListener('focusin', function (e) { var el = e.target.closest && e.target.closest('[data-tip]'); if (el) showTip(el); });
  document.addEventListener('focusout', function (e) { if (e.target.closest && e.target.closest('[data-tip]')) hideTip(); });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') hideTip(); });
  document.addEventListener('touchstart', function (e) { var el = e.target.closest && e.target.closest('[data-tip]'); if (el && el !== tipFor) showTip(el); else if (!el) hideTip(); }, { passive: true });
  addEventListener('scroll', hideTip, { passive: true }); addEventListener('resize', hideTip);
  document.addEventListener('scroll', function (e) { if (e.target !== document) hideTip(); }, true);

  var shells = [];
  function build(root) {
    var tabs = (root.getAttribute('data-tabs') || Object.keys(TABS).join(',')).split(',').map(function (s) { return s.trim(); }).filter(function (t) { return TABS[t]; });
    if (!tabs.length) return;
    if (root.getAttribute('data-gpu') && HW.gpus[root.getAttribute('data-gpu')]) state.gpu = root.getAttribute('data-gpu');
    if (root.getAttribute('data-device') && HW.devices[root.getAttribute('data-device')]) state.device = root.getAttribute('data-device');
    root.classList.add('vlab-wrap'); root.innerHTML = '';
    var box = document.createElement('div'); box.className = 'vlab';
    var head = document.createElement('div'); head.className = 'vlab-head';
    var first = tabs[0], kind0 = TABS[first].kind;
    var car = HW.carousel(kind0, kind0 === 'device' ? state.device : state.gpu, function (kind, key) { state[kind] = key; shells.forEach(function (sh) { if (sh.car !== car && sh.car.current().kind === kind) sh.car.select(key, true); }); notify(kind); });
    head.appendChild(car);
    var fs = document.createElement('button'); fs.type = 'button'; fs.className = 'vlab-fs'; fs.setAttribute('aria-label', 'Full screen'); fs.setAttribute('data-tip', 'Full screen. Esc leaves it.');
    var FS_IN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3M21 8V5a2 2 0 0 0-2-2h-3M3 16v3a2 2 0 0 0 2 2h3M16 21h3a2 2 0 0 0 2-2v-3"/></svg>';
    var FS_OUT = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3v3a2 2 0 0 1-2 2H3M21 8h-3a2 2 0 0 1-2-2V3M3 16h3a2 2 0 0 1 2 2v3M16 21v-3a2 2 0 0 1 2-2h3"/></svg>';
    fs.innerHTML = FS_IN; root.appendChild(fs);
    var bar = document.createElement('div'); bar.className = 'vlab-tabs'; bar.setAttribute('role', 'tablist');
    var body = document.createElement('div'); body.className = 'vlab-body';
    var panels = {};
    tabs.forEach(function (t, i) {
      var b = document.createElement('button'); b.className = 'vlab-tab'; b.type = 'button'; b.setAttribute('role', 'tab'); b.id = 'vlab-t-' + t + '-' + shells.length;
      b.setAttribute('aria-selected', i === 0 ? 'true' : 'false'); b.tabIndex = i === 0 ? 0 : -1; b.innerHTML = icon(TABS[t].icon) + '<span>' + TABS[t].label + '</span>'; b.setAttribute('data-tab', t);
      bar.appendChild(b);
      var p = document.createElement('div'); p.className = 'vlab-panel'; p.setAttribute('role', 'tabpanel'); p.setAttribute('aria-labelledby', b.id); p.id = 'vlab-p-' + t + '-' + shells.length;
      b.setAttribute('aria-controls', p.id);
      if (i) p.hidden = true;
      var m = document.createElement('div'); m.setAttribute('data-lab', t); p.appendChild(m); body.appendChild(p); panels[t] = { btn: b, panel: p };
    });
    var tw = document.createElement('div'); tw.className = 'vlab-tabwrap';
    tw.innerHTML = '<span class="vlab-tabhint l" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="m15 18-6-6 6-6"/></svg></span><span class="vlab-tabhint r" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="m9 18 6-6-6-6"/></svg></span>';
    tw.insertBefore(bar, tw.firstChild); root.appendChild(tw); HW.dragScroll(bar);
    function edges() { var l = bar.scrollLeft > 4, r = bar.scrollLeft + bar.clientWidth < bar.scrollWidth - 52; tw.classList.toggle('more-l', l); tw.classList.toggle('more-r', r); }
    bar.addEventListener('scroll', edges, { passive: true }); addEventListener('resize', edges);
    setTimeout(edges, 0);
    box.appendChild(head); box.appendChild(body);
    var note = document.createElement('div'); note.className = 'vlab-note';
    note.innerHTML = '<span>Estimates from public specifications; the formulas are under "How this is calculated" in each demo. Not measurements of a VitalOps product.</span>';
    box.appendChild(note); root.appendChild(box);
    var active = first;
    function show(t, focus) {
      if (!panels[t]) return;
      active = t;
      tabs.forEach(function (k) { var on = k === t; panels[k].btn.setAttribute('aria-selected', on); panels[k].btn.tabIndex = on ? 0 : -1; panels[k].panel.hidden = !on; });
      if (bar.scrollWidth > bar.clientWidth + 2) { var bt = panels[t].btn, left = bt.offsetLeft + bt.offsetWidth / 2 - bar.clientWidth / 2; bar.scrollTo({ left: Math.max(0, left), behavior: (window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches) ? 'auto' : 'smooth' }); }
      setTimeout(edges, 400);
      var kind = TABS[t].kind; car.setKind(kind, kind === 'device' ? state.device : state.gpu);
      body.scrollTop = 0;
      if (focus) panels[t].btn.focus();
      window.dispatchEvent(new Event('resize'));
    }
    bar.addEventListener('click', function (e) { var b = e.target.closest('.vlab-tab'); if (b) show(b.getAttribute('data-tab')); });
    bar.addEventListener('keydown', function (e) { var d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0; if (!d) return; e.preventDefault(); var i = tabs.indexOf(active); show(tabs[(i + d + tabs.length) % tabs.length], true); });
    /* deep link: /lab/#kernel opens that tab */
    function fromHash() { var h = location.hash.replace('#', ''); if (panels[h]) { show(h); setTimeout(function () { root.scrollIntoView({ block: 'start', behavior: 'auto' }); }, 0); } }
    bar.addEventListener('click', function (e) { if (e.target.closest('.vlab-tab') && history.replaceState) history.replaceState(null, '', '#' + active); });
    addEventListener('hashchange', fromHash); fromHash();
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { fit(); fromHash(); edges(); });   /* web fonts arrive after the first layout and move the page */
    /* height: the panel fits the window under the site header; the demo scrolls inside if it is taller */
    /* the whole shell fits the window under the site header, with a little air; the demo scrolls inside if it is taller */
    function fit() {
      if (root.classList.contains('vlab-full')) { box.style.removeProperty('--lab-h'); return; }
      var h = Math.max(360, Math.min(960, innerHeight - 62 - 12 - head.offsetHeight - tw.offsetHeight - 10 - note.offsetHeight));
      box.style.setProperty('--lab-h', h + 'px');
    }
    function setFull(on) {
      root.classList.toggle('vlab-full', on); document.body.classList.toggle('vlab-lock', on);
      fs.innerHTML = on ? FS_OUT : FS_IN; fs.setAttribute('aria-label', on ? 'Leave full screen' : 'Full screen'); fs.setAttribute('data-tip', on ? 'Leave full screen (Esc).' : 'Full screen. Esc leaves it.');
      if (on && root.requestFullscreen) root.requestFullscreen().catch(function () {});
      else if (!on && document.fullscreenElement === root && document.exitFullscreen) document.exitFullscreen().catch(function () {});
      fit(); window.dispatchEvent(new Event('resize'));
    }
    fs.addEventListener('click', function () { setFull(!root.classList.contains('vlab-full')); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && root.classList.contains('vlab-full')) setFull(false); });
    document.addEventListener('fullscreenchange', function () { if (!document.fullscreenElement && root.classList.contains('vlab-full')) setFull(false); });
    fit(); addEventListener('resize', fit);
    shells.push({ root: box, car: car, show: show });
  }
  function init() { [].forEach.call(document.querySelectorAll('[data-lab-shell]'), build); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
