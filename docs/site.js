/* vitalops.ai/oss: terminal tabs and copy buttons. */
(function () {
  'use strict';
  function init() {
    [].forEach.call(document.querySelectorAll('[data-tabs]'), function (t) {
      var tabs = [].slice.call(t.querySelectorAll('[role="tab"]'));
      function show(tab) {
        tabs.forEach(function (b) {
          var on = b === tab, p = document.getElementById(b.getAttribute('aria-controls'));
          b.setAttribute('aria-selected', on ? 'true' : 'false'); b.tabIndex = on ? 0 : -1;
          if (p) p.hidden = !on;
          var note = t.querySelector('[data-note-for="' + b.getAttribute('aria-controls') + '"]');
          if (note) note.hidden = !on;
        });
      }
      tabs.forEach(function (b, i) {
        b.addEventListener('click', function () { show(b); });
        b.addEventListener('keydown', function (e) {
          var d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
          if (d) { var n = tabs[(i + d + tabs.length) % tabs.length]; show(n); n.focus(); e.preventDefault(); }
        });
      });
    });
    document.addEventListener('click', function (e) {
      var b = e.target.closest('[data-copy]'); if (!b) return;
      var txt = b.getAttribute('data-copy');
      var done = function () { b.classList.add('done'); b.setAttribute('aria-label', 'Copied'); setTimeout(function () { b.classList.remove('done'); b.setAttribute('aria-label', 'Copy'); }, 1400); };
      if (navigator.clipboard) navigator.clipboard.writeText(txt).then(done, function () {}); else done();
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
