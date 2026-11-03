// vitalops docs: Markdown pages under content/, ordered by nav.json, rendered here. Routes are #<page> or #<page>/<heading>.
(function () {
  'use strict';
  var side = document.getElementById('side'), doc = document.getElementById('doc'), pager = document.getElementById('pager'), menu = document.getElementById('menu');
  var pages = [], bySlug = {}, cache = {};

  function esc(t) { return t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
  function slugify(t) { return t.toLowerCase().replace(/<[^>]+>/g, '').replace(/[`*_]/g, '').replace(/[^\w\s-]/g, '').trim().replace(/\s+/g, '-'); }

  // ---- inline Markdown: code, bold, italics, links
  function inline(t) {
    var out = '', i = 0;
    while (i < t.length) {
      var j = t.indexOf('`', i);
      if (j < 0) { out += span(t.slice(i)); break; }
      var k = t.indexOf('`', j + 1);
      if (k < 0) { out += span(t.slice(i)); break; }
      out += span(t.slice(i, j)) + '<code>' + esc(t.slice(j + 1, k)) + '</code>';
      i = k + 1;
    }
    return out;
  }
  function span(t) {
    t = esc(t);
    t = t.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, function (m, a, b) { return '<a href="' + b + '"' + (/^https?:/.test(b) ? ' target="_blank" rel="noopener"' : '') + '>' + a + '</a>'; });
    t = t.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    t = t.replace(/(^|[\s(])\*([^*\s][^*]*)\*/g, '$1<em>$2</em>');
    return t;
  }

  // ---- block Markdown: headings, paragraphs, fenced code, lists, tables, quotes
  function render(md) {
    var lines = md.replace(/\r/g, '').split('\n'), out = [], i = 0, toc = [];
    function para(buf) { if (buf.length) out.push('<p>' + inline(buf.join(' ')) + '</p>'); }
    var buf = [];
    while (i < lines.length) {
      var l = lines[i];
      if (/^```/.test(l)) {
        para(buf); buf = [];
        var code = []; i++;
        while (i < lines.length && !/^```/.test(lines[i])) { code.push(lines[i]); i++; }
        i++;
        out.push('<pre><code>' + esc(code.join('\n')) + '</code></pre>');
        continue;
      }
      var h = /^(#{1,3})\s+(.*)$/.exec(l);
      if (h) {
        para(buf); buf = [];
        var lvl = h[1].length, text = h[2].trim(), id = slugify(text);
        if (lvl === 1) out.push('<h1>' + inline(text) + '</h1>');
        else { out.push('<h' + lvl + ' id="' + id + '">' + inline(text) + '<a class="anchor" href="#' + current + '/' + id + '" aria-label="Link to this section">#</a></h' + lvl + '>'); toc.push([lvl, text, id]); }
        i++; continue;
      }
      if (/^\s*[-*]\s+/.test(l) || /^\s*\d+\.\s+/.test(l)) {
        para(buf); buf = [];
        var ordered = /^\s*\d+\./.test(l), items = [];
        while (i < lines.length && (/^\s*[-*]\s+/.test(lines[i]) || /^\s*\d+\.\s+/.test(lines[i]) || /^\s{2,}\S/.test(lines[i]))) {
          if (/^\s*[-*]\s+/.test(lines[i]) || /^\s*\d+\.\s+/.test(lines[i])) items.push(lines[i].replace(/^\s*([-*]|\d+\.)\s+/, ''));
          else items[items.length - 1] += ' ' + lines[i].trim();
          i++;
        }
        out.push((ordered ? '<ol>' : '<ul>') + items.map(function (x) { return '<li>' + inline(x) + '</li>'; }).join('') + (ordered ? '</ol>' : '</ul>'));
        continue;
      }
      if (/^\|/.test(l) && i + 1 < lines.length && /^\|?\s*:?-+/.test(lines[i + 1])) {
        para(buf); buf = [];
        var cells = function (r) { return r.replace(/^\|/, '').replace(/\|\s*$/, '').split('|').map(function (c) { return c.trim(); }); };
        var head = cells(l); i += 2; var rows = [];
        while (i < lines.length && /^\|/.test(lines[i])) { rows.push(cells(lines[i])); i++; }
        out.push('<table><thead><tr>' + head.map(function (c) { return '<th>' + inline(c) + '</th>'; }).join('') + '</tr></thead><tbody>' +
          rows.map(function (r) { return '<tr>' + r.map(function (c) { return '<td>' + inline(c) + '</td>'; }).join('') + '</tr>'; }).join('') + '</tbody></table>');
        continue;
      }
      if (/^>\s?/.test(l)) {
        para(buf); buf = [];
        var q = [];
        while (i < lines.length && /^>\s?/.test(lines[i])) { q.push(lines[i].replace(/^>\s?/, '')); i++; }
        out.push('<blockquote>' + render(q.join('\n')).html + '</blockquote>');
        continue;
      }
      if (!l.trim()) { para(buf); buf = []; i++; continue; }
      buf.push(l.trim()); i++;
    }
    para(buf);
    return { html: out.join('\n'), toc: toc };
  }

  // ---- navigation
  var current = null;
  function buildSide() {
    side.innerHTML = '';
    NAV.parts.forEach(function (part) {
      var h = document.createElement('h4'); h.textContent = part.title; side.appendChild(h);
      part.pages.forEach(function (p) { var a = document.createElement('a'); a.href = '#' + p[0]; a.textContent = p[1]; a.setAttribute('data-p', p[0]); side.appendChild(a); });
    });
  }
  function fetchPage(slug) {
    if (!cache[slug]) cache[slug] = fetch('content/' + slug + '.md', { cache: 'no-cache' }).then(function (r) { if (!r.ok) throw new Error(r.status); return r.text(); });
    return cache[slug];
  }
  function show(route) {
    var parts = (route || pages[0][0]).split('/'), slug = parts[0], anchor = parts[1];
    if (!bySlug[slug]) { slug = pages[0][0]; anchor = null; }
    current = slug;
    [].forEach.call(side.querySelectorAll('a'), function (a) { a.classList.toggle('on', a.getAttribute('data-p') === slug); });
    side.classList.remove('open'); menu.setAttribute('aria-expanded', 'false');
    fetchPage(slug).then(function (md) {
      var r = render(md);
      doc.innerHTML = r.html;
      document.title = bySlug[slug][1] + ' · vitalops docs';
      var k = pages.findIndex(function (p) { return p[0] === slug; }), prev = pages[k - 1], next = pages[k + 1];
      pager.innerHTML = (prev ? '<a href="#' + prev[0] + '"><small>Previous</small>' + prev[1] + '</a>' : '<span></span>') + (next ? '<a class="next" href="#' + next[0] + '"><small>Next</small>' + next[1] + '</a>' : '');
      if (anchor) { var el = document.getElementById(anchor); if (el) el.scrollIntoView(); } else window.scrollTo(0, 0);
    }).catch(function () { doc.innerHTML = '<h1>Not found</h1><p>No page called <code>' + esc(slug) + '</code>.</p>'; });
  }
  var NAV;
  fetch('nav.json', { cache: 'no-cache' }).then(function (r) { return r.json(); }).then(function (n) {
    NAV = n;
    n.parts.forEach(function (part) { part.pages.forEach(function (p) { pages.push(p); bySlug[p[0]] = p; }); });
    buildSide();
    show(location.hash.replace(/^#/, ''));
    addEventListener('hashchange', function () { show(location.hash.replace(/^#/, '')); });
  });
  menu.addEventListener('click', function () { var o = side.classList.toggle('open'); menu.setAttribute('aria-expanded', o); });
})();
