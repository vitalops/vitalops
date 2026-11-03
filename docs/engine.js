/* The planner's engine on vitalops's site: vitalops's own Python source (py/vitalops/, a copy of the
   package that tools/site.py keeps in sync) loaded into Pyodide. lab-planner.js calls
   vitalops.web.evaluate for every figure it shows. Pyodide has no sockets, so a Hugging Face config is
   fetched here and handed to vitalops's loader through its cache. */
(function () {
  'use strict';
  var PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
  var FILES = ['__init__.py', 'hw.py', 'models.py', 'physics.py', 'cli.py', 'web.py'];
  var HUB = 'https://huggingface.co/';
  var listeners = [], last = '', py = null, web = null;

  function status(m) { last = m; listeners.forEach(function (f) { f(m); }); }
  function script(src) {
    return new Promise(function (ok, bad) {
      var s = document.createElement('script'); s.src = src; s.onload = ok;
      s.onerror = function () { bad(new Error('could not load ' + src)); };
      document.head.appendChild(s);
    });
  }
  function text(path) {
    return fetch(path, { cache: 'no-cache' }).then(function (r) { if (!r.ok) throw new Error(path + ': ' + r.status); return r.text(); });
  }

  var E = {
    label: 'vitalops · Python',
    tip: 'Every figure here comes from vitalops\'s Python source, loaded into Pyodide (CPython compiled to WebAssembly), the same code pip installs. The command line below prints the same report.',
    models: [],
    version: '',
    onStatus: function (f) { listeners.push(f); if (last) f(last); }
  };

  E.ready = (async function () {
    status('Loading Pyodide, CPython compiled to WebAssembly (about 7 MB, cached after the first visit)…');
    var files = Promise.all(FILES.map(function (f) { return text('py/vitalops/' + f); }));
    await script(PYODIDE + 'pyodide.js');
    py = await window.loadPyodide({ indexURL: PYODIDE });
    status('Loading vitalops…');
    var srcs = await files;
    py.FS.mkdirTree('/home/pyodide/vitalops');
    FILES.forEach(function (f, i) { py.FS.writeFile('/home/pyodide/vitalops/' + f, srcs[i]); });
    py.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
    web = py.pyimport('vitalops.web');
    var d = JSON.parse(web.model_list_json());
    E.models = d.models; E.version = d.version;
    E.label = 'vitalops ' + d.version + ' · Python';
  })();

  E.evaluate = function (state) { return JSON.parse(web.evaluate_json(JSON.stringify(state))); };

  // config.json and the checkpoint size, as vitalops.models.fetch_hub reads them, then vitalops's own loader
  E.loadModel = async function (repo) {
    var base = HUB + repo + '/resolve/main/';
    var r = await fetch(base + 'config.json');
    if (r.status === 401 || r.status === 403) throw new Error(repo + ' is gated or private. The command line reads it with HF_TOKEN set.');
    if (r.status === 404) throw new Error('No config.json at ' + repo + '.');
    if (!r.ok) throw new Error('The Hub answered ' + r.status + ' for ' + repo + '.');
    var config = await r.json(), total = null;
    try {
      var ri = await fetch(base + 'model.safetensors.index.json');
      if (ri.ok) total = +((await ri.json()).metadata || {}).total_size || null;
    } catch (e) { /* no index */ }
    if (!total) {
      try {
        var rt = await fetch(HUB + 'api/models/' + repo + '/tree/main');
        if (rt.ok) (await rt.json()).forEach(function (f) { if (f.path === 'model.safetensors') total = (f.lfs && f.lfs.size) || f.size || null; });
      } catch (e) { /* size unknown: the config shapes are used */ }
    }
    web.preload_json(repo, JSON.stringify({ config: config, total_bytes: total }));
    var mi = JSON.parse(web.model_info_json(repo));
    if (mi.error) throw new Error(mi.error);
    return mi;
  };

  window.VO_PLANNER_ENGINE = E;
})();
