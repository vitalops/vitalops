/* The planner's engine under `tokplan ui`: the local server answers with tokplan.web, so every figure
   is the installed package's, and Hub models load through its loader (with HF_TOKEN if set). */
(function () {
  'use strict';
  function json(r) { if (!r.ok) throw new Error('the local tokplan server answered ' + r.status); return r.json(); }
  var E = {
    label: 'tokplan · Python',
    tip: 'Every figure here comes from the tokplan package this page is served by, through a small API on localhost. The command line below prints the same report.',
    models: [],
    version: ''
  };
  E.ready = fetch('api/models').then(json).then(function (d) { E.models = d.models; E.version = d.version; E.label = 'tokplan ' + d.version + ' · Python'; });
  E.evaluate = function (state) { return fetch('api/evaluate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(state) }).then(json); };
  E.loadModel = function (repo) {
    return fetch('api/model?id=' + encodeURIComponent(repo)).then(json).then(function (d) { if (d.error) throw new Error(d.error); return d; });
  };
  window.VO_PLANNER_ENGINE = E;
})();
