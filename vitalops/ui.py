"""`vitalops ui`: the interactive planner from vitalops.ai, served from this package on localhost.

The page calls back into this package for every figure (vitalops.web over a small JSON API), so it
shows exactly what the command line prints, and models from the Hub load with your HF_TOKEN.
Hardware photographs are loaded from vitalops.ai when there is a connection and left out otherwise.
"""
from __future__ import annotations

import http.server
import json
import os
import socketserver
import sys
import threading
import webbrowser
from urllib.parse import parse_qs, urlparse

from .web import evaluate, model_info, model_list
from . import __version__

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/models":
            return self._json({"version": __version__, "models": model_list()})
        if u.path == "/api/model":
            repo = (parse_qs(u.query).get("id") or [""])[0]
            try:
                return self._json(model_info(repo))
            except Exception as e:
                return self._json({"error": str(e.args[0] if e.args else e)})
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/evaluate":
            return self._json({"error": "not found"}, 404)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            return self._json(evaluate(json.loads(self.rfile.read(n) or b"{}")))
        except Exception as e:
            return self._json({"error": "%s: %s" % (type(e).__name__, e)})


def serve(port: int = 8765, no_browser: bool = False) -> int:
    socketserver.TCPServer.allow_reuse_address = True
    try:
        httpd = socketserver.TCPServer(("127.0.0.1", port), _Quiet)
    except OSError as e:
        print("vitalops: cannot listen on port %d (%s); try --port" % (port, e), file=sys.stderr)
        return 2
    url = "http://127.0.0.1:%d/" % port
    print("vitalops ui at %s  (Ctrl-C to stop)" % url)
    if not no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0
