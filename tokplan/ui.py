"""`tokplan ui`: the interactive planner from vitalops.ai, served from this package on localhost.

The page is static: the same hardware catalogue and the same physics as the command line, written in
JavaScript. Hardware photographs are loaded from vitalops.ai when there is a connection and left out
otherwise.
"""
from __future__ import annotations

import http.server
import os
import socketserver
import sys
import threading
import webbrowser

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def log_message(self, *a):
        pass


def serve(port: int = 8765, no_browser: bool = False) -> int:
    socketserver.TCPServer.allow_reuse_address = True
    try:
        httpd = socketserver.TCPServer(("127.0.0.1", port), _Quiet)
    except OSError as e:
        print("tokplan: cannot listen on port %d (%s); try --port" % (port, e), file=sys.stderr)
        return 2
    url = "http://127.0.0.1:%d/" % port
    print("tokplan ui at %s  (Ctrl-C to stop)" % url)
    if not no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0
