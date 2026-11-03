"""A small OpenAI-compatible server for testing `vitalops bench` without a GPU.

It streams words, not model output, on a made-up timing law that slows as more requests are in
flight: time between tokens = 4 ms + 0.25 ms × active requests; time to first token =
10 ms + 0.02 ms × prompt tokens + 2 ms × active requests. One word counts as one token, plus 3
for the template. Run it directly to point the CLI at it: python tests/mock_server.py 8000
"""
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "mock/llama-tiny"
active = [0]
lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    strict = False          # reject min_tokens/ignore_eos, like a strict hosted API

    def log_message(self, *a):
        pass

    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/v1/models":
            return self._send(200, {"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        chat = self.path == "/v1/chat/completions"
        if self.path not in ("/v1/completions", "/v1/chat/completions"):
            return self._send(404, {"error": "not found"})
        if self.strict and ("min_tokens" in body or "ignore_eos" in body):
            return self._send(400, {"error": {"message": "Unrecognized request argument supplied: min_tokens"}})
        text = body["messages"][-1]["content"] if chat else body["prompt"]
        ptok = len(text.split()) + 3
        want = int(body.get("max_tokens", 16))
        out = want if body.get("min_tokens") else max(1, min(want, want * 3 // 4))
        with lock:
            active[0] += 1
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def emit(obj):
                data = ("data: " + (obj if isinstance(obj, str) else json.dumps(obj)) + "\n\n").encode()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                self.wfile.flush()
            time.sleep(0.010 + 0.00002 * ptok + 0.002 * active[0])
            for i in range(out):
                if i:
                    time.sleep(0.004 + 0.00025 * active[0])
                ch = {"delta": {"content": " w%d" % i}} if chat else {"text": " w%d" % i}
                emit({"choices": [dict(ch, index=0)]})
            emit({"choices": [], "usage": {"prompt_tokens": ptok, "completion_tokens": out, "total_tokens": ptok + out}})
            emit("[DONE]")
            self.wfile.write(b"0\r\n\r\n")
        finally:
            with lock:
                active[0] -= 1


def serve(port=0, strict=False):
    h = type("H", (Handler,), {"strict": strict})
    s = ThreadingHTTPServer(("127.0.0.1", port), h)
    s.daemon_threads = True
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


if __name__ == "__main__":
    s = serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
    print("mock server on http://127.0.0.1:%d" % s.server_address[1])
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
