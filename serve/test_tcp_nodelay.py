"""serve/test_tcp_nodelay.py - accepted sockets have TCP_NODELAY on, so small SSE writes are not held back by Nagle.

    python -m unittest serve.test_tcp_nodelay -v
"""
from __future__ import annotations

import socket
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402
from serve.test_server import CTX  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class TcpNodelay(unittest.TestCase):
    def test_the_accepted_socket_has_nodelay(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "hi", max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        seen = []
        base = httpd.RequestHandlerClass

        class Probe(base):
            def setup(self):
                super().setup()
                seen.append(self.connection.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY))

        httpd.RequestHandlerClass = Probe
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_address[1]}/health", timeout=30) as r:
                r.read()
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertTrue(seen and seen[0])


if __name__ == "__main__":
    unittest.main()
