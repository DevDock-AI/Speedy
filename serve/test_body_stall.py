"""Strata #6107 port: a request body that stalls ends that request (408) instead of tying up the handler thread;
the next request is answered.  python -m unittest serve.test_body_stall -v"""
import json
import os
import socket
import time
import unittest
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, body_stall_s, serve

ROOT = Path(__file__).resolve().parent.parent
CRLF = b"\r\n"


class BodyStall(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = MockEngine(tok, "hello there", max_context=4096)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.env = mock.patch.dict(os.environ, {"STRATA_BODY_STALL_S": "0.5"})
        cls.env.start()
        cls.httpd = serve(cls.svc, port=0)
        cls.port = cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.env.stop()

    def read_all(self, s):
        data = b""
        while True:
            piece = s.recv(65536)
            if not piece:
                return data
            data += piece

    def test_stream_write_guard_is_on_by_default(self):
        import serve.server as server
        if "STRATA_STREAM_WRITE_TIMEOUT_S" not in os.environ:
            self.assertEqual(server.STREAM_WRITE_TIMEOUT_S, 120.0)

    def test_setting(self):
        self.assertEqual(body_stall_s({}), 30.0)
        self.assertEqual(body_stall_s({"STRATA_BODY_STALL_S": "0"}), 0.0)
        self.assertEqual(body_stall_s({"STRATA_BODY_STALL_S": "x"}), 30.0)

    def test_a_stalled_body_is_a_408_and_the_server_goes_on(self):
        body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode()
        with socket.create_connection(("127.0.0.1", self.port), timeout=20) as s:
            s.sendall(b"POST /v1/chat/completions HTTP/1.1" + CRLF + b"Host: 127.0.0.1" + CRLF +
                      b"Content-Type: application/json" + CRLF + b"Content-Length: %d" % (len(body) + 100) + CRLF + CRLF + body)
            t0 = time.monotonic()
            data = self.read_all(s)                         # the socket stays open on our side: only the guard ends it
            self.assertLess(time.monotonic() - t0, 10)
        self.assertEqual(int(data.split(b" ", 2)[1]), 408, data[:200])
        self.assertIn(b"stalled", data)
        with socket.create_connection(("127.0.0.1", self.port), timeout=20) as s:
            s.sendall(b"POST /v1/chat/completions HTTP/1.1" + CRLF + b"Host: 127.0.0.1" + CRLF + b"Connection: close" + CRLF +
                      b"Content-Type: application/json" + CRLF + b"Content-Length: %d" % len(body) + CRLF + CRLF + body)
            data = self.read_all(s)
        self.assertEqual(int(data.split(b" ", 2)[1]), 200, data[:200])

    def test_a_stalled_chunked_body_is_a_408(self):
        with socket.create_connection(("127.0.0.1", self.port), timeout=20) as s:
            s.sendall(b"POST /v1/chat/completions HTTP/1.1" + CRLF + b"Host: 127.0.0.1" + CRLF +
                      b"Content-Type: application/json" + CRLF + b"Transfer-Encoding: chunked" + CRLF + CRLF + b"5" + CRLF + b"{\"a\":")
            data = self.read_all(s)
        self.assertEqual(int(data.split(b" ", 2)[1]), 408, data[:200])


if __name__ == "__main__":
    unittest.main()
