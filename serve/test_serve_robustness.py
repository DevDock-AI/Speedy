"""Smaller serve-side robustness fixes ported from Strata (round 0213): #4271 a streaming client that stops reading,
#5044 a refused admission / a byte that is not UTF-8 on the engine's stdout, #4792 a failed engine start when another
thread closed the process, #3443 ConvCacheLog.poll without a lock.  The fake engine and helpers are those of
test_stream_ownership.py.
"""
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve.server import StrataEngine
from serve.test_stream_ownership import FakeEngines, canary_script, collect, expected_reply, prompt_ids


class StalledClient(unittest.TestCase):
    """#4271: a streaming client that stops reading, without closing its connection, must not hold the engine's turn
    for ever: the write that makes no progress ends the request as a disconnect and the next request is served."""

    def test_a_client_that_stops_reading_gives_the_turn_back(self):
        import json
        import socket
        import time
        import urllib.request
        import serve.server as server
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, Service, serve
        tok = ByteTokenizer()
        engine = MockEngine(tok, " ".join(str(i) for i in range(80000)), max_context=2_000_000)   # (not one repeated token)
        svc = Service(engine, tok, ChatTemplate(Path(server.__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        port = httpd.server_address[1]
        stuck = socket.socket()
        try:
            with mock.patch.object(server, "STREAM_WRITE_TIMEOUT_S", 0.5, create=True):
                stuck.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
                stuck.connect(("127.0.0.1", port))
                body = json.dumps({"messages": [{"role": "user", "content": "go"}], "stream": True,
                                   "max_tokens": 400000, "reasoning_effort": "none"}).encode()
                stuck.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                              b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
                # ... and it never reads.  Wait until the server is writing into a full buffer, then until it lets go.
                end = time.monotonic() + 3
                while time.monotonic() < end and not svc.fifo.locked():
                    time.sleep(0.01)
                self.assertTrue(svc.fifo.locked(), "the stuck client's request holds the engine's turn")
                end = time.monotonic() + 20
                while time.monotonic() < end and svc.fifo.locked():
                    time.sleep(0.05)
                self.assertFalse(svc.fifo.locked(), "the turn is still held by a client that does not read")
        finally:
            stuck.close()
            httpd.shutdown()
            httpd.server_close()

    def test_a_client_that_reads_but_waits_for_a_slow_engine_is_not_cut_off(self):
        """The write limit is for a write that makes no progress (the client stopped reading), not for a client that is
        idle while the engine thinks: 0.3 s between tokens, a 0.1 s write limit, the whole reply arrives."""
        import http.client
        import json
        import serve.server as server
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, Service, serve
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "abcdef", delay_s=0.3), tok,
                      ChatTemplate(Path(server.__file__).parent / "chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            with mock.patch.object(server, "STREAM_WRITE_TIMEOUT_S", 0.1, create=True):
                conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=30)
                conn.request("POST", "/v1/chat/completions", json.dumps(
                    {"messages": [{"role": "user", "content": "go"}], "stream": True, "max_tokens": 50,
                     "reasoning_effort": "none"}), {"Content-Type": "application/json"})
                body = conn.getresponse().read().decode()
                conn.close()
            text = "".join(json.loads(l[5:])["choices"][0]["delta"].get("content") or ""
                           for l in body.splitlines() if l.startswith("data:") and "[DONE]" not in l
                           and json.loads(l[5:]).get("choices"))
            self.assertEqual(text, "abcdef")
            self.assertIn("data: [DONE]", body)
        finally:
            httpd.shutdown()
            httpd.server_close()


