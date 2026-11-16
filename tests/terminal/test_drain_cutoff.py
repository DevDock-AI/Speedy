"""Upstream Strata #6055: a request the stopping server cut off ends with an error, not 0 tokens as "length"."""

import threading
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineDied, MockEngine, Service


class StoppedEngine(MockEngine):
    """An engine closed under the request: its STOP ends the pass before any token (no end-of-turn token)."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_prompt = list(ids)
        yield from ()


class DrainCutoffTests(unittest.TestCase):
    def svc(self, script, cls=MockEngine):
        tok = ByteTokenizer()
        return Service(cls(tok, script, max_context=4096), tok,
                       ChatTemplate(Path(__file__).resolve().parents[2] / "serve/chat_template.jinja")), tok

    def test_cut_off_while_draining_is_an_error(self):
        svc, tok = self.svc("", StoppedEngine)
        svc.begin_drain()
        with self.assertRaises(EngineDied):
            list(svc.run(tok.encode("Hi"), False, None, 64, {}, threading.Event()))

    def test_not_draining_keeps_length(self):
        svc, tok = self.svc("", StoppedEngine)
        out = list(svc.run(tok.encode("Hi"), False, None, 64, {}, threading.Event()))
        self.assertEqual(out[-1][1]["finish"], "length")

    def test_finished_reply_while_draining_is_kept(self):
        svc, tok = self.svc("Hello.")
        svc.begin_drain()
        out = list(svc.run(tok.encode("Hi"), False, None, 64, {}, threading.Event()))
        self.assertEqual(out[-1][1]["finish"], "stop")


if __name__ == "__main__":
    unittest.main()
