"""serve/test_default_sampling.py - 0.4.8 server-side default sampling (STRATA_DEFAULT_SAMPLING / "default_sampling").

Per field: a request's own fields win, an explicit temperature 0 stays greedy (no defaulted penalty), "off" gives
greedy for absent fields (no model-card defaults either), and the Anthropic route merges the same way.

    python -m unittest serve.test_default_sampling -v
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (DEFAULT_SAMPLING_ON, ByteTokenizer, MockEngine, Service, StrataEngine,  # noqa: E402
                          merge_default_sampling, resolve_default_sampling, serve)

ROOT = Path(__file__).resolve().parents[1]
ON = {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "min_p": 0.0, "repetition_penalty": 1.0,
      "presence_penalty": 0.0}


class Resolve(unittest.TestCase):
    def test_values(self):
        self.assertEqual(DEFAULT_SAMPLING_ON, ON)

    def test_env_on_off_json(self):
        self.assertEqual(resolve_default_sampling({}, {"STRATA_DEFAULT_SAMPLING": "on"}), (ON, "env"))
        self.assertEqual(resolve_default_sampling({"default_sampling": "on"}, {"STRATA_DEFAULT_SAMPLING": "off"}),
                         ({}, "off"))
        self.assertEqual(resolve_default_sampling({}, {"STRATA_DEFAULT_SAMPLING": '{"temperature": 0.5, "top_k": 10}'}),
                         ({"temperature": 0.5, "top_k": 10}, "env"))
        with self.assertRaises(SystemExit):
            resolve_default_sampling({}, {"STRATA_DEFAULT_SAMPLING": "{bad"})
        with self.assertRaises(SystemExit):
            resolve_default_sampling({}, {"STRATA_DEFAULT_SAMPLING": '{"temprature": 0.5}'})
        with self.assertRaises(SystemExit):
            resolve_default_sampling({}, {"STRATA_DEFAULT_SAMPLING": '{"top_k": 500}'})

    def test_config_key_and_legacy_block(self):
        self.assertEqual(resolve_default_sampling({"default_sampling": "on"}, {}), (ON, "config"))
        self.assertEqual(resolve_default_sampling({"default_sampling": False}, {}), ({}, "off"))
        self.assertEqual(resolve_default_sampling({}, {}), ({}, "none"))
        legacy = {"sampling": {"temperature": 0.6, "strata_tune": {"pcie_frac": 0.5}}}
        self.assertEqual(resolve_default_sampling(legacy, {}),
                         ({"temperature": 0.6, "strata_tune": {"pcie_frac": 0.5}}, "sampling"))
        # default_sampling replaces the block's sampling fields but keeps its other keys
        self.assertEqual(resolve_default_sampling({**legacy, "default_sampling": "on"}, {}),
                         ({"strata_tune": {"pcie_frac": 0.5}, **ON}, "config"))
        self.assertEqual(resolve_default_sampling({**legacy, "default_sampling": "off"}, {}),
                         ({"strata_tune": {"pcie_frac": 0.5}}, "off"))

    def test_merge_field_by_field(self):
        self.assertEqual(merge_default_sampling(ON, {}), ON)
        self.assertEqual(merge_default_sampling(ON, {"top_k": 40, "top_p": None}), {**ON, "top_k": 40})
        greedy = merge_default_sampling(ON, {"temperature": 0})
        self.assertEqual(greedy["temperature"], 0)
        self.assertNotIn("presence_penalty", greedy)        # no defaulted penalty under greedy
        self.assertEqual(merge_default_sampling(ON, {"temperature": 0, "presence_penalty": 0.5})["presence_penalty"],
                         0.5)
        self.assertEqual(merge_default_sampling({}, {"temperature": 0.2}), {"temperature": 0.2})


class Recording(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.sampling = dict(sampling or {})
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class Routes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.engine = Recording(cls.tok, "<think>\nhm\n</think>\n\nok", max_context=8192)
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self.svc.model_arch = "qwen4exp"
        self.svc.sampling_defaults = dict(ON)
        self.svc.default_sampling_source = "config"
        self.svc.shared = {}

    def tearDown(self):
        self.svc.model_arch = None
        self.svc.sampling_defaults = {}
        self.svc.default_sampling_source = "none"

    def post(self, path, body):
        self.engine.sampling = {}
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()
        return StrataEngine.sampling_keys(self.engine.sampling)

    def chat(self, **body):
        return self.post("/v1/chat/completions",
                         {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, **body})

    def anthropic(self, **body):
        return self.post("/v1/messages",
                         {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, **body})

    def test_absent_fields_get_defaults(self):
        for keys in (self.chat(), self.anthropic()):
            self.assertIn("temperature=0.6", keys)
            self.assertIn("top_k=20", keys)
            self.assertIn("top_p=0.95", keys)
            self.assertNotIn("min_p", keys)                 # 0 = off
            self.assertNotIn("penalty_present", keys)       # 0 = off
            self.assertNotIn("penalty_repeat", keys)        # 1.0 = off

    def test_filled_fields_are_logged(self):
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.chat(top_k=5)
        line = [x for x in out.getvalue().splitlines() if "default sampling for this request" in x]
        self.assertEqual(len(line), 1)
        self.assertIn("temperature=0.6", line[0])
        self.assertNotIn("top_k", line[0])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.chat(temperature=0)
        self.assertNotIn("default sampling for this request", out.getvalue())   # greedy: nothing logged

    def test_no_sampling_request_gets_every_default(self):
        # A request with no sampling fields at all fills (and logs, on stdout = the service journal, not the
        # engine log file) every configured default field
        import contextlib
        import io
        for send in (self.chat, self.anthropic):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                send()
            line = [x for x in out.getvalue().splitlines() if "default sampling for this request" in x]
            self.assertEqual(len(line), 1, out.getvalue())
            for field in ("temperature=0.6", "top_k=20", "top_p=0.95", "min_p=0.0",
                          "repetition_penalty=1.0", "presence_penalty=0.0"):
                self.assertIn(field, line[0])

    def test_client_fields_win(self):
        for keys in (self.chat(temperature=0.9, top_p=0.5, top_k=5), self.anthropic(temperature=0.9, top_p=0.5, top_k=5)):
            self.assertIn("temperature=0.9", keys)
            self.assertIn("top_p=0.5", keys)
            self.assertIn("top_k=5", keys)
            self.assertNotIn("min_p", keys)                 # the field it left out defaults to 0 (off)

    def test_explicit_temperature_zero_is_greedy(self):
        for keys in (self.chat(temperature=0), self.anthropic(temperature=0)):
            self.assertNotIn("temperature", keys)
            self.assertNotIn("penalty", keys)

    def test_off_is_greedy_without_model_card(self):
        self.svc.sampling_defaults = {}
        self.svc.default_sampling_source = "off"
        self.assertEqual(self.chat(), "")
        self.assertEqual(self.anthropic(), "")

    def test_forced_tool_choice_still_works(self):
        tools = [{"type": "function", "function": {"name": "get", "parameters": {"type": "object", "properties": {}}}}]
        keys = self.chat(tools=tools, tool_choice={"type": "function", "function": {"name": "get"}})
        self.assertIn("temperature=0.6", keys)

    def test_version_and_health_expose_defaults(self):
        for path in ("/speedy/version", "/health"):
            body = json.loads(urllib.request.urlopen(self.base + path, timeout=10).read())
            info = body if path == "/speedy/version" else body["speedy"]
            self.assertEqual(info["default_sampling"], {"source": "config", "values": ON})
        self.svc.default_sampling_source = "off"
        body = json.loads(urllib.request.urlopen(self.base + "/speedy/version", timeout=10).read())
        self.assertEqual(body["default_sampling"], {"source": "off", "values": {}})


if __name__ == "__main__":
    unittest.main()
