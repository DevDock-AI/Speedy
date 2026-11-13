"""Speedy 0.4.6 tool-call guards (serve/toolguard.py): constrained tool-call opening (STRATA_TOOLCALL_CONSTRAIN), old
thinking dropped at long context (STRATA_DROP_THINKING_ABOVE) and the repeat warning (STRATA_REPEAT_WARN).

    python -m unittest serve.test_toolguard
"""
import copy
import json
import os
import tempfile
import threading
import unittest
import urllib.request
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from serve import toolguard as g
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parent.parent
THINK = "I will read the file.</think>\n\n"
REST = "<parameter=path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>"


class CallTokenizer(ByteTokenizer):
    SPECIALS = ByteTokenizer.SPECIALS + ["<tool_call>"]
    ALWAYS = ("<tool_call>",)


def tool(name, *params):
    return {"type": "function", "function": {"name": name, "parameters": {
        "type": "object", "properties": {p: {"type": "string"} for p in params}, "required": list(params)}}}


TOOLS = [tool("list", "path"), tool("read", "path"), tool("read_all", "path"), tool("bash", "command")]


class BiasEngine(MockEngine):
    """First pass: the reply with the broken opening `<4>`.  A one-token request with a logit bias picks, among the
    allowed ids, the one that spells `want` further (else the lowest).  A prompt ending in "<function=NAME>\\n" gets
    the call's parameters."""

    def __init__(self, tok, first, want="read", bias_ok=True):
        super().__init__(tok, "x")
        self.first, self.want, self.calls = first, want, []
        self.info = {"logit_bias": 1} if bias_ok else {}

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls.append((list(ids), max_new, dict(sampling or {})))
        text = self.tok.decode(ids)
        bias = (sampling or {}).get("logit_bias")
        if bias:
            assert max_new == 1
            sofar = text[text.rfind("<function=") + len("<function="):]
            target = self.want + ">\n"
            nxt = target[len(sofar):len(sofar) + 1].encode()
            pick = nxt[0] if nxt and nxt[0] in bias else min(bias)
            yield pick
            return
        out = REST if text.endswith(">\n") and "<function=" in text[-40:] else self.first
        for t in (self.tok.encode(out, parse_special=True) + self.tok.encode("<|im_end|>", parse_special=True))[:max_new]:
            if cancel.is_set():
                return
            yield t


class Pure(unittest.TestCase):
    def test_allowed_next_only_continues_declared_names(self):
        index = {"r": [1], "re": [2], "read": [3], "x": [4], "4": [5], ">": [6], ">\n": [7], "=": [8], "b": [9]}
        targets = g.name_targets(["read", "bash"])
        self.assertEqual(g.allowed_next(index, "", targets), [1, 2, 3, 9])
        self.assertEqual(g.allowed_next(index, "read", targets), [6, 7])
        self.assertEqual(g.allowed_next(index, "rx", targets), [])

    def test_constrain_name_picks_and_forces(self):
        index = {"r": [1], "e": [2], "a": [3], "d": [4], "b": [5], "s": [6], "h": [7], ">": [8], "\n": [9], "_": [10],
                 "l": [11], "read": [12]}
        enc = lambda text: [index[c][0] for c in text]  # noqa: E731
        # one pick ("b") leaves one name: the rest is forced
        ids, name = g.constrain_name(lambda allowed: 5, enc, index, ["read", "bash"])
        self.assertEqual((ids, name), ([5] + enc("ash>\n"), "bash"))
        # read / read_all: "read" then ">" must be picked by the model
        seq = iter([12, 8])
        ids, name = g.constrain_name(lambda allowed: next(seq), enc, index, ["read", "read_all"])
        self.assertEqual(name, "read")
        self.assertEqual(ids, [12, 8, 9])
        # a token outside the mask (an engine that ignored the bias): no name
        self.assertIsNone(g.constrain_name(lambda allowed: 99, enc, index, ["read", "bash"]))

    def test_drop_boundary_moves_in_steps(self):
        self.assertEqual([g.drop_boundary(n, 4) for n in range(0, 11)], [0, 0, 0, 0, 0, 4, 4, 4, 4, 8, 8])
        for n in range(1, 200):
            self.assertLess(g.drop_boundary(n, 16), n)        # the last turn always keeps its thinking

    def test_drop_old_thinking_prefix_is_stable(self):
        def convo(k):
            out = [{"role": "user", "content": "go"}]
            for i in range(k):
                out += [{"role": "assistant", "content": "", "reasoning_content": f"think {i}",
                         "tool_calls": [{"function": {"name": "bash", "arguments": {"command": f"ls {i}"}}}]},
                        {"role": "tool", "content": f"out {i}"}]
            return out
        changes, prev = 0, None
        for k in range(2, 40):
            m = convo(k)
            dropped = g.drop_old_thinking(m, 8)
            self.assertEqual(m[-2]["reasoning_content"], f"think {k - 1}")   # the last turn keeps it
            state = [x.get("reasoning_content") for x in m if x["role"] == "assistant"][:-1]
            if prev is not None and state[:len(prev)] != prev:
                changes += 1                                  # an earlier turn changed: the cache prefix moved
            prev = state
            self.assertEqual(len(dropped), g.drop_boundary(k, 8))
        self.assertEqual(changes, 4)                          # k = 10, 18, 26, 34: once per 8 turns

    def test_mark_repeats(self):
        def call(cmd):
            return {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "bash", "arguments": {"command": cmd}}}]}
        m = [{"role": "user", "content": "go"}, call("ls"), {"role": "tool", "content": "a"},
             call("pwd"), {"role": "tool", "content": "b"}, call("ls"), {"role": "tool", "content": "a"}]
        reps = g.mark_repeats(m)
        self.assertEqual([(r["name"], r["times"]) for r in reps], [("bash", 2)])
        self.assertTrue(m[-1]["content"].endswith(g.REPEAT_NOTE + "]"))
        self.assertEqual(m[2]["content"], "a")
        # the next request (one more turn): the same note on the same turn, nothing new to log
        m2 = [x if x["role"] != "tool" else dict(x) for x in m[:-1]] + [{"role": "tool", "content": "a"},
                                                                         call("cat f"), {"role": "tool", "content": "c"}]
        m2[-3]["content"] = "a"
        self.assertEqual(g.mark_repeats(m2), [])
        self.assertTrue(m2[-3]["content"].endswith(g.REPEAT_NOTE + "]"))
        # outside the window of 3 calls: not a repeat
        m3 = [{"role": "user", "content": "go"}]
        for c in ("ls", "a", "b", "c", "ls"):
            m3 += [call(c), {"role": "tool", "content": "r"}]
        self.assertEqual(g.mark_repeats(m3, 3), [])
        # string arguments compare by value, not spelling
        m4 = [{"role": "user", "content": "go"},
              {"role": "assistant", "tool_calls": [{"function": {"name": "x", "arguments": '{"a": 1, "b": 2}'}}]},
              {"role": "tool", "content": "r"},
              {"role": "assistant", "tool_calls": [{"function": {"name": "x", "arguments": {"b": 2, "a": 1}}}]},
              {"role": "tool", "content": [{"type": "text", "text": "r"}]}]
        self.assertEqual(len(g.mark_repeats(m4)), 1)
        self.assertEqual(m4[-1]["content"][-1]["text"].strip(), f"[Note: {g.REPEAT_NOTE}]")

    def test_fail_log_has_no_default_path(self):
        from serve import server
        env = {k: v for k, v in os.environ.items() if k != "STRATA_TOOLCALL_FAIL_LOG"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch("builtins.open") as op, \
                mock.patch("os.makedirs") as mk:
            server.log_toolcall_failure({"outcome": "off"})
        op.assert_not_called()
        mk.assert_not_called()
        self.assertFalse(hasattr(server, "TOOLCALL_FAIL_LOG"))

    def test_flags(self):
        with mock.patch.dict(os.environ, {"STRATA_TOOLCALL_CONSTRAIN": ""}, clear=False):
            self.assertTrue(g.flag(g.env_value("STRATA_TOOLCALL_CONSTRAIN", {"STRATA_TOOLCALL_CONSTRAIN": "1"})))
        self.assertFalse(g.flag(None))
        self.assertEqual(g.int_setting("300000"), 300000)
        self.assertEqual(g.int_setting("x", 7), 7)


class Served(unittest.TestCase):
    def start(self, engine, **env):
        self.logdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.logdir.cleanup)
        self.failog = os.path.join(self.logdir.name, "toolcall-failures.jsonl")
        p = mock.patch.dict(os.environ, {"STRATA_TOOLCALL_FAIL_LOG": self.failog, **env})
        p.start()
        self.addCleanup(p.stop)
        self.tok = engine.tok
        self.engine = engine
        self.svc = Service(engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def post(self, messages, stream=False, tools=TOOLS):
        body = {"model": "m", "messages": messages, "tools": tools, "max_tokens": 400, "stream": stream}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "x-session-id": "ses_g1"})
        log = StringIO()
        with redirect_stdout(log), urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode()
        self.log = log.getvalue()
        if not stream:
            return json.loads(raw)["choices"][0]
        content, names = "", []
        for line in raw.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                for ch in json.loads(line[6:])["choices"]:
                    d = ch.get("delta") or {}
                    content += d.get("content") or ""
                    names += [(c.get("function") or {}).get("name") for c in d.get("tool_calls") or []
                              if (c.get("function") or {}).get("name")]
        return {"content": content, "names": names}

    BROKEN = THINK + "Let me look.\n\n<tool_call>\n<4>\n<parameter=path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>"
    ASK = [{"role": "user", "content": "read /tmp/x"}]

    def test_constrained_opening_turns_the_slip_into_a_declared_call(self):
        self.start(BiasEngine(CallTokenizer(), self.BROKEN), STRATA_TOOLCALL_CONSTRAIN="1")
        c = self.post(self.ASK)
        self.assertEqual(c["finish_reason"], "tool_calls")
        calls = c["message"]["tool_calls"]
        self.assertEqual([(t["function"]["name"], json.loads(t["function"]["arguments"])) for t in calls],
                         [("read", {"path": "/tmp/x"})])
        self.assertNotIn("<4>", c["message"].get("content") or "")
        prompts = [self.tok.decode(p) for p, _, _ in self.engine.calls]
        biased = [s for _, _, s in self.engine.calls if s.get("logit_bias")]
        # "read" vs "read_all": r, e, a, d and then ">" picked by the model; "\n" forced
        self.assertEqual(len(biased), 5)
        self.assertTrue(prompts[1].endswith("Let me look.\n\n<tool_call>\n<function="))
        self.assertTrue(prompts[-1].endswith("<tool_call>\n<function=read>\n"))
        # every later prompt continues the first one (the engine's cached prefix)
        for p, _, _ in self.engine.calls[1:]:
            self.assertEqual(p[:len(self.engine.calls[0][0])], self.engine.calls[0][0])
        self.assertIn("[strata] constrained tool call: read", self.log)

    def test_streaming(self):
        self.start(BiasEngine(CallTokenizer(), self.BROKEN, want="bash"), STRATA_TOOLCALL_CONSTRAIN="1")
        c = self.post(self.ASK, stream=True)
        self.assertEqual(c["names"], ["bash"])
        self.assertNotIn("<4>", c["content"])

    def test_off_and_unsupported_engine_unchanged(self):
        for env, ok in (({}, True), ({"STRATA_TOOLCALL_CONSTRAIN": "1"}, False)):
            with self.subTest(env=env):
                self.start(BiasEngine(CallTokenizer(), self.BROKEN, bias_ok=ok), **env)
                c = self.post(self.ASK)
                self.assertEqual(len(self.engine.calls), 1)
                self.assertFalse(any(s.get("logit_bias") for _, _, s in self.engine.calls))
                self.assertIn("<tool_call>", c["message"].get("content") or "")

    def test_with_repair_on(self):
        self.start(BiasEngine(CallTokenizer(), self.BROKEN, want="list"), STRATA_TOOLCALL_CONSTRAIN="1",
                   STRATA_TOOLCALL_REPAIR="1")
        c = self.post(self.ASK)
        self.assertEqual([t["function"]["name"] for t in c["message"]["tool_calls"]], ["list"])
        self.assertNotIn("tool call repaired", self.log)

    def test_no_tools_no_constraint(self):
        self.start(BiasEngine(CallTokenizer(), THINK + "hello"), STRATA_TOOLCALL_CONSTRAIN="1")
        c = self.post(self.ASK, tools=None)
        self.assertEqual(c["message"]["content"], "hello")
        self.assertEqual(len(self.engine.calls), 1)

    def history(self, k, repeat=False):
        m = [{"role": "user", "content": "go"}]
        for i in range(k):
            cmd = "ls" if repeat else f"ls {i}"
            m += [{"role": "assistant", "content": "", "reasoning_content": f"thinking number {i} " * 20,
                   "tool_calls": [{"id": f"c{i}", "type": "function",
                                   "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}]},
                  {"role": "tool", "tool_call_id": f"c{i}", "content": f"out {i}"}]
        return m

    def test_drop_thinking_above_threshold(self):
        self.start(MockEngine(CallTokenizer(), THINK + "done"), STRATA_DROP_THINKING_ABOVE="1000",
                   STRATA_DROP_THINKING_STEP="4")
        self.post(self.history(10))
        prompt = self.tok.decode(self.engine.last_prompt)
        self.assertNotIn("thinking number 0 ", prompt)
        self.assertNotIn("thinking number 7 ", prompt)
        self.assertIn("thinking number 8 ", prompt)            # boundary 8 of 10 turns: 8 and 9 keep theirs
        self.assertIn("thinking number 9 ", prompt)
        self.assertRegex(self.log, r"thinking of 8 older turns left out, \d+ tokens saved")

    def test_drop_thinking_below_threshold_untouched(self):
        self.start(MockEngine(CallTokenizer(), THINK + "done"), STRATA_DROP_THINKING_ABOVE="10000000")
        self.post(self.history(10))
        self.assertIn("thinking number 0 ", self.tok.decode(self.engine.last_prompt))

    def test_repeat_warning(self):
        self.start(MockEngine(CallTokenizer(), THINK + "done"), STRATA_REPEAT_WARN="1")
        self.post(self.history(2, repeat=True))
        self.assertIn(g.REPEAT_NOTE, self.tok.decode(self.engine.last_prompt))
        self.assertIn("[strata] repeated tool call: bash", self.log)
        rec = [json.loads(x) for x in Path(self.failog).read_text().splitlines()]
        self.assertEqual([(r["outcome"], r["name"], r["session"]) for r in rec], [("repeat", "bash", "ses_g1")])

    def test_repeat_warning_off(self):
        self.start(MockEngine(CallTokenizer(), THINK + "done"))
        self.post(self.history(2, repeat=True))
        self.assertNotIn(g.REPEAT_NOTE, self.tok.decode(self.engine.last_prompt))
        self.assertFalse(os.path.exists(self.failog))


if __name__ == "__main__":
    unittest.main()
