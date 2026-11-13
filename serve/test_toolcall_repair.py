"""Tool call repair (opt-in, STRATA_TOOLCALL_REPAIR): an answer whose `<tool_call>` block does not read as a call of a
declared tool is written again from that `<tool_call>`, with `<function=` forced.

Live (an agent on Speedy 0.4.4, Qwen3.8-Flash-Next, 12 tools declared): the `<function=NAME>` line of the final reply came
out as `<4>`; the parser found no call, the block went back as text, and the agent read it as its answer and stopped.

    python -m unittest serve.test_toolcall_repair
"""
import json
import os
import tempfile
import unittest
from unittest import mock
import urllib.request
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve, toolcall_repair_tries

ROOT = Path(__file__).resolve().parent.parent
OPEN = "<tool_call>\n<function="
# the reply as the model wrote it (the `<function=NAME>` line is `<4>`)
BROKEN = ("<tool_call>\n<4>\n<parameter name=\"path\">/Users/bubu/.t3/projects/3d-horse-bench\n</parameter>\n"
          "</function>\n</tool_call>")
FIXED = "list>\n<parameter=path>\n/Users/bubu/.t3/projects/3d-horse-bench\n</parameter>\n</function>\n</tool_call>"
THINK = "I will list the folder.</think>\n\n"          # the reply thinks first (thinking is on)
CLEAN = "<tool_call>\n<function=list>\n<parameter=path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>"


class CallTokenizer(ByteTokenizer):
    """The pack's `<tool_call>` is one token, matched everywhere (type 4); the byte tokenizer spells it in bytes."""
    SPECIALS = ByteTokenizer.SPECIALS + ["<tool_call>"]
    ALWAYS = ("<tool_call>",)


def tool(name, *params):
    return {"type": "function", "function": {"name": name, "parameters": {
        "type": "object", "properties": {p: {"type": "string"} for p in params}, "required": list(params)}}}


TOOLS = [tool("list", "path"), tool("read", "path"), tool("bash", "command")]


class ScriptEngine(MockEngine):
    """The first answer is `first`; a prompt that ends with the forced opening gets `retry` (or BROKEN again)."""

    def __init__(self, tok, first, retry=None):
        super().__init__(tok, "x")
        self.first, self.retry, self.prompts = first, retry, []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        text = self.tok.decode(ids)
        out = (self.retry if self.retry is not None else BROKEN[len(OPEN):]) if text.endswith(OPEN) else self.first
        for t in (self.tok.encode(out, parse_special=True) + self.tok.encode("<|im_end|>", parse_special=True))[:max_new]:
            if cancel.is_set():
                return
            yield t


class Repair(unittest.TestCase):
    def start(self, first, retry=None, tries=1):
        self.logdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.logdir.cleanup)
        self.failog = os.path.join(self.logdir.name, "logs", "toolcall-failures.jsonl")
        env = mock.patch.dict(os.environ, {"STRATA_TOOLCALL_FAIL_LOG": self.failog})
        env.start()
        self.addCleanup(env.stop)
        self.tok = CallTokenizer()
        self.engine = ScriptEngine(self.tok, first, retry)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.toolcall_repair = tries
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def post(self, stream):
        body = {"model": "m", "messages": [{"role": "user", "content": "list the project"}], "tools": TOOLS,
                "max_tokens": 400, "stream": stream}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "x-session-id": "ses_test1"})
        log = StringIO()
        with redirect_stdout(log), urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode()
        self.log = log.getvalue()
        if not stream:
            return json.loads(raw)["choices"][0]
        content, calls, finish = "", {}, None
        for line in raw.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ch = json.loads(line[6:])["choices"]
            if not ch:
                continue
            d = ch[0].get("delta") or {}
            content += d.get("content") or ""
            for c in d.get("tool_calls") or []:
                e = calls.setdefault(c["index"], {"name": "", "arguments": ""})
                e["name"] += (c.get("function") or {}).get("name") or ""
                e["arguments"] += (c.get("function") or {}).get("arguments") or ""
            finish = ch[0].get("finish_reason") or finish
        return {"content": content, "calls": list(calls.values()), "finish": finish}

    def test_the_broken_call_is_written_again(self):
        self.start(THINK + "Let me look.\n\n" + BROKEN, FIXED)
        c = self.post(False)
        self.assertEqual(len(self.engine.prompts), 2)
        self.assertTrue(self.tok.decode(self.engine.prompts[1]).endswith("Let me look.\n\n" + OPEN))
        # the retry continues the first prompt (the engine reuses its cached prefix)
        self.assertEqual(self.engine.prompts[1][:len(self.engine.prompts[0])], self.engine.prompts[0])
        m = c["message"]
        self.assertEqual(c["finish_reason"], "tool_calls")
        self.assertEqual([(t["function"]["name"], json.loads(t["function"]["arguments"])) for t in m["tool_calls"]],
                         [("list", {"path": "/Users/bubu/.t3/projects/3d-horse-bench"})])
        self.assertNotIn("<4>", m.get("content") or "")
        self.assertNotIn("<tool_call>", m.get("content") or "")
        self.assertIn("[strata] tool call repaired: list", self.log)

    def test_streaming_never_sends_the_broken_text(self):
        self.start(THINK + "Let me look.\n\n" + BROKEN, FIXED)
        r = self.post(True)
        self.assertEqual(r["finish"], "tool_calls")
        self.assertNotIn("<4>", r["content"])
        self.assertNotIn("tool_call", r["content"])
        self.assertTrue(r["content"].startswith("Let me look."))
        self.assertEqual([(c["name"], json.loads(c["arguments"])) for c in r["calls"]],
                         [("list", {"path": "/Users/bubu/.t3/projects/3d-horse-bench"})])
        self.assertIn("[strata] tool call repaired: list", self.log)

    def test_a_retry_that_fails_again_returns_the_first_reply_as_before(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.start(THINK + "Let me look.\n\n" + BROKEN, None, tries=2)
                c = self.post(stream)
                self.assertEqual(len(self.engine.prompts), 3)          # the first reply and two retries
                text = c["content"] if stream else c["message"]["content"]
                self.assertEqual(text, "Let me look.\n\n" + BROKEN)
                self.assertIn("tool call repair failed after 2 tries", self.log)
                self.assertNotIn("tool call repaired", self.log)

    def test_off_returns_the_text_as_before(self):
        self.start(THINK + "Let me look.\n\n" + BROKEN, FIXED, tries=0)
        c = self.post(False)
        self.assertEqual(len(self.engine.prompts), 1)
        self.assertEqual(c["message"]["content"], "Let me look.\n\n" + BROKEN)

    def test_a_clean_call_is_not_touched(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.start(THINK + "Let me look.\n\n" + CLEAN, FIXED)
                c = self.post(stream)
                self.assertEqual(len(self.engine.prompts), 1)
                calls = c["calls"] if stream else [t["function"] for t in c["message"]["tool_calls"]]
                self.assertEqual([(t["name"], json.loads(t["arguments"])) for t in calls], [("list", {"path": "/tmp/x"})])
                self.assertNotIn("tool call repair", self.log)

    def test_an_answer_without_a_call_is_not_touched(self):
        self.start(THINK + "All done.", FIXED)
        c = self.post(False)
        self.assertEqual(len(self.engine.prompts), 1)
        self.assertEqual(c["message"]["content"], "All done.")

    def failures(self):
        if not os.path.exists(self.failog):
            return []
        with open(self.failog, encoding="utf-8") as f:
            return [json.loads(x) for x in f if x.strip()]

    def test_failure_log_repaired(self):
        self.start(THINK + "Let me look.\n\n" + BROKEN, FIXED)
        self.post(False)
        (r,) = self.failures()
        self.assertEqual((r["outcome"], r["tries"], r["repaired_name"]), ("repaired", 1, "list"))
        self.assertEqual(r["session"], "ses_test1")
        self.assertEqual(r["tools"], ["bash", "list", "read"])
        self.assertIn("<tool_call>\n<4>", r["snippet"])
        self.assertLessEqual(len(r["snippet"]), 300)
        self.assertGreater(r["prompt_tokens"], 0)
        self.assertGreater(r["output_tokens"], 0)
        for k in ("ts", "model", "preset"):
            self.assertIn(k, r)

    def test_failure_log_failed_and_off(self):
        for tries, outcome in ((1, "failed"), (0, "off")):
            with self.subTest(outcome=outcome):
                self.start(THINK + "Let me look.\n\n" + BROKEN, None, tries=tries)
                self.post(True)
                (r,) = self.failures()
                self.assertEqual((r["outcome"], r["tries"]), (outcome, tries))
                self.assertNotIn("repaired_name", r)
                self.assertIn("<4>", r["snippet"])

    def test_failure_log_undeclared_name(self):
        self.start(THINK + CLEAN.replace("=list>", "=nosuch>"), FIXED, tries=0)
        self.post(False)
        (r,) = self.failures()
        self.assertEqual(r["outcome"], "off")
        self.assertEqual(r.get("undeclared_names"), ["nosuch"])

    def test_no_failure_no_line(self):
        self.start(THINK + "Let me look.\n\n" + CLEAN, FIXED)
        self.post(False)
        self.assertEqual(self.failures(), [])

    def test_an_unwritable_log_does_not_fail_the_request(self):
        self.start(THINK + "Let me look.\n\n" + BROKEN, FIXED)
        blocker = os.path.join(self.logdir.name, "file")
        open(blocker, "w").close()
        with mock.patch.dict(os.environ, {"STRATA_TOOLCALL_FAIL_LOG": os.path.join(blocker, "x.jsonl")}):
            c = self.post(False)
        self.assertEqual(c["finish_reason"], "tool_calls")
        self.assertIn("failure log not written", self.log)

    def test_the_env_value(self):
        self.assertEqual([toolcall_repair_tries(v) for v in (None, "", "0", "1", "2", "5", "x")], [0, 0, 0, 1, 2, 2, 0])


if __name__ == "__main__":
    unittest.main()
