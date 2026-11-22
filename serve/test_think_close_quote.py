"""serve/test_think_close_quote.py - a `</think>` right after a quote that closes a sentence ends the thinking.

`he said."</think>The answer` is the end of the thinking; a pair of quotes around the tag (`"</think>"`) and the
escaped `\\n</think>` form are still the model writing about the tag (#537).  STRATA_THINK_CLOSE_QUOTE=0 restores
the old rule (any quote right before the tag, no line break after it: a mention).

    python -m unittest serve.test_think_close_quote -v
"""
from __future__ import annotations

import json
import os
import sys
import unittest
import unittest.mock
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate, OutputParser  # noqa: E402
from serve.server import ByteTokenizer, Service, serve  # noqa: E402
from serve.test_server import ThinkingEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def run(text, width):
    p = OutputParser(thinking=True)
    evs = []
    for i in range(0, len(text), width):
        evs += p.feed(text[i:i + width])
    return evs + p.finish()


def joined(evs, kind):
    return "".join(e.text for e in evs if e.kind == kind)


CLOSING = ['he said "it is 4."', "she wrote 'done!'", 'is it "x?"', "the note: `stop.`", 'he said “fine…"']
PAIRS = ['split it on "</think>" first', "the tag `</think>` ends it", "split on '</think>' here",
         'he said."</think>" is the tag']


class Parser(unittest.TestCase):
    def test_a_sentence_closing_quote_ends_the_thinking(self):
        for thought in CLOSING:
            for width in (1, 2, 3, 100_000):
                with self.subTest(thought=thought, width=width):
                    evs = run(thought + "</think>The answer is 4", width)
                    self.assertEqual(joined(evs, "reasoning"), thought)
                    self.assertEqual(joined(evs, "content"), "The answer is 4")

    def test_a_quote_pair_stays_a_mention(self):
        for thought in PAIRS:
            text = "plan: " + thought + "\ndone\n</think>\n\nanswer"
            for width in (1, 2, 3, 100_000):
                with self.subTest(thought=thought, width=width):
                    evs = run(text, width)
                    self.assertEqual(joined(evs, "reasoning"), "plan: " + thought + "\ndone\n")
                    self.assertEqual(joined(evs, "content"), "answer")

    def test_a_quote_without_sentence_punctuation_stays_a_mention(self):
        for width in (1, 100_000):
            evs = run('he said "x"</think>more thinking\n</think>\n\nanswer', width)
            self.assertEqual(joined(evs, "content"), "answer")

    def test_the_escaped_form_stays_a_mention(self):
        for width in (1, 2, 3, 100_000):
            evs = run('the marker ".\\n</think>\\n\\n" appears\n</think>\n\nanswer', width)
            self.assertEqual(joined(evs, "content"), "answer")

    def test_switch_off_keeps_the_old_rule(self):
        with unittest.mock.patch.dict(os.environ, {"STRATA_THINK_CLOSE_QUOTE": "0"}):
            for width in (1, 100_000):
                evs = run('he said "it is 4."</think>The answer is 4', width)
                self.assertEqual(joined(evs, "content"), "")
                self.assertEqual(joined(evs, "reasoning"), 'he said "it is 4."</think>The answer is 4')


class QuoteEngine(ThinkingEngine):
    THOUGHT = 'He said "two plus two is 4."'
    ANSWER = "The answer is 4"

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        text = self.THOUGHT + "</think>" + self.ANSWER
        for t in (self.tok.encode(text) + self.tok.encode("<|im_end|>", parse_special=True))[:max_new]:
            if cancel.is_set():
                return
            yield t


class Endpoints(unittest.TestCase):
    """Chat completions, responses and anthropic messages, streaming and not."""

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = QuoteEngine(self.tok)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode()

    def chat(self, stream):
        raw = self.post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "2+2?"}],
                                                 "max_tokens": 300, "stream": stream})
        if not stream:
            m = json.loads(raw)["choices"][0]["message"]
            return m.get("reasoning_content") or "", m.get("content") or ""
        r = c = ""
        for line in raw.splitlines():
            if line.startswith("data: {"):
                for ch in json.loads(line[6:]).get("choices", []):
                    d = ch.get("delta", {})
                    r += d.get("reasoning_content") or ""
                    c += d.get("content") or ""
        return r, c

    def responses(self, stream):
        raw = self.post("/v1/responses", {"model": "m", "input": "2+2?", "stream": stream})
        if not stream:
            out = json.loads(raw)["output"]
            c = "".join(p.get("text", "") for o in out if o.get("type") == "message" for p in o.get("content", []))
            r = "".join(p.get("text", "") for o in out if o.get("type") == "reasoning"
                        for p in (o.get("content") or []) + (o.get("summary") or []))
            return r, c
        r = c = ""
        for line in raw.splitlines():
            if line.startswith("data: {"):
                e = json.loads(line[6:])
                if e.get("type") == "response.output_text.delta":
                    c += e["delta"]
                elif e.get("type", "").startswith("response.reasoning") and e.get("type", "").endswith(".delta"):
                    r += e["delta"]
        return r, c

    def messages(self, stream):
        raw = self.post("/v1/messages", {"model": "m", "max_tokens": 300, "stream": stream,
                                         "messages": [{"role": "user", "content": "2+2?"}]})
        if not stream:
            blocks = json.loads(raw)["content"]
            return ("".join(b.get("thinking", "") for b in blocks if b["type"] == "thinking"),
                    "".join(b.get("text", "") for b in blocks if b["type"] == "text"))
        r = c = ""
        for line in raw.splitlines():
            if line.startswith("data: {"):
                d = json.loads(line[6:]).get("delta", {})
                r += d.get("thinking", "") if d.get("type") == "thinking_delta" else ""
                c += d.get("text", "") if d.get("type") == "text_delta" else ""
        return r, c

    def test_every_endpoint_ends_the_thinking(self):
        for name in ("chat", "responses", "messages"):
            for stream in (False, True):
                with self.subTest(endpoint=name, stream=stream):
                    r, c = getattr(self, name)(stream)
                    self.assertEqual(c, "The answer is 4")
                    self.assertIn("two plus two is 4.", r)

    def test_switch_off_keeps_the_old_rule(self):
        with unittest.mock.patch.dict(os.environ, {"STRATA_THINK_CLOSE_QUOTE": "0"}):
            for stream in (False, True):
                with self.subTest(stream=stream):
                    r, c = self.chat(stream)
                    self.assertNotEqual(c, "The answer is 4")
                    self.assertIn("</think>", r + c)


if __name__ == "__main__":
    unittest.main()
