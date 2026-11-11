"""serve/test_restore_reasoning.py - an answer sent back without its thinking gets the thinking this server wrote for it
(mock engine, no GPU).

The template keeps a turn's thinking (preserve_thinking), and OpenAI chat clients drop reasoning_content from the
history, so the next prompt left the conversation cache at that turn and the whole answer was read again.

    python -m unittest serve.test_restore_reasoning -v
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import LITERAL_TAGS, ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class RestoreReasoning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.engine = MockEngine(cls.tok, "<think>\nmy private plan\n</think>\n\nthe answer", max_context=8192)
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self.svc.recent_reasoning.clear()

    def chat(self, messages, stream=False):
        req = urllib.request.Request(self.base + "/v1/chat/completions",
                                     data=json.dumps({"messages": messages, "stream": stream}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
        return self.tok.decode(self.engine.last_prompt)

    def test_dropped_thinking_comes_back(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.svc.recent_reasoning.clear()
                self.chat([{"role": "user", "content": "q1"}], stream)
                prompt = self.chat([{"role": "user", "content": "q1"}, {"role": "assistant", "content": "the answer"},
                                    {"role": "user", "content": "q2"}], stream)
                self.assertIn("<think>\nmy private plan\n</think>\n\nthe answer", prompt)

    def test_reply_cut_off_while_thinking_keeps_its_thinking(self):
        # finish "length" inside the thinking: no answer text, the client sends the turn back with content ""
        ids = self.tok.encode("<|im_start|>user\nwhich one is largest?<|im_end|>\n<|im_start|>assistant\n<think>\n")
        self.svc._remember_reasoning(ids, "half a plan", "", None, "half a plan")
        msgs = [{"role": "user", "content": "which one is largest?"}, {"role": "assistant", "content": ""},
                {"role": "user", "content": "and the smallest?"}]
        self.svc._restore_reasoning(msgs, log=False)
        self.assertEqual(msgs[1].get("reasoning_content"), "half a plan")
        other = [{"role": "user", "content": "a different question"}, {"role": "assistant", "content": ""},
                 {"role": "user", "content": "next"}]
        self.svc._restore_reasoning(other, log=False)
        self.assertNotIn("reasoning_content", other[1])   # another conversation's empty turn gets nothing

    def test_client_thinking_and_unknown_answers_stay(self):
        self.chat([{"role": "user", "content": "q1"}])
        prompt = self.chat([{"role": "user", "content": "q1"},
                            {"role": "assistant", "content": "the answer", "reasoning_content": "theirs"},
                            {"role": "user", "content": "q2"}])
        self.assertIn("<think>\ntheirs\n</think>\n\nthe answer", prompt)
        prompt = self.chat([{"role": "user", "content": "q1"}, {"role": "assistant", "content": "another answer"},
                            {"role": "user", "content": "q2"}])
        self.assertIn("<think>\n\n</think>\n\nanother answer", prompt)


class RestoreReasoningSameAnswer(unittest.TestCase):
    """Two replies with the same answer and different thinking: each turn of the history gets its own back (one key
    for both gave the older turn the newer thinking, and the prompt left the cache at the older turn)."""

    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.engine = MockEngine(cls.tok, ["<think>\nplan one\n</think>\n\nok", "<think>\nplan two\n</think>\n\nok"],
                                max_context=8192)
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self.svc.recent_reasoning.clear()
        self.engine.turns, self.engine.script = 0, self.engine.scripts[0]

    def chat(self, messages):
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps({"messages": messages}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
        return self.tok.decode(self.engine.last_prompt)

    def test_each_turn_gets_its_own_thinking(self):
        u = lambda s: {"role": "user", "content": s}
        ok = {"role": "assistant", "content": "ok"}
        self.chat([u("first question")])
        self.chat([u("first question"), ok, u("second question")])
        prompt = self.chat([u("first question"), ok, u("second question"), ok, u("third question")])
        one, two = prompt.index("plan one"), prompt.index("plan two")
        self.assertLess(prompt.index("first question"), one)
        self.assertLess(one, prompt.index("second question"))
        self.assertLess(prompt.index("second question"), two)

    def test_a_turn_without_its_own_thinking_gets_none(self):
        u = lambda s: {"role": "user", "content": s}
        ok = {"role": "assistant", "content": "ok"}
        self.chat([u("question x")])   # this reply's thinking is remembered with "question x" before it
        prompt = self.chat([u("question y"), ok, u("question z")])   # an "ok" that no reply here wrote
        self.assertNotIn("plan", prompt)


class RestoreReasoningToolCall(unittest.TestCase):
    """A tool-call turn has no answer text: its thinking is found by the call (name and arguments)."""
    TOOLS = [{"type": "function", "function": {"name": "get", "description": "d",
                                               "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]

    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.engine = MockEngine(cls.tok, "<think>\nplan the call\n</think>\n\n<tool_call>\n<function=get>\n<parameter=q>\nx\n"
                                         "</parameter>\n</function>\n</tool_call>", max_context=8192)
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def chat(self, messages):
        req = urllib.request.Request(self.base + "/v1/chat/completions",
                                     data=json.dumps({"messages": messages, "tools": self.TOOLS}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read())
        return body, self.tok.decode(self.engine.last_prompt)

    def test_tool_call_turn_gets_its_thinking(self):
        body, _ = self.chat([{"role": "user", "content": "find x"}])
        calls = body["choices"][0]["message"]["tool_calls"]
        _, prompt = self.chat([{"role": "user", "content": "find x"},
                               {"role": "assistant", "content": None, "tool_calls": calls},
                               {"role": "tool", "tool_call_id": calls[0]["id"], "content": "x is 1"}])
        self.assertIn("<think>\nplan the call\n</think>", prompt)


if __name__ == "__main__":
    unittest.main()


PACK_TOK = Path("/srv/models/strata-qwen38/packs/unsloth-ud-iq4_xs/tokenizer")


@unittest.skipUnless((PACK_TOK / "vocab.json").exists(), "needs the Qwen3.8 pack tokenizer")
class RestoredTurnKeepsItsIds(unittest.TestCase):
    """A reply whose thinking began with whitespace ('\\n', ' ', '\\n\\n'): the template trims it, and '<think>\\n' +
    '\\nOkay' re-tokenizes as '<think>' '\\n\\n' 'Okay', so the next prompt left the cache at the end of the turn's
    prompt.  The restored turn is encoded as the ids the engine generated (real tokenizer, no engine)."""

    def setUp(self):
        import types
        sys.path.insert(0, str(ROOT / "tools"))
        import strata_tokenizer as ST
        from serve.server import Service, _Written
        vocab = json.loads((PACK_TOK / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for t, i in vocab.items():
            tokens[i] = t
        tcfg = json.loads((PACK_TOK / "tokenizer.json").read_text(encoding="utf-8"))
        self.tok = ST.Tokenizer(tokens, (PACK_TOK / "merges.txt").read_text(encoding="utf-8").split("\n"),
                                json.loads((PACK_TOK / "token_type.json").read_text()), tcfg.get("pre") or "qwen35",
                                tcfg.get("special_ids") or None)
        self.tpl = ChatTemplate(ROOT / "serve/chat_template.jinja")
        self.Written = _Written
        self.svc = types.SimpleNamespace(tok=self.tok, stop_ids=set(self.tok.encode("<|im_end|>", parse_special=True)),
                                         prompts=None, literals=LITERAL_TAGS)
        self.svc.render_prompt = lambda m, t, kw: self.tpl.render(m, t, add_generation_prompt=True)
        self.svc._splice_restored = lambda m, p: Service._splice_restored(self.svc, m, p)
        self.svc._restored_ids = lambda r, m, o: Service._restored_ids(self.svc, r, m, o)
        self.encode_prompt = lambda m: Service.encode_prompt(self.svc, m, None, {})

    def test_whitespace_started_thinking_stays_in_cache(self):
        enc = lambda s: self.tok.encode(s, parse_special=True)
        u1 = {"role": "user", "content": "Explain caching."}
        prompt_ids = enc(self.tpl.render([u1], add_generation_prompt=True))
        wrap = "\n\nI have thought about this long enough; time to give my answer.\n</think>\n\n"
        for start in ("Okay, so", "\nOkay, so", "\n\nOkay, so", " Okay, so"):
            for close in ("\n</think>\n\n", wrap):
                with self.subTest(start=start, close=close[:4]):
                    gen = enc(start + " caching.") + enc(close) + enc("Caching stores resu")
                    reasoning = self.tok.decode(gen).split("</think>")[0].strip()
                    msgs = [u1, {"role": "assistant", "content": "Caching stores resu",
                                 "reasoning_content": self.Written(reasoning, prompt_ids, gen)},
                            {"role": "user", "content": "more"}]
                    ids = self.encode_prompt(msgs)
                    self.assertEqual(ids[:len(prompt_ids) + len(gen)], prompt_ids + gen)
                    rendered = self.tpl.render(msgs, add_generation_prompt=True)   # the same turns, the model's whitespace
                    self.assertEqual(self.tok.decode(ids).partition("Caching stores resu")[2],
                                     rendered.partition("Caching stores resu")[2])
                    plain = dict(msgs[1], reasoning_content=str(reasoning))    # not remembered: the plain encode
                    self.assertEqual(self.encode_prompt([u1, plain, msgs[2]]),
                                     enc(self.tpl.render([u1, plain, msgs[2]], add_generation_prompt=True)))

    def test_every_restored_turn_keeps_its_ids(self):
        # an agent thread: each request's prompt is what the engine cached so far, then the reply it generated
        enc = lambda s: self.tok.encode(s, parse_special=True)
        starts = ["\nFirst, so", " Second, so", "Third, so", "\n\nFourth, so"]
        msgs, cached = [], None
        for k, start in enumerate(starts):
            msgs.append({"role": "user", "content": f"question {k}"})
            prompt_ids = self.encode_prompt(msgs)
            if cached is not None:
                self.assertEqual(prompt_ids[:len(cached)], cached, f"turn {k} left the cache")
            answer = f"Answer {k} is cut of"
            gen = enc(start + f" thinking {k}.") + enc("\n</think>\n\n") + enc(answer)
            reasoning = self.tok.decode(gen).split("</think>")[0].strip()
            msgs.append({"role": "assistant", "content": answer,
                         "reasoning_content": self.Written(reasoning, prompt_ids, gen)})
            cached = prompt_ids + gen
        msgs.append({"role": "user", "content": "last"})
        ids = self.encode_prompt(msgs)
        self.assertEqual(ids[:len(cached)], cached)
        # a turn that cannot be checked (ids that do not decode to its text) falls back alone; the others still splice
        bad = [dict(m) for m in msgs]
        bad[3]["reasoning_content"] = self.Written(str(bad[3]["reasoning_content"]), bad[3]["reasoning_content"].prompt_ids,
                                                   enc("something else\n</think>\n\nnope"))
        ids = self.encode_prompt(bad)
        self.assertEqual(self.tok.decode(ids).count("\nFirst, so"), 1)
        self.assertEqual(self.tok.decode(ids).count("\n\nFourth, so"), 1)

    def test_length_cut_answer_is_spliced(self):
        # finish "length" mid-answer: no stop token, and the cut can fall after a space or newline, which the template
        # trims (content|trim) while the client sends the streamed text back as it was
        enc = lambda s: self.tok.encode(s, parse_special=True)
        u1 = {"role": "user", "content": "Explain caching."}
        prompt_ids = enc(self.tpl.render([u1], add_generation_prompt=True))
        for answer in ("Caching stores resu", "Caching stores ", "Caching stores.\n", "Caching stores.\n\n"):
            with self.subTest(answer=answer):
                gen = enc("\nOkay, so caching.") + enc("\n</think>\n\n") + enc(answer)
                reasoning = self.tok.decode(gen).split("</think>")[0].strip()
                msgs = [u1, {"role": "assistant", "content": answer,
                             "reasoning_content": self.Written(reasoning, prompt_ids, gen)},
                        {"role": "user", "content": "more"}]
                ids = self.encode_prompt(msgs)
                self.assertEqual(ids[:len(prompt_ids) + 8], prompt_ids + gen[:8])   # past the turn-1 prompt end
                keep = len(enc(answer.rstrip()))
                lost = len(prompt_ids) + len(gen) - next(
                    (i for i, (a, b) in enumerate(zip(ids, prompt_ids + gen)) if a != b), len(prompt_ids) + len(gen))
                self.assertLessEqual(lost, len(enc(answer)) - keep + 1)
