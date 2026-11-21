"""serve/test_reasoning_cache.py - a tool-call turn sent back without its thinking gets back the thinking of the reply
that made it, found by the reply's tool call ids: every turn its own (the same call made many times with the same
result included), on /v1/chat/completions and /v1/messages, streamed or not, with parallel calls, with the repeat note
(STRATA_REPEAT_WARN) in the prompt, and as the ids the engine generated (the prompt stays in the conversation cache).
Also: what is not restored (a rewritten turn, a restart, a bounded memory), the opt-in STRATA_SKIP_EMPTY_THINK and
chat_template_kwargs.preserve_thinking.  Mock engine, no GPU.

    python -m unittest serve.test_reasoning_cache -v
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import sys
import types
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import responses, toolguard  # noqa: E402
from serve.frontend import LITERAL_TAGS, CallList, ChatTemplate, anthropic_to_messages, openai_to_messages  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, _Written, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CALL = "<tool_call>\n<function={}>\n<parameter=path>\n{}\n</parameter>\n</function>\n</tool_call>"
TOOLS = [{"type": "function", "function": {"name": n, "description": n, "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}}}}} for n in ("run_tests", "read")]
A_TOOLS = [{"name": t["function"]["name"], "description": "d", "input_schema": t["function"]["parameters"]} for t in TOOLS]
USER = {"role": "user", "content": "fix the bug in a.py"}
EMPTY = "<think>\n\n</think>"
N = 40


def reply(k, calls=(("run_tests", "tests/"),), text=""):
    """What the model writes after the prompt's '<think>\\n': its plan for step k, then the calls (the template's layout)."""
    return f"plan {k}: step number {k}\n</think>\n\n" + text + "\n".join(CALL.format(*c) for c in calls)


class Base(unittest.TestCase):
    SCRIPTS = [reply(k) for k in range(N)]           # the same call every time (the run_tests loop), its own plan each

    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.engine = MockEngine(cls.tok, cls.SCRIPTS, max_context=65536, cycle=True)
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self.reset()
        self.svc.configure_toolguard({"STRATA_LOOP_BREAKER": "0"})   # these cases test the repeat note (0.5.0: breaker on by default)
        self.svc.template.skip_empty_think = False

    def reset(self):
        """A restarted server: nothing remembered, the script from its start."""
        self.svc.recent_reasoning.clear()
        self.svc.reasoning_by_call.clear()
        self.svc.reasoning_size[:] = [0, 0]
        self.engine.turns = 0

    def post(self, path, body):
        """(the reply or its raw stream, the prompt the engine got as text, as ids, the ids it generated, the log)."""
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        out = io.StringIO()
        with contextlib.redirect_stdout(out), urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode()
        ids = list(self.engine.last_prompt)
        return (raw if body.get("stream") else json.loads(raw)), self.tok.decode(ids), ids, \
            list(self.engine.script[:-1]), out.getvalue()

    def chat(self, messages, **extra):
        return self.post("/v1/chat/completions", {"messages": messages, "tools": TOOLS, **extra})

    def step(self, messages, result="3 failed, 2 passed", keep=False, **extra):
        """One agent step on /v1/chat/completions: the reply sent back WITHOUT reasoning_content (keep=True: with it),
        and a result per call.  Returns the new messages and (prompt ids, generated ids)."""
        data, _, ids, gen, _ = self.chat(messages, **extra)
        m = data["choices"][0]["message"]
        self.assertTrue(m.get("reasoning_content"))
        self.assertTrue(m.get("tool_calls"))
        if not keep:
            m.pop("reasoning_content")
        results = [{"role": "tool", "tool_call_id": c["id"], "content": result} for c in m["tool_calls"]]
        return messages + [m] + results, (ids, gen)

    def chain(self, n, keep=False, result="3 failed, 2 passed", **extra):
        msgs, runs = [USER], []
        for _ in range(n):
            msgs, run = self.step(msgs, result, keep, **extra)
            runs.append(run)
        return msgs, runs

    def plans(self, prompt):
        return [int(x.split(":")[0]) for x in prompt.split("<think>\nplan ")[1:]]


class ToolChain(Base):
    def test_same_call_same_result_every_turn_keeps_its_own_thinking(self):
        # the loop that lost its thinking: one call, one result, six times.  The answer and the message before it are
        # the same for every turn, so only the id tells the replies apart
        for warn in (False, True):
            with self.subTest(repeat_warn=warn):
                self.reset()
                self.svc.repeat_warn = warn
                msgs, _ = self.chain(6)
                _, prompt, _, _, log = self.chat(msgs)
                self.assertEqual(self.plans(prompt), [0, 1, 2, 3, 4, 5])
                self.assertNotIn(EMPTY, prompt)
                self.assertNotIn("not restored", log)
                self.assertEqual(toolguard.REPEAT_NOTE in prompt, warn)
                self.reset()                              # the same chain from a client that keeps the thinking
                kept, _ = self.chain(6, keep=True)
                _, truth, _, _, _ = self.chat(kept)
                self.assertEqual(prompt, truth)           # byte for byte the prompt of a client that sent it back

    def test_every_request_continues_the_one_before(self):
        # request k+1's prompt starts with request k's prompt and the ids its reply generated: the engine's session
        # for the conversation is reused whole (match_len needs ALL of its tokens)
        for warn in (False, True):
            with self.subTest(repeat_warn=warn):
                self.reset()
                self.svc.repeat_warn = warn
                msgs, runs = self.chain(7)
                _, _, last, _, log = self.chat(msgs)
                self.assertNotIn("not spliced", log)
                for (ids, gen), (after, _) in zip(runs, runs[1:] + [(last, None)]):
                    self.assertEqual(after[:len(ids) + len(gen)], ids + gen)
                    self.assertEqual(self.tok.decode(after[len(ids) + len(gen):])[:10], "<|im_end|>")

    def test_without_ids_the_repeat_note_no_longer_hides_the_turn(self):
        # a client that writes its own ids: the answer-keyed restore, whose prompt end now holds the note
        self.svc.repeat_warn = True
        msgs = [USER]
        for k in range(5):
            msgs, _ = self.step(msgs, f"output number {k}")
        for m in msgs:
            for c in m.get("tool_calls") or []:
                c["id"] = "x"
            if m.get("role") == "tool":
                m["tool_call_id"] = "x"
        _, prompt, _, _, log = self.chat(msgs)
        self.assertIn(toolguard.REPEAT_NOTE, prompt)
        self.assertEqual(self.plans(prompt), [0, 1, 2, 3, 4])
        self.assertNotIn("not restored", log)

    def test_a_result_that_ends_with_the_note_text_itself(self):
        # the tool printed the note's own words: both the prompt end as written and without the note are tried
        self.svc.repeat_warn = False
        note = f"\n\n[Note: {toolguard.REPEAT_NOTE}]"
        msgs, _ = self.step([USER], "the log says" + note)
        self.svc.reasoning_by_call.clear()
        _, prompt, _, _, _ = self.chat(msgs)
        self.assertEqual(self.plans(prompt), [0])

    def test_two_conversations_taking_turns(self):
        a, b = [USER], [{"role": "user", "content": "another task"}]
        for _ in range(4):                               # same calls, same results, in both
            a, _ = self.step(a)
            b, _ = self.step(b)
        _, pa, _, _, _ = self.chat(a)
        _, pb, _, _, _ = self.chat(b)
        self.assertEqual(self.plans(pa), [0, 2, 4, 6])
        self.assertEqual(self.plans(pb), [1, 3, 5, 7])

    def test_streamed_reply(self):
        raw, _, _, _, _ = self.chat([USER], stream=True)
        evs = [json.loads(x[6:]) for x in raw.splitlines() if x.startswith("data: {")]
        calls = [c for e in evs for ch in e.get("choices") or [] for c in ch.get("delta", {}).get("tool_calls") or []]
        cid = [c["id"] for c in calls if c.get("id")][0]
        msgs = [USER, {"role": "assistant", "content": None, "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": "run_tests", "arguments": "{\"path\": \"tests/\"}"}}]},
                {"role": "tool", "tool_call_id": cid, "content": "ok"}]
        _, prompt, _, _, log = self.chat(msgs, stream=True)
        self.assertEqual(self.plans(prompt), [0])
        self.assertNotIn("not spliced", log)

    def test_thinking_the_client_sent_is_kept(self):
        msgs, _ = self.step([USER])
        msgs[1]["reasoning_content"] = "the client's own text"
        _, prompt, _, _, _ = self.chat(msgs)
        self.assertIn("<think>\nthe client's own text\n</think>", prompt)
        self.assertEqual(self.plans(prompt), [])

    def test_text_beside_the_calls_may_change(self):
        # the id names the reply; clients trim the text or move it, the calls are what is compared
        msgs, _ = self.step([USER])
        msgs[1]["content"] = "(edited by the client)"
        self.svc.recent_reasoning.clear()
        _, prompt, _, _, log = self.chat(msgs)
        self.assertEqual(self.plans(prompt), [0])
        self.assertIn("restored turn not spliced", log)   # not the generated text any more: the plain encode
        self.assertIn("(edited by the client)", prompt)

    def test_a_rewritten_call_gets_nothing(self):
        msgs, _ = self.step([USER])
        msgs[1]["tool_calls"][0]["function"]["arguments"] = json.dumps({"path": "other/"})
        _, prompt, _, _, log = self.chat(msgs)
        self.assertEqual(self.plans(prompt), [])
        self.assertIn(EMPTY, prompt)
        self.assertIn("not restored by tool call id: its tool calls are not the ones", log)

    def test_an_id_on_another_call_gets_nothing(self):
        # the same id reused for a different call (a client that numbers its calls itself, or a mixed-up history)
        msgs, _ = self.step([USER])
        cid = msgs[1]["tool_calls"][0]["id"]
        other = [USER, {"role": "assistant", "content": "", "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": "read", "arguments": "{\"path\": \"a.py\"}"}}]},
                 {"role": "tool", "tool_call_id": cid, "content": "x = 1"}]
        _, prompt, _, _, _ = self.chat(other)
        self.assertEqual(self.plans(prompt), [])

    def test_an_id_given_again_names_the_newer_reply(self):
        enc = self.tok.encode
        call = types.SimpleNamespace(name="read", arguments={"path": "a.py"}, id="call_same")
        self.svc._remember_reasoning(enc("p1<think>\n"), True, "", [call], "first")
        self.svc._remember_reasoning(enc("p2<think>\n"), True, "", [call], "second")
        m = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_same", "function": {"name": "read", "arguments": {"path": "a.py"}}}]}
        self.svc._restore_reasoning([USER, m, {"role": "tool", "content": "r"}], log=False)
        self.assertEqual(m["reasoning_content"], "second")
        self.assertEqual(len(self.svc.reasoning_by_call), 1)
        self.assertEqual(self.svc.reasoning_size[1], self.svc.reasoning_by_call["call_same"][2])

    def test_restart_falls_back_quietly(self):
        msgs, _ = self.chain(3)
        self.reset()                                     # the server restarted: nothing remembered
        data, prompt, _, _, log = self.chat(msgs)
        self.assertEqual(prompt.count(EMPTY), 3)         # as before the cache: the template's empty block
        self.assertTrue(data["choices"][0]["message"]["tool_calls"])
        self.assertNotIn("Traceback", log)
        msgs2, _ = self.step(msgs)                       # and the conversation goes on: its new turns are remembered
        _, prompt, _, _, _ = self.chat(msgs2)
        self.assertEqual(prompt.count(EMPTY), 3)
        self.assertEqual(len(self.plans(prompt)), 1)

    def test_unknown_turns_never_borrow_a_newer_replys_thinking(self):
        # after a restart the old turns' ids are unknown.  The new replies make the same call and get the same result,
        # so their answer and the message before them are the old turns' too: they are still other replies
        msgs, _ = self.chain(3)
        self.reset()
        prompts = []
        for _ in range(3):
            msgs, _ = self.step(msgs)
            prompts.append(self.chat(msgs)[1])
            self.engine.turns -= 1                       # (that look at the prompt was not a step)
        for k, prompt in enumerate(prompts):
            self.assertEqual(prompt.count(EMPTY), 3)
            self.assertEqual(self.plans(prompt), list(range(k + 1)))
        for a, b in zip(prompts, prompts[1:]):           # and the old turns render the same on every request
            self.assertEqual(b[:len(a) - len("<|im_start|>assistant\n<think>\n")],
                             a[:len(a) - len("<|im_start|>assistant\n<think>\n")])

    def test_reply_cut_by_max_tokens(self):
        # cut inside the thinking: no call, no id, nothing to restore by id; the request after it still works
        data, _, _, _, _ = self.chat([USER], max_tokens=6)
        self.assertEqual(data["choices"][0]["finish_reason"], "length")
        self.assertEqual(len(self.svc.reasoning_by_call), 0)
        m = data["choices"][0]["message"]
        m.pop("reasoning_content", None)
        _, prompt, _, _, log = self.chat([USER, m, {"role": "user", "content": "go on"}])
        self.assertNotIn("Traceback", log)

    def test_switched_off(self):
        msgs, _ = self.chain(2)
        self.svc.configure_toolguard({"STRATA_REASONING_RESTORE": "0"})
        _, prompt, _, _, _ = self.chat(msgs)
        self.assertEqual(prompt.count(EMPTY), 2)
        self.svc.configure_toolguard({})
        _, prompt, _, _, _ = self.chat(msgs)
        self.assertEqual(self.plans(prompt), [0, 1])

    def test_thinking_off_request_still_renders_the_history_the_same(self):
        msgs, _ = self.chain(2)
        _, on, _, _, _ = self.chat(msgs)
        _, off, _, _, _ = self.chat(msgs, chat_template_kwargs={"enable_thinking": False})
        self.assertEqual(self.plans(on), [0, 1])
        self.assertEqual(self.plans(off), [0, 1])

    def test_count_bound(self):
        from serve import server as S
        with mock.patch.object(S, "REASONING_KEPT_MAX", 3):
            msgs = [USER]
            for k in range(6):
                msgs, _ = self.step(msgs, f"out {k}")
            self.assertEqual(len(self.svc.reasoning_by_call), 3)
            self.assertLessEqual(len(self.svc.recent_reasoning), 3)
            self.svc.recent_reasoning.clear()
            self.svc.reasoning_size[0] = 0
            _, prompt, _, _, _ = self.chat(msgs)
            self.assertEqual(self.plans(prompt), [3, 4, 5])      # the newest three; the older turns fall back
            self.assertEqual(prompt.count(EMPTY), 3)

    def held(self):
        s = self.svc
        return [sum(s._reasoning_size(e, w) for v in s.recent_reasoning.values() for e, w in v),
                sum(v[2] for v in s.reasoning_by_call.values())]

    def test_size_bound(self):
        one = None
        msgs = [USER]
        for k in range(8):
            msgs, _ = self.step(msgs, f"out {k}")
            self.assertEqual(self.svc.reasoning_size, self.held())
            if one is None:
                one = self.svc.reasoning_size[1]
                self.svc.reasoning_kept_bytes = int(one * 2.5)   # room for two replies
        self.assertEqual(len(self.svc.reasoning_by_call), 2)
        self.assertLessEqual(self.svc.reasoning_size[1], self.svc.reasoning_kept_bytes)
        self.assertLessEqual(self.svc.reasoning_size[0], self.svc.reasoning_kept_bytes)
        self.svc.reasoning_kept_bytes = 1                        # smaller than one reply: the newest still stays
        msgs, _ = self.step(msgs, "out 8")
        self.assertEqual(len(self.svc.reasoning_by_call), 1)
        self.assertEqual(len(self.svc.recent_reasoning), 1)
        self.assertEqual(self.svc.reasoning_size, self.held())
        _, prompt, _, _, _ = self.chat(msgs)
        self.assertEqual(self.plans(prompt), [8])

    def test_the_same_answer_33_times_keeps_the_sizes_right(self):
        enc = self.tok.encode
        for k in range(40):                                      # one key, more than the 32 kept per key
            self.svc._remember_reasoning(enc(f"prompt {k}<think>\n"), True, "same answer", None, f"thought {k}", [1, 2])
        self.assertEqual(len(self.svc.recent_reasoning), 1)
        self.assertEqual(len(next(iter(self.svc.recent_reasoning.values()))), 32)
        self.assertEqual(self.svc.reasoning_size, self.held())


class ParallelCalls(Base):
    SCRIPTS = [reply(k, (("read", f"a{k}.py"), ("read", f"b{k}.py")), text="Reading both.\n\n" if k % 2 else "")
               for k in range(N)]

    def test_two_calls_results_in_any_order(self):
        msgs = [USER]
        runs = []
        for k in range(4):
            msgs, run = self.step(msgs, f"content {k}")
            runs.append(run)
            msgs[-2], msgs[-1] = msgs[-1], msgs[-2]          # the results in the order the tools finished
        _, prompt, last, _, log = self.chat(msgs)
        self.assertEqual(self.plans(prompt), [0, 1, 2, 3])
        self.assertNotIn("not spliced", log)
        for (ids, gen), (after, _) in zip(runs, runs[1:] + [(last, None)]):
            self.assertEqual(after[:len(ids) + len(gen)], ids + gen)

    def test_one_message_for_the_whole_turn(self):
        # a client that keeps every call of the user turn in one assistant message (_split_tool_steps takes it apart
        # by the ids): each step gets its own thinking back
        msgs, _ = self.chain(3)
        calls = [c for m in msgs if m.get("tool_calls") for c in m["tool_calls"]]
        merged = [USER, {"role": "assistant", "content": "", "tool_calls": calls}] + \
            [m for m in msgs if m.get("role") == "tool"]
        _, prompt, _, _, _ = self.chat(merged)
        self.assertEqual(self.plans(prompt), [0, 1, 2])

    def test_a_reply_split_in_two_turns_gets_nothing(self):
        msgs, _ = self.step([USER])
        a, t1, t2 = msgs[1], msgs[2], msgs[3]
        split = [USER, dict(a, tool_calls=a["tool_calls"][:1]), t1, dict(a, tool_calls=a["tool_calls"][1:]), t2]
        _, prompt, _, _, log = self.chat(split)
        self.assertEqual(self.plans(prompt), [])             # not twice the same thinking
        self.assertIn("not restored by tool call id", log)

    def test_a_dropped_call_gets_nothing(self):
        msgs, _ = self.step([USER])
        msgs[1]["tool_calls"].pop()
        msgs.pop()
        _, prompt, _, _, _ = self.chat(msgs)
        self.assertEqual(self.plans(prompt), [])

    def test_calls_of_two_replies_in_one_turn_get_nothing(self):
        enc = self.tok.encode
        c1 = types.SimpleNamespace(name="read", arguments={"path": "a"}, id="id_1")
        c2 = types.SimpleNamespace(name="read", arguments={"path": "b"}, id="id_2")
        self.svc._remember_reasoning(enc("p1<think>\n"), True, "", [c1], "first")
        self.svc._remember_reasoning(enc("p2<think>\n"), True, "", [c2], "second")
        m = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "id_1", "function": {"name": "read", "arguments": {"path": "a"}}},
            {"id": "id_2", "function": {"name": "read", "arguments": {"path": "b"}}}]}
        self.svc._restore_reasoning([USER, m, {"role": "tool", "content": "r"}], log=False)
        self.assertNotIn("reasoning_content", m)
        dup = {"role": "assistant", "content": "", "tool_calls": [m["tool_calls"][0], m["tool_calls"][0]]}
        self.svc._restore_reasoning([USER, dup, {"role": "tool", "content": "r"}], log=False)
        self.assertNotIn("reasoning_content", dup)

    def anthropic(self, messages, **extra):
        return self.post("/v1/messages", {"model": "x", "max_tokens": 4000, "messages": messages, "tools": A_TOOLS,
                                          **extra})

    def test_messages_endpoint(self):
        # /v1/messages: the thinking blocks dropped, the tool_use ids sent back, results in the order they finished
        for warn in (False, True):
            with self.subTest(repeat_warn=warn):
                self.reset()
                self.svc.repeat_warn = warn
                msgs, runs = [USER], []
                for k in range(4):
                    data, _, ids, gen, _ = self.anthropic(msgs)
                    runs.append((ids, gen))
                    self.assertTrue([b for b in data["content"] if b["type"] == "thinking"])
                    kept = [b for b in data["content"] if b["type"] != "thinking"]
                    uses = [b for b in kept if b["type"] == "tool_use"]
                    self.assertEqual(len(uses), 2)
                    msgs = msgs + [{"role": "assistant", "content": kept}, {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": u["id"], "content": "same"} for u in reversed(uses)]}]
                self.svc.recent_reasoning.clear()        # only the ids can find the replies now
                _, prompt, last, _, log = self.anthropic(msgs)
                self.assertEqual(self.plans(prompt), [0, 1, 2, 3])
                self.assertNotIn(EMPTY, prompt)
                self.assertNotIn("not spliced", log)
                for (ids, gen), (after, _) in zip(runs, runs[1:] + [(last, None)]):
                    self.assertEqual(after[:len(ids) + len(gen)], ids + gen)

    def test_messages_endpoint_streamed(self):
        raw, _, _, _, _ = self.anthropic([USER], stream=True)
        evs = [json.loads(x[6:]) for x in raw.splitlines() if x.startswith("data: {")]
        uses = [e["content_block"] for e in evs if e.get("type") == "content_block_start"
                and e["content_block"].get("type") == "tool_use"]
        self.assertEqual(len(uses), 2)
        blocks = [{"type": "tool_use", "id": u["id"], "name": "read", "input": {"path": f"{x}0.py"}}
                  for u, x in zip(uses, "ab")]
        msgs = [USER, {"role": "assistant", "content": blocks}, {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": u["id"], "content": "text"} for u in uses]}]
        _, prompt, _, _, _ = self.anthropic(msgs, stream=True)
        self.assertEqual(self.plans(prompt), [0])


class Conversions(unittest.TestCase):
    """The emitted ids reach the restore from all three request forms, and never the template."""

    def test_chat(self):
        m, _, _ = openai_to_messages({"messages": [USER, {"role": "assistant", "content": None, "tool_calls": [
            {"id": "i1", "function": {"name": "read", "arguments": "{}"}},
            {"id": "i2", "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "i2", "content": "b"}, {"role": "tool", "tool_call_id": "i1", "content": "a"}]})
        self.assertEqual(m[1]["tool_calls"].ids, ["i1", "i2"])
        self.assertEqual([x["content"] for x in m[2:]], ["a", "b"])
        self.assertNotIn("i1", json.dumps(m))                # the ids are on no message and on no call
        self.assertEqual(Service._turn_call_ids(m[1]), ["i1", "i2"])
        legacy, _, _ = openai_to_messages({"messages": [USER, {"role": "assistant", "content": None, "tool_calls": [
            {"function": {"name": "read", "arguments": "{}"}}, {"function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "content": "a"}, {"role": "tool", "content": "b"}]})
        self.assertEqual(m, legacy)                          # the same messages as a history without ids
        self.assertEqual(Service._turn_call_ids(legacy[1]), [None, None])
        self.assertEqual(copy.deepcopy(m)[1]["tool_calls"].ids, ["i1", "i2"])

    def test_messages(self):
        m, _, _ = anthropic_to_messages({"messages": [USER, {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "a"}]}]})
        self.assertEqual(m[1]["tool_calls"].ids, ["toolu_1"])
        self.assertNotIn("toolu_1", json.dumps(m))

    def test_responses(self):
        msgs = [{"role": "user", "content": "q"},
                {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read", "arguments": {}},
                                                                     "_call_id": "c1"}]},
                {"role": "tool", "content": "r", "_call_id": "c1"}]
        responses._order_tool_results(msgs)
        self.assertEqual(msgs[1]["tool_calls"].ids, ["c1"])
        self.assertNotIn("c1", json.dumps(msgs))
        m = responses.input_messages({"input": [
            {"type": "message", "role": "user", "content": "q"},
            {"type": "function_call", "call_id": "c9", "name": "read", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c9", "output": "r"}]})
        self.assertEqual([getattr(x.get("tool_calls"), "ids", None) for x in m if x.get("tool_calls")], [["c9"]])

    def test_the_template_ignores_the_ids(self):
        t = ChatTemplate(ROOT / "serve/chat_template.jinja")
        m = [USER, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read", "arguments": {}}}]},
             {"role": "tool", "content": "r"}]
        with_ids = copy.deepcopy(m)
        with_ids[1]["tool_calls"] = CallList(with_ids[1]["tool_calls"], ["call_0123456789abcdef_0"])
        self.assertEqual(t.render(m, TOOLS), t.render(with_ids, TOOLS))


class SkipEmptyThink(Base):
    """STRATA_SKIP_EMPTY_THINK=1 (opt-in): a history turn whose thinking is not known renders without a thinking block."""

    def test_default_is_unchanged(self):
        msgs, _ = self.chain(2)
        self.reset()
        _, prompt, _, _, _ = self.chat(msgs)
        self.assertEqual(prompt.count(EMPTY), 2)
        self.svc.configure_toolguard({"STRATA_SKIP_EMPTY_THINK": "0", "STRATA_LOOP_BREAKER": "0"})
        _, again, _, _, _ = self.chat(msgs)
        self.assertEqual(prompt, again)

    def test_unknown_turns_get_no_block_and_known_ones_keep_theirs(self):
        msgs, _ = self.chain(2)
        self.reset()                                         # a restart: those two cannot be restored
        self.svc.configure_toolguard({"STRATA_SKIP_EMPTY_THINK": "1"})
        msgs, _ = self.step(msgs)                            # a third step, remembered
        _, prompt, _, _, log = self.chat(msgs)
        self.assertNotIn(EMPTY, prompt)
        self.assertEqual(prompt.count("<|im_start|>assistant\n<tool_call>"), 2)
        self.assertEqual(len(self.plans(prompt)), 1)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n"))    # the new reply still thinks
        self.assertNotIn("not spliced", log)
        _, again, _, _, _ = self.chat(msgs)                  # and the prompt is the same on every request
        self.assertEqual(prompt, again)

    def test_with_the_effort_turn_at_the_end(self):
        msgs, _ = self.chain(1)
        self.reset()
        self.svc.configure_toolguard({"STRATA_SKIP_EMPTY_THINK": "1"})
        with mock.patch.object(self.svc, "effort_end", True):
            for extra in ({"reasoning_effort": "low"}, {"chat_template_kwargs": {"enable_thinking": False}}, {}):
                _, prompt, _, _, _ = self.chat(msgs, **extra)
                self.assertNotIn(EMPTY, prompt.rsplit("<|im_start|>assistant", 1)[0])

    def test_plain_chat_turns(self):
        t = ChatTemplate(ROOT / "serve/chat_template.jinja")
        msgs = [USER, {"role": "assistant", "content": "done"}, {"role": "user", "content": "next"}]
        self.assertIn(EMPTY + "\n\ndone", t.render(msgs))
        t.skip_empty_think = True
        out = t.render(msgs)
        self.assertNotIn(EMPTY, out)
        self.assertIn("<|im_start|>assistant\ndone<|im_end|>", out)
        msgs[1]["reasoning_content"] = "why"
        self.assertIn("<think>\nwhy\n</think>\n\ndone", t.render(msgs))


class PreserveThinking(Base):
    def test_kwarg_passes_through(self):
        msgs = [USER, {"role": "assistant", "content": "done", "reasoning_content": "why"},
                {"role": "user", "content": "next"}]
        t = ChatTemplate(ROOT / "serve/chat_template.jinja")
        for ctk, want in (({"preserve_thinking": False}, False), ({"preserve_thinking": True}, True),
                          ({"preserve_thinking": "no"}, None), ({}, None),
                          ({"preserve_thinking": False, "enable_thinking": False}, False),
                          ({"enable_thinking": False, "preserve_thinking": False}, False),
                          ({"reasoning_effort": "low", "preserve_thinking": False}, False)):
            with self.subTest(ctk=ctk):
                _, _, kw = openai_to_messages({"messages": msgs, "chat_template_kwargs": ctk})
                self.assertIs(kw.get("preserve_thinking"), want)
                self.assertEqual("<think>\nwhy" in t.render(msgs, **kw), want is not False)
                if "enable_thinking" in ctk:
                    self.assertIs(kw.get("enable_thinking"), False)

    def test_false_drops_the_older_turns_thinking_only(self):
        msgs, _ = self.chain(2)
        msgs.append({"role": "assistant", "content": "ROOT CAUSE: x"})
        msgs.append({"role": "user", "content": "now the next bug"})
        self.engine.turns = 5
        msgs, _ = self.step(msgs)
        _, prompt, _, _, log = self.chat(msgs, chat_template_kwargs={"preserve_thinking": False})
        self.assertEqual(self.plans(prompt), [5])            # the running tool chain keeps its thinking
        self.assertNotIn("not spliced", log)
        _, prompt, _, _, _ = self.chat(msgs)
        self.assertEqual(self.plans(prompt), [0, 1, 5])


PACK_TOK = Path(os.environ.get("STRATA_TEST_PACK_TOK", ""))   # the Qwen3.8 pack tokenizer dir; the test skips when unset


@unittest.skipUnless(os.environ.get("STRATA_TEST_PACK_TOK") and (PACK_TOK / "vocab.json").exists(), "needs the Qwen3.8 pack tokenizer")
class ToolTurnKeepsItsIds(unittest.TestCase):
    """A restored tool-call turn is encoded as the ids the engine generated when they decode to what the template
    renders for it, also when its thinking began with whitespace (real tokenizer, no engine); else the plain encode."""

    def setUp(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import strata_tokenizer as ST
        vocab = json.loads((PACK_TOK / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for t, i in vocab.items():
            tokens[i] = t
        tcfg = json.loads((PACK_TOK / "tokenizer.json").read_text(encoding="utf-8"))
        self.tok = ST.Tokenizer(tokens, (PACK_TOK / "merges.txt").read_text(encoding="utf-8").split("\n"),
                                json.loads((PACK_TOK / "token_type.json").read_text()), tcfg.get("pre") or "qwen35",
                                tcfg.get("special_ids") or None)
        self.tpl = ChatTemplate(ROOT / "serve/chat_template.jinja")
        self.svc = types.SimpleNamespace(tok=self.tok, stop_ids=set(self.tok.encode("<|im_end|>", parse_special=True)),
                                         prompts=None, literals=LITERAL_TAGS)
        self.svc.render_prompt = lambda m, t, kw: self.tpl.render(m, t, add_generation_prompt=True)
        self.svc._splice_restored = lambda m, p: Service._splice_restored(self.svc, m, p)
        self.svc._restored_ids = lambda r, m, o: Service._restored_ids(self.svc, r, m, o)
        self.enc = lambda s: self.tok.encode(s, parse_special=True)

    def encode(self, msgs):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ids = Service.encode_prompt(self.svc, msgs, TOOLS, {})
        return ids, out.getvalue()

    def turn(self, prompt_ids, start, text, calls, tail=""):
        body = text + "\n".join(CALL.format(n, p) for n, p in calls)
        gen = self.enc(start + " the plan.") + self.enc("\n</think>\n\n") + self.enc(body + tail)
        reasoning = self.tok.decode(gen).split("</think>")[0].strip()
        return gen, {"role": "assistant", "content": text.strip(), "reasoning_content": _Written(reasoning, prompt_ids, gen),
                     "tool_calls": [{"function": {"name": n, "arguments": {"path": p}}} for n, p in calls]}

    def test_whitespace_started_thinking_stays_in_cache(self):
        cases = [("", [("read", "a.py")]), ("", [("read", "a.py"), ("run_tests", "tests/")]),
                 ("Let me look at the file.\n\n", [("read", "a.py")])]
        for start in ("Okay, so", "\nOkay, so", "\n\nOkay, so", " Okay, so"):
            for text, calls in cases:
                for tail in ("", "\n"):
                    with self.subTest(start=start, text=text[:6], calls=len(calls), tail=tail):
                        msgs, cached = [USER], None
                        for k in range(3):                   # a chain: every request continues the one before
                            prompt_ids, log = self.encode(msgs)
                            self.assertNotIn("not spliced", log)
                            if cached is not None:
                                self.assertEqual(prompt_ids[:len(cached)], cached, f"step {k} left the cache")
                            gen, m = self.turn(prompt_ids, start, text, calls, tail)
                            msgs = msgs + [m] + [{"role": "tool", "content": f"result {k}"} for _ in calls]
                            cached = prompt_ids + gen
                        ids, log = self.encode(msgs)
                        self.assertEqual(ids[:len(cached)], cached)
                        self.assertNotIn("not spliced", log)
                        # the text of the prompt is the rendered one up to the whitespace the model itself wrote
                        rendered = self.tpl.render(msgs, TOOLS, add_generation_prompt=True)
                        squeeze = lambda s: "".join(s.split())
                        self.assertEqual(squeeze(self.tok.decode(ids)), squeeze(rendered))

    def test_a_turn_that_differs_from_the_generated_one_is_encoded_plainly(self):
        prompt_ids, _ = self.encode([USER])
        for what in ("argument", "text", "layout"):
            with self.subTest(what=what):
                gen, m = self.turn(prompt_ids, "\nOkay, so", "", [("read", "a.py")])
                if what == "argument":
                    m["tool_calls"][0]["function"]["arguments"] = {"path": "b.py"}
                elif what == "text":
                    m["content"] = "added by the client"
                else:                                        # the model wrote the call in another layout
                    gen = self.enc("\nOkay, so the plan.\n</think>\n\n<tool_call>\n<function=read>"
                                   "<parameter=path>a.py</parameter>\n</function>\n</tool_call>")
                    m["reasoning_content"] = _Written(str(m["reasoning_content"]), prompt_ids, gen)
                msgs = [USER, m, {"role": "tool", "content": "r"}]
                ids, log = self.encode(msgs)
                self.assertIn("restored turn not spliced", log)
                self.assertEqual(ids, self.enc(self.tpl.render(msgs, TOOLS, add_generation_prompt=True)))


if __name__ == "__main__":
    unittest.main()
