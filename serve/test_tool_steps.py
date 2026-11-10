"""A client that keeps one assistant message per user turn sends every tool call of the turn in that message,
followed by all the results.  The model produced them step by step (calls, results, calls, results), so the
merged form moves each new call in front of the earlier results and the prompt stops being a continuation of
what the engine holds.  Call ids carry the response that issued them; openai_to_messages puts the steps back."""
import re
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate, OutputParser, openai_to_messages

TEMPLATE = Path(__file__).with_name("chat_template.jinja")


def call(cid, name="search", **args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args or {"q": cid}}}


def result(cid, text):
    return {"role": "tool", "tool_call_id": cid, "content": text}


A0, A1, B0, C0 = "call_aaaaaaaaaaaaaaaa_0", "call_aaaaaaaaaaaaaaaa_1", "call_bbbbbbbbbbbbbbbb_0", "call_cccccccccccccccc_0"
USER = {"role": "user", "content": "compare these"}


def roles(messages):
    return [(m["role"], len(m.get("tool_calls") or [])) for m in messages]


class CallIds(unittest.TestCase):
    def emitted(self, parser):
        text = ("<tool_call>\n<function=search>\n<parameter=q>\na\n</parameter>\n</function>\n</tool_call>\n"
                "<tool_call>\n<function=search>\n<parameter=q>\nb\n</parameter>\n</function>\n</tool_call>")
        events = parser.feed(text) + parser.finish()
        return [e.call.id for e in events if e.kind == "tool_call"]

    def test_calls_of_one_response_share_its_marker(self):
        for stream in (False, True):
            ids = self.emitted(OutputParser(thinking=False, stream_tools=stream))
            self.assertEqual(len(ids), 2)
            marks = [re.fullmatch(r"call_([0-9a-f]{16})_(\d+)", i) for i in ids]
            self.assertTrue(all(marks), ids)
            self.assertEqual(marks[0].group(1), marks[1].group(1))
            self.assertEqual([m.group(2) for m in marks], ["0", "1"])

    def test_another_response_has_another_marker(self):
        a = self.emitted(OutputParser(thinking=False))
        b = self.emitted(OutputParser(thinking=False))
        self.assertNotEqual(a[0][:21], b[0][:21])


class TagInProse(unittest.TestCase):
    """A <tool_call> is a call only when <function= follows it; the tag named in a sentence is text."""
    REAL = "<tool_call>\n<function=write>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>"

    def run_parser(self, text, piece):
        p = OutputParser(thinking=False, stream_tools=False)
        events = []
        for i in range(0, len(text), piece):
            events += p.feed(text[i:i + piece])
        events += p.finish()
        content = "".join(e.text for e in events if e.kind == "content")
        return content, [(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"]

    def test_tag_named_before_the_real_call(self):
        text = "I will use the `<tool_call>` format now.\n" + self.REAL
        for piece in (1, 3, 7, len(text)):
            with self.subTest(piece=piece):
                content, calls = self.run_parser(text, piece)
                self.assertEqual(content, "I will use the `<tool_call>` format now.")
                self.assertEqual(calls, [("write", {"path": "a.txt"})])

    def test_tag_alone_is_text(self):
        for text in ("The tag is <tool_call>", "Write <tool_call> and then </tool_call> around it."):
            for piece in (1, 5, len(text)):
                with self.subTest(text=text, piece=piece):
                    self.assertEqual(self.run_parser(text, piece), (text, []))

    def test_a_plain_call_still_parses(self):
        for piece in (1, 4, len(self.REAL)):
            with self.subTest(piece=piece):
                self.assertEqual(self.run_parser("Saving it.\n\n" + self.REAL, piece),
                                 ("Saving it.", [("write", {"path": "a.txt"})]))


class SplitSteps(unittest.TestCase):
    def test_merged_turn_is_split_back_into_its_steps(self):
        merged = [USER, {"role": "assistant", "content": "", "tool_calls": [call(A0), call(A1), call(B0)]},
                  result(A0, "ra0"), result(A1, "ra1"), result(B0, "rb0")]
        stepped = [USER, {"role": "assistant", "content": "", "tool_calls": [call(A0), call(A1)]},
                   result(A0, "ra0"), result(A1, "ra1"),
                   {"role": "assistant", "content": "", "tool_calls": [call(B0)]}, result(B0, "rb0")]
        got = openai_to_messages({"messages": merged})[0]
        self.assertEqual(roles(got), [("user", 0), ("assistant", 2), ("tool", 0), ("tool", 0), ("assistant", 1), ("tool", 0)])
        tpl = ChatTemplate(TEMPLATE)
        want = openai_to_messages({"messages": stepped})[0]
        self.assertEqual(tpl.render(got, enable_thinking=False), tpl.render(want, enable_thinking=False))

    def test_each_request_of_a_turn_continues_the_previous_one(self):
        tpl = ChatTemplate(TEMPLATE)
        calls, results, prompts = [], [], []
        for cid in (A0, B0, C0):
            calls.append(call(cid))
            results.append(result(cid, "r" + cid))
            msgs = [USER, {"role": "assistant", "content": "", "tool_calls": list(calls)}] + list(results)
            prompts.append(tpl.render(openai_to_messages({"messages": msgs})[0], add_generation_prompt=False))
        self.assertTrue(prompts[1].startswith(prompts[0]))
        self.assertTrue(prompts[2].startswith(prompts[1]))

    def test_answer_of_a_finished_turn_follows_its_last_results(self):
        merged = [USER, {"role": "assistant", "content": "the answer", "tool_calls": [call(A0), call(B0)]},
                  result(A0, "ra0"), result(B0, "rb0"), {"role": "user", "content": "thanks"}]
        got = openai_to_messages({"messages": merged})[0]
        self.assertEqual(roles(got), [("user", 0), ("assistant", 1), ("tool", 0), ("assistant", 1), ("tool", 0),
                                      ("assistant", 0), ("user", 0)])
        self.assertEqual(got[5]["content"], "the answer")
        self.assertFalse(got[1]["content"])

    def test_text_of_a_running_turn_stays_with_its_first_step(self):
        merged = [USER, {"role": "assistant", "content": "let me look", "reasoning_content": "hm",
                         "tool_calls": [call(A0), call(B0)]}, result(A0, "ra0"), result(B0, "rb0")]
        got = openai_to_messages({"messages": merged})[0]
        self.assertEqual(roles(got), [("user", 0), ("assistant", 1), ("tool", 0), ("assistant", 1), ("tool", 0)])
        self.assertEqual(got[1]["content"], "let me look")
        self.assertNotIn("reasoning_content", got[1])
        self.assertEqual(got[3]["reasoning_content"], "hm")

    def test_left_alone(self):
        old = ["call_" + "0123456789abcdef01234567", "call_" + "89abcdef0123456789abcdef"]
        cases = {
            "one response": [call(A0), call(A1)],
            "ids of another server": [call(old[0]), call(old[1])],
            "a response's calls apart": [call(A0), call(B0), call(A1)],
            "a repeated id": [call(A0), call(B0), call(B0)],
            "a foreign id beside ours": [call(A0), call(old[0])],
        }
        for name, calls in cases.items():
            with self.subTest(name):
                msgs = [USER, {"role": "assistant", "content": "", "tool_calls": calls}] + \
                       [result(c["id"], "r") for c in calls]
                self.assertEqual(roles(openai_to_messages({"messages": msgs})[0]),
                                 [("user", 0), ("assistant", len(calls))] + [("tool", 0)] * len(calls))
        with self.subTest("a result of an unknown call"):
            msgs = [USER, {"role": "assistant", "content": "", "tool_calls": [call(A0), call(B0)]},
                    result(A0, "r"), result("call_else", "r")]
            self.assertEqual(roles(openai_to_messages({"messages": msgs})[0]),
                             [("user", 0), ("assistant", 2), ("tool", 0), ("tool", 0)])

    def test_switched_off(self):
        import os
        from unittest import mock
        merged = [USER, {"role": "assistant", "content": "", "tool_calls": [call(A0), call(B0)]},
                  result(A0, "ra0"), result(B0, "rb0")]
        with mock.patch.dict(os.environ, {"STRATA_TOOL_STEP_SPLIT": "0"}):
            got = openai_to_messages({"messages": merged})[0]
        self.assertEqual(roles(got), [("user", 0), ("assistant", 2), ("tool", 0), ("tool", 0)])


if __name__ == "__main__":
    unittest.main()
