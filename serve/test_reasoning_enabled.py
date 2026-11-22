"""serve/test_reasoning_enabled.py - `reasoning: {"enabled": false}` is thinking off (Chat Completions and Responses),
and the effort precedence matrix: where the level comes from x API x tools x floor -> template kwargs and floor line.

    python -m unittest serve.test_reasoning_enabled -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import openai_to_messages  # noqa: E402
from serve.responses import template_kwargs  # noqa: E402
from serve.test_effort_floor import MSG, TOOLS, fake, run  # noqa: E402

OFF = {"enable_thinking": False}


def chat_kwargs(req):
    return openai_to_messages({"messages": MSG, **req})[2]


class ChatEnabledFalse(unittest.TestCase):
    def test_same_as_none(self):
        self.assertEqual(chat_kwargs({"reasoning": {"enabled": False}}), chat_kwargs({"reasoning_effort": "none"}))
        self.assertEqual(chat_kwargs({"reasoning": {"enabled": False}}), OFF)

    def test_explicit_level_wins(self):
        self.assertEqual(chat_kwargs({"reasoning": {"enabled": False, "effort": "high"}}),
                         chat_kwargs({"reasoning_effort": "high"}))
        self.assertEqual(chat_kwargs({"reasoning": {"enabled": False}, "reasoning_effort": "low"}),
                         chat_kwargs({"reasoning_effort": "low"}))

    def test_enabled_true_or_absent_unchanged(self):
        self.assertEqual(chat_kwargs({"reasoning": {"enabled": True}}), {})
        self.assertEqual(chat_kwargs({"reasoning": {}}), {})
        self.assertEqual(chat_kwargs({}), {})

    def test_template_kwargs_still_override(self):
        self.assertEqual(chat_kwargs({"reasoning": {"enabled": False},
                                      "chat_template_kwargs": {"enable_thinking": False}}), OFF)


class ResponsesEnabledFalse(unittest.TestCase):
    def test_same_as_none(self):
        self.assertEqual(template_kwargs({"reasoning": {"enabled": False}}, {}),
                         template_kwargs({"reasoning": {"effort": "none"}}, {}))

    def test_beats_shared_default(self):
        self.assertEqual(template_kwargs({"reasoning": {"enabled": False}}, {"reasoning_effort": "high"}), OFF)

    def test_explicit_level_wins(self):
        self.assertEqual(template_kwargs({"reasoning": {"enabled": False, "effort": "medium"}}, {}),
                         template_kwargs({"reasoning": {"effort": "medium"}}, {}))


class PrecedenceMatrix(unittest.TestCase):
    """Source of the level x API x tools x floor: rendered kwargs (chat) and whether the floor raises it."""
    OPENAI = [  # (request part, thinking off by the client, explicit level below the low/medium floor)
        ({}, False, None),
        ({"reasoning_effort": "none"}, True, "none"),
        ({"reasoning_effort": "low"}, False, "low"),
        ({"reasoning": {"effort": "none"}}, True, "none"),
        ({"reasoning": {"enabled": False}}, True, None),
        ({"reasoning": {"enabled": False, "effort": "low"}}, False, "low"),
        ({"chat_template_kwargs": {"enable_thinking": False}}, True, "none"),
        ({"chat_template_kwargs": {"reasoning_effort": "off"}}, True, "none"),
    ]
    ANTHROPIC = [
        ({}, False),
        ({"thinking": {"type": "disabled"}}, True),
        ({"output_config": {"effort": "none"}}, True),
        ({"output_config": {"effort": "low"}}, False),
    ]

    def test_openai(self):
        for part, off, _ in self.OPENAI:
            for tools in (False, True):
                for floor in (None, "medium"):
                    req = {"messages": MSG, **part, **({"tools": TOOLS} if tools else {})}
                    r, log = run(fake(floor), req)
                    raised = "effort floor" in log
                    should = bool(floor) and tools and not off and part != {} and \
                        (part.get("reasoning_effort") == "low" or part.get("reasoning", {}).get("effort") == "low")
                    self.assertEqual(raised, should, (part, tools, floor, log))
                    if off:
                        self.assertEqual(openai_to_messages(r)[2].get("enable_thinking"), False, (part, tools, floor))

    def test_anthropic(self):
        for part, off in self.ANTHROPIC:
            for tools in (False, True):
                for floor in (None, "medium"):
                    req = {"messages": MSG, **part, **({"tools": TOOLS} if tools else {})}
                    _, log = run(fake(floor), req, "anthropic")
                    should = bool(floor) and tools and not off and part != {}
                    self.assertEqual("effort floor" in log, should, (part, tools, floor, log))

    def test_off_is_not_raised_by_server_default(self):
        # a server default level only fills a request that names none: enabled:false names one (off)
        r, _ = run(fake(None, effort="high"), {"messages": MSG, "reasoning": {"enabled": False}})
        self.assertEqual(openai_to_messages(r)[2], OFF)


if __name__ == "__main__":
    unittest.main()
