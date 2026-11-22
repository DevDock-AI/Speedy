"""serve/test_effort_floor.py - reasoning effort floor (STRATA_REASONING_EFFORT_FLOOR / "reasoning_effort_floor",
STRATA_REASONING_EFFORT_FLOOR_CTX_TOKENS / "reasoning_effort_floor_ctx_tokens"): tool requests (and, opt-in, long
prompts) whose thinking level is below the floor get the floor.

    python -m unittest serve.test_effort_floor -v
"""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import Service, client_thinking_off, effort_below_floor, resolve_effort_floor  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
MSG = [{"role": "user", "content": "hi"}]


def fake(floor="low", ctx=0, effort=None):
    svc = Service.__new__(Service)
    svc.shared, svc.config_effort, svc.default_effort_source = {}, effort, "config"
    svc.effort_floor, svc.effort_floor_ctx_tokens = floor, ctx
    return svc


def run(svc, req, api="openai"):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        r = svc.with_shared(req, api)
    return r, out.getvalue()


class Resolve(unittest.TestCase):
    def test_order(self):
        self.assertEqual(resolve_effort_floor({}, {}), ("low", 0, "default"))
        self.assertEqual(resolve_effort_floor({"reasoning_effort_floor": "medium"}, {}), ("medium", 0, "config"))
        self.assertEqual(resolve_effort_floor({"reasoning_effort_floor": "off"}, {}), (None, 0, "off"))
        self.assertEqual(resolve_effort_floor({"reasoning_effort_floor": "medium"},
                                              {"STRATA_REASONING_EFFORT_FLOOR": "high"}), ("high", 0, "env"))
        # the config's env block counts for the server process too, the process environment first
        cfg = {"reasoning_effort_floor": "medium", "env": {"STRATA_REASONING_EFFORT_FLOOR": "none",
                                                            "STRATA_REASONING_EFFORT_FLOOR_CTX_TOKENS": "5000"}}
        self.assertEqual(resolve_effort_floor(cfg, {}), (None, 5000, "off"))
        self.assertEqual(resolve_effort_floor(cfg, {"STRATA_REASONING_EFFORT_FLOOR": "low"}), ("low", 5000, "env"))
        self.assertEqual(resolve_effort_floor({"reasoning_effort_floor_ctx_tokens": 32768}, {}), ("low", 32768, "default"))
        for bad in ({"reasoning_effort_floor": "max"}, {"reasoning_effort_floor": "huge"},
                    {"reasoning_effort_floor_ctx_tokens": "lots"}):
            with self.assertRaises(SystemExit):
                resolve_effort_floor(bad, {})

    def test_client_thinking_off(self):
        self.assertTrue(client_thinking_off({"reasoning_effort": "none"}, "openai"))
        self.assertTrue(client_thinking_off({"chat_template_kwargs": {"enable_thinking": False}}, "openai"))
        self.assertTrue(client_thinking_off({"reasoning": {"effort": "none"}}, "responses"))
        self.assertTrue(client_thinking_off({"thinking": {"type": "disabled"}}, "anthropic"))
        self.assertFalse(client_thinking_off({}, "openai"))
        self.assertFalse(client_thinking_off({"reasoning_effort": "low"}, "openai"))
        self.assertFalse(client_thinking_off({"reasoning_effort": "low",
                                              "chat_template_kwargs": {"enable_thinking": False}}, "openai"))
        self.assertFalse(client_thinking_off({"thinking": {"type": "enabled"}}, "anthropic"))

    def test_order_of_levels(self):
        self.assertTrue(effort_below_floor("none", "low"))
        self.assertTrue(effort_below_floor("off", "low"))
        self.assertTrue(effort_below_floor("low", "medium"))
        self.assertFalse(effort_below_floor("low", "low"))
        self.assertFalse(effort_below_floor("high", "medium"))
        self.assertFalse(effort_below_floor(None, "low"))
        self.assertFalse(effort_below_floor("none", None))


class Apply(unittest.TestCase):
    def test_client_off_respected(self):
        # #30: a client that turns thinking off itself keeps it off, tools or not
        for req in ({"messages": MSG, "tools": TOOLS, "reasoning_effort": "none"},
                    {"messages": MSG, "tools": TOOLS, "reasoning": {"effort": "none"}},
                    {"messages": MSG, "tools": TOOLS, "reasoning": {"enabled": False}},
                    {"messages": MSG, "tools": TOOLS, "chat_template_kwargs": {"enable_thinking": False}},
                    {"messages": MSG, "tools": TOOLS, "chat_template_kwargs": {"reasoning_effort": "off"}}):
            r, log = run(fake("medium"), req)
            self.assertEqual(r.get("reasoning_effort"), req.get("reasoning_effort"), req)
            self.assertEqual(r.get("chat_template_kwargs"), req.get("chat_template_kwargs"), req)
            self.assertEqual(log, "", req)

    def test_client_level_below_floor_raised(self):
        r, log = run(fake("medium"), {"messages": MSG, "tools": TOOLS, "reasoning_effort": "low"})
        self.assertEqual(r["reasoning_effort"], "medium")
        self.assertEqual(log.count("effort floor"), 1)

    def test_left_alone(self):
        svc = fake()
        for req in ({"messages": MSG, "reasoning_effort": "none"},                       # no tools
                    {"messages": MSG, "tools": TOOLS, "reasoning_effort": "high"},       # above the floor
                    {"messages": MSG, "tools": TOOLS},                                   # no level named
                    {"messages": MSG, "tools": TOOLS, "tool_choice": "none", "reasoning_effort": "none"},
                    {"messages": MSG, "tools": TOOLS, "reasoning_effort": "none", "reasoning_budget_tokens": 0}):
            r, log = run(svc, req)
            self.assertEqual(r.get("reasoning_effort"), req.get("reasoning_effort"), req)
            self.assertEqual(log, "")
        r, _ = run(fake(None), {"messages": MSG, "tools": TOOLS, "reasoning_effort": "none"})
        self.assertEqual(r["reasoning_effort"], "none")

    def test_server_default_none_raised(self):
        r, _ = run(fake(effort="none"), {"messages": MSG, "tools": TOOLS})
        self.assertEqual(r["reasoning_effort"], "low")

    def test_anthropic(self):
        r, _ = run(fake(), {"messages": MSG, "tools": [{"name": "f", "input_schema": {}}],
                            "thinking": {"type": "disabled"}}, "anthropic")
        self.assertEqual(r["thinking"], {"type": "disabled"})                         # #30: client off kept
        self.assertNotIn("output_config", r)
        r, _ = run(fake("medium"), {"messages": MSG, "tools": [{"name": "f", "input_schema": {}}],
                                    "output_config": {"effort": "low"}}, "anthropic")
        self.assertEqual(r["output_config"], {"effort": "medium"})
        r, _ = run(fake(), {"messages": MSG, "thinking": {"type": "disabled"}}, "anthropic")
        self.assertEqual(r["thinking"], {"type": "disabled"})

    def test_ctx_tokens(self):
        long = [{"role": "user", "content": "x" * 40000}]                               # ~10K tokens
        r, _ = run(fake(ctx=0), {"messages": long, "reasoning_effort": "none"})
        self.assertEqual(r["reasoning_effort"], "none")
        r, _ = run(fake(ctx=5000), {"messages": long, "reasoning_effort": "none"})   # #30: client off kept
        self.assertEqual(r["reasoning_effort"], "none")
        r, log = run(fake(ctx=5000, effort="none"), {"messages": long})                # server default none raised
        self.assertEqual(r["reasoning_effort"], "low")
        self.assertIn("long prompt", log)
        r, _ = run(fake(ctx=50000), {"messages": long, "reasoning_effort": "none"})
        self.assertEqual(r["reasoning_effort"], "none")


if __name__ == "__main__":
    unittest.main()
