"""serve/test_default_effort.py - 0.4.8 server default reasoning effort (STRATA_DEFAULT_REASONING_EFFORT /
"default_reasoning_effort"): used only when a request names no effort, thinking block, template kwargs or budget.

    python -m unittest serve.test_default_effort -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import Service, resolve_default_effort, speedy_version_info  # noqa: E402


class Resolve(unittest.TestCase):
    def test_order_and_levels(self):
        self.assertEqual(resolve_default_effort({}, {}), ("medium", "default"))   # Speedy 0.5.0
        self.assertEqual(resolve_default_effort({"default_reasoning_effort": "medium"}, {}), ("medium", "config"))
        self.assertEqual(resolve_default_effort({"reasoning_effort": "low"}, {}), ("low", "config"))
        self.assertEqual(resolve_default_effort({"default_reasoning_effort": "high", "reasoning_effort": "low"}, {}),
                         ("high", "config"))
        self.assertEqual(resolve_default_effort({"default_reasoning_effort": "medium"},
                                                {"STRATA_DEFAULT_REASONING_EFFORT": "max"}), ("max", "env"))
        self.assertEqual(resolve_default_effort({"default_reasoning_effort": "medium"},
                                                {"STRATA_DEFAULT_REASONING_EFFORT": "off"}), ("none", "off"))
        self.assertEqual(resolve_default_effort({}, {"STRATA_DEFAULT_REASONING_EFFORT": " "}), ("medium", "default"))
        with self.assertRaises(SystemExit):
            resolve_default_effort({}, {"STRATA_DEFAULT_REASONING_EFFORT": "huge"})
        with self.assertRaises(SystemExit):
            resolve_default_effort({"default_reasoning_effort": "x"}, {})


def fake(effort):
    svc = Service.__new__(Service)
    svc.shared, svc.config_effort, svc.default_effort_source = {}, effort, "config"
    return svc


class Apply(unittest.TestCase):
    def test_fills_only_when_absent(self):
        svc = fake("medium")
        self.assertEqual(svc.with_shared({}, "openai")["reasoning_effort"], "medium")
        self.assertEqual(svc.with_shared({}, "anthropic")["output_config"], {"effort": "medium"})

    def test_client_wins(self):
        svc = fake("medium")
        for req in ({"reasoning_effort": "high"}, {"reasoning": {"effort": "low"}},
                    {"chat_template_kwargs": {"enable_thinking": False}},
                    {"chat_template_kwargs": {"reasoning_effort": "low"}}, {"reasoning_budget_tokens": 512},
                    {"thinking_budget_tokens": 0}):
            self.assertEqual(svc.with_shared(req, "openai").get("reasoning_effort"), req.get("reasoning_effort"), req)
        for req in ({"thinking": {"type": "enabled", "budget_tokens": 2048}}, {"output_config": {"effort": "high"}}):
            out = svc.with_shared(req, "anthropic")
            self.assertEqual((out.get("thinking"), out.get("output_config")), (req.get("thinking"), req.get("output_config")))

    def test_off_and_unset(self):
        self.assertEqual(fake("none").with_shared({}, "anthropic")["thinking"], {"type": "disabled"})
        self.assertEqual(fake(None).with_shared({"x": 1}, "openai"), {"x": 1})

    def test_version_info(self):
        svc = SimpleNamespace(config_effort="medium", default_effort_source="config", update_checker=None)
        self.assertEqual(speedy_version_info(svc)["default_reasoning_effort"], {"source": "config", "value": "medium"})
        svc = SimpleNamespace(config_effort="none", default_effort_source="off", update_checker=None)
        self.assertEqual(speedy_version_info(svc)["default_reasoning_effort"], {"source": "off", "value": "off"})


if __name__ == "__main__":
    unittest.main()
