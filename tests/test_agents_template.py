"""tests/test_agents_template.py - the shipped coding-agent AGENTS template (docs/agents/AGENTS.coding.md) keeps its
loop marker and the two rules measured on Terminal-Bench: detached background processes stopped by PID, and a stated
limit tested below, at and above it.

    python -m unittest tests.test_agents_template -v
"""
from __future__ import annotations

import unittest
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1] / "docs" / "agents" / "AGENTS.coding.md"


class AgentsTemplate(unittest.TestCase):
    def setUp(self):
        self.text = TEMPLATE.read_text(encoding="utf-8")

    def test_structure(self):
        for heading in ("## Loop (each step)", "## Tool rules", "## Before you finish", "## Final message"):
            self.assertIn(heading, self.text)

    def test_background_rule(self):
        self.assertIn("nohup CMD >LOG 2>&1 </dev/null &", self.text)
        self.assertIn("Stop them by PID", self.text)

    def test_limit_rule(self):
        self.assertIn("try below it, at it and above it", self.text)

    def test_plain_text(self):
        self.assertTrue(self.text.isascii())
        self.assertNotIn("owner", self.text.lower())


if __name__ == "__main__":
    unittest.main()
