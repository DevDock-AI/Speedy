"""serve/test_tool_name_case.py - a declared tool written in another case is that tool, when only one declared tool fits.

Seen in a real gate: OpenCode declared `bash`; the model wrote `<function=Bash>`.  The block went back as text with finish "stop" and the
agent's turn ended.  The name now resolves exact first, then case-insensitively against the declared tools (and their aliases); an unknown
name stays text under the default "undeclared_tool_calls": "text", and a name two declared tools fit is not guessed.

    python -m unittest serve.test_tool_name_case -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import OutputParser  # noqa: E402


def tool(name, prop="command"):
    return {"name": name, "parameters": {"properties": {prop: {"type": "string"}}}}


def call(name, opener="<tool_call>\n"):
    return f"{opener}<function={name}>\n<parameter=command>\nls -la && cat ISSUE.md\n</parameter>\n</function>\n</tool_call>"


def run(text, tools, step=5, stream_tools=True):
    p = OutputParser(thinking=False, tools=tools, stream_tools=stream_tools, recover=True, declared_only=True)
    evs = []
    for i in range(0, len(text), step):
        evs += p.feed(text[i:i + step])
    evs += p.finish("stop")
    return evs


def calls(evs):
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


def text(evs):
    return "".join(e.text or "" for e in evs if e.kind == "content")


OPENCODE = [tool("bash"), tool("read", "filePath"), tool("edit", "filePath")]


class CaseInsensitiveName(unittest.TestCase):
    def test_capitalised_bash_is_bash(self):
        for step in (1, 5, 1000):
            for stream in (True, False):
                evs = run(call("Bash"), OPENCODE, step, stream)
                self.assertEqual(calls(evs), [("bash", {"command": "ls -la && cat ISSUE.md"})], (step, stream))
                self.assertNotIn("<function", text(evs))
                starts = [e for e in evs if e.kind == "tool_start"]
                if stream:
                    self.assertEqual([e.call.name if getattr(e, "call", None) else e.name for e in starts][:1], ["bash"])

    def test_upper_case_and_misspelt_opener(self):
        self.assertEqual(calls(run(call("BASH"), OPENCODE))[0][0], "bash")
        self.assertEqual(calls(run(call("Bash", "<toolcall>\n"), OPENCODE))[0][0], "bash")       # #5276
        self.assertEqual(calls(run(call("tools Bash", "<tool_call>\n"), OPENCODE))[0][0], "bash")  # #5409

    def test_exact_name_wins(self):
        tools = [tool("bash"), tool("Bash")]
        self.assertEqual(calls(run(call("Bash"), tools))[0][0], "Bash")
        self.assertEqual(calls(run(call("bash"), tools))[0][0], "bash")

    def test_unknown_name_stays_text(self):
        evs = run(call("Shell"), OPENCODE)
        self.assertEqual(calls(evs), [])
        self.assertIn("<function=Shell>", text(evs))

    def test_two_matches_stay_text(self):
        tools = [tool("bash"), tool("BASH"), tool("read", "filePath")]
        evs = run(call("Bash"), tools)
        self.assertEqual(calls(evs), [])
        self.assertIn("<function=Bash>", text(evs))


if __name__ == "__main__":
    unittest.main()
