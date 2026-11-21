"""#6285 / upstream Strata #6315: repaired arguments stream as the final call carries them (Speedy's streaming scanner
holds what follows a slip and sends the repaired members at the call's close)."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serve.frontend import OutputParser, repair_arguments  # noqa: E402

RC = {"name": "run_commands", "parameters": {"type": "object", "properties": {
    "commands": {"type": "array", "items": {"type": "string"}}, "cwd": {"type": "string"}}, "required": ["commands"]}}
CASES = {
    "plural slip after a sent string": ("<parameter=cwd>/x</parameter><parameter=command>ls -la</parameter>",
                                        {"cwd": "/x", "commands": ["ls -la"]}),
    "objects for strings": ('<parameter=cwd>/x</parameter><parameter=commands>[{"cmd":"ls"}]</parameter>',
                            {"cwd": "/x", "commands": ["ls"]}),
    "arguments member: own write wins": ('<parameter=cwd>/x</parameter><parameter=arguments>{"commands":["ls"],"cwd":"/y"}'
                                         "</parameter>", {"cwd": "/x", "commands": ["ls"]}),
    "nothing to repair": ('<parameter=commands>["a","b"]</parameter><parameter=cwd>/x</parameter>',
                          {"commands": ["a", "b"], "cwd": "/x"}),
}


class RepairStream(unittest.TestCase):
    def test_streamed_json_equals_final_call(self):
        for label, (params, expected) in CASES.items():
            text = f"<tool_call><function=run_commands>{params}</function></tool_call>"
            for size in (1, 3, 7, len(text)):
                with self.subTest(label=label, size=size):
                    p = OutputParser(thinking=False, tools=[RC], stream_tools=True)
                    ev = []
                    for i in range(0, len(text), size):
                        ev += p.feed(text[i:i + size])
                    ev += p.finish()
                    calls = [e.call.arguments for e in ev if e.kind == "tool_call"]
                    self.assertEqual(calls, [expected])
                    self.assertEqual(json.loads("".join(e.text for e in ev if e.kind == "tool_args")), expected)

    def test_no_schema_or_no_props_left_alone(self):
        self.assertEqual(repair_arguments({"command": "ls"}, None), {"command": "ls"})
        self.assertEqual(repair_arguments({"command": "ls"}, {"name": "x", "parameters": {}}), {"command": "ls"})


if __name__ == "__main__":
    unittest.main()
