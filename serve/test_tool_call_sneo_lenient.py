"""A missing </parameter> before the next declared parameter, and quoted numbers/booleans coerced by the schema.

    python -m unittest serve.test_tool_call_sneo_lenient -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import OutputParser, parse_tool_call  # noqa: E402

WRITE = {"name": "write", "parameters": {"properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}
READ = {"name": "read", "parameters": {"properties": {"path": {"type": "string"}, "limit": {"type": "integer"},
                                                       "ratio": {"type": "number"}, "all": {"type": "boolean"}}}}


def calls(text, step=5):
    p = OutputParser(thinking=False, tools=[WRITE, READ], stream_tools=True, recover=True)
    evs = []
    for i in range(0, len(text), step):
        evs += p.feed(text[i:i + step])
    evs += p.finish("stop")
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


class Lenient(unittest.TestCase):
    def test_missing_param_end(self):
        body = "<function=write>\n<parameter=path>\na.py\n<parameter=content>\nx = 1\n</parameter>\n</function>"
        self.assertEqual(parse_tool_call(body, WRITE).arguments, {"path": "a.py", "content": "x = 1"})

    def test_missing_param_end_parser(self):
        t = "<tool_call>\n<function=write>\n<parameter=path>\na.py\n<parameter=content>\nx = 1\n</parameter>\n</function>\n</tool_call>"
        self.assertEqual(calls(t), [("write", {"path": "a.py", "content": "x = 1"})])

    def test_quoted_format_kept(self):
        # a value holding its own </parameter> (a file that documents the format) is not cut
        v = "doc:\n<parameter=content>\nq\n</parameter>\nend"
        body = f"<function=write>\n<parameter=path>\n{v}\n</parameter>\n<parameter=content>\nz\n</parameter>\n</function>"
        self.assertEqual(parse_tool_call(body, WRITE).arguments["path"], v)

    def test_undeclared_tag_not_cut(self):
        body = "<function=write>\n<parameter=content>\nsee <parameter=foo>\nbar\n</parameter>\n</function>"
        self.assertEqual(parse_tool_call(body, WRITE).arguments, {"content": "see <parameter=foo>\nbar"})

    def test_quoted_scalars_coerced(self):
        body = ('<function=read>\n<parameter=path>\n"7"\n</parameter>\n<parameter=limit>\n"40"\n</parameter>\n'
                '<parameter=ratio>\n"0.5"\n</parameter>\n<parameter=all>\n"True"\n</parameter>\n</function>')
        self.assertEqual(parse_tool_call(body, READ).arguments, {"path": '"7"', "limit": 40, "ratio": 0.5, "all": True})

    def test_bare_function(self):
        t = "Reading it.\n<function=read>\n<parameter=path>\na.py\n</parameter>\n</function>"
        self.assertEqual(calls(t), [("read", {"path": "a.py"})])


if __name__ == "__main__":
    unittest.main()
