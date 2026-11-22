"""Repro: a non-string parameter written twice, the second opener without its `>`
(`replace_all = "false\\n</parameter>\\n<parameter=replace_all\\nfalse"`): the call must still come out as a call, the
first value kept, in streamed and whole parsing: python -m unittest serve.test_dup_param_reopen."""
import unittest

from serve.frontend import OutputParser, parse_tool_call

S, B = {"type": "string"}, {"type": "boolean"}
TOOLS = [{"name": "edit", "parameters": {"type": "object", "properties": {
    "path": S, "old_string": S, "new_string": S, "replace_all": B}}}]
HEAD = ("<tool_call>\n<function=edit>\n<parameter=path>\na.py\n</parameter>\n<parameter=old_string>\nx = 1\n"
        "</parameter>\n<parameter=new_string>\nx = 2\n</parameter>\n")
TAIL = "</function>\n</tool_call>"
WANT = [("edit", {"path": "a.py", "old_string": "x = 1", "new_string": "x = 2", "replace_all": False})]


def parse(text, stream):
    p = OutputParser(thinking=False, tools=TOOLS, stream_tools=stream, recover=False, declared_only=True)
    evs = []
    if stream:
        for ch in text:
            evs += p.feed(ch)
    else:
        evs += p.feed(text)
    evs += p.finish("stop")
    return ([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
            "".join(e.text or "" for e in evs if e.kind == "content"))


class DuplicateParam(unittest.TestCase):
    def check(self, body):
        for stream in (False, True):
            calls, content = parse(HEAD + body + TAIL, stream)
            self.assertEqual(calls, WANT, (stream, content))
            self.assertNotIn("<parameter", content)

    def test_second_opener_without_gt(self):
        self.check("<parameter=replace_all>\nfalse\n</parameter>\n<parameter=replace_all\nfalse\n</parameter>\n")

    def test_well_formed_duplicate(self):
        self.check("<parameter=replace_all>\nfalse\n</parameter>\n<parameter=replace_all>\nfalse\n</parameter>\n")

    def test_only_opener_without_gt(self):
        self.check("<parameter=replace_all\nfalse\n</parameter>\n")

    def test_first_value_is_kept(self):
        call = parse_tool_call("<function=edit>\n<parameter=replace_all>\nfalse\n</parameter>\n"
                               "<parameter=replace_all\ntrue\n</parameter>\n</function>", TOOLS[0])
        self.assertEqual(call.arguments, {"replace_all": False})

    def test_a_string_value_quoting_the_broken_opener_is_kept(self):
        doc = "see <parameter=replace_all\nfalse\n</parameter> in the log"
        call = parse_tool_call("<function=edit>\n<parameter=new_string>\n" + doc + "\n</parameter>\n</function>",
                               TOOLS[0])
        self.assertEqual(call.arguments, {"new_string": doc})


if __name__ == "__main__":
    unittest.main()
