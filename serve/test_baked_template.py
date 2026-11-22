"""serve/test_baked_template.py - #6481: a pack's baked chat template with the old `preserve_thinking is defined and`
predicate is normalized on load; a correct template (ours, our packs) is byte-identical after load.

    python -m unittest serve.test_baked_template -v
"""
from __future__ import annotations

import glob
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate, normalize_baked_chat_template  # noqa: E402

BROKEN_TEXT = "preserve_thinking is defined and preserve_thinking is true"


class BakedPreserveThinking(unittest.TestCase):
    _BROKEN = (
        "{%- set ns = namespace(last_query_index=-1) %}\n"
        "{%- for message in messages %}\n"
        "    {%- if message.role == \"user\" %}\n"
        "        {%- set ns.last_query_index = loop.index0 %}\n"
        "    {%- endif %}\n"
        "{%- endfor %}\n"
        "{%- for message in messages %}\n"
        "    {%- if message.role == \"assistant\" %}\n"
        "        {%- if (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index) %}\n"
        "            {{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' + message.content }}\n"
        "        {%- else %}\n"
        "            {{- '<|im_start|>assistant\\n' + message.content }}\n"
        "        {%- endif %}\n"
        "    {%- else %}\n"
        "        {{- '<|im_start|>' + message.role + '\\n' + message.content }}\n"
        "    {%- endif %}\n"
        "{%- endfor %}\n"
    )

    def test_unset_flag_keeps_the_empty_think_block(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat_template.jinja"
            path.write_text(self._BROKEN, encoding="utf-8")
            template = ChatTemplate(path)
        self.assertNotIn(BROKEN_TEXT, template.source)
        rendered = template.render([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": "again"},
        ], add_generation_prompt=False)
        self.assertIn("<think>", rendered)
        self.assertIn("OK", rendered)

    def test_a_correct_template_is_left_alone(self):
        path = Path(__file__).with_name("chat_template.jinja")
        self.assertEqual(ChatTemplate(path).source, path.read_text(encoding="utf-8"))
        self.assertNotIn(BROKEN_TEXT, ChatTemplate(path).source)

    def test_other_text_untouched(self):
        self.assertEqual(normalize_baked_chat_template("abc"), "abc")

    def test_local_packs_byte_identical(self):
        # packs installed on this machine (skipped elsewhere): already fixed, so loading must not change a byte
        packs = sorted(glob.glob("/srv/models/*/packs/*/tokenizer/chat_template.jinja"))
        if not packs:
            self.skipTest("no local packs")
        for p in packs:
            src = Path(p).read_text(encoding="utf-8")
            self.assertEqual(normalize_baked_chat_template(src), src, p)


if __name__ == "__main__":
    unittest.main()
