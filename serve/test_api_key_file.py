"""serve/test_api_key_file.py - --api-key-file / "api_key_file": the key read from a file (kept out of the process list).

    python -m unittest serve.test_api_key_file -v
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import api_key_of, configured_api_key, main, read_api_key_file  # noqa: E402


class KeyFile(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def write(self, text, name="key"):
        p = Path(self.dir.name) / name
        p.write_text(text, encoding="utf-8")
        return str(p)

    def test_read_and_newline(self):
        self.assertEqual(read_api_key_file(self.write("sk-abc\n")), "sk-abc")
        self.assertEqual(read_api_key_file(self.write("  sk-abc\r\n\n")), "sk-abc")
        self.assertEqual(api_key_of(read_api_key_file(self.write("k1,k2\n"))), api_key_of("k1,k2"))

    def test_missing_and_empty_are_errors(self):
        with self.assertRaises(ValueError):
            read_api_key_file(os.path.join(self.dir.name, "nope"))
        with self.assertRaises(ValueError):
            read_api_key_file(self.write("\n \n"))

    def test_precedence(self):
        f = self.write("from-file\n")
        self.assertEqual(configured_api_key("cli", f, {"api_key_file": f, "api_key": "cfg"}), "cli")
        self.assertEqual(configured_api_key("", f, {"api_key": "cfg"}), "from-file")
        self.assertEqual(configured_api_key("", None, {"api_key_file": f, "api_key": "cfg"}), "from-file")
        self.assertEqual(configured_api_key("", None, {"api_key": "cfg"}), "cfg")
        self.assertEqual(configured_api_key("", None, {}), "")

    def test_cli_wins_without_reading_a_bad_file(self):
        self.assertEqual(configured_api_key("cli", os.path.join(self.dir.name, "nope"), {}), "cli")

    def test_main_stops_on_a_missing_file(self):
        err = io.StringIO()
        old = os.environ.pop("STRATA_API_KEY", None)
        try:
            with contextlib.redirect_stderr(err):
                rc = main(["--api-key-file", os.path.join(self.dir.name, "nope")])
        finally:
            if old is not None:
                os.environ["STRATA_API_KEY"] = old
        self.assertEqual(rc, 2)
        self.assertIn("API key file", err.getvalue())


if __name__ == "__main__":
    unittest.main()
