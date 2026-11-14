"""Strata #5921: serve arms faulthandler (STRATA_NO_FAULTHANDLER=1 off); a native crash prints thread stacks."""
import faulthandler
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class FaultHandler(unittest.TestCase):
    def test_arm_and_opt_out(self):
        from serve import server
        was = faulthandler.is_enabled()
        try:
            self.assertFalse(server.arm_faulthandler({"STRATA_NO_FAULTHANDLER": "1"}))
            with tempfile.TemporaryFile("w+") as f:
                self.assertTrue(server.arm_faulthandler({}, stream=f))
                self.assertTrue(faulthandler.is_enabled())
        finally:
            faulthandler.disable()
            if was:
                faulthandler.enable()

    def test_crash_prints_stack(self):
        code = ("import sys; sys.path.insert(0, %r); from serve import server; server.arm_faulthandler({});"
                "import ctypes; ctypes.string_at(0)" % str(ROOT))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Fatal Python error", r.stderr)
        self.assertIn("most recent call first", r.stderr)


if __name__ == "__main__":
    unittest.main()
