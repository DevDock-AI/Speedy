"""serve/test_malloc_arena.py - MALLOC_ARENA_MAX for the engine from the config or STRATA_MALLOC_ARENA_MAX; default off.

    python -m unittest serve.test_malloc_arena -v
"""
from __future__ import annotations

import sys
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import server  # noqa: E402


class MallocArena(unittest.TestCase):
    def test_default_is_off(self):
        self.assertEqual(server.malloc_arena_max({}, {}), 0)
        with unittest.mock.patch.dict(server.os.environ, {}, clear=True):
            self.assertNotIn("MALLOC_ARENA_MAX", server.child_env({}))

    def test_config_and_env(self):
        self.assertEqual(server.malloc_arena_max({"malloc_arena_max": 2}, {}), 2)
        self.assertEqual(server.malloc_arena_max({"malloc_arena_max": 2}, {"STRATA_MALLOC_ARENA_MAX": "0"}), 0)
        self.assertEqual(server.malloc_arena_max({}, {"STRATA_MALLOC_ARENA_MAX": "x"}), 0)

    @unittest.skipUnless(sys.platform.startswith("linux"), "glibc only")
    def test_engine_env(self):
        with unittest.mock.patch.dict(server.os.environ, {}, clear=True):
            self.assertEqual(server.child_env({"malloc_arena_max": 2})["MALLOC_ARENA_MAX"], "2")
        with unittest.mock.patch.dict(server.os.environ, {"MALLOC_ARENA_MAX": "4"}, clear=True):
            self.assertEqual(server.child_env({"malloc_arena_max": 2})["MALLOC_ARENA_MAX"], "4")


if __name__ == "__main__":
    unittest.main()
