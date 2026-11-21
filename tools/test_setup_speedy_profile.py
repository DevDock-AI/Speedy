"""Speedy 0.5.0: setup's tuned 1-GPU profile and the MTP drafter choice.  python -m unittest tools.test_setup_speedy_profile"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from serve import server  # noqa: E402


class Profile(unittest.TestCase):
    def test_one_nvidia_card_ud_file(self):
        env, args, prefill = setup.speedy_profile("tuned", "unsloth", "UD-IQ4_XS", False, False,
                                                  {"arch": "86", "vram_gb": 24.0})
        self.assertEqual(env["STRATA_ROUTE_PRIOR"], "1.5")
        self.assertEqual(env["STRATA_TOOLCALL_REPAIR"], "1")
        self.assertIn("--adapt-async", args)
        self.assertEqual(prefill, "auto:16384")

    def test_rtx50_16gb_keeps_adapt_fetch(self):
        _, args, _ = setup.speedy_profile("tuned", "unsloth", "UD-Q4_K_XL", False, False, {"arch": "120", "vram_gb": 16.0})
        self.assertNotIn("--adapt-async", args)
        self.assertIn("--adapt-swaps", args)

    def test_multi_hip_other_family_get_tool_env_only(self):
        for fam, model, hip, multi in (("unsloth", "UD-IQ4_XS", False, True), ("unsloth", "UD-IQ4_XS", True, False),
                                       ("qwen", "Q2_0", False, False)):
            env, args, prefill = setup.speedy_profile("tuned", fam, model, hip, multi)
            self.assertEqual(env, setup.SPEEDY_TOOL_ENV)
            self.assertEqual((args, prefill), ([], None))

    def test_upstream_profile_writes_nothing(self):
        self.assertEqual(setup.speedy_profile("upstream", "unsloth", "UD-IQ4_XS", False, False), ({}, [], None))


class MtpDrafter(unittest.TestCase):
    def test_stock_switch(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            stock = Path(d) / "rt"
            stock.mkdir()
            cfg = {"args": ["--mtp", str(stock) + "-tuned", "--spec", "4"], "mtp_drafter": "stock"}
            server.choose_mtp_drafter(cfg, env={})
            self.assertEqual(cfg["args"][1], str(stock))
            cfg = {"args": ["--mtp", str(stock) + "-tuned"]}
            server.choose_mtp_drafter(cfg, env={})                   # default: the tuned one is kept
            self.assertTrue(cfg["args"][1].endswith("-tuned"))
            server.choose_mtp_drafter(cfg, env={"STRATA_MTP_DRAFTER": "stock"})
            self.assertEqual(cfg["args"][1], str(stock))

    def test_nudge_default_and_off(self):
        self.assertEqual(server.resolve_reasoning_nudge({}, env={}), [0.6, 0.8])
        self.assertIsNone(server.resolve_reasoning_nudge({"reasoning_nudge": False}, env={}))
        self.assertIsNone(server.resolve_reasoning_nudge({}, env={"STRATA_REASONING_NUDGE": "off"}))
        self.assertEqual(server.resolve_reasoning_nudge({"reasoning_nudge": "0.5"}, env={}), [0.5])

    def test_effort_default_medium(self):
        self.assertEqual(server.resolve_default_effort({}, env={}), ("medium", "default"))
        self.assertEqual(server.resolve_default_effort({"reasoning_effort": "off"}, env={})[0], "none")


if __name__ == "__main__":
    unittest.main()
