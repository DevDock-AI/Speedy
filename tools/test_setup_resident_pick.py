"""#23: on a card whose VRAM holds a better quant fully resident (experts + KV + draft + reserve), setup's --yes picks
the best one that fits (UD-Q4_K_XL, then UD-IQ4_XS) and says why; smaller cards and an explicit choice are unchanged.

    python tools/test_setup_resident_pick.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "serve"))
import setup  # noqa: E402
from test_setup_golden import card, install  # noqa: E402

FAMS = setup.menu_families(False)
PRO6000 = card(0, "NVIDIA RTX PRO 6000 Blackwell Server Edition", 97887 / 1024, "120")


class ResidentPick(unittest.TestCase):
    def test_96gb_sm120_picks_q4_k_xl(self):
        fam, model, why = setup.resident_pick(PRO6000["vram_gb"], 176.0, FAMS)
        self.assertEqual((fam, model), ("unsloth", "UD-Q4_K_XL"))
        self.assertIn("IQ3_S", why)

    def test_80gb_falls_back_to_iq4_xs(self):
        self.assertEqual(setup.resident_pick(80.0, 128.0, FAMS)[1], "UD-IQ4_XS")

    def test_small_cards_unchanged(self):
        for vram in (24.0, 31.8, 48.0):
            self.assertIsNone(setup.resident_pick(vram, 128.0, FAMS))

    def test_split_and_no_unsloth_unchanged(self):
        self.assertIsNone(setup.resident_pick(PRO6000["vram_gb"], 176.0, FAMS, multi=[0, 1]))
        self.assertIsNone(setup.resident_pick(PRO6000["vram_gb"], 176.0, ["qwen", "swift"]))

    def test_yes_on_96gb_installs_q4_k_xl(self):
        code, text, cfg, _ = install(175.0, [PRO6000], ["--yes"])
        self.assertIn("[ok] size: UD-Q4_K_XL", text)
        self.assertIn("holds all 77 GB of UD-Q4_K_XL's experts", text)

    def test_yes_on_24gb_unchanged(self):
        code, text, cfg, _ = install(127.8, [card(0, "NVIDIA GeForce RTX 4090", 24.0, "89")], ["--yes"])
        self.assertNotIn("holds all", text)
        self.assertIn("[ok] size: IQ3_S", text)

    def test_explicit_choice_kept(self):
        for argv in (["--yes", "--model", "IQ3_S"], ["--yes", "--family", "qwen"],
                     ["--yes", "--family", "unsloth", "--model", "UD-IQ4_XS"]):
            code, text, cfg, _ = install(175.0, [PRO6000], argv)
            self.assertNotIn("holds all", text, argv)
            want = "UD-IQ4_XS" if "UD-IQ4_XS" in argv else "IQ3_S"
            self.assertIn("[ok] size: " + want, text, argv)


if __name__ == "__main__":
    unittest.main()
