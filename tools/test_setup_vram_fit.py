"""#25: setup never writes a Flash-Next config whose start leaves the expert cache under its minimum (a 262144 context,
8-bit KV and the draft layer on a 10 GB RTX 3080: "no VRAM is left for the expert cache ... about 951 MiB more than
this card has free"): the context steps down, then the KV cache to 4 bits; the hint covers 10-12 GB cards; and the
server's start error says which context fits when the engine's log shows that failure.

    python tools/test_setup_vram_fit.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "serve"))
import setup  # noqa: E402
from test_setup_golden import card, install  # noqa: E402
from server import expert_cache_fail_hint, start_failure_hint  # noqa: E402

ISSUE25 = ("strata generate: no VRAM is left for the expert cache: it needs at least 296 slots (657 MiB), about 951 MiB "
           "more than this card has free. To make room: a smaller --max-context (the session, mostly its KV cache, "
           "takes 3741 MiB at 262144 tokens), --kv q4_0, the English draft subset (setup --draft-vocab en; the draft "
           "head takes 184 MiB now), images on the CPU, or close other programs that use the GPU\n")


def arg(cfg, flag):
    a = cfg["args"]
    return a[a.index(flag) + 1] if flag in a else None


class Estimate(unittest.TestCase):
    def test_issue25_numbers(self):
        # the engine's own figures: 951 MiB short at 262144 / int8 / draft layer on 10 GB
        self.assertAlmostEqual(setup.fit_slack_mib(10.0, 262144, "int8"), -951, delta=1)
        self.assertAlmostEqual(setup.fit_session_mib(262144, "int8"), 3741, delta=1)
        self.assertEqual(setup.fit_plan(10.0, 262144, "int8")[:2], (131072, "int8"))

    def test_context_first_then_kv(self):
        self.assertEqual(setup.fit_plan(10.0, 131072, "int8")[:2], (131072, "int8"))     # fits as asked
        self.assertEqual(setup.fit_plan(12.0, 262144, "int8")[:2], (262144, "int8"))
        self.assertEqual(setup.fit_plan(8.0, 32768, "int8")[:2], (32768, "int8"))
        # a big reserve on a 10 GB card: 32768 int8 does not fit, 4 bits does
        self.assertEqual(setup.fit_plan(10.0, 262144, "int8", reserve_mib=2800)[:2], (32768, "q4_0"))
        self.assertIsNone(setup.fit_plan(6.0, 262144, "int8"))

    def test_kv_streaming_keeps_only_the_resident_window(self):
        self.assertEqual(setup.fit_plan(10.0, 262144, "int8", resident=32768)[:2], (262144, "int8"))

    def test_no_draft_layer_frees_its_vram(self):
        self.assertGreater(setup.fit_slack_mib(10.0, 262144, "int8", mtp=False),
                           setup.fit_slack_mib(10.0, 262144, "int8") + 1000)

    def test_hint_for_10_to_12_gb_cards(self):
        self.assertTrue(any("--context 65536" in x for x in setup.fit_card_note(10.0, 131072, "int8")))
        self.assertTrue(setup.fit_card_note(12.0, 32768, "int8"))
        self.assertEqual(setup.fit_card_note(16.0, 32768, "int8"), [])
        self.assertEqual(setup.fit_card_note(6.0, 32768, "int8"), [])
        self.assertEqual(setup.fit_card_note(8.0, 32768, "int8"), [])


class Install(unittest.TestCase):
    def run_setup(self, vram, ctx, extra=()):
        argv = ["--family", "qwen", "--no-start", "--model", "IQ3_XXS", "--context", str(ctx), *extra]
        return install(62.0, [card(0, "NVIDIA GeForce RTX 3080", vram, "86")], argv, answers={})

    def test_issue25_config_steps_down_to_a_start_that_fits(self):
        code, out, cfg, _ = self.run_setup(10.0, 262144, ("--kv", "int8"))
        self.assertEqual(code, 0, out)
        self.assertEqual(arg(cfg, "--max-context"), "131072", out)
        self.assertEqual(arg(cfg, "--kv"), "int8")
        self.assertIn("context 262144 -> 131072", out)
        self.assertIn("#25", out)
        if "--kv-resident" not in cfg["args"]:
            slack = setup.fit_slack_mib(10.0, int(arg(cfg, "--max-context")), arg(cfg, "--kv"))
            self.assertGreaterEqual(slack, setup.FIT_MARGIN_MIB)

    def test_a_fitting_config_is_left_alone(self):
        code, out, cfg, _ = self.run_setup(10.0, 32768)
        self.assertEqual(code, 0, out)
        self.assertEqual(arg(cfg, "--max-context"), "32768")
        self.assertNotIn("would not start", out)

    def test_a_16_gb_card_is_not_touched(self):
        code, out, cfg, _ = self.run_setup(16.0, 262144)
        self.assertEqual(code, 0, out)
        self.assertEqual(arg(cfg, "--max-context"), "262144")


class ServerHint(unittest.TestCase):
    def test_issue25_log_names_the_context_that_fits(self):
        h = expert_cache_fail_hint(ISSUE25)
        self.assertIn("context is too large for this card", h)
        self.assertIn("--context 131072", h)
        self.assertIn("951 MiB short", h)

    def test_through_start_failure_hint(self):
        p = Path(tempfile.mkdtemp()) / "strata.log"
        p.write_text("earlier start\n" + ISSUE25, encoding="utf-8")
        self.assertIn("--context 131072", start_failure_hint(str(p), len("earlier start\n")))
        self.assertEqual(start_failure_hint(str(p), p.stat().st_size), "")

    def test_without_figures_still_says_what_to_do(self):
        self.assertIn("too large", expert_cache_fail_hint("strata generate: no VRAM is left for the expert cache\n"))
        self.assertEqual(expert_cache_fail_hint("some other failure\n"), "")


if __name__ == "__main__":
    unittest.main()
