"""Speedy 0.5.2: an update brings an earlier install's config to the 0.5 settings.
python -m unittest tools.test_setup_update_052"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

GPU = {"arch": "86", "vram_gb": 24.0}


class Upgrade052(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.rt = self.d / "mtp" / "rt"
        self.rt.mkdir(parents=True)
        for n in ("dense.bin", "dense.txt", "experts.bin", "draft_vocab.bin"):
            (self.rt / n).write_bytes(b"x")
        self.tuned = self.d / "mtp" / "rt-tuned"
        self.calls = 0

    def tearDown(self):
        self.tmp.cleanup()

    def fake_tuned(self, ok=True):
        def f(rt, url=None):
            self.calls += 1
            if not ok:
                return None
            self.tuned.mkdir(exist_ok=True)
            return self.tuned
        return f

    def old_cfg(self, **kw):
        cfg = {"exe": "x", "model_name": "qwen3.8-flash-next-unsloth-ud-iq4_xs",
               "args": ["--pack", "p", "--prefill", "auto", "--mtp", str(self.rt), "--spec", "4"],
               "env": {"STRATA_HC_Q8": "1"}, "reasoning_budget_tokens": 16384}
        cfg.update(kw)
        return cfg

    def run_up(self, cfg, ok=True, a=None):
        p = self.d / "strata-x.json"
        p.write_text(json.dumps(cfg))
        with mock.patch.object(setup, "tuned_mtp_dir", self.fake_tuned(ok)):
            ch = setup.upgrade_to_052(p, cfg, a, {}, GPU)
        return p, ch

    def test_old_04_config_upgraded(self):
        p, ch = self.run_up(self.old_cfg())
        cfg = json.loads(p.read_text())
        self.assertEqual(cfg["args"][cfg["args"].index("--mtp") + 1], str(self.tuned))
        self.assertTrue((self.tuned / "draft_vocab.bin").exists())
        self.assertNotIn("reasoning_budget_tokens", cfg)        # the server default 32768, as a fresh config
        self.assertEqual(cfg["env"]["STRATA_ROUTE_PRIOR"], "1.5")
        self.assertEqual(cfg["env"]["STRATA_TOOLCALL_CONSTRAIN"], "1")
        self.assertEqual(cfg["env"]["STRATA_HC_Q8"], "1")
        self.assertIn("--adapt-swaps", cfg["args"])
        self.assertEqual(cfg["args"][cfg["args"].index("--prefill") + 1], "auto:16384")
        self.assertEqual(cfg["speedy_profile"], "tuned")
        self.assertTrue((self.d / "strata-x.json.bak-0.5.2").exists())
        self.assertEqual(len(ch), 3)

    def test_idempotent(self):
        p, _ = self.run_up(self.old_cfg())
        cfg = json.loads(p.read_text())
        before = p.read_text()
        with mock.patch.object(setup, "tuned_mtp_dir", self.fake_tuned()):
            self.assertEqual(setup.upgrade_to_052(p, cfg, None, {}, GPU), [])
        self.assertEqual(p.read_text(), before)

    def test_opt_out_kept(self):
        for kw in ({"mtp_drafter": "stock"}, {"env": {"STRATA_MTP_DRAFTER": "stock"}}):
            _, ch = self.run_up(self.old_cfg(**kw))
            self.assertEqual(self.calls, 0)
            self.assertFalse(any("drafter" in c for c in ch))
        _, ch = self.run_up(self.old_cfg(), a=mock.Mock(mtp_drafter="stock", speedy_profile="tuned"))
        self.assertEqual(self.calls, 0)

    def test_upstream_profile_kept(self):
        p, _ = self.run_up(self.old_cfg(speedy_profile="upstream"))
        cfg = json.loads(p.read_text())
        self.assertNotIn("STRATA_ROUTE_PRIOR", cfg["env"])
        self.assertNotIn("--adapt-swaps", cfg["args"])

    def test_custom_budget_kept(self):
        p, _ = self.run_up(self.old_cfg(reasoning_budget_tokens=12000))
        self.assertEqual(json.loads(p.read_text())["reasoning_budget_tokens"], 12000)

    def test_low_preset_pair_dropped_but_lone_low_kept(self):
        p, _ = self.run_up(self.old_cfg(reasoning_budget_tokens=4096, reasoning_effort="low"))
        cfg = json.loads(p.read_text())
        self.assertNotIn("reasoning_effort", cfg)
        p, _ = self.run_up(self.old_cfg(reasoning_budget_tokens=20000, reasoning_effort="low"))
        self.assertEqual(json.loads(p.read_text())["reasoning_effort"], "low")

    def test_offline_keeps_stock(self):
        p, ch = self.run_up(self.old_cfg(), ok=False)
        cfg = json.loads(p.read_text())
        self.assertEqual(cfg["args"][cfg["args"].index("--mtp") + 1], str(self.rt))
        self.assertFalse(any("drafter" in c for c in ch))

    def test_user_env_not_overwritten(self):
        p, _ = self.run_up(self.old_cfg(env={"STRATA_ROUTE_PRIOR": "1.0"}))
        self.assertEqual(json.loads(p.read_text())["env"]["STRATA_ROUTE_PRIOR"], "1.0")

    def test_multi_gpu_gets_tool_env_only(self):
        p, _ = self.run_up(self.old_cfg(gpu=[0, 1]))
        cfg = json.loads(p.read_text())
        self.assertNotIn("STRATA_ROUTE_PRIOR", cfg["env"])
        self.assertEqual(cfg["env"]["STRATA_TOOLCALL_REPAIR"], "1")

    def test_own_path_and_deepseek_untouched(self):
        own = next(iter(setup.OWN_PATH_FAMILIES.values()))
        for cfg in (self.old_cfg(arch=own["arch"], model_name=own["name"] + "-q4"),
                    self.old_cfg(model_name="deepseek-v4.1-flash-ud-q8_k_xl")):
            p, ch = self.run_up(cfg)
            self.assertEqual(ch, [])
            self.assertEqual(json.loads(p.read_text())["reasoning_budget_tokens"], 16384)
            self.assertFalse((self.d / "strata-x.json.bak-0.5.2").exists())
        self.assertEqual(self.calls, 0)

    def test_signal_config_switched_only_drafter(self):
        cfg = self.old_cfg(model_name="signal-ap-q4_k_xl")
        cfg["args"] += ["--native", "/m/signal/AP-Q4_K_XL/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf"]
        p, ch = self.run_up(cfg)
        out = json.loads(p.read_text())
        self.assertEqual(out["args"][out["args"].index("--mtp") + 1], str(self.tuned))
        self.assertEqual(len(ch), 1)                              # the drafter only: the rest stays as written
        self.assertEqual(out["reasoning_budget_tokens"], 16384)
        self.assertEqual(out["env"], {"STRATA_HC_Q8": "1"})
        self.assertNotIn("--adapt-swaps", out["args"])
        self.assertEqual(out["args"][out["args"].index("--prefill") + 1], "auto")

    def test_non_flash_next_untouched(self):
        cfg = self.old_cfg(model_name="some-other-model")
        cfg["args"] += ["--native", "/m/other.gguf"]
        p, ch = self.run_up(cfg)
        self.assertEqual(ch, [])
        self.assertEqual(json.loads(p.read_text())["args"][5], str(self.rt))
        self.assertEqual(self.calls, 0)


if __name__ == "__main__":
    unittest.main()
