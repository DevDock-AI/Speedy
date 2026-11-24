"""The tuned MTP drafter from its Hugging Face repository: source order, SHA-256 check, cache, opt-out, fallback.
python -m unittest tools.test_setup_mtp_hf"""
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

HF_BYTES, GH_BYTES = b"hf-dense", b"gh-dense"
REV = "0" * 40


def pin(b):
    return len(b), hashlib.sha256(b).hexdigest()


class MtpHf(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rt = Path(self.tmp.name) / "mtp" / "rt"
        self.rt.mkdir(parents=True)
        for n in ("dense.bin", "dense.txt", "experts.bin", "draft_vocab.bin"):
            (self.rt / n).write_bytes(b"stock")
        self.dense = self.rt.parent / "rt-tuned" / "dense.bin"
        self.urls = []
        self.serve = {"huggingface.co": HF_BYTES, "github.com": GH_BYTES}
        size, sha = pin(GH_BYTES)
        self.patches = [mock.patch.object(setup, "MTP_HF_REVISION", REV),
                        mock.patch.dict(setup.MTP_HF_FILES, {"dense.bin": pin(HF_BYTES)}),
                        mock.patch.dict(setup.MTP_TUNED, {"size": size, "sha256": sha}),
                        mock.patch.object(setup, "download", self.fake_download),
                        mock.patch.object(setup, "warn", lambda *a, **k: None),
                        mock.patch.dict(os.environ, {"STRATA_MTP_HF": "", "HF_ENDPOINT": ""})]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def fake_download(self, url, dst, what=None):
        self.urls.append(url)
        host = url.split("/")[2]
        if self.serve.get(host) is None:
            raise SystemExit(f"cannot reach {host}")
        dst.write_bytes(self.serve[host])

    def test_hf_first_pinned(self):
        out = setup.tuned_mtp_dir(self.rt)
        self.assertEqual(out, self.rt.parent / "rt-tuned")
        self.assertEqual(self.dense.read_bytes(), HF_BYTES)
        self.assertEqual(self.urls, [f"https://huggingface.co/{setup.MTP_HF_REPO}/resolve/{REV}/dense.bin"])
        self.assertTrue((out / "experts.bin").exists() and (out / "dense.txt").exists())

    def test_cache_hit_skips_download(self):
        setup.tuned_mtp_dir(self.rt)
        self.urls.clear()
        self.assertIsNotNone(setup.tuned_mtp_dir(self.rt))
        self.assertEqual(self.urls, [])

    def test_bad_sha_falls_back_to_release(self):
        self.serve["huggingface.co"] = b"tampered"
        self.assertIsNotNone(setup.tuned_mtp_dir(self.rt))
        self.assertEqual(self.dense.read_bytes(), GH_BYTES)
        self.assertEqual(len(self.urls), 2)
        self.assertFalse(self.dense.with_name("dense.bin.new").exists())

    def test_all_bad_falls_back_to_stock(self):
        self.serve = {"huggingface.co": b"bad", "github.com": None}
        self.assertIsNone(setup.tuned_mtp_dir(self.rt))   # the caller keeps the stock drafter
        self.assertFalse(self.dense.exists())

    def test_offline_keeps_older_known_file(self):
        self.dense.parent.mkdir(parents=True)
        self.dense.write_bytes(GH_BYTES)                   # an earlier install's release-asset drafter
        self.serve = {}
        self.assertIsNotNone(setup.tuned_mtp_dir(self.rt))
        self.assertEqual(self.dense.read_bytes(), GH_BYTES)

    def test_older_file_upgraded_when_online(self):
        self.dense.parent.mkdir(parents=True)
        self.dense.write_bytes(GH_BYTES)
        setup.tuned_mtp_dir(self.rt)
        self.assertEqual(self.dense.read_bytes(), HF_BYTES)

    def test_opt_out_env_skips_hf(self):
        with mock.patch.dict(os.environ, {"STRATA_MTP_HF": "0"}):
            setup.tuned_mtp_dir(self.rt)
        self.assertEqual([u.split("/")[2] for u in self.urls], ["github.com"])

    def test_unpublished_revision_not_tried(self):
        with mock.patch.object(setup, "MTP_HF_REVISION", None):
            setup.tuned_mtp_dir(self.rt)
        self.assertEqual([u.split("/")[2] for u in self.urls], ["github.com"])

    def test_stock_drafter_never_downloads(self):
        cfg = {"model_name": "qwen3.8-flash-next-unsloth-ud-q4_k_xl", "mtp_drafter": "stock",
               "args": ["--mtp", str(self.rt)]}
        self.assertTrue(setup._env_says_stock(cfg))
        self.assertTrue(setup._env_says_stock({"env": {"STRATA_MTP_DRAFTER": "stock"}}))

    def test_which_models(self):
        for name in ("qwen3.8-flash-next-unsloth-ud-q4_k_xl", "qwen3.8-flash-next-unsloth-ud-iq4_xs",
                     "qwen3.8-flash-next-iq3_s"):
            self.assertTrue(setup.flash_next_config({"model_name": name}), name)
        self.assertFalse(setup.flash_next_config({"model_name": "qwen3.8-flash-next-unsloth-ud-q4_k_xl",
                                                  "backend": "metal"}))
        self.assertFalse(setup.flash_next_config({"model_name": "qwen3.6-35b-a3b", "arch": "qwen35moe"}))


if __name__ == "__main__":
    unittest.main()
