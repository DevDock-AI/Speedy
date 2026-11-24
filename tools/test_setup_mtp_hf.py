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


class MtpHfFull(unittest.TestCase):
    """0.5.5: the whole drafter from Hugging Face on a fresh install, the Qwen download + pack as the fallback."""
    FILES = {"dense.txt": b"index", "experts.bin": b"experts" * 9, "draft_vocab.bin": b"vocab", "dense.bin": b"tuned"}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mtp = Path(self.tmp.name) / "mtp"
        self.rt = self.mtp / "rt"
        self.urls, self.ran, self.fetched = [], [], []
        self.serve = dict(self.FILES)
        self.patches = [mock.patch.object(setup, "MTP_HF_REVISION", REV),
                        mock.patch.dict(setup.MTP_HF_FILES, {n: pin(b) for n, b in self.FILES.items()}),
                        mock.patch.object(setup, "download", self.fake_download),
                        mock.patch.object(setup, "fetch_mtp", lambda mtp, env: self.fetched.append(mtp)),
                        mock.patch.object(setup, "run", lambda cmd, **k: self.ran.append(Path(str(cmd[1])).name)),
                        mock.patch.object(setup, "warn", lambda *a, **k: None),
                        mock.patch.object(setup, "ok", lambda *a, **k: None),
                        mock.patch.object(setup, "say", lambda *a, **k: None),
                        mock.patch.dict(os.environ, {"STRATA_MTP_HF": "", "HF_ENDPOINT": "", "STRATA_MTP_DRAFTER": ""})]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def fake_download(self, url, dst, what=None):
        self.urls.append(url)
        body = self.serve.get(url.rsplit("/", 1)[1])
        if body is None:
            raise SystemExit("cannot reach huggingface.co")
        dst.write_bytes(body)

    def layer(self, drafter="tuned", corrupt=False, mtp="on"):
        a = mock.Mock(mtp=mtp, mtp_drafter=drafter)
        return setup.fetch_mtp_layer(self.mtp, self.rt, {}, a, corrupt)

    def old_path_used(self):
        self.assertEqual(self.fetched, [self.mtp])
        self.assertEqual(self.ran, ["mtp_pack.py", "mtp_rt.py"])

    def test_fresh_install_hf_full_skips_mtp_fetch(self):
        out = self.layer()
        self.assertEqual(out, self.mtp / "rt-tuned")
        for n, b in self.FILES.items():
            self.assertEqual((out / n).read_bytes(), b)
        self.assertEqual((self.fetched, self.ran), ([], []))
        self.assertTrue(all(f"/{setup.MTP_HF_REPO}/resolve/{REV}/" in u for u in self.urls))
        self.assertFalse(any(p.name.endswith((".new", ".part")) for p in out.iterdir()))
        self.urls.clear()
        self.assertEqual(self.layer(), out)                # cached: no network the second time
        self.assertEqual(self.urls, [])
        self.assertEqual(self.layer(mtp="off"), out)       # --mtp off keeps a drafter that is already there

    def test_hf_mirror(self):
        with mock.patch.dict(os.environ, {"HF_ENDPOINT": "https://hf-mirror.com"}):
            self.layer()
        self.assertTrue(all(u.startswith("https://hf-mirror.com/") for u in self.urls))

    def test_hf_failure_old_path(self):
        self.serve.pop("experts.bin")
        self.assertIsNone(self.layer())
        self.old_path_used()

    def test_sha_mismatch_old_path(self):
        self.serve["experts.bin"] = b"tampered"
        self.assertIsNone(self.layer())
        self.old_path_used()
        self.assertFalse((self.mtp / "rt-tuned" / "experts.bin").exists())
        self.assertFalse((self.mtp / "rt-tuned" / "experts.bin.new.part").exists())

    def test_stock_choice_old_path(self):
        self.assertIsNone(self.layer("stock"))
        self.old_path_used()
        self.assertEqual(self.urls, [])

    def test_stock_env_old_path(self):
        with mock.patch.dict(os.environ, {"STRATA_MTP_DRAFTER": "stock"}):
            self.assertIsNone(self.layer())
        self.old_path_used()

    def test_hf_off_old_path(self):
        with mock.patch.dict(os.environ, {"STRATA_MTP_HF": "0"}):
            self.assertIsNone(self.layer())
        self.old_path_used()
        self.assertEqual(self.urls, [])

    def test_existing_stock_layer_untouched(self):
        self.rt.mkdir(parents=True)
        (self.rt / "experts.bin").write_bytes(b"stock")
        self.assertIsNone(self.layer())
        self.assertEqual((self.urls, self.fetched, self.ran), ([], [], []))

    def test_any_draft_vocab_from_hf_files(self):
        out = self.layer()
        (out / "draft_vocab.bin").write_bytes((ROOT / "data" / "draft_vocab_en.bin").read_bytes())
        setup.refresh_draft_vocab(out, "cyrillic")
        self.assertEqual((out / "draft_vocab.bin").read_bytes(),
                         (ROOT / "data" / "draft_vocab_cyrillic.bin").read_bytes())


class SignalConfig(unittest.TestCase):
    def test_signal_detected(self):
        cfg = {"model_name": "signal-xl", "args": ["--native", "/m/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf"]}
        self.assertTrue(setup.flash_next_config(cfg))
        self.assertTrue(setup.flash_next_config({"model_name": "Signal-3.8-Flash-Next-AP-Q4_K_XL"}))
        self.assertFalse(setup.flash_next_config({"model_name": "other", "args": ["--native", "/m/x.gguf"]}))
        self.assertFalse(setup.flash_next_config({"model_name": "signal-3.8-flash-next", "backend": "metal"}))


if __name__ == "__main__":
    unittest.main()
