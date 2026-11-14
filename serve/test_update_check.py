"""0.4.7 update check and speedy-update (serve/update_check.py, serve/update.py); HTTP is mocked, no GPU."""
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import update_check as uc  # noqa: E402
import update as upd  # noqa: E402


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def release(tag="v0.4.8", body="Fixes.\n\n- a\n- b", assets=()):
    return {"tag_name": tag, "html_url": f"https://github.com/DevDock-AI/Speedy/releases/tag/{tag}", "body": body,
            "assets": list(assets)}


class Opener:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, req, timeout=None):
        self.calls.append(req)
        v = self.routes[req.full_url]
        if isinstance(v, Exception):
            raise v
        return Resp(v if isinstance(v, bytes) else json.dumps(v).encode())


class Versions(unittest.TestCase):
    def test_parse_compare(self):
        self.assertEqual(uc.parse_version("v0.4.7"), (0, 4, 7))
        self.assertEqual(uc.parse_version("0.5"), (0, 5, 0))
        self.assertIsNone(uc.parse_version("nightly"))
        self.assertTrue(uc.is_newer("v0.4.10", "0.4.9"))
        self.assertFalse(uc.is_newer("v0.4.7", "0.4.7"))
        self.assertFalse(uc.is_newer("garbage", "0.4.7"))

    def test_env(self):
        self.assertTrue(uc.enabled({}))
        self.assertFalse(uc.enabled({"STRATA_UPDATE_CHECK": "0"}))
        self.assertFalse(uc.enabled({"STRATA_OFFLINE": "1"}))
        self.assertEqual(uc.interval_s({}), 3 * 3600)
        self.assertEqual(uc.interval_s({"STRATA_UPDATE_CHECK_HOURS": "24"}), 24 * 3600)
        self.assertEqual(uc.interval_s({"STRATA_UPDATE_CHECK_HOURS": "x"}), 3 * 3600)
        self.assertEqual(uc.interval_s({"STRATA_UPDATE_CHECK_HOURS": "0"}), 900)


class Checker(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = {"STRATA_UPDATE_CACHE": os.path.join(self.tmp.name, "c.json")}
        self.lines = []
        self.now = [1_000_000.0]

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, routes, current="0.4.7", env=None):
        return uc.UpdateChecker(current, env={**self.env, **(env or {})}, opener=Opener(routes),
                                log=self.lines.append, clock=lambda: self.now[0])

    def test_request_is_anonymous(self):
        op = Opener({uc.LATEST_URL: release()})
        uc.fetch_latest("0.4.7", opener=op)
        hdrs = {k.lower(): v for k, v in op.calls[0].header_items()}
        self.assertEqual(hdrs["user-agent"], "Speedy/0.4.7")
        self.assertEqual(set(hdrs), {"user-agent", "accept"})
        self.assertIsNone(op.calls[0].data)

    def test_newer_logs_once_and_caches(self):
        c = self.make({uc.LATEST_URL: release("v0.4.8")})
        self.assertTrue(c.check_now())
        self.assertEqual(len(self.lines), 1)
        self.assertIn("Speedy v0.4.8 is available (you run v0.4.7)", self.lines[0])
        self.assertIn("update with:", self.lines[0])
        c.check_now()
        self.assertEqual(len(self.lines), 1)            # not again for the same version
        st = c.status()
        self.assertTrue(st["update_available"])
        self.assertEqual((st["current"], st["latest"]), ("0.4.7", "0.4.8"))
        self.assertTrue(st["notes_excerpt"].startswith("Fixes."))
        d = json.loads(Path(self.env["STRATA_UPDATE_CACHE"]).read_text())
        self.assertEqual(d["latest"]["version"], "0.4.8")

    def test_same_version_quiet(self):
        c = self.make({uc.LATEST_URL: release("v0.4.7")})
        c.check_now()
        self.assertEqual(self.lines, [])
        self.assertIsNone(c.update_available())
        self.assertFalse(c.status()["update_available"])

    def test_offline_quiet(self):
        c = self.make({uc.LATEST_URL: urllib.error.URLError("down")})
        self.assertFalse(c.check_now())
        self.assertTrue(c.status()["offline"])
        self.assertEqual(self.lines, [])

    def test_cache_ttl_follows_interval(self):
        self.make({uc.LATEST_URL: release("v0.4.8")}).check_now()
        fresh = self.make({})
        self.assertTrue(fresh.load_cache())
        self.now[0] += 3 * 3600 + 1
        self.assertFalse(self.make({}).load_cache())    # older than 3 h: ask again
        self.assertTrue(self.make({}, env={"STRATA_UPDATE_CHECK_HOURS": "24"}).load_cache())

    def test_run_uses_fresh_cache_without_network(self):
        self.make({uc.LATEST_URL: release("v0.4.8")}).check_now()
        self.lines.clear()
        c = self.make({uc.LATEST_URL: AssertionError("no fetch expected")})
        c.stop.set()
        c.run()
        self.assertEqual(len(self.lines), 1)            # the start notice from the cache

    def test_disabled_never_starts(self):
        c = self.make({}, env={"STRATA_UPDATE_CHECK": "0"})
        c.start()
        self.assertIsNone(c.thread)


class ServerWiring(unittest.TestCase):
    def test_health_header_and_endpoint(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from serve import server
        import http.client

        class Svc:
            pass
        c = uc.UpdateChecker("0.4.7", env={"STRATA_UPDATE_CACHE": os.devnull}, opener=Opener({uc.LATEST_URL: release()}),
                             log=lambda s: None)
        c.check_now()
        svc = Svc()
        svc.update_checker = c
        self.assertEqual(server.speedy_version_info(svc)["latest"], "0.4.8")
        Svc.update_checker = None
        self.assertEqual(server.speedy_version_info(Svc())["current"], server.SPEEDY_VERSION)


class Updater(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def zip_bytes(self):
        b = io.BytesIO()
        with zipfile.ZipFile(b, "w") as z:
            z.writestr("strata", "ENGINE")
            z.writestr("BUILD.json", json.dumps({"version": "0.4.8", "archs": [86]}))
        return b.getvalue()

    def latest(self, data, digest=True, sums=None):
        name = upd.asset_name("rtx30")
        assets = [{"name": name, "size": len(data), "url": "https://x/" + name,
                   "digest": "sha256:" + hashlib.sha256(data).hexdigest() if digest else None}]
        if sums is not None:
            assets.append({"name": "SHA256SUMS", "size": len(sums), "url": "https://x/SHA256SUMS"})
        return {"tag": "v0.4.8", "version": "0.4.8", "url": "u", "notes": "", "assets": assets}, name

    def test_zip_side_by_side(self):
        (self.root / "engine").mkdir()
        (self.root / "engine" / "BUILD.json").write_text(json.dumps({"archs": [86]}))
        (self.root / "engine" / "strata").write_text("OLD")
        data = self.zip_bytes()
        latest, name = self.latest(data)
        rc = upd.update_zip(latest, True, self.root, Opener({"https://x/" + name: data}))
        self.assertEqual(rc, 0)
        self.assertEqual((self.root / "engine-v0.4.8" / "strata").read_text(), "ENGINE")
        self.assertEqual((self.root / "engine" / "strata").read_text(), "OLD")
        self.assertEqual(upd.update_zip(latest, True, self.root, Opener({})), 1)   # never overwrites

    def test_zip_sha256sums_mismatch_refused(self):
        data = self.zip_bytes()
        latest, name = self.latest(data, digest=False, sums=b"")
        sums = f"{'0' * 64}  {name}\n".encode()
        with self.assertRaises(SystemExit):
            upd.update_zip(latest, True, self.root,
                           Opener({"https://x/" + name: data, "https://x/SHA256SUMS": sums}), family="rtx30")
        self.assertFalse((self.root / "engine-v0.4.8").exists())

    def test_zip_size_mismatch_refused(self):
        data = self.zip_bytes()
        latest, name = self.latest(data)
        latest["assets"][0]["size"] += 1
        with self.assertRaises(SystemExit):
            upd.update_zip(latest, True, self.root, Opener({"https://x/" + name: data}), family="rtx30")

    def test_no_confirmation_no_change(self):
        data = self.zip_bytes()
        latest, name = self.latest(data)
        old = sys.stdin
        sys.stdin = io.StringIO("")
        try:
            self.assertEqual(upd.update_zip(latest, False, self.root, Opener({}), family="rtx30"), 1)
        finally:
            sys.stdin = old
        self.assertFalse((self.root / "engine-v0.4.8").exists())

    def test_check_exit_codes(self):
        self.assertEqual(upd.main(["--check"], opener=Opener({uc.LATEST_URL: release("v99.0.0")})), 10)
        self.assertEqual(upd.main(["--check"], opener=Opener({uc.LATEST_URL: release("v0.0.1")})), 0)
        self.assertEqual(upd.main(["--check"], opener=Opener({uc.LATEST_URL: urllib.error.URLError("x")})), 2)

    def test_git_update_and_rollback(self):
        r = self.root / "repo"
        r.mkdir()
        g = lambda *a: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=r, check=True,
                                      capture_output=True)
        g("init", "-q", "-b", "main")
        (r / "f").write_text("1")
        g("add", "f")
        g("commit", "-qm", "one")
        first = upd.git("rev-parse", "HEAD", root=r)
        (r / "f").write_text("2")
        g("commit", "-qam", "two")
        g("tag", "v0.4.8")
        g("reset", "-q", "--hard", first)
        (r / "build").mkdir()
        (r / "build" / "strata").write_text("OLDBIN")
        g("remote", "add", "origin", str(r))           # fetch from itself: no network
        rc = upd.update_git({"tag": "v0.4.8", "version": "0.4.8"}, True, r, build=False)
        self.assertEqual(rc, 0)
        self.assertEqual((r / "f").read_text(), "2")
        (r / "build" / "strata").write_text("NEWBIN")
        self.assertEqual(upd.rollback(r, yes=True), 0)
        self.assertEqual(upd.git("rev-parse", "HEAD", root=r), first)
        self.assertEqual((r / "build" / "strata").read_text(), "OLDBIN")


if __name__ == "__main__":
    unittest.main()
