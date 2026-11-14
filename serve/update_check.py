"""New-release check (0.4.7).

On start and then every STRATA_UPDATE_CHECK_HOURS (default 3) hours a background thread asks GitHub for the latest
Speedy release (GET /repos/DevDock-AI/Speedy/releases/latest, 5 s timeout, User-Agent "Speedy/<version>", nothing
else is sent) and compares it with SPEEDY_VERSION.  The answer is cached in <user cache dir>/speedy/update-check.json
for the same interval, so restarts do not ask again.  STRATA_UPDATE_CHECK=0 turns it off; with no network it stays
quiet and asks again at the next interval.  It never updates anything and never touches model output: the result is
a log line, the "speedy" field of /health, GET /speedy/version and the X-Speedy-Update response header.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from version import SPEEDY_VERSION
except ImportError:                                    # imported as serve.update_check
    from serve.version import SPEEDY_VERSION

REPO = "DevDock-AI/Speedy"
LATEST_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
TIMEOUT_S = 5
DEFAULT_HOURS = 3.0
UPDATE_COMMAND = "python serve/update.py" if sys.platform == "win32" else "./speedy-update"


def parse_version(v) -> tuple[int, ...] | None:
    """'v0.4.7' / '0.4.7' / '0.4.7-rc1' -> (0, 4, 7); None if it does not start with a number."""
    m = re.match(r"\s*v?(\d+(?:\.\d+)*)", str(v or ""))
    if not m:
        return None
    t = tuple(int(x) for x in m.group(1).split("."))
    return t + (0,) * (3 - len(t)) if len(t) < 3 else t


def is_newer(latest, current) -> bool:
    a, b = parse_version(latest), parse_version(current)
    return bool(a and b and a > b)


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    if str(env.get("STRATA_OFFLINE", "")).strip() in ("1", "true") or str(env.get("HF_HUB_OFFLINE", "")).strip() == "1":
        return False                                   # strict local mode: no network at all
    return str(env.get("STRATA_UPDATE_CHECK", "1")).strip().lower() not in ("0", "off", "false", "no")


def interval_s(env=None) -> float:
    env = os.environ if env is None else env
    try:
        h = float(env.get("STRATA_UPDATE_CHECK_HOURS", DEFAULT_HOURS))
    except (TypeError, ValueError):
        h = DEFAULT_HOURS
    return max(h, 0.25) * 3600.0                       # at least 15 min: GitHub allows 60 unauthenticated calls/hour


def cache_path(env=None) -> Path:
    env = os.environ if env is None else env
    if env.get("STRATA_UPDATE_CACHE"):
        return Path(env["STRATA_UPDATE_CACHE"])
    if sys.platform == "win32":
        base = Path(env.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(env.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "speedy" / "update-check.json"


def notes_excerpt(body, limit=400) -> str:
    text = re.sub(r"\s+", " ", str(body or "")).strip()
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


def fetch_latest(current=SPEEDY_VERSION, timeout=TIMEOUT_S, opener=None) -> dict:
    """The latest release, as {tag, version, url, notes, published, assets}.  Raises on any failure."""
    req = urllib.request.Request(LATEST_URL, headers={"User-Agent": f"Speedy/{current}",
                                                      "Accept": "application/vnd.github+json"})
    with (opener or urllib.request.urlopen)(req, timeout=timeout) as r:
        rel = json.loads(r.read().decode("utf-8"))
    tag = str(rel.get("tag_name") or "")
    if parse_version(tag) is None:
        raise ValueError(f"release tag {tag!r} is not a version")
    return {"tag": tag, "version": ".".join(map(str, parse_version(tag))),
            "url": rel.get("html_url") or f"https://github.com/{REPO}/releases/tag/{tag}",
            "notes": str(rel.get("body") or ""), "published": rel.get("published_at"),
            "assets": [{"name": a.get("name"), "size": a.get("size"), "digest": a.get("digest"),
                        "url": a.get("browser_download_url")} for a in rel.get("assets") or []]}


def notice(current: str, latest: dict) -> str:
    return (f"Speedy v{latest['version']} is available (you run v{current}): {latest['url']} "
            f"— update with: {UPDATE_COMMAND}")


class UpdateChecker:
    """Thread-safe holder of the last answer; `start()` runs the periodic check in a daemon thread."""

    def __init__(self, current=SPEEDY_VERSION, env=None, opener=None, log=None, clock=time.time):
        self.current = current
        self.env = dict(os.environ if env is None else env)
        self.opener = opener
        self.log = log or (lambda s: print(f"[strata] {s}", flush=True))
        self.clock = clock
        self.enabled = enabled(self.env)
        self.interval = interval_s(self.env)
        self.path = cache_path(self.env)
        self.lock = threading.Lock()
        self.latest: dict | None = None
        self.checked_at: float | None = None
        self.offline = False
        self.announced: str | None = None
        self.stop = threading.Event()
        self.thread = None

    # -- cache ------------------------------------------------------------------------------------------------
    def load_cache(self) -> bool:
        """Take a cached answer that is younger than the interval; True if it was."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            at, latest = float(data["checked_at"]), data["latest"]
            if not isinstance(latest, dict) or parse_version(latest.get("version")) is None:
                return False
        except Exception:
            return False
        with self.lock:
            self.latest, self.checked_at = latest, at
            self.announced = data.get("announced")
        return 0 <= self.clock() - at < self.interval

    def save_cache(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            with self.lock:
                data = {"checked_at": self.checked_at, "current": self.current, "latest": self.latest,
                        "announced": self.announced}
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass                                       # a read-only home is not a reason to complain

    # -- state ------------------------------------------------------------------------------------------------
    def update_available(self) -> str | None:
        with self.lock:
            v = self.latest and self.latest.get("version")
        return v if v and is_newer(v, self.current) else None

    def status(self) -> dict:
        with self.lock:
            latest = dict(self.latest) if self.latest else None
            at, offline = self.checked_at, self.offline
        avail = bool(latest and is_newer(latest["version"], self.current))
        return {"current": self.current, "latest": latest["version"] if latest else None,
                "update_available": avail, "url": latest.get("url") if latest else None,
                "notes_excerpt": notes_excerpt(latest.get("notes")) if latest else None,
                "update_command": UPDATE_COMMAND if avail else None,
                "checked_at": at, "check_enabled": self.enabled, "offline": offline}

    def announce(self, force=False) -> None:
        """One log line when an update is known: at start (force) and whenever a version is newly found."""
        v = self.update_available()
        if not v or (v == self.announced and not force):
            return
        with self.lock:
            latest = dict(self.latest)
            self.announced = v
        self.log(notice(self.current, latest))

    def check_now(self) -> bool:
        try:
            latest = fetch_latest(self.current, opener=self.opener)
        except (urllib.error.URLError, OSError, TimeoutError):
            with self.lock:
                self.offline = True                    # offline: quiet, ask again at the next interval
            return False
        except Exception:
            return False
        with self.lock:
            self.latest, self.checked_at, self.offline = latest, self.clock(), False
        self.announce()
        self.save_cache()
        return True

    def run(self) -> None:
        fresh = self.load_cache()
        self.announce(force=True)                      # the cached answer at start, before any network
        wait = 0.0
        if fresh:
            wait = max(0.0, self.interval - (self.clock() - self.checked_at))
        while not self.stop.wait(wait):
            self.check_now()
            wait = self.interval

    def start(self) -> "UpdateChecker":
        if self.enabled and self.thread is None:
            self.thread = threading.Thread(target=self.run, name="speedy-update-check", daemon=True)
            self.thread.start()
        return self
