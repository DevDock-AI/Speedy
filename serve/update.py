#!/usr/bin/env python3
"""speedy-update: move this Speedy install to the latest release, only when you run it (0.4.7).

  ./speedy-update            show the new release and its notes, ask, then update
  ./speedy-update --check    only report (exit 0: up to date, 10: an update is available, 2: could not check)
  ./speedy-update --yes      update without asking
  ./speedy-update --rollback go back to what ran before the last update

A git checkout fetches the tags, checks out the release tag (a fast-forward when main already contains it) and
rebuilds the engine in the existing build dir (the arch it was configured for); the old commit and a copy of the old
binary are kept for --rollback.  A ready-made (zip) install downloads the release's engine zip for this OS and card
family, checks it (SHA256SUMS when the release has one, else GitHub's size and digest) and unpacks it beside the old
one in engine-vX.Y.Z/; the running engine is never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import update_check  # noqa: E402
from version import SPEEDY_VERSION  # noqa: E402

ROOT = HERE.parent
STATE = ".speedy-update"                     # <root>/.speedy-update/previous.json + the kept binaries
WIN = sys.platform == "win32"
EXE = "strata.exe" if WIN else "strata"
FAMILY = {75: "rtx20", 86: "rtx30", 89: "rtx40", 120: "rtx50"}


def say(s=""):
    print(s, flush=True)


def run(cmd, cwd=ROOT, check=True, capture=False):
    say("$ " + " ".join(map(str, cmd)))
    return subprocess.run(list(map(str, cmd)), cwd=cwd, check=check, text=True,
                          stdout=subprocess.PIPE if capture else None)


def git(*a, root=ROOT, check=True) -> str:
    r = subprocess.run(["git", *a], cwd=root, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return r.stdout.strip()


def install_kind(root=ROOT) -> str:
    return "git" if (root / ".git").exists() else "zip"


def family_of(root=ROOT, archs=None) -> str | None:
    """The card family of the installed ready-made engine (engine/BUILD.json), or of `archs`."""
    if archs is None:
        try:
            meta = json.loads((root / "engine" / "BUILD.json").read_text(encoding="utf-8"))
            archs = meta.get("archs") or []
        except Exception:
            return None
    fams = {FAMILY.get(int(a)) for a in archs}
    return fams.pop() if len(fams) == 1 and None not in fams else None


def asset_name(family: str) -> str:
    return f"strata-{'windows' if WIN else 'linux'}-x64-{family}.zip"


def confirm(question: str, yes: bool) -> bool:
    if yes:
        return True
    if not sys.stdin.isatty():
        say("Not a terminal: run again with --yes to update without a question.")
        return False
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, opener=None) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": f"Speedy/{SPEEDY_VERSION}"})
    with (opener or urllib.request.urlopen)(req, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def expected_hash(latest: dict, name: str, workdir: Path, opener=None) -> tuple[str | None, int | None, str]:
    """(sha256 hex, size, where from): SHA256SUMS(.txt) in the release first, else GitHub's asset digest + size."""
    assets = {a["name"]: a for a in latest.get("assets") or []}
    for sums in ("SHA256SUMS", "SHA256SUMS.txt", "sha256sums.txt"):
        if sums in assets and assets[sums].get("url"):
            p = workdir / sums
            download(assets[sums]["url"], p, opener)
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[-1].lstrip("*") == name:
                    return parts[0].lower(), assets[name].get("size"), sums
    a = assets.get(name) or {}
    d = str(a.get("digest") or "")
    return (d.split(":", 1)[1].lower() if d.startswith("sha256:") else None), a.get("size"), "GitHub digest"


def verify(path: Path, sha: str | None, size: int | None) -> None:
    if size is not None and path.stat().st_size != int(size):
        raise SystemExit(f"Download size {path.stat().st_size} != {size}: not installed.")
    if sha is not None and sha256_of(path) != sha:
        raise SystemExit(f"SHA-256 of {path.name} does not match the release: not installed.")
    if sha is None and size is None:
        raise SystemExit("The release publishes no checksum, size or digest for this file: not installed.")


# -- git installs ------------------------------------------------------------------------------------------------
def remote_name(root=ROOT) -> str:
    for line in git("remote", "-v", root=root, check=False).splitlines():
        parts = line.split()
        if len(parts) >= 2 and "DevDock-AI/Speedy" in parts[1]:
            return parts[0]
    return "origin"


def engine_binary(root=ROOT) -> Path | None:
    for p in (root / "build" / EXE, root / "build" / "bin" / EXE):
        if p.exists():
            return p
    return None


def save_state(root: Path, state: dict) -> None:
    d = root / STATE
    d.mkdir(exist_ok=True)
    (d / "previous.json").write_text(json.dumps(state, indent=1), encoding="utf-8")


def load_state(root: Path) -> dict | None:
    try:
        return json.loads((root / STATE / "previous.json").read_text(encoding="utf-8"))
    except Exception:
        return None


def rebuild(root: Path) -> None:
    if (root / "build" / "CMakeCache.txt").exists():   # the arch and options it was configured with stay
        run(["cmake", "--build", "build", "--target", "strata", "-j", str(os.cpu_count() or 4)], cwd=root)
    else:
        run([sys.executable, "setup.py", "--update"], cwd=root)


def update_git(latest: dict, yes: bool, root=ROOT, build=True) -> int:
    if git("status", "--porcelain", "--untracked-files=no", root=root):
        say("You have changes to tracked files here; commit or stash them first (git status lists them).")
        return 1
    remote = remote_name(root)
    run(["git", "fetch", "--tags", remote], cwd=root)
    tag = latest["tag"]
    head = git("rev-parse", "HEAD", root=root)
    branch = git("symbolic-ref", "--short", "-q", "HEAD", root=root, check=False)
    if not confirm(f"Update this checkout to {tag} and rebuild the engine?", yes):
        say("Nothing changed.")
        return 1
    state = {"kind": "git", "commit": head, "branch": branch, "from": SPEEDY_VERSION, "to": latest["version"]}
    exe = engine_binary(root)
    if exe is not None:
        keep = root / STATE / f"{EXE}.{head[:8]}"
        (root / STATE).mkdir(exist_ok=True)
        shutil.copy2(exe, keep)
        state.update(binary=str(exe), binary_backup=str(keep))
    save_state(root, state)
    ff = branch and subprocess.run(["git", "merge-base", "--is-ancestor", "HEAD", tag], cwd=root).returncode == 0
    if ff:
        run(["git", "merge", "--ff-only", tag], cwd=root)
    else:
        run(["git", "checkout", "--detach", tag], cwd=root)
    if build:
        rebuild(root)
    say(f"Updated to {tag}. Restart the server to use it; ./speedy-update --rollback goes back to {head[:8]}.")
    return 0


def rollback(root=ROOT, yes=False) -> int:
    st = load_state(root)
    if not st:
        say("No earlier update recorded here: nothing to roll back.")
        return 1
    if st.get("kind") == "zip":
        say(f"A ready-made update never replaces the old engine: it is still in {st.get('old_engine')}. "
            f"Point the config's \"exe\" back at it (the new one is in {st.get('new_engine')}).")
        return 0
    if not confirm(f"Go back to {st['commit'][:8]} (v{st.get('from')})?", yes):
        return 1
    if st.get("branch"):
        run(["git", "checkout", st["branch"]], cwd=root)
        run(["git", "reset", "--keep", st["commit"]], cwd=root)
    else:
        run(["git", "checkout", "--detach", st["commit"]], cwd=root)
    if st.get("binary_backup") and Path(st["binary_backup"]).exists():
        shutil.copy2(st["binary_backup"], st["binary"])
        say(f"Engine binary restored from {st['binary_backup']}.")
    say("Rolled back. Restart the server.")
    return 0


# -- zip installs ------------------------------------------------------------------------------------------------
def update_zip(latest: dict, yes: bool, root=ROOT, opener=None, family=None) -> int:
    family = family or family_of(root)
    if family is None:
        say("Could not tell this install's card family (engine/BUILD.json); pass --family rtx20|rtx30|rtx40|rtx50.")
        return 1
    name = asset_name(family)
    assets = {a["name"]: a for a in latest.get("assets") or []}
    if name not in assets:
        say(f"The release {latest['tag']} has no {name}; build from source instead (docs/BUILD-LINUX.md).")
        return 1
    dest = root / f"engine-v{latest['version']}"
    if dest.exists():
        say(f"{dest} already exists: nothing downloaded.")
        return 1
    if not confirm(f"Download {name} ({latest['tag']}) into {dest.name}/?", yes):
        say("Nothing changed.")
        return 1
    tmp = root / f".engine-v{latest['version']}.part"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        z = tmp / name
        download(assets[name]["url"], z, opener)
        sha, size, src = expected_hash(latest, name, tmp, opener)
        verify(z, sha, size)
        say(f"Checked {name} ({src}).")
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                if m.startswith(("/", "\\")) or ".." in Path(m).parts:
                    raise SystemExit(f"Unsafe path in the zip: {m}")
            zf.extractall(tmp / "unpacked")
        os.replace(tmp / "unpacked", dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    save_state(root, {"kind": "zip", "old_engine": str(root / "engine"), "new_engine": str(dest),
                      "from": SPEEDY_VERSION, "to": latest["version"]})
    say(f"New engine in {dest} (the old one in engine/ is untouched). Set the config's \"exe\" to "
        f"{dest / EXE} and restart; get the matching server files from {latest['url']}.")
    return 0


def main(argv=None, opener=None) -> int:
    ap = argparse.ArgumentParser(prog="speedy-update", description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="only report whether a newer release exists")
    ap.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation")
    ap.add_argument("--rollback", action="store_true", help="go back to what ran before the last update")
    ap.add_argument("--family", choices=sorted(set(FAMILY.values())), help="zip installs: the card family")
    ap.add_argument("--no-build", action="store_true", help="git installs: check out only, do not rebuild")
    a = ap.parse_args(argv)
    if a.rollback:
        return rollback(ROOT, a.yes)
    try:
        latest = update_check.fetch_latest(SPEEDY_VERSION, timeout=15, opener=opener)
    except Exception as e:
        say(f"Could not reach GitHub ({type(e).__name__}): nothing checked.")
        return 2
    if not update_check.is_newer(latest["version"], SPEEDY_VERSION):
        say(f"Speedy v{SPEEDY_VERSION} is up to date (latest release: v{latest['version']}).")
        return 0
    say(f"Speedy v{latest['version']} is available (you run v{SPEEDY_VERSION}): {latest['url']}")
    if a.check:
        return 10
    say("\nRelease notes:\n" + (latest["notes"].strip() or "(none)") + "\n")
    if install_kind(ROOT) == "git":
        return update_git(latest, a.yes, ROOT, build=not a.no_build)
    return update_zip(latest, a.yes, ROOT, opener, a.family)


if __name__ == "__main__":
    sys.exit(main())
