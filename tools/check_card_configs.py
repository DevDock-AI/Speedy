#!/usr/bin/env python3
"""Check configs/cards/*: every card has a filled CARD.md and at least one <N>x/ preset folder, every JSON parses and
looks like a serve config, and nothing machine-specific (absolute home paths, GPU UUIDs, IPs) slipped in.
  python3 tools/check_card_configs.py            # exit 1 on any problem"""
import json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent / "configs" / "cards"
FIELDS = ["Status", "GPU", "Build", "GPU unlock", "Power limit", "PCIe", "CPU", "RAM", "Storage", "OS / driver / CUDA",
          "Speedy commit", "Maintainer"]
LEAKS = [(re.compile(r"GPU-[0-9a-f]{8}-"), "GPU UUID"), (re.compile(r"/(home|root|Users)/[^\s\"<]+"), "home path"),
         (re.compile(r"[A-Za-z]:\\\\Users\\\\"), "Windows user path"),
         (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b(?<!127\.0\.0\.1)"), "IP address")]
errs = []

def leaks(p):
    t = p.read_text(encoding="utf-8", errors="replace")
    for rx, what in LEAKS:
        if rx.search(t): errs.append(f"{p}: contains a {what}; use the placeholders in docs/CONFIG-PLACEHOLDERS.md")

for card in sorted(d for d in ROOT.iterdir() if d.is_dir()):
    template = card.name == "_template"
    if not template and not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*-\d+gb", card.name):
        errs.append(f"{card}: folder name must look like <model>-<vram>gb, e.g. rtx4090-24gb")
    md = card / "CARD.md"
    if not md.is_file(): errs.append(f"{card}: missing CARD.md"); continue
    rows = dict(re.findall(r"^\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|\s*$", md.read_text(encoding="utf-8"), re.M))
    for f in FIELDS:
        if f not in rows: errs.append(f"{md}: missing row '{f}'")
        elif not template and not rows[f]: errs.append(f"{md}: row '{f}' is empty")
    if not template and "## Measured" not in md.read_text(encoding="utf-8"): errs.append(f"{md}: missing '## Measured' section")
    leaks(md)
    gpus = [d for d in card.iterdir() if d.is_dir() and re.fullmatch(r"\d+x", d.name)]
    if not gpus: errs.append(f"{card}: needs at least one <N>x/ folder (1x/, 2x/, ...)")
    for p in sorted(card.rglob("*")):
        if p.suffix == ".json":
            try: c = json.loads(p.read_text(encoding="utf-8"))
            except ValueError as e: errs.append(f"{p}: invalid JSON ({e})"); continue
            for k in ("exe", "args", "model_name"):
                if k not in c: errs.append(f"{p}: missing '{k}'")
            leaks(p)
        elif p.suffix in (".yaml", ".yml"): leaks(p)

print("\n".join(errs) if errs else f"ok: {sum(1 for d in ROOT.iterdir() if d.is_dir())} card folders checked")
sys.exit(1 if errs else 0)
