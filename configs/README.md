# Configs

```
configs/
├── cards/                      one folder per GPU model, tuned and measured on that card
│   ├── rtx3080-10gb/           reference card (all README numbers)
│   │   ├── CARD.md             hardware, build flags, measured numbers, tuning notes
│   │   ├── 1x/                 presets for one card (+ llama-swap.yaml)
│   │   └── 2x/                 presets for two cards (layer split)
│   └── _template/              copy this to add a card
├── generic/                    presets with /opt paths, written by make_configs.py
└── make_configs.py             writes generic presets for your paths
```

**Pick a preset:** use the folder for your card and GPU count. If your card has no folder yet, start from the closest
one (same VRAM first, then same architecture) and fill the placeholders ([docs/CONFIG-PLACEHOLDERS.md](../docs/CONFIG-PLACEHOLDERS.md)).

## Add a card

1. `cp -r configs/cards/_template configs/cards/<model>-<vram>gb` (lower case, e.g. `rtx4090-24gb`, `rtx3090-24gb`, `a6000-48gb`).
2. Put your presets in `1x/`, `2x/`, ... (one folder per GPU count). Start from the closest existing card and change
   only what you measured. Keep the placeholders; never commit real paths, GPU UUIDs or hostnames.
3. Fill every row of `CARD.md` and the **Measured** table (bench/speedbench.py, cold prompts at about 60K / 120K / 200K).
4. Run `python3 tools/check_card_configs.py` (CI runs it too).
5. Open a PR with the full hardware table (see [CONTRIBUTING.md](../CONTRIBUTING.md)).

A new card folder is **community** status until a maintainer reproduces it; then it becomes **verified**.
Card folders never change the engine's defaults; a default change still needs the full evidence bar in CONTRIBUTING.md.
