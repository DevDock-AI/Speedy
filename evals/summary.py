import json, os
def write_summary(out, bench, n, score_lines, ours, ref, extra=()):
    items = [json.loads(l) for l in open(f"{out}/items.jsonl")] if os.path.exists(f"{out}/items.jsonl") else []
    tok = sum(i.get("completion_tokens") or 0 for i in items); wall = sum(i.get("wall_s") or 0 for i in items)
    ln = sum(i.get("finish") == "length" for i in items)
    L = [f"# {bench} - Qwen3.8-Flash-Next UD-Q4_K_XL (local, reasoning high, T1.0 top_p0.95 top_k20, cap 32768)", "",
         *[f"- {s}" for s in score_lines],
         f"- problems: {n} (generated {len(items)})",
         f"- total completion tokens: {tok:,}  (avg {tok/max(1,len(items)):,.0f}/item; hit 32768 cap / finish=length: {ln})",
         f"- total generation time: {wall/3600:.2f} h  (avg {wall/max(1,len(items)):.1f} s/item, {tok/max(1e-9,wall):.1f} tok/s overall)",
         *[f"- {e}" for e in extra], "",
         "| Model | Score |", "|---|---|", f"| **{ours[0]}** | **{ours[1]:.1f}** |", *[f"| {a} | {b} |" for a, b in ref], "",
         "Card numbers: official BF16, uncapped output (up to 262k thinking), likely multi-sample; ours: 1 sample, 32k cap."]
    open(f"{out}/SUMMARY.md", "w").write("\n".join(L) + "\n")
