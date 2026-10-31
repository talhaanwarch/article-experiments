"""Summarise ../data/results/raw.jsonl into the tables used in the article.

Usage:
    python analyze.py > ../data/results/summary.md

Style bias follows the paper's definition: the share of verdicts that pick the
markdown answer minus the share that pick the prose answer, over both answer
orders. +1 means always markdown, 0 means no preference, -1 means always prose.
The 95% interval comes from resampling the pairs (2,000 bootstrap draws).

It is reported twice: on all 50 pairs, and on the "clean" pairs where both
answers end on a complete sentence and neither is less than 75% of the other's
word count. In the other pairs one answer is cut off or much shorter, so
picking the other one is a fair call rather than a style preference.

The main tables use the first run of every cell (rep 1). Cells that were
re-asked with run_judges.py --repeat get their own table at the end.
"""

import json
import re
import random
from collections import defaultdict
from pathlib import Path

from run_judges import JUDGES, PROMPTS, RESULTS_FILE, is_clean, load_pairs


def bias(rows: list[dict]) -> float:
    md = sum(r["picked"] == "markdown" for r in rows)
    pr = sum(r["picked"] == "prose" for r in rows)
    return (md - pr) / len(rows)


def bootstrap_ci(rows: list[dict], draws: int = 2000) -> tuple[float, float]:
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[r["pair"]].append(r)
    pairs = list(by_pair)
    rng = random.Random(0)
    stats = []
    for _ in range(draws):
        sample = [r for p in rng.choices(pairs, k=len(pairs)) for r in by_pair[p]]
        stats.append(bias(sample))
    stats.sort()
    return stats[int(0.025 * draws)], stats[int(0.975 * draws) - 1]


def share(rows: list[dict], value: str) -> str:
    return f"{100 * sum(r['picked'] == value for r in rows) / len(rows):.0f}%"


def main() -> None:
    pairs = load_pairs()
    clean = {p["id"] for p in pairs if is_clean(p)}
    all_rows = [json.loads(line) for line in Path(RESULTS_FILE).read_text().splitlines()]
    rows = [r for r in all_rows if r.get("rep", 1) == 1]
    valid = [r for r in rows if r["picked"] in ("markdown", "prose", "tie")]
    judges = JUDGES + sorted({r["judge"] for r in rows} - set(JUDGES))

    print(f"Calls: {len(rows)}. Valid verdicts: {len(valid)}. "
          f"Invalid or failed: {len(rows) - len(valid)}. "
          f"Cost of the first run: ${sum(r.get('cost_usd') or 0 for r in rows):.4f}. "
          f"Cost including repeats: ${sum(r.get('cost_usd') or 0 for r in all_rows):.4f}\n")
    print(f"Clean pairs ({len(clean)}): {', '.join(sorted(clean))}\n")

    print("## Style bias (markdown vs prose, same content)\n")
    print("| Judge | Prompt | All 50 pairs | 95% CI | Clean pairs | 95% CI | Markdown / prose / tie (clean) |")
    print("|---|---|---|---|---|---|---|")
    for judge in judges:
        for prompt in PROMPTS:
            sel = [r for r in valid if r["judge"] == judge and r["prompt"] == prompt and r["experiment"] == "style"]
            sel_clean = [r for r in sel if r["pair"] in clean]
            if not sel or not sel_clean:
                continue
            lo, hi = bootstrap_ci(sel)
            clo, chi = bootstrap_ci(sel_clean)
            print(f"| {judge} | {prompt} | {bias(sel):+.2f} | {lo:+.2f} to {hi:+.2f} "
                  f"| {bias(sel_clean):+.2f} | {clo:+.2f} to {chi:+.2f} "
                  f"| {share(sel_clean, 'markdown')} / {share(sel_clean, 'prose')} / {share(sel_clean, 'tie')} |")

    # Split the pairs that are not clean: a short prose answer leaves out content,
    # while a cut-off last sentence leaves both answers at a similar length.
    def words(text: str) -> int:
        return len(re.findall(r"[A-Za-z0-9]+", text))

    short = {p["id"] for p in pairs if p["id"] not in clean and words(p["prose"]) / words(p["markdown"]) < 0.75}
    cut = {p["id"] for p in pairs if p["id"] not in clean and p["id"] not in short}
    print("\n## Style bias by pair group, plain prompt\n")
    print(f"| Judge | Clean ({len(clean)}) | Cut off, similar length ({len(cut)}) | Prose much shorter ({len(short)}) |")
    print("|---|---|---|---|")
    for judge in judges + ["all judges"]:
        sel = [r for r in valid if r["prompt"] == "naive" and r["experiment"] == "style"
               and (judge == "all judges" or r["judge"] == judge)]
        groups = [[r for r in sel if r["pair"] in ids] for ids in (clean, cut, short)]
        if all(groups):
            print(f"| {judge} | " + " | ".join(f"{bias(g):+.2f}" for g in groups) + " |")

    print("\n## Planted error (markdown answer wrong, prose answer right)\n")
    print("| Judge | Prompt | Picks correct prose | Picks wrong markdown | Tie | n |")
    print("|---|---|---|---|---|---|")
    for judge in judges:
        for prompt in PROMPTS:
            sel = [r for r in valid if r["judge"] == judge and r["prompt"] == prompt and r["experiment"] == "error"]
            if sel:
                print(f"| {judge} | {prompt} | {share(sel, 'prose')} | {share(sel, 'markdown')} | {share(sel, 'tie')} | {len(sel)} |")

    print("\n## First-slot preference on the style pairs (verdict A minus verdict B)\n")
    print("| Judge | Prompt | A minus B | Tie rate |")
    print("|---|---|---|---|")
    for judge in judges:
        for prompt in PROMPTS:
            sel = [r for r in valid if r["judge"] == judge and r["prompt"] == prompt and r["experiment"] == "style"]
            if sel:
                a = sum(r["verdict"] == "A" for r in sel)
                b = sum(r["verdict"] == "B" for r in sel)
                print(f"| {judge} | {prompt} | {(a - b) / len(sel):+.2f} | {share(sel, 'tie')} |")

    print("\n## Cost and tokens per judge\n")
    print("| Judge | Calls | Cost | Avg output tokens | Avg reasoning tokens |")
    print("|---|---|---|---|---|")
    for judge in judges:
        sel = [r for r in rows if r["judge"] == judge]
        if not sel:
            continue
        out = [r.get("completion_tokens") or 0 for r in sel]
        rea = [r.get("reasoning_tokens") or 0 for r in sel]
        print(f"| {judge} | {len(sel)} | ${sum(r.get('cost_usd') or 0 for r in sel):.4f} "
              f"| {sum(out) / len(sel):.0f} | {sum(rea) / len(sel):.0f} |")

    repeated = sorted({(r["judge"], r["prompt"], r["experiment"]) for r in all_rows if r.get("rep", 1) > 1})
    if repeated:
        print("\n## Repeated cells (planted error: wrong markdown picked, per run)\n")
        print("| Judge | Prompt | Run 1 | Run 2 | Run 3 | Pairs where markdown won |")
        print("|---|---|---|---|---|---|")
        for judge, prompt, experiment in repeated:
            cells, won = [], set()
            for rep in (1, 2, 3):
                sel = [r for r in all_rows if r["judge"] == judge and r["prompt"] == prompt
                       and r["experiment"] == experiment and r.get("rep", 1) == rep]
                bad = [r for r in sel if r["picked"] == "markdown"]
                won.update(r["pair"] for r in bad)
                cells.append(f"{len(bad)}/{len(sel)}")
            print(f"| {judge} | {prompt} | {' | '.join(cells)} | {', '.join(sorted(won))} |")


if __name__ == "__main__":
    main()
