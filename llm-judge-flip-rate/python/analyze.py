"""Turn ../data/results/raw.jsonl into the tables in ../data/results/summary.md.

Usage:
    python analyze.py > ../data/results/summary.md

Definitions follow the paper (Yagubyan 2026, section 3):
  flip rate of one (judge, pair) cell = share of valid verdicts that differ
  from that cell's majority verdict. Ties count as verdicts. Unparseable
  replies are dropped from the flip rate, as in the paper, and counted
  separately.

"Winner" is the model whose answer won: mini (GPT-4o-mini, the paper's
Response A) or 4o (GPT-4o, the paper's Response B), whatever slot it sat in.
"""

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "data" / "results" / "raw.jsonl"
SHORT = {
    "anthropic/claude-haiku-5.5": "Claude Haiku 5.5",
    "openai/gpt-6-luna": "GPT-6 Luna",
    "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
    "qwen/qwen3.8-flash": "Qwen3.8 Flash",
    "google/gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
}


def flip(verdicts):
    valid = [v for v in verdicts if v is not None]
    if len(valid) < 2:
        return None, None
    top, n = Counter(valid).most_common(1)[0]
    # a cell whose top two counts are equal has no single majority
    counts = Counter(valid).most_common()
    majority = top if len(counts) == 1 or counts[0][1] > counts[1][1] else "split"
    return 1 - n / len(valid), majority


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def main():
    rows = [json.loads(line) for line in RESULTS.read_text().splitlines()]
    ok = [r for r in rows if r["verdict"] != "error"]
    judges = [j for j in SHORT if any(r["judge"] == j for r in ok)]
    cells = defaultdict(list)  # (judge, experiment, order, pair) -> winners
    slots = defaultdict(list)  # same key -> slot verdicts
    for r in ok:
        k = (r["judge"], r["experiment"], r["order"], r["pair"])
        cells[k].append(r["winner"])
        slots[k].append(r["verdict"])
    pairs = sorted({r["pair"] for r in ok})
    category = {r["pair"]: r["category"] for r in ok}

    cost = sum(r.get("cost_usd") or 0 for r in rows)
    print(f"Calls with a reply: {len(ok)}. Failed attempts that were retried (mostly HTTP 429 from Qwen): "
          f"{len(rows) - len(ok)}. Replies with no verdict: {sum(r['verdict'] is None for r in ok)}. "
          f"Cost: ${cost:.2f}\n")
    reps = Counter((r["judge"], r["experiment"], r["order"], r["pair"]) for r in ok)
    print(f"Verdicts per cell: min {min(reps.values())}, max {max(reps.values())}\n")

    def fr_table(exp, order):
        out = {}
        for j in judges:
            frs = [flip(cells[(j, exp, order, p)])[0] for p in pairs]
            frs = [f for f in frs if f is not None]
            out[j] = frs
        return out

    # 1. flip rate, paper setting
    print("## 1. Flip rate, paper prompt, temperature 1.0, paper's order\n")
    print("| Judge | Mean flip rate | Pairs with 0% | Pairs over 20% | Max | Pairs where both answers won at least once |")
    print("|---|---|---|---|---|---|")
    t1 = fr_table("repeat", "original")
    for j in judges:
        frs = t1[j]
        both = sum(1 for p in pairs if {"mini", "4o"} <= set(cells[(j, "repeat", "original", p)]))
        print(f"| {SHORT[j]} | {pct(statistics.mean(frs))} | {sum(f == 0 for f in frs)}/{len(frs)} | "
              f"{sum(f > 0.2 for f in frs)}/{len(frs)} | {pct(max(frs))} | {both}/{len(pairs)} |")
    allfr = [f for j in judges for f in t1[j]]
    print(f"\nAll five judges: mean {pct(statistics.mean(allfr))}. Paper: 13.3% (GPT-4o-mini), 13.9% (GPT-4.1-mini).\n")

    print("Flip rate with the answers swapped (same prompt, temperature 1.0):\n")
    print("| Judge | Mean flip rate, swapped order |")
    print("|---|---|")
    ts = fr_table("repeat", "swapped")
    for j in judges:
        print(f"| {SHORT[j]} | {pct(statistics.mean(ts[j]))} |")
    print()

    # 2. ties
    print("## 2. Share of verdicts that are ties (paper prompt, t=1.0, both orders)\n")
    print("| Judge | Ties |")
    print("|---|---|")
    for j in judges:
        v = [r["verdict"] for r in ok if r["judge"] == j and r["experiment"] == "repeat" and r["verdict"]]
        print(f"| {SHORT[j]} | {pct(v.count('tie') / len(v))} |")
    print()

    # 3. position
    print("## 3. Position: the paper's 'A wins' index, then with the answers swapped\n")
    print("PBI = share of pairs whose majority verdict is slot A (paper's definition). "
          "In the paper's order slot A is GPT-4o-mini's answer; swapped, slot A is GPT-4o's.\n")
    print("| Judge | PBI, paper order | PBI, swapped | Slot A share of A/B verdicts, both orders | "
          "Pairs whose majority winner changes when swapped |")
    print("|---|---|---|---|---|")
    for j in judges:
        pbi = {}
        for order in ("original", "swapped"):
            maj = [flip(slots[(j, "repeat", order, p)])[1] for p in pairs]
            pbi[order] = sum(m == "A" for m in maj) / len(pairs)
        v = [r["verdict"] for r in ok if r["judge"] == j and r["experiment"] == "repeat" and r["verdict"] in ("A", "B")]
        changed = 0
        for p in pairs:
            a = flip(cells[(j, "repeat", "original", p)])[1]
            b = flip(cells[(j, "repeat", "swapped", p)])[1]
            changed += a != b
        print(f"| {SHORT[j]} | {pct(pbi['original'])} | {pct(pbi['swapped'])} | {pct(v.count('A') / len(v))} | "
              f"{changed}/{len(pairs)} |")
    print("\nPaper: PBI 72% (GPT-4o-mini), 59% (GPT-4.1-mini), paper order only.\n")

    print("Which model's answer wins the majority, by order (pairs out of 29):\n")
    print("| Judge | Order | GPT-4o-mini | GPT-4o | tie | no majority |")
    print("|---|---|---|---|---|---|")
    for j in judges:
        for order in ("original", "swapped"):
            c = Counter(flip(cells[(j, "repeat", order, p)])[1] for p in pairs)
            print(f"| {SHORT[j]} | {order} | {c['mini']} | {c['4o']} | {c['tie']} | {c['split']} |")
    print()

    # 4. temperature 0
    print("## 4. Temperature 0 (paper prompt, paper's order)\n")
    print("| Judge | Flip rate t=1.0 | Flip rate t=0 | Pairs with any flip at t=0 | Majority differs from t=1.0 |")
    print("|---|---|---|---|---|")
    t0 = fr_table("temp0", "original")
    for j in judges:
        nonzero = sum(f > 0 for f in t0[j])
        diff = sum(flip(cells[(j, "temp0", "original", p)])[1] != flip(cells[(j, "repeat", "original", p)])[1]
                   for p in pairs)
        print(f"| {SHORT[j]} | {pct(statistics.mean(t1[j]))} | {pct(statistics.mean(t0[j]))} | "
              f"{nonzero}/{len(t0[j])} | {diff}/{len(pairs)} |")
    print("\nPaper: 13.3% -> 2.8% (GPT-4o-mini), 13.9% -> 7.9% (GPT-4.1-mini).\n")

    # 5. prompt sensitivity
    print("## 5. Prompt wording (temperature 1.0, paper's order)\n")
    print("| Judge | Unparseable: paper prompt | Unparseable: prompt B | Unparseable: prompt B fixed | "
          "Majority differs, paper vs B | Majority differs, paper vs B fixed | Flip rate B | Flip rate B fixed |")
    print("|---|---|---|---|---|---|---|---|")
    for j in judges:
        un = {}
        for e in ("repeat", "prompt_b", "prompt_b_fixed"):
            v = [r["verdict"] for r in ok if r["judge"] == j and r["experiment"] == e and r["order"] == "original"]
            un[e] = f"{sum(x is None for x in v)}/{len(v)}"
        base = {p: flip(cells[(j, "repeat", "original", p)])[1] for p in pairs}
        d = {e: sum(flip(cells[(j, e, "original", p)])[1] != base[p] for p in pairs) for e in ("prompt_b", "prompt_b_fixed")}
        fb = statistics.mean(fr_table("prompt_b", "original")[j])
        ff = statistics.mean(fr_table("prompt_b_fixed", "original")[j])
        print(f"| {SHORT[j]} | {un['repeat']} | {un['prompt_b']} | {un['prompt_b_fixed']} | "
              f"{d['prompt_b']}/{len(pairs)} | {d['prompt_b_fixed']}/{len(pairs)} | {pct(fb)} | {pct(ff)} |")
    print("\nPaper: majority changed in 5 of 20 (judge, question) cases on a 10-question subset.\n")

    # 6. cross-judge
    print("## 6. Do the five judges agree with each other? (paper prompt, t=1.0, both orders pooled)\n")
    maj = {j: {p: flip(cells[(j, "repeat", "original", p)] + cells[(j, "repeat", "swapped", p)])[1] for p in pairs}
           for j in judges}
    print("| | " + " | ".join(SHORT[j] for j in judges) + " |")
    print("|---" * (len(judges) + 1) + "|")
    for a in judges:
        cellsrow = []
        for b in judges:
            cellsrow.append("–" if a == b else f"{sum(maj[a][p] == maj[b][p] for p in pairs)}/{len(pairs)}")
        print(f"| {SHORT[a]} | " + " | ".join(cellsrow) + " |")
    unanimous = sum(len({maj[j][p] for j in judges}) == 1 for p in pairs)
    print(f"\nPairs where all five judges give the same majority winner: {unanimous}/{len(pairs)}. "
          f"Paper: the two OpenAI judges agreed on 22 of 29.\n")

    # 7. by category
    print("## 7. Mean flip rate by category (paper prompt, t=1.0, paper's order, all judges)\n")
    print("| Category | Mean flip rate |")
    print("|---|---|")
    bycat = defaultdict(list)
    for j in judges:
        for p in pairs:
            f = flip(cells[(j, "repeat", "original", p)])[0]
            if f is not None:
                bycat[category[p]].append(f)
    for c, v in sorted(bycat.items(), key=lambda x: -statistics.mean(x[1])):
        print(f"| {c} | {pct(statistics.mean(v))} |")
    print()

    # 8. how often does one call match the 10-call majority
    print("## 8. One call vs the 10-call majority (paper prompt, t=1.0, paper's order)\n")
    print("| Judge | One call matches the 10-call majority |")
    print("|---|---|")
    for j in judges:
        hit = tot = 0
        for p in pairs:
            ws = [w for w in cells[(j, "repeat", "original", p)] if w is not None]
            hit += Counter(ws).most_common(1)[0][1]  # a 5-5 split counts 5, as in the flip rate
            tot += len(ws)
        print(f"| {SHORT[j]} | {pct(hit / tot)} |")
    print("\nPaper: 86.6% against the 50-call majority.\n")

    # 9. noise baseline for the prompt comparison
    print("## 9. Is the prompt effect bigger than re-running the same prompt?\n")
    print("Majorities from 5 calls each (pairs out of 29 whose majority winner differs). "
          "Baseline: repeats 1-5 vs repeats 6-10 of the paper prompt.\n")
    print("| Judge | Same prompt, two halves | Paper prompt vs prompt B | Paper prompt vs prompt B fixed |")
    print("|---|---|---|---|")
    by_rep = defaultdict(dict)
    for r in ok:
        by_rep[(r["judge"], r["experiment"], r["order"], r["pair"])][r["rep"]] = r["winner"]

    def maj5(j, e, p, reps):
        return flip([by_rep[(j, e, "original", p)].get(i) for i in reps])[1]

    first, second = range(1, 6), range(6, 11)
    for j in judges:
        half = sum(maj5(j, "repeat", p, first) != maj5(j, "repeat", p, second) for p in pairs)
        pb = sum(maj5(j, "repeat", p, first) != maj5(j, "prompt_b", p, first) for p in pairs)
        pf = sum(maj5(j, "repeat", p, first) != maj5(j, "prompt_b_fixed", p, first) for p in pairs)
        print(f"| {SHORT[j]} | {half}/29 | {pb}/29 | {pf}/29 |")
    print()

    # 10. what kind of flip
    print("## 10. What a flip is (paper prompt, t=1.0, paper's order)\n")
    print("Each verdict that differs from its cell's majority, split by whether a tie is involved.\n")
    print("| Judge | Non-majority verdicts | Involve a tie | Opposite winner |")
    print("|---|---|---|---|")
    for j in judges:
        tie = opp = 0
        for p in pairs:
            ws = [w for w in cells[(j, "repeat", "original", p)] if w is not None]
            m = Counter(ws).most_common(1)[0][0]  # for a 5-5 split, one side stands in as majority
            for w in ws:
                if w == m:
                    continue
                if m == "tie" or w == "tie":
                    tie += 1
                else:
                    opp += 1
        print(f"| {SHORT[j]} | {tie + opp} | {tie} | {opp} |")
    print()

    # 11. which answer wins once order is balanced
    print("## 11. Which model's answer wins, both orders pooled\n")
    print("| Judge | GPT-4o-mini share of non-tie verdicts |")
    print("|---|---|")
    for j in judges:
        w = [r["winner"] for r in ok if r["judge"] == j and r["experiment"] == "repeat" and r["winner"] in ("mini", "4o")]
        print(f"| {SHORT[j]} | {pct(w.count('mini') / len(w))} |")
    print()

    # 12. reasoning tokens
    print("## 12. Mean reasoning tokens per call\n")
    print("| Judge | t=1.0 | t=0 |")
    print("|---|---|---|")
    for j in judges:
        t = {e: statistics.mean([r.get("reasoning_tokens") or 0 for r in ok if r["judge"] == j and r["experiment"] == e])
             for e in ("repeat", "temp0")}
        print(f"| {SHORT[j]} | {t['repeat']:.0f} | {t['temp0']:.0f} |")
    print()

    # 13. parser check
    diff = sum(1 for r in ok if r.get("verdict_paper_parser") != r["verdict"])
    print(f"## 13. Parser check\n\nReplies where the paper's parser (first [[A]] wins) and ours (last verdict "
          f"wins) disagree: {diff}/{len(ok)}.\n")


if __name__ == "__main__":
    main()
