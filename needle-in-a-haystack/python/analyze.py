"""Tables and hypothesis checks for the NoLiMa re-test (run_niah.py output).

Accuracy is NoLiMa's rule: the reply contains the right name. "Kept pairs" drops, per
setup, the main pairs the setup gets wrong at 2 or more of the 3 depths at 1K, so a
model that does not know the fact (that Dresden is in Saxony) is not counted as a
long-input failure. Differences come with a 95% range from a paired bootstrap over
pairs (2,000 resamples).

Usage:
    python analyze.py            # setups
    python analyze.py --rebuild  # H0
"""

import argparse
import collections
import json
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "data" / "results"
RNG = np.random.default_rng(0)


def load(*paths):
    rows = [json.loads(l) for path in paths if path.exists() for l in path.read_text().splitlines()]
    latest = {}
    for r in rows:
        if r["status"] != "error":
            latest[r["key"]] = r
    return list(latest.values())


def pct(x):
    return f"{100 * x:5.1f}" if x is not None else "    -"


def acc(rows):
    return sum(r["correct"] for r in rows) / len(rows) if rows else None


def pair_scores(rows):
    """pair -> mean correct over its depths"""
    d = collections.defaultdict(list)
    for r in rows:
        d[r["pair"]].append(r["correct"])
    return {p: sum(v) / len(v) for p, v in d.items()}


def paired_diff(a_rows, b_rows):
    """mean(b) - mean(a) over shared pairs, with a bootstrap 95% range."""
    a, b = pair_scores(a_rows), pair_scores(b_rows)
    common = sorted(set(a) & set(b))
    if not common:
        return None, None, None
    da = np.array([b[p] - a[p] for p in common])
    boots = [RNG.choice(da, len(da)).mean() for _ in range(2000)]
    return da.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def kept_pairs(rows):
    s = collections.defaultdict(int)
    for r in rows:
        if r["condition"] == "main" and r["length"] == 1000:
            s[r["pair"]] += (not r["correct"])
    return {p for p, wrong in s.items() if wrong < 2}


def sel(rows, cond=None, length=None, pairs=None, hop=None):
    out = rows
    if cond:
        out = [r for r in out if r["condition"] == cond]
    if length:
        out = [r for r in out if r["length"] == length]
    if pairs is not None:
        out = [r for r in out if r["pair"] in pairs]
    if hop == "one":
        out = [r for r in out if r["qtype"] == "onehop"]
    if hop == "two":
        out = [r for r in out if r["qtype"].startswith("twohop")]
    return out


def fmt_diff(d):
    m, lo, hi = d
    return "-" if m is None else f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"


def rebuild():
    rows = load(ROOT / "rebuild_raw.jsonl")
    print("H0 rebuild: Llama 3.3 70B, NoLiMa main set, 58 pairs x 3 depths\n")
    print("| Length | All | One-hop | Two-hop | n |")
    print("|---|---|---|---|---|")
    for L in (1000, 32000):
        r = sel(rows, "main", L)
        print(f"| {L // 1000}K | {pct(acc(r))} | {pct(acc(sel(r, hop='one')))} | {pct(acc(sel(r, hop='two')))} | {len(r)} |")
    a1, a32 = acc(sel(rows, "main", 1000)), acc(sel(rows, "main", 32000))
    print(f"\n32K / 1K = {a32 / a1:.2f} (gate: below 0.60). Paper: 97.3 base, 42.7 at 32K (0.44).")
    print("Gate", "PASSES: the drop reproduces." if a32 / a1 < 0.60 else "FAILS: stop and rethink.")
    print(f"Spent ${sum(r.get('cost_usd') or 0 for r in rows):.2f}")


def setups():
    rows = load(ROOT / "raw.jsonl", ROOT / "raw_gemini.jsonl", ROOT / "raw_gemini_rw.jsonl")
    arms = sorted({r["arm"] for r in rows}, key=lambda a: (a.split("/")[0], a.split("/")[1] != "think"))
    by = {a: [r for r in rows if r["arm"] == a] for a in arms}

    print("## Accuracy (%), all pairs\n")
    print("| Setup | Main 1K | Main 32K | 32K/1K | One-hop 1K→32K | Two-hop 1K→32K | Rerun 32K | Direct 1K | Direct 32K | Distractor 1K | Distractor 32K |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for a, r in by.items():
        m1, m32 = acc(sel(r, "main", 1000)), acc(sel(r, "main", 32000))
        print(f"| {a} | {pct(m1)} | {pct(m32)} | {m32 / m1 if m1 and m32 is not None else 0:.2f} | "
              f"{pct(acc(sel(r, 'main', 1000, hop='one')))}→{pct(acc(sel(r, 'main', 32000, hop='one')))} | "
              f"{pct(acc(sel(r, 'main', 1000, hop='two')))}→{pct(acc(sel(r, 'main', 32000, hop='two')))} | "
              f"{pct(acc(sel(r, 'main_rerun', 32000)))} | {pct(acc(sel(r, 'direct', 1000)))} | "
              f"{pct(acc(sel(r, 'direct', 32000)))} | {pct(acc(sel(r, 'distractor', 1000)))} | "
              f"{pct(acc(sel(r, 'distractor', 32000)))} |")

    print("\n## Kept pairs only (dropped: pairs the setup misses at 1K)\n")
    print("| Setup | Kept | Main 1K | Main 32K | 32K/1K | One-hop 32K | Two-hop 32K | 32K − 1K |")
    print("|---|---|---|---|---|---|---|---|")
    kept = {}
    for a, r in by.items():
        k = kept[a] = kept_pairs(r)
        m1, m32 = acc(sel(r, "main", 1000, k)), acc(sel(r, "main", 32000, k))
        print(f"| {a} | {len(k)} | {pct(m1)} | {pct(m32)} | {m32 / m1 if m1 and m32 is not None else 0:.2f} | "
              f"{pct(acc(sel(r, 'main', 32000, k, 'one')))} | {pct(acc(sel(r, 'main', 32000, k, 'two')))} | "
              f"{fmt_diff(paired_diff(sel(r, 'main', 1000, k), sel(r, 'main', 32000, k)))} |")

    print("\n## Hypotheses\n")
    # H1
    h1 = {a: acc(sel(r, "direct", 32000)) for a, r in by.items()}
    print("H1 direct (shares words) at 32K >= 95% on every setup:",
          all(v is not None and v >= 0.95 for v in h1.values()), {a: pct(v) for a, v in h1.items()})
    # H2
    ratios = {}
    for a, r in by.items():
        k = kept[a]
        m1, m32 = acc(sel(r, "main", 1000, k)), acc(sel(r, "main", 32000, k))
        if m1 and m32 is not None:
            ratios[a] = m32 / m1
    below = [a for a, v in ratios.items() if v < 0.85]
    print(f"H2 main 32K below 85% of 1K (kept pairs) on at least half the setups: {len(below)}/{len(ratios)} ->",
          len(below) >= len(ratios) / 2, below)
    # length vs no-shared-words split
    print("\nLength cost vs missing-shared-words cost at 32K (points, all pairs):")
    for a, r in by.items():
        d1, d32 = acc(sel(r, "direct", 1000)), acc(sel(r, "direct", 32000))
        m1, m32 = acc(sel(r, "main", 1000)), acc(sel(r, "main", 32000))
        if None in (d1, d32, m1, m32):
            continue
        print(f"  {a:30s} direct drop {100 * (d1 - d32):5.1f} | main drop {100 * (m1 - m32):5.1f} | "
              f"extra from no shared words {100 * ((m1 - m32) - (d1 - d32)):5.1f}")
    # H3
    print("\nH3 reasoning on minus off at 32K (kept pairs of the off setup), paired by pair:")
    wins = 0
    for base in sorted({a.split("/")[0] for a in arms}):
        on, off = by.get(base + "/think"), by.get(base + "/off")
        if not on or not off:
            continue
        k = kept[base + "/off"] & kept[base + "/think"]
        d = paired_diff(sel(off, "main", 32000, k), sel(on, "main", 32000, k))
        on32, on1 = acc(sel(on, "main", 32000, k)), acc(sel(on, "main", 1000, k))
        wins += d[0] is not None and d[0] >= 0.10
        print(f"  {base:22s} {fmt_diff(d)} | on 32K {pct(on32)} vs on 1K {pct(on1)}")
    print(f"  +10 points or more on {wins} of the on/off models (prediction: at least 2)")
    # H4
    print("\nH4 distractor minus main at 1K (all pairs), and the noise floor (rerun minus main, 32K, depth 50%):")
    for a, r in by.items():
        d = paired_diff(sel(r, "main", 1000), sel(r, "distractor", 1000))
        main50 = [x for x in sel(r, "main", 32000) if x["depth"] == 0.5]
        noise = paired_diff(main50, sel(r, "main_rerun", 32000))
        d32 = paired_diff([x for x in sel(r, "main", 32000) if x["depth"] == 0.5], sel(r, "distractor", 32000))
        print(f"  {a:30s} 1K {fmt_diff(d)} | 32K {fmt_diff(d32)} | noise {fmt_diff(noise)}")
    # H5
    print("\nH5 median output tokens (reasoning included), main condition, 1K vs 32K:")
    fewer = 0
    n = 0
    for a, r in by.items():
        if not a.endswith("/think"):
            continue
        t1 = st.median([x.get("completion_tokens") or 0 for x in sel(r, "main", 1000)] or [0])
        t32 = st.median([x.get("completion_tokens") or 0 for x in sel(r, "main", 32000)] or [0])
        right = [x.get("completion_tokens") or 0 for x in sel(r, "main", 32000) if x["correct"]]
        wrong = [x.get("completion_tokens") or 0 for x in sel(r, "main", 32000) if not x["correct"]]
        n += 1
        fewer += t32 < t1
        print(f"  {a:30s} 1K {t1:7.0f} | 32K {t32:7.0f} | 32K right {st.median(right) if right else 0:7.0f} "
              f"wrong {st.median(wrong) if wrong else 0:7.0f}")
    print(f"  fewer at 32K on {fewer} of {n} (prediction: at least 3)")

    print("\n## Cost and failures\n")
    print("| Setup | Calls | Spent $ | $ per correct (main 32K) | Cap hits | Empty | Says 'not mentioned' (main 32K) |")
    print("|---|---|---|---|---|---|---|")
    for a, r in by.items():
        m32 = sel(r, "main", 32000)
        c = sum(x.get("cost_usd") or 0 for x in m32)
        right = sum(x["correct"] for x in m32)
        none = sum(1 for x in m32 if not x["correct"] and any(
            w in (x.get("content") or "").lower() for w in ("not mention", "no character", "none", "does not", "doesn't", "not stated", "no mention", "not specified")))
        print(f"| {a} | {len(r)} | {sum(x.get('cost_usd') or 0 for x in r):.2f} | "
              f"{c / right if right else 0:.4f} | {sum(x['status'] == 'cap_hit' for x in r)} | "
              f"{sum(x['status'] == 'empty' for x in r)} | {none}/{len(m32) - right} |")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()
    rebuild() if args.rebuild else setups()
