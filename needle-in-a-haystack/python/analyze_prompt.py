"""Prompt follow-up: does NoLiMa's "Use the information provided in the book snippet"
make models answer "not mentioned"? Compares three prompts on the main condition:

  nolima         NoLiMa's template, unchanged (the original run; Haiku rerun on Runware)
  no_snippet     that sentence removed
  own_knowledge  that sentence plus "You can also use your own knowledge to connect the dots."

For each setup and prompt: accuracy at 1K and 32K, the kept-pairs ratio (pairs right at
2 or more of 3 depths at 1K; 32K accuracy over 1K accuracy on those pairs), and the share
of replies that say the answer is not in the text.

Usage: python analyze_prompt.py
"""

import collections
import json
import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "data" / "results"

# The original NoLiMa-prompt rows for each setup. Haiku's came from Go, so it was rerun on
# Runware with the NoLiMa prompt to keep the provider the same across the three prompts.
BASELINE = {
    "claude-haiku-5.5-rw/off": ("prompt_nolima_rw.jsonl", "claude-haiku-5.5-rw/off"),
    "claude-haiku-5.5-rw/think": ("prompt_nolima_rw.jsonl", "claude-haiku-5.5-rw/think"),
    "gemini-3.5-flash-lite-rw/off": ("raw_gemini_rw.jsonl", "gemini-3.5-flash-lite-rw/off"),
    "gemini-3.5-flash-lite-rw/think": ("raw_gemini_rw.jsonl", "gemini-3.5-flash-lite-rw/think"),
    "gemma-4-31b/think": ("raw.jsonl", "gemma-4-31b/think"),
}
VARIANTS = {"no_snippet": "prompt_no_snippet.jsonl", "own_knowledge": "prompt_own_knowledge.jsonl"}

REFUSAL = re.compile(
    r"not (explicitly )?(mention|stat|specif|provid|given|includ|contain)|no (character|information|mention)"
    r"|cannot be determined|can't be determined|unable to|does not (say|mention|indicate)|doesn't"
    r"|don't see|not enough|^none|unknown|insufficient", re.I)


def rows(file, arm):
    path = RESULTS / file
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["arm"] == arm and r["condition"] == "main" and r["status"] != "error":
            out.append(r)
    return out


def summary(rs):
    by = {(r["pair"], r["depth"], r["length"]): r for r in rs}
    acc = {}
    ref = {}
    n = {}
    for length in (1000, 32000):
        sel = [r for r in rs if r["length"] == length]
        n[length] = len(sel)
        acc[length] = 100 * sum(bool(r.get("correct")) for r in sel) / len(sel) if sel else float("nan")
        ref[length] = 100 * sum(bool(REFUSAL.search(r.get("content") or "")) for r in sel) / len(sel) if sel else float("nan")
    pairs = {r["pair"] for r in rs}
    kept = [p for p in pairs if sum(bool(by.get((p, d, 1000), {}).get("correct")) for d in (0.2, 0.5, 0.8)) >= 2]
    def kacc(length):
        sel = [by[(p, d, length)] for p in kept for d in (0.2, 0.5, 0.8) if (p, d, length) in by]
        return 100 * sum(bool(r.get("correct")) for r in sel) / len(sel) if sel else float("nan")
    k1, k32 = kacc(1000), kacc(32000)
    return {"n1": n[1000], "n32": n[32000], "acc1": acc[1000], "acc32": acc[32000],
            "ref1": ref[1000], "ref32": ref[32000], "kept": len(kept),
            "ratio": k32 / k1 if k1 else float("nan"), "kacc1": k1, "kacc32": k32}


def main():
    print(f"{'setup':32} {'prompt':14} {'n 1K/32K':>9} {'acc 1K':>7} {'acc 32K':>8} "
          f"{'kept':>5} {'kept 32K':>9} {'ratio':>6} {'refuse 1K':>10} {'refuse 32K':>11}")
    for arm, (bfile, barm) in BASELINE.items():
        for name, rs in [("nolima", rows(bfile, barm))] + [(v, rows(f, arm)) for v, f in VARIANTS.items()]:
            if not rs:
                print(f"{arm:32} {name:14} (no rows)")
                continue
            s = summary(rs)
            print(f"{arm:32} {name:14} {s['n1']:>4}/{s['n32']:<4} {s['acc1']:6.1f}% {s['acc32']:7.1f}% "
                  f"{s['kept']:>5} {s['kacc32']:8.1f}% {s['ratio']:6.2f} {s['ref1']:9.1f}% {s['ref32']:10.1f}%")
        print()


if __name__ == "__main__":
    main()
