"""Wrong-answer rate by reasoning length. For each reasoning-on setup in the main test
(NoLiMa's prompt), sorts that setup's calls at one length by output tokens, splits them
into four equal groups, and reports the share of wrong answers in each group.

Usage: python analyze_reasoning_length.py
"""

import collections
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "data" / "results"
FILES = ["raw.jsonl", "raw_gemini_rw.jsonl"]


def load():
    out = []
    for f in FILES:
        for line in (RESULTS / f).read_text().splitlines():
            r = json.loads(line)
            if r["condition"] == "main" and r["status"] != "error" and r["arm"].endswith("/think"):
                out.append(r)
    return out


def main():
    rows = load()
    arms = sorted({r["arm"] for r in rows})
    for length in (1000, 32000):
        print(f"== {length // 1000}K")
        total = collections.Counter()
        for arm in arms:
            rs = sorted((r for r in rows if r["arm"] == arm and r["length"] == length and r["completion_tokens"]),
                        key=lambda r: r["completion_tokens"])
            n = len(rs)
            parts = []
            for i in range(4):
                b = rs[i * n // 4:(i + 1) * n // 4]
                wrong = sum(not r["correct"] for r in b)
                total[i, "w"] += wrong
                total[i, "n"] += len(b)
                parts.append(f"Q{i + 1} {b[0]['completion_tokens']}-{b[-1]['completion_tokens']} tok: "
                             f"{100 * wrong / len(b):.0f}% wrong")
            print(f"{arm:32} " + " | ".join(parts))
        print("all setups: " + " | ".join(
            f"Q{i + 1} {100 * total[i, 'w'] / total[i, 'n']:.0f}% ({total[i, 'w']}/{total[i, 'n']})" for i in range(4)))
        print()


if __name__ == "__main__":
    main()
