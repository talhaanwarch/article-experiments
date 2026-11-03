"""Quick table for a pilot or partial run: accuracy, unparsed replies, tokens and
cost per setup, task and prompt. Usage: python pilot_summary.py ../data/results/pilot_raw.jsonl
"""

import json
import statistics
import sys
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1])]
cells = defaultdict(list)
for r in rows:
    cells[(r["arm"], r["task"], r["condition"])].append(r)

print(f"{'setup':28} {'task':5} {'prompt':13} {'n':>3} {'acc':>6} {'noans':>5} {'err':>3} "
      f"{'med out tok':>11} {'med reas tok':>12} {'med chars':>9} {'cost $':>7}")
for (arm, task, cond), rs in sorted(cells.items()):
    ok = [r for r in rs if r["status"] != "error"]
    acc = sum(r["correct"] for r in ok) / len(ok) if ok else float("nan")
    noans = sum(r["status"] == "no_answer" for r in ok)
    out_tok = statistics.median([r.get("completion_tokens") or 0 for r in ok]) if ok else 0
    reas = statistics.median([r.get("reasoning_tokens") or 0 for r in ok]) if ok else 0
    chars = statistics.median([r.get("content_chars") or 0 for r in ok]) if ok else 0
    cost = sum(r.get("cost_usd") or 0 for r in rs)
    extra = ""
    if "paper_correct" in rs[0]:
        extra = f"  paper {sum(r['paper_correct'] for r in rs) / len(rs):.0%}"
    print(f"{arm:28} {task:5} {cond:13} {len(rs):>3} {acc:>6.1%} {noans:>5} {len(rs) - len(ok):>3} "
          f"{out_tok:>11.0f} {reas:>12.0f} {chars:>9.0f} {cost:>7.3f}{extra}")
print(f"total cost ${sum(r.get('cost_usd') or 0 for r in rows):.3f}, {len(rows)} calls")
