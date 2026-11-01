"""Turn ../data/results/raw.jsonl into the tables in ../data/results/summary.md.

Usage:
    python analyze.py > ../data/results/summary.md

Definitions:
  accuracy      share of calls whose parsed option equals the answer key. A reply
                with no option ("no answer") counts as not correct, and is also
                reported on its own.
  effect        for each question, accuracy under the role (averaged over runs)
                minus accuracy with no role (averaged over runs), then averaged
                over questions. Reported in percentage points.
  95% CI        bootstrap over questions (10,000 resamples, seed 767 for each interval).
  noise floor   the same effect computed for "no role" against "no role" asked
                again in separate calls. Nothing changed between the two, so its
                size shows how far an effect can move by chance alone.
"""

import gzip
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "data" / "results"
B = 10_000
SEED = 767
PAPER_REBUILD = 0.802905  # Qwen2.5-72B control accuracy, plot.ipynb in the paper's repository

NAMES = {
    "claude-haiku-5.5/think": "Claude Haiku 5.5",
    "gpt-6-luna/think": "GPT-6 Luna",
    "deepseek-v4.1-flash/think": "DeepSeek V4.1 Flash",
    "qwen3.8-flash/think": "Qwen3.8 Flash",
    "glm-5.3-flash/think": "GLM-5.3 Flash",
    "deepseek-v4-pro/think": "DeepSeek V4 Pro",
    "claude-haiku-5.5/off": "Claude Haiku 5.5",
    "gpt-6-luna/off": "GPT-6 Luna",
    "deepseek-v4.1-flash/off": "DeepSeek V4.1 Flash",
    "qwen3.8-flash/off": "Qwen3.8 Flash",
    "gemini-3.5-flash-lite/none": "Gemini 3.5 Flash-Lite",
    "qwen2.5-72b/none": "Qwen2.5-72B (Zheng's model)",
}
GROUP_LABEL = {"think": "reasoning on", "off": "reasoning off", "none": "never reasons"}
ROLES = ["helpful", "in_domain", "out_domain", "low_knowledge"]
ROLE_LABEL = {
    "no_role_rerun": "no role, asked again (noise floor)",
    "helpful": "helpful assistant",
    "in_domain": "in-domain expert",
    "out_domain": "out-of-domain expert",
    "low_knowledge": "4-year-old toddler",
}


def pts(x):
    return f"{100 * x:+.1f}"


def pct(x):
    return f"{100 * x:.1f}%"


def boot_ci(diffs, rng=None):
    # A fresh generator per interval, so each interval is reproducible on its own and
    # does not depend on how many intervals were computed before it.
    rng = random.Random(SEED)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(B))
    return means[int(0.025 * B)], means[int(0.975 * B) - 1]


def fmt_effect(d, rng):
    m = sum(d) / len(d)
    lo, hi = boot_ci(d, rng)
    star = "" if lo <= 0 <= hi else " *"
    return m, lo, hi, f"{pts(m)} [{pts(lo)}, {pts(hi)}]{star}"


def main():
    rng = random.Random(SEED)
    raw = RESULTS / "raw.jsonl"
    if not raw.exists():  # the repository stores the run gzipped
        text = gzip.open(RESULTS / "raw.jsonl.gz", "rt").read()
    else:
        text = raw.read_text()
    rows = [json.loads(line) for line in text.splitlines()]
    ok = [r for r in rows if r["status"] != "error"]
    arms = [a for a in NAMES if any(r["arm"] == a for r in ok)]

    # per (arm, condition, qid): list of correct flags across runs
    cell = defaultdict(list)
    noans = defaultdict(list)
    for r in ok:
        cell[(r["arm"], r["condition"], r["qid"])].append(1.0 if r["correct"] else 0.0)
        noans[(r["arm"], r["condition"])].append(r["status"] == "no_answer")
    qids = sorted({r["qid"] for r in ok})
    domain = {r["qid"]: r["domain"] for r in ok}

    def qmean(arm, cond, q):
        v = cell.get((arm, cond, q))
        return sum(v) / len(v) if v else None

    def diffs(arm, cond, base="no_role", qs=None):
        out = []
        for q in qs or qids:
            a, b = qmean(arm, cond, q), qmean(arm, base, q)
            if a is not None and b is not None:
                out.append(a - b)
        return out

    cost = sum(r.get("cost_usd") or 0 for r in ok)
    print(f"Calls with a reply: {len(ok)}. Failed attempts, retried: {len(rows) - len(ok)}. "
          f"Cost at list prices: ${cost:.2f}.\n")

    print("## 1. Setups, calls and cost\n")
    print("Go does not report reasoning tokens for Claude Haiku 5.5; its output tokens include them, and "
          "the last column shows how often a reply came back with reasoning text.\n")
    print("| Model | Reasoning | Calls | Runs per cell | Mean output tokens | Mean reasoning tokens | Cost "
          "| Calls with reasoning text |")
    print("|---|---|---|---|---|---|---|---|")
    for a in arms:
        rs = [r for r in ok if r["arm"] == a]
        runs = Counter((r["condition"], r["qid"]) for r in rs)
        print(f"| {NAMES[a]} | {GROUP_LABEL[rs[0]['group']]} | {len(rs)} | {min(runs.values())}-{max(runs.values())} | "
              f"{statistics.mean(r['completion_tokens'] or 0 for r in rs):.0f} | "
              f"{statistics.mean(r['reasoning_tokens'] or 0 for r in rs):.0f} | "
              f"${sum(r.get('cost_usd') or 0 for r in rs):.2f} | "
              f"{pct(sum((r.get('reasoning_chars') or 0) > 0 for r in rs) / len(rs))} |")

    conds = ["no_role", "no_role_rerun"] + ROLES
    print("\n## 2. Accuracy by condition\n")
    print("| Model | Reasoning | " + " | ".join(["no role", "no role again", "helpful", "in-domain",
                                                "out-of-domain", "toddler"]) + " |")
    print("|---|---|" + "---|" * len(conds))
    for a in arms:
        accs = []
        for c in conds:
            v = [x for q in qids for x in cell.get((a, c, q), [])]
            accs.append(pct(sum(v) / len(v)) if v else "n/a")
        print(f"| {NAMES[a]} | {GROUP_LABEL[a.split('/')[1]]} | " + " | ".join(accs) + " |")

    print("\n## 3. Effect of each role against no role, in points [95% CI]\n")
    print("`*` = the interval excludes zero. The first column is the noise floor: no role against no role.\n")
    print("| Model | Reasoning | " + " | ".join(ROLE_LABEL[c] for c in ["no_role_rerun"] + ROLES) + " |")
    print("|---|---|" + "---|" * (len(ROLES) + 1))
    effects = {}
    for a in arms:
        cells_txt = []
        for c in ["no_role_rerun"] + ROLES:
            m, lo, hi, txt = fmt_effect(diffs(a, c), rng)
            effects[(a, c)] = (m, lo, hi)
            cells_txt.append(txt)
        print(f"| {NAMES[a]} | {GROUP_LABEL[a.split('/')[1]]} | " + " | ".join(cells_txt) + " |")

    print("\n## 4. Average effect by group (models weighted equally), in points\n")
    print("| Group | Models | " + " | ".join(ROLE_LABEL[c] for c in ["no_role_rerun"] + ROLES) + " |")
    print("|---|---|" + "---|" * (len(ROLES) + 1))
    groups = {
        "reasoning on, small": [a for a in arms if a.endswith("/think") and "v4-pro" not in a],
        "reasoning on, big (DeepSeek V4 Pro)": [a for a in arms if "v4-pro" in a],
        "reasoning off (same models)": [a for a in arms if a.endswith("/off")],
        "never reasons": [a for a in arms if a.endswith("/none")],
    }
    for g, members in groups.items():
        if not members:
            continue
        vals = [pts(statistics.mean(effects[(a, c)][0] for a in members)) for c in ["no_role_rerun"] + ROLES]
        print(f"| {g} | {len(members)} | " + " | ".join(vals) + " |")

    print("\n## 5. Effects that beat the noise floor\n")
    print("A role effect counts here only if its CI excludes zero AND its size is larger than the largest "
          "noise-floor gap seen on any setup.\n")
    floor = max(abs(effects[(a, "no_role_rerun")][0]) for a in arms)
    print(f"Largest noise-floor gap: {pts(floor)} points "
          f"(range {pts(min(effects[(a, 'no_role_rerun')][0] for a in arms))} to "
          f"{pts(max(effects[(a, 'no_role_rerun')][0] for a in arms))}).\n")
    hits = [(a, c) for a in arms for c in ROLES
            if not (effects[(a, c)][1] <= 0 <= effects[(a, c)][2]) and abs(effects[(a, c)][0]) > floor]
    for a, c in hits:
        print(f"- {NAMES[a]} ({GROUP_LABEL[a.split('/')[1]]}), {ROLE_LABEL[c]}: {pts(effects[(a, c)][0])} points")
    if not hits:
        print("- none")
    n_sig = sum(not (effects[(a, c)][1] <= 0 <= effects[(a, c)][2]) for a in arms for c in ROLES)
    n_floor_sig = sum(not (effects[(a, 'no_role_rerun')][1] <= 0 <= effects[(a, 'no_role_rerun')][2]) for a in arms)
    print(f"\nRole contrasts whose CI excludes zero: {n_sig} of {len(arms) * len(ROLES)}. "
          f"Noise-floor contrasts whose CI excludes zero: {n_floor_sig} of {len(arms)}.")

    print("\n## 6. Same model, reasoning on vs off: role effect in each, in points\n")
    print("| Model | Role | Effect, reasoning on | Effect, reasoning off | Off minus on [95% CI] |")
    print("|---|---|---|---|---|")
    for base in ["claude-haiku-5.5", "gpt-6-luna", "deepseek-v4.1-flash", "qwen3.8-flash"]:
        on, off = f"{base}/think", f"{base}/off"
        if on not in arms or off not in arms:
            continue
        for c in ["no_role_rerun"] + ROLES:
            d_on, d_off = diffs(on, c), diffs(off, c)
            if len(d_on) != len(d_off):
                continue
            dd = [y - x for x, y in zip(d_on, d_off)]
            print(f"| {NAMES[on]} | {ROLE_LABEL[c]} | {pts(effects[(on, c)][0])} | {pts(effects[(off, c)][0])} | "
                  f"{fmt_effect(dd, rng)[3]} |")
    print("\nNo-role accuracy, reasoning on vs off:\n")
    for base in ["claude-haiku-5.5", "gpt-6-luna", "deepseek-v4.1-flash", "qwen3.8-flash"]:
        on, off = f"{base}/think", f"{base}/off"
        if on in arms and off in arms:
            d = [qmean(on, "no_role", q) - qmean(off, "no_role", q) for q in qids
                 if qmean(on, "no_role", q) is not None and qmean(off, "no_role", q) is not None]
            print(f"- {NAMES[on]}: reasoning on minus off {fmt_effect(d, rng)[3]} points")

    print("\n## 7. In-domain minus out-of-domain expert, in points [95% CI]\n")
    print("Zheng et al. report +0.4 points for this contrast (mixed model over their models).\n")
    print("| Model | Reasoning | In-domain minus out-of-domain |")
    print("|---|---|---|")
    for a in arms:
        print(f"| {NAMES[a]} | {GROUP_LABEL[a.split('/')[1]]} | "
              f"{fmt_effect(diffs(a, 'in_domain', base='out_domain'), rng)[3]} |")

    print("\n## 8. In-domain expert effect by question domain, all setups pooled (models weighted equally)\n")
    print("| Domain | Questions | In-domain expert | Out-of-domain expert | Toddler |")
    print("|---|---|---|---|---|")
    for dname in sorted(set(domain.values())):
        qs = [q for q in qids if domain[q] == dname]
        vals = []
        for c in ["in_domain", "out_domain", "low_knowledge"]:
            per_arm = [statistics.mean(d) for a in arms if (d := diffs(a, c, qs=qs))]
            vals.append(pts(statistics.mean(per_arm)))
        print(f"| {dname} | {len(qs)} | " + " | ".join(vals) + " |")

    print("\n## 9. Replies with no option number (refusals and format failures)\n")
    print("| Model | Reasoning | " + " | ".join(["no role", "no role again", "helpful", "in-domain",
                                                "out-of-domain", "toddler"]) + " |")
    print("|---|---|" + "---|" * len(conds))
    for a in arms:
        vals = []
        for c in conds:
            v = noans[(a, c)]
            vals.append(f"{sum(v)}/{len(v)}")
        print(f"| {NAMES[a]} | {GROUP_LABEL[a.split('/')[1]]} | " + " | ".join(vals) + " |")
    examples = [r for r in ok if r["status"] == "no_answer"]
    if examples:
        print("\nUp to 15 examples:\n")
        for r in examples[:15]:
            text = r["content"].replace("\n", " ")[:200]
            print(f"- {NAMES[r['arm']]} ({r['arm'].split('/')[1]}), {r['condition']}: \"{text}\"")

    print("\n## 10. Output length by condition (mean completion tokens, reasoning included)\n")
    print("| Model | Reasoning | " + " | ".join(["no role", "no role again", "helpful", "in-domain",
                                                "out-of-domain", "toddler"]) + " |")
    print("|---|---|" + "---|" * len(conds))
    for a in arms:
        vals = []
        for c in conds:
            v = [r["completion_tokens"] or 0 for r in ok if r["arm"] == a and r["condition"] == c]
            vals.append(f"{statistics.mean(v):.0f}" if v else "n/a")
        print(f"| {NAMES[a]} | {GROUP_LABEL[a.split('/')[1]]} | " + " | ".join(vals) + " |")

    print("\n## 12. Toddler effect, strict vs lenient parsing, in points [95% CI]\n")
    print("Strict is the parser used everywhere above. Lenient also reads a no-answer reply that "
          "holds exactly one distinct number from 1 to 4 (\"Moon is yummy cheese! ... 3!\") as that "
          "option, in both the toddler and the no-role calls.\n")
    print("| Model | Reasoning | No-answer replies, toddler | Of those, one number 1-4 | That number is right "
          "| Toddler effect, strict | Toddler effect, lenient |")
    print("|---|---|---|---|---|---|---|")

    def lone(r):
        nums = set(re.findall(r"\b[1-4]\b", r["content"] or ""))
        return int(nums.pop()) if len(nums) == 1 else None

    lcell = defaultdict(list)
    for r in ok:
        c = bool(r["correct"])
        if r["status"] == "no_answer" and lone(r) is not None:
            c = lone(r) == int(r["true_option"])
        lcell[(r["arm"], r["condition"], r["qid"])].append(1.0 if c else 0.0)
    for a in arms:
        na = [r for r in ok if r["arm"] == a and r["condition"] == "low_knowledge" and r["status"] == "no_answer"]
        one = [r for r in na if lone(r) is not None]
        right = [r for r in one if lone(r) == int(r["true_option"])]
        ld = []
        for q in qids:
            x, y = lcell.get((a, "low_knowledge", q)), lcell.get((a, "no_role", q))
            if x and y:
                ld.append(sum(x) / len(x) - sum(y) / len(y))
        print(f"| {NAMES[a]} | {GROUP_LABEL[a.split('/')[1]]} | {len(na)} | {len(one)} | {len(right)} | "
              f"{fmt_effect(diffs(a, 'low_knowledge'), rng)[3]} | {fmt_effect(ld, rng)[3]} |")

    rb_file = RESULTS / "rebuild_raw.jsonl"
    if rb_file.exists():
        rb = [json.loads(line) for line in rb_file.read_text().splitlines()]
        rb = [r for r in rb if r["status"] != "error"]
        acc = sum(r["correct"] for r in rb) / len(rb)
        na = sum(r["status"] == "no_answer" for r in rb)
        print("\n## 11. Rebuild check: the paper's Qwen2.5-72B-Instruct, no role, all 2,410 questions\n")
        print(f"Our accuracy: {pct(acc)} on {len(rb)} questions ({na} with no option number). "
              f"Paper: {pct(PAPER_REBUILD)}. Gap: {pts(acc - PAPER_REBUILD)} points. "
              f"Cost: ${sum(r.get('cost_usd') or 0 for r in rb):.2f}.")


if __name__ == "__main__":
    main()
