"""Re-test the markdown style bias from "Judging the Judges" on current cheap judges.

The paper (Soumik, TMLR 2026, arXiv 2604.23178) found that LLM judges prefer a
markdown answer over a plain-prose answer with the same content. This script
asks five current low-cost judges the same question, with the paper's own test
pairs and judge prompts, and adds one test the paper does not run: the markdown
answer carries one planted factual error and the prose answer is correct.

The 50 test pairs are downloaded from the paper's repository (MIT licence),
pinned to one commit, so this run and the paper read the same data.

Usage:
    OPENROUTER_API_KEY=... python run_judges.py
    OPENROUTER_API_KEY=... python run_judges.py --judge openai/gpt-6-luna --clean-only
    OPENROUTER_API_KEY=... python run_judges.py --repeat 2 --judge google/gemini-3.5-flash-lite --prompt naive --experiment error

The second form re-asks one cell of the grid; each repeat is stored as its
own row with "rep" set, so the first run (rep 1) is never overwritten.

Every call is appended to ../data/results/raw.jsonl as soon as it returns, so
an interrupted run resumes where it stopped. The run stops by itself once the
spend reported by OpenRouter passes BUDGET_USD.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import httpx

DATASET_URL = (
    "https://raw.githubusercontent.com/sksoumik/judging-the-judges/"
    "6ff578c28366964af8e0558dba35930b1bd8a622/data/custom/controlled_pairs.jsonl"
)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_FILE = DATA_DIR / "cache" / "controlled_pairs.jsonl"
ERRORS_FILE = DATA_DIR / "planted_errors.json"
RESULTS_FILE = DATA_DIR / "results" / "raw.jsonl"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
BUDGET_USD = 2.8
CONCURRENCY = 8
TEMPERATURE = 0.1  # the paper's setting

# Five recent low-cost judges, one per vendor. Each runs with its default
# settings, so the ones that reason before answering do so here too.
JUDGES = [
    "anthropic/claude-haiku-5.5",
    "openai/gpt-6-luna",
    "deepseek/deepseek-v4.1-flash",
    "qwen/qwen3.8-flash",
    "google/gemini-3.5-flash-lite",
]

# The paper's three single-call prompts, copied word for word from
# src/judges/prompts.py at the pinned commit: the plain baseline (B0), the
# rubric that tells the judge to ignore formatting (S4), and the
# step-by-step prompt (S5).
PROMPTS = {
    "naive": """\
You are an expert evaluator. Given a question and two responses, determine which response is better.

Question: {question}

Response A: {response_a}

Response B: {response_b}

Evaluate both responses and provide your verdict. Output a JSON object:
{{
  "verdict": "A" or "B" or "tie",
  "score_a": <float 1-10>,
  "score_b": <float 1-10>,
  "reasoning": "<brief explanation>"
}}""",
    "rubric": """\
You are an expert evaluator. Evaluate two responses using the following rubric. Score each criterion from 1-5.

**Rubric:**
1. **Accuracy** (1-5): Is the information factually correct?
   - 1: Major factual errors  2: Some errors  3: Mostly correct  4: Minor issues only  5: Fully accurate
2. **Relevance** (1-5): Does it address the question directly?
   - 1: Off-topic  2: Partially relevant  3: Mostly relevant  4: Directly relevant  5: Perfectly targeted
3. **Completeness** (1-5): Does it cover all aspects of the question?
   - 1: Very incomplete  2: Missing major parts  3: Covers basics  4: Thorough  5: Comprehensive
4. **Clarity** (1-5): Is it well-organized and easy to understand?
   - 1: Confusing  2: Poorly organized  3: Adequate  4: Clear  5: Exceptionally clear
5. **Reasoning** (1-5): Does it show sound logical reasoning?
   - 1: Illogical  2: Weak reasoning  3: Adequate  4: Sound  5: Exceptional reasoning

**IMPORTANT:** Do NOT consider response length or formatting style. Focus only on the criteria above.

Question: {question}

Response A: {response_a}

Response B: {response_b}

Score each response on all 5 criteria, then determine the winner. Output JSON:
{{
  "scores_a": {{"accuracy": X, "relevance": X, "completeness": X, "clarity": X, "reasoning": X}},
  "scores_b": {{"accuracy": X, "relevance": X, "completeness": X, "clarity": X, "reasoning": X}},
  "total_a": X,
  "total_b": X,
  "verdict": "A" or "B" or "tie",
  "reasoning": "<brief explanation>"
}}""",
    "cot": """\
You are an expert evaluator. Analyze two responses step by step before making a judgment.

Question: {question}

Response A: {response_a}

Response B: {response_b}

Follow these steps EXACTLY:

Step 1: Identify the key requirements of the question.
Step 2: Evaluate Response A's strengths and weaknesses against those requirements.
Step 3: Evaluate Response B's strengths and weaknesses against those requirements.
Step 4: Compare the two responses directly on substance (ignore length and formatting).
Step 5: Provide your final verdict.

Output JSON:
{{
  "step1_requirements": "<key requirements>",
  "step2_analysis_a": "<Response A analysis>",
  "step3_analysis_b": "<Response B analysis>",
  "step4_comparison": "<direct comparison>",
  "verdict": "A" or "B" or "tie",
  "score_a": <float 1-10>,
  "score_b": <float 1-10>
}}""",
}

VERDICT_RE = re.compile(r'"verdict"\s*:\s*"(A|B|tie)"', re.IGNORECASE)


def load_pairs() -> list[dict]:
    """Return the paper's 50 style pairs, downloading them once."""
    if not CACHE_FILE.exists():
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        response = httpx.get(DATASET_URL, timeout=60)
        response.raise_for_status()
        CACHE_FILE.write_text(response.text)
    rows = [json.loads(line) for line in CACHE_FILE.read_text().splitlines() if line]
    return [
        {
            "id": r["id"],
            "category": r["metadata"]["category"],
            "question": r["question"],
            "markdown": r["response_a"],
            "prose": r["response_b"],
        }
        for r in rows
        if r.get("bias_type") == "style"
    ]


ENDINGS = (".", "!", "?", '"', ")", "*", "$", "|", "`")


def is_clean(pair: dict) -> bool:
    """Both answers end on a complete sentence and neither is under 75% of the other's word count."""
    words = [len(re.findall(r"[A-Za-z0-9]+", pair[k])) for k in ("markdown", "prose")]
    complete = all(pair[k].rstrip().endswith(ENDINGS) for k in ("markdown", "prose"))
    return complete and min(words) / max(words) >= 0.75


def build_tasks(pairs: list[dict]) -> list[dict]:
    """One task per (experiment, pair, order). The judge never sees which is which."""
    by_id = {p["id"]: p for p in pairs}
    tasks = []
    for p in pairs:
        for order in ("md_first", "prose_first"):
            tasks.append({"experiment": "style", "pair": p["id"], "order": order,
                          "question": p["question"], "markdown": p["markdown"], "prose": p["prose"]})
    for e in json.loads(ERRORS_FILE.read_text()):
        p = by_id[e["id"]]
        if p["markdown"].count(e["find"]) != 1:
            sys.exit(f"planted error for {e['id']} does not match exactly once")
        wrong = p["markdown"].replace(e["find"], e["replace"])
        for order in ("md_first", "prose_first"):
            tasks.append({"experiment": "error", "pair": p["id"], "order": order,
                          "question": p["question"], "markdown": wrong, "prose": p["prose"]})
    return tasks


def parse_verdict(text: str) -> str:
    """Last verdict in the reply wins; replies that never state one count as invalid."""
    matches = VERDICT_RE.findall(text or "")
    if not matches:
        return "invalid"
    v = matches[-1]
    return "tie" if v.lower() == "tie" else v.upper()


def task_key(judge: str, prompt: str, t: dict, rep: int = 1) -> str:
    key = f"{judge}|{prompt}|{t['experiment']}|{t['pair']}|{t['order']}"
    return key if rep == 1 else f"{key}|rep{rep}"


async def judge_once(client: httpx.AsyncClient, key: str, judge: str, prompt: str, t: dict, rep: int) -> dict:
    md_first = t["order"] == "md_first"
    response_a = t["markdown"] if md_first else t["prose"]
    response_b = t["prose"] if md_first else t["markdown"]
    content = PROMPTS[prompt].format(question=t["question"], response_a=response_a, response_b=response_b)
    body = {
        "model": judge,
        "temperature": TEMPERATURE,
        "max_tokens": 6000,
        "messages": [{"role": "user", "content": content}],
        "usage": {"include": True},
    }
    last_error = None
    for attempt in range(4):
        try:
            r = await client.post(OPENROUTER_URL, json=body)
            data = r.json()
            if r.status_code != 200 or "choices" not in data:
                raise RuntimeError(f"HTTP {r.status_code}: {json.dumps(data)[:300]}")
            text = data["choices"][0]["message"].get("content") or ""
            verdict = parse_verdict(text)
            picked = None
            if verdict in ("A", "B"):
                first = verdict == "A"
                picked = "markdown" if first == md_first else "prose"
            usage = data.get("usage", {})
            return {
                "key": key, "rep": rep, "judge": judge, "prompt": prompt,
                "experiment": t["experiment"], "pair": t["pair"], "order": t["order"],
                "verdict": verdict, "picked": picked or verdict,
                "answered_by": data.get("model"),
                "finish_reason": data["choices"][0].get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "cost_usd": usage.get("cost", 0.0),
                "reply_tail": text[-400:],
            }
        except (httpx.HTTPError, RuntimeError, json.JSONDecodeError, KeyError) as exc:
            last_error = str(exc)
            await asyncio.sleep(2 * (attempt + 1))
    return {"key": key, "rep": rep, "judge": judge, "prompt": prompt, "experiment": t["experiment"],
            "pair": t["pair"], "order": t["order"], "verdict": "error", "picked": "error",
            "error": last_error, "cost_usd": 0.0}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1, help="extra repeats of the selected cells")
    parser.add_argument("--judge", help="any OpenRouter model id; defaults to the five JUDGES")
    parser.add_argument("--prompt", choices=list(PROMPTS))
    parser.add_argument("--experiment", choices=["style", "error"])
    parser.add_argument("--clean-only", action="store_true", help="skip the 26 style pairs that are not clean")
    args = parser.parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("set OPENROUTER_API_KEY")
    pairs = load_pairs()
    tasks = build_tasks(pairs)
    if args.clean_only:
        clean = {p["id"] for p in pairs if is_clean(p)}
        tasks = [t for t in tasks if t["pair"] in clean]

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    done, spent = set(), 0.0
    if RESULTS_FILE.exists():
        for line in RESULTS_FILE.read_text().splitlines():
            row = json.loads(line)
            spent += row.get("cost_usd") or 0.0
            if row["verdict"] != "error":
                done.add(row["key"])

    judges = [args.judge] if args.judge else JUDGES
    prompts = [args.prompt] if args.prompt else list(PROMPTS)
    tasks = [t for t in tasks if not args.experiment or t["experiment"] == args.experiment]
    reps = range(1, 2) if args.repeat == 1 else range(2, args.repeat + 2)
    todo = [(task_key(j, p, t, r), j, p, t, r) for j in judges for p in prompts for t in tasks for r in reps]
    todo = [x for x in todo if x[0] not in done]
    print(f"{len(todo)} calls to make, ${spent:.4f} already spent", file=sys.stderr)

    sem = asyncio.Semaphore(CONCURRENCY)
    state = {"spent": spent, "n": 0}
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(headers=headers, timeout=180) as client:
        with RESULTS_FILE.open("a") as out:

            async def run(item):
                key, judge, prompt, t, rep = item
                async with sem:
                    if state["spent"] > BUDGET_USD:
                        return
                    row = await judge_once(client, key, judge, prompt, t, rep)
                state["spent"] += row.get("cost_usd") or 0.0
                state["n"] += 1
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                if state["n"] % 50 == 0:
                    print(f"{state['n']}/{len(todo)} done, ${state['spent']:.4f}", file=sys.stderr)

            await asyncio.gather(*(run(item) for item in todo))
    print(f"finished, total spend ${state['spent']:.4f}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
