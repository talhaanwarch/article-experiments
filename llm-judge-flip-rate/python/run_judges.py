"""Re-test "The Coin Flip Judge" (Yagubyan 2026, arXiv 2606.13685) on five current judges.

The paper asked two OpenAI judges the same pairwise question many times and
counted how often the verdict changed. Its authors say a cross-provider
replication is still missing. This script runs the paper's own 29 pairs and
judge prompts on five low-cost judges from five vendors, and adds two checks
the paper's published results do not contain:

  * swapped order: the paper always shows GPT-4o-mini's answer first, so its
    "position bias" cannot be told apart from a preference for that answer.
    Here every pair is also judged with the two answers swapped.
  * a fixed prompt B: the paper's second template labels the answers
    "Answer 1" and "Answer 2" but asks for [[A]] or [[B]]. Here the same
    template is also run with the labels made to match.

Experiments (10 repeats of every cell):
  repeat      paper prompt, temperature 1.0, original and swapped order
  temp0       paper prompt, temperature 0, original order
  prompt_b    the paper's prompt B as published, temperature 1.0, original order
  prompt_b_fixed  prompt B with "Answer A"/"Answer B" labels, same settings

The 29 pairs are downloaded from the paper's repository (MIT licence) at a
pinned commit. Every call is appended to ../data/results/raw.jsonl as soon as
it returns, so a stopped run resumes where it left off. The run stops by
itself once the spend reported by OpenRouter passes BUDGET_USD.

Usage:
    OPENROUTER_API_KEY=... python run_judges.py
    OPENROUTER_API_KEY=... python run_judges.py --judge <openrouter-model-id> --experiment repeat
    OPENROUTER_API_KEY=... python run_judges.py --judge qwen/qwen3.8-flash --concurrency 4

Rows with verdict "error" are not counted as done, so running the script again
retries them.
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
    "https://raw.githubusercontent.com/Abelo9996/llm-judge-consistency/"
    "067ef9e6a509f3da081d5ffb02cd2af7234629ab/data/eval_pairs.json"
)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_FILE = DATA_DIR / "cache" / "eval_pairs.json"
RESULTS_FILE = DATA_DIR / "results" / "raw.jsonl"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
BUDGET_USD = 10.0
CONCURRENCY = 64
REPEATS = 10

# The same five judges as part 1 of the series, one per vendor, each with its
# default reasoning settings.
JUDGES = [
    "anthropic/claude-haiku-5.5",
    "openai/gpt-6-luna",
    "deepseek/deepseek-v4.1-flash",
    "qwen/qwen3.8-flash",
    "google/gemini-3.5-flash-lite",
]

# Copied word for word from the paper's repository at the pinned commit:
# PAIRWISE_PROMPT in src/evaluators/judges.py (also PROMPT_A in
# run_prompt_var.py) and PROMPT_B in run_prompt_var.py.
PROMPT_PAPER = """You are an impartial judge. Compare the two responses below to the given question and decide which is better.

[Question]
{question}

[Response A]
{response_a}

[Response B]
{response_b}

Evaluate based on helpfulness, relevance, accuracy, depth, and clarity.
First provide a brief explanation, then output your verdict as exactly one of:
[[A]] if Response A is better
[[B]] if Response B is better
[[tie]] if they are equally good"""

PROMPT_B = """Please act as a fair and unbiased evaluator. You will be given a question and two candidate answers. Your task is to determine which answer better addresses the question.

Question: {question}

Answer 1:
{response_a}

Answer 2:
{response_b}

Consider accuracy, completeness, clarity, and overall quality. Think step by step, then conclude with your final judgment in this exact format:
Winner: [[A]] or [[B]] or [[tie]]"""

# Our change: the two labels match the verdict format. Nothing else differs.
PROMPT_B_FIXED = PROMPT_B.replace("Answer 1:", "Answer A:").replace("Answer 2:", "Answer B:")

# experiment -> (prompt, temperature, orders)
EXPERIMENTS = {
    "repeat": (PROMPT_PAPER, 1.0, ["original", "swapped"]),
    "temp0": (PROMPT_PAPER, 0.0, ["original"]),
    "prompt_b": (PROMPT_B, 1.0, ["original"]),
    "prompt_b_fixed": (PROMPT_B_FIXED, 1.0, ["original"]),
}

VERDICT_RE = re.compile(r"\[\[\s*(A|B|tie)\s*\]\]", re.IGNORECASE)


def load_pairs() -> list[dict]:
    if not CACHE_FILE.exists():
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        r = httpx.get(DATASET_URL, timeout=60)
        r.raise_for_status()
        CACHE_FILE.write_bytes(r.content)
    return json.loads(CACHE_FILE.read_text())


def parse_verdict(text: str) -> str | None:
    """The last [[A]], [[B]] or [[tie]] in the reply.

    The paper's parser checks for "[[A]]" anywhere first, so a reply that
    quotes the format before giving [[B]] counts as A. Taking the last match
    avoids that; parse_verdict_paper keeps the paper's rule for comparison.
    """
    found = VERDICT_RE.findall(text)
    if not found:
        return None
    v = found[-1]
    return "tie" if v.lower() == "tie" else v.upper()


def parse_verdict_paper(text: str) -> str | None:
    if "[[A]]" in text:
        return "A"
    if "[[B]]" in text:
        return "B"
    if "[[tie]]" in text.lower():
        return "tie"
    return None


def winner_of(verdict: str | None, order: str) -> str | None:
    """Map a slot verdict to the model whose answer won: mini (GPT-4o-mini) or 4o."""
    if verdict in (None, "tie"):
        return verdict
    first = "mini" if order == "original" else "4o"
    second = "4o" if order == "original" else "mini"
    return first if verdict == "A" else second


def task_key(judge, experiment, pair_id, order, rep) -> str:
    return f"{judge}|{experiment}|{pair_id}|{order}|{rep}"


async def judge_once(client, judge, experiment, pair, order, rep) -> dict:
    prompt, temperature, _ = EXPERIMENTS[experiment]
    a, b = pair["response_a"]["text"], pair["response_b"]["text"]
    if order == "swapped":
        a, b = b, a
    body = {
        "model": judge,
        "temperature": temperature,
        "max_tokens": 6000,
        "messages": [{"role": "user", "content": prompt.format(question=pair["question"], response_a=a, response_b=b)}],
        "usage": {"include": True},
    }
    base = {"key": task_key(judge, experiment, pair["id"], order, rep), "judge": judge,
            "experiment": experiment, "pair": pair["id"], "category": pair["category"],
            "order": order, "rep": rep, "temperature": temperature}
    last_error = None
    for attempt in range(4):
        try:
            r = await client.post(OPENROUTER_URL, json=body)
            data = r.json()
            if r.status_code != 200 or "choices" not in data:
                raise RuntimeError(f"HTTP {r.status_code}: {json.dumps(data)[:300]}")
            text = data["choices"][0]["message"].get("content") or ""
            verdict = parse_verdict(text)
            usage = data.get("usage", {})
            return base | {
                "verdict": verdict,
                "verdict_paper_parser": parse_verdict_paper(text),
                "winner": winner_of(verdict, order),
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
    return base | {"verdict": "error", "winner": "error", "error": last_error, "cost_usd": 0.0}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", help="any OpenRouter model id; defaults to the five JUDGES")
    parser.add_argument("--experiment", choices=list(EXPERIMENTS))
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY,
                        help="lower it if a provider returns HTTP 429")
    args = parser.parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("set OPENROUTER_API_KEY")
    pairs = load_pairs()

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    done, spent = set(), 0.0
    if RESULTS_FILE.exists():
        for line in RESULTS_FILE.read_text().splitlines():
            row = json.loads(line)
            spent += row.get("cost_usd") or 0.0
            if row["verdict"] != "error":
                done.add(row["key"])

    judges = [args.judge] if args.judge else JUDGES
    experiments = [args.experiment] if args.experiment else list(EXPERIMENTS)
    todo = [
        (j, e, p, o, r)
        for r in range(1, args.repeats + 1)
        for e in experiments
        for j in judges
        for p in pairs
        for o in EXPERIMENTS[e][2]
        if task_key(j, e, p["id"], o, r) not in done
    ]
    print(f"{len(todo)} calls to make, ${spent:.4f} already spent", file=sys.stderr)

    sem = asyncio.Semaphore(args.concurrency)
    state = {"spent": spent, "n": 0}
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(headers=headers, timeout=240) as client:
        with RESULTS_FILE.open("a") as out:

            async def run(item):
                async with sem:
                    if state["spent"] > BUDGET_USD:
                        return
                    row = await judge_once(client, *item)
                state["spent"] += row.get("cost_usd") or 0.0
                state["n"] += 1
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                if state["n"] % 200 == 0:
                    print(f"{state['n']}/{len(todo)} done, ${state['spent']:.4f}", file=sys.stderr)

            await asyncio.gather(*(run(item) for item in todo))
    print(f"finished, total spend ${state['spent']:.4f}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
