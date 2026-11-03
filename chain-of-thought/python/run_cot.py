"""Re-test Sprague et al., "To CoT or not to CoT? Chain-of-thought helps mainly
on math and symbolic reasoning" (ICLR 2025, arXiv 2409.12183) on 2026 models.

and Liu et al., "Mind Your Step (by Step): Chain-of-Thought can Reduce
Performance on Tasks where Thinking Makes Humans Worse" (arXiv 2410.21333).

Sprague et al. found CoT helps mostly on math; Liu et al. found it hurts on an
artificial-grammar task, with o1-preview 36.3 points below GPT-4o. This script
asks one task from each (build_questions.py: 200 MATH, 200 grammar questions) to
2026 models with reasoning switched on and off, with four user prompts and no
system prompt:

  direct         the question, then "Only write the answer, with no explanation."
  direct_rerun   the same prompt in separate calls: the noise floor
  plain          the question, then only the answer-format line (what most people send)
  step_by_step   plain, then Kojima et al.'s "Let's think step by step."

Model setups (ARMS) are the ones from the role-prompting re-test, in three groups:
  think     the model reasons before answering
  off       the same model with reasoning switched off
  none      a model that does not reason at all

`--rebuild` runs each paper's own model with that paper's exact zero-shot
messages (direct and CoT), temperature 0, with the paper's token limits:
Llama-3.1-8B-Instruct on MATH (stored next to Sprague et al.'s answer for the
same question) and GPT-4o (2024-08-06) on the grammar task (Liu et al. report
87.5% direct, 64.4% CoT on all 4,400 questions).

Every call is appended to ../data/results/raw.jsonl as soon as it returns; a
stopped run resumes where it left off. Rows with status "error" are retried.

Usage:
    OPENCODE_KEY=... OPENROUTER_API_KEY=... python run_cot.py --limit 5 --runs 1   # smoke test
    python run_cot.py
    python run_cot.py --rebuild
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_FILE = DATA_DIR / "results" / "raw.jsonl"
REBUILD_FILE = DATA_DIR / "results" / "rebuild_raw.jsonl"

GO_URL = "https://opencode.ai/inference/go/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
USER_AGENT = "article-experiments/1.0"

CONDITIONS = ["direct", "direct_rerun", "plain", "step_by_step"]
KOJIMA = "Let's think step by step."  # Kojima et al. 2022, the zero-shot CoT trigger
# The direct line follows the paper's own direct prompt ("Only write the answer. Write the
# answer in the following format: "Answer: <your answer>"."), with "no explanation" added
# because 2026 models explain unless told not to.
DIRECT_LINE = "Only write the answer, with no explanation. Write it in the format \"Answer: {slot}\"."
FORMAT_LINE = "End your reply with \"Answer: {slot}\"."

# Go list prices per million tokens (https://opencode.ai/inference/go/v1/models,
# read 2026-10-31). DeepSeek doubles on weekdays 01-04 and 06-10 UTC.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "gpt-6-luna": (0.10, 0.50),
    "deepseek-v4.1-flash": (0.15, 0.60),
    "qwen3.8-flash": (0.15, 0.47),
    "glm-5.3-flash": (0.15, 0.50),
    "deepseek-v4-pro": (0.66, 1.98),
}
PEAK_PRICED = {"deepseek-v4.1-flash", "deepseek-v4-pro"}

# name -> backend, model, extra request fields, runs, max_tokens, group
ARMS = {
    "claude-haiku-5.5/think": ("go", "claude-haiku-5-5", {"reasoning_effort": "high"}, 3, 16000, "think"),
    "gpt-6-luna/think": ("go", "gpt-6-luna", {}, 3, 16000, "think"),
    "deepseek-v4.1-flash/think": ("go", "deepseek-v4.1-flash", {}, 3, 16000, "think"),
    "qwen3.8-flash/think": ("go", "qwen3.8-flash", {}, 3, 16000, "think"),
    "glm-5.3-flash/think": ("go", "glm-5.3-flash", {}, 3, 16000, "think"),
    "deepseek-v4-pro/think": ("go", "deepseek-v4-pro", {}, 2, 16000, "think"),
    "claude-haiku-5.5/off": ("go", "claude-haiku-5-5", {"reasoning_effort": "none"}, 3, 4096, "off"),
    "gpt-6-luna/off": ("go", "gpt-6-luna", {"reasoning_effort": "none"}, 3, 4096, "off"),
    "deepseek-v4.1-flash/off": ("go", "deepseek-v4.1-flash", {"thinking": {"type": "disabled"}}, 3, 4096, "off"),
    "qwen3.8-flash/off": ("go", "qwen3.8-flash", {"enable_thinking": False}, 3, 4096, "off"),
    "gemini-3.5-flash-lite/none": ("openrouter", "google/gemini-3.5-flash-lite",
                                   {"provider": {"order": ["Google AI Studio"], "allow_fallbacks": False}},
                                   3, 4096, "none"),
    "qwen2.5-72b/none": ("openrouter", "qwen/qwen-2.5-72b-instruct",
                         {"provider": {"order": ["DeepInfra"], "allow_fallbacks": False}}, 3, 4096, "none"),
}
# The paper ran Llama-3.1-8B-Instruct in bf16 with vLLM, temperature 0, top_p 1.
# CoreWeave is the bf16 endpoint on OpenRouter.
REBUILD_ARMS = {
    "math": ("llama-3.1-8b/rebuild", ("openrouter", "meta-llama/llama-3.1-8b-instruct",
             {"provider": {"order": ["CoreWeave"], "allow_fallbacks": False}, "top_p": 1.0, "seed": 42},
             1, None, "none")),
    # Liu et al. ran GPT-4o at temperature 0 in autumn 2024, when "gpt-4o" pointed to 2024-08-06.
    "agl": ("gpt-4o/rebuild", ("openrouter", "openai/gpt-4o-2024-08-06",
            {"provider": {"order": ["OpenAI"], "allow_fallbacks": False}}, 1, None, "none")),
}

TEMPERATURE = 1.0


# ---------- prompts ----------

def question_block(q: dict) -> str:
    if q["task"] == "agl":
        # Liu et al.'s question text, without their answer-format lines.
        return ("Consider the following letter strings:\n" + q["train"] + "\n"
                "These strings were generated according to a certain set of rules. "
                "Does the following string also follow the same set of rules?\n" + q["test_string"])
    return f"Problem: {q['question']}"


def slot(q: dict) -> str:
    return "<Yes or No>" if q["task"] == "agl" else "<number>"


def user_prompt(condition: str, q: dict) -> str:
    block = question_block(q)
    if condition in ("direct", "direct_rerun"):
        return f"{block}\n\n{DIRECT_LINE.format(slot=slot(q))}"
    tail = FORMAT_LINE.format(slot=slot(q))
    if condition == "plain":
        return f"{block}\n\n{tail}"
    if condition == "step_by_step":
        return f"{block}\n\n{tail}\n\n{KOJIMA}"
    raise ValueError(condition)


# ---------- answer parsing ----------

def parse_yes_no(text: str) -> str | None:
    """Yes or No from a reply, or None.

    Takes Liu et al.'s ```answer block if there is one (their extract_answer), else the
    last "Answer: Yes/No", else a reply that is only Yes or No.
    """
    t = (text or "").replace("*", "")
    m = re.findall(r"```answer\s*\n(.*?)\n?```", t, re.DOTALL | re.IGNORECASE)
    if m:
        w = m[-1].strip().strip(".").lower()
        return {"yes": "Yes", "no": "No"}.get(w)
    cues = re.findall(r"answer\s*(?:is)?\s*:?\s*\(?\s*(yes|no)\b", t, re.IGNORECASE)
    if cues:
        return cues[-1].capitalize()
    w = t.strip().strip(".!").lower()
    return {"yes": "Yes", "no": "No"}.get(w)


def last_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    k = j
    while k < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        k += 1
    return text[j:k - 1] if depth == 0 else None


def to_number(s: str) -> float | None:
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:,|!|;|\s)", "", s)
    s = s.replace("\\left", "").replace("\\right", "").replace("^\\circ", "").replace("^{\\circ}", "")
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s)  # "x = 4"
    s = s.rstrip(".").strip()
    s = re.sub(r"_\{?[a-zA-Z0-9]+\}?$", "", s)  # base subscript: 100100_b, 101_{2}
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # 1,000
    s = re.sub(r"\s*(?:degrees|units|cm|inches|dollars|cents|square units)$", "", s).strip()
    s = s.strip("$").strip()
    try:
        return float(s)
    except ValueError:
        return None


def parse_number(text: str) -> float | None:
    """The numeric answer a reply gives, or None.

    Takes the last "Answer: ..." line, else the last \\boxed{...}. Only plain numbers
    count (the math set holds only whole-number answers).
    """
    t = (text or "").replace("*", "")
    cues = re.findall(r"answer(?: is)?\s*:?\s*(.+)", t, re.IGNORECASE)
    for cand in reversed(cues):
        inner = last_boxed(cand)
        n = to_number(inner if inner is not None else cand.split("I hope")[0])
        if n is not None:
            return n
    inner = last_boxed(t)
    if inner is not None:
        return to_number(inner)
    stripped = t.strip()
    return to_number(stripped) if len(stripped) <= 12 else None


def grade(q: dict, text: str):
    if q["task"] == "agl":
        got = parse_yes_no(text)
        return got, got == q["answer_key"]
    got = parse_number(text)
    return got, (got is not None and abs(got - float(q["answer_key"])) < 1e-9)


# ---------- calls ----------

def is_peak(now: datetime) -> bool:
    return now.weekday() < 5 and (1 <= now.hour < 4 or 6 <= now.hour < 10)


def list_cost(model: str, prompt_tokens, completion_tokens, peak: bool) -> float | None:
    if model not in PRICES or prompt_tokens is None:
        return None
    pin, pout = PRICES[model]
    cost = (prompt_tokens * pin + (completion_tokens or 0) * pout) / 1e6
    return cost * (2 if peak and model in PEAK_PRICED else 1)


def task_key(arm, condition, qid, run) -> str:
    return f"{arm}|{condition}|{qid}|{run}"


async def call_chat(client, url, headers, model, extra, messages, max_tokens, temperature):
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, **extra}
    r = await client.post(url, json=body, headers=headers)
    data = r.json()
    if r.status_code != 200 or "choices" not in data:
        raise RuntimeError(f"HTTP {r.status_code}: {json.dumps(data)[:300]}")
    msg = data["choices"][0]["message"]
    usage = data.get("usage") or {}
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return {
        "content": msg.get("content") or "",
        "finish_reason": data["choices"][0].get("finish_reason"),
        "answered_by": data.get("model"),
        "provider": data.get("provider"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "reasoning_chars": len(reasoning),
        "reported_cost_usd": usage.get("cost"),
    }


async def ask(client, keys, arm_name, arm, condition, q, run, rebuild=False) -> dict:
    backend, model, extra, _, max_tokens, group = arm
    if rebuild:
        messages = q["paper"][f"{condition}_messages"]
        max_tokens = q["paper"][f"{condition}_max_tokens"]
        temperature = 0.0
    else:
        messages = [{"role": "user", "content": user_prompt(condition, q)}]
        temperature = TEMPERATURE
    prefill = messages[-1]["content"] if messages[-1]["role"] == "assistant" else ""
    base = {"key": task_key(arm_name, condition, q["qid"], run), "arm": arm_name, "group": group,
            "model": model, "condition": condition, "qid": q["qid"], "task": q["task"],
            "run": run, "answer_key": q["answer_key"], "temperature": temperature, "max_tokens": max_tokens,
            "request_extra": {k: v for k, v in extra.items() if k != "provider"}}
    if rebuild and f"{condition}_correct" in q["paper"]:
        base["paper_correct"] = q["paper"][f"{condition}_correct"]
        base["paper_answer"] = q["paper"][f"{condition}_answer"]
    else:
        base["user_prompt"] = messages[0]["content"]
    last_error = None
    attempt = 0
    while attempt < 6:
        attempt += 1
        started = datetime.now(timezone.utc)
        try:
            t0 = time.monotonic()
            if backend == "go":
                headers = {"Authorization": f"Bearer {keys['go']}", "x-opencode-session": str(uuid.uuid4()),
                           "User-Agent": USER_AGENT}
                out = await call_chat(client, GO_URL, headers, model, extra, messages, max_tokens, temperature)
            else:
                headers = {"Authorization": f"Bearer {keys['openrouter']}", "User-Agent": USER_AGENT}
                out = await call_chat(client, OPENROUTER_URL, headers, model, {**extra, "usage": {"include": True}},
                                      messages, max_tokens, temperature)
            graded_text = (prefill + " " + out["content"]) if prefill else out["content"]
            parsed, correct = grade(q, graded_text)
            cost = out["reported_cost_usd"]
            if cost is None or backend == "go":  # Go returns cost 0; use its list prices
                cost = list_cost(model, out["prompt_tokens"], out["completion_tokens"], is_peak(started))
            return base | out | {
                "status": "answered" if parsed is not None else "no_answer",
                "parsed_answer": parsed,
                "correct": correct,
                "cost_usd": cost,
                "peak_hour": is_peak(started),
                "latency_s": round(time.monotonic() - t0, 2),
                "timestamp": started.isoformat(timespec="seconds"),
                "content": out["content"],
                "content_chars": len(out["content"]),
            }
        except (httpx.HTTPError, RuntimeError, json.JSONDecodeError, KeyError, IndexError) as exc:
            last_error = str(exc)
            if "GoUsageLimitError" in last_error:
                # Go's 5-hour / weekly limit covers the whole workspace. Wait for it
                # to reset instead of logging every call as an error.
                attempt -= 1
                await asyncio.sleep(600)
                continue
            wait = 30 if "429" in last_error else 3 * attempt
            await asyncio.sleep(wait)
    return base | {"status": "error", "error": last_error, "cost_usd": 0.0}


def load_questions() -> list[dict]:
    return [json.loads(l) for l in (DATA_DIR / "questions.jsonl").read_text().splitlines()]


def spread(questions: list[dict], limit: int) -> list[dict]:
    """The first `limit` questions of each task, so a pilot covers both."""
    seen: dict[str, int] = {}
    out = []
    for q in questions:
        if seen.get(q["task"], 0) < limit:
            out.append(q)
            seen[q["task"]] = seen.get(q["task"], 0) + 1
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", choices=list(ARMS), help="repeatable; default all")
    parser.add_argument("--condition", action="append", choices=CONDITIONS)
    parser.add_argument("--task", action="append", choices=["math", "agl"])
    parser.add_argument("--limit", type=int, help="only the first N questions of each subset (smoke test)")
    parser.add_argument("--runs", type=int, help="override runs per cell")
    parser.add_argument("--concurrency", type=int, default=8, help="per arm")
    parser.add_argument("--rebuild", action="store_true", help="each paper's own model with its own prompts")
    parser.add_argument("--skip-peak", action="store_true", help="pause DeepSeek arms in peak-price hours")
    parser.add_argument("--out", type=Path, help="results file (default raw.jsonl / rebuild_raw.jsonl)")
    args = parser.parse_args()

    keys = {"go": os.environ.get("OPENCODE_KEY"),
            "openrouter": os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")}
    questions = load_questions()
    if args.task:
        questions = [q for q in questions if q["task"] in args.task]
    if args.limit:
        questions = spread(questions, args.limit)

    if args.rebuild:
        # each rebuild arm only gets its own paper's task
        arms = {REBUILD_ARMS[t][0]: REBUILD_ARMS[t][1] for t in (args.task or REBUILD_ARMS)}
        arm_task = {REBUILD_ARMS[t][0]: t for t in REBUILD_ARMS}
        conditions = ["direct", "cot"]
        out_file = args.out or REBUILD_FILE
    else:
        arms = {a: ARMS[a] for a in (args.arm or ARMS)}
        conditions = args.condition or CONDITIONS
        out_file = args.out or RESULTS_FILE
    for a in arms.values():
        if not keys[a[0]]:
            sys.exit(f"missing key for backend {a[0]}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            row = json.loads(line)
            if row["status"] != "error":
                done.add(row["key"])

    lock = asyncio.Lock()
    async with httpx.AsyncClient(timeout=900) as client:
        with out_file.open("a") as out:

            async def run_arm(name, arm):
                runs = args.runs or arm[3]
                qs = [q for q in questions if not args.rebuild or q["task"] == arm_task[name]]
                todo = [(c, q, r) for r in range(1, runs + 1) for q in qs for c in conditions
                        if task_key(name, c, q["qid"], r) not in done]
                print(f"{name}: {len(todo)} calls", file=sys.stderr)
                sem = asyncio.Semaphore(args.concurrency)
                stats = {"n": 0, "cost": 0.0, "err": 0}

                async def one(c, q, r):
                    async with sem:
                        while args.skip_peak and arm[1] in PEAK_PRICED and is_peak(datetime.now(timezone.utc)):
                            await asyncio.sleep(300)
                        row = await ask(client, keys, name, arm, c, q, r, rebuild=args.rebuild)
                    async with lock:
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out.flush()
                    stats["n"] += 1
                    stats["cost"] += row.get("cost_usd") or 0.0
                    stats["err"] += row["status"] == "error"
                    if stats["n"] % 500 == 0:
                        print(f"{name}: {stats['n']}/{len(todo)}, ${stats['cost']:.3f}, {stats['err']} errors",
                              file=sys.stderr)

                await asyncio.gather(*(one(c, q, r) for c, q, r in todo))
                print(f"{name}: finished {stats['n']} calls, ${stats['cost']:.3f}, {stats['err']} errors",
                      file=sys.stderr)

            await asyncio.gather(*(run_arm(n, a) for n, a in arms.items()))


if __name__ == "__main__":
    asyncio.run(main())
