"""Re-test Zheng et al., "When 'A Helpful Assistant' Is Not Really Helpful"
(Findings of EMNLP 2024, arXiv 2311.10054 v3) on 2026 models.

The paper puts a persona in the system prompt ("You are a/an {role}.") and asks
MMLU questions with "Reply with only the option number." It found that no
persona beat the no-persona control. This script asks the same questions in the
same format, with six conditions:

  no_role        no system message (the paper's control)
  no_role_rerun  the same, as a separate condition: the noise floor
  helpful        "You are a helpful assistant."
  in_domain      an expert whose field matches the question (lawyer for law, ...)
  out_domain     an expert from an unrelated field (chemist for law, ...)
  low_knowledge  "You are a 4-year-old toddler who thinks the moon is made of
                 cheese." (Basil et al. 2025's strongest low-knowledge persona,
                 a positive control: a test that cannot see this cannot see anything)

Every role except the toddler comes from the paper's role list
(data/role_info.csv), placed with the paper's template "You are a/an {role}."

Model setups (ARMS) fall into three groups:
  think     the model reasons before answering
  off       the same model with reasoning switched off
  none      a model that does not reason at all

`--rebuild` runs one of the paper's own models (Qwen2.5-72B-Instruct) with no
role on all 2,410 questions, once, with the paper's sampling settings, to compare
against the paper's 80.3% control accuracy.

Every call is appended to ../data/results/raw.jsonl as soon as it returns; a
stopped run resumes where it left off. Rows with status "error" are retried.

Usage:
    OPENCODE_KEY=... GEMINI_API_KEY=... OPENROUTER_API_KEY=... python run_roles.py
    python run_roles.py --arm qwen3.8-flash/think --limit 20     # smoke test
    python run_roles.py --rebuild
"""

import argparse
import asyncio
import csv
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
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
USER_AGENT = "article-experiments/1.0"

# The paper's user turn (llama3 template in scripts/vllm_inference_pipeline.py).
USER_TEMPLATE = ("The following is a multiple choice question (with answers). "
                 "Reply with only the option number. {question}")
ROLE_TEMPLATE = "You are {article} {role}."  # "You are {indefinite} {role}."

# In-domain expert per subject. All are in the paper's role list and tagged with
# the matching domain there (column occupation cate).
IN_DOMAIN = {
    "professional_law": "lawyer", "international_law": "lawyer",
    "clinical_knowledge": "doctor", "college_medicine": "doctor", "professional_medicine": "doctor",
    "electrical_engineering": "electrical engineer",
    "college_computer_science": "software engineer", "high_school_computer_science": "software engineer",
    "high_school_statistics": "statistician",
    "college_mathematics": "mathematician", "high_school_mathematics": "mathematician",
    "us_foreign_policy": "politician", "high_school_government_and_politics": "politician",
    "professional_psychology": "psychologist", "high_school_psychology": "psychologist",
    "college_physics": "physicist", "high_school_physics": "physicist",
    "college_chemistry": "chemist", "high_school_chemistry": "chemist",
    "college_biology": "biologist", "high_school_biology": "biologist",
    "management": "economist", "professional_accounting": "economist", "econometrics": "economist",
    "high_school_macroeconomics": "economist", "high_school_microeconomics": "economist",
}
# Out-of-domain expert per domain: an expert from a clearly unrelated domain.
OUT_DOMAIN = {
    "law": "chemist",
    "medicine": "lawyer",
    "eecs": "psychologist",
    "math": "politician",
    "politics": "mathematician",
    "psychology": "electrical engineer",
    "natural science": "lawyer",
    "econ": "biologist",
}
TODDLER = "You are a 4-year-old toddler who thinks the moon is made of cheese."
CONDITIONS = ["no_role", "no_role_rerun", "helpful", "in_domain", "out_domain", "low_knowledge"]

# Go list prices per million tokens (https://opencode.ai/inference/go/v1/models,
# read 2026-10-31). DeepSeek doubles on weekdays 01-04 and 06-10 UTC.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "gpt-6-luna": (0.10, 0.50),
    "deepseek-v4.1-flash": (0.15, 0.60),
    "qwen3.8-flash": (0.15, 0.47),
    "glm-5.3-flash": (0.15, 0.50),
    "deepseek-v4-pro": (0.66, 1.98),
    "gemini-3.5-flash-lite": (0.30, 2.50),  # OpenRouter list price; the free key returns no cost
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
    # With no setting Haiku still thinks on some questions (hidden, 50-240 tokens);
    # "none" stops it (3 output tokens on every smoke-test call).
    "claude-haiku-5.5/off": ("go", "claude-haiku-5-5", {"reasoning_effort": "none"}, 3, 1024, "off"),
    "gpt-6-luna/off": ("go", "gpt-6-luna", {"reasoning_effort": "none"}, 3, 1024, "off"),
    "deepseek-v4.1-flash/off": ("go", "deepseek-v4.1-flash", {"thinking": {"type": "disabled"}}, 3, 1024, "off"),
    "qwen3.8-flash/off": ("go", "qwen3.8-flash", {"enable_thinking": False}, 3, 1024, "off"),
    # No off setup for GLM-5.3 Flash: "thinking: disabled" still thinks on some
    # questions, and reasoning_effort "none" returns "[1210] This model always
    # engages in thinking".
    # The free Gemini API key managed about 11 calls a minute, so Flash-Lite runs
    # through OpenRouter, pinned to Google AI Studio. --gemini-direct sends it to
    # the Gemini API instead (GEMINI_API_KEY).
    "gemini-3.5-flash-lite/none": ("openrouter", "google/gemini-3.5-flash-lite",
                                   {"provider": {"order": ["Google AI Studio"], "allow_fallbacks": False}},
                                   3, 1024, "none"),
    "qwen2.5-72b/none": ("openrouter", "qwen/qwen-2.5-72b-instruct",
                         {"provider": {"order": ["DeepInfra"], "allow_fallbacks": False}}, 3, 1024, "none"),
}
# The paper's vLLM settings: SamplingParams(n=1, repetition_penalty=1.1, max_tokens=32),
# temperature left at vLLM's default of 1.0.
REBUILD_ARM = ("openrouter", "qwen/qwen-2.5-72b-instruct",
               {"provider": {"order": ["DeepInfra"], "allow_fallbacks": False},
                "repetition_penalty": 1.1}, 1, 32, "none")

TEMPERATURE = 1.0
NUM_RE = re.compile(r"(?<![\d.,/^])([1-4])(?![\d.,/]\d|\d)")
CUE_RE = re.compile(r"(?:option|answer|choice)(?:\s+(?:number|is|would be))*\s*[:=]?\s*\(?\*{0,2}([1-4])\b",
                    re.IGNORECASE)


def parse_option(text: str) -> int | None:
    """The option number the reply gives, or None if it gives none.

    Short replies ("4", "Option 4", "4.") take the only number in them. Longer
    replies take the last "option/answer is N", else a final line holding one
    number 1-4. Anything else is None and is counted as no answer.
    """
    t = (text or "").replace("*", "").strip()
    if not t:
        return None
    nums = NUM_RE.findall(t)
    if len(t) <= 20 and len(set(nums)) == 1:
        return int(nums[0])
    lead = re.match(r"\(?([1-4])[.):]\s", t)  # "2. Income is transferred ..."
    if lead:
        return int(lead.group(1))
    cues = CUE_RE.findall(t)
    if cues:
        return int(cues[-1])
    last = t.splitlines()[-1].strip()
    last_nums = NUM_RE.findall(last)
    if len(last) <= 20 and len(set(last_nums)) == 1:
        return int(last_nums[0])
    return None


def article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"  # the paper's get_indefinite_article


def system_prompt(condition: str, q: dict) -> str | None:
    if condition in ("no_role", "no_role_rerun"):
        return None
    if condition == "helpful":
        role = "helpful assistant"
    elif condition == "in_domain":
        role = IN_DOMAIN[q["subject"]]
    elif condition == "out_domain":
        role = OUT_DOMAIN[q["domain"]]
    elif condition == "low_knowledge":
        return TODDLER
    else:
        raise ValueError(condition)
    return ROLE_TEMPLATE.format(article=article(role), role=role)


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


async def call_chat(client, url, headers, model, extra, system, user, max_tokens):
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    body = {"model": model, "messages": messages, "temperature": TEMPERATURE, "max_tokens": max_tokens, **extra}
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


async def call_gemini(client, key, model, system, user, max_tokens):
    body = {"contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": TEMPERATURE, "maxOutputTokens": max_tokens}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    r = await client.post(GEMINI_URL.format(model=model), json=body, headers={"x-goog-api-key": key})
    data = r.json()
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {json.dumps(data)[:300]}")
    cand = (data.get("candidates") or [{}])[0]
    parts = (cand.get("content") or {}).get("parts") or []
    usage = data.get("usageMetadata") or {}
    return {
        "content": "".join(p.get("text", "") for p in parts if not p.get("thought")),
        "finish_reason": cand.get("finishReason"),
        "answered_by": data.get("modelVersion"),
        "provider": "google-ai-studio",
        "prompt_tokens": usage.get("promptTokenCount"),
        "completion_tokens": (usage.get("candidatesTokenCount") or 0) + (usage.get("thoughtsTokenCount") or 0),
        "reasoning_tokens": usage.get("thoughtsTokenCount") or 0,
        "reasoning_chars": 0,
        "reported_cost_usd": None,
    }


async def ask(client, keys, arm_name, arm, condition, q, run) -> dict:
    backend, model, extra, _, max_tokens, group = arm
    system = system_prompt(condition, q)
    user = USER_TEMPLATE.format(question=q["full_question"])
    base = {"key": task_key(arm_name, condition, q["qid"], run), "arm": arm_name, "group": group,
            "model": model, "condition": condition, "qid": q["qid"], "subject": q["subject"],
            "domain": q["domain"], "run": run, "true_option": int(q["true_option"]),
            "system_prompt": system, "temperature": TEMPERATURE, "max_tokens": max_tokens,
            "request_extra": {k: v for k, v in extra.items() if k != "provider"}}
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
                out = await call_chat(client, GO_URL, headers, model, extra, system, user, max_tokens)
            elif backend == "openrouter":
                headers = {"Authorization": f"Bearer {keys['openrouter']}", "User-Agent": USER_AGENT}
                out = await call_chat(client, OPENROUTER_URL, headers, model, {**extra, "usage": {"include": True}},
                                      system, user, max_tokens)
            else:
                out = await call_gemini(client, keys["gemini"], model, system, user, max_tokens)
            parsed = parse_option(out["content"])
            cost = out["reported_cost_usd"]
            if cost is None or backend == "go":  # Go returns cost 0; use its list prices
                cost = list_cost(model, out["prompt_tokens"], out["completion_tokens"], is_peak(started))
            return base | out | {
                "status": "answered" if parsed is not None else "no_answer",
                "parsed_option": parsed,
                "correct": parsed == int(q["true_option"]),
                "cost_usd": cost,
                "peak_hour": is_peak(started),
                "latency_s": round(time.monotonic() - t0, 2),
                "timestamp": started.isoformat(timespec="seconds"),
                "content": out["content"][:3000],
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


def load_questions(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", choices=list(ARMS), help="repeatable; default all")
    parser.add_argument("--condition", action="append", choices=CONDITIONS)
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    parser.add_argument("--runs", type=int, help="override runs per cell")
    parser.add_argument("--concurrency", type=int, default=8, help="per arm")
    parser.add_argument("--rebuild", action="store_true", help="the paper's Qwen2.5-72B, no role, 2,410 questions")
    parser.add_argument("--skip-peak", action="store_true", help="pause DeepSeek arms in peak-price hours")
    parser.add_argument("--gemini-direct", action="store_true",
                        help="send Gemini 3.5 Flash-Lite to the Gemini API instead of OpenRouter")
    parser.add_argument("--qwen-openrouter", action="store_true",
                        help="send Qwen3.8 Flash (reasoning on) to OpenRouter, pinned to Alibaba, instead of Go")
    args = parser.parse_args()
    if args.qwen_openrouter:
        # Same model and list price as on Go. Used to finish the last calls of this setup
        # after the Go workspace limit was reached; each row records its provider.
        ARMS["qwen3.8-flash/think"] = ("openrouter", "qwen/qwen3.8-flash",
                                       {"provider": {"order": ["Alibaba"], "allow_fallbacks": False},
                                        "reasoning": {"enabled": True}}, 3, 16000, "think")
    if args.gemini_direct:
        ARMS["gemini-3.5-flash-lite/none"] = ("gemini", "gemini-3.5-flash-lite", {}, 3, 1024, "none")

    keys = {"go": os.environ.get("OPENCODE_KEY"), "gemini": os.environ.get("GEMINI_API_KEY"),
            "openrouter": os.environ.get("OPENROUTER_API_KEY")}

    if args.rebuild:
        arms = {"qwen2.5-72b/rebuild": REBUILD_ARM}
        questions = load_questions(DATA_DIR / "questions_all.csv")
        conditions = ["no_role"]
        out_file = REBUILD_FILE
    else:
        arms = {a: ARMS[a] for a in (args.arm or ARMS)}
        questions = load_questions(DATA_DIR / "questions.csv")
        conditions = args.condition or CONDITIONS
        out_file = RESULTS_FILE
    if args.limit:
        questions = questions[: args.limit]
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
    async with httpx.AsyncClient(timeout=600) as client:
        with out_file.open("a") as out:

            async def run_arm(name, arm):
                runs = args.runs or arm[3]
                todo = [(c, q, r) for r in range(1, runs + 1) for q in questions for c in conditions
                        if task_key(name, c, q["qid"], r) not in done]
                print(f"{name}: {len(todo)} calls", file=sys.stderr)
                sem = asyncio.Semaphore(args.concurrency)
                stats = {"n": 0, "cost": 0.0, "err": 0}

                async def one(c, q, r):
                    async with sem:
                        while args.skip_peak and arm[1] in PEAK_PRICED and is_peak(datetime.now(timezone.utc)):
                            await asyncio.sleep(300)
                        row = await ask(client, keys, name, arm, c, q, r)
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
