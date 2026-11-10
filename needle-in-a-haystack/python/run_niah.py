"""Re-test Modarressi et al., "NoLiMa: Long-Context Evaluation Beyond Literal Matching"
(ICML 2025, arXiv 2502.05167 v3) on 2026 models, with reasoning switched on and off.

NoLiMa hides one sentence (the needle) in a long book snippet and asks a question that
shares no words with it: needle "Actually, Yuki lives next to the Semper Opera House.",
question "Which character has been to Dresden?". The paper found that most models fall
below half their short-input score at 32K tokens.

This script uses NoLiMa's own needle set, haystack, prompt template, needle-placement
code and grading rule (the reply contains the right name). The needle set and haystack
are downloaded from Hugging Face into ../data/cache/ (Adobe Research License,
non-commercial, so they are not committed here).

Conditions (CONDITIONS below):
  main         NoLiMa's 58 question-needle pairs (one-hop and two-hop), needle at 20/50/80%
  main_rerun   main at 50% depth, asked again in separate calls: the noise floor (32K only)
  direct       NoLiMa's control set: the question repeats the needle's words
               ("Which character lives next to the Semper Opera House?"), 28 pairs, 20/80%
  distractor   main plus NoLiMa's distractor sentence, which shares the question's key word
               ("There was an article about Dresden in the daily newspaper.")

Lengths are haystack sizes in o200k_base tokens (GPT-4o's tokenizer; NoLiMa counted with
each model's own tokenizer). Every setup gets the same text for the same test.

`--rebuild` runs Llama 3.3 70B on OpenRouter with the paper's settings (temperature 0,
192 tokens) on the main condition at 1K and 32K: the H0 stop gate.

Every call is appended to ../data/results/raw.jsonl as soon as it returns; a stopped run
resumes where it left off. Rows with status "error" are retried.

Keys come from the environment: OPENCODE_KEY, OPENROUTER_KEY, RUNWARE_API_KEY, GEMINI_API_KEY.

Usage:
    python run_niah.py --rebuild --max-spend 2
    python run_niah.py --arm claude-haiku-5.5/think --limit 6 --max-spend 0.5   # pilot
    python run_niah.py --arm gpt-6-luna/off --max-spend 3
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import tiktoken

ROOT = Path(__file__).resolve().parent.parent / "data"
CACHE = ROOT / "cache"
RESULTS_FILE = ROOT / "results" / "raw.jsonl"
REBUILD_FILE = ROOT / "results" / "rebuild_raw.jsonl"

HF = "https://huggingface.co/datasets/amodaresi/NoLiMa/resolve/main/"
SOURCES = {
    "main": "needlesets/needle_set.json",
    "direct": "needlesets/needle_set_ONLYDirect.json",
    "distractor": "needlesets/needle_set_w_Distractor.json",
    "haystack": "haystack/rand_shuffle/rand_book_1.txt",
    "haystack_long": "haystack/rand_shuffle_long/rand_book_1.txt",
}

GO_URL = "https://opencode.ai/inference/go/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
RUNWARE_URL = "https://api.runware.ai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
USER_AGENT = "article-experiments/1.0"

# NoLiMa's run_tests.py: use_default_system_prompt is true in every config, and the
# connector's default is "You are a helpful assistant".
SYSTEM_PROMPT = "You are a helpful assistant"

# condition -> needle set, depths, lengths
CONDITIONS = {
    "main": ("main", [0.2, 0.5, 0.8], [1000, 32000]),
    "main_rerun": ("main", [0.5], [32000]),
    "direct": ("direct", [0.2, 0.8], [1000, 32000]),
    "distractor": ("distractor", None, None),  # depths depend on length, see DISTRACTOR_DEPTHS
}
DISTRACTOR_DEPTHS = {1000: [0.2, 0.5, 0.8], 32000: [0.5]}
LONG = 120000  # optional third length (--long), main condition only

# list prices per million tokens: (input, output, cached input)
PRICES = {
    ("go", "claude-haiku-5-5"): (0.10, 0.50, None),
    ("runware", "openai-gpt-6-luna"): (0.10, 0.50, None),
    ("runware", "deepseek-v4-1-flash"): (0.15, 0.60, 0.01),
    ("runware", "zai-glm-5-3-flash"): (0.15, 0.50, 0.03),
    ("runware", "google-gemma-4-31b"): (0.102, 0.297, 0.012),
    ("gemini", "gemini-3.5-flash-lite"): (0.0, 0.0, 0.0),  # free key
    ("runware", "google-gemini-3-5-flash-lite"): (0.30, 2.50, 0.03),
}

# name -> backend, model, extra request fields, max output tokens, temperature, group
REASONING_CAP = 16000
ARMS = {
    "claude-haiku-5.5/think": ("go", "claude-haiku-5-5", {"reasoning_effort": "medium"}, REASONING_CAP, 1.0, "think"),
    "claude-haiku-5.5/off": ("go", "claude-haiku-5-5", {"reasoning_effort": "none"}, 1024, 1.0, "off"),
    "gpt-6-luna/think": ("runware", "openai-gpt-6-luna", {"reasoning_effort": "medium"}, REASONING_CAP, 1.0, "think"),
    "gpt-6-luna/off": ("runware", "openai-gpt-6-luna", {"reasoning_effort": "none"}, 1024, 1.0, "off"),
    "deepseek-v4.1-flash/think": ("runware", "deepseek-v4-1-flash", {"reasoning_effort": "medium"}, REASONING_CAP, 1.0, "think"),
    "deepseek-v4.1-flash/off": ("runware", "deepseek-v4-1-flash", {"reasoning_effort": "none"}, 1024, 1.0, "off"),
    "gemini-3.5-flash-lite/think": ("gemini", "gemini-3.5-flash-lite", {"thinkingConfig": {"thinkingLevel": "medium", "includeThoughts": False}}, REASONING_CAP, 1.0, "think"),
    "gemini-3.5-flash-lite/off": ("gemini", "gemini-3.5-flash-lite", {}, 1024, 1.0, "off"),
    # The free Gemini key ran out of quota part-way, so Flash-Lite runs in full on Runware.
    # Runware: no reasoning field means no thinking; reasoning_effort medium thinks (checked 2026-11-10).
    "gemini-3.5-flash-lite-rw/think": ("runware", "google-gemini-3-5-flash-lite", {"reasoning_effort": "medium"}, REASONING_CAP, 1.0, "think"),
    "gemini-3.5-flash-lite-rw/off": ("runware", "google-gemini-3-5-flash-lite", {}, 1024, 1.0, "off"),
    "glm-5.3-flash/think": ("runware", "zai-glm-5-3-flash", {}, REASONING_CAP, 1.0, "think"),
    "gemma-4-31b/think": ("runware", "google-gemma-4-31b", {"reasoning_effort": "high"}, REASONING_CAP, 1.0, "think"),
}
# The paper ran Llama 3.3 70B with vLLM in bf16, temperature 0, top_p 1, 192 tokens.
# OpenRouter's only 16-bit host (CoreWeave) costs 7x more, so this uses DeepInfra's fp8.
REBUILD_ARM = ("llama-3.3-70b/rebuild",
               ("openrouter", "meta-llama/llama-3.3-70b-instruct",
                {"top_p": 1.0, "provider": {"order": ["DeepInfra"], "allow_fallbacks": False}}, 192, 0.0, "none"))


# ---------- data ----------

def fetch(name: str) -> Path:
    path = CACHE / SOURCES[name]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = httpx.get(HF + SOURCES[name], follow_redirects=True, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def needle_tests(set_name: str) -> list[dict]:
    """NoLiMa's run_tests.py, unchanged in logic: one test per (group, question type, test)."""
    out = []
    for exp in json.loads(fetch(set_name).read_text()):
        for qtype, question in exp["questions"].items():
            for tid, test in exp["tests"].items():
                needle, q, distractor = exp["needle"], question, None
                for i, arg in enumerate(test["input_args"]):
                    ph = "{" + str(i + 1) + "}"
                    q = q.replace(ph, arg)
                    needle = needle.replace(ph, arg)
                    if "distractors" in exp and ph in exp["distractors"][qtype]:
                        distractor = exp["distractors"][qtype].replace(ph, arg)
                out.append({"pair": f"{exp['id']}_{tid}_{qtype}", "group": exp["id"], "qtype": qtype,
                            "needle": needle, "question": q, "distractor": distractor,
                            "characters": exp["character_set"], "template": exp["task_template"]})
    return out


class Haystack:
    """NoLiMa's BookHaystack._generate_w_needle_placement and its distractor placement."""

    def __init__(self, path: Path):
        self.enc = tiktoken.get_encoding("o200k_base")
        self.text = path.read_text()
        self.ids = self.enc.encode(self.text)
        self.newlines = np.array([0] + [i for i, t in enumerate(self.ids) if "\n" in self.enc.decode([t])])

    def place(self, needle: str, length: int, depth: float) -> str:
        delta = self.newlines / (length + 1) - depth
        delta[delta < 0] = np.inf
        k = int(np.argmin(delta))
        start = self.newlines[k] - int(np.round(length * depth))
        end = start + length
        assert start >= 0 and end <= len(self.ids)
        pre = self.enc.decode(self.ids[start:self.newlines[k] + 1])
        pre = pre[:max(pre.rfind("\n"), 0)]
        post = self.enc.decode(self.ids[self.newlines[k] + 1:end])
        return pre + " " + needle + "\n" + post

    def place_with_distractor(self, needle: str, distractor: str, length: int, depth: float,
                              rng: np.random.Generator, free_zone: float = 0.2) -> str:
        left = max(0, depth - 2 * free_zone)
        right = max(0, 1 - (depth + 2 * free_zone))
        d = rng.uniform(0, left + right)
        d = depth + free_zone + (d - left) if d > left else d + free_zone
        with_d = self.place(distractor, length, d)
        pos = with_d.index(distractor)
        span = with_d[max(pos - 50, 0):pos - 1]
        text = self.place(needle, length, depth)
        if text.count(span) != 1:
            raise IndexError("distractor anchor not unique")
        loc = text.index(span) + 49
        return text[:loc] + " " + distractor + "\n" + text[loc:]


def seed_for(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def build_tests(lengths_override=None, long=False) -> list[dict]:
    """Every test, identical for every setup: same character, same text."""
    hay = Haystack(fetch("haystack"))
    hay_long = None
    tests = []
    for cond, (set_name, depths, lengths) in CONDITIONS.items():
        for t in needle_tests(set_name):
            lens = DISTRACTOR_DEPTHS.keys() if cond == "distractor" else lengths
            for length in lens:
                if lengths_override and length not in lengths_override:
                    continue
                for depth in (DISTRACTOR_DEPTHS[length] if cond == "distractor" else depths):
                    tests.append(make_test(hay, cond, t, length, depth))
    if long:
        hay_long = Haystack(fetch("haystack_long"))
        for t in needle_tests("main"):
            for depth in [0.2, 0.5, 0.8]:
                tests.append(make_test(hay_long, "main", t, LONG, depth))
    return tests


def make_test(hay: Haystack, cond: str, t: dict, length: int, depth: float) -> dict:
    # main_rerun reuses main's character so the two are the same prompt
    base_cond = "main" if cond == "main_rerun" else cond
    rng = np.random.default_rng(seed_for(base_cond, t["pair"], length, depth))
    char = str(rng.choice(t["characters"]))
    needle = t["needle"].replace("{CHAR}", char)
    if cond == "distractor":
        text = hay.place_with_distractor(needle, t["distractor"], length, depth, rng)
    else:
        text = hay.place(needle, length, depth)
    prompt = t["template"].format(haystack=text, question=t["question"])
    return {"id": f"{cond}|{t['pair']}|{length}|{depth}", "condition": cond, "pair": t["pair"],
            "group": t["group"], "qtype": t["qtype"], "length": length, "depth": depth,
            "answer": char, "question": t["question"], "needle": needle, "prompt": prompt}


# ---------- calls ----------

def list_cost(backend, model, prompt_tokens, completion_tokens, cached_tokens) -> float | None:
    if (backend, model) not in PRICES or prompt_tokens is None:
        return None
    pin, pout, pcache = PRICES[(backend, model)]
    cached = (cached_tokens or 0) if pcache is not None else 0
    return ((prompt_tokens - cached) * pin + cached * (pcache or 0) + (completion_tokens or 0) * pout) / 1e6


async def call_openai(client, url, headers, model, extra, prompt, max_tokens, temperature, token_field):
    body = {"model": model, "temperature": temperature, token_field: max_tokens, **extra,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]}
    r = await client.post(url, json=body, headers=headers)
    data = r.json()
    if r.status_code != 200 or "choices" not in data:
        raise RuntimeError(f"HTTP {r.status_code}: {json.dumps(data)[:300]}")
    msg = data["choices"][0]["message"]
    usage = data.get("usage") or {}
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    completion = usage.get("completion_tokens")
    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    # Runware's Gemini reports thought tokens only inside total_tokens.
    hidden = (usage.get("total_tokens") or 0) - (usage.get("prompt_tokens") or 0) - (completion or 0)
    if reasoning_tokens is None and hidden > 0 and "gemini" in model:
        reasoning_tokens = hidden
        completion = (completion or 0) + hidden
    return {
        "content": msg.get("content") or "",
        "finish_reason": data["choices"][0].get("finish_reason"),
        "answered_by": data.get("model"),
        "provider": data.get("provider"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion,
        "reasoning_tokens": reasoning_tokens,
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "reasoning_chars": len(reasoning),
        "reported_cost_usd": usage.get("cost"),
    }


async def call_gemini(client, key, model, extra, prompt, max_tokens, temperature):
    body = {"systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens, **extra}}
    r = await client.post(GEMINI_URL.format(model=model), json=body,
                          headers={"x-goog-api-key": key, "User-Agent": USER_AGENT})
    data = r.json()
    if r.status_code != 200 or "candidates" not in data:
        raise RuntimeError(f"HTTP {r.status_code}: {json.dumps(data)[:300]}")
    cand = data["candidates"][0]
    parts = (cand.get("content") or {}).get("parts") or []
    u = data.get("usageMetadata") or {}
    thoughts = u.get("thoughtsTokenCount") or 0
    return {
        "content": "".join(p.get("text", "") for p in parts if not p.get("thought")),
        "finish_reason": cand.get("finishReason"),
        "answered_by": data.get("modelVersion"),
        "provider": "google",
        "prompt_tokens": u.get("promptTokenCount"),
        "completion_tokens": (u.get("candidatesTokenCount") or 0) + thoughts,
        "reasoning_tokens": thoughts,
        "cached_tokens": u.get("cachedContentTokenCount"),
        "reasoning_chars": 0,
        "reported_cost_usd": None,
    }


async def ask(client, keys, arm_name, arm, test) -> dict:
    backend, model, extra, max_tokens, temperature, group = arm
    base = {"key": f"{arm_name}|{test['id']}", "arm": arm_name, "group": group, "model": model,
            "backend": backend, "temperature": temperature, "max_tokens": max_tokens,
            "request_extra": {k: v for k, v in extra.items() if k != "provider"},
            **{k: test[k] for k in ("id", "condition", "pair", "group", "qtype", "length", "depth",
                                    "answer", "question", "needle")}}
    base["group"] = group  # setup group, not needle group
    base["needle_group"] = test["group"]
    last_error = None
    attempt = 0
    while attempt < 8:
        attempt += 1
        started = datetime.now(timezone.utc)
        try:
            t0 = time.monotonic()
            if backend == "gemini":
                out = await call_gemini(client, keys["gemini"], model, extra, test["prompt"], max_tokens, temperature)
            elif backend == "go":
                headers = {"Authorization": f"Bearer {keys['go']}", "x-opencode-session": str(uuid.uuid4()),
                           "User-Agent": USER_AGENT}
                out = await call_openai(client, GO_URL, headers, model, extra, test["prompt"], max_tokens,
                                        temperature, "max_tokens")
            elif backend == "runware":
                headers = {"Authorization": f"Bearer {keys['runware']}", "User-Agent": USER_AGENT}
                out = await call_openai(client, RUNWARE_URL, headers, model, extra, test["prompt"], max_tokens,
                                        temperature, "max_completion_tokens")
            else:
                headers = {"Authorization": f"Bearer {keys['openrouter']}", "User-Agent": USER_AGENT}
                out = await call_openai(client, OPENROUTER_URL, headers, model, {**extra, "usage": {"include": True}},
                                        test["prompt"], max_tokens, temperature, "max_tokens")
            content = out["content"]
            # NoLiMa's "contains" metric: the reply contains the right name
            correct = test["answer"] in content
            if not content.strip():
                status = "cap_hit" if out["finish_reason"] in ("length", "MAX_TOKENS") else "empty"
            else:
                status = "answered"
            cost = out["reported_cost_usd"]
            if cost is None or backend == "go":
                cost = list_cost(backend, model, out["prompt_tokens"], out["completion_tokens"], out["cached_tokens"])
            return base | out | {"status": status, "correct": correct, "cost_usd": cost,
                                 "latency_s": round(time.monotonic() - t0, 2),
                                 "timestamp": started.isoformat(timespec="seconds")}
        except (httpx.HTTPError, RuntimeError, json.JSONDecodeError, KeyError, IndexError) as exc:
            last_error = str(exc)
            if "GoUsageLimitError" in last_error:
                attempt -= 1
                await asyncio.sleep(600)
                continue
            wait = 60 if ("429" in last_error or "RESOURCE_EXHAUSTED" in last_error) else 3 * attempt
            await asyncio.sleep(wait)
    return base | {"status": "error", "error": last_error, "cost_usd": 0.0}


def pick(tests, limit):
    """A pilot slice: the first `limit` pairs of each condition and length, all depths."""
    pairs = {}
    out = []
    for t in tests:
        k = (t["condition"], t["length"])
        seen = pairs.setdefault(k, [])
        if t["pair"] not in seen:
            if len(seen) >= limit:
                continue
            seen.append(t["pair"])
        out.append(t)
    return out


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", action="append", choices=list(ARMS), help="repeatable")
    p.add_argument("--condition", action="append", choices=list(CONDITIONS))
    p.add_argument("--length", action="append", type=int)
    p.add_argument("--long", action="store_true", help="only the 120K main tests")
    p.add_argument("--limit", type=int, help="pilot: first N pairs per condition and length")
    p.add_argument("--concurrency", type=int, default=6, help="per arm")
    p.add_argument("--rebuild", action="store_true", help="Llama 3.3 70B with the paper's settings (H0)")
    p.add_argument("--max-spend", type=float, help="stop starting new calls after this many dollars")
    p.add_argument("--dry", action="store_true", help="build the tests and print counts only")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    keys = {"go": os.environ.get("OPENCODE_KEY"),
            "openrouter": os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY"),
            "runware": os.environ.get("RUNWARE_API_KEY"),
            "gemini": os.environ.get("GEMINI_API_KEY")}

    if args.long:
        tests = [t for t in build_tests(long=True) if t["length"] == LONG]
    else:
        tests = build_tests(lengths_override=args.length)
    if args.rebuild:
        tests = [t for t in tests if t["condition"] == "main"]
        arms = dict([REBUILD_ARM])
        out_file = args.out or REBUILD_FILE
    else:
        arms = {a: ARMS[a] for a in (args.arm or [])}
        out_file = args.out or RESULTS_FILE
    if args.condition:
        tests = [t for t in tests if t["condition"] in args.condition]
    if args.limit:
        tests = pick(tests, args.limit)
    enc = tiktoken.get_encoding("o200k_base")
    counts = {}
    for t in tests:
        counts[(t["condition"], t["length"])] = counts.get((t["condition"], t["length"]), 0) + 1
    print("tests:", {f"{c}@{l}": n for (c, l), n in sorted(counts.items())}, file=sys.stderr)
    if args.dry:
        print("prompt tokens (o200k):", sum(len(enc.encode(t["prompt"])) for t in tests), file=sys.stderr)
        return
    if not arms:
        sys.exit("pick --arm or --rebuild")
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

    # Order by length, then depth, then pair, so calls that share a long prefix run close
    # together and the provider's prompt cache can hit.
    tests.sort(key=lambda t: (t["length"], t["condition"] == "distractor", t["depth"], t["pair"]))
    lock = asyncio.Lock()
    spent = {"total": 0.0, "stopped": False}
    async with httpx.AsyncClient(timeout=900) as client:
        with out_file.open("a") as out:
            async def run_arm(name, arm):
                todo = [t for t in tests if f"{name}|{t['id']}" not in done]
                print(f"{name}: {len(todo)} calls", file=sys.stderr)
                conc = 2 if arm[0] == "gemini" else args.concurrency
                sem = asyncio.Semaphore(conc)
                stats = {"n": 0, "cost": 0.0, "err": 0, "ok": 0}

                async def one(t):
                    async with sem:
                        if args.max_spend is not None and spent["total"] >= args.max_spend:
                            if not spent["stopped"]:
                                spent["stopped"] = True
                                print(f"spend cap ${args.max_spend:.2f} reached; rerun to resume", file=sys.stderr)
                            return
                        row = await ask(client, keys, name, arm, t)
                    async with lock:
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out.flush()
                    stats["n"] += 1
                    stats["cost"] += row.get("cost_usd") or 0.0
                    spent["total"] += row.get("cost_usd") or 0.0
                    stats["err"] += row["status"] == "error"
                    stats["ok"] += bool(row.get("correct"))
                    if stats["n"] % 50 == 0:
                        print(f"{name}: {stats['n']}/{len(todo)}, ${stats['cost']:.3f}, "
                              f"{stats['ok']} correct, {stats['err']} errors", file=sys.stderr, flush=True)

                await asyncio.gather(*(one(t) for t in todo))
                print(f"{name}: finished {stats['n']} calls, ${stats['cost']:.3f}, {stats['ok']} correct, "
                      f"{stats['err']} errors", file=sys.stderr, flush=True)

            await asyncio.gather(*(run_arm(n, a) for n, a in arms.items()))
    print(f"total spent this run: ${spent['total']:.3f}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
