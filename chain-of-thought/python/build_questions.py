"""Build the two question sets for the chain-of-thought re-test (seed 767).

  math  200 MATH questions, levels 3-5, whole-number answers: the task where
        Sprague et al. ("To CoT or not to CoT?", ICLR 2025) found CoT helps most.
        Taken from the outputs they published for Llama-3.1-8B-Instruct, so the
        rebuild check can compare answers question by question.
  agl   200 artificial-grammar questions: the task where Liu et al. ("Mind Your
        Step (by Step)", arXiv 2410.21333) found CoT hurts. Taken from the
        authors' agl.json (100 grammars x 44 questions); 2 per grammar, one string
        that follows the grammar (answer Yes) and one that breaks it (answer No).

Only MATH rows where the paper ran both zero-shot settings are used.

Writes ../data/questions.jsonl. Each row holds the paper's own prompts for the
rebuild check.
"""

import hashlib
import json
import random
import sys
import urllib.request
from pathlib import Path

import pandas as pd

SEED = 767
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
CACHE = DATA / "download"

SPRAGUE = "TAUR-Lab/Taur_CoT_Analysis_Project___meta-llama__Meta-Llama-3.1-8B-Instruct"
SPRAGUE_SHA = "b52c0e3d7f8c1b7e6456a8f714d16dd529a328bf"  # dataset repo commit, read 2026-11-03
MATH_PARQUET = ("https://huggingface.co/datasets/" + SPRAGUE +
                "/resolve/refs%2Fconvert%2Fparquet/math/latest/0000.parquet")
NOT_RUN = "[invalid] - configuration was not run"

# agl.json is in the authors' Google Drive data folder linked from
# https://github.com/JiayiGeng/CoT_overthinking (commit ee9b9c5). Download it with
#   gdown 1mbxqE2tD_uhNmkC-ExNizcNH9WXWmaIn -O ../data/download/agl.json
AGL_SHA256 = "dc6c4d98a65b11b26f27bb9fec47d34ed7c22be79492fffa5bc2ff293c276a27"
# prompts/agl.yaml in the authors' repository, verbatim.
LIU_ZEROSHOT = """Consider the following letter strings:
{train_examples}
These strings were generated according to a certain set of rules. Does the following string also follow the same set of rules?
{test_example}
Please ONLY answer "Yes" or "No" in the following format:
```answer
YOUR_ANSWER
```"""
LIU_COT = """Consider the following letter strings:
{train_examples}
These strings were generated according to a certain set of rules. Does the following string also follow the same set of rules?
{test_example}
Please reason about your answer before answering "Yes" or "No" in the following format:
```reasoning
YOUR_REASONING
```
```answer
YOUR_ANSWER
```"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def math_rows(rng: random.Random) -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "math.parquet"
    if not path.exists():
        print("downloading math.parquet", file=sys.stderr)
        urllib.request.urlretrieve(MATH_PARQUET, path)
    print(f"math.parquet sha256 {sha256(path)}", file=sys.stderr)
    df = pd.read_parquet(path)
    both_ran = ~(df.zero_shot_direct_parsed_model_answer.astype(str).str.startswith(NOT_RUN)
                 | df.zero_shot_cot_parsed_model_answer.astype(str).str.startswith(NOT_RUN))
    level = df.additional_information.apply(lambda s: json.loads(s).get("level"))
    whole = df.answerKey.astype(str).str.fullmatch(r"-?\d+")
    df = df[both_ran & whole & level.isin(["Level 3", "Level 4", "Level 5"])]
    pool = []
    for idx, r in df.iterrows():
        info = json.loads(r.additional_information)
        pool.append({
            "qid": f"math/{idx}",
            "task": "math",
            "question": r.question,
            "answer_key": str(r.answerKey),
            "level": info.get("level"),
            "type": info.get("type"),
            "paper": {
                "direct_messages": json.loads(r.zero_shot_direct_messages),
                "cot_messages": json.loads(r.zero_shot_cot_messages),
                "direct_max_tokens": json.loads(r.zero_shot_direct_payload).get("max_tokens"),
                "cot_max_tokens": json.loads(r.zero_shot_cot_payload).get("max_tokens"),
                "direct_correct": bool(r.zero_shot_direct_is_correct),
                "cot_correct": bool(r.zero_shot_cot_is_correct),
                "direct_answer": r.zero_shot_direct_parsed_model_answer,
                "cot_answer": r.zero_shot_cot_parsed_model_answer,
            },
        })
    return rng.sample(pool, 200)


def agl_rows(rng: random.Random) -> list[dict]:
    path = CACHE / "agl.json"
    if not path.exists():
        sys.exit(f"missing {path}: see the comment above AGL_SHA256")
    if sha256(path) != AGL_SHA256:
        sys.exit("agl.json does not match the expected sha256")
    grammars = json.loads(path.read_text())
    out = []
    for g_idx, g in enumerate(grammars):
        train = "\n".join(g["train"])  # as in the authors' dataloading.py
        for kind, pool, answer in (("follows", g["test"], "Yes"), ("breaks", g["fakes"], "No")):
            i = rng.randrange(len(pool))
            s = pool[i]
            out.append({
                "qid": f"agl/{g_idx}/{kind}/{i}",
                "task": "agl",
                "grammar": g_idx,
                "train": train,
                "test_string": s,
                "answer_key": answer,
                "paper": {
                    "direct_messages": [{"role": "user", "content": LIU_ZEROSHOT.format(
                        train_examples=train, test_example=s)}],
                    "cot_messages": [{"role": "user", "content": LIU_COT.format(
                        train_examples=train, test_example=s)}],
                    "direct_max_tokens": 10,   # the paper's setting for zero-shot
                    "cot_max_tokens": 1000,    # and for CoT
                },
            })
    return out


def main() -> None:
    rng = random.Random(SEED)
    out = math_rows(rng) + agl_rows(rng)
    path = DATA / "questions.jsonl"
    with path.open("w") as f:
        for q in out:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    m = [q for q in out if q["task"] == "math"]
    a = [q for q in out if q["task"] == "agl"]
    print(f"wrote {len(out)} questions to {path}: math {len(m)}, agl {len(a)} "
          f"({sum(q['answer_key'] == 'Yes' for q in a)} Yes)", file=sys.stderr)
    d = sum(q["paper"]["direct_correct"] for q in m) / len(m)
    c = sum(q["paper"]["cot_correct"] for q in m) / len(m)
    print(f"  math: paper Llama-3.1-8B direct {d:.1%}, CoT {c:.1%}", file=sys.stderr)


if __name__ == "__main__":
    main()
