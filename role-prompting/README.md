# Role prompting

A re-test of Zheng et al.,
[When "A Helpful Assistant" Is Not Really Helpful: Personas in System Prompts Do Not Improve Performances of Large Language Models](https://aclanthology.org/2024.findings-emnlp.888/)
(Findings of EMNLP 2024, [arXiv 2311.10054](https://arxiv.org/abs/2311.10054) v3). The
paper put a persona in the system prompt ("You are a lawyer.") and asked 2,410 MMLU
questions. No persona beat the same prompt with no persona.

This folder asks 300 of the paper's questions, in the paper's exact format, to 2026
models, in three groups:

- **Reasoning on:** Claude Haiku 5.5, GPT-6 Luna, DeepSeek V4.1 Flash, Qwen3.8 Flash,
  GLM-5.3 Flash, and DeepSeek V4 Pro as the big model.
- **The same models with reasoning switched off:** Haiku 5.5, GPT-6 Luna, DeepSeek V4.1
  Flash, Qwen3.8 Flash. GLM-5.3 Flash cannot switch it off.
- **Never reasons:** Gemini 3.5 Flash-Lite and Qwen2.5-72B-Instruct, one of the paper's
  own models.

Six conditions, each asked 3 times per question (2 for DeepSeek V4 Pro):

| Condition | System prompt |
|---|---|
| `no_role` | none (the paper's control) |
| `no_role_rerun` | none, asked again in separate calls: the noise floor |
| `helpful` | You are a helpful assistant. |
| `in_domain` | an expert in the question's field, e.g. "You are a lawyer." for law |
| `out_domain` | an expert in an unrelated field, e.g. "You are a chemist." for law |
| `low_knowledge` | You are a 4-year-old toddler who thinks the moon is made of cheese. |

Every role except the toddler comes from the paper's role list. The toddler is
the strongest low-knowledge persona in [Basil et al. 2025](https://arxiv.org/abs/2512.05858),
used here as a positive control. The role maps are in `run_roles.py` (`IN_DOMAIN`,
`OUT_DOMAIN`).

A rebuild check runs Qwen2.5-72B-Instruct with no role on all 2,410 questions, once, with
the paper's sampling settings (temperature 1.0, repetition penalty 1.1, 32 tokens), and
compares the result with the paper's 80.3%.

## Files

- `python/build_questions.py` downloads the paper's question file at a pinned commit,
  removes duplicates as the paper's notebook does (2,457 → 2,410), and draws the
  300-question sample: 37 or 38 per domain, seed 767.
- `data/questions.csv` (the 300) and `data/questions_all.csv` (the 2,410). Question IDs
  are `subject/question_id` from the paper's file.
- `python/run_roles.py` sends every call and appends each reply to
  `data/results/raw.jsonl`. A stopped run resumes where it left off.
- `python/analyze.py` turns the raw file into `data/results/summary.md`.
- `data/results/raw.jsonl.gz`: one row per call, with the system prompt, the reply, the
  parsed option, token counts and cost.
- `data/results/rebuild_raw.jsonl`: the rebuild check.
- `data/results/smoke_raw.jsonl`: the smoke test (4 questions, 1 run), not used in the
  results.

## Providers

The Go models run through [OpenCode Go](https://opencode.ai). Each call sends its own
`x-opencode-session` ID and the User-Agent `article-experiments/1.0`. Go returns no
cost, so cost is computed from Go's list prices (in `run_roles.py`). Gemini 3.5
Flash-Lite and Qwen2.5-72B run through OpenRouter, pinned to one provider each, which
returns the cost of each call.

Qwen3.8 Flash with reasoning on ran its first 3,014 calls through Go. The last 2,386
(the rest of run 2 and all of run 3) went through OpenRouter's Alibaba endpoint, at the
same list price, after Go's usage limit was reached (`--qwen-openrouter`). Each row's
`provider` field shows which one answered; Go rows have none. All six conditions of a
question in a run went through the same provider.

The full run made 63,000 calls and cost $10.11 at list prices; the rebuild check cost
$0.09.

## Run it

```bash
pip install httpx
cd python
python build_questions.py
export OPENCODE_KEY=... OPENROUTER_API_KEY=...
python run_roles.py --arm qwen3.8-flash/think --limit 20 --runs 1   # small test
python run_roles.py                 # all setups
python run_roles.py --rebuild       # the paper's Qwen2.5-72B on 2,410 questions
python analyze.py > ../data/results/summary.md
```

The repository stores the run as `raw.jsonl.gz` (51 MB unpacked). `analyze.py` reads it
as it is. To resume or extend the run, unpack it first: `gunzip -k ../data/results/raw.jsonl.gz`.

`--arm` takes any name in `ARMS`, and can be repeated. To test your own model, add a
line to `ARMS`.

## Licence

The scripts are Apache 2.0, like the rest of this repository. The questions come from
[MMLU](https://github.com/hendrycks/test) (MIT licence) through the paper's
[repository](https://github.com/Jiaxin-Pei/Prompting-with-Social-Roles).
