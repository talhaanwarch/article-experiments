# Chain of thought (work in progress)

**Status: pilot done, full run not started.** See the tracking issue for progress,
pricing and open decisions.

Does "Let's think step by step." help, do nothing, or hurt on 2026 models? Three
papers disagree:

- **Helps** on math and logic: Sprague et al.,
  [To CoT or not to CoT?](https://arxiv.org/abs/2409.12183) (ICLR 2025,
  [code](https://github.com/Zayne-sprague/To-CoT-or-not-to-CoT)).
- **Hurts** on tasks where thinking also hurts humans: Liu et al.,
  [Mind Your Step (by Step)](https://arxiv.org/abs/2410.21333)
  ([code](https://github.com/JiayiGeng/CoT_overthinking)).
- **Mostly obsolete:** Meincke et al. (Wharton),
  [The Decreasing Value of Chain of Thought in Prompting](https://arxiv.org/abs/2506.07142)
  (2025 preprint).

This folder asks one task from each of the first two papers (200 MATH questions,
200 artificial-grammar questions) to the same 2026 models with reasoning switched
on and off through the API. The comparison none of the papers could make is the same
model with reasoning on and off.

Prompts (user turn, no system prompt): `direct` (answer only), `direct_rerun` (the
noise floor), `plain` (answer-format line only) and `step_by_step` (`plain` +
"Let's think step by step.").

## Files

- `python/build_questions.py` builds `data/questions.jsonl`. MATH comes from the
  paper's published Llama-3.1-8B outputs on Hugging Face. The grammar questions come
  from the authors' `agl.json`, which is in their Google Drive folder:
  `gdown 1mbxqE2tD_uhNmkC-ExNizcNH9WXWmaIn -O data/download/agl.json`.
  Downloads go to `data/download/`, which git ignores.
- `python/run_cot.py` sends every call and appends each reply to a JSONL file. A
  stopped run resumes where it left off.
- `python/pilot_summary.py` prints accuracy, unparsed replies, tokens and cost for
  each setup, task and prompt.
- `data/results/pilot_raw.jsonl`: the pilot (Haiku 5.5 on and off, Gemini 3.5
  Flash-Lite; 30 questions per task, 1 run).
- `data/results/pilot_tokens.jsonl`: a token probe of the other setups (8 questions
  per task). The Qwen3.8 Flash and DeepSeek V4 Pro rows are incomplete: the probe
  hit its 20-minute limit.
- `data/results/pilot_rebuild.jsonl`: 10 questions per paper on the papers' own
  models (Llama-3.1-8B on MATH, GPT-4o on the grammar task).

## Run it

```bash
pip install httpx pandas pyarrow
cd python
python build_questions.py
export OPENCODE_KEY=... OPENROUTER_API_KEY=...
python run_cot.py --rebuild                                   # each paper's own model
python run_cot.py --arm claude-haiku-5.5/off --limit 30 --runs 1 --out ../data/results/test.jsonl
python pilot_summary.py ../data/results/test.jsonl
```

## Licence

The scripts are Apache 2.0, like the rest of this repository. The MATH questions are
from [MATH](https://github.com/hendrycks/math) (MIT licence). The grammar questions
are from the Mind Your Step authors' data. Their repository gives no licence, so check
with them before this folder is merged.
