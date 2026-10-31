# LLM judge flip rate

A cross-provider re-test of
[The Coin Flip Judge](https://arxiv.org/abs/2606.13685) (Yagubyan, 2026). The paper
asked two OpenAI judges the same pairwise question many times and counted how often
the verdict changed. This folder runs the paper's 29 pairs and judge prompts on five
current low-cost judges from five vendors, and adds two checks the paper's published
results do not contain:

- **Swapped order.** The paper always shows GPT-4o-mini's answer in slot A, so its
  first-position result cannot be told apart from a preference for that answer. Every
  pair is also judged here with the answers swapped.
- **A fixed prompt B.** The paper's second template names the answers "Answer 1" and
  "Answer 2" but asks for `[[A]]` or `[[B]]`. It is run here as published and with
  the labels made to match.

The 29 pairs are not copied here. `run_judges.py` downloads them from the paper's
[repository](https://github.com/Abelo9996/llm-judge-consistency) (MIT licence) at a
pinned commit.

## Files

- `python/run_judges.py` sends every judge call through OpenRouter and appends each
  reply to `data/results/raw.jsonl`. A stopped run resumes where it left off.
- `python/analyze.py` turns the raw file into the tables in `data/results/summary.md`.
  Its flip-rate function reproduces the paper's 13.3% and 13.9% from the paper's own
  raw results.
- `data/results/raw.jsonl` is the raw output of the run behind the article.

## Run it

```bash
pip install httpx
export OPENROUTER_API_KEY=...
cd python
python run_judges.py                       # 7,250 calls
python analyze.py > ../data/results/summary.md

# your own judge, one experiment
python run_judges.py --judge <openrouter-model-id> --experiment repeat
```

## Licence

The scripts are Apache 2.0, like the rest of this repository. The 29 test pairs come
from the paper's repository and stay under its MIT licence.
