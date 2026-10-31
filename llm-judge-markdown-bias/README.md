# LLM judge markdown bias

A re-test of the style-bias result in
[Judging the Judges](https://arxiv.org/abs/2604.23178) (Soumik, TMLR 2026) on five
current low-cost judges, plus one test the paper does not run: the markdown answer
carries a planted factual error and the plain-prose answer is correct.

The 50 test pairs are not copied here. `run_judges.py` downloads them from the
paper's [repository](https://github.com/sksoumik/judging-the-judges) (MIT licence)
at a pinned commit.

## Files

- `python/run_judges.py` sends every judge call through OpenRouter and appends
  each reply to `data/results/raw.jsonl`. A stopped run resumes where it left off.
- `python/analyze.py` turns the raw file into the tables in `data/results/summary.md`.
- `data/planted_errors.json` lists the one-fact change made to each of the 10
  markdown answers in the planted-error test.
- `data/results/raw.jsonl` is the raw output of the run behind the article: 1,880
  calls, run on 2026-10-31.
- `data/results/rerun-gpt-6-luna-clean.jsonl` is a second run of one judge on the
  clean pairs, kept to show run-to-run variation.

## Run it

```bash
pip install httpx
export OPENROUTER_API_KEY=...
cd python
python run_judges.py                       # 1,800 calls, about $1
python analyze.py > ../data/results/summary.md

# your own judge, clean pairs and planted errors only (204 calls)
python run_judges.py --judge <openrouter-model-id> --clean-only
```

## Licence

The scripts are Apache 2.0, like the rest of this repository. The 50 test pairs and
the answer text quoted in `data/planted_errors.json` come from the paper's repository
and stay under its MIT licence.
