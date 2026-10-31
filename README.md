# Article experiments

Code and data behind articles from [AcruxCore](https://github.com/AcruxCore/AcruxCore)
that test claims about LLMs: a paper re-run on newer models, a benchmark, or a measurement.
Each folder holds one article's scripts, its inputs and the raw output of the run the
article reports, so every number in the article can be checked or re-run.

| Folder | Article | What it tests |
|---|---|---|
| [`llm-judge-markdown-bias`](llm-judge-markdown-bias) | LLM judge bias (Medium, 2026) | Whether five low-cost 2026 LLM judges prefer markdown answers, re-running the style-bias test from [Judging the Judges](https://arxiv.org/abs/2604.23178) |

The scripts call model providers directly. Most need only an
[OpenRouter](https://openrouter.ai) key in `OPENROUTER_API_KEY`.

Code is licensed under Apache 2.0. Data taken from other projects keeps its own
licence, which each folder's README names.
