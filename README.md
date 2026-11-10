# Article experiments

Code and data behind articles from [AcruxCore](https://github.com/AcruxCore/AcruxCore)
that test claims about LLMs: a paper re-run on newer models, a benchmark, or a measurement.
Each folder holds one article's scripts, its inputs and the raw output of the run the
article reports, so every number in the article can be checked or re-run.

| Folder | Article | What it tests |
|---|---|---|
| [`llm-judge-markdown-bias`](llm-judge-markdown-bias) | [LLM-as-a-Judge Bias](https://talhaanwar.com/blog/llm-as-a-judge-bias/) (2026) | Whether five low-cost 2026 LLM judges prefer markdown answers, re-running the style-bias test from [Judging the Judges](https://arxiv.org/abs/2604.23178) |
| [`llm-judge-flip-rate`](llm-judge-flip-rate) | [LLM-as-a-Judge Consistency](https://talhaanwar.com/blog/llm-as-a-judge-consistency/) (2026) | How often the same five judges change their verdict on identical input, re-running [The Coin Flip Judge](https://arxiv.org/abs/2606.13685) across vendors, with swapped answer order |
| [`role-prompting`](role-prompting) | Does "You Are an Expert" Make an LLM More Accurate? (2026, not yet published) | Whether a role in the system prompt ("You are a lawyer.") changes accuracy on 300 MMLU questions, for seven 2026 models with reasoning on and off, re-running [Zheng et al. 2024](https://aclanthology.org/2024.findings-emnlp.888/) |

The scripts call model providers directly. The two LLM-as-a-judge folders need only an
[OpenRouter](https://openrouter.ai) key in `OPENROUTER_API_KEY`. `role-prompting` also
needs an [OpenCode Go](https://opencode.ai) key in `OPENCODE_KEY`.

Code is licensed under Apache 2.0. Data taken from other projects keeps its own
licence, which each folder's README names.
