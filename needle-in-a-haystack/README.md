# Needle in a haystack (NoLiMa)

A re-test of Modarressi et al.,
[NoLiMa: Long-Context Evaluation Beyond Literal Matching](https://arxiv.org/abs/2502.05167)
(ICML 2025, v3; [code](https://github.com/adobe-research/NoLiMa)). NoLiMa hides one
sentence in a book snippet and asks a question that shares no words with it:

- Needle: "Actually, Yuki lives next to the Semper Opera House."
- Question: "Which character has been to Dresden?"

The paper found that 11 of 13 models fell below half their short-input score at 32K
tokens. This folder asks NoLiMa's own questions to 2026 models, with reasoning switched
on and off on the same model.

## What is new here

1. **The same model with reasoning on and off**, with a 16,000-token cap (NoLiMa capped
   reasoning at 1,536 tokens and compared different models).
2. **Length against missing shared words.** NoLiMa's control set, where the question
   repeats the needle's words, is asked at 1K and 32K. Its drop is the cost of length.
   The extra drop on the main set is the cost of the question sharing no words.
3. **Reasoning tokens at 1K against 32K.**

## Design

| Condition | Needle set | Depths | 1K | 32K |
|---|---|---|---|---|
| `main` | `needle_set.json`, 58 pairs (one-hop and two-hop) | 20/50/80% | 174 | 174 |
| `main_rerun` | same, asked again: the noise floor | 50% | | 58 |
| `direct` | `needle_set_ONLYDirect.json`, 28 pairs, the question repeats the needle | 20/80% | 56 | 56 |
| `distractor` | `needle_set_w_Distractor.json`, 58 pairs | 20/50/80% at 1K, 50% at 32K | 174 | 58 |

That is 750 calls per setup. Everything else is NoLiMa's:

- **Prompt:** NoLiMa's task template, with "You are a helpful assistant" as the system
  prompt (NoLiMa's default).
- **Haystack:** `rand_shuffle/rand_book_1.txt`, with NoLiMa's needle and distractor
  placement code ported to `run_niah.py`.
- **Grading:** the reply contains the right name.
- **Lengths:** haystack size in o200k_base tokens. Every setup gets the same text and the
  same character name for the same test.

The setups run at temperature 1.0, so reasoning on and off differ only in the reasoning
setting.

| Setup | Provider | Reasoning setting |
|---|---|---|
| Claude Haiku 5.5 on / off | OpenCode Go | `reasoning_effort` medium / none |
| GPT-6 Luna on / off | Runware | `reasoning_effort` medium / none |
| DeepSeek V4.1 Flash on / off | Runware | `reasoning_effort` medium / none |
| Gemini 3.5 Flash-Lite on / off | Runware | `reasoning_effort` medium / default (no thinking) |
| GLM-5.3 Flash | Runware | default (always thinks) |
| Gemma 4 31B | Runware | `reasoning_effort` high |

Go does not report Haiku's reasoning tokens, so reasoning is compared by output tokens,
which include reasoning, for every setup.

**Rebuild check (H0):** Llama 3.3 70B on OpenRouter (DeepInfra, fp8; the paper ran bf16),
temperature 0, 192 tokens, main condition at 1K and 32K.

## Files

- `python/run_niah.py` downloads the NoLiMa data into `data/cache/` (Adobe Research
  License, non-commercial, so it is not committed) and runs the calls.
- `python/analyze.py` prints the tables and the hypothesis checks.
- `data/results/raw.jsonl` and `data/results/rebuild_raw.jsonl` hold every call.

## Run

```bash
python python/run_niah.py --rebuild --max-spend 2
python python/run_niah.py --arm gpt-6-luna/think --arm gpt-6-luna/off --max-spend 5
python python/analyze.py --rebuild
python python/analyze.py
```

## Results

Claude Haiku 5.5 with reasoning on has 612 of 750 calls (Go usage limit); its numbers
will change. The other nine setups are complete. Full tables: `python python/analyze.py`.

| Setup | Main 1K | Main 32K | 32K/1K (kept pairs) | Control 32K |
|---|---|---|---|---|
| DeepSeek V4.1 Flash on | 97.1 | 83.9 | 0.86 | 100 |
| GPT-6 Luna on | 97.7 | 79.3 | 0.82 | 100 |
| GLM-5.3 Flash | 98.9 | 80.5 | 0.81 | 100 |
| DeepSeek V4.1 Flash off | 86.2 | 69.5 | 0.82 | 100 |
| Claude Haiku 5.5 on (partial) | 91.4 | 57.5 | 0.63 | 100 |
| Gemma 4 31B | 96.6 | 59.8 | 0.62 | 100 |
| GPT-6 Luna off | 91.4 | 54.6 | 0.60 | 100 |
| Gemini 3.5 Flash-Lite on | 86.2 | 51.7 | 0.59 | 100 |
| Gemini 3.5 Flash-Lite off | 64.9 | 35.6 | 0.50 | 100 |
| Claude Haiku 5.5 off | 65.5 | 28.2 | 0.47 | 100 |
| Llama 3.3 70B (H0, temp 0) | 90.8 | 35.1 | 0.39 | |

- **H0 passes:** Llama keeps 0.39 of its 1K score (paper: 0.44).
- **H1 holds:** the control is 100% at 32K on every setup, so the drop is not length alone.
- **H2 holds:** 9 of 10 setups below 0.85.
- **H3 holds:** reasoning on minus off at 32K is +13.6 to +37.2 points on all 4 models,
  and every on setup stays below its 1K score.
- **H4 holds at 1K:** the distractor costs 5.2 to 31.0 points on 8 of 10 setups.
- **H5 fails:** median output tokens rise at 32K on all 6 reasoning setups; wrong answers
  use the most (DeepSeek V4.1 Flash: 209 right, 11,238 wrong).

Spend: $18.35 for 7,362 calls, plus $0.58 for H0. `raw_gemini.jsonl` is a first
Flash-Lite run on the free Gemini key that stopped at its quota; it is not analysed.
