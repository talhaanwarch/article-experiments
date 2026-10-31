Calls: 1800. Valid verdicts: 1799. Invalid or failed: 1. Cost of the first run: $1.0189. Cost including repeats: $1.0940

Clean pairs (24): style_001, style_005, style_007, style_008, style_009, style_011, style_013, style_020, style_021, style_022, style_024, style_025, style_026, style_028, style_029, style_032, style_036, style_040, style_041, style_042, style_043, style_045, style_046, style_048

## Style bias (markdown vs prose, same content)

| Judge | Prompt | All 50 pairs | 95% CI | Clean pairs | 95% CI | Markdown / prose / tie (clean) |
|---|---|---|---|---|---|---|
| anthropic/claude-haiku-5.5 | naive | +0.46 | +0.23 to +0.67 | +0.29 | -0.04 to +0.62 | 58% / 29% / 12% |
| anthropic/claude-haiku-5.5 | rubric | +0.32 | +0.14 to +0.49 | +0.10 | -0.06 to +0.25 | 19% / 8% / 73% |
| anthropic/claude-haiku-5.5 | cot | +0.28 | +0.11 to +0.45 | +0.12 | +0.00 to +0.27 | 15% / 2% / 83% |
| openai/gpt-6-luna | naive | +0.40 | +0.14 to +0.64 | +0.25 | -0.12 to +0.58 | 62% / 38% / 0% |
| openai/gpt-6-luna | rubric | +0.38 | +0.21 to +0.55 | +0.27 | +0.12 to +0.44 | 29% / 2% / 69% |
| openai/gpt-6-luna | cot | +0.25 | +0.09 to +0.41 | +0.08 | +0.00 to +0.19 | 8% / 0% / 92% |
| deepseek/deepseek-v4.1-flash | naive | +0.56 | +0.36 to +0.75 | +0.42 | +0.10 to +0.71 | 69% / 27% / 4% |
| deepseek/deepseek-v4.1-flash | rubric | +0.29 | +0.13 to +0.44 | +0.08 | +0.00 to +0.19 | 8% / 0% / 92% |
| deepseek/deepseek-v4.1-flash | cot | +0.27 | +0.09 to +0.45 | +0.04 | -0.15 to +0.24 | 17% / 13% / 70% |
| qwen/qwen3.8-flash | naive | +0.44 | +0.20 to +0.66 | +0.25 | -0.12 to +0.60 | 60% / 35% / 4% |
| qwen/qwen3.8-flash | rubric | +0.49 | +0.34 to +0.62 | +0.33 | +0.19 to +0.48 | 33% / 0% / 67% |
| qwen/qwen3.8-flash | cot | +0.29 | +0.10 to +0.48 | +0.08 | -0.12 to +0.29 | 19% / 10% / 71% |
| google/gemini-3.5-flash-lite | naive | +0.66 | +0.46 to +0.83 | +0.44 | +0.12 to +0.73 | 69% / 25% / 6% |
| google/gemini-3.5-flash-lite | rubric | +0.43 | +0.29 to +0.57 | +0.25 | +0.10 to +0.42 | 25% / 0% / 75% |
| google/gemini-3.5-flash-lite | cot | +0.57 | +0.40 to +0.73 | +0.33 | +0.04 to +0.60 | 58% / 25% / 17% |

## Style bias by pair group, plain prompt

| Judge | Clean (24) | Cut off, similar length (12) | Prose much shorter (14) |
|---|---|---|---|
| anthropic/claude-haiku-5.5 | +0.29 | +0.17 | +1.00 |
| openai/gpt-6-luna | +0.25 | +0.00 | +1.00 |
| deepseek/deepseek-v4.1-flash | +0.42 | +0.33 | +1.00 |
| qwen/qwen3.8-flash | +0.25 | +0.17 | +1.00 |
| google/gemini-3.5-flash-lite | +0.44 | +0.71 | +1.00 |
| all judges | +0.33 | +0.28 | +1.00 |

## Planted error (markdown answer wrong, prose answer right)

| Judge | Prompt | Picks correct prose | Picks wrong markdown | Tie | n |
|---|---|---|---|---|---|
| anthropic/claude-haiku-5.5 | naive | 100% | 0% | 0% | 20 |
| anthropic/claude-haiku-5.5 | rubric | 100% | 0% | 0% | 20 |
| anthropic/claude-haiku-5.5 | cot | 100% | 0% | 0% | 20 |
| openai/gpt-6-luna | naive | 100% | 0% | 0% | 20 |
| openai/gpt-6-luna | rubric | 100% | 0% | 0% | 20 |
| openai/gpt-6-luna | cot | 100% | 0% | 0% | 20 |
| deepseek/deepseek-v4.1-flash | naive | 100% | 0% | 0% | 20 |
| deepseek/deepseek-v4.1-flash | rubric | 100% | 0% | 0% | 20 |
| deepseek/deepseek-v4.1-flash | cot | 100% | 0% | 0% | 20 |
| qwen/qwen3.8-flash | naive | 100% | 0% | 0% | 20 |
| qwen/qwen3.8-flash | rubric | 100% | 0% | 0% | 20 |
| qwen/qwen3.8-flash | cot | 100% | 0% | 0% | 20 |
| google/gemini-3.5-flash-lite | naive | 70% | 30% | 0% | 20 |
| google/gemini-3.5-flash-lite | rubric | 80% | 5% | 15% | 20 |
| google/gemini-3.5-flash-lite | cot | 100% | 0% | 0% | 20 |

## First-slot preference on the style pairs (verdict A minus verdict B)

| Judge | Prompt | A minus B | Tie rate |
|---|---|---|---|
| anthropic/claude-haiku-5.5 | naive | +0.04 | 6% |
| anthropic/claude-haiku-5.5 | rubric | +0.06 | 44% |
| anthropic/claude-haiku-5.5 | cot | +0.00 | 48% |
| openai/gpt-6-luna | naive | +0.00 | 0% |
| openai/gpt-6-luna | rubric | -0.08 | 38% |
| openai/gpt-6-luna | cot | -0.01 | 55% |
| deepseek/deepseek-v4.1-flash | naive | +0.14 | 2% |
| deepseek/deepseek-v4.1-flash | rubric | +0.03 | 57% |
| deepseek/deepseek-v4.1-flash | cot | +0.01 | 44% |
| qwen/qwen3.8-flash | naive | -0.06 | 4% |
| qwen/qwen3.8-flash | rubric | +0.01 | 39% |
| qwen/qwen3.8-flash | cot | +0.01 | 43% |
| google/gemini-3.5-flash-lite | naive | +0.04 | 4% |
| google/gemini-3.5-flash-lite | rubric | +0.07 | 49% |
| google/gemini-3.5-flash-lite | cot | +0.13 | 13% |

## Cost and tokens per judge

| Judge | Calls | Cost | Avg output tokens | Avg reasoning tokens |
|---|---|---|---|---|
| anthropic/claude-haiku-5.5 | 360 | $0.1915 | 718 | 288 |
| openai/gpt-6-luna | 360 | $0.1071 | 329 | 158 |
| deepseek/deepseek-v4.1-flash | 360 | $0.2371 | 784 | 594 |
| qwen/qwen3.8-flash | 360 | $0.1795 | 722 | 529 |
| google/gemini-3.5-flash-lite | 360 | $0.3037 | 197 | 0 |

## Repeated cells (planted error: wrong markdown picked, per run)

| Judge | Prompt | Run 1 | Run 2 | Run 3 | Pairs where markdown won |
|---|---|---|---|---|---|
| google/gemini-3.5-flash-lite | cot | 0/20 | 2/20 | 1/20 | style_036 |
| google/gemini-3.5-flash-lite | naive | 6/20 | 6/20 | 6/20 | style_008, style_013, style_032, style_036 |
