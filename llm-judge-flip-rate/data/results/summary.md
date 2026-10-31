Calls with a reply: 7250. Failed attempts that were retried (mostly HTTP 429 from Qwen): 419. Replies with no verdict: 15. Cost: $3.11

Verdicts per cell: min 10, max 10

## 1. Flip rate, paper prompt, temperature 1.0, paper's order

| Judge | Mean flip rate | Pairs with 0% | Pairs over 20% | Max | Pairs where both answers won at least once |
|---|---|---|---|---|---|
| Claude Haiku 5.5 | 11.0% | 19/29 | 6/29 | 50.0% | 3/29 |
| GPT-6 Luna | 10.7% | 17/29 | 6/29 | 40.0% | 10/29 |
| DeepSeek V4.1 Flash | 12.2% | 17/29 | 8/29 | 50.0% | 5/29 |
| Qwen3.8 Flash | 14.8% | 16/29 | 8/29 | 50.0% | 11/29 |
| Gemini 3.5 Flash-Lite | 5.5% | 22/29 | 2/29 | 50.0% | 4/29 |

All five judges: mean 10.9%. Paper: 13.3% (GPT-4o-mini), 13.9% (GPT-4.1-mini).

Flip rate with the answers swapped (same prompt, temperature 1.0):

| Judge | Mean flip rate, swapped order |
|---|---|
| Claude Haiku 5.5 | 10.7% |
| GPT-6 Luna | 11.4% |
| DeepSeek V4.1 Flash | 16.7% |
| Qwen3.8 Flash | 12.8% |
| Gemini 3.5 Flash-Lite | 10.0% |

## 2. Share of verdicts that are ties (paper prompt, t=1.0, both orders)

| Judge | Ties |
|---|---|
| Claude Haiku 5.5 | 21.6% |
| GPT-6 Luna | 18.5% |
| DeepSeek V4.1 Flash | 22.6% |
| Qwen3.8 Flash | 17.3% |
| Gemini 3.5 Flash-Lite | 16.4% |

## 3. Position: the paper's 'A wins' index, then with the answers swapped

PBI = share of pairs whose majority verdict is slot A (paper's definition). In the paper's order slot A is GPT-4o-mini's answer; swapped, slot A is GPT-4o's.

| Judge | PBI, paper order | PBI, swapped | Slot A share of A/B verdicts, both orders | Pairs whose majority winner changes when swapped |
|---|---|---|---|---|
| Claude Haiku 5.5 | 48.3% | 31.0% | 52.3% | 5/29 |
| GPT-6 Luna | 41.4% | 17.2% | 40.0% | 7/29 |
| DeepSeek V4.1 Flash | 51.7% | 48.3% | 68.5% | 13/29 |
| Qwen3.8 Flash | 44.8% | 27.6% | 49.1% | 5/29 |
| Gemini 3.5 Flash-Lite | 51.7% | 51.7% | 63.7% | 10/29 |

Paper: PBI 72% (GPT-4o-mini), 59% (GPT-4.1-mini), paper order only.

Which model's answer wins the majority, by order (pairs out of 29):

| Judge | Order | GPT-4o-mini | GPT-4o | tie | no majority |
|---|---|---|---|---|---|
| Claude Haiku 5.5 | original | 14 | 7 | 7 | 1 |
| Claude Haiku 5.5 | swapped | 13 | 9 | 7 | 0 |
| GPT-6 Luna | original | 12 | 12 | 5 | 0 |
| GPT-6 Luna | swapped | 18 | 5 | 5 | 1 |
| DeepSeek V4.1 Flash | original | 15 | 8 | 6 | 0 |
| DeepSeek V4.1 Flash | swapped | 5 | 14 | 6 | 4 |
| Qwen3.8 Flash | original | 13 | 9 | 5 | 2 |
| Qwen3.8 Flash | swapped | 15 | 8 | 4 | 2 |
| Gemini 3.5 Flash-Lite | original | 15 | 9 | 5 | 0 |
| Gemini 3.5 Flash-Lite | swapped | 10 | 15 | 4 | 0 |

## 4. Temperature 0 (paper prompt, paper's order)

| Judge | Flip rate t=1.0 | Flip rate t=0 | Pairs with any flip at t=0 | Majority differs from t=1.0 |
|---|---|---|---|---|
| Claude Haiku 5.5 | 11.0% | 9.3% | 10/29 | 3/29 |
| GPT-6 Luna | 10.7% | 10.3% | 11/29 | 2/29 |
| DeepSeek V4.1 Flash | 12.2% | 7.2% | 8/29 | 5/29 |
| Qwen3.8 Flash | 14.8% | 8.6% | 10/29 | 3/29 |
| Gemini 3.5 Flash-Lite | 5.5% | 9.0% | 11/29 | 1/29 |

Paper: 13.3% -> 2.8% (GPT-4o-mini), 13.9% -> 7.9% (GPT-4.1-mini).

## 5. Prompt wording (temperature 1.0, paper's order)

| Judge | Unparseable: paper prompt | Unparseable: prompt B | Unparseable: prompt B fixed | Majority differs, paper vs B | Majority differs, paper vs B fixed | Flip rate B | Flip rate B fixed |
|---|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | 0/290 | 0/290 | 0/290 | 6/29 | 6/29 | 9.3% | 6.6% |
| GPT-6 Luna | 1/290 | 0/290 | 0/290 | 5/29 | 5/29 | 13.1% | 10.7% |
| DeepSeek V4.1 Flash | 2/290 | 2/290 | 2/290 | 2/29 | 3/29 | 7.9% | 7.9% |
| Qwen3.8 Flash | 1/290 | 0/290 | 1/290 | 5/29 | 3/29 | 11.7% | 12.8% |
| Gemini 3.5 Flash-Lite | 0/290 | 0/290 | 0/290 | 8/29 | 7/29 | 11.0% | 10.7% |

Paper: majority changed in 5 of 20 (judge, question) cases on a 10-question subset.

## 6. Do the five judges agree with each other? (paper prompt, t=1.0, both orders pooled)

| | Claude Haiku 5.5 | GPT-6 Luna | DeepSeek V4.1 Flash | Qwen3.8 Flash | Gemini 3.5 Flash-Lite |
|---|---|---|---|---|---|
| Claude Haiku 5.5 | – | 21/29 | 21/29 | 21/29 | 18/29 |
| GPT-6 Luna | 21/29 | – | 26/29 | 27/29 | 19/29 |
| DeepSeek V4.1 Flash | 21/29 | 26/29 | – | 27/29 | 18/29 |
| Qwen3.8 Flash | 21/29 | 27/29 | 27/29 | – | 19/29 |
| Gemini 3.5 Flash-Lite | 18/29 | 19/29 | 18/29 | 19/29 | – |

Pairs where all five judges give the same majority winner: 15/29. Paper: the two OpenAI judges agreed on 22 of 29.

## 7. Mean flip rate by category (paper prompt, t=1.0, paper's order, all judges)

| Category | Mean flip rate |
|---|---|
| extraction | 23.3% |
| coding | 16.7% |
| math | 13.3% |
| instruction | 10.7% |
| knowledge | 10.0% |
| writing | 9.3% |
| ethics | 8.0% |
| roleplay | 6.0% |
| reasoning | 6.0% |
| hard | 4.7% |

## 8. One call vs the 10-call majority (paper prompt, t=1.0, paper's order)

| Judge | One call matches the 10-call majority |
|---|---|
| Claude Haiku 5.5 | 89.0% |
| GPT-6 Luna | 89.3% |
| DeepSeek V4.1 Flash | 87.8% |
| Qwen3.8 Flash | 85.1% |
| Gemini 3.5 Flash-Lite | 94.5% |

Paper: 86.6% against the 50-call majority.

## 9. Is the prompt effect bigger than re-running the same prompt?

Majorities from 5 calls each (pairs out of 29 whose majority winner differs). Baseline: repeats 1-5 vs repeats 6-10 of the paper prompt.

| Judge | Same prompt, two halves | Paper prompt vs prompt B | Paper prompt vs prompt B fixed |
|---|---|---|---|
| Claude Haiku 5.5 | 4/29 | 6/29 | 5/29 |
| GPT-6 Luna | 1/29 | 5/29 | 6/29 |
| DeepSeek V4.1 Flash | 5/29 | 2/29 | 3/29 |
| Qwen3.8 Flash | 6/29 | 4/29 | 5/29 |
| Gemini 3.5 Flash-Lite | 2/29 | 7/29 | 7/29 |

## 10. What a flip is (paper prompt, t=1.0, paper's order)

Each verdict that differs from its cell's majority, split by whether a tie is involved.

| Judge | Non-majority verdicts | Involve a tie | Opposite winner |
|---|---|---|---|
| Claude Haiku 5.5 | 32 | 26 | 6 |
| GPT-6 Luna | 31 | 8 | 23 |
| DeepSeek V4.1 Flash | 35 | 23 | 12 |
| Qwen3.8 Flash | 43 | 18 | 25 |
| Gemini 3.5 Flash-Lite | 16 | 9 | 7 |

## 11. Which model's answer wins, both orders pooled

| Judge | GPT-4o-mini share of non-tie verdicts |
|---|---|
| Claude Haiku 5.5 | 61.8% |
| GPT-6 Luna | 58.1% |
| DeepSeek V4.1 Flash | 50.6% |
| Qwen3.8 Flash | 57.4% |
| Gemini 3.5 Flash-Lite | 51.3% |

## 12. Mean reasoning tokens per call

| Judge | t=1.0 | t=0 |
|---|---|---|
| Claude Haiku 5.5 | 295 | 304 |
| GPT-6 Luna | 163 | 165 |
| DeepSeek V4.1 Flash | 990 | 838 |
| Qwen3.8 Flash | 627 | 538 |
| Gemini 3.5 Flash-Lite | 0 | 0 |

## 13. Parser check

Replies where the paper's parser (first [[A]] wins) and ours (last verdict wins) disagree: 0/7250.

