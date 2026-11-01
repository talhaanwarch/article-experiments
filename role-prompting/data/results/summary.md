Calls with a reply: 63000. Failed attempts, retried: 791. Cost at list prices: $10.11.

## 1. Setups, calls and cost

Go does not report reasoning tokens for Claude Haiku 5.5; its output tokens include them, and the last column shows how often a reply came back with reasoning text.

| Model | Reasoning | Calls | Runs per cell | Mean output tokens | Mean reasoning tokens | Cost | Calls with reasoning text |
|---|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | reasoning on | 5400 | 3-3 | 156 | 0 | $0.51 | 86.3% |
| GPT-6 Luna | reasoning on | 5400 | 3-3 | 70 | 63 | $0.25 | 0.0% |
| DeepSeek V4.1 Flash | reasoning on | 5400 | 3-3 | 323 | 321 | $1.16 | 100.0% |
| Qwen3.8 Flash | reasoning on | 5400 | 3-3 | 641 | 638 | $1.75 | 99.8% |
| GLM-5.3 Flash | reasoning on | 5400 | 3-3 | 117 | 85 | $0.41 | 74.5% |
| DeepSeek V4 Pro | reasoning on | 3600 | 2-2 | 507 | 505 | $4.06 | 100.0% |
| Claude Haiku 5.5 | reasoning off | 5400 | 3-3 | 19 | 0 | $0.14 | 0.0% |
| GPT-6 Luna | reasoning off | 5400 | 3-3 | 5 | 0 | $0.07 | 0.0% |
| DeepSeek V4.1 Flash | reasoning off | 5400 | 3-3 | 14 | 0 | $0.13 | 0.0% |
| Qwen3.8 Flash | reasoning off | 5400 | 3-3 | 4 | 0 | $0.11 | 0.0% |
| Gemini 3.5 Flash-Lite | never reasons | 5400 | 3-3 | 84 | 0 | $1.30 | 0.0% |
| Qwen2.5-72B (Zheng's model) | never reasons | 5400 | 3-3 | 1 | 0 | $0.23 | 0.0% |

## 2. Accuracy by condition

| Model | Reasoning | no role | no role again | helpful | in-domain | out-of-domain | toddler |
|---|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | reasoning on | 96.9% | 96.6% | 96.0% | 97.0% | 96.9% | 96.8% |
| GPT-6 Luna | reasoning on | 96.2% | 96.6% | 96.6% | 96.2% | 96.0% | 96.3% |
| DeepSeek V4.1 Flash | reasoning on | 97.2% | 97.1% | 97.4% | 97.3% | 97.1% | 97.2% |
| Qwen3.8 Flash | reasoning on | 96.6% | 96.8% | 96.7% | 96.6% | 96.4% | 95.3% |
| GLM-5.3 Flash | reasoning on | 94.7% | 94.8% | 94.3% | 95.7% | 93.9% | 85.4% |
| DeepSeek V4 Pro | reasoning on | 98.5% | 97.5% | 97.5% | 97.7% | 97.7% | 97.8% |
| Claude Haiku 5.5 | reasoning off | 89.4% | 88.4% | 87.6% | 89.7% | 87.3% | 80.8% |
| GPT-6 Luna | reasoning off | 89.0% | 89.6% | 88.1% | 89.6% | 88.2% | 87.3% |
| DeepSeek V4.1 Flash | reasoning off | 91.9% | 92.1% | 90.9% | 91.8% | 92.6% | 90.7% |
| Qwen3.8 Flash | reasoning off | 88.0% | 88.8% | 88.0% | 88.3% | 89.6% | 84.8% |
| Gemini 3.5 Flash-Lite | never reasons | 94.9% | 95.1% | 94.4% | 93.9% | 93.7% | 65.6% |
| Qwen2.5-72B (Zheng's model) | never reasons | 85.3% | 85.2% | 85.0% | 84.4% | 82.3% | 81.7% |

## 3. Effect of each role against no role, in points [95% CI]

`*` = the interval excludes zero. The first column is the noise floor: no role against no role.

| Model | Reasoning | no role, asked again (noise floor) | helpful assistant | in-domain expert | out-of-domain expert | 4-year-old toddler |
|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | reasoning on | -0.3 [-1.3, +0.7] | -0.9 [-2.0, +0.2] | +0.1 [-0.6, +0.8] | +0.0 [-1.1, +1.0] | -0.1 [-1.4, +1.2] |
| GPT-6 Luna | reasoning on | +0.3 [-0.4, +1.2] | +0.3 [-0.7, +1.3] | -0.0 [-1.1, +1.0] | -0.2 [-1.2, +0.7] | +0.1 [-0.6, +0.8] |
| DeepSeek V4.1 Flash | reasoning on | -0.1 [-0.9, +0.7] | +0.2 [-0.3, +0.8] | +0.1 [-0.6, +0.9] | -0.1 [-0.9, +0.6] | +0.0 [-0.9, +0.9] |
| Qwen3.8 Flash | reasoning on | +0.2 [-0.6, +1.1] | +0.1 [-0.7, +1.0] | +0.0 [-0.6, +0.7] | -0.1 [-1.1, +0.9] | -1.2 [-2.3, -0.2] * |
| GLM-5.3 Flash | reasoning on | +0.1 [-1.6, +1.8] | -0.3 [-1.9, +1.2] | +1.0 [-0.6, +2.6] | -0.8 [-2.6, +1.0] | -9.2 [-11.7, -6.8] * |
| DeepSeek V4 Pro | reasoning on | -1.0 [-2.2, -0.2] * | -1.0 [-2.2, -0.2] * | -0.8 [-2.0, +0.0] | -0.8 [-2.2, +0.3] | -0.7 [-1.8, +0.2] |
| Claude Haiku 5.5 | reasoning off | -1.0 [-2.2, +0.2] | -1.9 [-3.8, -0.1] * | +0.2 [-2.1, +2.7] | -2.1 [-4.4, +0.2] | -8.7 [-11.8, -5.8] * |
| GPT-6 Luna | reasoning off | +0.6 [-0.6, +1.8] | -0.9 [-2.4, +0.6] | +0.6 [-1.0, +2.2] | -0.8 [-2.6, +0.9] | -1.7 [-3.7, +0.2] |
| DeepSeek V4.1 Flash | reasoning off | +0.2 [-1.0, +1.6] | -1.0 [-2.6, +0.4] | -0.1 [-1.6, +1.3] | +0.7 [-0.8, +2.2] | -1.2 [-2.7, +0.2] |
| Qwen3.8 Flash | reasoning off | +0.8 [-0.9, +2.4] | +0.0 [-1.6, +1.6] | +0.3 [-1.6, +2.3] | +1.6 [-0.2, +3.4] | -3.2 [-5.3, -1.1] * |
| Gemini 3.5 Flash-Lite | never reasons | +0.2 [-0.9, +1.3] | -0.4 [-2.0, +1.1] | -1.0 [-2.8, +0.8] | -1.2 [-2.9, +0.3] | -29.3 [-33.3, -25.3] * |
| Qwen2.5-72B (Zheng's model) | never reasons | -0.1 [-1.6, +1.3] | -0.3 [-1.8, +1.1] | -0.9 [-2.4, +0.7] | -3.0 [-5.0, -1.2] * | -3.7 [-6.0, -1.6] * |

## 4. Average effect by group (models weighted equally), in points

| Group | Models | no role, asked again (noise floor) | helpful assistant | in-domain expert | out-of-domain expert | 4-year-old toddler |
|---|---|---|---|---|---|---|
| reasoning on, small | 5 | +0.0 | -0.1 | +0.2 | -0.2 | -2.1 |
| reasoning on, big (DeepSeek V4 Pro) | 1 | -1.0 | -1.0 | -0.8 | -0.8 | -0.7 |
| reasoning off (same models) | 4 | +0.1 | -0.9 | +0.3 | -0.2 | -3.7 |
| never reasons | 2 | +0.1 | -0.4 | -0.9 | -2.1 | -16.5 |

## 5. Effects that beat the noise floor

A role effect counts here only if its CI excludes zero AND its size is larger than the largest noise-floor gap seen on any setup.

Largest noise-floor gap: +1.0 points (range -1.0 to +0.8).

- Qwen3.8 Flash (reasoning on), 4-year-old toddler: -1.2 points
- GLM-5.3 Flash (reasoning on), 4-year-old toddler: -9.2 points
- Claude Haiku 5.5 (reasoning off), helpful assistant: -1.9 points
- Claude Haiku 5.5 (reasoning off), 4-year-old toddler: -8.7 points
- Qwen3.8 Flash (reasoning off), 4-year-old toddler: -3.2 points
- Gemini 3.5 Flash-Lite (never reasons), 4-year-old toddler: -29.3 points
- Qwen2.5-72B (Zheng's model) (never reasons), out-of-domain expert: -3.0 points
- Qwen2.5-72B (Zheng's model) (never reasons), 4-year-old toddler: -3.7 points

Role contrasts whose CI excludes zero: 9 of 48. Noise-floor contrasts whose CI excludes zero: 1 of 12.

## 6. Same model, reasoning on vs off: role effect in each, in points

| Model | Role | Effect, reasoning on | Effect, reasoning off | Off minus on [95% CI] |
|---|---|---|---|---|
| Claude Haiku 5.5 | no role, asked again (noise floor) | -0.3 | -1.0 | -0.7 [-2.2, +0.9] |
| Claude Haiku 5.5 | helpful assistant | -0.9 | -1.9 | -1.0 [-3.2, +1.2] |
| Claude Haiku 5.5 | in-domain expert | +0.1 | +0.2 | +0.1 [-2.3, +2.6] |
| Claude Haiku 5.5 | out-of-domain expert | +0.0 | -2.1 | -2.1 [-4.9, +0.9] |
| Claude Haiku 5.5 | 4-year-old toddler | -0.1 | -8.7 | -8.6 [-11.9, -5.3] * |
| GPT-6 Luna | no role, asked again (noise floor) | +0.3 | +0.6 | +0.2 [-1.2, +1.8] |
| GPT-6 Luna | helpful assistant | +0.3 | -0.9 | -1.2 [-3.0, +0.4] |
| GPT-6 Luna | in-domain expert | -0.0 | +0.6 | +0.6 [-1.3, +2.6] |
| GPT-6 Luna | out-of-domain expert | -0.2 | -0.8 | -0.6 [-2.6, +1.4] |
| GPT-6 Luna | 4-year-old toddler | +0.1 | -1.7 | -1.8 [-3.9, +0.2] |
| DeepSeek V4.1 Flash | no role, asked again (noise floor) | -0.1 | +0.2 | +0.3 [-1.0, +1.8] |
| DeepSeek V4.1 Flash | helpful assistant | +0.2 | -1.0 | -1.2 [-2.9, +0.3] |
| DeepSeek V4.1 Flash | in-domain expert | +0.1 | -0.1 | -0.2 [-1.7, +1.2] |
| DeepSeek V4.1 Flash | out-of-domain expert | -0.1 | +0.7 | +0.8 [-0.8, +2.4] |
| DeepSeek V4.1 Flash | 4-year-old toddler | +0.0 | -1.2 | -1.2 [-2.9, +0.3] |
| Qwen3.8 Flash | no role, asked again (noise floor) | +0.2 | +0.8 | +0.6 [-1.3, +2.4] |
| Qwen3.8 Flash | helpful assistant | +0.1 | +0.0 | -0.1 [-2.0, +1.8] |
| Qwen3.8 Flash | in-domain expert | +0.0 | +0.3 | +0.3 [-1.7, +2.3] |
| Qwen3.8 Flash | out-of-domain expert | -0.1 | +1.6 | +1.7 [-0.6, +3.9] |
| Qwen3.8 Flash | 4-year-old toddler | -1.2 | -3.2 | -2.0 [-4.3, +0.2] |

No-role accuracy, reasoning on vs off:

- Claude Haiku 5.5: reasoning on minus off +7.4 [+4.3, +10.7] * points
- GPT-6 Luna: reasoning on minus off +7.2 [+4.7, +10.0] * points
- DeepSeek V4.1 Flash: reasoning on minus off +5.3 [+2.8, +8.0] * points
- Qwen3.8 Flash: reasoning on minus off +8.6 [+5.7, +11.7] * points

## 7. In-domain minus out-of-domain expert, in points [95% CI]

Zheng et al. report +0.4 points for this contrast (mixed model over their models).

| Model | Reasoning | In-domain minus out-of-domain |
|---|---|---|
| Claude Haiku 5.5 | reasoning on | +0.1 [-0.8, +1.1] |
| GPT-6 Luna | reasoning on | +0.2 [-0.6, +1.0] |
| DeepSeek V4.1 Flash | reasoning on | +0.2 [-0.8, +1.2] |
| Qwen3.8 Flash | reasoning on | +0.1 [-0.8, +1.0] |
| GLM-5.3 Flash | reasoning on | +1.8 [+0.3, +3.2] * |
| DeepSeek V4 Pro | reasoning on | +0.0 [-1.2, +1.0] |
| Claude Haiku 5.5 | reasoning off | +2.3 [+0.0, +4.8] |
| GPT-6 Luna | reasoning off | +1.3 [-0.3, +3.1] |
| DeepSeek V4.1 Flash | reasoning off | -0.8 [-2.4, +0.8] |
| Qwen3.8 Flash | reasoning off | -1.2 [-3.1, +0.7] |
| Gemini 3.5 Flash-Lite | never reasons | +0.2 [-1.7, +2.1] |
| Qwen2.5-72B (Zheng's model) | never reasons | +2.1 [+0.3, +4.0] * |

## 8. In-domain expert effect by question domain, all setups pooled (models weighted equally)

| Domain | Questions | In-domain expert | Out-of-domain expert | Toddler |
|---|---|---|---|---|
| econ | 38 | +0.2 | -0.1 | -3.3 |
| eecs | 38 | -1.2 | -1.9 | -6.3 |
| law | 37 | +0.2 | -1.7 | -5.0 |
| math | 38 | +0.6 | +0.0 | -6.4 |
| medicine | 37 | -0.5 | -0.8 | -4.5 |
| natural science | 38 | +1.2 | +1.0 | -4.4 |
| politics | 37 | -0.6 | -0.8 | -5.1 |
| psychology | 37 | -0.2 | -0.6 | -4.3 |

## 9. Replies with no option number (refusals and format failures)

| Model | Reasoning | no role | no role again | helpful | in-domain | out-of-domain | toddler |
|---|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | reasoning on | 0/900 | 0/900 | 0/900 | 0/900 | 0/900 | 2/900 |
| GPT-6 Luna | reasoning on | 0/900 | 0/900 | 0/900 | 0/900 | 0/900 | 0/900 |
| DeepSeek V4.1 Flash | reasoning on | 1/900 | 3/900 | 2/900 | 2/900 | 1/900 | 1/900 |
| Qwen3.8 Flash | reasoning on | 4/900 | 1/900 | 4/900 | 4/900 | 5/900 | 5/900 |
| GLM-5.3 Flash | reasoning on | 1/900 | 2/900 | 3/900 | 0/900 | 2/900 | 72/900 |
| DeepSeek V4 Pro | reasoning on | 0/600 | 0/600 | 0/600 | 0/600 | 1/600 | 0/600 |
| Claude Haiku 5.5 | reasoning off | 4/900 | 6/900 | 3/900 | 3/900 | 3/900 | 65/900 |
| GPT-6 Luna | reasoning off | 2/900 | 3/900 | 3/900 | 0/900 | 2/900 | 3/900 |
| DeepSeek V4.1 Flash | reasoning off | 3/900 | 0/900 | 2/900 | 4/900 | 0/900 | 0/900 |
| Qwen3.8 Flash | reasoning off | 1/900 | 0/900 | 3/900 | 3/900 | 1/900 | 1/900 |
| Gemini 3.5 Flash-Lite | never reasons | 3/900 | 2/900 | 3/900 | 8/900 | 2/900 | 244/900 |
| Qwen2.5-72B (Zheng's model) | never reasons | 2/900 | 2/900 | 0/900 | 6/900 | 3/900 | 4/900 |

Up to 15 examples:

- GLM-5.3 Flash (think), low_knowledge: "1! Hehe, dat's da one! 🧀 ...can the moon cheese be a client too?"
- Qwen3.8 Flash (off), out_domain: "exprethabdsmantavadosNXNNNNkNnKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKkKk"
- Claude Haiku 5.5 (off), no_role: "I don't have the table or figure that this question refers to, so I can't determine the number of units of labor hired at a $4.50 wage. Could you share the labor demand data, the marginal revenue prod"
- Claude Haiku 5.5 (off), helpful: "I don't have the table or figure that accompanies this question, so I can't determine the answer from the information given. Could you share the relevant data, such as the marginal revenue product of "
- Claude Haiku 5.5 (off), no_role_rerun: "I can't tell which answer is correct without the data this question depends on, such as the marginal revenue product of labor schedule or the labor supply and demand figures. Your message doesn't incl"
- Claude Haiku 5.5 (off), out_domain: "I can't tell which answer is correct because the question doesn't include the marginal revenue product or labor demand schedule needed to determine how many units are hired at $4.50. Without that data"
- Claude Haiku 5.5 (off), in_domain: "I can't determine the answer without the accompanying table or figure showing the marginal revenue product of labor or the labor demand schedule. Could you share the data, such as the marginal revenue"
- GPT-6 Luna (off), helpful: "Insufficient information."
- Claude Haiku 5.5 (off), low_knowledge: "I don't know what the wage thing means, but the moon is totally cheese, so I'm gonna say 4!"
- GPT-6 Luna (off), no_role_rerun: "Cannot determine without the labor demand or marginal revenue product data."
- DeepSeek V4.1 Flash (off), no_role: "I’m unable to determine the answer from the information given because the question appears to refer to a table or graph showing the value of marginal product or demand for labor, which isn’t provided "
- GPT-6 Luna (off), out_domain: "Unable to determine from the information provided."
- GPT-6 Luna (off), low_knowledge: "Cannot be determined from the information provided."
- Qwen2.5-72B (Zheng's model) (none), in_domain: "To accurately choose the correct option, I would need more information, such as the demand for labor or the marginal product of labor. However, based on the information given and assuming that the que"
- Claude Haiku 5.5 (off), low_knowledge: "I don't know the answer to that one, but I do know the moon is made of cheese!"

## 10. Output length by condition (mean completion tokens, reasoning included)

| Model | Reasoning | no role | no role again | helpful | in-domain | out-of-domain | toddler |
|---|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | reasoning on | 137 | 140 | 137 | 138 | 142 | 242 |
| GPT-6 Luna | reasoning on | 76 | 69 | 68 | 70 | 68 | 67 |
| DeepSeek V4.1 Flash | reasoning on | 339 | 382 | 313 | 333 | 316 | 256 |
| Qwen3.8 Flash | reasoning on | 713 | 683 | 710 | 661 | 605 | 476 |
| GLM-5.3 Flash | reasoning on | 86 | 151 | 104 | 97 | 90 | 177 |
| DeepSeek V4 Pro | reasoning on | 588 | 567 | 445 | 486 | 485 | 471 |
| Claude Haiku 5.5 | reasoning off | 21 | 20 | 19 | 23 | 21 | 12 |
| GPT-6 Luna | reasoning off | 5 | 5 | 5 | 5 | 5 | 5 |
| DeepSeek V4.1 Flash | reasoning off | 18 | 20 | 11 | 18 | 11 | 8 |
| Qwen3.8 Flash | reasoning off | 6 | 4 | 4 | 4 | 2 | 1 |
| Gemini 3.5 Flash-Lite | never reasons | 78 | 78 | 112 | 118 | 102 | 18 |
| Qwen2.5-72B (Zheng's model) | never reasons | 1 | 1 | 1 | 2 | 1 | 1 |

## 12. Toddler effect, strict vs lenient parsing, in points [95% CI]

Strict is the parser used everywhere above. Lenient also reads a no-answer reply that holds exactly one distinct number from 1 to 4 ("Moon is yummy cheese! ... 3!") as that option, in both the toddler and the no-role calls.

| Model | Reasoning | No-answer replies, toddler | Of those, one number 1-4 | That number is right | Toddler effect, strict | Toddler effect, lenient |
|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | reasoning on | 2 | 1 | 0 | -0.1 [-1.4, +1.2] | -0.1 [-1.4, +1.2] |
| GPT-6 Luna | reasoning on | 0 | 0 | 0 | +0.1 [-0.6, +0.8] | +0.1 [-0.6, +0.8] |
| DeepSeek V4.1 Flash | reasoning on | 1 | 0 | 0 | +0.0 [-0.9, +0.9] | +0.0 [-0.9, +0.9] |
| Qwen3.8 Flash | reasoning on | 5 | 0 | 0 | -1.2 [-2.3, -0.2] * | -1.2 [-2.3, -0.2] * |
| GLM-5.3 Flash | reasoning on | 72 | 68 | 65 | -9.2 [-11.7, -6.8] * | -2.0 [-4.0, -0.0] * |
| DeepSeek V4 Pro | reasoning on | 0 | 0 | 0 | -0.7 [-1.8, +0.2] | -0.7 [-1.8, +0.2] |
| Claude Haiku 5.5 | reasoning off | 65 | 51 | 25 | -8.7 [-11.8, -5.8] * | -5.9 [-8.7, -3.2] * |
| GPT-6 Luna | reasoning off | 3 | 0 | 0 | -1.7 [-3.7, +0.2] | -1.7 [-3.7, +0.2] |
| DeepSeek V4.1 Flash | reasoning off | 0 | 0 | 0 | -1.2 [-2.7, +0.2] | -1.2 [-2.7, +0.2] |
| Qwen3.8 Flash | reasoning off | 1 | 0 | 0 | -3.2 [-5.3, -1.1] * | -3.2 [-5.3, -1.1] * |
| Gemini 3.5 Flash-Lite | never reasons | 244 | 220 | 189 | -29.3 [-33.3, -25.3] * | -8.3 [-11.3, -5.6] * |
| Qwen2.5-72B (Zheng's model) | never reasons | 4 | 1 | 0 | -3.7 [-6.0, -1.6] * | -3.7 [-6.0, -1.6] * |

## 11. Rebuild check: the paper's Qwen2.5-72B-Instruct, no role, all 2,410 questions

Our accuracy: 79.7% on 2410 questions (9 with no option number). Paper: 80.3%. Gap: -0.6 points. Cost: $0.09.
