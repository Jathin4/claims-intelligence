# Agent benchmark: are the investigation reports right?

The number checker (`backend/app/verify.py`) proves that every number, date and ID in a finding appears in the evidence it cites. It can't prove the sentence is **true**. So I ran the investigation graph on a fixed set of claims and graded every finding that passed the checker **by hand** against its evidence.

- Command: `python backend/manage.py benchmark-agents` → `docs/agent-benchmark.json` (round 2) · round 1: `docs/agent-benchmark-round1.json`
- LLM: qwen3:8b via Ollama, temperature 0, CPU only. No cases are created; nothing is written to the database.

## The claims (picked by rules, not by hand)

| Rule | Claim | Why it's hard |
|---|---|---|
| Highest risk score | CLM-021505 | Many strong signals at once |
| Just above the threshold | CLM-007384 | Score 0.2336 vs threshold 0.2336 |
| Just below the threshold | CLM-006288 | Score 0.2335: one rounding step from the other answer |
| Unusual but not recommended | CLM-019910 | The two models disagree |
| Documents missing, recommended | CLM-029807 | A missing value |
| Customer's first claim | CLM-011032 | No history at all |
| From the drifted batch | CLM-040322 | Data unlike the training data |

## Grading

- **Correct:** true according to the cited evidence (loose wording allowed if nothing is false).
- **Speculative:** not false, but says more than the evidence supports ("may indicate…").
- **Wrong:** contradicts the evidence.

## Results

| | Round 1 | Round 2 (after fixes) |
|---|---|---|
| Findings that passed the checker | 34 | 35 |
| Findings the checker stopped | 1 (a made-up "3 of them in the home category") | 0 |
| Rewrites needed | 1 claim | 0 |
| **Correct** | **26 (76%)** | **33 (94%)** |
| Speculative | 4 | 1 |
| Wrong | 4 | 1 |
| Median time per claim | 110 s | 104 s |

### What was wrong in round 1, and the fix

| Finding (shortened) | Problem | Root cause | Fix |
|---|---|---|---|
| "The risk score is just above the threshold, but the recommendation is not to review" (CLM-006288) | False: 0.2335 is below 0.2336 | **My evidence** rounded the score to 3 decimals (0.234) next to a 4-decimal threshold | Score shown with 4 decimals |
| "2 claims in the last 14 days, which raises the risk by 0.95" (CLM-019910) | 0.95 belongs to "days since previous claim"; the 14-day count isn't a model input | Number attached to the wrong label: the checker's known blind spot | The 14-day count is labelled "used by the anomaly check, not by the risk model"; prompt: attach a SHAP value only to its own factor |
| "Slightly below the median, which contributes to a lower risk score" (CLM-007384) | SHAP says this factor **raises** the risk | The LLM guessed the direction | Prompt: raises/lowers comes only from the SHAP evidence |
| "11.8%, slightly below the overall 12.4%, suggesting this claim's risk is higher" (CLM-011032) | Non sequitur | Reasoning | No direct fix: the reviewer sees the evidence |
| "Not flagged, but close to the threshold, so it may have unusual characteristics" (3 claims) | Speculative | The LLM hedges | Prompt: don't suggest an unflagged claim is unusual |

**Missed in round 1:** CLM-019910's customer filed **the same amount (1,023.90) three times in five days**, the strongest signal in the case, and no finding mentioned it. A new evidence fact counts earlier claims with exactly the same amount; in round 2 the LLM wrote *"has submitted 2 claims with the same amount as this one, which may indicate a pattern"* and asked the reviewer to check them.

### What is still wrong in round 2

- **Wrong (1):** "a recent claim in the same category" (CLM-007384): the earlier claims were vehicle and home; this claim is device. The checker can't catch it: there's no number in the sentence.
- **Speculative (1):** "may indicate a higher risk of needing manual review despite the model's low risk score" (CLM-006288): second-guesses the model without evidence.

## Held-out test: 14 claims never used for tuning

`python backend/manage.py benchmark-agents --heldout` → `docs/agent-benchmark-heldout.json`. Same 7 rules, the **next 2 claims** for each, excluding the 7 above. Nothing was changed between round 2 and this run. Graded the same way (19 Nov 2026).

| | Held-out |
|---|---|
| Findings that passed the checker | 69 |
| Findings the checker stopped | 1 |
| Claims that needed a rewrite | 5 of 14 |
| **Correct** | **61 (88%)** |
| Speculative | 4 |
| Wrong | 4 |
| Median time per claim | 176 s (slower than the 104 s before; other work was running on the laptop) |

**88% on unseen claims vs 94% on the tuning claims**: the fixes generalised, but not completely, which is what a held-out test is for.

| Wrong finding (shortened) | Problem |
|---|---|
| "Days since the previous claim is 3, which lowers the risk" (CLM-000673) | SHAP says it **raises** the risk by 0.74: the direction rule from round 1 reduced these errors but didn't remove them |
| "Raising the risk by 10.61 times the median" (CLM-010460) | 10.61 is the amount ratio, not a change in risk: a number attached to the wrong concept |
| "7 claims in 12 months, above the category baseline" (CLM-016834) | There is no baseline for claim counts: an invented comparison |
| "The amount vs category median slightly raises the risk" (CLM-015858) | It's the **largest** factor (+0.89): understated |

The speculative ones: "may suggest a pattern of frequent claims" (one earlier claim), "may indicate a potential anomaly or a lower-risk claim", and two findings calling a +0.20 effect "significant" next to a +2.58 one.

**The checker's own mistake:** the one stopped finding was stopped for the wrong reason. The comparison check read "12" from "in the last 12 months" as one side of "a higher rate (13.6%)". The sentence was wrong anyway (it called the claim "the third in the device category"; it was the second), but the check shouldn't have fired. Fixed afterwards: time windows like "the last 12 months" are ignored by the comparison check (test added).

## Honest caveats

- **Round 2 isn't a clean test.** The fixes were made after reading round 1 on the **same 7 claims**, so its 94% is optimistic. The held-out run above (88%) is the fair number.
- **One grader (me), 21 claims, 138 findings in total.** Enough to find failure patterns, not to give a precise accuracy.
- **Numbers written as words** ("one claim", "three claims") bypass the checker. The prompt asks for digits; the LLM doesn't always comply.
- **Run-to-run variation:** even at temperature 0 the wording changes between runs.

## What this means for the design

The checker catches invented numbers and fake citations reliably; it can't make a sentence true. That's why the report shows the evidence next to every finding, the summary is written by code, and a second person must decide, with a reason.
