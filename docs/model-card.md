# Model card: claims-review v1 and claims-anomaly v1

> **All data is synthetic.** These models were built for a portfolio project. They have never seen a real claim, and no result here says anything about real customers or insurers.

## 1. What the models do

| | claims-review v1 | claims-anomaly v1 |
|---|---|---|
| Question | Will this claim need **manual review**? | Does this claim look **unlike typical claims**? |
| Output | A calibrated probability (0–1) and "recommended for review" when it's at or above **0.2336** | An anomaly score; "unusual" when it's above **0.6085** (about the most unusual 2% of training claims) |
| Method | Gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`), isotonic calibration with time-ordered cross-validation | Isolation Forest, 300 trees |
| Explanation | SHAP values per claim (`/api/claims/{id}/explanation`) | None per feature; the score only |
| Files | `model-artifacts/claims-review-v1.joblib` + `.json` (SHA-256 checked before every load) | `model-artifacts/claims-anomaly-v1.joblib` + `.json` |

**Intended use:** help a claims team decide **which claims to look at first**. A person always makes the decision; the app enforces a second person for investigation decisions.

**Not intended for:** deciding to pay or refuse a claim, accusing anyone of fraud, pricing, or any use on real people without retraining, re-validation and a legal review. A high score is a reason to review, **never proof of wrongdoing**.

## 2. Data

- 30,166 synthetic claims from 10,000 customers, 1 Nov 2023 – 31 Oct 2026 (`ml/generate_data.py --seed 42`; same seed gives byte-identical data, checked by a test against the SHA-256 stored with the model).
- The first year is **history only** (so every modelled claim has a full year of past claims); the 20,087 claims from 1 Nov 2024 are split **by time**: train 70% → validation 15% → test 15%.
- Label `manual_review_required`: ~12% positive, drawn from a hidden risk with random noise, so even a perfect model can't be perfect (see the ceiling below).
- Features (7), all known **at the time of the claim**: amount, amount ÷ category median (medians from training data only), category, documents, policy age, claims in the previous 365 days, days since the previous claim.
- **Not used:** IDs, dates, `region` (kept out so it can be used to check fairness), the generator's hidden columns. A test fails if any of them becomes a feature.

Details: [data.md](data.md), [eda/README.md](eda/README.md).

## 3. Results

### claims-review v1 (held-out test period, used once)

| Metric | Value | For comparison |
|---|---|---|
| PR-AUC | **0.342** | Random ordering: 0.137 (the positive rate). Ceiling (the generator's own hidden probability): 0.448 |
| ROC-AUC | 0.747 | |
| Brier score | 0.106 | Always predicting the base rate: 0.118 |
| At the threshold | Review rate **14.5%**, precision **36%**, recall **38%** | The team can review at most 15% |

In plain words: of every 100 claims the model sends to review, about 36 really needed it (vs about 14 if picked at random), and it finds about 38% of all claims that needed review while keeping the workload under 15%.

**Why this threshold:** a missed claim was counted as 10× as costly as an unnecessary review, and the review rate was capped at 15%. The threshold with the lowest cost on the **validation** period was kept. Full comparison of the three candidate models, curves and calibration: [model/README.md](model/README.md).

**Fairness check by region** (test): flag rates 14.0–15.3%, recall 35–41%, false-positive rates 10.5–11.1%. No region is treated very differently; a test fails if a region's flag rate drifts more than 3 points or its recall more than 10 points from the overall figure. Region is the only group in the data, so other groups (age, gender…) could not be checked.

**What drives the score** (mean |SHAP|): amount vs category median 0.69, days since previous claim 0.34, claims in the past year 0.23, documents 0.16, policy age 0.09, amount 0.08, category 0.06.

### claims-anomaly v1

The data contains planted anomalies (huge amounts, bursts of claims, claims without documents), about 1% of claims. On the test period the detector flagged 1.8% of claims; **13%** of its flags were planted anomalies, against **1.4%** for random flagging, and it caught **7 of 41** (17%). It is a weak but real signal: useful as a second opinion next to the risk score, not on its own.

## 4. Monitoring

Every CSV import is compared with the training data (PSI per feature, category mix, missing documents). On a deliberately drifted test batch it flagged the amount ratio (PSI 0.59), the category mix (0.40) and missing documents (27% vs 3%), and the model's review rate doubled to 31.5%, which is exactly the warning a team needs before the workload explodes. Known natural drift: policy age grows over time because the synthetic customer base is closed.

## 5. Limitations

- **Synthetic data:** the relationships were written by me; real claims are messier and the real label (did a reviewer find a problem?) depends on past review policies, which can carry bias into the model.
- **Noisy label:** no model can score above about 0.448 PR-AUC on this data. Ours reaches 0.342, about two-thirds of the way from random (0.137) to that ceiling.
- **Recall 38%:** most claims that need review are still not flagged. Raising recall means a bigger review workload; the threshold is a business choice, not a technical one.
- **No feature for document content**, adjuster notes or external data.
- **Drift:** the model must be retrained when the data moves; the drift check warns but doesn't retrain.
- **The investigation agents** (LLM) are covered separately: [agent-benchmark.md](agent-benchmark.md).

## 6. Responsible use

- The model **prioritises** work; a person decides. Investigation decisions need a second person and a written reason, both audited.
- Every score is stored with the exact model version, threshold and inputs that produced it.
- Explanations describe **the model**, not the cause in the real world.
- Never use the words fraud or fraudster about a flagged claim; the agents' reports are checked for this in code.
