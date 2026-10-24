# M1 Proof of Concept and Roadmap

**Owner:** Irfan (doing all M1 work)  
**Written:** Oct 18, 2026  
**Assumed M1 due date:** Fri Oct 23, 2026 (confirm on the course page)

## 1. Status at a glance

| Task | Item | Status |
|---|---|---|
| T01 | Charter + Databricks capability decision | Done except recording scope approval and AWS cost note |
| T02 | Repo scaffold + Bundle smoke test | Done |
| T03 | Dataset chosen, profiled, data dictionary | **Done in this POC** (`data_dictionary.md`) |
| T04 | Metric spec (4 qualities) + split rules | Draft in section 4; confirm against the rubric |
| T05 | Governance note + model card template | To do |
| T06 | Provider-disjoint split + train-only preprocessing | **Proven locally** (`src/fwa/split.py`); port to Databricks |
| T07 | LR baseline in MLflow | **Proven locally**; port to Databricks + MLflow |
| T08 | Bronze ingestion in Unity Catalog | To do |
| T09 / T14 | Provider Gold features + Delta manifest | **Logic proven locally** (`src/fwa/features.py`); port to PySpark |
| T12 | HGB challenger + comparison | **Proven locally**; port to Databricks + MLflow |
| T13 | FastAPI `/health` + `/recommendations` | To do (reuse lab Module 8 app) |
| T15 | Ranked queue in Lakebase | To do (Lakebase must be recreated) |
| T10 | M1 report, four sections | To do |
| T11 | Acceptance check + submit | To do |

## 2. What the POC is

A local, pandas + scikit-learn version of the whole M1 modeling path on the **real Kaggle data**:

```
4 Train CSVs -> provider Gold features (26) -> provider split 70/15/15 -> LR and HGB
             -> 4 quality measures -> champion -> ranked top-20 providers
```

It runs in about 15 seconds and confirms the plan works before spending Databricks quota.

### Files

| Path | Purpose |
|---|---|
| `src/fwa/features.py` | Builds one row per provider from claims + patients (no label) |
| `src/fwa/split.py` | Stratified, provider-disjoint split + manifest with seed and SHA-256 of the assignment |
| `src/fwa/evaluate.py` | Precision@K, PR-AUC, latency, robustness, deterministic ranking |
| `poc/run_m1_poc.py` | End-to-end runner |
| `tests/test_split_and_ranking.py` | 7 unit tests (no data needed) |
| `docs/milestones/M1/poc_results/` | Metrics, comparison table and split manifest from the run below |

### How to run

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests
python poc/run_m1_poc.py --data-dir ../archive --out-dir poc/output
```

`poc/output/` is ignored by Git (it contains the full provider split with labels).

## 3. POC results (seed 42)

**Split:** 3,787 / 811 / 812 providers, with 354 / 76 / 76 fraud (9.4% in every set). No provider appears in two sets.

| Model | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Test P@20 | Test P@50 | p95 latency | Top-20 overlap under 5% noise |
|---|---|---|---|---|---|---|---|
| Logistic Regression | **0.753** | 0.709 | 0.944 | 0.90 | 0.78 | 4.5 ms | 0.84 |
| HistGradientBoosting | 0.713 | 0.693 | 0.932 | 0.95 | 0.68 | 19.3 ms | **0.96** |

- **Champion by validation PR-AUC: Logistic Regression.** The test set was used once per model, after the choice.
- Base rate is 9.4%, so P@20 of 0.90 means about **9.6 times better than random** review.
- **Trade-off worth discussing in the report:** LR ranks slightly better, HGB is much more stable under input noise (0.96 vs 0.84 overlap).
- Latency is **in-process** scoring of 812 providers. The real API latency (network + serving) must be measured in T13.

### Known issues to fix in the Databricks version

1. **LR scores saturate at 1.0** for the top providers, so ties are broken by Provider ID. Log-transform heavy-tailed features (amount sums, counts) before scaling to spread scores out.
2. Only 76 fraud providers in each holdout set, so P@20 moves in steps of 0.05. Report it with that caveat, or add P@50.
3. HGB uses defaults. Any tuning must use the validation set only.

## 4. Draft of the four M1 qualities (T04)

Confirm the exact required qualities in the course rubric, then fill each as metric / data / operationalization.

| Quality | Metric | Data | How it is measured |
|---|---|---|---|
| Ranking usefulness | Precision@20 | test providers | share of top-20 that are labeled Yes |
| Overall ranking quality | PR-AUC | test providers | average precision over all scores |
| Responsiveness | p95 latency (ms) | 50+ requests to `/recommendations?limit=20` | time from request to ranked response |
| Robustness | top-20 overlap | test features with 5% multiplicative noise, 20 trials | share of the original top 20 still in the top 20 |

**Split and leakage rules:** split on unique Provider IDs before any aggregation; stratify on the label; fixed seed 42; fit imputer and scaler on train only (inside the sklearn pipeline); never use `Provider`, `PotentialFraud`, raw IDs or demographics as inputs; select the champion on validation; touch test once.

## 5. Roadmap to finish M1

| Day | Work | Done when |
|---|---|---|
| **Sun 18 (night)** | `git pull`, branch `M1-pipeline`, commit this POC; save official M1 instructions in this folder | POC on GitHub |
| **Mon 19** | T08: upload 4 Train CSVs to a UC Volume, create 4 Bronze Delta tables. T09/T14: port `features.py` to PySpark, write `provider_features_gold`, record Delta version. T05: one-page governance note | Gold has 5,410 rows, one per provider, version recorded |
| **Tue 20** | T06: write `provider_split` Delta table from `split.py` (same seed, same SHA-256). T07: LR in MLflow with probabilities, PR-AUC, P@20, seed, Gold Delta version, Git SHA. Fix LR saturation | LR run logged and reproducible |
| **Wed 21** | T12: HGB on the same split, comparison table on all 4 qualities, register champion in UC with `@champion`, deploy serving endpoint (lab Module 5 code) | Champion served |
| **Thu 22** | T13: FastAPI `/health` and `/recommendations?limit=20` on Databricks Apps (lab Module 8). T15: recreate Lakebase, store request_id, provider, rank, score, model version, data version, timestamp. Measure real p95 latency | Endpoint returns ranked top-20 |
| **Fri 23** | T10: four report sections (Learning, Model comparison, Prediction service, Team process). T11: clean rerun, overlap = 0 check, screenshots and links. Submit | M1 submitted |

**If time runs short, cut in this order:** T15 Lakebase (describe as next step), robustness detail, filled model card (keep the template). Never cut: two models on the same provider split, four measured qualities, working `/health` + top-20, report.

## 6. Open questions

1. Exact M1 rubric qualities and due date.
2. Kaggle dataset license wording for the report.
3. Team process section: describe the real division of work honestly.
