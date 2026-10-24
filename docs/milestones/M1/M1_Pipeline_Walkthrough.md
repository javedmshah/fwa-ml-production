# M1 Pipeline Walkthrough (plain-language guide)

**What this is:** a step-by-step explanation of everything built for Milestone 1 on Oct 23, 2026: every notebook cell, the model, the serving endpoint, and the API app. It is written for teammates who do not code much. You do not need to read the code to follow it.

**Where things live (Databricks Free Edition workspace):**

| Thing | Name / location |
|---|---|
| Training notebook | `M1_train_and_register` (in Irfan's workspace folder) |
| MLflow experiment | `/Users/ishai21@uic.edu/fwa_m1_training` |
| Registered model | `workspace.default.fwa_provider_ranker`, version 1, alias `@champion` |
| MLflow run of the champion | `f89c359d142a4d0898494b90374918c0` |
| Serving endpoint (the model online) | `fwa-provider-ranker` |
| API app | `fwa-provider-ranking` (Databricks Apps) |
| App source folder | `/Workspace/fwa_app_source` |

---

## 1. The big picture

**Goal:** rank healthcare providers so a human reviewer knows who to look at first. The output is the **top 20 riskiest providers**. A score is only a ranking signal. It is **not** a fraud finding, and nothing is denied automatically.

```
Kaggle data
   |
Gold table (one row of features per provider)  +  labels (PotentialFraud Yes/No)
   |
Frozen split: train 70% / validation 15% / test 15% of providers
   |
Train 3 candidates: simple baseline, Logistic Regression, HistGradientBoosting
   |
Pick the champion on validation, test once, register it (@champion)
   |
Model Serving endpoint "fwa-provider-ranker"  (the model as an online service)
   |
FastAPI app "fwa-provider-ranking"  ->  /health  and  /recommendations?limit=20
```

**The four quality measures (defined in T04):**

| Code | Quality | What it means in plain words |
|---|---|---|
| Q1 | Ranking usefulness | Of the top 20 providers, how many have the label "Yes"? (Precision@20). PR-AUC is a second, whole-list score |
| Q2 | Responsiveness | How long one ranking request takes (p95 = 95% of requests are faster than this) |
| Q3 | Robustness | If the input numbers wobble a little, how much does the top 20 get worse? |
| Q4 | Auditability | Does every top-20 record carry all the IDs needed to trace where it came from? |

---

## 2. The data

- **Source:** Kaggle "Healthcare Provider Fraud Detection Analysis" (public data).
- **Label:** `PotentialFraud` (Yes/No), one per provider. 506 of 5,410 providers are "Yes" (9.4%). It is a **proxy label**, not a proven fraud finding.
- **Gold table** `workspace.default.gold_provider_features`: one row per provider, 5,410 rows, built by the data engineering step. It summarizes each provider's claims: number of claims, amounts reimbursed, number of patients, length of stay, diagnosis variety, physicians, patient age and chronic conditions, peer comparisons, and so on.
- **Labels table** `workspace.default.irfan_provider_labels`: the Kaggle file `Train-1542865627584.csv` loaded as a table (Cell 2).

**Important:** the Kaggle "Test" files have **no labels**, so all evaluation uses held-out providers from the labeled "Train" file.

---

## 3. The notebook, cell by cell

### Cell 1: Settings
**What it does:** writes down, in one place, the names of every table, the model, the experiment, the random seed (42) and the top-K size (20).
**Why:** every later cell reads these names, so we change them in one spot only. The experiment folder is built from the logged-in user's name, so the notebook works for any teammate without editing.
**Output:** nothing printed.

### Cell 2: Create the labels table (one-time step)
**What it does:** reads the Kaggle labels CSV we uploaded to a Databricks Volume (`/Volumes/workspace/default/irfan_files/`) and saves it as a table.
**Why:** the shared labels table was not readable with our permissions, and the labels are public data, so we recreated them.
**Output:** No 4,904 and Yes 506 providers.

### Cell 3: Load the features and labels
**What it does:**
1. Reads the Gold table (version 1) into memory.
2. Converts any "decimal" number columns to normal numbers, so none are dropped by mistake.
3. Removes any column that looks like the label (safety check against cheating; none were found).
4. Keeps only numeric columns as model inputs: **42 features**. Text columns such as `size_band` and `peer_group`, and date columns, are skipped.
5. Checks that every labeled provider has a Gold row.

**Output:** `Gold v1: 5410 providers, 42 numeric features`, labels {No: 4904, Yes: 506}.

### Cell 4: Freeze the provider split
**What it does:** divides the 5,410 providers into three groups:

| Group | Providers | "Yes" providers | Used for |
|---|---|---|---|
| Train | 3,787 | 354 | the model learns from these |
| Validation | 811 | 76 | choosing between models |
| Test | 812 | 76 | the final exam, used once |

**Key rules:**
- The split is by **provider**, so a provider is never in two groups. If the model saw a provider during training and was tested on the same provider, the result would be unfairly good.
- It is **stratified**: each group has the same share of "Yes" (9.4%).
- Providers are sorted by ID before splitting and the seed is fixed, so the split is **identical every time**.
- A fingerprint (SHA-256 hash) of the split is saved: `f5ea90429ae0a396...`. It **matches** the hash from the local proof of concept on a laptop, which proves the split is reproducible.

**Output:** the table above, "overlap 0", saved to `workspace.default.provider_split`.

### Cell 5: Prepare the groups and the scoring rules
**What it does:**
1. Attaches the 42 features to every provider in each group.
2. Defines how we rank: sort by score, highest first; if two scores tie, the lower provider ID comes first (so the order never changes randomly).
3. Defines the measures: Precision@20 and PR-AUC (Q1), and the robustness test (Q3).

**Output:** group sizes (3787, 811, 812 rows, 42 columns each), 0 missing values.

### Cell 6: Train three candidates and record them in MLflow
**MLflow** is Databricks' lab notebook for experiments: it saves each training run with its settings, scores and the model file.

| Candidate | Plain explanation |
|---|---|
| **Baseline** | No machine learning. Just rank providers by `total_reimbursed` (who billed the most). It tells us whether ML adds anything. |
| **Logistic Regression (LR)** | A simple, explainable model. Each feature gets a weight. Very large numbers are first compressed (`arcsinh`, similar to a log) and put on the same scale. |
| **HistGradientBoosting (HGB)** | Many small decision trees combined. Catches non-obvious patterns. |

Both models use `class_weight="balanced"`, which tells them to pay extra attention to the rare "Yes" providers. All data preparation (filling gaps, scaling) is learned **from the training group only**, to avoid leaking information from validation or test.

Each run is tagged with: Gold table version, split hash, seed, feature cutoff date and Git commit. The models are saved so that, when served, they return a **probability between 0 and 1**, not just Yes/No.

**Validation results:**

| | P@20 | PR-AUC | ROC-AUC |
|---|---|---|---|
| Baseline | 0.90 | 0.717 | 0.947 |
| LR | 0.90 | 0.722 | 0.959 |
| HGB | 0.95 | 0.727 | 0.950 |

### Cell 7: Pick the champion, test once, register
**What it does:**
1. Applies the rule we fixed **before** looking at test results: the champion is the model with the **highest validation PR-AUC**. That is **HGB**.
2. Scores every candidate on the **test** group, once.
3. Runs the robustness test (Q3): multiplies inputs by small random noise (5% and 10%, 10 times each) and records the worst drop in P@20.
4. Registers HGB in Unity Catalog as `fwa_provider_ranker` **version 1** with the alias **`@champion`**, plus lineage tags.

**Test results:**

| | Test PR-AUC | Test P@20 | Test ROC-AUC | Q3 worst P@20 drop |
|---|---|---|---|---|
| Baseline | 0.694 | 0.80 | 0.942 | 0 points |
| LR | 0.717 | 0.90 | 0.944 | 5 points |
| **HGB (champion)** | **0.722** | **0.95** | 0.930 | 5 points |

**How to read it:** HGB's top 20 contains 19 "Yes" providers. Random picking would find about 2 (9.4% of 20). A 5-point drop means at most one provider in the top 20 changed under noise.

## M1: MLflow Model Comparison Summary

The MLflow results suggest that **Histogram Gradient Boosting (HGB) is the stronger candidate** for our FWA provider investigation-ranking system, although its advantage over Logistic Regression (LR) is modest.

HGB achieves a test **Precision@20 of 0.95** (19 of 20 providers), compared with LR's 0.90 (18 of 20). HGB also slightly outperforms LR in PR-AUC (0.722 vs. 0.717), while LR achieves better ROC-AUC (0.944 vs. 0.930). Both models were evaluated on data with a 9.4% positive-label prevalence. These labels represent potential fraud rather than independently confirmed fraud.

HGB demonstrates promising inference performance, with median latency of 158 ms, P95 latency of 188 ms, and zero recorded failures. Both models experience a maximum five-percentage-point Precision@20 decline in robustness testing. HGB also achieves 100% replay ranking agreement and trace completeness. However, corresponding latency and traceability measurements are missing for LR.

Importantly, a simple baseline ranking providers by total reimbursement appears surprisingly competitive, raising questions about whether the models primarily capture billing volume rather than suspicious behavior.

**HGB is provisionally preferred for production**, but model selection should follow complete performance comparisons, ranking-stability testing, and investigation of possible confounding factors and fairness implications.

### Cell 8: Score the test providers and check traceability (Q4)
**What it does:** ranks all 812 test providers with the champion and adds every field from the T04 output contract:

| Field | Meaning |
|---|---|
| `rank`, `provider_id`, `risk_score` | the ranking itself |
| `model_version`, `run_id` | which model produced it |
| `scored_at_utc` | when |
| `dataset_snapshot_id` | which data version (`gold_provider_features@v1`) |
| `feature_cutoff_at` | last date of data the features can use (2009-12-31) |
| `score_semantics` | reminder that a score is a proxy-label rank, not a fraud finding |

Then it checks the top 20: unique providers, ranks 1 to 20, sorted correctly, all fields present.
**Output:** **Q4 trace completeness 20/20**, contract check True. Saved to `workspace.default.test_cohort_scores`.

### Cell 9: Put the model online (Model Serving)
**What it does:** creates the serving endpoint **`fwa-provider-ranker`**, which runs the `@champion` model (version 1) as an online service that answers scoring requests.
- **Size:** Small. **Scale to zero:** on (it sleeps when unused, so it does not use Free Edition quota while idle).
- If the endpoint already exists and already serves the champion, it does nothing. If the champion changed, it updates the endpoint.
- The first creation took about 10 to 20 minutes (Databricks builds a small server with the model and its libraries).

**Output:** `READY`, serving version 1.

### Cell 10: Save the scoring cohort
**What it does:** saves the 812 test providers and their 42 features as the table `workspace.default.recommendation_cohort`, with three trace columns (data snapshot, cutoff date, split hash).
**Why:** this is the frozen group the API ranks. Saving it in a table we own avoids needing extra permissions on the shared Gold table.

### Cell 11: Check the endpoint gives the same answers
**What it does:** sends 5 providers to the online endpoint and compares its scores with the model in the notebook.
**Output:** identical scores (`Match: True`), first response in 0.12 s. This proves the online model is exactly the trained model.

### Cell 12: Hand off files to the app
**What it does:** writes two files into the app folder `/Workspace/fwa_app_source`:
- `cohort.csv`: the 812 providers and their features (all as numbers, exactly as the model expects).
- `app_config.json`: model name and version, run ID, endpoint name, data snapshot, cutoff date, split hash, score meaning, and the list of 42 feature names.

**Why separate:** the training notebook is the **model layer**; the app is the **service layer**. They only share these two handoff files. When the model is retrained, these two files are exported again; the app code does not change.

### Cell 13: Give the app permission to use the model
**What it does:** every Databricks App runs as its own "robot account" (a **service principal**). This cell gives the app's robot account **Can Query** permission on the endpoint `fwa-provider-ranker`, and prints the permission list to confirm.
**Why:** without it, the app would get "permission denied" when asking the model for scores. The app gets only this one permission (least privilege).

### Cell 14: Measure speed (Q2) and replay (Q4)
**What it does:** this cell is self-contained (it reloads everything from saved tables and the model registry) because the notebook's memory was reset after being idle.
1. One warm-up call to wake the endpoint.
2. 30 timed requests, each scoring all 812 providers and taking the top 20.
3. Compares the endpoint's top 20 with the stored top 20 from Cell 8.

**Output:**
- **Q2:** p50 0.158 s, **p95 0.188 s**, 0 failures out of 30.
- **Q4 replay:** the endpoint's top-20 order matches the stored scores (True).

**Note:** right after the endpoint has been idle, the first request takes about 35 s (it has to wake up, a "cold start"). That is the trade-off of scale-to-zero, which saves Free Edition quota. We report it separately from the warm p95.

---

## 4. The API app (service layer)

The app lives in `/Workspace/fwa_app_source` and runs on **Databricks Apps** under the name **`fwa-provider-ranking`**. It has five files:

| File | Written by | Plain explanation |
|---|---|---|
| `app.py` | by hand | The program. Uses **FastAPI**, a Python tool for building web APIs. |
| `app.yaml` | by hand | Tells Databricks how to start the app (`python app.py`) and which endpoint to call (`SERVING_ENDPOINT = fwa-provider-ranker`). |
| `requirements.txt` | by hand | The Python libraries the app needs: fastapi, uvicorn (the web server), pandas, databricks-sdk. |
| `cohort.csv` | Cell 12 | The 812 providers to rank. |
| `app_config.json` | Cell 12 | Model and data identifiers for traceability. |

**How `app.py` works when it starts:** it reads `app_config.json` and `cohort.csv`, connects to Databricks with the app's own robot identity, and waits for requests.

### The API endpoints

| Address | What you get |
|---|---|
| `/` | A short description of the service and the list of endpoints |
| `/health` | Is everything OK? Shows the endpoint state, the model version being served, whether it matches the config (`version_match`), and how many providers are in the cohort |
| `/recommendations?limit=20` | The ranked list (details below) |
| `/docs` | An interactive web page, generated by FastAPI, where you can try each endpoint with a button |

**What `/recommendations?limit=20` does, step by step:**
1. Checks there are at least 20 providers (otherwise it answers "top-20 not evaluable", as T04 requires).
2. Creates a unique `request_id` and starts a timer.
3. Sends all 812 providers' features to the model endpoint in one request.
4. Gets back a probability for each provider.
5. Sorts by score (highest first, ties by provider ID) and keeps the top N.
6. Returns each provider with all contract fields, plus `request_id`, `providers_scored` and `latency_ms`.

If the model endpoint is unreachable, it answers with error 503 and a message, instead of crashing.

**Sample `/health` response:**
```json
{"status": "ok", "serving_endpoint": "fwa-provider-ranker", "endpoint_state": "EndpointStateReady.READY",
 "model_name": "workspace.default.fwa_provider_ranker", "served_model_version": "1",
 "config_model_version": "1", "version_match": true, "cohort_providers": 812,
 "dataset_snapshot_id": "workspace.default.gold_provider_features@v1"}
```

**Sample `/recommendations?limit=20` (first item):**
```json
{"rank": 1, "provider_id": "PRV55462", "risk_score": 0.999985, "model_version": "1",
 "run_id": "f89c359d142a4d0898494b90374918c0", "scored_at_utc": "2026-10-24T03:44:30Z",
 "dataset_snapshot_id": "workspace.default.gold_provider_features@v1",
 "feature_cutoff_at": "2009-12-31",
 "score_semantics": "higher = higher proxy-label risk rank; review-only, not a fraud finding"}
```

---

## 5. How everything connects (traceability chain)

```
Gold table v1  --+
Split hash f5ea9042...  --+--> MLflow run f89c359d... --> Model v1 (@champion)
                          |                                   |
                          |                       Endpoint fwa-provider-ranker
                          |                                   |
                          +--> cohort.csv + app_config.json --> App /recommendations
                                                                 (every row carries
                                                                  model version, run ID,
                                                                  data snapshot, time)
```

From any recommendation row you can walk back to the exact model, the training run, the data version and the split.

---

## 6. Honest findings and limitations (for the report)

1. **A simple baseline is strong.** Ranking by total reimbursement alone reaches PR-AUC 0.694 on test. The models add a modest gain (0.717 and 0.722) and clearly improve the top 20 (0.80 to 0.95). The proxy label is strongly linked to provider size.
2. **Small holdout.** Each holdout group has only 76 "Yes" providers. Differences of about 0.005 in PR-AUC are within noise; P@20 moves in steps of 0.05 (one provider).
3. **Proxy label.** `PotentialFraud` is not an adjudicated finding. Results describe agreement with the proxy label only.
4. **Peer features.** The peer-comparison features in Gold were computed over all providers (without labels). Strictly, T04 says aggregates should use training data only; to revisit in M2.
5. **Pilot run.** A local proof of concept used the same frozen split before the final run. It informed the thresholds; no model or feature choice was tuned on test results.
6. **Cold start.** About 35 s for the first request after the endpoint has been idle (scale-to-zero). Warm p95 is 0.188 s.
7. **Frozen cohort file.** The app ranks a frozen snapshot packaged with the app, not live data. Reading live from Delta is an M2 improvement.

## 7. Not done in M1 (next steps)

- **T15:** store each recommendation request in Lakebase (request ID, providers, ranks, model and data versions).
- Push the notebook to GitHub (blocked by a repo permission issue on Oct 23).
- **M2/M3:** turn the notebook into an automatic pipeline (Lakeflow Job): features, split, train, gate, deploy only if the champion changes, export for the app, monitoring.

## 8. Glossary

| Term | Meaning |
|---|---|
| Gold table | Cleaned, summarized data, one row per provider, ready for the model |
| Proxy label | A label that stands in for the real truth (here: "potential" fraud, not proven) |
| Provider-disjoint split | No provider appears in more than one of train, validation and test |
| Stratified | Each group keeps the same share of "Yes" |
| Seed | A fixed number that makes random steps repeatable |
| Precision@20 (P@20) | Share of the top 20 that are labeled "Yes" |
| PR-AUC | One number for how well the whole list is ordered (1.0 is perfect) |
| p95 latency | 95% of requests finish faster than this |
| MLflow | Databricks' tool that records each training run (settings, scores, model) |
| Unity Catalog | Databricks' catalog of tables and models, with permissions |
| `@champion` | A label on the model version currently in use |
| Serving endpoint | The model running online, answering scoring requests |
| Scale to zero | The endpoint sleeps when idle to save resources; first call wakes it |
| Databricks Apps | Databricks' hosting for small web apps and APIs |
| FastAPI | A Python tool for building web APIs |
| Service principal | The robot account an app uses to access other Databricks resources |
