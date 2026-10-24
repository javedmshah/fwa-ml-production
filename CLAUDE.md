# CLAUDE.md: context for Claude Code

Read this first. It carries over the planning done in a previous Claude chat.

## Project

**FWA Investigation Prioritization** (CS 544/590, Machine Learning in Production, UIC, Fall 2026).
Rank healthcare **providers** for human fraud, waste and abuse review. Scores prioritize investigations; they are **never** fraud findings and **never** auto-deny anything.

- Data: Kaggle "Healthcare Provider Fraud Detection Analysis". Provider-level proxy label `PotentialFraud` (506 of 5,410 = 9.4%). Test CSVs have no labels, so all holdouts come from Train providers. Details: `docs/milestones/M1/data_dictionary.md`.
- Platform: **Databricks Free Edition only** (serverless compute, `workspace` catalog, Unity Catalog, Delta, MLflow, Model Serving, Databricks Apps, Lakebase, AI/BI dashboards, SQL Alerts, Bundles). Plus GitHub, and a Docker scorer for M3.
- Milestones: M1 model comparison + recommendation API; M2 evaluation, telemetry, tests, CI; M3 serving, Docker, retraining, monitoring, A/B, provenance; M4 fairness, feedback loops, security.

## Who does what

Irfan is doing **all** remaining work himself (teammates are busy). Ignore the per-person split in `docs/planning/FWA_Project_Task_Tracker.xlsx`; use its task IDs (T01 to T45) only as a checklist.

## Current state (Oct 18, 2026)

- T01, T02 done. T03 done (data dictionary).
- **Local POC works end to end** on the real data: `poc/run_m1_poc.py` (features, split, LR vs HGB, 4 qualities, top-20). Results and roadmap: `docs/milestones/M1/M1_POC_and_Roadmap.md`.
- Next: port the POC to Databricks, following the roadmap in that doc (Bronze, Gold, split table, MLflow runs, champion, serving, FastAPI app, Lakebase queue, report).
- `app/` still holds the old Gradio smoke test; it will be replaced by the FastAPI app (T13).

## Commands

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests                                        # unit tests, no data needed
python poc/run_m1_poc.py --data-dir ../archive --out-dir poc/output   # full POC (~15 s)
```

Raw Kaggle CSVs live in `../archive/` (outside the repo).

## Rules

1. **Never commit data:** no CSVs, no `poc/output/`, no model files, no credentials. Check `.gitignore`.
2. **Provider-disjoint split always.** Split on unique Provider IDs, stratified, seed 42. Fit preprocessing on train only. Pick the champion on validation; use test once.
3. **Never use as model inputs:** `Provider`, `PotentialFraud`, raw IDs, Race, Gender, State, County.
4. **Feature logic must match** between training (Gold) and serving (API), to avoid training-serving skew. Keep it in `src/fwa/` and reuse it.
5. **Record provenance:** model version, MLflow run ID, Gold Delta version, Git commit SHA, request/prediction ID.
6. **Writing style:** never use em dashes in any docs, comments or reports Claude writes. Plain, simple English.
7. One branch per task, PR to `main`.

## Databricks lessons already learned (save time)

- Spark 4 is ANSI strict: use `try_cast` when converting raw text.
- `%sql` must be the first line of its own notebook cell.
- Serverless streaming: use `trigger(availableNow=True)`; always-on streams are not available.
- Log sklearn models with `pyfunc_predict_fn="predict_proba"` so serving returns probabilities, not 0/1.
- Moving the `@champion` alias does not update a serving endpoint. Redeploy by code (compare served version with champion version).
- A "top 10%" threshold breaks when scores saturate. Route with a policy threshold and sort by score.
- Databricks Apps: the Lakebase resource key is `database` (`valueFrom: database`). The serving endpoint could not be added as an app resource in the new UI: set `SERVING_ENDPOINT` as a plain `value` and grant the app's service principal **Can Query** on the endpoint.
- Deploy apps from a workspace folder path; the Deployments tab unlocks after the first deploy.
- Git folders cannot hold MLflow experiments, dashboards or alerts. Keep experiments outside the Git folder (for example `/Users/<me>/fwa_training`).
- SQL Alerts need a schedule before "Run alert" works; set the condition column, operator and value explicitly.
- Free Edition quota: stop the app, pause job/alert schedules, use scale-to-zero serving, `workload_size="Small"`.
