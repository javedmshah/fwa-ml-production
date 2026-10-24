# M1 Completion Roadmap

**Owner:** Irfan (all tasks)  
**Written:** Mon Oct 19, 2026  
**Assumed due date:** Fri Oct 23, 2026 (confirm on the course page)  
**Based on:** the course M1 handout (movie recommendation version), adapted to FWA provider ranking. See `M1_POC_and_Roadmap.md` for the POC results this plan builds on.

## 1. What M1 needs, in our terms

| Handout item | Our version |
|---|---|
| Train at least two models | Logistic Regression (LR) and HistGradientBoosting (HGB) on the same provider-disjoint split |
| Compare four qualities | (1) accuracy, (2) training cost, (3) inference cost, (4) model size, each as metric / data / operationalization |
| Pick a production model | Champion chosen on validation PR-AUC, with trade-offs explained |
| Inference service, top 20 in ranked order, under 800 ms | FastAPI `GET /recommendations?limit=20` returns provider IDs ranked by risk score |
| Report with four sections | Learning, Model comparison, Prediction service, Team process |

## 2. Rubric checklist

Each line is pass/fail. Tick only when the evidence exists.

| Pts | Requirement | Evidence we will provide | Task |
|---|---|---|---|
| 10 | Data + learning technique described, with reasons | Report section 1 | R1 |
| 10 | Link to training code | GitHub link to the training notebook/script | D4, D5 |
| 5 | Accuracy: metric, data, operationalization, code link, result | Report section 2 + `src/fwa/evaluate.py` | Q1 |
| 5 | Training cost: same | Report section 2 + measurement code | Q2 |
| 5 | Inference cost: same | Report section 2 + measurement code | Q3 |
| 5 | Model size: same | Report section 2 + measurement code | Q4 |
| 10 | Argue which model goes to production | Report section 2, last paragraph | R2 |
| 10 | Service described, ranking explained | Report section 3 | R3 |
| 10 | Link to service code | GitHub link to `app/` | S1 |
| 10 | Team organization + notes with who / what / by when | Report section 4 + this file (section 5 table) | R4 |
| 10 | 2000 successful requests in 24 h | **Ask instructor.** Plan: our own load test of 2000 requests, report success rate and p95 | S3 |
| 10 | Personalized answers for 10% of requests | **Ask instructor.** Plan: `GET /score/{provider_id}` gives a different answer per request | S2 |

## 3. The four qualities (draft for the report)

| Quality | Metric | Data gathered | Operationalization |
|---|---|---|---|
| (1) Prediction accuracy | PR-AUC (primary), Precision@20 (secondary) | 812 test providers (76 labeled Yes), scored once after champion selection | PR-AUC = sklearn `average_precision_score(y, score)`. P@20 = share of the 20 highest-scored providers labeled Yes (ties broken by Provider ID) |
| (2) Training cost | Training wall-clock time in seconds; peak memory in MB | 3,787 train providers, 26 features, same compute for both models | Time `pipeline.fit(X_train, y_train)` with `time.perf_counter()`, 5 repeats, report the median. Peak memory with `tracemalloc` during one fit |
| (3) Inference cost | Throughput (providers scored per second, in process); p95 latency of the API in ms | In process: 812 test providers, 50 repeats. API: 200 calls to `/recommendations?limit=20` after 10 warm-up calls | Throughput = 812 / median time of `predict_proba(X_test)`. API p95 = 95th percentile of client-measured request time |
| (4) Model size | Serialized model size in KB | The fitted sklearn pipeline (imputer + scaler + model) | Size in bytes of `joblib.dump(pipeline, file)`, also the size of the MLflow model folder |
| (extra) Robustness | Top-20 overlap under noise | Test features times (1 + N(0, 0.05)), 20 trials | Average share of the original top 20 still in the top 20 |

## 4. Prediction service spec

- `GET /health` returns status, model name, model version.
- `GET /recommendations?limit=20` returns a list of up to 20 objects `{rank, provider_id, score}`, highest score first. Optional plain text form: comma-separated provider IDs on one line, like the handout.
- `GET /score/{provider_id}` returns that provider's score and rank.
- Ranking: score = champion `predict_proba` class 1 from the Model Serving endpoint; sort by score descending; ties broken by Provider ID ascending.
- Every response carries `request_id`, model version, Gold Delta version. Responses go into the Lakebase queue table if time allows.
- Target: p95 under 800 ms (the handout limit).
- Scores rank providers for human review. They are never fraud findings and never auto-deny anything.

## 5. Day-by-day plan (also serves as the work-division notes)

| ID | Task | Owner | Due | Done when |
|---|---|---|---|---|
| **Mon Oct 19** | | | | |
| P1 | Move the M1 handout into `docs/milestones/M1/`, commit POC on branch `M1-pipeline` | Irfan | Mon | Files on GitHub |
| P2 | Add training time, peak memory, throughput and model size to `poc/run_m1_poc.py` | Irfan | Mon | All four qualities have local numbers for LR and HGB |
| P3 | Fix LR score saturation (log1p on amount and count features) | Irfan | Mon | Fewer tied top scores, rerun table saved |
| D1 | T08: upload 4 Train CSVs to a UC Volume, create 4 Bronze Delta tables | Irfan | Mon | Bronze row counts match the CSVs |
| **Tue Oct 20** | | | | |
| D2 | T09/T14: port `features.py` to PySpark, write `provider_features_gold` | Irfan | Tue | 5,410 rows, one per provider, Delta version recorded |
| D3 | T06: write `provider_split` table from `split.py` (seed 42, same SHA-256) | Irfan | Tue | Counts 3,787 / 811 / 812, zero overlap |
| D4 | T07: LR run in MLflow (probabilities, all four qualities, seed, Gold version, Git SHA) | Irfan | Tue | Run logged, rerun gives same metrics |
| G1 | T05: one-page governance note (review-only use, no auto-deny, excluded inputs) | Irfan | Tue | Note in `docs/milestones/M1/` |
| **Wed Oct 21** | | | | |
| D5 | T12: HGB run in MLflow on the same split, same measurements | Irfan | Wed | Both runs in one experiment |
| D6 | Comparison table from MLflow, pick champion on validation, test once | Irfan | Wed | Table saved, champion named |
| D7 | Register champion in UC with `@champion`, deploy scale-to-zero serving endpoint | Irfan | Wed | Endpoint returns probabilities |
| **Thu Oct 22** | | | | |
| S1 | T13: FastAPI app on Databricks Apps (`/health`, `/recommendations`, `/score`) replacing the Gradio smoke test | Irfan | Thu | Top 20 returned in ranked order |
| S2 | Feature logic reused from `src/fwa/` (no skew), one unit test for the endpoints | Irfan | Thu | Tests pass |
| S3 | Load test: 2000 requests, success rate and p95 | Irfan | Thu | Numbers saved for the report |
| S4 | T15: Lakebase queue table (request_id, provider, rank, score, versions, time) | Irfan | Thu | Rows written per request (cut first if late) |
| **Fri Oct 23** | | | | |
| R1 to R4 | T10: write the four report sections with links and screenshots | Irfan | Fri noon | Every rubric line in section 2 ticked |
| A1 | T11: clean rerun, overlap = 0 check, no data in Git, merge PR to `main`, submit | Irfan | Fri | Submitted on Gradescope |
| A2 | Stop the app, pause schedules, endpoint scaled to zero | Irfan | Fri | Quota safe |

## 6. If time runs short

Cut in this order: S4 Lakebase (describe as next step), robustness detail, `/score` endpoint, filled model card.  
Never cut: two models on the same split, all four required qualities measured, working service returning a ranked top 20, the report.

## 7. Risks

| Risk | Plan |
|---|---|
| Free Edition quota or serving cold starts push latency past 800 ms | Warm up before the load test; report cold and warm p95 separately |
| Databricks port takes longer than one day | POC numbers already exist locally, so the report can use them while the port finishes |
| Grading for the 2000-request and personalization lines is unclear | Ask instructor on Mon; our load test is the fallback evidence |
| Only 76 positives in test | Report P@20 with the 0.05-step caveat, add P@50 |

## 8. Questions for the instructor (send Mon)

1. Confirm the M1 due date.
2. How are the "2000 requests" and "personalized 10%" lines graded for a research project with no course traffic?
3. Does a one-person team still submit meeting notes, or is this plan with owners and due dates enough?
