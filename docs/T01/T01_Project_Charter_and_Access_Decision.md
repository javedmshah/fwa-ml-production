# T01 — Project charter and Databricks access/quotas decision

**Owner:** Javed  
**Planned dates:** October 10–11, 2026  
**Milestone:** M1  
**Closure status:** Technical assessment completed; team scope approval and AWS cost confirmation still need recording.

## 1. Proposed project charter

**Working title:** Healthcare Fraud, Waste, and Abuse (FWA) Investigation Prioritization.

**Problem and product:** Build an ML-powered system that recommends a ranked list of healthcare providers for an investigator to review, with risk scores and explanations. The system prioritizes investigation; it does not automatically determine fraud, deny claims, or make adverse decisions.

**Data boundary:** Use a public benchmark and/or explicitly synthetic claims. No company data, PHI, private credentials, or company models are to enter this coursework environment.

**Timeline:** October 10–November 27, 2026; M1 model comparison and recommendation API; M2 evaluation, telemetry, data checks, tests and CI; M3 serving, containerization, retraining, monitoring, experiments and provenance; M4 fairness, feedback-loop and security analysis.

**Responsibilities:** Javed—scope, governance, configuration, metrics and evidence; Irfan—model development, API and deployment; Vipul—data engineering, storage, telemetry and analysis. All members contribute to reviews and milestone submissions.

**Platform decision:** Databricks Free Edition is the primary development/test environment, supported by Unity Catalog/Delta, Lakeflow, MLflow, Model Serving, Databricks Apps, Lakebase, AI/BI dashboards, and Databricks Bundles. AWS S3 backs an external Unity Catalog location required for the tested Lakebase CDF path. GitHub is used for shared code/CI, and a separately deployed Docker scorer remains required for M3.

## 2. Service-access decision matrix

| Capability group | Result from hands-on testing | Evidence / limitation |
|---|---|---|
| SQL Editor, Unity Catalog and Delta reads/writes | **PASS** | Catalog/schema queries, Delta table creation and insertion |
| Python, model training, MLflow and Unity Catalog Registry | **PASS** | LR/HGB training, logged runs, registered models |
| Model Serving | **PASS** | Serving endpoint returned classes 0 and 1; 0–4 concurrency resolved initial quota block |
| Databricks Apps and App → Model Serving | **PASS** | Custom Gradio UI successfully submitted eight values and displayed inference response |
| Lakebase and App → Lakebase | **PASS** | SQL and app-level read/write produced `read_write_test: PASSED` |
| AWS S3 external catalog and Lakebase CDF → Delta | **PASS** | External-backed catalog enabled working synchronization |
| Lakeflow Jobs, scheduling and retraining | **PASS** | Scheduled trigger available, Job succeeded, model registered/aliased |
| Declarative pipeline + validation rule | **PASS, with qualification** | Pipeline ran; separate dropped-row metrics not independently recorded |
| AI/BI dashboard and SQL alert | **PASS, with qualification** | Dashboard published; alert evaluation OK, threshold-trigger behavior not proven |
| CLI + Databricks Bundles | **PASS, with qualification** | Authentication, bundle validate and deploy succeeded; GitHub Actions deferred |
| Sticky A/B and statistical calculation | **PASS, simulation only** | Stable assignments and Fisher test; no live randomized experiment yet |
| Model/training-data/prediction provenance | **PASS, synthetic demonstration** | Exact model/run/Delta version/audit record traced; live serving audit not yet implemented |
| Native data profiling and statistical drift analysis | **PENDING** | Profiling initiated; completion and drift comparison not confirmed |
| Standalone deployed Docker scorer | **NOT TESTED** | M3 work remains |

**Detailed reproducibility evidence:** `FWA_Databricks_Capability_Test_Runbook.md` and/or `.docx` contains all 21 individual tests, full commands and code, outputs, names, troubleshooting and qualifiers. Upload the runbook alongside this file to your project repository and link it from the T01 GitHub issue.

## 3. Quota and cost decision

- **Observed quota risk:** Free Edition serverless pipeline compute initially returned `RESOURCE_EXHAUSTED`; after freeing resources, the pipeline succeeded. Serving creation initially exceeded quota until configured for **0–4 concurrency**. Availability at this moment does not guarantee unrestricted ongoing capacity.
- **Working constraint:** Use minimal compute, stop idle endpoints/apps, avoid concurrent smoke-test workloads, and prefer lightweight scheduled Jobs. Treat quota errors as a capacity risk to the milestone schedule.
- **External cost exposure:** The AWS S3 bucket/external storage setup is outside the Databricks Free Edition allowance. No AWS bill amount or budget cap was established during these tests. Check AWS Billing and set a budget/alert before representing the cost review as complete.
- **Fallback:** If Free Edition limits block essential demonstrations, sequence tests in separate windows or use a separately approved workspace/deployment for the affected capability. Do not claim production capacity from smoke tests alone.

## 4. What counts as T01 completion

- [x] Project purpose, in-scope functionality, schedule and role split written down.
- [x] Databricks platform choice justified by executed access tests.
- [x] Service-access matrix and reproducible runbook documented.
- [x] Serverless quota limitations and recovery observed and logged.
- [x] Synthetic/public-only data boundary stated.
- [ ] Javed, Irfan and Vipul acknowledge/approve the project scope (for example, comments in the T01 GitHub issue or kickoff meeting notes).
- [ ] AWS cost status is recorded (current observed spend or budget/alert), even if the amount is zero.

**Close T01** after the final two items are documented. Pending profiling, GitHub Actions and Docker checks are deliberately tracked as later-milestone work, not T01 blockers.

## Suggested GitHub issue completion comment

> **T01 access assessment complete:** We tested Databricks Free Edition end-to-end and verified Delta/Unity Catalog, model training and registration, Model Serving, Databricks Apps, Lakebase, S3-backed Lakebase CDF, Lakeflow Jobs and pipelines, dashboards/alert evaluation, retraining, experiment-analysis calculations, Databricks Bundles and model/data/prediction provenance. Full runnable evidence is in the capability runbook. We selected Databricks-first architecture with 0–4 serving concurrency and documented Free Edition compute-quota limits and AWS S3 cost exposure. Team scope approval and AWS cost confirmation are recorded below before closure.
