# FWA

**CS 544 / CS 590 — Machine Learning in Production (Fall 2026)**

**Team:** Javed (governance/configuration), Irfan (ML/software development), Vipul (data engineering/databases).  
**Project window:** October 10–November 27, 2026.  
**Status:** T01 technical access validation completed; team scope approval and AWS cost confirmation still to record.

## Objective

Build a Databricks-first application that ranks healthcare providers for **human fraud, waste, and abuse investigation**. Scores are investigation-prioritization signals, not confirmed fraud findings and never automatic denials. Use public and/or synthetic data only.

## Platform

- **Data:** Unity Catalog, Bronze/Silver/Gold Delta tables, Lakeflow Jobs/pipelines, PySpark
- **Models:** scikit-learn, MLflow, Unity Catalog Model Registry, Databricks Model Serving
- **Application and telemetry:** Databricks Apps, Lakebase PostgreSQL, Lakebase CDF, Delta audit tables
- **Monitoring:** Databricks AI/BI dashboards and SQL Alerts
- **Delivery:** Databricks Declarative Automation Bundles, GitHub; Docker scorer required for M3

Successful access tests are documented in [`docs/T01/FWA_Databricks_Capability_Test_Runbook.md`](docs/T01/FWA_Databricks_Capability_Test_Runbook.md).

## Repository layout

| Path | Purpose |
|---|---|
| `app/` | Working Databricks Apps prototype (Gradio now; FastAPI recommendation API later) |
| `src/fwa/` | Reusable Python ingestion, features, training, scoring, evaluation, telemetry code |
| `notebooks/` | Notebook entrypoints, experiments, and exploration |
| `resources/` | Databricks Bundle resource YAML definitions for Jobs/pipelines |
| `sql/` | Analytical SQL, metric and monitoring queries |
| `tests/` | Unit, integration, and data-quality tests |
| `config/` | Non-secret environment, model and quality configuration |
| `analysis/` | M4 fairness, feedback-loop and security investigation notebooks/scripts |
| `docs/T01/` | Project charter, access decision, and capability test runbook |
| `docs/milestones/` | Submission evidence for M1, M2, M3, M4 |
| `docs/planning/` | Master project task tracker |
| `docs/architecture/` | System architecture diagram |
| `.github/workflows/` | GitHub Actions workflows when implemented |
| `databricks.yml` | Main Databricks Bundle configuration (minimal initial skeleton) |

## Development conventions

1. Create a feature branch per task, e.g. `T02-ingestion-schema`.
2. Open a pull request to `main` and request at least one teammate's review.
3. Link tests, notebooks, dashboard screenshots, and MLflow runs in the pull request or milestone document.
4. Do not commit private credentials, real patient/company data, source archives, or model/data artifacts. Keep data in approved storage and record its version.
5. Record deployed model version, MLflow run ID, training Delta version, pipeline Git commit, and prediction ID for provenance.

## T01 closure

See [`docs/T01/T01_Project_Charter_and_Access_Decision.md`](docs/T01/T01_Project_Charter_and_Access_Decision.md). Before closing the T01 issue, record **team scope approval** and **AWS cost/alert review**. Some later tests (native profiling/drift, Docker and GitHub Actions) remain pending.
