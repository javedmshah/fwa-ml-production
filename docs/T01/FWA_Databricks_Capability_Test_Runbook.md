# Databricks Capability Validation Runbook

FWA investigation recommendation platform — Executed tests, code and configuration

Record date: 10 October 2026 | Environment: Databricks Free Edition + AWS + Windows

Purpose: a repeatable audit of what this particular Databricks setup was shown to support. The document is a technical evidence register, not a delivery schedule or a substitute for live validation in the final project.

| Verified capability groups | Pending / not started | Known follow-ups |
|---|---|---|
| 19 of 21 | Test 15; Test 21 | Quality counts; alert firing; GitHub Actions; live A/B and serving provenance |

Evidence convention: “Passed” means the specific smoke test described here succeeded. It does not certify production readiness, unrestricted quotas, long-term availability, or that a full course deliverable is complete.

Data policy: all demonstrated feature vectors, provider IDs, reviewer events and experiment outcomes were synthetic. No credentials or private source datasets are included.

# 1. Capability checklist and outcomes

| # | Capability | Status | Recorded evidence |
|---|---|---|---|
| 01 | SQL Editor / workspace | PASS | SHOW CATALOGS ran; workspace visible |
| 02 | Unity Catalog schema visibility | PASS | workspace.default and information_schema visible |
| 03 | Create Delta table | PASS | workspace.default.fwa_access_test created |
| 04 | Insert Delta records | PASS | INSERT succeeded |
| 05 | Python + MLflow run tracking | PASS | fwa_test, metric accuracy=0.95 logged |
| 06 | Train LR and HGB models | PASS | Both training-success messages |
| 07 | Register Unity Catalog model | PASS | fwa_lr_access_test registered |
| 08 | Model Serving / inference | PASS | 0–4 concurrency; prediction returned |
| 09 | Databricks Apps template hosting | PASS | Gradio Hello World deployed/opened |
| 10 | Lakebase PostgreSQL read/write | PASS | Event row created and read |
| 11 | S3 + Lakebase CDF to Delta | PASS | External catalog; history Delta row appeared |
| 12 | Lakeflow Job / scheduling | PASS | Job succeeded; Delta output; schedule visible |
| 13 | Declarative pipeline execution | PASS* | Pipeline worked after quota recovery |
| 14 | Dashboard / SQL alert execution | PASS* | Dashboard published; alert evaluated OK |
| 15 | Native profiling + drift | PENDING | Profiling initiated; completion not confirmed |
| 16 | Custom App → Model / Lakebase | PASS | Prediction [1] and database R/W test passed |
| 17 | Scheduled retraining / model alias | PASS | Job; new version; FWA_TestChampion alias |
| 18 | CLI / Databricks Bundles | PASS* | Login, validation, deploy; Actions deferred |
| 19 | Sticky A/B + Fisher test | PASS | Assignments; p=0.0300, simulated outcomes |
| 20 | Model / data / prediction lineage | PASS | Run IDs and synthetic audit row confirmed |
| 21 | Standalone Docker scorer | NOT RUN | docker version / hello-world not attempted |

> PASS* limitations: Test 13 did not separately confirm 2 retained / 1 dropped in pipeline quality metrics; Test 14 did not prove that the alert fired; Test 18 excluded GitHub Actions CI.

# 2. Environment and resolved configuration

| Item | Configured / observed value |
|---|---|
| Platform | Databricks Free Edition (AWS-backed workspace) |
| Workspace host | dbc-1da8d8d7-1178.cloud.databricks.com |
| Initial Unity Catalog scope | workspace.default |
| External S3 bucket used in test | s3://fwa-databricks-test-2026/ (the bucket name used in our successful test commands) |
| External location | fwa_s3_location; automatic AWS setup succeeded |
| External-backed catalog | fwa_cdf_catalog, schema telemetry |
| Serving endpoint | fwa-test-endpoint; concurrency 0–4 |
| Registered serving model | workspace.default.fwa_lr_access_test (served version 1 in lineage test) |
| Lakebase project | fwa-project-test; branch production |
| Postgres database | databricks_postgres |
| App source folder | fwa_app_source; existing Gradio Hello World app converted to custom app |
| Lakebase app resource key | database — NOT postgres |
| App config binding | ENDPOINT_NAME / valueFrom: database |
| Provenance model / training Delta | workspace.default.fwa_provenance_test / fwa_training_provenance_test |
| Local tooling | PowerShell, Databricks CLI, Declarative Automation Bundles |

> Names are those exercised or supplied during the conversation; recheck them in the workspace before reusing commands in another account. No AWS IAM role ARN, storage-credential ID, personal access token or OAuth secret was supplied or stored.

# 3. Executed tests and runnable code

SQL labeled “Lakehouse SQL” runs in the main Databricks SQL Editor. SQL labeled “Lakebase PostgreSQL” runs in Lakebase’s PostgreSQL editor; the two SQL dialects and permissions differ. Python snippets run in Databricks notebooks unless noted otherwise.

## Test 01–02 — Workspace, SQL and schemas

STATUS: PASS  |  SCOPE: Read-only discovery

OBSERVED: The workspace catalog was visible; workspace schemas included default and information_schema.

```sql
SELECT current_user(), current_catalog(), current_schema();
SHOW CATALOGS;
SHOW SCHEMAS IN workspace;
```

> The word Result on the UI was the query result tab, not a Unity Catalog catalog.

## Test 03–04 — Delta table creation and insertion

STATUS: PASS  |  SCOPE: Unity Catalog writes

OBSERVED: CREATE TABLE returned no rows (normal); INSERT succeeded.

```sql
CREATE TABLE workspace.default.fwa_access_test (
    provider_id INT,
    risk_score DOUBLE
) USING DELTA;

INSERT INTO workspace.default.fwa_access_test
VALUES (1001, 0.42);

SELECT * FROM workspace.default.fwa_access_test;
```

## Test 05 — MLflow experiment tracking

STATUS: PASS  |  SCOPE: Notebook + MLflow

OBSERVED: Experiment run fwa_test recorded accuracy = 0.95.

```python
import mlflow

with mlflow.start_run(run_name="fwa_test"):
    mlflow.log_metric("accuracy", 0.95)

print("MLflow test successful")
```

> 0.95 was a deliberately logged test value, not a measured model accuracy.

## Test 06 — Two model families train

STATUS: PASS  |  SCOPE: scikit-learn in notebook

OBSERVED: Logistic Regression: Training successful; HistGradientBoosting: Training successful.

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

X, y = make_classification(
    n_samples=300, n_features=8, random_state=42
)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "HistGradientBoosting": HistGradientBoostingClassifier()
}
for name, model in models.items():
    model.fit(X, y)
    print(f"{name}: Training successful")
```

## Test 07 — Unity Catalog Model Registry

STATUS: PASS  |  SCOPE: MLflow → Unity Catalog

OBSERVED: Registered workspace.default.fwa_lr_access_test.

```python
import mlflow
import mlflow.sklearn

mlflow.set_registry_uri("databricks-uc")
with mlflow.start_run(run_name="fwa_registry_test"):
    result = mlflow.sklearn.log_model(
        sk_model=models["Logistic Regression"],
        name="model",
        input_example=X[:1],
        registered_model_name="workspace.default.fwa_lr_access_test"
    )
print("Model registration:", result.model_uri)
```

## Test 08 — Model Serving and prediction

STATUS: PASS  |  SCOPE: Serving endpoint, UI query

OBSERVED: Endpoint could be created at 0–4 concurrency; response predictions [0], later [1] through App.

- Serving > Create endpoint > model workspace.default.fwa_lr_access_test > CPU > concurrency 0–4 > Create.
- Higher configuration initially returned “quota exceeded”; reducing to 0–4 permitted deployment.
- Serving endpoint fwa-test-endpoint returned prediction when queried. It was stopped temporarily to free capacity and started again for app integration.

> Serverless quota and endpoint availability may vary by time and Free Edition allowances.

## Test 09 — Databricks Apps template

STATUS: PASS  |  SCOPE: Managed application deployment

OBSERVED: Gradio Hello World template opened and responded to interaction.

- Apps > Create app > Gradio Hello World; deploy and open App URL.
- For custom code, use a persistent workspace source folder fwa_app_source, not a generated deployment snapshot.
- After custom deployment, the working UI showed editable feature fields and a Test Prediction button.

## Test 10 — Lakebase PostgreSQL transaction

STATUS: PASS  |  SCOPE: Lakebase SQL Editor

OBSERVED: fwa-project-test created; one synthetic reviewer event inserted and read.

```sql
CREATE TABLE IF NOT EXISTS fwa_access_events (
    event_id INT PRIMARY KEY,
    provider_id INT,
    reviewer_id TEXT,
    ab_variant TEXT,
    risk_score DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
INSERT INTO fwa_access_events
(event_id, provider_id, reviewer_id, ab_variant, risk_score)
VALUES (1, 1001, 'test_reviewer', 'A', 0.42)
ON CONFLICT (event_id)
DO UPDATE SET risk_score = EXCLUDED.risk_score;
SELECT * FROM fwa_access_events;
```

## Test 11 — External S3 + Lakebase CDF → Delta

STATUS: PASS  |  SCOPE: AWS and Databricks integrations

OBSERVED: External location connected, external-backed catalog created, CDF started and Delta history record appeared.

Preflight in Lakehouse SQL: initial SHOW commands both returned no rows. After creating the AWS S3 bucket and automatic Databricks external location fwa_s3_location, its connection test passed.

```sql
SHOW EXTERNAL LOCATIONS;
SHOW STORAGE CREDENTIALS;

CREATE CATALOG fwa_cdf_catalog
MANAGED LOCATION 's3://fwa-databricks-test-2026/fwa-cdf-managed/';
CREATE SCHEMA fwa_cdf_catalog.telemetry;
DESCRIBE CATALOG EXTENDED fwa_cdf_catalog;
```

Lakebase PostgreSQL source setup:

```sql
CREATE SCHEMA IF NOT EXISTS fwa_cdf_test;
CREATE TABLE IF NOT EXISTS fwa_cdf_test.probe_events (
    event_id INT PRIMARY KEY,
    risk_score DOUBLE PRECISION
);
ALTER TABLE fwa_cdf_test.probe_events REPLICA IDENTITY FULL;
INSERT INTO fwa_cdf_test.probe_events VALUES (1002, 0.73)
ON CONFLICT (event_id)
DO UPDATE SET risk_score = EXCLUDED.risk_score;
```

CDF setup: Source project fwa-project-test, production branch, database databricks_postgres, schema fwa_cdf_test; destination catalog fwa_cdf_catalog, schema telemetry. Verification in Lakehouse SQL:

```sql
SELECT event_id, risk_score, _pg_change_type
FROM fwa_cdf_catalog.telemetry.lb_probe_events_history
WHERE event_id = 1002;
```

> The default workspace catalog was rejected for CDF because it used Databricks Default Storage; the external-backed catalog resolved that issue. External S3 policies, IAM details, and exact CDF runtime metadata were not recorded.

## Test 12 — Lakeflow Job, Delta write and schedule

STATUS: PASS  |  SCOPE: Jobs & Pipelines

OBSERVED: Job succeeded; JOB SUCCESS in logs; Delta row visible and scheduled trigger configurable.

```python
spark.sql("""
CREATE TABLE IF NOT EXISTS workspace.default.fwa_job_test (
    run_id STRING,
    executed_at TIMESTAMP,
    status STRING
) USING DELTA
""")
spark.sql("""
INSERT INTO workspace.default.fwa_job_test
SELECT uuid(), current_timestamp(), 'SUCCESS'
""")
print("JOB SUCCESS")
```

```sql
SELECT * FROM workspace.default.fwa_job_test
ORDER BY executed_at DESC;
```

Job: Jobs & Pipelines > Create Job > Notebook task referencing fwa_job_test > Serverless > Run now. Schedule UI supported a daily trigger (not activated).

> Startup ran longer than ten minutes; logs showed GC/runtime initialization and eventually JOB SUCCESS. Long runtime was not evidence of task failure.

## Test 13 — Declarative Pipeline expectation

STATUS: PASS — pipeline execution; count unconfirmed  |  SCOPE: Data quality

OBSERVED: Initial serverless RESOURCE_EXHAUSTED; after freeing capacity the pipeline succeeded.

```sql
CREATE OR REPLACE TABLE workspace.default.fwa_quality_input
USING DELTA AS
SELECT * FROM VALUES
  (1001, 0.42),
  (1002, 0.93),
  (1003, 1.40)
AS t(provider_id, risk_score);
```

```python
from pyspark import pipelines as dp

@dp.materialized_view(name="fwa_valid_scores")
@dp.expect_or_drop(
    "valid_risk_score",
    "risk_score BETWEEN 0 AND 1"
)
def fwa_valid_scores():
    return spark.read.table("workspace.default.fwa_quality_input")
```

```sql
SELECT * FROM workspace.default.fwa_valid_scores;
```

Expected: two retained rows (0.42, 0.93) and one excluded row (1.40). The expectation filters the materialized-view output; it does not delete the input row. The pipeline eventually worked, but the separate 2/1 dashboard expectation counts were not explicitly confirmed.

## Test 14 — Dashboard + SQL Alert

STATUS: PASS — execution; alert firing unverified  |  SCOPE: AI/BI + SQL

OBSERVED: Dashboard published, SQL alert evaluated with status OK.

```sql
-- AI/BI dashboard dataset
SELECT COUNT(*) AS total_providers,
       ROUND(AVG(risk_score), 3) AS avg_risk
FROM workspace.default.fwa_valid_scores;

-- SQL Alert query
SELECT COUNT(*) AS invalid_scores
FROM workspace.default.fwa_quality_input
WHERE risk_score NOT BETWEEN 0 AND 1;
```

Configured alert condition: invalid_scores > 0. Input data should yield invalid_scores = 1. However, user reported “evaluated OK”, not a confirmed triggered event. Alert notification path and trigger semantics remain to be verified.

## Test 15 — Native Data Profiling and drift

STATUS: PENDING  |  SCOPE: Unity Catalog Quality tab

OBSERVED: Quality tab showed Anomaly detection Disabled and Data profiling Disabled; profiling launched but result not reported.

- Catalog > workspace.default.fwa_valid_scores > Quality > Data profiling > Configure.
- Snapshot/manual refresh was attempted; operation took time and was deferred.
- Not yet confirmed: refresh succeeded, metrics tables generated, drift comparison, or any anomaly-detection alert.

# 4. Working custom Databricks App — Test 16

## Test 16 — App → Model Serving + Lakebase

STATUS: PASS  |  SCOPE: Persistent Gradio app

OBSERVED: Class 0 and class 1 requests succeeded; Lakebase read_write_test: PASSED.

Important resolved configuration: the Lakebase app resource key in this workspace was database (not postgres). The app loaded app.yaml (FWA_CONFIG_CHECK=loaded); using valueFrom: database resolved the missing ENDPOINT_NAME variable. The model URL also needed an explicit https:// scheme.

Workspace source folder: fwa_app_source. The existing Hello World app was redeployed from that folder. Files needed: app.py, app.yaml, requirements.txt.

Final app.yaml:

```yaml
command: ['python', 'app.py']
env:
  - name: ENDPOINT_NAME
    valueFrom: database
  - name: FWA_CONFIG_CHECK
    value: "loaded"
```

Final requirements.txt:

```text
gradio==4.44.0
requests
databricks-sdk>=0.89.0
psycopg[binary]>=3.2,<4
```

Working app.py recorded from the final test:

```python
import os
import gradio as gr
import requests
import psycopg
from databricks.sdk import WorkspaceClient

POSITIVE_EXAMPLE = [
    -1.27416, 1.31799, -2.42388, 0.90695,
    -1.58390, -0.33181, -0.45231, 0.76041
]
ZERO_EXAMPLE = [0.0] * 8


def predict(*values):
    features = [float(v) for v in values]
    w = WorkspaceClient()
    host = os.environ["DATABRICKS_HOST"].rstrip("/")
    if not host.startswith(("https://", "http://")):
        host = "https://" + host
    url = (
        f"{host}/serving-endpoints/"
        "fwa-test-endpoint/invocations"
    )
    response = requests.post(
        url,
        headers=w.config.authenticate(),
        json={"inputs": [features]},
        timeout=120
    )
    response.raise_for_status()
    return {
        "submitted_features": features,
        "model_response": response.json()
    }


def test_lakebase():
    w = WorkspaceClient()
    credential = w.postgres.generate_database_credential(
        endpoint=os.environ["ENDPOINT_NAME"]
    )
    with psycopg.connect(
        host=os.environ["PGHOST"],
        port=os.environ.get("PGPORT", "5432"),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=credential.token,
        sslmode="require"
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE SCHEMA IF NOT EXISTS fwa_app_probe")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fwa_app_probe.test_events (
                    id INT PRIMARY KEY,
                    status TEXT
                )
            """)
            cur.execute("""
                INSERT INTO fwa_app_probe.test_events
                VALUES (1, 'connected')
                ON CONFLICT (id)
                DO UPDATE SET status = EXCLUDED.status
            """)
            cur.execute("""
                SELECT id, status FROM fwa_app_probe.test_events
                WHERE id = 1
            """)
            row = cur.fetchone()
    return {
        "database": "Lakebase PostgreSQL",
        "project": "fwa-project-test",
        "test_record_id": row[0],
        "status": row[1],
        "read_write_test": "PASSED"
    }


with gr.Blocks() as app:
    gr.Markdown("# FWA Databricks Integration Test")
    gr.Markdown(
        "Test ML predictions and Lakebase PostgreSQL connectivity."
    )
    gr.Markdown("## 1. Model Prediction")
    gr.Markdown("Enter eight synthetic feature values and run inference.")
    inputs = []
    for start in (0, 4):
        with gr.Row():
            for i in range(start, start + 4):
                inputs.append(
                    gr.Number(
                        label=f"Feature {i+1}",
                        value=POSITIVE_EXAMPLE[i]
                    )
                )
    with gr.Row():
        gr.Button("Load Class 1 Example").click(
            fn=lambda: POSITIVE_EXAMPLE, outputs=inputs
        )
        gr.Button("Load Zero Example").click(
            fn=lambda: ZERO_EXAMPLE, outputs=inputs
        )
    prediction_output = gr.JSON(label="Model Prediction Result")
    gr.Button("Predict", variant="primary").click(
        fn=predict, inputs=inputs, outputs=prediction_output
    )
    gr.Markdown("## 2. Lakebase PostgreSQL")
    gr.Markdown(
        "Test database authentication, table creation, "
        "record insertion, and retrieval."
    )
    database_output = gr.JSON(label="Lakebase Connection Result")
    gr.Button("Test Lakebase", variant="primary").click(
        fn=test_lakebase, outputs=database_output
    )

app.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
)
```

Observed example prediction and database response:

```json
{
  "submitted_features": [-1.27416, 1.31799, -2.42388,
      0.90695, -1.5839, -0.33181, -0.45231, 0.76041],
  "model_response": {"predictions": [1]}
}
{
  "database": "Lakebase PostgreSQL",
  "project": "fwa-project-test",
  "test_record_id": 1,
  "status": "connected",
  "read_write_test": "PASSED"
}
```

> The eight example features are anonymous synthetic variables. The class 1 outcome was confirmed empirically, but these values do not represent real FWA business fields. The app uses the registered model’s class predictions, not risk probabilities.

## Test 17 — Automated retraining, registration and alias

STATUS: PASS  |  SCOPE: Lakeflow Job + MLflow

OBSERVED: Scheduled-capable Job retrained, registered a new model version and moved the FWA_TestChampion alias; output version numbers matched.

```python
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_registry_uri("databricks-uc")
model_name = "workspace.default.fwa_lr_access_test"
X, y = make_classification(
    n_samples=300, n_features=8, random_state=43
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

with mlflow.start_run(run_name="fwa_auto_retrain_test"):
    mlflow.log_metric(
        "test_accuracy",
        accuracy_score(y_test, model.predict(X_test))
    )
    info = mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=X_train[:2]
    )
version = mlflow.register_model(info.model_uri, model_name)
client = MlflowClient()
client.set_registered_model_alias(
    name=model_name,
    alias="FWA_TestChampion",
    version=version.version
)
print("NEW MODEL VERSION:", version.version)
print("CHAMPION VERSION:",
      client.get_model_version_by_alias(
          model_name, "FWA_TestChampion"
      ).version)
```

> Alias reassignment does not automatically update the currently deployed serving endpoint. The test did not implement a quality gate or fully automatic serving rollout.

## Test 18 — CLI and Databricks Bundles

STATUS: PASS — GitHub Actions deferred  |  SCOPE: Windows PowerShell

OBSERVED: CLI authentication, bundle validate and bundle deploy to dev all succeeded.

```powershell
databricks version
# Only if CLI was not yet installed:
winget install Databricks.DatabricksCLI

databricks auth login --host https://dbc-1da8d8d7-1178.cloud.databricks.com
databricks current-user me

cd C:\Github
databricks bundle init default-python
cd C:\Github\fwa-bundle-test

databricks bundle validate
databricks bundle deploy --target dev
```

The test established local, authenticated deployment of Databricks bundle resources. GitHub Actions triggers, repository secrets/federation and automated CI execution were explicitly postponed.

# 5. Experimentation checks — Test 19

## Test 19A — Sticky A/B assignment and Delta recording

STATUS: PASS  |  SCOPE: Python + Delta

OBSERVED: Two independent assignments matched; both A and B appeared across 100 reviewers.

```python
import hashlib

def assign_variant(reviewer_id):
    key = f"fwa-experiment-v1:{reviewer_id}"
    hash_value = hashlib.sha256(key.encode()).hexdigest()
    return "A" if int(hash_value, 16) % 2 == 0 else "B"

reviewers = [f"reviewer_{i}" for i in range(100)]
first = [assign_variant(r) for r in reviewers]
second = [assign_variant(r) for r in reviewers]
assert first == second
print("Sticky assignment: PASSED")
rows = list(zip(reviewers, first))
df = spark.createDataFrame(rows, ["reviewer_id", "variant"])
df.write.mode("overwrite").format("delta").saveAsTable(
    "workspace.default.fwa_ab_access_test"
)
display(spark.sql("""
SELECT variant, COUNT(*) AS reviewers
FROM workspace.default.fwa_ab_access_test
GROUP BY variant ORDER BY variant
"""))
```

## Test 19B — Fisher exact test for simulated outcomes

STATUS: PASS  |  SCOPE: Python + SciPy

OBSERVED: A=19.57%, B=40.74%, diff=21.18 pp, p=0.0300.

```python
import numpy as np
from scipy.stats import fisher_exact

df = spark.table(
    "workspace.default.fwa_ab_access_test"
).toPandas()
rng = np.random.default_rng(544)
df["confirmed_positive"] = [
    int(rng.random() < (0.30 if v == "A" else 0.45))
    for v in df["variant"]
]
a = df[df["variant"] == "A"]["confirmed_positive"]
b = df[df["variant"] == "B"]["confirmed_positive"]
table = [
    [int(a.sum()), int(len(a) - a.sum())],
    [int(b.sum()), int(len(b) - b.sum())]
]
odds_ratio, p_value = fisher_exact(table)
print(f"Model A success rate: {a.mean():.2%}")
print(f"Model B success rate: {b.mean():.2%}")
print(f"Difference (B - A): {(b.mean()-a.mean()):.2%}")
print(f"Fisher exact p-value: {p_value:.4f}")
print("Statistical analysis: PASSED")
```

> This is a deterministic simulation, not a real comparison of models or investigators. In deployment, add stable reviewer assignment in the app, route requests to both real serving configurations, log exposure and investigator action, and avoid treating simulated p-values as empirical effect evidence.

# 6. Provenance validation — Test 20

## Test 20A — Serving endpoint to registered model and run

STATUS: PASS  |  SCOPE: Model metadata API

OBSERVED: Served version=1, MLflow run=50b136871fb14e96a38246cae4cb77e6.

```python
from databricks.sdk import WorkspaceClient
from mlflow import MlflowClient

w = WorkspaceClient()
client = MlflowClient(registry_uri="databricks-uc")
endpoint = w.serving_endpoints.get(name="fwa-test-endpoint")
entities = (
    endpoint.config.served_entities
    or endpoint.config.served_models
    or []
)
for entity in entities:
    model_name = entity.entity_name or entity.model_name
    model_version = entity.entity_version or entity.model_version
    version = client.get_model_version(model_name, str(model_version))
    print("SERVING ENDPOINT:", endpoint.name)
    print("REGISTERED MODEL:", model_name)
    print("MODEL VERSION:", model_version)
    print("MLFLOW RUN ID:", version.run_id or "Not recorded")
```

Observed: fwa-test-endpoint → workspace.default.fwa_lr_access_test → version 1 → MLflow run 50b136871fb14e96a38246cae4cb77e6. This original model had no recorded Delta training snapshot.

## Test 20B — Registered model → immutable training snapshot

STATUS: PASS  |  SCOPE: MLflow dataset lineage

OBSERVED: Model fwa_provenance_test v1, Delta version 0, training rows 300; lineage PASSED.

```python
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

mlflow.set_registry_uri("databricks-uc")
TABLE = "workspace.default.fwa_training_provenance_test"
MODEL = "workspace.default.fwa_provenance_test"
X, y = make_classification(
    n_samples=300, n_features=8, random_state=544
)
columns = [f"feature_{i}" for i in range(1, 9)]
data = pd.DataFrame(X, columns=columns)
data["label"] = y
spark.createDataFrame(data).write \
    .format("delta").mode("overwrite") \
    .option("overwriteSchema", "true").saveAsTable(TABLE)
version = spark.sql(
    f"DESCRIBE HISTORY {TABLE}"
).select("version").first()[0]
dataset = mlflow.data.load_delta(
    table_name=TABLE, version=str(version), targets="label"
)
training = dataset.df.toPandas()
with mlflow.start_run(run_name="fwa_provenance_test") as run:
    mlflow.log_input(dataset, context="training")
    mlflow.set_tag("pipeline_version", "fwa_test_v1")
    mlflow.set_tag("delta_version", str(version))
    model = LogisticRegression(max_iter=1000)
    model.fit(training[columns], training["label"])
    result = mlflow.sklearn.log_model(
        sk_model=model, name="model",
        input_example=training[columns].iloc[:2],
        registered_model_name=MODEL
    )
    run_id = run.info.run_id
recorded = mlflow.get_run(run_id)
print("TRAINING TABLE:", TABLE)
print("DELTA VERSION:", version)
print("TRAINING ROWS:", len(training))
print("MODEL:", MODEL)
print("MLFLOW RUN ID:", run_id)
print("DATASET LINEAGE:",
      "PASSED" if recorded.inputs.dataset_inputs else "MISSING")
```

Observed model run: 5b658f3380da4c2b99480e326b74571c; dataset version 0; 300 rows.

## Test 20C — Synthetic prediction audit linked to lineage

STATUS: PASS  |  SCOPE: Registered model + Delta audit

OBSERVED: Prediction audit PASSED; ID 05312977-dae6-405a-9cd5-3db0b1536968.

```python
import uuid
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
MODEL = "workspace.default.fwa_provenance_test"
AUDIT = "workspace.default.fwa_prediction_audit_test"
client = MlflowClient(registry_uri="databricks-uc")
version = client.get_model_version(MODEL, "1")
run = mlflow.get_run(version.run_id)
table = "workspace.default.fwa_training_provenance_test"
delta_version = int(run.data.tags["delta_version"])
pipeline_version = run.data.tags["pipeline_version"]
data = spark.read.option(
    "versionAsOf", delta_version
).table(table)
features = data.limit(1).toPandas().drop(columns=["label"])
model = mlflow.sklearn.load_model(f"models:/{MODEL}/1")
prediction = int(model.predict(features)[0])
prediction_id = str(uuid.uuid4())
record = [{
    "prediction_id": prediction_id,
    "provider_id": "synthetic_provider_1001",
    "prediction": prediction,
    "model_name": MODEL,
    "model_version": 1,
    "mlflow_run_id": version.run_id,
    "training_table": table,
    "training_delta_version": delta_version,
    "pipeline_version": pipeline_version
}]
spark.createDataFrame(record).write \
    .format("delta").mode("append").saveAsTable(AUDIT)
display(spark.table(AUDIT).filter(
    f"prediction_id = '{prediction_id}'"
))
print("PREDICTION AUDIT: PASSED")
print("PREDICTION ID:", prediction_id)
print("MODEL VERSION:", version.version)
print("TRAINING DELTA VERSION:", delta_version)
print("PIPELINE VERSION:", pipeline_version)
```

Observed: prediction ID 05312977-dae6-405a-9cd5-3db0b1536968; model version 1; training Delta version 0; pipeline label fwa_test_v1.

> This was a notebook-produced synthetic prediction, NOT an audited production serving event. The pipeline identifier is a manual test tag, not a verified Git commit. The test record does not preserve full request feature values, timestamp or experiment assignment. Add these for true replay and end-to-end M3 provenance.

# 7. Remaining validation and follow-up

## Test 21 — Docker runtime and standalone scorer

STATUS: NOT RUN  |  SCOPE: Local Windows/Docker

OBSERVED: Docker commands were proposed but no result was reported.

```powershell
docker version
docker run --rm hello-world
```

After confirming local Docker: containerize a FastAPI scorer, build and run locally, call its inference endpoint, and deploy the image to an accessible standalone hosting environment. Databricks Apps alone do not meet the explicit containerization demonstration requirement.

# 8. Prioritized outstanding checks

- P1 — Verify Test 15 native snapshot profiling actually completes; create a baseline-vs-current drift comparison and preserve metrics table names.
- P1 — Finish Test 21: verify Docker and deploy an independently callable containerized inference service.
- P1 — Finish live prediction audit: log serving model/version, input feature hash or payload, runtime timestamp, Git commit, Delta training version and response ID.
- P2 — Recheck Test 13 pipeline expectation counts explicitly (two valid / one dropped) and capture its event-log metrics.
- P2 — Verify Test 14 SQL Alert enters a triggered state and sends an approved notification (only if notification capability is needed).
- P2 — Add an actual two-model A/B routing demonstration with user-stable assignment, exposure telemetry and non-simulated or clearly labeled replay outcomes.
- P2 — GitHub Actions CI/CD: intentionally deferred; validate workflow permissions and automated bundle deployment when resumed.
- P2 — Verify collaborator access: Free Edition sharing policy, whether Irfan and Vipul can open Apps/SQL/Jobs, and quota limits.

# 9. Lessons from the test sequence

| Problem encountered | Working resolution / implication |
|---|---|
| Serving quota exceeded | Use a supported 0–4 concurrency endpoint; quota may still affect other serverless resources. |
| CDF rejected default catalog | External AWS S3 storage, storage credential/external location and a new externally backed catalog enabled CDF. |
| Long-running Lakeflow Job | Read task logs; “JOB SUCCESS” and final status Succeeded confirmed execution. |
| ETL pipeline RESOURCE_EXHAUSTED | Free Edition serverless usage/capacity was temporarily limiting; later run succeeded. |
| App showed old Hello World code | Deploy from persistent Workspace folder fwa_app_source rather than editing generated deployment snapshot. |
| Prediction request failed MissingSchema | Normalize DATABRICKS_HOST by adding https:// when absent. |
| App KeyError ENDPOINT_NAME | app.yaml loaded; actual Lakebase resource key was database, not postgres. Set valueFrom: database. |
| A/B p-value significant | Outcomes were artificial; result proves library/test availability, not superiority of any model. |
| Provenance was incomplete for served model | Separate test registered versioned Delta input and logged a synthetic prediction record; live serving audit remains to build. |

# 10. Evidence and reproducibility notes

Preserve alongside this runbook: (1) job run URLs/screenshots, (2) MLflow experiment/model version links, (3) pipeline event-log and data quality panel screenshots, (4) app deployed source path, (5) Databricks Bundle YAML in GitHub, (6) one Delta audit query output showing a chosen prediction ID, (7) app resources showing Lakebase key database. Do not paste credentials, AWS role secret material, database passwords, or company data into a course repository.

Before sharing publicly, verify whether the workspace hostname, project names, endpoint names and storage bucket naming are appropriate to disclose. The code is a capability probe, not hardened production inference: add authentication/authorization review, least privilege, validation, retries, timeouts, monitoring, rollback policy, cost controls and scheduled clean-up.

