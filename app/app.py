"""Synthetic FWA Databricks connectivity smoke test (not production scoring)."""
import os
import gradio as gr
import requests
import psycopg
from databricks.sdk import WorkspaceClient

POSITIVE_EXAMPLE = [-1.27416, 1.31799, -2.42388, 0.90695,
                    -1.58390, -0.33181, -0.45231, 0.76041]
ZERO_EXAMPLE = [0.0] * 8


def predict(*values):
    features = [float(v) for v in values]
    client = WorkspaceClient()
    host = os.environ["DATABRICKS_HOST"].rstrip("/")
    if not host.startswith(("https://", "http://")):
        host = "https://" + host
    response = requests.post(
        f"{host}/serving-endpoints/fwa-test-endpoint/invocations",
        headers=client.config.authenticate(),
        json={"inputs": [features]},
        timeout=120,
    )
    response.raise_for_status()
    return {"submitted_features": features, "model_response": response.json()}


def test_lakebase():
    client = WorkspaceClient()
    credential = client.postgres.generate_database_credential(
        endpoint=os.environ["ENDPOINT_NAME"]
    )
    with psycopg.connect(
        host=os.environ["PGHOST"],
        port=os.environ.get("PGPORT", "5432"),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=credential.token,
        sslmode="require",
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("CREATE SCHEMA IF NOT EXISTS fwa_app_probe")
            cursor.execute("""CREATE TABLE IF NOT EXISTS fwa_app_probe.test_events
                              (id INT PRIMARY KEY, status TEXT)""")
            cursor.execute("""INSERT INTO fwa_app_probe.test_events VALUES (1, 'connected')
                              ON CONFLICT (id) DO UPDATE SET status = EXCLUDED.status""")
            cursor.execute("SELECT id, status FROM fwa_app_probe.test_events WHERE id = 1")
            row = cursor.fetchone()
    return {"database": "Lakebase PostgreSQL", "project": "fwa-project-test",
            "test_record_id": row[0], "status": row[1], "read_write_test": "PASSED"}


with gr.Blocks() as app:
    gr.Markdown("# FWA Databricks Integration Test")
    gr.Markdown("This prototype uses synthetic data only.")
    gr.Markdown("## 1. Model Prediction")
    inputs = []
    for start in (0, 4):
        with gr.Row():
            for i in range(start, start + 4):
                inputs.append(gr.Number(label=f"Feature {i + 1}", value=POSITIVE_EXAMPLE[i]))
    with gr.Row():
        gr.Button("Load Class 1 Example").click(lambda: POSITIVE_EXAMPLE, outputs=inputs)
        gr.Button("Load Zero Example").click(lambda: ZERO_EXAMPLE, outputs=inputs)
    prediction_output = gr.JSON(label="Model Prediction Result")
    gr.Button("Predict", variant="primary").click(
        fn=predict, inputs=inputs, outputs=prediction_output
    )
    gr.Markdown("## 2. Lakebase PostgreSQL")
    database_output = gr.JSON(label="Lakebase Connection Result")
    gr.Button("Test Lakebase").click(fn=test_lakebase, outputs=database_output)

app.launch(server_name="0.0.0.0",
           server_port=int(os.environ.get("DATABRICKS_APP_PORT", "8000")))
