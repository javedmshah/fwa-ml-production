# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # T09 test notebook: Gold provider features on real Bronze data
# MAGIC
# MAGIC Run the cells top to bottom (or **Run all**). Nothing is written unless you set `write_tables = yes`.
# MAGIC Every check prints PASS/FAIL; the notebook stops at the first failure.
# MAGIC
# MAGIC Keep this notebook in the **same folder** as `provider_features.py` so the import works.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "default")
dbutils.widgets.text("bronze_inpatient", "bronze_inpatient")
dbutils.widgets.text("bronze_outpatient", "bronze_outpatient")
dbutils.widgets.text("bronze_beneficiary", "bronze_beneficiary")
dbutils.widgets.text("bronze_labels", "bronze_provider_labels")
dbutils.widgets.dropdown("write_tables", "no", ["no", "yes"])

w = {k: dbutils.widgets.get(k) for k in
     ["catalog", "schema", "bronze_inpatient", "bronze_outpatient", "bronze_beneficiary", "bronze_labels", "write_tables"]}
fq = lambda t: f"{w['catalog']}.{w['schema']}.{t}"
print(w)

# COMMAND ----------

# MAGIC %md ## 1. Import the T09 code

# COMMAND ----------

import importlib, os, sys
sys.path.insert(0, os.getcwd())          # folder of this notebook
import provider_features as pf
importlib.reload(pf)                      # pick up edits without restarting
print("loaded", pf.FEATURE_SET_VERSION, "from", pf.__file__)

# COMMAND ----------

# MAGIC %md ## 2. Bronze tables exist and have the expected row counts

# COMMAND ----------

expected = {"bronze_inpatient": 40474, "bronze_outpatient": 517737,
            "bronze_beneficiary": 138556, "bronze_labels": 5410}
bronze = {}
for key, n in expected.items():
    df = spark.table(fq(w[key]))
    got = df.count()
    print(f"[{'PASS' if got == n else 'FAIL'}] {fq(w[key])}: {got:,} rows (expected {n:,})")
    bronze[key] = df
assert all(bronze[k].count() == n for k, n in expected.items()), "Bronze row counts differ from the Kaggle train files"

# COMMAND ----------

# MAGIC %md ## 3. Build Silver + Gold and run the 16 validation checks

# COMMAND ----------

out = pf.build_all(bronze["bronze_inpatient"], bronze["bronze_outpatient"],
                   bronze["bronze_beneficiary"], bronze["bronze_labels"])
for k, v in out["checks"].items():
    print(f"[{'PASS' if v['passed'] else 'FAIL'}] {k}: {v['detail']}")
failed = [k for k, v in out["checks"].items() if not v["passed"]]
assert not failed, f"failed checks: {failed}"
print(f"\nALL {len(out['checks'])} CHECKS PASSED")

# COMMAND ----------

# MAGIC %md ## 4. Spot checks against the known local results

# COMMAND ----------

from pyspark.sql import functions as F
gold, labels = out["gold_provider_features"], out["gold_provider_labels"]

# 4a. "NA" text in Bronze became real nulls in Silver
na_text = bronze["bronze_inpatient"].where(F.col("OperatingPhysician") == "NA").count()
nulls = out["silver_claims"].where("claim_type = 'inpatient' AND operating_physician IS NULL").count()
print(f"[{'PASS' if na_text == nulls else 'FAIL'}] 'NA' strings in Bronze: {na_text:,} -> nulls in Silver: {nulls:,}")

# 4b. Label balance (expect 506 flagged, 4,904 not)
display(labels.groupBy("is_potential_fraud").count())

# 4c. The two providers from the training session (expect PRV51857 ~ $6,728/claim, peer_rz ~ +2.1)
display(gold.where(F.col("provider_id").isin("PRV51857", "PRV52405"))
            .select("provider_id", "n_claims", "reimbursed_per_claim", "inpatient_share",
                    "peer_group", "peer_rz_reimbursed_per_claim"))

# COMMAND ----------

# 4d. Flag rate by peer group (expect ~0.2% for XS_OP up to ~52% for L_IP)
display(gold.join(labels, "provider_id")
            .groupBy("peer_group")
            .agg(F.count("*").alias("providers"), F.round(F.avg("is_potential_fraud"), 3).alias("flag_rate"))
            .orderBy("flag_rate"))

# COMMAND ----------

# MAGIC %md ## 5. (Optional) Write the tables. The Lakeflow job does this too; set `write_tables = yes` to do it here.

# COMMAND ----------

if w["write_tables"] == "yes":
    for name in ["silver_claims", "silver_beneficiaries", "gold_provider_features", "gold_provider_labels"]:
        out[name].write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(fq(name))
        print("wrote", fq(name), spark.table(fq(name)).count())
else:
    print("write_tables = no: nothing written")