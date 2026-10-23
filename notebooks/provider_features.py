"""
T09 - Provider-level Gold features for the FWA provider-ranking project.

Pure PySpark transformations, no I/O, so they run the same on Databricks and
in local tests:

    Bronze (raw CSV strings) -> Silver (typed, cleaned) -> Gold (1 row per provider)

Design rules (see docs/T09/feature_definitions.md):
  * Grain of Gold = one row per Provider. Validated, not assumed.
  * The label (PotentialFraud) is NEVER a Gold column. It lives in its own
    table (gold_provider_labels) and is joined only at training time.
  * Inpatient and outpatient claims are UNIONed (with a claim_type column)
    before aggregation. They are never joined to each other.
  * Beneficiaries join to claims many-to-one on bene_id; the join is checked
    so it cannot inflate claim counts.
  * Race and Gender stay in Silver for fairness audits only; they are not
    features.
  * Raw CSV encodes missing values as the literal string "NA"; Silver turns
    those into real nulls.
"""
from __future__ import annotations

from functools import reduce

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

FEATURE_SET_VERSION = "provider_features_v1"

N_DIAG_COLS = 10
N_PROC_COLS = 6
CHRONIC_COLS = [
    "ChronicCond_Alzheimer", "ChronicCond_Heartfailure", "ChronicCond_KidneyDisease",
    "ChronicCond_Cancer", "ChronicCond_ObstrPulmonary", "ChronicCond_Depression",
    "ChronicCond_Diabetes", "ChronicCond_IschemicHeart", "ChronicCond_Osteoporasis",
    "ChronicCond_rheumatoidarthritis", "ChronicCond_stroke",
]

# Fixed (not quantile-based) size bands so a provider's peer group does not
# change just because other providers were added to the data.
SIZE_BANDS = [(10, "XS"), (50, "S"), (200, "M")]  # < 10 -> XS, < 50 -> S, < 200 -> M, else L

# Rate features that get a peer-relative version (robust z-score in peer group).
PEER_RELATIVE_FEATURES = [
    "reimbursed_per_claim",
    "reimbursed_per_beneficiary",
    "claims_per_beneficiary",
    "avg_claim_days",
    "diag_codes_per_claim",
    "attending_physicians_per_claim",
    "avg_chronic_conditions",
]

# Columns that identify the provider or describe the row, not model inputs.
NON_FEATURE_COLUMNS = [
    "provider_id", "peer_group", "size_band", "feature_set_version",
    "feature_window_start", "feature_window_end",
]


# --------------------------------------------------------------------------- #
# Silver
# --------------------------------------------------------------------------- #
def _clean(col_name: str):
    """Trim a raw string column and turn '', 'NA', 'NaN', 'null' into a real null."""
    c = F.trim(F.col(col_name).cast("string"))
    return F.when(c.isNull() | c.isin("", "NA", "NaN", "nan", "null", "NULL"), None).otherwise(c)


def _date(col_name: str):
    return F.to_date(_clean(col_name), "yyyy-MM-dd")


def _num(col_name: str):
    return _clean(col_name).cast("double")


def build_silver_claims(bronze_inpatient: DataFrame, bronze_outpatient: DataFrame) -> DataFrame:
    """One row per claim, both claim types, typed and cleaned."""

    def _one(df: DataFrame, claim_type: str) -> DataFrame:
        has = set(df.columns)
        diag = [_clean(f"ClmDiagnosisCode_{i}") for i in range(1, N_DIAG_COLS + 1)]
        proc = [
            _clean(f"ClmProcedureCode_{i}") if f"ClmProcedureCode_{i}" in has else F.lit(None).cast("string")
            for i in range(1, N_PROC_COLS + 1)
        ]
        opt = lambda c, fn: fn(c) if c in has else F.lit(None)  # noqa: E731 (columns only inpatient has)
        return df.select(
            _clean("ClaimID").alias("claim_id"),
            _clean("BeneID").alias("bene_id"),
            _clean("Provider").alias("provider_id"),
            F.lit(claim_type).alias("claim_type"),
            _date("ClaimStartDt").alias("claim_start_dt"),
            _date("ClaimEndDt").alias("claim_end_dt"),
            opt("AdmissionDt", _date).cast("date").alias("admission_dt"),
            opt("DischargeDt", _date).cast("date").alias("discharge_dt"),
            _num("InscClaimAmtReimbursed").alias("reimbursed_amt"),
            _num("DeductibleAmtPaid").alias("deductible_amt"),
            _clean("AttendingPhysician").alias("attending_physician"),
            _clean("OperatingPhysician").alias("operating_physician"),
            _clean("OtherPhysician").alias("other_physician"),
            _clean("ClmAdmitDiagnosisCode").alias("admit_diag_code"),
            opt("DiagnosisGroupCode", _clean).cast("string").alias("drg_code"),
            F.filter(F.array(*diag), lambda x: x.isNotNull()).alias("diag_codes"),
            F.filter(F.array(*proc), lambda x: x.isNotNull()).alias("proc_codes"),
        )

    claims = _one(bronze_inpatient, "inpatient").unionByName(_one(bronze_outpatient, "outpatient"))
    return (
        claims
        .withColumn("n_diag_codes", F.size("diag_codes"))
        .withColumn("n_proc_codes", F.size("proc_codes"))
        .withColumn("claim_days", F.datediff("claim_end_dt", "claim_start_dt"))
        .withColumn("los_days", F.datediff("discharge_dt", "admission_dt"))
    )


def build_silver_beneficiaries(bronze_beneficiary: DataFrame) -> DataFrame:
    """One row per beneficiary. Codes recoded; age computed later against the as-of date."""
    chronic = [
        F.when(_clean(c) == "1", 1).when(_clean(c) == "2", 0).otherwise(None).alias(
            "cc_" + c.replace("ChronicCond_", "").lower())
        for c in CHRONIC_COLS
    ]
    df = bronze_beneficiary.select(
        _clean("BeneID").alias("bene_id"),
        _date("DOB").alias("dob"),
        _date("DOD").alias("dod"),
        _clean("Gender").alias("gender_code"),      # fairness audit only, not a feature
        _clean("Race").alias("race_code"),          # fairness audit only, not a feature
        _clean("State").alias("state_code"),
        _clean("County").alias("county_code"),
        F.when(_clean("RenalDiseaseIndicator") == "Y", 1).otherwise(0).alias("renal_disease"),
        _num("NoOfMonths_PartACov").cast("int").alias("part_a_months"),
        _num("NoOfMonths_PartBCov").cast("int").alias("part_b_months"),
        *chronic,
        _num("IPAnnualReimbursementAmt").alias("ip_annual_reimbursed_amt"),
        _num("OPAnnualReimbursementAmt").alias("op_annual_reimbursed_amt"),
    )
    cc_cols = [c for c in df.columns if c.startswith("cc_")]
    return df.withColumn("n_chronic_conditions", reduce(lambda a, b: a + b, [F.coalesce(F.col(c), F.lit(0)) for c in cc_cols]))


def build_provider_labels(bronze_labels: DataFrame) -> DataFrame:
    """Labels table, kept separate from features. 1 = flagged PotentialFraud."""
    return bronze_labels.select(
        _clean("Provider").alias("provider_id"),
        F.when(F.upper(_clean("PotentialFraud")) == "YES", 1)
         .when(F.upper(_clean("PotentialFraud")) == "NO", 0)
         .otherwise(None).cast("int").alias("is_potential_fraud"),
    ).withColumn("label_source", F.lit("kaggle_healthcare_provider_fraud_train"))


# --------------------------------------------------------------------------- #
# Gold
# --------------------------------------------------------------------------- #
def _size_band(n_claims_col):
    expr = F.lit("L")
    for limit, name in reversed(SIZE_BANDS):
        expr = F.when(n_claims_col < limit, name).otherwise(expr)
    return expr


def build_gold_provider_features(silver_claims: DataFrame, silver_beneficiaries: DataFrame) -> DataFrame:
    """One row per provider. No label columns."""
    # As-of date = last service date in the data. Features use only claims up to it
    # (the whole window here); the label has no date, so this is documented as an assumption.
    window = silver_claims.agg(F.min("claim_start_dt").alias("ws"), F.max("claim_end_dt").alias("we")).first()
    window_start, window_end = window["ws"], window["we"]

    bene = silver_beneficiaries.withColumn(
        "age_years", F.floor(F.months_between(F.lit(window_end), F.col("dob")) / 12).cast("int"))

    # Shared-patient signal: how many providers each beneficiary visits.
    bene_providers = silver_claims.groupBy("bene_id").agg(F.countDistinct("provider_id").alias("providers_per_bene"))
    # Physician network signal: how many providers each attending physician bills through.
    phys_providers = (silver_claims.where(F.col("attending_physician").isNotNull())
                      .groupBy("attending_physician")
                      .agg(F.countDistinct("provider_id").alias("providers_per_physician")))

    c = (silver_claims
         .join(bene.select("bene_id", "age_years", "dod", "renal_disease", "n_chronic_conditions"), "bene_id", "left")
         .join(bene_providers, "bene_id", "left")
         .join(phys_providers, "attending_physician", "left"))

    is_inp = F.col("claim_type") == "inpatient"
    is_out = F.col("claim_type") == "outpatient"
    claim_month = F.date_format("claim_start_dt", "yyyy-MM")

    agg = c.groupBy("provider_id").agg(
        # volume (size) - kept, but size alone is a confounder; see docs
        F.count("*").alias("n_claims"),
        F.sum(is_inp.cast("int")).alias("n_inpatient_claims"),
        F.sum(is_out.cast("int")).alias("n_outpatient_claims"),
        F.countDistinct("bene_id").alias("n_beneficiaries"),
        F.sum("reimbursed_amt").alias("total_reimbursed"),
        F.sum(F.when(is_inp, F.col("reimbursed_amt"))).alias("inpatient_total_reimbursed"),
        F.sum(F.when(is_out, F.col("reimbursed_amt"))).alias("outpatient_total_reimbursed"),
        F.countDistinct(claim_month).alias("n_active_months"),
        # amounts
        F.avg("reimbursed_amt").alias("reimbursed_per_claim"),
        F.avg(F.when(is_inp, F.col("reimbursed_amt"))).alias("inpatient_reimbursed_per_claim"),
        F.avg(F.when(is_out, F.col("reimbursed_amt"))).alias("outpatient_reimbursed_per_claim"),
        F.max("reimbursed_amt").alias("max_claim_reimbursed"),
        F.avg((F.col("reimbursed_amt") == 0).cast("int")).alias("zero_reimbursed_share"),
        F.avg(F.when(is_out, F.coalesce(F.col("deductible_amt"), F.lit(0.0)))).alias("outpatient_deductible_per_claim"),
        # utilisation / coding
        F.avg("claim_days").alias("avg_claim_days"),
        F.avg(F.when(is_inp, F.col("los_days"))).alias("avg_inpatient_los"),
        F.avg("n_diag_codes").alias("diag_codes_per_claim"),
        F.size(F.array_distinct(F.flatten(F.collect_list("diag_codes")))).alias("n_distinct_diag_codes"),
        F.avg(F.when(is_inp, F.col("n_proc_codes"))).alias("proc_codes_per_inpatient_claim"),
        F.countDistinct("drg_code").alias("n_distinct_drg_codes"),
        # physicians
        F.countDistinct("attending_physician").alias("n_attending_physicians"),
        F.avg(F.col("operating_physician").isNotNull().cast("int")).alias("operating_physician_share"),
        F.avg(F.when(F.col("providers_per_physician") > 1, 1).otherwise(0)).alias("multi_provider_physician_claim_share"),
        # patient mix (claim-weighted)
        F.avg("age_years").alias("avg_beneficiary_age"),
        F.avg("n_chronic_conditions").alias("avg_chronic_conditions"),
        F.avg("renal_disease").alias("renal_disease_share"),
        F.avg(F.col("dod").isNotNull().cast("int")).alias("deceased_beneficiary_share"),
        F.avg("providers_per_bene").alias("avg_providers_per_beneficiary"),
    )
    # Note: "claims after the patient's death" is a classic FWA rule, but it is 0 for
    # every provider in this dataset, so it is reported by validate() instead of
    # being stored as a constant (useless) feature.

    gold = (agg
        .withColumn("inpatient_share", F.col("n_inpatient_claims") / F.col("n_claims"))
        .withColumn("claims_per_beneficiary", F.col("n_claims") / F.col("n_beneficiaries"))
        .withColumn("reimbursed_per_beneficiary", F.col("total_reimbursed") / F.col("n_beneficiaries"))
        .withColumn("claims_per_active_month", F.col("n_claims") / F.col("n_active_months"))
        .withColumn("distinct_diag_codes_per_claim", F.col("n_distinct_diag_codes") / F.col("n_claims"))
        .withColumn("attending_physicians_per_claim", F.col("n_attending_physicians") / F.col("n_claims"))
        .withColumn("has_inpatient", (F.col("n_inpatient_claims") > 0).cast("int"))
        .withColumn("size_band", _size_band(F.col("n_claims")))
        .withColumn("peer_group", F.concat_ws("_", F.col("size_band"),
                                              F.when(F.col("has_inpatient") == 1, "IP").otherwise("OP")))
    )

    # Null handling: inpatient-only averages are undefined for outpatient-only
    # providers. Fill with 0 and keep has_inpatient so the model can tell them apart.
    fill_zero = ["inpatient_total_reimbursed", "outpatient_total_reimbursed",
                 "inpatient_reimbursed_per_claim", "outpatient_reimbursed_per_claim",
                 "outpatient_deductible_per_claim", "avg_inpatient_los",
                 "proc_codes_per_inpatient_claim", "avg_beneficiary_age"]
    gold = gold.fillna(0.0, subset=fill_zero)

    gold = add_peer_relative_features(gold)

    return (gold
            .withColumn("feature_set_version", F.lit(FEATURE_SET_VERSION))
            .withColumn("feature_window_start", F.lit(window_start).cast("date"))
            .withColumn("feature_window_end", F.lit(window_end).cast("date")))


def add_peer_relative_features(gold: DataFrame, eps: float = 1e-6) -> DataFrame:
    """Robust z-score inside the peer group: (x - peer median) / peer IQR.

    Uses no labels. Peer statistics are computed over the scored population, the
    same way a production FWA system compares a provider with its current peers.
    """
    w = Window.partitionBy("peer_group")
    for f in PEER_RELATIVE_FEATURES:
        med = F.percentile_approx(f, 0.5, 10000).over(w)
        iqr = F.percentile_approx(f, 0.75, 10000).over(w) - F.percentile_approx(f, 0.25, 10000).over(w)
        gold = gold.withColumn(f"peer_rz_{f}", (F.col(f) - med) / F.greatest(iqr, F.lit(eps)))
    # Clip extreme values so a tiny IQR cannot create absurd scores.
    for f in PEER_RELATIVE_FEATURES:
        gold = gold.withColumn(f"peer_rz_{f}", F.least(F.greatest(F.col(f"peer_rz_{f}"), F.lit(-20.0)), F.lit(20.0)))
    return gold


def feature_columns(gold: DataFrame) -> list[str]:
    return [c for c in gold.columns if c not in NON_FEATURE_COLUMNS]


def _maybe_cache(df: DataFrame) -> DataFrame:
    """Cache when the cluster allows it. Databricks serverless compute rejects
    df.cache() (NOT_SUPPORTED_WITH_SERVERLESS); there we simply recompute."""
    try:
        return df.cache()
    except Exception:
        return df


def build_all(bronze_inpatient: DataFrame, bronze_outpatient: DataFrame,
              bronze_beneficiary: DataFrame, bronze_labels: DataFrame | None = None) -> dict:
    """Bronze in, Silver + Gold + labels out, plus validation results."""
    silver_claims = _maybe_cache(build_silver_claims(bronze_inpatient, bronze_outpatient))
    silver_bene = _maybe_cache(build_silver_beneficiaries(bronze_beneficiary))
    gold = _maybe_cache(build_gold_provider_features(silver_claims, silver_bene))
    labels = _maybe_cache(build_provider_labels(bronze_labels)) if bronze_labels is not None else None
    checks = validate(silver_claims, silver_bene, gold, labels)
    return {"silver_claims": silver_claims, "silver_beneficiaries": silver_bene,
            "gold_provider_features": gold, "gold_provider_labels": labels, "checks": checks}


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate(silver_claims: DataFrame, silver_beneficiaries: DataFrame, gold: DataFrame,
             labels: DataFrame | None = None) -> dict:
    """Return a dict of named checks -> {passed, detail}. Caller decides whether to fail the job."""
    checks: dict[str, dict] = {}

    def check(name, passed, detail):
        checks[name] = {"passed": bool(passed), "detail": detail}

    n_claims = silver_claims.count()
    n_claim_ids = silver_claims.select("claim_id").distinct().count()
    check("claim_id_unique", n_claims == n_claim_ids, f"{n_claims} claims, {n_claim_ids} distinct ids")

    n_bene = silver_beneficiaries.count()
    n_bene_ids = silver_beneficiaries.select("bene_id").distinct().count()
    check("bene_id_unique", n_bene == n_bene_ids, f"{n_bene} rows, {n_bene_ids} distinct")

    joined = silver_claims.join(silver_beneficiaries.select("bene_id"), "bene_id", "left").count()
    check("bene_join_no_inflation", joined == n_claims, f"{n_claims} claims before join, {joined} after")

    orphan = silver_claims.join(silver_beneficiaries.select("bene_id"), "bene_id", "left_anti").count()
    check("claims_have_beneficiary", orphan == 0, f"{orphan} claims without a beneficiary")

    # Informational rule check (always passes): would become a feature if non-zero in new data.
    after_death = (silver_claims.join(silver_beneficiaries.select("bene_id", "dod"), "bene_id")
                   .where(F.col("claim_start_dt") > F.col("dod")).count())
    check("info_claims_after_death", True, f"{after_death} claims start after the patient's date of death")

    bad_dates = silver_claims.where(F.col("claim_start_dt").isNull() | (F.col("claim_end_dt") < F.col("claim_start_dt"))).count()
    check("claim_dates_valid", bad_dates == 0, f"{bad_dates} claims with missing start date or end before start")

    n_gold = gold.count()
    n_gold_ids = gold.select("provider_id").distinct().count()
    n_claim_providers = silver_claims.select("provider_id").distinct().count()
    check("gold_one_row_per_provider", n_gold == n_gold_ids, f"{n_gold} rows, {n_gold_ids} distinct providers")
    check("gold_covers_all_claim_providers", n_gold == n_claim_providers,
          f"{n_gold} gold rows vs {n_claim_providers} providers in claims")

    gold_claims = gold.agg(F.sum("n_claims")).first()[0]
    check("gold_claim_total_matches_silver", gold_claims == n_claims, f"sum(n_claims)={gold_claims} vs {n_claims}")

    leaky = [c for c in gold.columns if any(k in c.lower() for k in ("fraud", "label", "target"))]
    check("no_label_columns_in_gold", not leaky, f"suspicious columns: {leaky}")

    feats = feature_columns(gold)
    null_counts = gold.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in feats]).first().asDict()
    nulls = {k: v for k, v in null_counts.items() if v}
    check("no_null_features", not nulls, f"null counts: {nulls}")

    share_cols = [c for c in feats if c.endswith("_share") and not c.startswith("peer_rz_")]
    bad = gold.where(reduce(lambda a, b: a | b,
                            [(F.col(c) < 0) | (F.col(c) > 1) for c in share_cols])).count()
    check("shares_between_0_and_1", bad == 0, f"{bad} rows with a share outside [0,1] ({len(share_cols)} share columns)")

    if labels is not None:
        n_lab = labels.count()
        n_lab_ids = labels.select("provider_id").distinct().count()
        check("labels_one_row_per_provider", n_lab == n_lab_ids, f"{n_lab} rows, {n_lab_ids} distinct")
        unlabeled = gold.join(labels, "provider_id", "left_anti").count()
        no_features = labels.join(gold, "provider_id", "left_anti").count()
        check("every_gold_provider_has_label", unlabeled == 0, f"{unlabeled} gold providers without a label")
        check("every_label_has_features", no_features == 0, f"{no_features} labeled providers without features")
        bad_lab = labels.where(F.col("is_potential_fraud").isNull()).count()
        check("labels_parsed", bad_lab == 0, f"{bad_lab} labels not Yes/No")

    return checks
