"""Provider-level (Gold) features built from the Kaggle healthcare-provider-fraud files.

One output row per Provider. The label (PotentialFraud) is never used here, so this
module can run on any set of claims, including providers we have never seen.

The same logic is meant to be ported to PySpark for the Databricks Gold table, so keep
every feature simple, documented and reproducible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# File name prefixes in the Kaggle archive (the numeric suffix differs per download).
TRAIN_FILES = {
    "labels": "Train-",
    "inpatient": "Train_Inpatientdata-",
    "outpatient": "Train_Outpatientdata-",
    "beneficiary": "Train_Beneficiarydata-",
}

CHRONIC_COLS = [
    "ChronicCond_Alzheimer", "ChronicCond_Heartfailure", "ChronicCond_KidneyDisease",
    "ChronicCond_Cancer", "ChronicCond_ObstrPulmonary", "ChronicCond_Depression",
    "ChronicCond_Diabetes", "ChronicCond_IschemicHeart", "ChronicCond_Osteoporasis",
    "ChronicCond_rheumatoidarthritis", "ChronicCond_stroke",
]
DIAG_COLS = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
PROC_COLS = [f"ClmProcedureCode_{i}" for i in range(1, 7)]

# Columns that must never become model inputs.
ID_COLS = ["Provider"]
LABEL_COL = "PotentialFraud"


def find_file(data_dir: str | Path, prefix: str) -> Path:
    matches = sorted(Path(data_dir).glob(f"{prefix}*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one file starting with '{prefix}' in {data_dir}, found {matches}")
    return matches[0]


def load_raw(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Read the four Train CSVs (the Test CSVs have no labels, so they are not used)."""
    return {name: pd.read_csv(find_file(data_dir, prefix)) for name, prefix in TRAIN_FILES.items()}


def _prepare_claims(inpatient: pd.DataFrame, outpatient: pd.DataFrame,
                    beneficiary: pd.DataFrame) -> pd.DataFrame:
    ip = inpatient.copy()
    op = outpatient.copy()
    ip["is_inpatient"] = 1
    op["is_inpatient"] = 0
    claims = pd.concat([ip, op], ignore_index=True, sort=False)

    for col in ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt"]:
        claims[col] = pd.to_datetime(claims[col], errors="coerce")

    claims["claim_days"] = (claims["ClaimEndDt"] - claims["ClaimStartDt"]).dt.days
    claims["length_of_stay"] = (claims["DischargeDt"] - claims["AdmissionDt"]).dt.days
    claims["n_diag_codes"] = claims[DIAG_COLS].notna().sum(axis=1)
    claims["n_proc_codes"] = claims[PROC_COLS].notna().sum(axis=1)
    claims["n_physicians"] = claims[["AttendingPhysician", "OperatingPhysician", "OtherPhysician"]].notna().sum(axis=1)
    claims["DeductibleAmtPaid"] = claims["DeductibleAmtPaid"].fillna(0)

    # Patient (beneficiary) attributes, one row per claim (BeneID is unique in beneficiary).
    bene = beneficiary.copy()
    bene["DOB"] = pd.to_datetime(bene["DOB"], errors="coerce")
    bene["is_deceased"] = bene["DOD"].notna().astype(int)
    bene["n_chronic"] = (bene[CHRONIC_COLS] == 1).sum(axis=1)  # 1 = yes, 2 = no in this dataset
    bene["is_renal"] = (bene["RenalDiseaseIndicator"] == "Y").astype(int)
    bene = bene[["BeneID", "DOB", "is_deceased", "n_chronic", "is_renal", "Gender"]]
    if bene["BeneID"].duplicated().any():
        raise ValueError("BeneID is not unique in the beneficiary file; join would inflate rows")

    n_before = len(claims)
    claims = claims.merge(bene, on="BeneID", how="left", validate="many_to_one")
    assert len(claims) == n_before, "beneficiary join changed the claim count"
    claims["patient_age"] = (claims["ClaimStartDt"] - claims["DOB"]).dt.days / 365.25
    return claims


def build_provider_features(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return one row per Provider with numeric features (no label column)."""
    claims = _prepare_claims(raw["inpatient"], raw["outpatient"], raw["beneficiary"])
    g = claims.groupby("Provider")

    feats = pd.DataFrame({
        "n_claims": g.size(),
        "n_inpatient_claims": g["is_inpatient"].sum(),
        "inpatient_share": g["is_inpatient"].mean(),
        "reimb_sum": g["InscClaimAmtReimbursed"].sum(),
        "reimb_mean": g["InscClaimAmtReimbursed"].mean(),
        "reimb_max": g["InscClaimAmtReimbursed"].max(),
        "reimb_std": g["InscClaimAmtReimbursed"].std(),
        "deductible_mean": g["DeductibleAmtPaid"].mean(),
        "n_unique_patients": g["BeneID"].nunique(),
        "n_unique_attending": g["AttendingPhysician"].nunique(),
        "n_unique_operating": g["OperatingPhysician"].nunique(),
        "n_unique_diag_1": g["ClmDiagnosisCode_1"].nunique(),
        "n_unique_drg": g["DiagnosisGroupCode"].nunique(),
        "claim_days_mean": g["claim_days"].mean(),
        "length_of_stay_mean": g["length_of_stay"].mean(),
        "diag_codes_per_claim": g["n_diag_codes"].mean(),
        "proc_codes_per_claim": g["n_proc_codes"].mean(),
        "physicians_per_claim": g["n_physicians"].mean(),
        "patient_age_mean": g["patient_age"].mean(),
        "patient_deceased_share": g["is_deceased"].mean(),
        "patient_chronic_mean": g["n_chronic"].mean(),
        "patient_renal_share": g["is_renal"].mean(),
    })

    # Ratio features (more stable than raw totals when volume differs a lot).
    feats["claims_per_patient"] = feats["n_claims"] / feats["n_unique_patients"]
    feats["claims_per_attending"] = feats["n_claims"] / feats["n_unique_attending"].replace(0, np.nan)
    feats["log_n_claims"] = np.log1p(feats["n_claims"])
    feats["log_reimb_sum"] = np.log1p(feats["reimb_sum"].clip(lower=0))

    feats = feats.reset_index()
    if feats["Provider"].duplicated().any():
        raise ValueError("Gold features must have exactly one row per Provider")
    return feats


def feature_columns(features: pd.DataFrame) -> list[str]:
    """All model inputs: every column except IDs and the label."""
    return [c for c in features.columns if c not in ID_COLS + [LABEL_COL]]
