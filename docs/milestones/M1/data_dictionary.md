# M1 Data Dictionary and Dataset Inventory (T03)

## Source

- **Dataset:** Healthcare Provider Fraud Detection Analysis (Kaggle)
- **Link:** https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis
- **Terms:** check and record the Kaggle license on the dataset page before submission.
- **Storage rule:** raw CSVs stay outside Git (local `archive/` folder and a Unity Catalog Volume). Never commit them.

## File inventory (Train files only)

| File | Rows | Columns | Grain | Key |
|---|---|---|---|---|
| `Train-1542865627584.csv` | 5,410 | 2 | one row per provider | `Provider` (unique) |
| `Train_Inpatientdata-1542865627584.csv` | 40,474 | 30 | one row per inpatient claim | `ClaimID` |
| `Train_Outpatientdata-1542865627584.csv` | 517,737 | 27 | one row per outpatient claim | `ClaimID` |
| `Train_Beneficiarydata-1542865627584.csv` | 138,556 | 25 | one row per patient | `BeneID` (unique) |

**Test files are not usable for evaluation.** `Test-1542969243754.csv` contains only 1,353 provider IDs and **no labels** (Kaggle competition holdout). All validation and test sets are carved out of the 5,410 labeled Train providers.

## Labels

| Value of `PotentialFraud` | Providers | Share |
|---|---|---|
| No | 4,904 | 90.6% |
| Yes | 506 | 9.4% |

- The label is **provider level**. There is no claim-level label.
- It is a **proxy label** ("potential" fraud), not an adjudicated fraud finding. It may carry investigation and selection bias.
- 38% of all claims belong to "Yes" providers, because flagged providers file many more claims.

## Joins

| From | To | Key | Cardinality | Check |
|---|---|---|---|---|
| Inpatient / Outpatient | Labels | `Provider` | many to one | every labeled provider has at least one claim |
| Inpatient / Outpatient | Beneficiary | `BeneID` | many to one | `BeneID` is unique in the beneficiary file, so the join does not inflate rows |

- Claims (inpatient + outpatient): 558,211, all `ClaimID` values unique.
- Providers with inpatient claims only: 398. Outpatient only: 3,318. Both: 1,694.
- 66% of patients appear under more than one provider, so patient-level information must be aggregated per provider, never used to link providers across splits.

## Date range

- Inpatient claims: 2008-11-27 to 2009-12-31
- Outpatient claims: 2008-12-12 to 2009-12-31
- Labels have no date, so the dataset supports a static provider-level model, not a time-based backtest.

## Important columns

### Claims (inpatient and outpatient)

| Column | Meaning | Missing (IP / OP) |
|---|---|---|
| `InscClaimAmtReimbursed` | amount reimbursed for the claim | 0% / 0% |
| `DeductibleAmtPaid` | deductible paid | 2.2% / 0% |
| `AttendingPhysician` | attending physician ID | 0.3% / 0.3% |
| `OperatingPhysician` | operating physician ID | 41.1% / 82.5% |
| `ClmDiagnosisCode_1..10` | ICD diagnosis codes | code 1: 0% / 2.0% |
| `ClmProcedureCode_1..6` | ICD procedure codes | code 1: 42.8% / 100%; codes 5 and 6 empty |
| `ClmAdmitDiagnosisCode` | admitting diagnosis | 0% / 79.6% |
| `AdmissionDt`, `DischargeDt`, `DiagnosisGroupCode` | inpatient only | |

Reimbursement: inpatient median $7,000 (max $125,000); outpatient median $80 (max $102,500).

### Beneficiary

| Column | Meaning | Notes |
|---|---|---|
| `DOB`, `DOD` | birth and death date | DOD present for 1.0% |
| `Gender`, `Race`, `State`, `County` | demographics | sensitive; not used as model features in M1 |
| `RenalDiseaseIndicator` | `Y` or `0` | |
| `ChronicCond_*` (11 columns) | chronic condition flags | **1 = yes, 2 = no** |
| `IP/OPAnnualReimbursementAmt`, `IP/OPAnnualDeductibleAmt` | yearly totals per patient | |

## Provider volume

Claims per provider: min 1, median 31, mean 103, max 8,240. Small providers produce noisy averages; keep `n_claims` (and its log) as features so models can weigh that.

## Columns excluded from model inputs

- `Provider` (ID), `PotentialFraud` (label)
- Raw IDs (`BeneID`, `ClaimID`, physician IDs): only used for counts
- `Race`, `Gender`, `State`, `County`: excluded in M1 to avoid unsupported fairness claims (revisit in M4)
