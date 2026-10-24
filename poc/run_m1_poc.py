"""M1 proof of concept: Kaggle data -> provider features -> provider split -> LR vs HGB -> top-20.

Runs locally (pandas + scikit-learn) in about a minute. It is a POC to check that the M1
plan works end to end on the real dataset BEFORE porting each step to Databricks
(Bronze/Gold Delta tables, MLflow, Model Serving, Apps).

Usage (from the repo root):
    python poc/run_m1_poc.py --data-dir ../archive --out-dir poc/output

The data directory must contain the four Kaggle Train*.csv files. Raw data is never
copied into the repository.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fwa.evaluate import latency_ms, quality_metrics, rank_providers, robustness  # noqa: E402
from fwa.features import build_provider_features, feature_columns, find_file, load_raw  # noqa: E402
from fwa.split import make_provider_split, split_manifest, to_json  # noqa: E402

SEED = 42


def build_models() -> dict:
    # Preprocessing lives INSIDE each pipeline, so it is fit on the training set only.
    return {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                           random_state=SEED)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="folder with the Kaggle Train*.csv files")
    parser.add_argument("--out-dir", default="poc/output")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Data + Gold features (one row per provider, no label)
    raw = load_raw(args.data_dir)
    features = build_provider_features(raw)
    cols = feature_columns(features)
    print(f"Gold features: {len(features)} providers x {len(cols)} features")

    # 2) Provider-disjoint stratified split (T06)
    split = make_provider_split(raw["labels"], seed=SEED)
    manifest = split_manifest(split, SEED, find_file(args.data_dir, "Train-").name)
    (out / "split_manifest.json").write_text(to_json(manifest))
    split.to_csv(out / "provider_split.csv", index=False)
    print("Split:", json.dumps(manifest["sets"]))

    data = split.merge(features, on="Provider", how="left", validate="one_to_one")
    assert data[cols].notna().any(axis=1).all(), "every labeled provider should have claims"
    parts = {name: data[data["split"] == name].reset_index(drop=True) for name in ["train", "validation", "test"]}
    X = {k: v[cols] for k, v in parts.items()}
    y = {k: v["label"].to_numpy() for k, v in parts.items()}
    ids = {k: v["Provider"] for k, v in parts.items()}

    # 3) Train both models on train; compare on validation only (T07, T12)
    results, models = {}, {}
    for name, model in build_models().items():
        model.fit(X["train"], y["train"])
        models[name] = model
        results[name] = {"validation": quality_metrics(y["validation"], model.predict_proba(X["validation"])[:, 1],
                                                        ids["validation"])}

    champion = max(results, key=lambda n: results[n]["validation"]["pr_auc"])

    # 4) Test set: used once per model, after the champion is chosen
    for name, model in models.items():
        results[name]["test"] = quality_metrics(y["test"], model.predict_proba(X["test"])[:, 1], ids["test"])
        results[name]["latency"] = latency_ms(model, X["test"], ids["test"])
        results[name]["robustness"] = robustness(model, X["test"], ids["test"])

    summary = {"champion_by_validation_pr_auc": champion, "seed": SEED, "n_features": len(cols),
               "features": cols, "results": results}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))

    # 5) What /recommendations?limit=20 would return (scored on the unseen test providers)
    top20 = rank_providers(ids["test"], models[champion].predict_proba(X["test"])[:, 1], limit=20)
    top20["model"] = champion
    top20.to_csv(out / "top20_test_providers.csv", index=False)

    rows = []
    for name, r in results.items():
        rows.append({"model": name,
                     "val_pr_auc": r["validation"]["pr_auc"], "test_pr_auc": r["test"]["pr_auc"],
                     "test_roc_auc": r["test"]["roc_auc"], "test_p@20": r["test"]["precision_at_20"],
                     "test_p@50": r["test"]["precision_at_50"], "p95_ms": r["latency"]["p95_ms"],
                     "top20_overlap": r["robustness"]["top20_overlap_mean"]})
    table = pd.DataFrame(rows).round(3)
    table.to_csv(out / "model_comparison.csv", index=False)
    print("\n" + table.to_string(index=False))
    print(f"\nChampion (chosen on validation PR-AUC): {champion}")
    print(f"Test base rate: {results[champion]['test']['base_rate']:.3f}")
    print("\nTop 5 recommended providers:\n" + top20.head().to_string(index=False))
    print(f"\nOutputs written to {out.resolve()}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
