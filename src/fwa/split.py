"""Provider-disjoint, label-stratified train / validation / test split (T06).

The split is done on unique Provider IDs, so no provider can appear in two sets.
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_COL = "split"


def make_provider_split(labels: pd.DataFrame, seed: int = 42,
                        val_frac: float = 0.15, test_frac: float = 0.15) -> pd.DataFrame:
    """Return a DataFrame with columns Provider, label, split (train / validation / test)."""
    df = labels[["Provider", "PotentialFraud"]].copy()
    if df["Provider"].duplicated().any():
        raise ValueError("Label file must have one row per Provider")
    df["label"] = (df["PotentialFraud"] == "Yes").astype(int)

    holdout_frac = val_frac + test_frac
    train_ids, rest = train_test_split(df, test_size=holdout_frac, stratify=df["label"], random_state=seed)
    val_ids, test_ids = train_test_split(rest, test_size=test_frac / holdout_frac,
                                         stratify=rest["label"], random_state=seed)

    out = pd.concat([
        train_ids.assign(**{SPLIT_COL: "train"}),
        val_ids.assign(**{SPLIT_COL: "validation"}),
        test_ids.assign(**{SPLIT_COL: "test"}),
    ])[["Provider", "label", SPLIT_COL]].sort_values("Provider").reset_index(drop=True)

    assert_provider_disjoint(out)
    return out


def assert_provider_disjoint(split_df: pd.DataFrame) -> None:
    sets = {name: set(g["Provider"]) for name, g in split_df.groupby(SPLIT_COL)}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = sets[a] & sets[b]
            if overlap:
                raise AssertionError(f"{len(overlap)} providers appear in both {a} and {b}")


def split_manifest(split_df: pd.DataFrame, seed: int, labels_file: str) -> dict:
    """Small, human-readable record of exactly how the split was made."""
    summary = (split_df.groupby(SPLIT_COL)["label"].agg(providers="size", fraud="sum").reset_index())
    ids_hash = hashlib.sha256(
        ",".join(split_df.sort_values("Provider")["Provider"] + ":" + split_df[SPLIT_COL]).encode()
    ).hexdigest()
    return {
        "method": "stratified on PotentialFraud, split on unique Provider IDs",
        "seed": seed,
        "fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "labels_file": labels_file,
        "split_assignment_sha256": ids_hash,
        "sets": {row[SPLIT_COL]: {"providers": int(row["providers"]), "fraud": int(row["fraud"]),
                                   "fraud_rate": round(row["fraud"] / row["providers"], 4)}
                 for _, row in summary.iterrows()},
    }


def to_json(obj: dict) -> str:
    return json.dumps(obj, indent=2)
