from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from backend.config import BASE_CSV_PATH, FEATURES_CSV_PATH


FEATURE_COLUMNS = [
    "agency_prior_protest_rate",
    "group_protest_rate",
    "group_sustain_rate",
    "median_resolution_days",
    "group_median_resolution_days",
    "value_millions",
    "log_value",
    "value_band_enc",
    "incumbent_displacement_flag",
    "sow_ambiguity_score",
    "sow_incumbent_interaction",
    "protester_history",
    "days_since_last_decision",
]

VALUE_BAND_MAP = {"LT1M": 0, "1-10M": 1, "10-50M": 2, "50M+": 3}


def build_features() -> Tuple[pd.DataFrame, list[str]]:
    base = pd.read_csv(
        BASE_CSV_PATH,
        parse_dates=["posted_date", "due_date", "last_decision_date"],
    )

    grouped = (
        base.groupby(["agency_id", "naics", "value_band"], dropna=False)
        .agg(
            group_protest_rate=("protested", "mean"),
            group_sustain_rate=("sustain_rate_opp", "mean"),
            group_median_resolution_days=("median_resolution_days", "median"),
        )
        .reset_index()
    )

    features = base.merge(
        grouped,
        on=["agency_id", "naics", "value_band"],
        how="left",
        suffixes=("", "_group"),
    )

    features["value_millions"] = features["value"] / 1_000_000.0
    features["log_value"] = np.log1p(features["value"])
    features["value_band_enc"] = features["value_band"].map(VALUE_BAND_MAP).fillna(-1).astype(int)
    features["sow_incumbent_interaction"] = (
        features["sow_ambiguity_score"] * features["incumbent_displacement_flag"]
    )
    features["days_since_last_decision"] = (
        (features["posted_date"] - features["last_decision_date"]).dt.days.fillna(365)
    )
    features["days_since_last_decision"] = features["days_since_last_decision"].clip(lower=-365, upper=365)

    numeric_cols = FEATURE_COLUMNS + [
        "protested",
        "sustain_rate_opp",
        "group_median_resolution_days",
    ]
    for col in FEATURE_COLUMNS:
        if col not in features.columns:
            features[col] = 0.0

    numeric_cols_present = [col for col in numeric_cols if col in features.columns]
    for col in numeric_cols_present:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    features[FEATURE_COLUMNS] = features[FEATURE_COLUMNS].fillna(0.0)

    FEATURES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_CSV_PATH, index=False)
    return features, FEATURE_COLUMNS
