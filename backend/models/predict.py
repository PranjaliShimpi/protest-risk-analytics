from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from backend.models.artifacts import (
    load_feature_columns,
    load_features_df,
    load_model,
    load_sustain_model,
)


FEATURE_LABELS = {
    "agency_prior_protest_rate": "Agency prior protest rate",
    "group_protest_rate": "Segment protest rate",
    "group_sustain_rate": "Segment sustain rate",
    "median_resolution_days": "Median resolution days",
    "group_median_resolution_days": "Segment median resolution",
    "value_millions": "Award value (M)",
    "log_value": "Log value",
    "value_band_enc": "Value band (encoded)",
    "incumbent_displacement_flag": "Incumbent displaced",
    "sow_ambiguity_score": "SOW ambiguity",
    "sow_incumbent_interaction": "SOW × incumbent",
    "protester_history": "Protester history",
    "days_since_last_decision": "Days since last decision",
}


def _label(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    return feature.replace("_", " ").title()

def _compute_driver_impacts(model, X: pd.DataFrame, feature_columns):
    
    lr = model.named_steps.get("lr")
    if lr is None:
        return {}
    coefs = lr.coef_.ravel()
    vals = X.iloc[0].values.astype(float)
    raw = {_label(f): float(c * v) for f, c, v in zip(feature_columns, coefs, vals)}
    # Sort & normalize
    top = sorted(raw.items(), key=lambda kv: -abs(kv[1]))[:5]
    denom = sum(abs(v) for _, v in top) or 1.0
    return {k: float(v / denom) for k, v in top}

def predict_protest_risk(opp_id: int) -> Optional[Dict]:
    feats = load_features_df()
    feature_columns = load_feature_columns()
    row = feats[feats["opp_id"] == opp_id]
    if row.empty:
        return None
    X = row[feature_columns]
    protest_model = load_model()
    proba = float(protest_model.predict_proba(X)[0, 1])
    drivers = _compute_driver_impacts(protest_model, X, feature_columns)

    sustain_model = load_sustain_model()
    sustain_proba = 0.0
    sustain_drivers = {}
    if sustain_model is not None:
        sustain_proba = float(sustain_model.predict_proba(X)[0, 1])
        sustain_drivers = _compute_driver_impacts(sustain_model, X, feature_columns)

    return {
        "oppId": int(opp_id),
        "probability": proba,
        "drivers": drivers,
        "sustain_probability": sustain_proba,
        "sustain_drivers": sustain_drivers,
    }
