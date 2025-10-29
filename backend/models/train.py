from __future__ import annotations

import json
from typing import Dict

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.config import (
    FEATURE_COLUMNS_PATH,
    FEATURES_CSV_PATH,
    METRICS_PATH,
    MODEL_PATH,
    SUSTAIN_MODEL_PATH,
)
from backend.features.build_features import FEATURE_COLUMNS


def _train_pipeline(X, y) -> Pipeline:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=1.5,
                    random_state=42,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return pipe


def _compute_metrics(model: Pipeline, X_test, y_test) -> Dict[str, float]:
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba) if len(set(y_test)) > 1 else 0.5
    brier = brier_score_loss(y_test, proba)
    calibration_error = abs(proba.mean() - y_test.mean())
    return {
        "auc": float(auc),
        "brier": float(brier),
        "calibration_error": float(calibration_error),
        "n_test": int(len(y_test)),
    }


def train() -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(FEATURES_CSV_PATH)
    FEATURE_COLUMNS_PATH.write_text(json.dumps(FEATURE_COLUMNS))

    X = df[FEATURE_COLUMNS]
    y_protest = df["protested"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_protest, test_size=0.3, random_state=42, stratify=y_protest
    )

    protest_model = _train_pipeline(X_tr, y_tr)
    protest_metrics = _compute_metrics(protest_model, X_te, y_te)
    protest_metrics.update({"n_train": int(len(y_tr))})

    joblib.dump(protest_model, MODEL_PATH)

    sustain_df = df[df["protested"] == 1].copy()
    sustain_df = sustain_df[sustain_df["sustained_flag"].isin([0, 1])]
    sustain_results = {}

    if len(sustain_df["sustained_flag"].unique()) > 1:
        Xs = sustain_df[FEATURE_COLUMNS]
        ys = sustain_df["sustained_flag"].astype(int)
        Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
            Xs, ys, test_size=0.3, random_state=42, stratify=ys
        )
        sustain_model = _train_pipeline(Xs_tr, ys_tr)
        sustain_metrics = _compute_metrics(sustain_model, Xs_te, ys_te)
        sustain_metrics.update({"n_train": int(len(ys_tr))})
        joblib.dump(sustain_model, SUSTAIN_MODEL_PATH)
        sustain_results = sustain_metrics
    else:
        joblib.dump(None, SUSTAIN_MODEL_PATH)
        sustain_results = {"auc": 0.5, "brier": 0.25, "calibration_error": 0.0, "n_train": 0, "n_test": 0}

    metrics = {"protest": protest_metrics, "sustain": sustain_results}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    print(train())
