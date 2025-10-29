from __future__ import annotations
from functools import lru_cache
import json
import joblib
import pandas as pd

from backend.config import (
    BASE_CSV_PATH,
    FEATURE_COLUMNS_PATH,
    FEATURES_CSV_PATH,
    METRICS_PATH,
    MODEL_PATH,
    SUSTAIN_MODEL_PATH,
)

@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_sustain_model():
    try:
        return joblib.load(SUSTAIN_MODEL_PATH)
    except FileNotFoundError:
        return None

@lru_cache(maxsize=1)
def load_feature_columns():
    if FEATURE_COLUMNS_PATH.exists():
        return json.loads(FEATURE_COLUMNS_PATH.read_text())
    return []

@lru_cache(maxsize=1)
def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}

@lru_cache(maxsize=1)
def load_base_df():
    return pd.read_csv(BASE_CSV_PATH)

@lru_cache(maxsize=1)
def load_features_df():
    return pd.read_csv(FEATURES_CSV_PATH)
