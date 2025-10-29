from __future__ import annotations
import numpy as np, pandas as pd
from typing import Optional
from backend.models.artifacts import load_base_df


def _norm(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    return value.upper()


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["resolution_for_median"] = np.where(
        df["protested"].astype(int) == 1,
        df["median_resolution_days"],
        np.nan,
    )
    return df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["agency_id", "naics", "value_band"], dropna=False)
        .agg(
            opportunity_count=("opp_id", "size"),
            protested_sum=("protested", "sum"),
            sustained_sum=("sustained_flag", "sum"),
            median_resolution_days=("resolution_for_median", "median"),
        )
        .reset_index()
    )
    grouped["protest_rate"] = grouped["protested_sum"] / grouped["opportunity_count"]
    grouped["sustain_rate"] = np.where(
        grouped["protested_sum"] > 0,
        grouped["sustained_sum"] / grouped["protested_sum"],
        0.0,
    )
    grouped = grouped.drop(columns=["protested_sum", "sustained_sum"])
    grouped["opportunity_count"] = grouped["opportunity_count"].astype(int)
    grouped[["protest_rate", "sustain_rate", "median_resolution_days"]] = grouped[
        ["protest_rate", "sustain_rate", "median_resolution_days"]
    ].fillna(0.0)
    return grouped


def get_protest_aggregates(
    agency_id: Optional[str] = None,
    naics: Optional[str] = None,
    value_band: Optional[str] = None,
):
    df = load_base_df()
    df = _prepare_dataframe(df)

    if agency_id := _norm(agency_id):
        df = df[df["agency_id"].astype(str).str.upper() == agency_id]
    if naics := _norm(naics):
        df = df[df["naics"].astype(str).str.upper().str.startswith(naics)]
    if value_band := _norm(value_band):
        df = df[df["value_band"].astype(str).str.upper() == value_band]

    if df.empty:
        return {"records": [], "total": 0}

    grouped = _aggregate(df)
    return {
        "records": grouped.to_dict(orient="records"),
        "total": int(grouped["opportunity_count"].sum()),
    }


def build_aggregate_table() -> list[dict]:
    df = load_base_df()
    grouped = _aggregate(_prepare_dataframe(df))
    return grouped.to_dict(orient="records")
