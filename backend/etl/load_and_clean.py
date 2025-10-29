from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import BASE_CSV_PATH, DATA_DIR


def load_sources(data_dir: Path = DATA_DIR):
    """
    Source inputs:
      - gao_decisions_sample.csv: opp-level protest outcomes (can contain duplicates per opportunity)
      - usa_contracts_sample.csv: contract attributes (value, incumbent, agency, NAICS)
      - sam_opps_sample.csv: opportunity metadata (posted_date, due_date, sow_text)
    """
    gao = pd.read_csv(
        data_dir / "gao_decisions_sample.csv",
        parse_dates=["decision_date"],
    )
    usa = pd.read_csv(data_dir / "usa_contracts_sample.csv")
    sam = pd.read_csv(
        data_dir / "sam_opps_sample.csv",
        parse_dates=["posted_date", "due_date"],
    )
    return gao, usa, sam


def _value_band(value: float) -> str:
    if value < 1_000_000:
        return "LT1M"
    if value < 10_000_000:
        return "1-10M"
    if value < 50_000_000:
        return "10-50M"
    return "50M+"


def build_base() -> pd.DataFrame:
    gao, usa, sam = load_sources()

    gao["opp_id"] = gao["opp_id"].astype(int)
    usa["opp_id"] = usa["opp_id"].astype(int)
    sam["opp_id"] = sam["opp_id"].astype(int)

    gao_sorted = gao.sort_values(["protester_id", "decision_date", "opp_id"])
    gao_sorted["prior_protests"] = gao_sorted.groupby("protester_id").cumcount()

    g_agg = (
        gao_sorted.groupby("opp_id")
        .agg(
            protests=("opp_id", "size"),
            sustained=("sustained", "sum"),
            median_resolution_days=("resolution_days", "median"),
            primary_protester=("protester_id", "first"),
            protester_history=("prior_protests", "max"),
            last_decision_date=("decision_date", "max"),
        )
        .reset_index()
    )

    base = (
        usa.merge(sam, on="opp_id", how="left")
        .merge(g_agg, on="opp_id", how="left")
    )

    base["protests"] = base["protests"].fillna(0).astype(int)
    base["sustained"] = base["sustained"].fillna(0).astype(int)
    base["protested"] = (base["protests"] > 0).astype(int)
    base["protester_history"] = base["protester_history"].fillna(0).astype(int)
    base["median_resolution_days"] = base["median_resolution_days"].fillna(0.0)
    base["last_decision_date"] = pd.to_datetime(
        base["last_decision_date"], errors="coerce"
    )
    base["sustained_flag"] = (base["sustained"] > 0).astype(int)

    base["sustain_rate_opp"] = np.where(
        base["protests"] > 0, base["sustained"] / base["protests"], 0.0
    )

    base["value_band"] = base["value"].apply(_value_band)
    base["incumbent_displacement_flag"] = base["incumbent_changed"].fillna(0).astype(int)
    base["sow_ambiguity_score"] = pd.to_numeric(
        base.get("sow_ambiguity_score", 0.0), errors="coerce"
    ).fillna(0.0)

    base["log_value"] = np.log1p(base["value"])

    base = base.sort_values("posted_date").reset_index(drop=True)
    agency_expanding = (
        base.groupby("agency_id")["protested"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    base["agency_prior_protest_rate"] = agency_expanding
    base["agency_prior_protest_rate"] = base["agency_prior_protest_rate"].fillna(
        base.groupby("agency_id")["protested"].transform("mean")
    )
    base["agency_prior_protest_rate"] = base["agency_prior_protest_rate"].fillna(
        base["protested"].mean()
    )

    BASE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(BASE_CSV_PATH, index=False)
    return base
