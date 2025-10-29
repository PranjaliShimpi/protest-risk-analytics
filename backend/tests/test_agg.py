import pandas as pd
import pytest

from backend.models import agg


@pytest.fixture
def sample_base_df():
    return pd.DataFrame(
        [
            {
                "opp_id": 1,
                "agency_id": "DOD",
                "naics": "541511",
                "value_band": "LARGE",
                "value": 2_500_000,
                "protests": 1,
                "sustained": 0,
                "protested": 1,
                "sustained_flag": 0,
                "median_resolution_days": 60.0,
            },
            {
                "opp_id": 2,
                "agency_id": "DOD",
                "naics": "541511",
                "value_band": "LARGE",
                "value": 1_200_000,
                "protests": 0,
                "sustained": 0,
                "protested": 0,
                "sustained_flag": 0,
                "median_resolution_days": 0.0,
            },
            {
                "opp_id": 3,
                "agency_id": "GSA",
                "naics": "541512",
                "value_band": "SMALL",
                "value": 500_000,
                "protests": 1,
                "sustained": 1,
                "protested": 1,
                "sustained_flag": 1,
                "median_resolution_days": 20.0,
            },
        ]
    )


def test_get_protest_aggregates(monkeypatch, sample_base_df):
    monkeypatch.setattr(agg, "load_base_df", lambda: sample_base_df)
    metrics = agg.get_protest_aggregates("DOD", "541511", "LARGE")
    assert metrics["total"] == 2
    assert len(metrics["records"]) == 1
    record = metrics["records"][0]
    assert record["opportunity_count"] == 2
    assert pytest.approx(record["protest_rate"], 0.01) == 0.5
    assert record["sustain_rate"] == 0.0
    assert record["median_resolution_days"] == 60.0


def test_get_protest_aggregates_case_insensitive(monkeypatch, sample_base_df):
    monkeypatch.setattr(agg, "load_base_df", lambda: sample_base_df)
    metrics = agg.get_protest_aggregates("dod", "541511", "large")
    assert metrics["total"] == 2


def test_get_protest_aggregates_empty(monkeypatch, sample_base_df):
    monkeypatch.setattr(agg, "load_base_df", lambda: sample_base_df)
    metrics = agg.get_protest_aggregates("NASA")
    assert metrics["total"] == 0
    assert metrics["records"] == []


def test_build_aggregate_table(monkeypatch, sample_base_df):
    monkeypatch.setattr(agg, "load_base_df", lambda: sample_base_df)
    rows = agg.build_aggregate_table()
    assert len(rows) == 2
    dod_row = next(r for r in rows if r["agency_id"] == "DOD")
    assert dod_row["opportunity_count"] == 2
    assert pytest.approx(dod_row["protest_rate"], 0.01) == 0.5
