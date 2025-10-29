import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.api import graphql_schema, main
from backend.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_protest_agg_success(monkeypatch, client):
    monkeypatch.setattr(
        main,
        "get_protest_aggregates",
        lambda *args, **kwargs: {
            "records": [
                {
                    "agency_id": "DOD",
                    "naics": "541511",
                    "value_band": "LARGE",
                    "opportunity_count": 5,
                    "protest_rate": 0.2,
                    "sustain_rate": 0.4,
                    "median_resolution_days": 45.0,
                }
            ],
            "total": 5,
        },
    )
    resp = client.get("/protestAgg", params={"agencyId": "DOD"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 5
    assert data["records"][0]["protest_rate"] == 0.2


def test_protest_agg_empty(monkeypatch, client):
    monkeypatch.setattr(
        main,
        "get_protest_aggregates",
        lambda *args, **kwargs: {
            "records": [],
            "total": 0,
        },
    )
    resp = client.get("/protestAgg", params={"agencyId": "DOD"})
    assert resp.status_code == 200
    assert resp.json()["total"] == 0


def test_protest_risk_success(monkeypatch, client):
    monkeypatch.setattr(
        main,
        "predict_protest_risk",
        lambda opp_id: {
            "oppId": opp_id,
            "probability": 0.75,
            "drivers": {"value": 0.2},
            "sustain_probability": 0.4,
            "sustain_drivers": {"sow": 0.3},
        },
    )
    resp = client.get("/protestRisk", params={"oppId": 101})
    assert resp.status_code == 200
    data = resp.json()
    assert data["probability"] == 0.75
    assert data["sustain_probability"] == 0.4


def test_protest_risk_not_found(monkeypatch, client):
    monkeypatch.setattr(main, "predict_protest_risk", lambda opp_id: None)
    resp = client.get("/protestRisk", params={"oppId": 404})
    assert resp.status_code == 404


def test_model_metrics(monkeypatch, client):
    monkeypatch.setattr(
        main,
        "load_metrics",
        lambda: {
            "protest": {
                "auc": 0.82,
                "calibration_error": 0.03,
                "brier": 0.17,
                "n_train": 100,
                "n_test": 80,
            },
            "sustain": {
                "auc": 0.74,
                "calibration_error": 0.04,
                "brier": 0.21,
                "n_train": 40,
                "n_test": 20,
            },
        },
    )
    resp = client.get("/modelMetrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["protest"]["auc"] == 0.82
    assert data["sustain"]["n_test"] == 20


def test_model_metrics_fallback(monkeypatch, client):
    monkeypatch.setattr(main, "load_metrics", lambda: {})
    resp = client.get("/modelMetrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["protest"]["auc"] == 0.0
    assert data["sustain"]["n_test"] == 0


def test_protest_agg_table(monkeypatch, client):
    df = pd.DataFrame(
        [
            {
                "agency_id": "DOD",
                "naics": "541511",
                "value_band": "LARGE",
                "opportunity_count": 2,
                "protest_rate": 0.5,
                "sustain_rate": 0.0,
                "median_resolution_days": 30.0,
            }
        ]
    )
    monkeypatch.setattr(main, "build_aggregate_table", lambda: df)
    resp = client.get("/protestAggTable")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["agency_id"] == "DOD"


def test_export_aggregates(monkeypatch, client):
    df = pd.DataFrame(
        [
            {
                "agency_id": "DOD",
                "naics": "541511",
                "value_band": "LARGE",
                "opportunity_count": 2,
                "protest_rate": 0.5,
                "sustain_rate": 0.0,
                "median_resolution_days": 30.0,
            }
        ]
    )
    monkeypatch.setattr(main, "build_aggregate_table", lambda: df)
    resp = client.get("/exportAggregates")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]


def test_graphql_protest_risk(monkeypatch, client):
    monkeypatch.setattr(
        graphql_schema,
        "predict_protest_risk",
        lambda opp_id: {
            "oppId": opp_id,
            "probability": 0.55,
            "drivers": {"Agency prior protest rate": 0.4},
            "sustain_probability": 0.25,
            "sustain_drivers": {"Segment sustain rate": 0.3},
        },
    )
    query = """
    query($oppId: Int!) {
      protestRisk(oppId: $oppId) {
        oppId
        probability
        sustainProbability
        protestDrivers { label weight }
        sustainDrivers { label weight }
      }
    }
    """
    resp = client.post(
        "/graphql",
        json={"query": query, "variables": {"oppId": 1000}},
    )
    assert resp.status_code == 200
    data = resp.json()["data"]["protestRisk"]
    assert data["oppId"] == 1000
    assert data["probability"] == 0.55
    assert data["sustainProbability"] == 0.25
    assert data["protestDrivers"][0]["label"] == "Agency prior protest rate"


def test_graphql_protest_agg(monkeypatch, client):
    monkeypatch.setattr(
        graphql_schema,
        "get_protest_aggregates",
        lambda *args, **kwargs: {
            "records": [
                {
                    "agency_id": "NASA",
                    "naics": "541512",
                    "value_band": "1-10M",
                    "opportunity_count": 3,
                    "protest_rate": 0.33,
                    "sustain_rate": 0.1,
                    "median_resolution_days": 42.0,
                }
            ],
            "total": 3,
        },
    )
    query = """
    query {
      protestAgg(agencyId: \"NASA\") {
        agencyId
        protestRate
        opportunityCount
      }
    }
    """
    resp = client.post("/graphql", json={"query": query})
    assert resp.status_code == 200
    nodes = resp.json()["data"]["protestAgg"]
    assert nodes[0]["agencyId"] == "NASA"
    assert nodes[0]["opportunityCount"] == 3
