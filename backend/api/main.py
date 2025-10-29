from __future__ import annotations
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from strawberry.asgi import GraphQL
import io
import json
import numpy as np
import pandas as pd

from backend.models.agg import get_protest_aggregates, build_aggregate_table
from backend.models.predict import predict_protest_risk
from backend.models.artifacts import load_metrics
from backend.api.graphql_schema import schema as graphql_schema

# FastAPI initialization

app = FastAPI(title="Protest Risk Analytics API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models

class ProtestAggOut(BaseModel):
    records: list
    total: int


class ProtestRiskOut(BaseModel):
    oppId: int
    probability: float
    sustain_probability: float
    drivers: dict
    sustain_drivers: dict


class ModelMetric(BaseModel):
    auc: float
    calibration_error: float
    brier: float
    n_train: int
    n_test: int


class ModelMetricsOut(BaseModel):
    protest: ModelMetric
    sustain: ModelMetric


# Routes

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/protestAgg", response_model=ProtestAggOut)
def protest_agg(
    agencyId: str | None = Query(None),
    naics: str | None = Query(None),
    valueBand: str | None = Query(None)
):
    result = get_protest_aggregates(
        agency_id=agencyId,
        naics=naics,
        value_band=valueBand
    )
    return result


@app.get("/protestRisk", response_model=ProtestRiskOut)
def protest_risk(oppId: int = Query(...)):
    result = predict_protest_risk(oppId)
    if not result:
        raise HTTPException(status_code=404, detail="Opportunity not found")
    return result


@app.get("/exportAggregates")
def export_aggregates():
    table = build_aggregate_table()
    if isinstance(table, pd.DataFrame):
        df = table.copy()
    else:
        if not table:
            raise HTTPException(status_code=404, detail="No aggregate data available.")
        df = pd.DataFrame(table)
    if df.empty:
        raise HTTPException(status_code=404, detail="No aggregate data available.")
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=aggregates.csv"},
    )


# Converts DataFrame to JSON-serializable format
@app.get("/protestAggTable")
def protest_agg_table():
    df = build_aggregate_table()
    if df is None or df.empty:
        return []
    
    data = json.loads(df.replace({np.nan: None}).to_json(orient="records"))
    return data


@app.get("/modelMetrics", response_model=ModelMetricsOut)
def model_metrics():
    m = load_metrics() or {}

    def normalize(key: str) -> ModelMetric:
        data = m.get(key, {}) if isinstance(m, dict) else {}
        return ModelMetric(
            auc=float(data.get("auc", 0.0)),
            calibration_error=float(data.get("calibration_error", 0.0)),
            brier=float(data.get("brier", 0.0)),
            n_train=int(data.get("n_train", 0)),
            n_test=int(data.get("n_test", 0)),
        )

    return {"protest": normalize("protest"), "sustain": normalize("sustain")}


# GraphQL integration

graphql_app = GraphQL(graphql_schema)
app.add_route("/graphql", graphql_app)
app.add_websocket_route("/graphql", graphql_app)
