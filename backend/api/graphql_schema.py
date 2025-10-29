from __future__ import annotations

import typing as t

import strawberry

from backend.models.agg import get_protest_aggregates
from backend.models.predict import predict_protest_risk


@strawberry.type
class ProtestAggregate:
    agency_id: t.Optional[str]
    naics: t.Optional[str]
    value_band: t.Optional[str]
    opportunity_count: int
    protest_rate: float
    sustain_rate: float
    median_resolution_days: float


@strawberry.type
class DriverImpact:
    label: str
    weight: float


@strawberry.type
class ProtestRisk:
    opp_id: int
    probability: float
    sustain_probability: float
    protest_drivers: t.List[DriverImpact]
    sustain_drivers: t.List[DriverImpact]


@strawberry.type
class Query:
    @strawberry.field(description="Aggregate protest metrics for an agency/NAICS/value band.")
    def protestAgg(
        self,
        agency_id: t.Optional[str] = strawberry.UNSET,
        naics: t.Optional[str] = strawberry.UNSET,
        value_band: t.Optional[str] = strawberry.UNSET,
    ) -> t.List[ProtestAggregate]:
        result = get_protest_aggregates(
            agency_id=agency_id if agency_id is not strawberry.UNSET else None,
            naics=naics if naics is not strawberry.UNSET else None,
            value_band=value_band if value_band is not strawberry.UNSET else None,
        )
        records = result.get("records", [])
        aggregates: t.List[ProtestAggregate] = [
            ProtestAggregate(
                agency_id=row.get("agency_id"),
                naics=row.get("naics"),
                value_band=row.get("value_band"),
                opportunity_count=int(row.get("opportunity_count", 0)),
                protest_rate=float(row.get("protest_rate", 0.0)),
                sustain_rate=float(row.get("sustain_rate", 0.0)),
                median_resolution_days=float(row.get("median_resolution_days", 0.0)),
            )
            for row in records
        ]
        return aggregates

    @strawberry.field(description="Predict protest and sustain probability for an opportunity.")
    def protestRisk(self, opp_id: int) -> t.Optional[ProtestRisk]:
        result = predict_protest_risk(opp_id)
        if not result:
            return None
        protest_drivers = [
            DriverImpact(label=label, weight=float(weight))
            for label, weight in result.get("drivers", {}).items()
        ]
        sustain_drivers = [
            DriverImpact(label=label, weight=float(weight))
            for label, weight in result.get("sustain_drivers", {}).items()
        ]
        return ProtestRisk(
            opp_id=int(result["oppId"]),
            probability=float(result["probability"]),
            sustain_probability=float(result.get("sustain_probability", 0.0)),
            protest_drivers=protest_drivers,
            sustain_drivers=sustain_drivers,
        )


schema = strawberry.Schema(query=Query)
