
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main(seed: int = 42, n_opps: int = 400) -> None:
    rng = np.random.default_rng(seed)

    opp_ids = np.arange(1000, 1000 + n_opps)
    agencies = np.array(["DHS", "HHS", "VA", "DOD", "NASA", "GSA"])
    naics_codes = np.array(["541512", "541511", "541611", "541519", "541330"])

    agency_id = rng.choice(agencies, size=n_opps, replace=True)
    naics = rng.choice(naics_codes, size=n_opps, replace=True)
    incumbent_changed = rng.binomial(1, 0.35, size=n_opps)
    value = rng.lognormal(mean=14.4, sigma=0.75, size=n_opps)  # ≈ $0.5M – $100M
    sow_ambiguity = np.clip(rng.normal(loc=0.45, scale=0.25, size=n_opps), 0.0, 1.0)
    posted_dates = pd.date_range("2022-01-01", periods=n_opps, freq="5D")

    agency_bias = {
        "DHS": 0.65,
        "HHS": 0.35,
        "VA": 0.45,
        "DOD": 0.55,
        "NASA": 0.25,
        "GSA": 0.4,
    }
    naics_bias = {
        "541512": 0.5,
        "541511": 0.45,
        "541611": 0.25,
        "541519": 0.35,
        "541330": 0.15,
    }

    value_scaled = (np.log10(value) - np.log10(value).mean()) / np.log10(value).std()
    score = (
        -1.6
        + 1.1 * incumbent_changed
        + 1.25 * sow_ambiguity
        + 0.8 * value_scaled
        + np.vectorize(agency_bias.get)(agency_id)
        + np.vectorize(naics_bias.get)(naics)
    )
    protest_prob = logistic(score)
    protested = rng.binomial(1, protest_prob).astype(int)

    protester_pool = {
        "DHS": [f"DHS_{i}" for i in range(1, 15)],
        "HHS": [f"HHS_{i}" for i in range(1, 12)],
        "VA": [f"VA_{i}" for i in range(1, 18)],
        "DOD": [f"DOD_{i}" for i in range(1, 22)],
        "NASA": [f"NASA_{i}" for i in range(1, 10)],
        "GSA": [f"GSA_{i}" for i in range(1, 14)],
    }

    incumbent_vendors = rng.integers(1, 40, size=n_opps)
    awardees = rng.integers(1, 45, size=n_opps)

    usa_contracts = pd.DataFrame(
        {
            "opp_id": opp_ids,
            "agency_id": agency_id,
            "naics": naics,
            "value": value,
            "incumbent_changed": incumbent_changed,
            "incumbent_vendor_id": incumbent_vendors,
            "awardee_id": awardees,
            "sow_ambiguity_score": sow_ambiguity,
        }
    )

    sow_snippets = [
        "Requirements include complex integrations of legacy platforms with modern cloud services.",
        "Scope is clearly defined with firm metrics and acceptance criteria.",
        "Ambiguous deliverables with TBD interfaces and heavy coordination requirements.",
        "Multi-vendor environment with significant transition risk from incumbent service provider.",
    ]
    sam_opps = pd.DataFrame(
        {
            "opp_id": opp_ids,
            "title": [f"Opportunity {i}" for i in opp_ids],
            "posted_date": posted_dates,
            "due_date": posted_dates + pd.to_timedelta(rng.integers(30, 120, size=n_opps), unit="D"),
            "sow_text": rng.choice(sow_snippets, size=n_opps),
        }
    )

    gao_rows = []
    decision_start = dt.date(2023, 1, 1)
    for i, opp_id in enumerate(opp_ids):
        if protested[i] == 0:
            continue
        events = rng.integers(1, 3)
        agency = agency_id[i]
        protester_candidates = protester_pool[agency]
        protester_id = rng.choice(protester_candidates)
        for j in range(events):
            sustained_score = (
                -0.2
                + 1.1 * sow_ambiguity[i]
                + 0.35 * (1 - incumbent_changed[i])
                + 0.25 * np.vectorize(naics_bias.get)([naics[i]])[0]
            )
            sustain_prob = logistic(sustained_score)
            sustained = rng.binomial(1, sustain_prob)
            res_days = int(np.clip(rng.normal(loc=45 + 40 * sow_ambiguity[i], scale=18), 12, 160))
            gao_rows.append(
                {
                    "opp_id": opp_id,
                    "agency_id": agency,
                    "naics": naics[i],
                    "protester_id": protester_id,
                    "awardee_id": awardees[i],
                    "sustained": sustained,
                    "resolution_days": res_days,
                    "decision_date": decision_start
                    + dt.timedelta(days=int(rng.integers(0, 365) + 30 * j)),
                    "rationale": rng.choice(
                        [
                            "Evaluation criteria were not applied consistently.",
                            "Agency failed to conduct meaningful discussions.",
                            "Solicitation contains ambiguous terms.",
                            "Award decision followed the stated criteria.",
                        ]
                    ),
                }
            )

    gao_decisions = pd.DataFrame(gao_rows)

    gao_decisions.to_csv(DATA_DIR / "gao_decisions_sample.csv", index=False)
    usa_contracts.to_csv(DATA_DIR / "usa_contracts_sample.csv", index=False)
    sam_opps.to_csv(DATA_DIR / "sam_opps_sample.csv", index=False)

    print(f"Sample data refreshed under {DATA_DIR}")


if __name__ == "__main__":
    main()
