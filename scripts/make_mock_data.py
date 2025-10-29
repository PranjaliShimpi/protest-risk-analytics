import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# Agencies, categories, outcomes
agencies = ["DOD", "DOE", "HHS", "GSA"]
naics = ["541511", "541512", "541513"]
outcomes = ["sustained", "denied"]

# GAO protests (historical data)
gao = pd.DataFrame({
    "opp_id": range(1000, 1100),
    "agency_id": rng.choice(agencies, 100),
    "naics": rng.choice(naics, 100),
    "outcome": rng.choice(outcomes, 100, p=[0.3, 0.7]),
    "decision_date": pd.date_range("2022-01-01", periods=100, freq="7D"),
    "protester_id": rng.integers(1, 20, 100),
    "awardee_id": rng.integers(1, 20, 100),
    "resolution_days": rng.integers(15, 120, 100)
})
gao.to_csv("data/gao_protests.csv", index=False)

# USASpending / FPDS (award values)
def band(v):
    if v < 200_000:
        return "SMALL"
    elif v < 2_000_000:
        return "MID"
    else:
        return "LARGE"

values = rng.integers(50_000, 5_000_000, 100)
usa = pd.DataFrame({
    "opp_id": range(1000, 1100),
    "agency_id": rng.choice(agencies, 100),
    "value": values,
    "value_band": [band(v) for v in values],
    "incumbent_vendor_id": rng.integers(1, 20, 100)
})
usa.to_csv("data/usaspending.csv", index=False)

# SAM opportunities (solicitations)
sow = [
    "Clear SOW with detailed scope and deliverables.",
    "Ambiguous requirements; integration unclear; TBD interfaces.",
    "Complex multi-vendor environment; heavy coordination."
]
sam = pd.DataFrame({
    "opp_id": range(1000, 1100),
    "naics": rng.choice(naics, 100),
    "sow_text": rng.choice(sow, 100),
    "due_date": pd.date_range("2025-09-01", periods=100, freq="3D")
})
sam.to_csv("data/sam_opps.csv", index=False)

print("Mock data created successfully in /data folder!")
