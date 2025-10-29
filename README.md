## Protest Risk Analytics

Analytics stack for forecasting the protest probability of federal opportunities using GAO protest outcomes, FPDS/USAspending metadata, and SAM.gov solicitations.

### Project Structure

- `backend/` – FastAPI application, ETL/feature engineering scripts, dual logistic regression models, and automated tests.
- `frontend/` – React dashboard that surfaces agency aggregates, model diagnostics, and opportunity-level risk.
- `data/` – Raw CSV inputs plus lightweight samples used by the ETL pipeline.
- `artifacts/` – Persisted CSV/model assets (`base.csv`, `features.csv`, `model.pkl`, `feature_columns.json`, `metrics.json`).
- `scripts/` – Utility scripts, including `build_pipeline.py` (orchestration) and `generate_sample_data.py` (seed synthetic inputs).
- `Dockerfile.backend`, `Dockerfile.frontend`, `docker-compose.yml` – Optional containerisation for a one-command local stack.

### Getting Started

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r backend/requirements.txt
   ```

2. **Rebuild artifacts (ETL → features → model)**

   ```bash
   python scripts/build_pipeline.py
   ```

   The pipeline produces `artifacts/base.csv`, `artifacts/features.csv`, `artifacts/model.pkl`, `artifacts/feature_columns.json`, and `artifacts/metrics.json`.

3. **Run automated tests**

   ```bash
   PYTHONPATH=. pytest backend/tests
   ```

4. **Start the API**

   ```bash
   uvicorn backend.api.main:app --reload
   ```

5. **Launch the dashboard**

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

   Set `VITE_API_URL` if the API is not running on `http://127.0.0.1:8000`.

### Docker Quick Start (optional)

1. [Install Docker Desktop](https://www.docker.com/products/docker-desktop) and ensure the `docker` CLI is available.
2. From the project root, build the images:

   ```bash
   docker compose build
   ```

3. Launch both services (FastAPI at port `8000`, frontend at port `5173`):

   ```bash
   docker compose up
   ```

   The frontend container is configured with `VITE_API_URL=http://backend:8000`, so the React app automatically targets the containerised API. Visit `http://localhost:5173` in your browser.

4. To stop the stack:

   ```bash
   docker compose down
   ```

### Key Endpoints

- `GET /health` – Service heartbeat.
- `GET /protestAgg` – Aggregated protest rate, sustain rate, median resolution days, and opportunity counts (supports optional `agencyId`, `naics`, `valueBand` filters).
- `GET /protestRisk?oppId=...` – Opportunity-level protest probability, sustain probability, and top drivers for each.
- `GET /protestAggTable` – Agency × NAICS drill-down (JSON).
- `GET /exportAggregates` – Same aggregates as CSV download.
- `GET /modelMetrics` – Persisted AUC, calibration error, and Brier scores for protest and sustain models.
- `POST /graphql` – Strawberry GraphQL endpoint exposing `protestAgg` and `protestRisk` queries (see below).

Sample GraphQL query:

```graphql
query Example($opp: Int!) {
  protestRisk(oppId: $opp) {
    oppId
    probability
    sustainProbability
    protestDrivers { label weight }
  }
  protestAgg(agencyId: "DHS") {
    agencyId
    opportunityCount
    protestRate
  }
}
```

### Model + Feature Engineering Highlights

- Normalized GAO, USAspending, and SAM.gov datasets into a unified opportunity table with protest labels.
- Engineered protest rate, sustain rate, and resolution metrics at the agency × NAICS × value band grain.
- Derived feature signals including incumbent displacement, SOW ambiguity, agency prior protest rates, and protester history.
- Two logistic regression pipelines (scikit-learn) trained on standardized features with balanced class weights: one models protest likelihood, the other models sustain probability conditional on protest; metrics for both are serialized for runtime availability.
- Current benchmark (`scripts/build_pipeline.py`) produces protest AUC ≈ 1.00, sustain AUC ≈ 0.87, with calibration errors ≈ 0.002 and 0.140 respectively (see `artifacts/metrics.json` or `GET /modelMetrics`).

### Frontend Overview

- Agency “Risk Meter” card showing protest/sustain rates, resolution timelines, and coverage counts.
- Opportunity sidebar with probability and driver explanation (tooltips convey impact definitions).
- Drill-down table with CSV export and inline model diagnostics (AUC, calibration error, Brier score).
- Lighthouse desktop performance: FCP ~0.4s, LCP ~0.9s, performance score 99 (uncached).

### Evaluation Checklist

- **Model performance** – Targeted AUC ≥ 0.70 and calibration error ≤ 0.05 (verify via `/modelMetrics` after training).
- **Data integrity** – `data/*.sample.csv` files provide quick validation of ETL outputs; tests patch artifacts to ensure aggregation logic matches expectations.
- **API reliability** – FastAPI endpoints covered via unit tests with dependency injection.
- **UI/UX** – Responsive cards, loading states, and tooltips for data freshness and feature impact.

### Notes

- Run `python scripts/generate_sample_data.py` to regenerate mock source files if needed.
- The pipeline persists compact CSV artifacts so they can be inspected or versioned easily (no parquet dependency).
- For reproducible experiments, adjust `random_state` in `backend/models/train.py` or extend the pipeline with cross-validation.
- Pytest is known to require Python 3.11 on macOS/arm64 (pandas 2.0.3 + numpy 1.24.4); the supplied `conda` instructions in discussion logs reproduce the test run (`PYTHONPATH=. pytest backend/tests`).
