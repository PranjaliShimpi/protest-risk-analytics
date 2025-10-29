
from __future__ import annotations

from backend.etl.load_and_clean import run as run_etl
from backend.features.build_features import run as run_features
from backend.models.train import run as run_training


def main() -> None:
    base_path = run_etl()
    print(f"ETL complete: {base_path}")
    features_path = run_features()
    print(f"Features complete: {features_path}")
    metrics = run_training()
    print("Model metrics:", metrics)


if __name__ == "__main__":
    main()
