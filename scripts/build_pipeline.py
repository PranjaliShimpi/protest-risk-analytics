import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from backend.etl.load_and_clean import build_base
from backend.features.build_features import build_features
from backend.models.train import train

if __name__ == "__main__":
    print("Building base...")
    build_base()
    print("Building features...")
    build_features()
    print("Training model...")
    print(train())
    print("Done.")
