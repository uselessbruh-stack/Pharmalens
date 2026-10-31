"""Train a model from an auditable experiment config."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.experiment import main

if __name__ == "__main__":
    raise SystemExit(main())
