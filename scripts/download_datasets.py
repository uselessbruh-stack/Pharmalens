"""Run the dataset acquisition CLI without installing PharmaLens dependencies."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.acquisition import main

if __name__ == "__main__":
    raise SystemExit(main())
