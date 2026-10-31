"""Normalize acquired source data into endpoint-specific, versioned datasets."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preparation import main

if __name__ == "__main__":
    raise SystemExit(main())
