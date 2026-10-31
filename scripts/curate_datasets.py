"""Combine prepared endpoint datasets with provenance and overlap checks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.curation import main

if __name__ == "__main__":
    raise SystemExit(main())
