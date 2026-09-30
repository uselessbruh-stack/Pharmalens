"""
PharmaLens Utilities — Configuration, Paths, Seeds, Device Management
"""

import os
import json
import random
import logging
from pathlib import Path
from datetime import datetime

import numpy as np

# ==============================================================================
# Project Paths
# ==============================================================================

# Root of the project (two levels up from src/utils/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
SPLITS_DIR = PROCESSED_DATA_DIR / "splits"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_MODELS_DIR = MODELS_DIR / "baselines"
XGBOOST_MODELS_DIR = MODELS_DIR / "xgboost"
DEEPTDA_MODELS_DIR = MODELS_DIR / "deeptda"
GNN_MODELS_DIR = MODELS_DIR / "gnn"
BEST_MODEL_DIR = MODELS_DIR / "best"

# Experiment directories
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
LOGS_DIR = EXPERIMENTS_DIR / "logs"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Notebook directory
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# All directories to create
ALL_DIRS = [
    RAW_DATA_DIR, PROCESSED_DATA_DIR, METADATA_DIR, SPLITS_DIR,
    BASELINE_MODELS_DIR, XGBOOST_MODELS_DIR, DEEPTDA_MODELS_DIR,
    GNN_MODELS_DIR, BEST_MODEL_DIR,
    CONFIGS_DIR, RESULTS_DIR, LOGS_DIR,
    FIGURES_DIR, TABLES_DIR, REPORTS_DIR,
    NOTEBOOKS_DIR,
]


def ensure_dirs():
    """Create all project directories if they don't exist."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Reproducibility
# ==============================================================================

DEFAULT_SEEDS = [42, 123, 456]


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ==============================================================================
# Device Management
# ==============================================================================

def get_device():
    """
    Get the best available compute device.

    Priority: CUDA GPU > Intel XPU (Arc) > CPU

    Returns:
        torch.device: The selected device.
    """
    try:
        import torch

        # Check NVIDIA CUDA
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            logging.info(f"Using CUDA GPU: {gpu_name}")
            return device

        # Check Intel Arc GPU via IPEX
        try:
            import intel_extension_for_pytorch as ipex
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                device = torch.device("xpu")
                logging.info("Using Intel XPU (Arc GPU) via IPEX")
                return device
        except ImportError:
            pass

        # Fallback to CPU
        device = torch.device("cpu")
        logging.info("Using CPU")
        return device

    except ImportError:
        logging.warning("PyTorch not installed. Device detection unavailable.")
        return None


# ==============================================================================
# Logging
# ==============================================================================

def setup_logging(name: str = "pharmalens", level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with console and file output.

    Args:
        name: Logger name.
        level: Logging level.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler
        ensure_dirs()
        log_file = LOGS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ==============================================================================
# JSON/CSV Utilities
# ==============================================================================

def save_json(data: dict, path: Path):
    """Save a dictionary as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> dict:
    """Load a JSON file as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results_csv(results: list[dict], path: Path):
    """Save a list of result dictionaries as a CSV."""
    import pandas as pd
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)


# ==============================================================================
# Plotting Defaults
# ==============================================================================

def set_plot_style():
    """
    Configure matplotlib for publication-quality figures.
    Uses a clean style suitable for thesis/presentation.
    """
    import matplotlib.pyplot as plt
    import matplotlib

    plt.style.use("seaborn-v0_8-whitegrid")

    matplotlib.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # PharmaLens color palette
    PHARMALENS_COLORS = {
        "primary": "#6366F1",      # Indigo
        "secondary": "#8B5CF6",    # Violet
        "accent": "#06B6D4",       # Cyan
        "success": "#10B981",      # Emerald
        "warning": "#F59E0B",      # Amber
        "error": "#EF4444",        # Red
        "info": "#3B82F6",         # Blue
        "dark": "#1E1B4B",         # Dark indigo
        "light": "#F8FAFC",        # Slate 50
    }

    return PHARMALENS_COLORS


# Color palette for model comparison charts
MODEL_COLORS = {
    "MeanPredictor": "#94A3B8",
    "Ridge": "#64748B",
    "RandomForest": "#10B981",
    "XGBoost": "#F59E0B",
    "DeepDTA": "#6366F1",
    "GNN": "#EF4444",
    "GNN-GCN": "#EF4444",
    "GNN-GAT": "#EC4899",
    "GraphDTA_GCN": "#EF4444",
    "GraphDTA_GAT": "#EC4899",
    "Advanced": "#8B5CF6",
}
