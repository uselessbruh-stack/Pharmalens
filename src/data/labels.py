"""Explicit affinity contracts shared by preparation, inference and exports."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LabelContract:
    endpoint: str
    label_name: str
    unit: str
    higher_is_stronger: bool = True

    def as_dict(self):
        return asdict(self)


CONTRACTS = {
    "KIBA": LabelContract("KIBA", "Transformed KIBA score", "KIBA score"),
    "Ki": LabelContract("Ki", "pKi", "negative log10 molar"),
    "Kd": LabelContract("Kd", "pKd", "negative log10 molar"),
    "IC50": LabelContract("IC50", "pIC50", "negative log10 molar"),
}


def concentration_to_p(value, unit="nM", relation="="):
    """Convert exact positive concentration; never coerce censored boundaries."""
    if relation != "=":
        raise ValueError("Censored observations require a separate interval-aware task")
    factors = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "µM": 1e-6, "μM": 1e-6, "nM": 1e-9, "pM": 1e-12}
    if unit not in factors:
        raise ValueError(f"Unsupported concentration unit: {unit}")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Concentration must be finite and positive")
    return -math.log10(value) - math.log10(factors[unit])
