"""Optional feature-matched branch for hybrid affinity networks."""

import torch
from torch import nn


class AuxiliaryEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=128, interactions=False):
        super().__init__()
        self.register_buffer("mean", torch.zeros(input_dim))
        self.register_buffer("scale", torch.ones(input_dim))
        self.interactions = interactions
        if interactions:
            if input_dim <= 1034:
                raise ValueError("Interaction branch requires drug and protein features")
            self.drug = nn.Sequential(
                nn.Linear(1034, 256),
                nn.ReLU(),
                nn.Linear(256, output_dim),
                nn.LayerNorm(output_dim),
            )
            self.protein = nn.Sequential(
                nn.Linear(input_dim - 1034, 256),
                nn.ReLU(),
                nn.Linear(256, output_dim),
                nn.LayerNorm(output_dim),
            )
            self.network = nn.Sequential(nn.Linear(output_dim * 4, output_dim), nn.ReLU())
            return
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, output_dim), nn.ReLU()
        )

    def fit_normalization(self, training_features):
        values = torch.as_tensor(training_features, dtype=torch.float32)
        if (
            values.ndim != 2
            or values.shape[1] != self.mean.numel()
            or not torch.isfinite(values).all()
        ):
            raise ValueError("Invalid training features for auxiliary normalization")
        with torch.no_grad():
            self.mean.copy_(values.mean(0))
            scale = values.std(0, unbiased=False)
            self.scale.copy_(torch.where(scale > 1e-6, scale, torch.ones_like(scale)))

    def forward(self, features):
        if features is None or features.ndim != 2 or features.shape[1] != self.mean.numel():
            raise ValueError("Hybrid model requires matching auxiliary features")
        normalized = ((features - self.mean) / self.scale).clamp(-10, 10)
        if self.interactions:
            drug, protein = self.drug(normalized[:, :1034]), self.protein(normalized[:, 1034:])
            return self.network(
                torch.cat([drug, protein, drug * protein, (drug - protein).abs()], dim=1)
            )
        return self.network(normalized)
