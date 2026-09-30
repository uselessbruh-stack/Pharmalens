"""
PharmaLens — DeepDTA Model

CNN-based Drug-Target Binding Affinity prediction.
Architecture from: Öztürk et al., 2018 — "DeepDTA: deep drug-target binding affinity prediction"

Drug SMILES → Embedding → Conv1D ×3 → GlobalMaxPool → drug_repr
Protein Seq → Embedding → Conv1D ×3 → GlobalMaxPool → prot_repr
[drug_repr ⊕ prot_repr] → FC(1024) → FC(512) → FC(1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepDTA(nn.Module):
    """
    DeepDTA: CNN-based model for drug-target binding affinity prediction.

    Processes drug SMILES and protein sequences as 1D character-level inputs
    through separate CNN encoders, then fuses representations through FC layers.

    Args:
        drug_vocab_size: Number of unique SMILES characters + 1 (for padding).
        protein_vocab_size: Number of amino acids + 1 (for padding).
        drug_embed_dim: Drug embedding dimension.
        protein_embed_dim: Protein embedding dimension.
        num_filters: Number of filters per CNN layer.
        drug_filter_sizes: Kernel sizes for drug CNN layers.
        protein_filter_sizes: Kernel sizes for protein CNN layers.
        fc_dims: Fully connected layer dimensions.
        dropout: Dropout rate.
        drug_max_len: Maximum SMILES length.
        protein_max_len: Maximum protein sequence length.
    """

    def __init__(
        self,
        drug_vocab_size: int = 63,
        protein_vocab_size: int = 21,
        drug_embed_dim: int = 128,
        protein_embed_dim: int = 128,
        num_filters: int = 32,
        drug_filter_sizes: list = None,
        protein_filter_sizes: list = None,
        fc_dims: list = None,
        dropout: float = 0.2,
        drug_max_len: int = 100,
        protein_max_len: int = 1000,
    ):
        super().__init__()

        if drug_filter_sizes is None:
            drug_filter_sizes = [4, 6, 8]
        if protein_filter_sizes is None:
            protein_filter_sizes = [4, 8, 12]
        if fc_dims is None:
            fc_dims = [1024, 512]

        self.drug_max_len = drug_max_len
        self.protein_max_len = protein_max_len

        # Drug encoder
        self.drug_embedding = nn.Embedding(drug_vocab_size + 1, drug_embed_dim, padding_idx=0)
        self.drug_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=drug_embed_dim if i == 0 else num_filters,
                out_channels=num_filters,
                kernel_size=drug_filter_sizes[i],
                padding=drug_filter_sizes[i] // 2,
            )
            for i in range(len(drug_filter_sizes))
        ])

        # Protein encoder
        self.protein_embedding = nn.Embedding(protein_vocab_size + 1, protein_embed_dim, padding_idx=0)
        self.protein_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=protein_embed_dim if i == 0 else num_filters,
                out_channels=num_filters,
                kernel_size=protein_filter_sizes[i],
                padding=protein_filter_sizes[i] // 2,
            )
            for i in range(len(protein_filter_sizes))
        ])

        # Fully connected layers for combined representation
        fc_input_dim = num_filters * 2  # drug + protein pooled outputs
        self.fc_layers = nn.ModuleList()
        self.fc_dropouts = nn.ModuleList()

        prev_dim = fc_input_dim
        for dim in fc_dims:
            self.fc_layers.append(nn.Linear(prev_dim, dim))
            self.fc_dropouts.append(nn.Dropout(dropout))
            prev_dim = dim

        # Output layer
        self.output = nn.Linear(prev_dim, 1)

        # Store config for serialization
        self.config = {
            "drug_vocab_size": drug_vocab_size,
            "protein_vocab_size": protein_vocab_size,
            "drug_embed_dim": drug_embed_dim,
            "protein_embed_dim": protein_embed_dim,
            "num_filters": num_filters,
            "drug_filter_sizes": drug_filter_sizes,
            "protein_filter_sizes": protein_filter_sizes,
            "fc_dims": fc_dims,
            "dropout": dropout,
            "drug_max_len": drug_max_len,
            "protein_max_len": protein_max_len,
        }

    def encode_drug(self, drug_input):
        """Encode drug SMILES through CNN."""
        # drug_input: (batch, drug_max_len) — integer indices
        x = self.drug_embedding(drug_input)  # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len) for Conv1d

        for conv in self.drug_convs:
            x = F.relu(conv(x))

        # Global max pooling
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (batch, num_filters)
        return x

    def encode_protein(self, protein_input):
        """Encode protein sequence through CNN."""
        # protein_input: (batch, protein_max_len) — integer indices
        x = self.protein_embedding(protein_input)
        x = x.permute(0, 2, 1)

        for conv in self.protein_convs:
            x = F.relu(conv(x))

        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        return x

    def forward(self, drug_input, protein_input):
        """
        Forward pass.

        Args:
            drug_input: (batch, drug_max_len) integer tensor
            protein_input: (batch, protein_max_len) integer tensor

        Returns:
            (batch, 1) predicted interaction scores
        """
        drug_repr = self.encode_drug(drug_input)
        protein_repr = self.encode_protein(protein_input)

        # Concatenate representations
        combined = torch.cat([drug_repr, protein_repr], dim=1)

        # FC layers
        for fc, dropout in zip(self.fc_layers, self.fc_dropouts):
            combined = F.relu(fc(combined))
            combined = dropout(combined)

        # Output
        out = self.output(combined)
        return out.squeeze(-1)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
