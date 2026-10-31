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


def masked_cnn_pool(embedded, token_ids, convolutions):
    """Prevent convolution biases in padding positions from winning pooling."""
    mask = token_ids.ne(0).unsqueeze(1)
    if not mask.any(dim=-1).all():
        raise ValueError("Every sequence must contain at least one encoded token")
    x = embedded.permute(0, 2, 1)
    for conv in convolutions:
        x = F.relu(conv(x))
        # Existing even-kernel layers add one right-hand output position.
        if x.shape[-1] > mask.shape[-1]:
            mask = F.pad(mask, (0, x.shape[-1] - mask.shape[-1]), value=False)
        x = x.masked_fill(~mask, 0)
    return x.masked_fill(~mask, float("-inf")).amax(dim=-1)


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
        mask_padding: bool = False,
        deduplicate_proteins: bool = False,
        auxiliary_dim: int = 0,
        auxiliary_interactions: bool = False,
        protein_embedding_cache=None,
        use_protein_cnn: bool = True,
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
        self.mask_padding = mask_padding
        self.deduplicate_proteins = deduplicate_proteins
        self.auxiliary_encoder = None
        self.protein_embedding_cache = protein_embedding_cache
        self.use_protein_cnn = use_protein_cnn
        if not use_protein_cnn and auxiliary_dim <= 1034:
            raise ValueError("A protein feature branch is required when protein CNN is disabled")
        if auxiliary_dim:
            from src.models.auxiliary import AuxiliaryEncoder

            self.auxiliary_encoder = AuxiliaryEncoder(
                auxiliary_dim, interactions=auxiliary_interactions
            )

        # Drug encoder
        self.drug_embedding = nn.Embedding(drug_vocab_size + 1, drug_embed_dim, padding_idx=0)
        self.drug_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=drug_embed_dim if i == 0 else num_filters,
                    out_channels=num_filters,
                    kernel_size=drug_filter_sizes[i],
                    padding=drug_filter_sizes[i] // 2,
                )
                for i in range(len(drug_filter_sizes))
            ]
        )

        # Protein encoder
        self.protein_embedding = nn.Embedding(
            protein_vocab_size + 1, protein_embed_dim, padding_idx=0
        )
        self.protein_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=protein_embed_dim if i == 0 else num_filters,
                    out_channels=num_filters,
                    kernel_size=protein_filter_sizes[i],
                    padding=protein_filter_sizes[i] // 2,
                )
                for i in range(len(protein_filter_sizes))
            ]
        )

        # Fully connected layers for combined representation
        fc_input_dim = num_filters * (2 if use_protein_cnn else 1) + (128 if auxiliary_dim else 0)
        if not use_protein_cnn:
            self.protein_embedding = None
            self.protein_convs = nn.ModuleList()
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
            "mask_padding": mask_padding,
            "deduplicate_proteins": deduplicate_proteins,
            "auxiliary_dim": auxiliary_dim,
            "auxiliary_interactions": auxiliary_interactions,
            "protein_embedding_cache": protein_embedding_cache,
            "use_protein_cnn": use_protein_cnn,
        }

    def encode_drug(self, drug_input):
        """Encode drug SMILES through CNN."""
        # drug_input: (batch, drug_max_len) — integer indices
        x = self.drug_embedding(drug_input)  # (batch, seq_len, embed_dim)
        if self.mask_padding:
            return masked_cnn_pool(x, drug_input, self.drug_convs)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len) for Conv1d

        for conv in self.drug_convs:
            x = F.relu(conv(x))

        # Global max pooling
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # (batch, num_filters)
        return x

    def encode_protein(self, protein_input):
        """Encode protein sequence through CNN."""
        if self.deduplicate_proteins and protein_input.shape[0] > 1:
            unique, inverse = torch.unique(protein_input, dim=0, return_inverse=True)
            return self._encode_protein(unique)[inverse]
        return self._encode_protein(protein_input)

    def _encode_protein(self, protein_input):
        # protein_input: (batch, protein_max_len) — integer indices
        x = self.protein_embedding(protein_input)
        if self.mask_padding:
            return masked_cnn_pool(x, protein_input, self.protein_convs)
        x = x.permute(0, 2, 1)

        for conv in self.protein_convs:
            x = F.relu(conv(x))

        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        return x

    def forward(self, drug_input, protein_input, auxiliary=None):
        """
        Forward pass.

        Args:
            drug_input: (batch, drug_max_len) integer tensor
            protein_input: (batch, protein_max_len) integer tensor

        Returns:
            (batch, 1) predicted interaction scores
        """
        drug_repr = self.encode_drug(drug_input)
        protein_repr = self.encode_protein(protein_input) if self.use_protein_cnn else None

        # Concatenate representations
        combined = (
            torch.cat([drug_repr, protein_repr], dim=1) if protein_repr is not None else drug_repr
        )
        if self.auxiliary_encoder is not None:
            combined = torch.cat([combined, self.auxiliary_encoder(auxiliary)], dim=1)

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
