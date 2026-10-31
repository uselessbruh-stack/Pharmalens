"""
PharmaLens — Graph Neural Network for DTI Prediction

Uses molecular graphs (from SMILES) + protein sequence encoding
to predict drug-target binding affinity.

Architecture:
    Drug SMILES → Molecular Graph → GCN/GAT layers → Graph Pooling → drug_repr
    Protein Seq → Embedding → Conv1D ×3 → GlobalMaxPool → prot_repr
    [drug_repr ⊕ prot_repr] → FC → score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINEConv, global_max_pool, global_mean_pool

from src.models.deeptda import masked_cnn_pool


class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder for molecular graphs.

    Processes atom features through multiple GNN layers with
    residual connections, then pools into a fixed-size representation.

    Args:
        node_feature_dim: Input node feature dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output representation dimension.
        num_layers: Number of GNN message-passing layers.
        gnn_type: 'gcn', 'gat' or bond-aware 'gine'.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        node_feature_dim: int = 30,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
        gnn_type: str = "gcn",
        dropout: float = 0.2,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        if gnn_type not in {"gcn", "gat", "gine"}:
            raise ValueError("gnn_type must be gcn, gat or gine")
        if num_layers < 2:
            raise ValueError("At least two message-passing layers are required")

        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer: node_feature_dim → hidden_dim
        if gnn_type == "gcn":
            self.convs.append(GCNConv(node_feature_dim, hidden_dim))
        elif gnn_type == "gat":
            self.convs.append(GATConv(node_feature_dim, hidden_dim, heads=4, concat=False))
        elif gnn_type == "gine":
            self.convs.append(
                GINEConv(
                    nn.Sequential(
                        nn.Linear(node_feature_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    ),
                    edge_dim=6,
                )
            )
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Intermediate layers: hidden_dim → hidden_dim
        for _ in range(num_layers - 2):
            if gnn_type == "gcn":
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == "gat":
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            elif gnn_type == "gine":
                self.convs.append(
                    GINEConv(
                        nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                        ),
                        edge_dim=6,
                    )
                )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Last layer: hidden_dim → output_dim
        if gnn_type == "gcn":
            self.convs.append(GCNConv(hidden_dim, output_dim))
        elif gnn_type == "gat":
            self.convs.append(GATConv(hidden_dim, output_dim, heads=4, concat=False))
        elif gnn_type == "gine":
            self.convs.append(
                GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, output_dim),
                        nn.ReLU(),
                        nn.Linear(output_dim, output_dim),
                    ),
                    edge_dim=6,
                )
            )
        self.batch_norms.append(nn.BatchNorm1d(output_dim))
        if gnn_type == "gine":
            self.batch_norms = nn.ModuleList(
                [nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)]
                + [nn.LayerNorm(output_dim)]
            )

        # Final projection
        self.fc_out = nn.Linear(output_dim, output_dim)

    def forward(self, x, edge_index, batch, edge_attr=None):
        """
        Forward pass through GNN.

        Args:
            x: Node features (num_nodes, node_feat_dim)
            edge_index: Graph connectivity (2, num_edges)
            batch: Batch assignment vector

        Returns:
            graph-level representation (batch_size, output_dim)
        """
        for i in range(self.num_layers):
            x_residual = x if i > 0 else None
            if self.gnn_type == "gine":
                if edge_attr is None or edge_attr.shape != (edge_index.shape[1], 6):
                    raise ValueError("GINE requires six bond features per directed edge")
                x = self.convs[i](x, edge_index, edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection (if dimensions match)
            if x_residual is not None and x.shape == x_residual.shape:
                x = x + x_residual

        # Global pooling: combine max and mean
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x = x_max + x_mean  # Element-wise sum

        x = self.fc_out(x)
        return x


class ProteinCNN(nn.Module):
    """CNN encoder for protein sequences (same as DeepDTA protein branch)."""

    def __init__(
        self,
        vocab_size=21,
        embed_dim=128,
        num_filters=32,
        filter_sizes=None,
        max_len=1000,
        mask_padding=False,
        deduplicate=False,
    ):
        super().__init__()
        self.mask_padding = mask_padding
        self.deduplicate = deduplicate

        if filter_sizes is None:
            filter_sizes = [4, 8, 12]

        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim if i == 0 else num_filters,
                    out_channels=num_filters,
                    kernel_size=filter_sizes[i],
                    padding=filter_sizes[i] // 2,
                )
                for i in range(len(filter_sizes))
            ]
        )

    def forward(self, x):
        if self.deduplicate and x.shape[0] > 1:
            unique, inverse = torch.unique(x, dim=0, return_inverse=True)
            return self._encode(unique)[inverse]
        return self._encode(x)

    def _encode(self, x):
        if self.mask_padding:
            return masked_cnn_pool(self.embedding(x), x, self.convs)
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        for conv in self.convs:
            x = F.relu(conv(x))
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        return x


class GraphDTA(nn.Module):
    """
    GraphDTA: GNN-based drug encoder + CNN-based protein encoder.

    Architecture:
        Drug → GNN(molecular graph) → drug_repr
        Protein → CNN(sequence) → prot_repr
        [drug_repr ⊕ prot_repr] → FC → score

    Args:
        node_feature_dim: Input atom feature dimension.
        gnn_hidden_dim: GNN hidden dimension.
        gnn_output_dim: GNN output dimension.
        gnn_layers: Number of GNN layers.
        gnn_type: 'gcn' or 'gat'.
        protein_vocab_size: Protein vocabulary size.
        protein_embed_dim: Protein embedding dimension.
        protein_num_filters: CNN filter count.
        protein_max_len: Max protein sequence length.
        fc_dims: FC layer dimensions.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        node_feature_dim: int = 30,
        gnn_hidden_dim: int = 128,
        gnn_output_dim: int = 128,
        gnn_layers: int = 3,
        gnn_type: str = "gcn",
        protein_vocab_size: int = 21,
        protein_embed_dim: int = 128,
        protein_num_filters: int = 32,
        protein_max_len: int = 1000,
        fc_dims: list = None,
        dropout: float = 0.2,
        mask_padding: bool = False,
        deduplicate_proteins: bool = False,
        auxiliary_dim: int = 0,
        auxiliary_interactions: bool = False,
        protein_embedding_cache=None,
        use_protein_cnn: bool = True,
    ):
        super().__init__()

        if fc_dims is None:
            fc_dims = [512, 256]
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

        # Drug encoder (GNN)
        self.drug_encoder = GNNEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            num_layers=gnn_layers,
            gnn_type=gnn_type,
            dropout=dropout,
        )

        # Protein encoder (CNN)
        self.protein_encoder = ProteinCNN(
            vocab_size=protein_vocab_size,
            embed_dim=protein_embed_dim,
            num_filters=protein_num_filters,
            max_len=protein_max_len,
            mask_padding=mask_padding,
            deduplicate=deduplicate_proteins,
        )

        # FC fusion layers
        fc_input_dim = (
            gnn_output_dim
            + (protein_num_filters if use_protein_cnn else 0)
            + (128 if auxiliary_dim else 0)
        )
        if not use_protein_cnn:
            self.protein_encoder = None
        self.fc_layers = nn.ModuleList()
        self.fc_dropouts = nn.ModuleList()

        prev_dim = fc_input_dim
        for dim in fc_dims:
            self.fc_layers.append(nn.Linear(prev_dim, dim))
            self.fc_dropouts.append(nn.Dropout(dropout))
            prev_dim = dim

        self.output = nn.Linear(prev_dim, 1)

        # Store config
        self.config = {
            "node_feature_dim": node_feature_dim,
            "gnn_hidden_dim": gnn_hidden_dim,
            "gnn_output_dim": gnn_output_dim,
            "gnn_layers": gnn_layers,
            "gnn_type": gnn_type,
            "protein_vocab_size": protein_vocab_size,
            "protein_max_len": protein_max_len,
            "protein_embed_dim": protein_embed_dim,
            "protein_num_filters": protein_num_filters,
            "fc_dims": fc_dims,
            "dropout": dropout,
            "mask_padding": mask_padding,
            "deduplicate_proteins": deduplicate_proteins,
            "auxiliary_dim": auxiliary_dim,
            "auxiliary_interactions": auxiliary_interactions,
            "protein_embedding_cache": protein_embedding_cache,
            "use_protein_cnn": use_protein_cnn,
        }

    def forward(self, data):
        """
        Forward pass.

        Args:
            data: PyG Batch object with:
                - x: Node features
                - edge_index: Graph connectivity
                - batch: Batch assignment
                - protein_input: Encoded protein sequences

        Returns:
            (batch_size,) predicted scores
        """
        # Drug encoding via GNN
        drug_repr = self.drug_encoder(
            data.x, data.edge_index, data.batch, getattr(data, "edge_attr", None)
        )

        # Protein encoding via CNN
        protein_repr = self.protein_encoder(data.protein_input) if self.use_protein_cnn else None

        # Fusion
        combined = (
            torch.cat([drug_repr, protein_repr], dim=1) if protein_repr is not None else drug_repr
        )
        if self.auxiliary_encoder is not None:
            combined = torch.cat([combined, self.auxiliary_encoder(data.auxiliary)], dim=1)
        for fc, dropout in zip(self.fc_layers, self.fc_dropouts):
            combined = F.relu(fc(combined))
            combined = dropout(combined)

        out = self.output(combined)
        return out.squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
