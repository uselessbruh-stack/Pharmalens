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
from torch_geometric.nn import GCNConv, GATConv, global_max_pool, global_mean_pool


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
        gnn_type: 'gcn' or 'gat'.
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

        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer: node_feature_dim → hidden_dim
        if gnn_type == "gcn":
            self.convs.append(GCNConv(node_feature_dim, hidden_dim))
        elif gnn_type == "gat":
            self.convs.append(GATConv(node_feature_dim, hidden_dim, heads=4, concat=False))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Intermediate layers: hidden_dim → hidden_dim
        for _ in range(num_layers - 2):
            if gnn_type == "gcn":
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == "gat":
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Last layer: hidden_dim → output_dim
        if gnn_type == "gcn":
            self.convs.append(GCNConv(hidden_dim, output_dim))
        elif gnn_type == "gat":
            self.convs.append(GATConv(hidden_dim, output_dim, heads=4, concat=False))
        self.batch_norms.append(nn.BatchNorm1d(output_dim))

        # Final projection
        self.fc_out = nn.Linear(output_dim, output_dim)

    def forward(self, x, edge_index, batch):
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
            x_residual = x if i > 0 and x.shape[1] == self.convs[i].out_channels else None

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

    def __init__(self, vocab_size=21, embed_dim=128, num_filters=32,
                 filter_sizes=None, max_len=1000):
        super().__init__()

        if filter_sizes is None:
            filter_sizes = [4, 8, 12]

        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim if i == 0 else num_filters,
                out_channels=num_filters,
                kernel_size=filter_sizes[i],
                padding=filter_sizes[i] // 2,
            )
            for i in range(len(filter_sizes))
        ])

    def forward(self, x):
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
    ):
        super().__init__()

        if fc_dims is None:
            fc_dims = [512, 256]

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
        )

        # FC fusion layers
        fc_input_dim = gnn_output_dim + protein_num_filters
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
            "protein_embed_dim": protein_embed_dim,
            "protein_num_filters": protein_num_filters,
            "fc_dims": fc_dims,
            "dropout": dropout,
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
        drug_repr = self.drug_encoder(data.x, data.edge_index, data.batch)

        # Protein encoding via CNN
        protein_repr = self.protein_encoder(data.protein_input)

        # Fusion
        combined = torch.cat([drug_repr, protein_repr], dim=1)
        for fc, dropout in zip(self.fc_layers, self.fc_dropouts):
            combined = F.relu(fc(combined))
            combined = dropout(combined)

        out = self.output(combined)
        return out.squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
