"""
PharmaLens — GNN Dataset

PyTorch Geometric Dataset for drug molecular graphs + protein sequences.
Each sample pairs a molecular graph with an encoded protein sequence.
"""

import logging
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

logger = logging.getLogger("pharmalens.models.gnn_dataset")


class DTIGraphDataset(InMemoryDataset):
    """
    PyTorch Geometric Dataset for DTI prediction using molecular graphs.

    Each Data object contains:
        - x: Node (atom) features
        - edge_index: Bond connectivity
        - edge_attr: Bond features
        - protein_input: Encoded protein sequence (integer tensor)
        - y: Interaction score
    """

    def __init__(
        self,
        drug_ids,
        target_ids,
        scores,
        smiles_list_dict: dict,
        target_sequence_dict: dict,
        protein_max_len: int = 1000,
        transform=None,
    ):
        """
        Args:
            drug_ids: Drug ID list.
            target_ids: Target ID list.
            scores: Score list.
            smiles_list_dict: {drug_id: SMILES string}
            target_sequence_dict: {target_id: encoded_sequence np.ndarray}
            protein_max_len: Max protein sequence length.
        """
        super().__init__('.', transform)

        from src.features.mol_graph import smiles_to_graph
        from src.features.protein_features import encode_sequence

        data_list = []
        n_failed = 0

        for i in range(len(drug_ids)):
            drug_id = drug_ids[i]
            target_id = target_ids[i]

            if drug_id not in smiles_list_dict or target_id not in target_sequence_dict:
                n_failed += 1
                continue

            smiles = smiles_list_dict[drug_id]

            # Convert SMILES to graph
            graph = smiles_to_graph(smiles)
            if graph is None:
                n_failed += 1
                continue

            # Get protein encoding
            if isinstance(target_sequence_dict[target_id], np.ndarray):
                protein_enc = target_sequence_dict[target_id]
            else:
                protein_enc = encode_sequence(target_sequence_dict[target_id], protein_max_len)

            # Create PyG Data object
            data = Data(
                x=torch.tensor(graph["node_features"], dtype=torch.float),
                edge_index=torch.tensor(graph["edge_index"], dtype=torch.long),
                edge_attr=torch.tensor(graph["edge_features"], dtype=torch.float),
                y=torch.tensor([float(scores[i])], dtype=torch.float),
                protein_input=torch.tensor(protein_enc, dtype=torch.long).unsqueeze(0),
            )

            data_list.append(data)

        if n_failed > 0:
            logger.warning(f"Failed to create {n_failed} graph samples")

        self.data, self.slices = self.collate(data_list)
        self._data_list = data_list

    def len(self):
        return len(self._data_list)

    def get(self, idx):
        return self._data_list[idx]


def collate_dti_graphs(batch):
    """
    Custom collate function for DTI graph batches.

    Handles batching of both molecular graphs (variable size)
    and protein sequences (fixed size) together.
    """
    from torch_geometric.data import Batch

    # Separate protein inputs and create PyG batch
    pyg_batch = Batch.from_data_list(batch)

    # Stack protein inputs
    protein_inputs = torch.cat([data.protein_input for data in batch], dim=0)
    pyg_batch.protein_input = protein_inputs

    return pyg_batch
