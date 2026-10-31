"""
PharmaLens — GNN Dataset

PyTorch Geometric Dataset for drug molecular graphs + protein sequences.
Each sample pairs a molecular graph with an encoded protein sequence.
"""

import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset

logger = logging.getLogger("pharmalens.models.gnn_dataset")


class CachedGraphPairDataset(Dataset):
    """One cached graph/sequence per identity; never silently omit a pair."""

    def __init__(
        self, frame, protein_max_len=1000, graph_cache=None, protein_cache=None, auxiliary=None
    ):
        from src.features.mol_graph import smiles_to_graph
        from src.features.protein_features import encode_sequence

        self.graphs = {} if graph_cache is None else graph_cache
        self.proteins = {} if protein_cache is None else protein_cache
        self.pairs = list(frame[["smiles", "sequence"]].itertuples(index=False, name=None))
        self.scores = torch.tensor(frame.score.to_numpy(dtype=np.float32))
        self.auxiliary = auxiliary
        for smiles, sequence in self.pairs:
            if smiles not in self.graphs:
                graph = smiles_to_graph(smiles)
                if graph is None or not graph["num_atoms"]:
                    raise ValueError("Invalid molecular graph; prepare inputs before training")
                self.graphs[smiles] = {
                    "x": torch.tensor(graph["node_features"], dtype=torch.float),
                    "edge_index": torch.tensor(graph["edge_index"], dtype=torch.long),
                    "edge_attr": torch.tensor(graph["edge_features"], dtype=torch.float),
                }
            key = (sequence, protein_max_len)
            if key not in self.proteins:
                self.proteins[key] = torch.tensor(
                    encode_sequence(sequence, protein_max_len), dtype=torch.long
                ).unsqueeze(0)
        self.protein_max_len = protein_max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        smiles, sequence = self.pairs[index]
        data = Data(
            **self.graphs[smiles],
            protein_input=self.proteins[(sequence, self.protein_max_len)],
            y=self.scores[index].reshape(1),
        )
        if self.auxiliary is not None:
            data.auxiliary = torch.as_tensor(self.auxiliary[index], dtype=torch.float).unsqueeze(0)
        return data


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
        super().__init__(".", transform)

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
