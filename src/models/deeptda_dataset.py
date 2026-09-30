"""
PharmaLens — DeepDTA Dataset

PyTorch Dataset for DeepDTA-style training using encoded SMILES + sequences.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class DTIDataset(Dataset):
    """
    PyTorch Dataset for Drug-Target Interaction prediction.

    Used by DeepDTA and similar sequence-based models.
    Each sample contains:
        - Encoded drug SMILES (integer sequence)
        - Encoded protein sequence (integer sequence)
        - Interaction score (regression target)

    Args:
        drug_ids: List of drug identifiers.
        target_ids: List of target identifiers.
        scores: List/array of interaction scores.
        drug_encoding_dict: {drug_id: np.ndarray} encoded SMILES vectors.
        target_encoding_dict: {target_id: np.ndarray} encoded sequence vectors.
        drug_max_len: Maximum drug sequence length.
        target_max_len: Maximum protein sequence length.
    """

    def __init__(
        self,
        drug_ids,
        target_ids,
        scores,
        drug_encoding_dict: dict,
        target_encoding_dict: dict,
        drug_max_len: int = 100,
        target_max_len: int = 1000,
    ):
        self.drug_ids = list(drug_ids)
        self.target_ids = list(target_ids)
        self.scores = np.array(scores, dtype=np.float32)
        self.drug_encoding_dict = drug_encoding_dict
        self.target_encoding_dict = target_encoding_dict
        self.drug_max_len = drug_max_len
        self.target_max_len = target_max_len

        # Filter to valid entries only
        self.valid_indices = []
        for i in range(len(self.drug_ids)):
            if self.drug_ids[i] in self.drug_encoding_dict and \
               self.target_ids[i] in self.target_encoding_dict:
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]

        drug_enc = self.drug_encoding_dict[self.drug_ids[real_idx]]
        target_enc = self.target_encoding_dict[self.target_ids[real_idx]]
        score = self.scores[real_idx]

        # Ensure correct length
        drug_tensor = torch.zeros(self.drug_max_len, dtype=torch.long)
        target_tensor = torch.zeros(self.target_max_len, dtype=torch.long)

        drug_len = min(len(drug_enc), self.drug_max_len)
        target_len = min(len(target_enc), self.target_max_len)

        drug_tensor[:drug_len] = torch.tensor(drug_enc[:drug_len], dtype=torch.long)
        target_tensor[:target_len] = torch.tensor(target_enc[:target_len], dtype=torch.long)

        return drug_tensor, target_tensor, torch.tensor(score, dtype=torch.float)


def create_encoding_dicts(df, smiles_col='smiles', seq_col='sequence',
                          drug_id_col='drug_id', target_id_col='target_id',
                          drug_max_len=100, target_max_len=1000):
    """
    Create encoding dictionaries from a DataFrame.

    Encodes all unique SMILES and sequences into integer vectors.

    Returns:
        tuple: (drug_encoding_dict, target_encoding_dict)
    """
    from src.features.drug_features import encode_smiles
    from src.features.protein_features import encode_sequence

    # Encode unique drugs
    drug_enc = {}
    for _, row in df.drop_duplicates(drug_id_col).iterrows():
        drug_enc[row[drug_id_col]] = encode_smiles(row[smiles_col], drug_max_len)

    # Encode unique targets
    target_enc = {}
    for _, row in df.drop_duplicates(target_id_col).iterrows():
        target_enc[row[target_id_col]] = encode_sequence(row[seq_col], target_max_len)

    return drug_enc, target_enc
