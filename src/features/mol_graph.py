"""
PharmaLens — Molecular Graph Construction

Converts drug SMILES strings into graph representations for GNN models.
Atoms become nodes, chemical bonds become edges.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("pharmalens.features.mol_graph")


# ==============================================================================
# Atom (Node) Features
# ==============================================================================

# Allowed atom types for one-hot encoding
ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si', 'B', 'Se', 'other']
HYBRIDIZATION_TYPES = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'other']


def atom_features(atom) -> list:
    """
    Extract features for a single atom.

    Features (total ~39 dimensions):
        - Atom type one-hot (13 types)
        - Degree one-hot (0-5, 6 values)
        - Formal charge (1)
        - Num H (1)
        - Hybridization one-hot (6 types)
        - Is aromatic (1)
        - Is in ring (1)
        - Atomic number / 100 (normalized, 1)

    Args:
        atom: RDKit Atom object.

    Returns:
        list of float: Feature vector.
    """
    from rdkit.Chem import rdchem

    # Atom type one-hot
    symbol = atom.GetSymbol()
    atom_type = [0] * len(ATOM_TYPES)
    if symbol in ATOM_TYPES:
        atom_type[ATOM_TYPES.index(symbol)] = 1
    else:
        atom_type[-1] = 1  # 'other'

    # Degree one-hot (0-5)
    degree = atom.GetDegree()
    degree_onehot = [0] * 6
    degree_onehot[min(degree, 5)] = 1

    # Hybridization one-hot
    hybridization = str(atom.GetHybridization())
    hyb_onehot = [0] * len(HYBRIDIZATION_TYPES)
    if hybridization in HYBRIDIZATION_TYPES:
        hyb_onehot[HYBRIDIZATION_TYPES.index(hybridization)] = 1
    else:
        hyb_onehot[-1] = 1

    features = (
        atom_type
        + degree_onehot
        + [atom.GetFormalCharge()]
        + [atom.GetTotalNumHs()]
        + hyb_onehot
        + [1 if atom.GetIsAromatic() else 0]
        + [1 if atom.IsInRing() else 0]
        + [atom.GetAtomicNum() / 100.0]
    )

    return features


def get_atom_feature_dim() -> int:
    """Get the dimensionality of atom feature vectors."""
    return len(ATOM_TYPES) + 6 + 1 + 1 + len(HYBRIDIZATION_TYPES) + 1 + 1 + 1  # = 30


# ==============================================================================
# Bond (Edge) Features
# ==============================================================================

def bond_features(bond) -> list:
    """
    Extract features for a single bond.

    Features (4 dimensions):
        - Bond type one-hot (single, double, triple, aromatic)
        - Is conjugated (1)
        - Is in ring (1)

    Args:
        bond: RDKit Bond object.

    Returns:
        list of float: Feature vector.
    """
    from rdkit.Chem import rdchem

    bt = bond.GetBondType()
    bond_type = [
        bt == rdchem.BondType.SINGLE,
        bt == rdchem.BondType.DOUBLE,
        bt == rdchem.BondType.TRIPLE,
        bt == rdchem.BondType.AROMATIC,
    ]

    features = (
        [int(x) for x in bond_type]
        + [int(bond.GetIsConjugated())]
        + [int(bond.IsInRing())]
    )

    return features


# ==============================================================================
# SMILES → Graph Conversion
# ==============================================================================

def smiles_to_graph(smiles: str) -> Optional[dict]:
    """
    Convert a SMILES string to a molecular graph representation.

    Returns a dictionary with:
        - node_features: np.ndarray of shape (num_atoms, node_feat_dim)
        - edge_index: np.ndarray of shape (2, num_edges) — COO format
        - edge_features: np.ndarray of shape (num_edges, edge_feat_dim)
        - num_atoms: int
        - num_bonds: int

    The edge_index is undirected: each bond appears twice (i→j and j→i).

    Args:
        smiles: SMILES string.

    Returns:
        dict with graph data, or None if invalid.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append(atom_features(atom))
    node_feats = np.array(node_feats, dtype=np.float32)

    # Edge index and features (undirected: add both directions)
    edge_indices = []
    edge_feats = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)

        # Add both directions
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_feats.append(bf)
        edge_feats.append(bf)

    if len(edge_indices) == 0:
        # Single atom molecule (no bonds)
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_feats_array = np.zeros((0, 6), dtype=np.float32)
    else:
        edge_index = np.array(edge_indices, dtype=np.int64).T  # Shape: (2, num_edges)
        edge_feats_array = np.array(edge_feats, dtype=np.float32)

    return {
        "node_features": node_feats,
        "edge_index": edge_index,
        "edge_features": edge_feats_array,
        "num_atoms": mol.GetNumAtoms(),
        "num_bonds": mol.GetNumBonds(),
        "smiles": smiles,
    }


def smiles_to_pyg_data(smiles: str, y: Optional[float] = None):
    """
    Convert a SMILES string to a PyTorch Geometric Data object.

    Args:
        smiles: SMILES string.
        y: Optional target value (binding score).

    Returns:
        torch_geometric.data.Data object, or None if invalid.
    """
    import torch
    from torch_geometric.data import Data

    graph = smiles_to_graph(smiles)
    if graph is None:
        return None

    data = Data(
        x=torch.tensor(graph["node_features"], dtype=torch.float),
        edge_index=torch.tensor(graph["edge_index"], dtype=torch.long),
        edge_attr=torch.tensor(graph["edge_features"], dtype=torch.float),
    )

    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)

    data.smiles = smiles

    return data


def visualize_molecular_graph(smiles: str, save_path: Optional[str] = None):
    """
    Visualize a molecular graph using RDKit's 2D drawing.

    Args:
        smiles: SMILES string.
        save_path: Optional path to save the image.

    Returns:
        PIL Image or None.
    """
    from rdkit import Chem
    from rdkit.Chem import Draw

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning(f"Cannot visualize invalid SMILES: {smiles}")
        return None

    img = Draw.MolToImage(mol, size=(400, 400))

    if save_path:
        img.save(save_path)
        logger.info(f"Molecular structure saved to {save_path}")

    return img
