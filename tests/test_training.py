import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.xgboost_model import XGBoostDTI
from src.training.trainer import Trainer


class TinyPair(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, drug, protein):
        return drug.reshape(-1) * self.weight


def test_validation_weights_short_last_batch_by_sample_count():
    model = TinyPair()
    trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01))
    inputs = torch.tensor([0.0, 0.0, 3.0])
    loader = DataLoader(TensorDataset(inputs, inputs, torch.zeros(3)), batch_size=2)
    loss, predictions, truth = trainer.validate(loader)
    assert loss == pytest.approx(3.0)
    assert predictions.shape == truth.shape == (3,)
    with pytest.raises(ValueError, match="empty"):
        trainer.validate([])


def test_real_graph_single_sample_batch():
    from torch_geometric.data import Batch, Data

    from src.models.gnn_model import GraphDTA

    model = GraphDTA(
        node_feature_dim=3,
        gnn_hidden_dim=8,
        gnn_output_dim=8,
        protein_embed_dim=8,
        protein_num_filters=4,
        fc_dims=[8],
    )
    trainer = Trainer(model, torch.optim.Adam(model.parameters()))
    batch = Batch.from_data_list(
        [
            Data(
                x=torch.ones(2, 3),
                edge_index=torch.tensor([[0, 1], [1, 0]]),
                y=torch.tensor([1.0]),
                protein_input=torch.ones(1, 30, dtype=torch.long),
            )
        ]
    )
    loss, predictions, labels = trainer.validate([batch])
    assert np.isfinite(loss)
    assert predictions.shape == labels.shape == (1,)


def test_xgboost_early_stopping_and_reload(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 4))
    model = XGBoostDTI(n_estimators=50, early_stopping_rounds=3, max_depth=2, n_jobs=1)
    model.fit(x[:70], x[:70, 0], x[70:], -x[70:, 0], verbose=False)
    assert model.model.get_params()["early_stopping_rounds"] == 3
    assert model.best_iteration < 49
    path = tmp_path / "model.json"
    model.save(path)
    loaded = XGBoostDTI().load(path)
    np.testing.assert_allclose(model.predict(x), loaded.predict(x))


def test_masked_cnn_is_invariant_to_extra_right_padding():
    from src.models.deeptda import DeepDTA

    model = DeepDTA(
        drug_embed_dim=8, protein_embed_dim=8, num_filters=4, fc_dims=[8], mask_padding=True
    ).eval()
    short = torch.tensor([[1, 2, 3, 4, 5] + [0] * 10])
    long = torch.nn.functional.pad(short, (0, 20))
    torch.testing.assert_close(model.encode_drug(short), model.encode_drug(long))
    with pytest.raises(ValueError, match="encoded token"):
        model.encode_drug(torch.zeros_like(short))


def test_deduplicated_protein_encoding_preserves_predictions_and_gradients():
    from src.models.deeptda import DeepDTA

    repeated = DeepDTA(
        drug_embed_dim=8, protein_embed_dim=8, num_filters=4, fc_dims=[8], mask_padding=True
    )
    deduplicated = DeepDTA(**{**repeated.config, "deduplicate_proteins": True})
    deduplicated.load_state_dict(repeated.state_dict())
    tokens = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 0, 0], [3, 2, 1, 0, 0]])
    a, b = repeated.encode_protein(tokens), deduplicated.encode_protein(tokens)
    torch.testing.assert_close(a, b)
    a.sum().backward()
    b.sum().backward()
    for left, right in zip(
        repeated.protein_convs.parameters(), deduplicated.protein_convs.parameters()
    ):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-6)
