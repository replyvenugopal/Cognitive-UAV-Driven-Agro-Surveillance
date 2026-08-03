import torch

from agrosurveillance.models.baselines import CNNBaseline, CNNLSTMBaseline, TemporalTransformerBaseline
from agrosurveillance.models.geospatio_trinet import ABLATION_COMPONENTS, GeoSpatioTriNet
from agrosurveillance.models.vulnerability import causal_decayed_sum, vulnerability_persistence, vulnerability_trajectory


def test_geospatio_trinet_forward_shapes():
    x = torch.randn(5, 8, 13)
    valid_mask = torch.ones(5, 8)
    model = GeoSpatioTriNet(input_dim=13, feature_dim=16, spatial_heads=2, temporal_hidden=16, temporal_layers=1)
    out = model(x, valid_mask=valid_mask)
    for key in ["stress_logits", "stress_prob", "vulnerability", "vulnerability_persistence"]:
        assert out[key].shape == (5, 8)


def test_geospatio_trinet_all_ablation_combinations_run():
    x = torch.randn(4, 6, 9)
    from itertools import combinations

    for r in range(len(ABLATION_COMPONENTS) + 1):
        for combo in combinations(ABLATION_COMPONENTS, r):
            model = GeoSpatioTriNet(input_dim=9, feature_dim=8, spatial_heads=2, temporal_hidden=8,
                                     temporal_layers=1, disable=combo)
            out = model(x)
            assert out["vulnerability"].shape == (4, 6)


def test_torch_baselines_forward_shapes():
    x = torch.randn(3, 7, 5)
    for cls in [CNNBaseline, CNNLSTMBaseline, TemporalTransformerBaseline]:
        model = cls(input_dim=5)
        out = model(x)
        assert out.shape == (3, 7)


def test_vulnerability_trajectory_is_causal():
    stress = torch.zeros(2, 6)
    stress[:, 3] = 1.0  # a single spike
    vuln = vulnerability_trajectory(stress, eta1=0.6, eta2=0.4, decay=0.85, horizon=5)
    # Nothing before the spike should be affected.
    assert torch.allclose(vuln[:, :3], torch.zeros(2, 3))
    assert torch.all(vuln[:, 3] > 0)


def test_vulnerability_persistence_uses_history():
    vuln = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
    persistence = vulnerability_persistence(vuln, decay=0.9, horizon=3)
    assert persistence[0, 3] > 0  # should feel the spike at t=2
    assert persistence[0, 0] == 0
