import pandas as pd

from agrosurveillance.data.dataset import AgroStressDataset, build_zone_sequences, causal_split, collate_zone_batch
from agrosurveillance.data.schema import DEFAULT_SCHEMA, derive_health_label
from agrosurveillance.data.synthetic import generate_synthetic_dataset


def _make_df() -> pd.DataFrame:
    df = generate_synthetic_dataset(n_fields=3, rows_per_field=40, seed=7)
    df[DEFAULT_SCHEMA.health_label_col] = derive_health_label(df[DEFAULT_SCHEMA.stress_indicator_col])
    return df


def test_build_zone_sequences_basic_shapes():
    df = _make_df()
    seqs = build_zone_sequences(df, sequence_length=10, baseline_window=3)
    assert len(seqs) > 0
    for seq in seqs:
        t = seq.features.shape[0]
        assert seq.stress_gradient.shape == seq.features.shape
        assert seq.event_mask.shape[0] == t
        assert seq.stress_label.shape[0] == t
        assert seq.health_label.shape[0] == t


def test_causal_split_is_zone_exclusive():
    df = _make_df()
    seqs = build_zone_sequences(df, sequence_length=10, baseline_window=3)
    train, test = causal_split(seqs, train_fraction=0.7)
    train_zones = {s.zone_id for s in train}
    test_zones = {s.zone_id for s in test}
    assert train_zones.isdisjoint(test_zones)
    assert len(train) > 0 and len(test) > 0


def test_dataset_and_collate_padding():
    df = _make_df()
    seqs = build_zone_sequences(df, sequence_length=10, baseline_window=3)
    ds = AgroStressDataset(seqs)
    item = ds[0]
    assert item["features"].shape[0] == item["gradient"].shape[0]

    from torch.utils.data import DataLoader

    dl = DataLoader(ds, batch_size=4, collate_fn=collate_zone_batch)
    batch = next(iter(dl))
    b, t, d = batch["features"].shape
    assert batch["valid_mask"].shape == (b, t)
    assert batch["health_label"].shape == (b, t)
