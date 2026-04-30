"""Synthetic data generator should be deterministic and well-shaped."""

from __future__ import annotations

import pandas as pd

from app.synth import SynthConfig, generate_history, split_history


def test_generate_history_is_deterministic() -> None:
    cfg = SynthConfig(n_employees=10, months=2, seed=7)
    a = generate_history(cfg)
    b = generate_history(cfg)
    pd.testing.assert_frame_equal(a, b)


def test_generate_history_shape() -> None:
    cfg = SynthConfig(n_employees=5, months=2, seed=1)
    df = generate_history(cfg)
    assert {"Associate ID", "Associate Name", "Date", "Weekday", "Booked"} <= set(df.columns)
    assert df["Booked"].isin([0, 1]).all()
    assert df["Weekday"].isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]).all()
    assert df["Associate ID"].nunique() == 5


def test_split_history_partitions_correctly() -> None:
    cfg = SynthConfig(n_employees=3, months=4, seed=2)
    df = generate_history(cfg)
    feature_df, train_df, test_df = split_history(df, [1, 2], 3, 4)
    assert set(feature_df["Date"].dt.month.unique()) == {1, 2}
    assert set(train_df["Date"].dt.month.unique()) == {3}
    assert set(test_df["Date"].dt.month.unique()) == {4}
