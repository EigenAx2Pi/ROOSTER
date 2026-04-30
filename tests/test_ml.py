"""ML pipeline smoke tests."""

from __future__ import annotations

import numpy as np

from app.eval import evaluate
from app.ml import (
    FEATURE_NAMES,
    build_feature_matrix,
    compute_employee_stats,
    train_model,
)
from app.synth import SynthConfig, generate_history, split_history


def test_feature_matrix_shape_matches_feature_names() -> None:
    cfg = SynthConfig(n_employees=5, months=3, seed=11)
    df = generate_history(cfg)
    feature_df, train_df, _ = split_history(df, [1], 2, 3)
    stats = compute_employee_stats(feature_df)
    X, y = build_feature_matrix(train_df, stats, feature_df)
    assert X.shape[1] == len(FEATURE_NAMES)
    assert X.shape[0] == y.shape[0] == len(train_df)


def test_model_learns_signal() -> None:
    cfg = SynthConfig(n_employees=20, months=4, seed=13)
    df = generate_history(cfg)
    feature_df, train_df, test_df = split_history(df, [1, 2], 3, 4)
    stats = compute_employee_stats(feature_df)
    X_train, y_train = build_feature_matrix(train_df, stats, feature_df)
    model = train_model(X_train, y_train)
    train_acc = (model.predict(X_train) == y_train).mean()
    assert train_acc > 0.65, f"model not learning: train acc {train_acc:.3f}"


def test_evaluate_returns_both_models() -> None:
    cfg = SynthConfig(n_employees=15, months=4, seed=17)
    df = generate_history(cfg)
    feature_df, train_df, test_df = split_history(df, [1, 2], 3, 4)
    results = evaluate(feature_df, train_df, test_df)
    assert {"rule", "ml"} <= set(results.keys())
    for r in results.values():
        assert 0.0 <= r.accuracy <= 1.0
        assert 0.0 <= r.f1 <= 1.0


def test_no_label_leakage_in_features() -> None:
    """Features for a row must not depend on its own label."""
    cfg = SynthConfig(n_employees=4, months=3, seed=19)
    df = generate_history(cfg)
    feature_df, train_df, _ = split_history(df, [1], 2, 3)
    stats = compute_employee_stats(feature_df)
    X1, _ = build_feature_matrix(train_df, stats, feature_df)
    flipped = train_df.copy()
    flipped["Booked"] = 1 - flipped["Booked"]
    X2, _ = build_feature_matrix(flipped, stats, feature_df)
    assert np.allclose(X1, X2), "feature matrix changed when labels flipped — leakage!"
