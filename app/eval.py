"""Held-out-month evaluation: rule-based vs gradient-boosted model.

Runs the leakage-safe protocol:
  - feature window: months [1..N-1]
  - train labels:   month  N
  - test labels:    month  N+1   (held out, never seen by either model)

Both models predict on the test month and are scored on the same labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.core import generate_booking_predictions
from app.ml import (
    build_feature_matrix,
    compute_employee_stats,
    enforce_min_days_per_week,
    predict_month,
    train_model,
)


@dataclass
class Metrics:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float | None

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "model": self.name,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "auc": round(self.auc, 4) if self.auc is not None else None,
        }


def _score(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> Metrics:
    auc = float(roc_auc_score(y_true, y_proba)) if y_proba is not None and len(set(y_true)) > 1 else None
    return Metrics(
        name=name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        auc=auc,
    )


def _rule_baseline(
    feature_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    min_days_per_week: int = 3,
    threshold: float = 0.6,
) -> pd.DataFrame:
    """Run the existing rule-based predictor against the test month."""
    workdays = (
        test_df[["Date", "Weekday"]].drop_duplicates().sort_values("Date").reset_index(drop=True)
    )
    predicted = generate_booking_predictions(
        feature_df,
        workdays,
        min_days_per_week=min_days_per_week,
        threshold=threshold,
    )
    return predicted.rename(columns={"Booked": "Predicted"})


def evaluate(
    feature_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    min_days_per_week: int = 3,
    rule_threshold: float = 0.6,
    ml_threshold: float = 0.5,
) -> dict[str, Metrics]:
    """Train ML on `train_df` (with stats from `feature_df`), test on `test_df`."""
    stats_for_train = compute_employee_stats(feature_df)
    X_train, y_train = build_feature_matrix(train_df, stats_for_train, feature_df)
    model = train_model(X_train, y_train)

    history_for_test = pd.concat([feature_df, train_df], ignore_index=True)
    stats_for_test = compute_employee_stats(history_for_test)
    ml_pred = predict_month(model, test_df, stats_for_test, history_for_test, threshold=ml_threshold)
    ml_pred_constrained = enforce_min_days_per_week(ml_pred, min_days=min_days_per_week)

    rule_pred = _rule_baseline(
        history_for_test, test_df,
        min_days_per_week=min_days_per_week, threshold=rule_threshold,
    )

    truth = test_df.sort_values(["Associate ID", "Date"]).reset_index(drop=True)
    truth_key = list(zip(truth["Associate ID"], truth["Date"]))
    truth_y = truth["Booked"].to_numpy()

    def align(pred_df: pd.DataFrame, score_col: str) -> np.ndarray:
        pred_df = pred_df.copy()
        pred_df["Date"] = pd.to_datetime(pred_df["Date"])
        idx = pred_df.set_index([pred_df["Associate ID"], pd.to_datetime(pred_df["Date"])])
        return np.asarray([idx.loc[k, score_col] for k in truth_key])

    ml_proba_aligned = align(ml_pred, "Probability")
    ml_pred_aligned = align(ml_pred_constrained, "Predicted")
    rule_pred_aligned = align(rule_pred, "Predicted")

    results = {
        "rule": _score("Rule-based (baseline)", truth_y, rule_pred_aligned.astype(int), None),
        "ml": _score("Gradient-boosted (ML)", truth_y, ml_pred_aligned.astype(int), ml_proba_aligned),
    }
    return results


def metrics_to_markdown(results: dict[str, Metrics]) -> str:
    """Render an evaluation result dict as a Markdown table."""
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for metrics in results.values():
        r = metrics.as_dict()
        auc_str = f"{r['auc']:.3f}" if r["auc"] is not None else "n/a"
        lines.append(
            f"| {r['model']} | {r['accuracy']:.3f} | {r['precision']:.3f} | "
            f"{r['recall']:.3f} | {r['f1']:.3f} | {auc_str} |"
        )
    return "\n".join(lines)
