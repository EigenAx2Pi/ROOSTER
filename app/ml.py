"""Gradient-boosted booking predictor.

A leakage-safe per-(employee, workday) classifier that predicts whether an
employee will book a given future weekday. Compares against the rule-based
baseline in `app.core` on a held-out month.

Features (computed only from the feature window — never from labels):
  - weekday one-hot
  - week-of-month, month
  - employee overall booking rate
  - employee per-weekday booking rate
  - employee recent (last-month-of-feature-window) booking rate
  - last Thursday booking signal (when predicting Friday)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


@dataclass
class EmployeeStats:
    """Aggregated per-employee booking stats from a feature window."""
    overall_rate: float
    weekday_rate: dict[str, float]
    recent_rate: float


def compute_employee_stats(feature_df: pd.DataFrame) -> dict[str, EmployeeStats]:
    """Aggregate features per employee from a strictly-historical window."""
    feature_df = feature_df.copy()
    feature_df["Date"] = pd.to_datetime(feature_df["Date"])
    cutoff = feature_df["Date"].max() - pd.Timedelta(days=30)

    stats: dict[str, EmployeeStats] = {}
    for emp_id, grp in feature_df.groupby("Associate ID"):
        weekday_rate = (
            grp.groupby("Weekday")["Booked"].mean().reindex(WEEKDAYS).fillna(0.5).to_dict()
        )
        recent = grp[grp["Date"] >= cutoff]
        stats[emp_id] = EmployeeStats(
            overall_rate=float(grp["Booked"].mean()),
            weekday_rate={k: float(v) for k, v in weekday_rate.items()},
            recent_rate=float(recent["Booked"].mean()) if len(recent) else float(grp["Booked"].mean()),
        )
    return stats


def _row_features(row: pd.Series, stats: dict[str, EmployeeStats], prev_thu: dict[str, int | None]) -> list[float]:
    emp_id = row["Associate ID"]
    weekday = row["Weekday"]
    s = stats.get(emp_id)
    if s is None:
        s = EmployeeStats(overall_rate=0.5, weekday_rate={w: 0.5 for w in WEEKDAYS}, recent_rate=0.5)

    one_hot = [1.0 if weekday == w else 0.0 for w in WEEKDAYS]
    week_of_month = (row["Date"].day - 1) // 7 + 1
    month = row["Date"].month
    last_thu = prev_thu.get(emp_id)
    last_thu_booked = float(last_thu) if last_thu is not None else 0.5
    is_friday = 1.0 if weekday == "Friday" else 0.0

    return [
        *one_hot,
        float(week_of_month),
        float(month),
        s.overall_rate,
        s.weekday_rate.get(weekday, 0.5),
        s.recent_rate,
        last_thu_booked,
        is_friday,
    ]


FEATURE_NAMES = [
    *(f"is_{w.lower()}" for w in WEEKDAYS),
    "week_of_month",
    "month",
    "emp_overall_rate",
    "emp_weekday_rate",
    "emp_recent_rate",
    "prev_thu_booked",
    "is_friday",
]


def build_feature_matrix(
    label_df: pd.DataFrame,
    stats: dict[str, EmployeeStats],
    history_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialise (X, y) for a label window, using stats computed elsewhere.

    The `history_df` is consulted to find each employee's last Thursday booking
    immediately before each target date — without ever consulting the label.
    """
    label_df = label_df.copy().sort_values(["Associate ID", "Date"]).reset_index(drop=True)
    history_df = history_df.copy()
    history_df["Date"] = pd.to_datetime(history_df["Date"])
    label_df["Date"] = pd.to_datetime(label_df["Date"])

    thu_history = (
        history_df[history_df["Weekday"] == "Thursday"]
        .sort_values("Date")
        .groupby("Associate ID")
    )
    last_thu_per_emp_by_date: dict[str, list[tuple[pd.Timestamp, int]]] = {
        emp: list(zip(g["Date"], g["Booked"])) for emp, g in thu_history
    }

    rows: list[list[float]] = []
    labels: list[int] = []
    for _, row in label_df.iterrows():
        prev_thu_booked = None
        thu_records = last_thu_per_emp_by_date.get(row["Associate ID"], [])
        for thu_date, booked in reversed(thu_records):
            if thu_date < row["Date"]:
                prev_thu_booked = int(booked)
                break
        rows.append(_row_features(row, stats, {row["Associate ID"]: prev_thu_booked}))
        labels.append(int(row["Booked"]))
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int)


def train_model(X: np.ndarray, y: np.ndarray, *, random_state: int = 42) -> Pipeline:
    """Fit a gradient-boosted classifier with feature scaling."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("gbm", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=random_state,
        )),
    ])
    pipeline.fit(X, y)
    return pipeline


def predict_month(
    model: Pipeline,
    target_df: pd.DataFrame,
    stats: dict[str, EmployeeStats],
    history_df: pd.DataFrame,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Predict bookings for a future month using a trained model.

    Returns a DataFrame with columns: Associate ID, Associate Name, Date, Weekday, Booked.
    """
    X, _ = build_feature_matrix(target_df, stats, history_df)
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
    out = target_df.sort_values(["Associate ID", "Date"]).reset_index(drop=True).copy()
    out["Probability"] = proba
    out["Predicted"] = (proba >= threshold).astype(int)
    return out[["Associate ID", "Associate Name", "Date", "Weekday", "Probability", "Predicted"]]


def enforce_min_days_per_week(predicted: pd.DataFrame, min_days: int = 3) -> pd.DataFrame:
    """Ensure each (employee, week-of-month) has at least `min_days` bookings.

    If fewer than `min_days` were predicted, top up with the highest-probability
    days for that week. Mirrors the rule-based fallback in `app.core`.
    """
    df = predicted.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Week"] = df["Date"].apply(lambda d: (d.day - 1) // 7 + 1)

    for (_emp, _week), idx in df.groupby(["Associate ID", "Week"]).groups.items():
        group = df.loc[idx]
        if int(group["Predicted"].sum()) < min_days:
            top = group.sort_values("Probability", ascending=False).head(min_days)
            df.loc[top.index, "Predicted"] = 1
    return df.drop(columns="Week")
