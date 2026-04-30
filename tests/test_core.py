"""Rule-based predictor invariants."""

from __future__ import annotations

import pandas as pd

from app.core import generate_booking_predictions
from app.synth import SynthConfig, generate_history


def _workdays_for_month(history: pd.DataFrame, month: int) -> pd.DataFrame:
    sub = history[history["Date"].dt.month == month]
    return sub[["Date", "Weekday"]].drop_duplicates().sort_values("Date").reset_index(drop=True)


def test_min_days_per_week_is_enforced() -> None:
    history = generate_history(SynthConfig(n_employees=8, months=3, seed=3))
    feature_df = history[history["Date"].dt.month <= 2]
    workdays = _workdays_for_month(history, 3)
    pred = generate_booking_predictions(feature_df, workdays, min_days_per_week=3, threshold=0.6)
    pred = pred.copy()
    pred["Date"] = pd.to_datetime(pred["Date"])
    pred["Week"] = pred["Date"].apply(lambda d: (d.day - 1) // 7 + 1)

    week_size = pred.groupby(["Associate ID", "Week"]).size()
    booked_per_week = pred.groupby(["Associate ID", "Week"])["Booked"].sum()
    target = booked_per_week.index.map(lambda k: min(3, int(week_size.loc[k])))
    underbooked = booked_per_week[booked_per_week.values < target]
    assert underbooked.empty, f"some weeks under-booked relative to available days: {underbooked}"


def test_threshold_extremes_behave_sensibly() -> None:
    history = generate_history(SynthConfig(n_employees=4, months=2, seed=5))
    feature_df = history[history["Date"].dt.month == 1]
    workdays = _workdays_for_month(history, 2)

    high = generate_booking_predictions(feature_df, workdays, min_days_per_week=1, threshold=0.99)
    low = generate_booking_predictions(feature_df, workdays, min_days_per_week=1, threshold=0.0)
    assert high["Booked"].sum() <= low["Booked"].sum()
    assert low["Booked"].sum() == len(low)
