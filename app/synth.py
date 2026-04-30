"""Synthetic roster data generator.

Produces realistic-looking employee booking histories with the same shape as
the real anonymised inputs, so the benchmark and tests are reproducible without
shipping any production data.

Each employee is sampled with:
  - a base attendance rate
  - per-weekday preferences (some are MWF people, some TThF, some all-week)
  - a small chance of mid-history drift (changes pattern after month 3)
  - a "Thursday→Friday" correlation for a subset (ML-only signal)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


@dataclass(frozen=True)
class SynthConfig:
    n_employees: int = 50
    months: int = 6
    start: date = date(2025, 1, 1)
    drift_fraction: float = 0.20
    th_fr_corr_fraction: float = 0.25
    seed: int = 42


def _employee_profile(rng: np.random.Generator) -> dict:
    """Sample an employee's behavioural profile."""
    archetype = rng.choice(["mwf", "tthf", "allweek", "mtwt", "sparse"], p=[0.25, 0.20, 0.30, 0.15, 0.10])
    base_rate = float(rng.uniform(0.55, 0.95))

    weekday_pref = {wd: 0.5 for wd in WEEKDAYS}
    if archetype == "mwf":
        weekday_pref.update({"Monday": 0.9, "Wednesday": 0.9, "Friday": 0.85, "Tuesday": 0.2, "Thursday": 0.2})
    elif archetype == "tthf":
        weekday_pref.update({"Tuesday": 0.9, "Thursday": 0.9, "Friday": 0.7, "Monday": 0.25, "Wednesday": 0.3})
    elif archetype == "allweek":
        weekday_pref = {wd: float(rng.uniform(0.75, 0.95)) for wd in WEEKDAYS}
    elif archetype == "mtwt":
        weekday_pref.update({"Monday": 0.9, "Tuesday": 0.85, "Wednesday": 0.85, "Thursday": 0.85, "Friday": 0.2})
    else:
        weekday_pref = {wd: float(rng.uniform(0.25, 0.55)) for wd in WEEKDAYS}

    return {"archetype": archetype, "base_rate": base_rate, "weekday_pref": weekday_pref}


def _month_workdays(start: date, months: int) -> list[date]:
    """All weekdays (Mon-Fri) in the [start, start+months) window."""
    cur = start
    end_year, end_month = (start.year + (start.month - 1 + months) // 12,
                           ((start.month - 1 + months) % 12) + 1)
    end = date(end_year, end_month, 1)
    days: list[date] = []
    while cur < end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def generate_history(cfg: SynthConfig = SynthConfig()) -> pd.DataFrame:
    """Generate a long-format booking history.

    Returns a DataFrame with columns matching the cleaned roster format used by
    `app.core.clean_roster_excel`:
        Associate ID, Associate Name, Project ID, Project Allocation End Date,
        Project Manager ID, Start Time, End Time, City, Facility, Weekday, Date, Booked
    """
    rng = np.random.default_rng(cfg.seed)
    workdays = _month_workdays(cfg.start, cfg.months)
    half = len(workdays) // 2

    employees = []
    for i in range(cfg.n_employees):
        emp_id = f"E{1000 + i:04d}"
        profile = _employee_profile(rng)
        drifts = bool(rng.random() < cfg.drift_fraction)
        th_fr_corr = bool(rng.random() < cfg.th_fr_corr_fraction)
        employees.append({
            "id": emp_id,
            "name": f"Employee {i + 1:03d}",
            "profile": profile,
            "drifts": drifts,
            "th_fr_corr": th_fr_corr,
            "drift_profile": _employee_profile(rng) if drifts else None,
        })

    rows = []
    for emp in employees:
        last_thursday_booked: bool | None = None
        for idx, day in enumerate(workdays):
            if emp["drifts"] and idx >= half:
                profile = emp["drift_profile"]
            else:
                profile = emp["profile"]
            weekday = WEEKDAYS[day.weekday()]
            p = profile["base_rate"] * profile["weekday_pref"][weekday]

            if emp["th_fr_corr"] and weekday == "Friday" and last_thursday_booked is not None:
                p = 0.85 if last_thursday_booked else 0.15

            booked = int(rng.random() < min(0.99, max(0.01, p)))
            if weekday == "Thursday":
                last_thursday_booked = bool(booked)

            rows.append({
                "Associate ID": emp["id"],
                "Associate Name": emp["name"],
                "Project ID": "P0001",
                "Project Allocation End Date": pd.Timestamp(cfg.start) + pd.DateOffset(years=1),
                "Project Manager ID": "M0001",
                "Start Time": "09:00",
                "End Time": "18:00",
                "City": "Kochi",
                "Facility": "HQ",
                "Weekday": weekday,
                "Date": pd.Timestamp(day),
                "Booked": booked,
            })

    return pd.DataFrame(rows)


def split_history(
    df: pd.DataFrame,
    feature_months: Sequence[int],
    train_label_month: int,
    test_label_month: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split for leakage-safe evaluation.

    Returns (feature_df, train_df, test_df) where:
      - feature_df is used only to compute aggregate features (no labels exposed in training)
      - train_df labels come from train_label_month
      - test_df labels come from test_label_month
    """
    df = df.copy()
    df["_month"] = df["Date"].dt.month
    feature_df = df[df["_month"].isin(feature_months)].drop(columns="_month")
    train_df = df[df["_month"] == train_label_month].drop(columns="_month")
    test_df = df[df["_month"] == test_label_month].drop(columns="_month")
    return feature_df, train_df, test_df
