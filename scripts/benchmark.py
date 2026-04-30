"""End-to-end benchmark on synthetic data.

Run with:
    python -m scripts.benchmark

Generates a 6-month synthetic roster, trains the gradient-boosted model on
month 5 with features from months 1-4, evaluates against the rule-based
baseline on the held-out month 6, and prints a Markdown metrics table.
"""

from __future__ import annotations

import argparse

from app.eval import evaluate, metrics_to_markdown
from app.synth import SynthConfig, generate_history, split_history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--employees", type=int, default=50)
    parser.add_argument("--months", type=int, default=6)
    args = parser.parse_args()

    cfg = SynthConfig(n_employees=args.employees, months=args.months, seed=args.seed)
    history = generate_history(cfg)

    feature_months = list(range(1, cfg.months - 1))
    train_label_month = cfg.months - 1
    test_label_month = cfg.months

    feature_df, train_df, test_df = split_history(
        history, feature_months, train_label_month, test_label_month
    )

    results = evaluate(feature_df, train_df, test_df)

    print(f"# Rooster benchmark (synthetic, n={cfg.n_employees}, months={cfg.months}, seed={cfg.seed})\n")
    print(f"Held-out month: {test_label_month}\n")
    print(metrics_to_markdown(results))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
