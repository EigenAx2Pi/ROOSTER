# rooster — Roster Predictor (Rule-based + ML benchmark)

Python + FastAPI app that predicts employee rosters from historical Excel
attendance data. Ships **two predictors**:

1. **Production:** rule-based weekday-frequency threshold + min-days-per-week
   top-up. Lives in `app/core.py`. Interpretable, calibrated, zero retrain.
2. **Benchmark:** sklearn `GradientBoostingClassifier` with leakage-safe
   per-(employee, weekday) features. Lives in `app/ml.py`. Used to quantify
   the ceiling above the rule baseline.

See [`README.md`](./README.md) for the human-facing how-to and the design
rationale ("why rules, not ML").

## Layout

```
rooster/
├── app/
│   ├── core.py        # rule-based predictor (production)
│   ├── ml.py          # GBM benchmark + leakage-safe feature builder
│   ├── synth.py       # synthetic data generator (drift + Thu→Fri signal)
│   ├── eval.py        # held-out month evaluation
│   └── main.py        # FastAPI (upload → Excel out)
├── scripts/benchmark.py
├── static/            # vanilla HTML/CSS/JS UI
├── tests/             # pytest, incl. explicit no-leakage test
├── .github/workflows/ci.yml
├── Dockerfile
├── pyproject.toml
└── requirements.txt   # kept for backwards-compat
```

## Stack

- Python 3.10+ · FastAPI · pandas · openpyxl · scikit-learn ≥ 1.4
- ruff · pytest · GitHub Actions · Docker

## Working notes

- **Don't recompute features from full history.** The training-time feature
  window is months `1..N-2`; the test-time feature window is `1..N-1`. Each
  row's features must be derivable strictly from data dated *before* its
  label's date. `tests/test_ml.py::test_no_label_leakage_in_features` enforces
  this.
- **The rule predictor is not a strawman.** On data where the underlying
  signal is `P(book | employee, weekday)`, a frequency threshold *is*
  Bayes-optimal. ML wins where independence assumptions break (drift, Thu→Fri
  correlation in synth data).
- **`min_days_per_week` is enforced *after* thresholding**, not during. This
  way the constraint never silently overrides a high-confidence "no".
- **Synthetic data is for benchmarks and tests only.** Real data is PII —
  must not be committed.
- **Web UI** has no build step. Edit `static/*.html|css|js` directly.

## Origin

Re-imported from public `EigenAx2Pi/ROOSTER` (commit `d227b50`) for
production-grade upgrade before re-graduating. The old public repo will be
replaced or archived after `/graduate rooster` runs.

## Status

See [`STATUS.md`](./STATUS.md).
