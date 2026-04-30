# rooster — Status

> Last updated: 2026-04-30

## Current focus

Re-imported from public `EigenAx2Pi/ROOSTER` and **upgraded for graduation**:
gradient-boosted ML benchmark, leakage-safe evaluation harness, synthetic data
generator, type hints, tests, CI, Dockerfile, recruiter-grade README.

## In flight

- `/graduate rooster` (user-run from `~/repo/playfield/`) → push to public
  `EigenAx2Pi/ROOSTER` (existing repo) or graduate to a new repo.

## Next up (post-graduation)

- Replace synthetic-data screenshot in README with a real anonymised demo
  screenshot of the web UI.
- Optional: add a `/api/benchmark` endpoint that runs the held-out evaluation
  on uploaded data and returns the metrics table.
- Optional: ONNX export of the GBM for cold-start latency.

## Blockers

- (none)

## Recently completed

- Renamed legacy `app/rooster.py` → `app/core.py`; dropped duplicate
  `ML RULE BASED ROOSTER.py`.
- Added `app/ml.py` (sklearn `GradientBoostingClassifier` with leakage-safe
  features), `app/synth.py` (deterministic synthetic 6-month roster generator
  with drift + Thu→Fri correlation), `app/eval.py` (held-out evaluation, both
  models scored on the same labels).
- Added `scripts/benchmark.py` — one-command rule-vs-ML comparison.
- `pyproject.toml` (PEP 621), ruff config, pytest config, MIT `LICENSE`.
- Tests: `tests/test_synth.py`, `tests/test_core.py`, `tests/test_ml.py`
  (incl. an explicit no-leakage test).
- `Dockerfile` (multi-stage, non-root user, healthcheck) + GitHub Actions CI
  matrix (Python 3.10/3.11/3.12).
- Rewrote README for recruiter / CTO audience: problem framing, results table,
  architecture diagram, "why rules not ML" section, what-I-learned.

## Graduation readiness

- [x] `README.md` (rewritten for external audience)
- [x] No hardcoded secrets (re-verified)
- [x] Happy path works (FastAPI + uvicorn boot, web UI loads)
- [x] `CLAUDE.md` (updated for new architecture)
- [x] Enhancement scope decided and shipped (ML benchmark + production polish)
- [x] Tests + CI
- [x] Dockerfile
