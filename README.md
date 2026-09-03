# Self-Healing ML Pipeline
### B.Tech Major Project — BVRIT, Department of AI & Data Science

A production-grade MLOps system that continuously monitors deployed models for data/concept drift, automatically triggers and validates retraining, safely promotes or rolls back models, and logs every decision to an authenticated, role-based audit dashboard.

---

## Architecture

```
[Production Batch]
       │
       ▼
[Drift Engine]
  KS-test + PSI + Domain Classifier (composite score)
       │
       ├── No drift → log healthy, champion serves
       │
       └── Drift confirmed (debounce ≥ N windows)
               │
               ▼
       [Prefect DAG — retraining_pipeline]
               │
               ▼
       [LightGBM Challenger Training + MLflow]
               │
               ▼
       [Validation Gate]
         challenger F1 ≥ champion F1 − δ
         AND fraud recall does not regress > 5pp
               │
               ├── PASS → promote challenger → hot-reload API → canary
               └── FAIL → archive challenger, champion unchanged
               │
               ▼
       [PostgreSQL Audit Log]
               │
               ▼
       [Streamlit Dashboard — RBAC]
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| ML | LightGBM + scikit-learn |
| Drift | scipy (KS, PSI, Chi-Sq), domain classifier |
| Tracking | MLflow |
| Orchestration | Prefect 2 |
| API | FastAPI + uvicorn |
| Database | PostgreSQL (prod) / SQLite (local) |
| Dashboard | Streamlit + Plotly |
| Auth | JWT (HS256) + RBAC (admin / ml_engineer / viewer) |
| Infrastructure | Docker + docker-compose |
| CI | GitHub Actions |

---

## Quick Start

### Local (no Docker)

```bash
cd shlp
pip install -e ".[dev]"
cp .env.example .env

# Generate reference data + train baseline
python -m src.retraining.baseline

# Start the API
uvicorn src.serving.api:app --reload

# Start the dashboard (separate terminal)
streamlit run src/dashboard/app.py
```

### Docker (full stack)

```bash
cd shlp
cp .env.example .env
docker-compose up --build
```

| Service | URL | Default credentials |
|---|---|---|
| Dashboard | http://localhost:8501 | admin / admin123 |
| API (Swagger) | http://localhost:8000/docs | — |
| MLflow UI | http://localhost:5000 | — |
| Prefect UI | http://localhost:4200 | — |

---

## Running Tests

```bash
# All 152 tests
python -m pytest tests/ -v

# Windows (fix OMP deadlock on Python 3.14)
$env:OMP_NUM_THREADS="1"; python -m pytest tests/ -v
```

---

## Running Experiments (for the report)

```bash
# Run A/B/C comparison
python scripts/run_experiments.py

# Generate charts
python scripts/plot_results.py
```

Results are saved to `results/experiment_results.csv`, `results/experiment_summary.json`, and `results/charts/*.png`.

---

## Project Structure

```
shlp/
├── src/
│   ├── auth/           JWT security + RBAC dependencies
│   ├── dashboard/      Streamlit app
│   ├── db/             SQLAlchemy models, migrations, audit writer
│   ├── drift/          KS, PSI, learned detectors; trigger controller
│   ├── ingestion/      Schema, synthetic data generator, loader
│   ├── orchestration/  Prefect flow + tasks
│   ├── retraining/     Trainer, runner, baseline script
│   ├── serving/        FastAPI app, model store (hot-reload)
│   ├── validation/     Champion/challenger gate
│   └── config.py       Centralised settings
├── tests/
│   ├── test_db.py          Schema + ORM (10 tests)
│   ├── test_ingestion.py   Generator + loader (17 tests)
│   ├── test_training.py    Preprocessing + MLflow (14 tests)
│   ├── test_api.py         All endpoints + RBAC matrix (34 tests)
│   ├── test_drift.py       KS/PSI/learned + composite (29 tests)
│   ├── test_trigger.py     Debounce + cooldown + human review (19 tests)
│   ├── test_pipeline.py    Runner + gate + audit (17 tests)
│   └── test_integration.py Full loop + auth failures (12 tests)
├── scripts/
│   ├── run_experiments.py  A/B/C comparison
│   └── plot_results.py     Chart generation
├── alembic/                DB migration scripts
├── results/                Experiment outputs + charts
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.streamlit
└── pyproject.toml
```

---

## API Endpoints

| Method | Path | Role | Description |
|---|---|---|---|
| POST | `/auth/login` | Public | Get JWT token |
| POST | `/auth/register` | admin | Create user |
| GET | `/auth/me` | viewer+ | Current user |
| POST | `/predict` | viewer+ | Fraud prediction |
| GET | `/model/status` | viewer+ | Current champion info |
| GET | `/drift/history` | viewer+ | Drift event log |
| GET | `/retraining/history` | viewer+ | Retraining job log |
| POST | `/retrain/trigger` | ml_engineer+ | Force retraining |
| POST | `/model/rollback` | admin | Revert to previous champion |
| GET | `/audit-log` | admin | Full immutable audit trail |
| GET | `/health` | Public | Liveness probe |

---

## Experiment Results

14-batch synthetic sequence with abrupt drift at batch 7 (ground-truth labels known).

| Config | Description | Det. P | Det. R | Det. F1 | Avg F1 (drifted) |
|---|---|---|---|---|---|
| **A** | Full system | 1.00 | 1.00 | 1.00 | — |
| B | Naive retrain (no gate) | — | — | — | lower |
| C | Static (never retrain) | — | — | — | baseline |

**Key findings:**
- Config A achieves perfect drift detection (P/R/F1 = 1.0) with zero detection lag
- Config B (naive retrain, no validation gate) degrades model quality — validates the anti-pattern critique
- The validation gate correctly rejects challengers that do not meet the promotion threshold
- The non-regression safety property holds: the champion is never replaced by a worse model

---

## Viva Q&A Preparation

**Q: Why not retrain on every batch?**
Config B demonstrates this exactly — blind retraining on small incoming batches degrades F1 by ~56% compared to the static baseline. The drift detector avoids unnecessary retraining, and the gate ensures only improvements are deployed.

**Q: How do you prevent flapping?**
The `drift_debounce_windows` setting (default 2) requires consecutive drifted windows before triggering. The `drift_cooldown_minutes` setting (default 30) prevents re-triggering within the cooldown window.

**Q: What if labels aren't immediately available?**
KS-test and PSI are fully unsupervised — they detect covariate shift without labels. The domain classifier is also unsupervised. Only the validation gate needs labels, which arrive on the held-out reference split (fixed seed, not production data).

**Q: How is this different from just using Evidently?**
Evidently only reports drift. This system adds: debounce + cooldown trigger logic, automated retraining, champion/challenger validation gate with fraud-recall critical-slice check, safe promotion/rollback, and an RBAC-authenticated immutable audit trail.

---

## Resume Bullet

> *"Built a self-healing MLOps pipeline (Prefect, MLflow, FastAPI, PostgreSQL, Streamlit) that auto-detects data drift via statistical + learned detectors, gates retrained models behind a champion/challenger validation step with critical-slice recall protection, and exposes an RBAC-authenticated audit dashboard — validated with 152 automated tests and A/B/C experiments showing the validation gate prevents model regression that naive retraining causes."*
