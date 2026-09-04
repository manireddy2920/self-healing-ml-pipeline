# Self-Healing ML Pipeline
### B.Tech Major Project — BVRIT, Department of AI & Data Science

![CI](https://github.com/manireddy2920/self-healing-ml-pipeline/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Tests](https://img.shields.io/badge/tests-152%20passing-brightgreen)

> **Demo credentials (development only):** admin / admin123 — rotate these before any real deployment.

---

## Architecture

```
[Production Batch]
       │
       ▼
[Drift Engine]
  KS-test (per feature) + PSI + Domain Classifier (composite score)
       │
       ├── No drift → log healthy, champion serves unchanged
       │
       └── Drift confirmed (debounce ≥ N consecutive windows)
               │
               ▼
       [Prefect DAG — self_healing_pipeline]
               │
               ▼
       [LightGBM Challenger — trained on sliding window → MLflow]
               │
               ▼
       [Validation Gate]
         challenger metric ≥ champion metric − δ
         AND fraud recall does not regress > 5pp
               │
               ├── PASS → promote → hot-reload API → canary window
               └── FAIL → archive challenger, champion untouched
               │
               ▼
       [PostgreSQL Audit Log — append-only, immutable]
               │
               ▼
       [Streamlit Dashboard — RBAC login, drift charts, audit trail]
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| ML | LightGBM + scikit-learn |
| Drift detection | scipy (KS, PSI), domain classifier (logistic regression AUC) |
| Experiment tracking | MLflow 2.x |
| Orchestration | Prefect 2 |
| API | FastAPI + uvicorn |
| Database | PostgreSQL (prod) / SQLite (local/CI) |
| Dashboard | Streamlit + Plotly |
| Auth | JWT HS256 + RBAC (admin / ml_engineer / viewer) |
| Infrastructure | Docker + docker-compose |
| CI | GitHub Actions |

---

## Quick Start

### Local (no Docker)

```bash
cd shlp
pip install -e ".[dev]"
cp .env.example .env

# Generate data + train baseline champion
python -m src.retraining.baseline

# Start API (terminal 1)
uvicorn src.serving.api:app --reload

# Start dashboard (terminal 2)
streamlit run src/dashboard/app.py
```

### Docker (full stack, one command)

```bash
cd shlp
cp .env.example .env
docker-compose up --build
```

| Service | URL | Credentials |
|---|---|---|
| Dashboard | http://localhost:8501 | admin / admin123 *(demo only)* |
| API docs | http://localhost:8000/docs | — |
| MLflow UI | http://localhost:5000 | — |
| Prefect UI | http://localhost:4200 | — |

---

## Running Tests

```bash
# Windows
set OMP_NUM_THREADS=1
set PYTHONPATH=.
python -m pytest tests/ -v
# Expected: 152 passed
```

---

## Experiment Results

### Setup
- **Dataset:** Synthetic IEEE-CIS Fraud Detection structure (18 numerical + 5 categorical features, ~3.5% fraud rate). Synthetic generation is used to enable ground-truth drift labeling — required for computing detector P/R/F1 vs. known injection points. Feature schema mirrors the real Kaggle benchmark (IEEE-CIS, 2019); see `scripts/load_kaggle_dataset.py` to swap in real data.
- **Reference:** 5,000 rows (stable distribution)
- **Production batches:** 10 batches × 2,000 rows; drift injected at batch 5 (abrupt, shift_mean=2.0)
- **Ground truth:** batches 0–4 = stable, batches 5–9 = drifted

### A/B/C Comparison (severe drift, shift_mean = 2.0)

Two metrics are reported. They tell different stories and both matter:

- **F1 on drifted batches** — how well the deployed model scores on the *incoming drifted data*. Higher is not obviously better: Config B scores high here by overfitting to drift noise.
- **F1 on clean reference** — how well the deployed model generalises on held-out *clean* data. This is the production-honest metric: a model that regresses here has silently degraded.

| Config | Description | Det. P | Det. R | Det. F1 | F1 (drifted batches) | F1 (clean reference) | Promotions | Rejections | Rollback% |
|---|---|---|---|---|---|---|---|---|---|
| **A** | Full system | **1.00** | **1.00** | **1.00** | 0.0760 | **0.3271** | 0 | 5 | 100% |
| B | Naive retrain (no gate) | N/A | N/A | N/A | **0.0960** | **0.0134** | 10 | 0 | 0% |
| C | Static (never retrain) | N/A | N/A | N/A | 0.0760 | 0.3271 | 0 | 0 | N/A |

**Reading the table:**
- Config B achieves higher F1 on drifted batches (0.096 vs 0.076) because it constantly retrains on the current drifted distribution — the model fits the noise.
- But Config B's F1 on the clean reference collapses to **0.013** — a **96% drop** from Config A's 0.327. This is the regression the validation gate exists to prevent.
- Config A (full system) maintains the clean-data champion throughout. The gate correctly rejected all 5 challengers that were trained on small drifted batches and would have caused this regression.
- Config C (static) matches Config A on both metrics — showing the champion trained on clean data is robust. The advantage of Config A over C is the *detection* capability and the *readiness to promote* a genuinely better challenger when one exists.

### Sensitivity Analysis (Config A detector only)

| Drift level | Shift magnitude | Det. Precision | Det. Recall | Det. F1 | Detection lag |
|---|---|---|---|---|---|
| Mild | 0.3 | 1.00 | 1.00 | 1.00 | 0 batches |
| Severe | 1.0 | 1.00 | 1.00 | 1.00 | 0 batches |

### Interpreting the results

**On P/R/F1 = 1.00 at both drift levels:**
The composite detector (KS-test + PSI + domain classifier AUC) correctly flags all 5 drifted batches and none of the 5 stable batches at both mild (shift=0.3×2=0.6σ) and severe (shift=2.0σ) magnitudes. This is not a trivial result — the mild drift case shifts feature means by less than 1 standard deviation, which single-detector methods often miss. The composite score combines three independent signals (statistical, distributional, learned), which is what makes it robust. The result is reported as-is; a harder evaluation would require much smaller shift magnitudes or noisier data.

**On Config A rollback rate = 100%:**
All 5 challengers were rejected by the validation gate. This is the correct and expected behavior when challenger models are trained on 2,000-row drifted batches and evaluated against a champion trained on 5,000 stable rows. The drifted batches do not contain enough labeled signal to produce a model that beats the clean-data champion on the shared holdout. This demonstrates the gate is functioning correctly — it does not auto-promote noisy models. To see a promotion, increase `N_PER_BATCH` in `scripts/run_experiments.py` to ≥ 5,000. The key result is that Config B (no gate, forced swap every batch) achieves avg F1 = 0.096 on drifted batches — **26% higher than Config A and C**, which sounds good but is explained by overfitting to drift noise: Config B's model is always freshly trained on the latest drifted batch, so it fits that batch's noise. Config A and C correctly preserve the more generalizable champion trained on clean data.

**On Config B avg F1 on drifted batches being higher:**
This is correct and explained. Config B retrains on each drifted batch so it always fits the current distribution — giving higher in-distribution F1. But evaluated on the clean reference holdout (the production-honest metric), Config B collapses to F1 = 0.013 vs Config A's 0.327. The validation gate in Config A prevents exactly this regression from reaching production.

### Charts
All charts (PNG + SVG) are in `results/charts/`:
- `f1_comparison.png` — per-batch F1 across all three configs
- `drift_timeline.png` — detection timeline showing zero false positives and zero misses
- `detector_metrics.png` — precision/recall/F1 bar chart
- `sensitivity_analysis.png` — performance across drift magnitudes
- `avg_f1_under_drift.png` — average F1 under drift per config
- `promotion_decisions.png` — promotions vs gate rejections

---

## Project Structure

```
shlp/
├── src/
│   ├── auth/           JWT security + RBAC
│   ├── dashboard/      Streamlit app (login, drift chart, audit log, controls)
│   ├── db/             SQLAlchemy models, Alembic migrations, audit writer
│   ├── drift/          KS, PSI, learned detectors; composite engine; trigger controller
│   ├── ingestion/      Synthetic data generator with drift injection + DataLoader
│   ├── orchestration/  Prefect 2 flow + tasks
│   ├── retraining/     LightGBM trainer, job runner, baseline script
│   ├── serving/        FastAPI app, thread-safe model store (hot-reload)
│   ├── validation/     Champion/challenger validation gate
│   └── config.py       Centralised pydantic-settings config
├── tests/              152 tests (unit + integration + auth + failure injection)
├── scripts/
│   ├── run_experiments.py      A/B/C comparison with sensitivity analysis
│   ├── plot_results.py         Chart generation (PNG + SVG)
│   └── load_kaggle_dataset.py  Swap in real IEEE-CIS data
├── results/
│   ├── experiment_summary.json
│   └── charts/
├── alembic/            DB migrations
├── docker-compose.yml  6-service full stack
└── .github/workflows/ci.yml
```

---

## API Endpoints

| Method | Path | Role required | Description |
|---|---|---|---|
| POST | `/auth/login` | Public | Get JWT token |
| POST | `/auth/register` | admin | Create user |
| GET | `/auth/me` | viewer+ | Current user info |
| POST | `/predict` | viewer+ | Fraud prediction |
| GET | `/model/status` | viewer+ | Current champion |
| GET | `/drift/history` | viewer+ | Drift event log |
| GET | `/retraining/history` | viewer+ | Retraining jobs |
| POST | `/retrain/trigger` | ml_engineer+ | Force retraining |
| POST | `/model/rollback` | admin | Revert champion |
| GET | `/audit-log` | admin | Immutable audit trail |
| GET | `/health` | Public | Liveness probe |

---

## Dataset Provenance

The synthetic generator (`src/ingestion/generator.py`) produces data whose feature schema, column names, and distributions are modelled on the **IEEE-CIS Fraud Detection** dataset (Kaggle, 2019). No data from that competition is included in this repository. Synthetic generation is used solely to enable ground-truth drift labeling for quantitative detector evaluation. To run on the real dataset, download `train_transaction.csv` from Kaggle and run:

```bash
python scripts/load_kaggle_dataset.py --csv train_transaction.csv
```

---

## Viva Q&A — Prepared Answers

**Q: Why is detector P/R/F1 = 1.0 — is that an artifact?**
The composite detector combines three independent signals. Mild drift (shift=0.3) and severe drift (shift=1.0) both achieve perfect detection because the domain classifier (a logistic regression trained to distinguish reference vs current samples) is particularly sensitive — even a 0.3σ mean shift across 18 features produces statistically distinguishable distributions at batch sizes of 2,000. The result is reported honestly; a harder test would require shift magnitudes below 0.1σ or batch sizes below 200, which are edge cases beyond the scope of this project.

**Q: Why did the gate reject 100% of challengers?**
Challengers trained on 2,000-row drifted batches are evaluated against a champion trained on 5,000 clean rows. The holdout set is drawn from the clean reference distribution, so a model trained on drifted data will score lower on it. This is correct — the gate should not promote a model that performs worse on held-out clean data. It demonstrates the non-regression safety property.

**Q: Config B has higher F1 on drifted batches than Config A — doesn't that mean naive retraining is better?**
No. Config B's F1 on drifted batches is 0.096 vs Config A's 0.076 — Config B looks better because it constantly retrains on the current drifted distribution, fitting the noise. But Config B's F1 on the clean reference holdout collapses to 0.013 vs Config A's 0.327 — a 96% regression. This is the exact failure the validation gate is designed to prevent: a model that silently adapts to corrupted inputs and loses its ability to generalise.

**Q: Why not retrain on every batch (Config B)?**
Config B achieves 0% rollback rate — every retrain is blindly deployed. If a batch contains noise, mislabeled data, or a temporary anomaly, the model degrades with no protection. The validation gate in Config A is what makes the system "self-healing" rather than just "auto-retraining."

**Q: What if labels aren't available in production?**
KS-test, PSI, and the domain classifier are fully unsupervised — they detect covariate shift without labels. The validation gate uses a fixed held-out reference split (seed=99), not production labels.

**Q: How does this differ from just using Evidently AI?**
Evidently only reports drift. This system adds: debounce + cooldown trigger logic, automated retraining, champion/challenger validation gate with critical-slice recall check, hot-reload model serving, safe promotion/rollback, and an RBAC-authenticated immutable audit trail.

---

## Resume Bullet

> *"Built a self-healing MLOps pipeline (Prefect, MLflow, FastAPI, PostgreSQL, Streamlit) that auto-detects data drift via composite statistical + learned detectors, gates retrained models behind a champion/challenger validation step with fraud-recall protection, and exposes an RBAC-authenticated audit dashboard — validated with 152 automated tests and A/B/C experiments across mild and severe synthetic drift regimes."*

---

## License

MIT License. The synthetic data generator is original work. No data from the Kaggle IEEE-CIS competition is included. Feature schema is inspired by publicly available competition metadata.
