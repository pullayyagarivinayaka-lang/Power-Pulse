# PowerPulse - Model 2: Appliance-Level Energy Estimation

Part of **PowerPulse - AI-Driven Energy Consumption & Cost Optimization**.
This model turns appliance power data into **appliance-wise kWh and estimated cost**, always saying whether a number is
*measured*, *calculated* (rated-power formula) or *ML-estimated*.

> Status: implementation of the PRISM Model 2 draft (PP-M02 v0.1). Dataset release/licence, owners, tariff and approvals are **TBD** until your team confirms them. Nothing here is a benchmark result until you run it on approved, real data.

## What it does

| Stage | Module | PRISM link |
|---|---|---|
| Load readings + channel->appliance mapping from config | `src/ingest.py`, `configs/*.json` | FR-01, AC-03 |
| Validate units, timestamps, gaps, duplicates, outliers; write quality report | `src/preprocess.py` | FR-02, AC-01 |
| Detect operating sessions (per-appliance thresholds) and integrate **measured kWh** | `src/energy.py` | FR-03, FR-04, AC-02 |
| Rated-power baseline `kWh = W x h / 1000` | `src/baseline.py` | FR-05 |
| Leakage-safe features + chronological purged split | `src/features.py`, `src/split.py` | AC-04 |
| Random Forest, Gradient Boosting, hybrid (baseline + ML correction); ML only "justified" if it beats the baseline on validation | `src/models.py`, `src/train.py` | FR-06 |
| MAE/RMSE vs baseline, per-appliance + daily error analysis, figures | `src/evaluate.py` | AC-05 |
| Energy + cost with provenance, assumptions, warnings (JSON contract) | `src/predict.py`, `src/tariff.py` | FR-07, AC-06 |
| Monthly appliance-wise measured energy/cost table | `src/summary.py` | FR-07 |
| Orchestration | `src/pipeline.py` | M-checkpoints |

## Quick start (works without any real data)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
make test            # or: python -m pytest -q
make run-synth       # generate SYNTHETIC data, run full pipeline (~1-2 min)
make predict-demo    # one appliance estimate as JSON
```
No `make` (Windows)? Run the commands inside the `Makefile` directly.

Outputs (per run name):
`reports/<run>/` -> `data_quality.md`, `evaluation.md`, `metrics.json`, `appliance_energy_summary.csv`, `figures/`
`artifacts/<run>/` -> `metadata.json` (+ `model.joblib` only when ML beat the baseline)
`data/processed/<run>/` -> sessions + features CSV

## Using real REFIT data

See [`data/README.md`](data/README.md). In short: put a REFIT house CSV in `data/raw/`, **verify the channel mapping** in `configs/refit_house1.json`, set `mapping_verified`, record dataset version/licence, then:

```bash
make run-refit
# very large files: restrict candidates to speed things up
python -m src.pipeline --config configs/refit_house1.json --raw data/raw/CLEAN_House1.csv \
    --tariff configs/tariff.json --run-name refit --families gradient_boosting hybrid_gradient_boosting
```

## Predicting

```bash
# Theoretical/ML estimate for a planned or past session
python -m src.predict --artifact-dir artifacts/refit --appliance washing_machine --duration-h 1.2 \
    --start 2026-10-24T19:30 --tariff configs/tariff.json

# Measured energy from a power trace (CSV columns: timestamp, power_w)
python -m src.predict --artifact-dir artifacts/refit --appliance fridge --trace my_trace.csv --tariff configs/tariff.json
```
Or from Python (e.g. your backend):
```python
from src.predict import load_artifacts, estimate_appliance
from src.tariff import load_tariff
art = load_artifacts("artifacts/refit")
result = estimate_appliance("fridge", duration_h=0.5, art=art, tariff=load_tariff("configs/tariff.json"))
```

### Output contract (schema v1.0)
`status` (`ok | warning | error`), `errors[]`, `warnings[]`, `appliance`, `units`, `inputs`, `energy_kwh`,
`energy_provenance` (`measured | calculated | ml_estimated`), `method`, `baseline_energy_kwh`, `ml_energy_kwh`,
`cost`, `currency`, `tariff_is_placeholder`, `assumptions[]`, `model_version`, `generated_at_utc`, `disclaimer`.
Invalid input returns `status="error"` with a reason; there are no silent defaults.

## Design decisions worth knowing

- **Target** = measured session energy (kWh) from integrating power. Sessions are found with per-appliance ON thresholds, merge gaps and minimum durations (`configs/*.json`). Fridges/freezers therefore produce one session per compressor cycle.
- **No target leakage:** the session's own measured mean/peak power are *not* features. An optional history feature uses only the mean power of the appliance's *previous* sessions.
- **Honest model choice:** if ML does not beat the baseline by `min_improvement_pct` on validation, the baseline is deployed and `model.joblib` is not written (PRISM M9).
- **Test set is touched once**, after model selection. Final models are refit on train+validation.
- **Gaps are never interpolated:** intervals longer than `max_gap_s` contribute zero energy and are reported as lower coverage.
- **Tariff** in `configs/tariff.json` is a **placeholder**. Replace it with your electricity provider's published tariff (a slab example is in `configs/tariff_slab_example.json`).

## Known limitations / open TBDs (from the PRISM risk register)

- R-01 rated power != real operating power - rated values here are assumptions until replaced by nameplate data.
- R-02 ON thresholds need per-appliance validation (use `notebooks/01_data_audit.ipynb`, section 4).
- R-03 REFIT is UK data; validate on Indian household data before any production claim.
- R-05 the synthetic demo is not evidence of real-world accuracy.
- Held-out-by-household evaluation needs more than one house; not included yet.
- Latency targets, deployment, API framework, owners and approvals: **TBD**.

## Repository layout
```
configs/        appliance mapping/thresholds/rated power, tariff examples
data/           README (source/licence/schema); raw/ & processed/ are git-ignored
notebooks/      01_data_audit.ipynb
src/            ingest, preprocess, energy, baseline, features, split, models, train, evaluate, summary, tariff, predict, pipeline
scripts/        generate_synthetic_data.py, make_audit_notebook.py
tests/          unit, data-validation, model-contract and E2E tests
artifacts/      metadata.json per run (model.joblib ignored by default)
reports/        generated per run (git-ignored except .gitkeep)
.github/        CI (tests + synthetic smoke run)
```
