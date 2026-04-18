# Runner Injury Risk Prediction 🇬🇧 [🇮🇹](README_IT.md)

![Test & Coverage](https://github.com/aleattene/runner-injury-risk-prediction/actions/workflows/test.yml/badge.svg)
![Lint & Format](https://github.com/aleattene/runner-injury-risk-prediction/actions/workflows/lint.yml/badge.svg)
[![codecov](https://codecov.io/gh/aleattene/runner-injury-risk-prediction/graph/badge.svg?token=9PXXMFOPE2)](https://codecov.io/gh/aleattene/runner-injury-risk-prediction)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)
![SHAP](https://img.shields.io/badge/SHAP-Interpretability-purple)
![License](https://img.shields.io/badge/License-MIT-blue)
![Last Commit](https://img.shields.io/github/last-commit/aleattene/runner-injury-risk-prediction)

A **Data Science** project that replicates and extends the study by Lovdal et al. (2021),
predicting injury risk in competitive runners using machine learning on training load
time-series data — covering EDA, modeling, SHAP interpretability, and fairness analysis.

---

## Research Context

This project builds on the following academic research:

> **Lovdal, S., Den Hartigh, R.J.R., & Azzopardi, G.** (2021).
> *Injury Prediction in Competitive Runners With Machine Learning.*
> International Journal of Sports Physiology and Performance, 16(10), 1522-1531.
> DOI: [10.1123/ijspp.2020-0518](https://doi.org/10.1123/ijspp.2020-0518)

**Dataset:** Training logs from 74 elite Dutch middle/long-distance runners (2012-2019),
combining GPS-measured training load with subjective wellness metrics.
Available on [Kaggle](https://www.kaggle.com/datasets/shashwatwork/injury-prediction-for-competitive-runners/data)
and [DataverseNL](https://doi.org/10.34894/UWU9PV).

**Paper benchmarks:** AUC-ROC 0.724 (day approach) and 0.678 (week approach) with bagged XGBoost.

---

## Business Questions

This analysis addresses key questions for sports scientists, coaches, and team physicians:

1. **Can we predict injuries before they happen?** Using the previous 7 days of training data, how accurately can ML models flag injury risk?
2. **Which training patterns increase risk?** What are the most predictive features — intensity spikes, volume, recovery deficits?
3. **Daily vs weekly monitoring — which is better?** Does fine-grained daily data outperform weekly aggregations for prediction?
4. **Is the model fair across athlete types?** Does prediction accuracy differ for high-volume vs low-volume athletes, or for previously injured vs uninjured runners?
5. **How should coaches use these predictions?** What decision threshold balances catching injuries vs false alarms?

---

## Key Findings

- **Week approach outperforms day approach** — AUC-ROC 0.624 (week) vs 0.588 (day), winning on 5 out of 6 metrics. This reverses the paper's finding, likely because a single XGBoost model benefits more from aggregated, less noisy weekly features than from fine-grained daily data.
- **92% of paper benchmark reached** — the week model achieves 92.0% of the paper's AUC-ROC (0.678), while the day model reaches 81.2% of the paper's 0.724. The gap is primarily attributable to our single model vs the paper's 100-bag ensemble.
- **Extreme class imbalance remains the core challenge** — with only ~1.2% injury rate, the day model produces zero recall at the optimal threshold (0.63), while the week model achieves modest recall (6.8%) at threshold 0.64.
- **SHAP reveals consistent injury risk factors across both approaches** — training volume (total km), subjective markers (perceived exertion, training success), and high-intensity load (zone distributions) drive predictions in both temporal frameworks.
- **No systematic fairness bias detected** — proxy-group analysis (volume, injury history, data density) shows similar performance profiles across athlete subgroups, though the absence of demographic data limits this assessment.

---

## Dataset

| Approach | Rows | Columns | Target | Window |
|----------|------|---------|--------|--------|
| **Day** | 42,766 | 73 | Binary (0/1) — 1.36% injury rate | 7 days x 10 features |
| **Week** | 42,798 | 72 | Continuous (0.0-1.5+) → binarized at 0.5 | 3 weeks x 22 features + 3 ratios |

**74 athletes** — 27 women, 47 men — national/international level, 800m to marathon.

### Feature categories

| Category | Day features | Week features |
|----------|-------------|---------------|
| Volume | total km, nr. sessions | total kms, max km one day, nr. sessions |
| Intensity | km Z3-4, km Z5-T1-T2, km sprinting | km Z3-4, km Z5-T1-T2, nr. tough sessions, interval days |
| Cross-training | strength training, hours alternative | nr. strength trainings, total hours alternative |
| Subjective wellness | perceived exertion, training success, recovery | avg/min/max exertion, training success, recovery |
| Load progression | — | rel total kms week-to-week ratios |

---

## Project Structure

```text
runner-injury-risk-prediction/
├── src/
│   ├── config.py                      # Paths, seed, constants
│   ├── data_loading.py                # CSV loading + column renaming
│   ├── preprocessing/                 # Sentinel handling, splitting, binarization
│   ├── modeling/                      # Model factory, training, evaluation
│   ├── interpretability/              # SHAP analysis
│   ├── fairness/                      # Proxy group audit
│   └── utils/                         # Logging, plotting, reproducibility
├── notebooks/
│   ├── 01_eda_day_approach.ipynb
│   ├── 02_eda_week_approach.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_modeling_day.ipynb
│   ├── 05_modeling_week.ipynb
│   ├── 06_interpretability.ipynb
│   ├── 07_fairness_analysis.ipynb
│   └── 08_comparison_summary.ipynb
├── reports/
│   ├── REPORT.md                      # Executive report
│   └── figures/                       # All exported charts
├── data/raw/                          # Original CSVs (gitignored)
├── data_sample/                       # Synthetic data for tests (committed)
├── tests/                             # pytest suite (≥85% coverage)
├── docs/ADR.md                        # Architecture Decision Records
└── .github/workflows/                 # CI/CD (test + lint)
```

---

## Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.13 |
| ML framework | scikit-learn, XGBoost |
| Interpretability | SHAP |
| Imbalance handling | imbalanced-learn (SMOTE), class weighting |
| Data manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Notebook | Jupyter |
| Testing | pytest, pytest-cov |
| Formatting & linting | black, ruff |
| CI/CD | GitHub Actions |

---

## Reproducibility

```bash
# 1. Install dependencies
pip install pip-tools
pip-compile requirements.in
pip-compile requirements-dev.in
pip-sync requirements-dev.txt

# 2. Install pre-commit hooks
pre-commit install

# 3. Run the test suite
pytest
pytest --cov=src --cov-report=term-missing  # with coverage

# 4. Run notebooks in order
jupyter notebook notebooks/01_eda_day_approach.ipynb
```

---

## Report & Dashboard

- [Executive Report](reports/REPORT.md)
- [EDA Notebook](notebooks/01_eda_day_approach.ipynb)
- Looker Studio Dashboard — *deferred*



## Privacy & Ethics

- **Athlete IDs are masked** — no personally identifiable information in the dataset
- **Ethics approval** — original study conducted under Declaration of Helsinki
- **Fairness analysis** — model performance evaluated across proxy athlete groups
- **Limitations discussed** — no demographic attributes available for full equity audit

---

## Author

Alessandro Attene
