# Previsione del Rischio di Infortunio nei Runner 🇮🇹 [🇬🇧](README.md)

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

Un progetto di **Data Science** che replica ed estende lo studio di Lovdal et al. (2021),
prevedendo il rischio di infortunio nei runner agonisti tramite machine learning su dati
di carico di allenamento in serie temporali — coprendo EDA, modellazione, interpretabilità
SHAP e analisi di equità.

---

## Contesto della Ricerca

Questo progetto si basa sulla seguente ricerca accademica:

> **Lovdal, S., Den Hartigh, R.J.R., & Azzopardi, G.** (2021).
> *Injury Prediction in Competitive Runners With Machine Learning.*
> International Journal of Sports Physiology and Performance, 16(10), 1522-1531.
> DOI: [10.1123/ijspp.2020-0518](https://doi.org/10.1123/ijspp.2020-0518)

**Dataset:** log di allenamento di 74 runner olandesi d'élite di mezzofondo/fondo (2012-2019),
che combinano carico di allenamento misurato tramite GPS con metriche soggettive di benessere.
Disponibile su [Kaggle](https://www.kaggle.com/datasets/shashwatwork/injury-prediction-for-competitive-runners/data)
e [DataverseNL](https://doi.org/10.34894/UWU9PV).

**Benchmark dell'articolo:** AUC-ROC 0.724 (approccio giornaliero) e 0.678 (approccio settimanale) con XGBoost in ensemble bagging.

---

## Domande di Business

Questa analisi risponde a domande chiave per scienziati dello sport, allenatori e medici sportivi:

1. **Possiamo prevedere gli infortuni prima che accadano?** Usando i 7 giorni precedenti di dati di allenamento, quanto accuratamente i modelli ML possono segnalare il rischio di infortunio?
2. **Quali pattern di allenamento aumentano il rischio?** Quali sono le feature più predittive — picchi di intensità, volume, deficit di recupero?
3. **Monitoraggio giornaliero vs settimanale — quale è migliore?** I dati giornalieri a grana fine superano le aggregazioni settimanali nella previsione?
4. **Il modello è equo tra tipi di atleta?** L'accuratezza della previsione varia tra atleti ad alto vs basso volume, o tra runner precedentemente infortunati vs non infortunati?
5. **Come dovrebbero usare queste previsioni gli allenatori?** Quale soglia di decisione bilancia la cattura degli infortuni vs i falsi allarmi?

---

## Risultati Principali

- **L'approccio settimanale supera quello giornaliero** — AUC-ROC 0.624 (settimanale) vs 0.588 (giornaliero), vincendo su 5 delle 6 metriche. Questo inverte il risultato dell'articolo, probabilmente perché un singolo modello XGBoost beneficia maggiormente da feature settimanali aggregate e meno rumorose rispetto ai dati giornalieri a grana fine.
- **Raggiunto il 92% del benchmark dell'articolo** — il modello settimanale raggiunge il 92.0% dell'AUC-ROC dell'articolo (0.678), mentre il modello giornaliero raggiunge l'81.2% dello 0.724 dell'articolo. Il divario è principalmente attribuibile al nostro singolo modello vs l'ensemble a 100 bag dell'articolo.
- **Lo sbilanciamento estremo delle classi resta la sfida principale** — con solo ~1.2% di tasso di infortunio, il modello giornaliero produce zero recall alla soglia ottimale (0.63), mentre il modello settimanale raggiunge un modesto recall (6.8%) alla soglia 0.64.
- **SHAP rivela fattori di rischio coerenti tra entrambi gli approcci** — volume di allenamento (km totali), indicatori soggettivi (sforzo percepito, successo dell'allenamento) e carico ad alta intensità (distribuzioni per zone) guidano le previsioni in entrambi i framework temporali.
- **Nessun bias sistematico di equità rilevato** — l'analisi per gruppi proxy (volume, storia infortuni, densità dati) mostra profili di performance simili tra i sottogruppi di atleti, sebbene l'assenza di dati demografici limiti questa valutazione.

---

## Dataset

| Approccio | Righe | Colonne | Target | Finestra |
|-----------|-------|---------|--------|----------|
| **Giornaliero** | 42.766 | 73 | Binario (0/1) — 1.36% tasso infortuni | 7 giorni x 10 feature |
| **Settimanale** | 42.798 | 72 | Continuo (0.0-1.5+) → binarizzato a 0.5 | 3 settimane x 22 feature + 3 rapporti |

**74 atleti** — 27 donne, 47 uomini — livello nazionale/internazionale, dagli 800m alla maratona.

### Categorie di feature

| Categoria | Feature giornaliere | Feature settimanali |
|-----------|--------------------|--------------------|
| Volume | km totali, nr. sessioni | km totali, km max in un giorno, nr. sessioni |
| Intensità | km Z3-4, km Z5-T1-T2, km sprint | km Z3-4, km Z5-T1-T2, nr. sessioni intense, giorni intervalli |
| Cross-training | allenamento forza, ore alternative | nr. allenamenti forza, ore totali alternative |
| Benessere soggettivo | sforzo percepito, successo allenamento, recupero | sforzo medio/min/max, successo allenamento, recupero |
| Progressione del carico | — | rapporti km totali relativi settimana-su-settimana |

---

## Struttura del Progetto

```text
runner-injury-risk-prediction/
├── src/
│   ├── config.py                      # Percorsi, seed, costanti
│   ├── data_loading.py                # Caricamento CSV + rinomina colonne
│   ├── preprocessing/                 # Gestione sentinel, split, binarizzazione
│   ├── modeling/                      # Factory modelli, addestramento, valutazione
│   ├── interpretability/              # Analisi SHAP
│   ├── fairness/                      # Audit gruppi proxy
│   └── utils/                         # Logging, plotting, riproducibilità
├── notebooks/
│   ├── 01_eda_day_approach_IT.ipynb
│   ├── 02_eda_week_approach_IT.ipynb
│   ├── 03_preprocessing_IT.ipynb
│   ├── 04_modeling_day_IT.ipynb
│   ├── 05_modeling_week_IT.ipynb
│   ├── 06_interpretability_IT.ipynb
│   ├── 07_fairness_analysis_IT.ipynb
│   └── 08_comparison_summary_IT.ipynb
├── reports/
│   ├── REPORT_IT.md                   # Report esecutivo
│   └── figures/                       # Tutti i grafici esportati
├── data/raw/                          # CSV originali (gitignored)
├── data_sample/                       # Dati sintetici per i test (committati)
├── tests/                             # Suite pytest (≥85% copertura)
├── docs/ADR_IT.md                     # Architecture Decision Records
└── .github/workflows/                 # CI/CD (test + lint)
```

---

## Stack Tecnologico

| Componente | Tecnologia |
|------------|------------|
| Linguaggio | Python 3.13 |
| Framework ML | scikit-learn, XGBoost |
| Interpretabilità | SHAP |
| Gestione sbilanciamento | imbalanced-learn (SMOTE), class weighting |
| Manipolazione dati | Pandas, NumPy |
| Visualizzazione | Matplotlib, Seaborn |
| Notebook | Jupyter |
| Testing | pytest, pytest-cov |
| Formattazione e linting | black, ruff |
| CI/CD | GitHub Actions |

---

## Riproducibilità

```bash
# 1. Installare le dipendenze
pip install pip-tools
pip-compile requirements.in
pip-compile requirements-dev.in
pip-sync requirements-dev.txt

# 2. Installare i pre-commit hook
pre-commit install

# 3. Eseguire la suite di test
pytest
pytest --cov=src --cov-report=term-missing  # con copertura

# 4. Eseguire i notebook in ordine
jupyter notebook notebooks/01_eda_day_approach_IT.ipynb
```

---

## Report e Dashboard

- [Report Esecutivo](reports/REPORT_IT.md)
- [Notebook EDA](notebooks/01_eda_day_approach_IT.ipynb)
- Dashboard Looker Studio — *in programma*



## Privacy ed Etica

- **Gli ID degli atleti sono mascherati** — nessuna informazione personale identificabile nel dataset
- **Approvazione etica** — lo studio originale è stato condotto secondo la Dichiarazione di Helsinki
- **Analisi di equità** — le performance del modello sono state valutate su gruppi proxy di atleti
- **Limiti discussi** — nessun attributo demografico disponibile per un audit completo di equità

---

## Autore

Alessandro Attene
