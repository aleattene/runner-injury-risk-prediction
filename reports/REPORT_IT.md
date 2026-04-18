# Report Esecutivo — Previsione del Rischio di Infortunio nei Runner 🇮🇹 [🇬🇧](REPORT.md)

**Progetto**: Replica ed estensione di Lovdal, Den Hartigh & Azzopardi (2021)
**Data**: Aprile 2026
**Autore**: Alessandro Attene

---

## Abstract

Questo progetto replica ed estende la metodologia di machine learning
di Lovdal et al. (2021) per prevedere il rischio di infortunio nei runner
agonisti utilizzando dati di carico di allenamento basati su GPS di 74 atleti
olandesi d'élite (2012–2019). Vengono confrontati due approcci temporali —
**giornaliero** (finestre mobili di 7 giorni) e **settimanale** (aggregazioni
di 3 settimane) — utilizzando un singolo modello XGBoost ottimizzato per
approccio. La pipeline garantisce una rigorosa separazione a livello di atleta
(GroupKFold / GroupShuffleSplit) per prevenire il data leakage, applica
interpretabilità basata su SHAP per collegare le previsioni a concetti di
scienza dello sport e include un audit di equità su gruppi proxy demografici.

Il nostro miglior risultato è un **AUC-ROC di 0.624** (approccio settimanale),
raggiungendo il **92.0%** del benchmark XGBoost con bagging dell'articolo (0.678).
L'approccio giornaliero raggiunge un AUC-ROC di 0.588 (81.2% dello 0.724
dell'articolo). Entrambi i modelli faticano con lo sbilanciamento estremo
delle classi (~1.2% tasso di infortunio), producendo basso recall e precisione
alle soglie ottimali. Il divario con l'articolo è principalmente attribuibile
all'uso di un singolo modello XGBoost rispetto all'ensemble a 100 bag
dell'articolo, e a una separazione più rigorosa a livello di atleta.

---

## Indice

1. [Contesto della Ricerca](#1-contesto-della-ricerca)
2. [Dataset](#2-dataset)
3. [Metodologia](#3-metodologia)
4. [Risultati](#4-risultati)
5. [Confronto con i Benchmark dell'Articolo](#5-confronto-con-i-benchmark-dellarticolo)
6. [Interpretabilità (SHAP)](#6-interpretabilità-shap)
7. [Analisi di Equità](#7-analisi-di-equità)
8. [Limitazioni](#8-limitazioni)
9. [Sviluppi Futuri](#9-sviluppi-futuri)
10. [Riferimenti](#10-riferimenti)

---

## 1. Contesto della Ricerca

Gli infortuni da sovraccarico sono la preoccupazione principale negli sport di
endurance. Il monitoraggio del carico di allenamento — che combina metriche GPS,
punteggi soggettivi di benessere e indicatori fisiologici — è ampiamente
utilizzato per gestire il rischio di infortunio. Lovdal et al. (2021) hanno
dimostrato che modelli di machine learning addestrati su feature giornaliere e
settimanali possono prevedere il rischio di infortunio del giorno o della
settimana successiva con moderata capacità discriminativa (AUC-ROC 0.724 per
il giornaliero, 0.678 per il settimanale) utilizzando un ensemble XGBoost
con bagging (100 bag).

Questo progetto replica la loro metodologia principale con semplificazioni
deliberate: un singolo modello XGBoost ottimizzato (senza bagging),
separazione più rigorosa a livello di atleta e interpretabilità trasparente
basata su SHAP — rendendo la pipeline più accessibile e riproducibile pur
accettando un compromesso sulle performance.

---

## 2. Dataset

| Proprietà | Approccio giornaliero | Approccio settimanale |
|---|---|---|
| Fonte | Lovdal et al. (2021) — runner olandesi d'élite, 2012–2019 | Come l'approccio giornaliero |
| Atleti | 74 (ID mascherati) | 74 (ID mascherati) |
| Righe totali | 42.766 | 42.798 |
| Feature | 70 (7 giorni x 10 metriche) | 69 (3 settimane x 22 metriche + 3 rapporti km) |
| Target | Binario (0/1) | Continuo → binarizzato a 0.5 (ADR-002) |
| Tasso di infortunio (test set) | 1.21% | 1.19% |
| Training set | 36.584 righe | 36.588 righe |
| Test set | 6.182 righe | 6.210 righe |

**Feature principali**: distanza totale (km), distribuzioni per zone di
allenamento (z1–z5, t1–t2), sforzo percepito, stato di recupero, sessioni
di allenamento della forza e indicatori soggettivi di benessere. L'approccio
settimanale include inoltre i rapporti acuto-cronico dei km (settimana 0/1,
settimana 0/2, settimana 0/(1+2)).

**Gestione sentinel** (ADR-007): i dati originali usano -0.01 per indicare
i giorni di riposo (nessun allenamento). Li abbiamo sostituiti con 0.0 — il
valore semanticamente corretto per "nessuna attività".

---

## 3. Metodologia

### 3.1 Preprocessing

- **Sostituzione sentinel**: -0.01 → 0.0 per i giorni di riposo (ADR-007)
- **Binarizzazione target settimanale**: valori continui binarizzati alla soglia 0.5 (ADR-002)
- **Split train/test**: GroupShuffleSplit per ID Atleta con split approssimativo
  80/20 a livello di gruppo-atleta, garantendo che tutte le osservazioni di un
  atleta appartengano a un solo split; le proporzioni a livello di riga possono
  differire perché gli atleti contribuiscono con un numero diseguale di
  osservazioni (ADR-006)
- **Scaling delle feature**: StandardScaler adattato solo sui dati di training,
  applicato sia al train che al test set

![Panoramica preprocessing](figures/preprocessing/03_train_test_athlete_assignment.png)

### 3.2 Pipeline di modellazione

Tre famiglie di modelli sono state confrontate tramite cross-validation
GroupKFold (k=5):

| Modello | AUC-ROC giornaliero (CV) | AUC-ROC settimanale (CV) |
|---|---|---|
| Logistic Regression | baseline | baseline |
| Random Forest | moderato | moderato |
| **XGBoost** | **migliore** | **migliore** |

XGBoost è stato selezionato come modello con le migliori performance per
entrambi gli approcci e ulteriormente ottimizzato tramite RandomizedSearchCV
(30 iterazioni, GroupKFold).

**Gestione dello sbilanciamento delle classi** (ADR-003):
- Primaria: `scale_pos_weight` (XGBoost) / `class_weight='balanced'` (LogReg, RF)
- Soglia di classificazione ottimizzata sulle previsioni di training
  (massimizzando F1) e applicata al test set — nessun leakage del test nella
  selezione della soglia

### 3.3 Strategia di valutazione

- **Metrica primaria**: AUC-ROC (capacità di ranking indipendente dalla soglia)
- **Metriche secondarie**: AUC-PR (sensibile allo sbilanciamento), recall,
  precisione, F1 (classe positiva), Brier score (calibrazione)
- **Mai l'accuratezza**: inappropriata con ~1.2% di tasso positivo

### 3.4 Interpretabilità

- SHAP TreeExplainer per XGBoost, LinearExplainer per Logistic Regression
- Summary plot, dependence plot, waterfall plot per previsioni individuali
- Aggregazione a livello di concetto: rimozione dei prefissi temporali per
  confrontare l'importanza delle feature base tra gli approcci

### 3.5 Audit di equità

- Nessun dato demografico disponibile (età, sesso, ecc.) — utilizzati gruppi proxy
- Tre strategie di raggruppamento: volume di allenamento (km), storia infortuni,
  densità dati
- Metriche per gruppo e rapporti di disparità calcolati per entrambi gli approcci

---

## 4. Risultati

### 4.1 Metriche sul test set

Entrambi i modelli valutati alle soglie ottimali selezionate sul training
(giornaliero: 0.63, settimanale: 0.64).

| Metrica | XGBoost giornaliero | XGBoost settimanale | Vincitore |
|---|---|---|---|
| AUC-ROC | 0.5878 | **0.6237** | Settimanale |
| AUC-PR | 0.0146 | **0.0194** | Settimanale |
| Recall | 0.0000 | **0.0676** | Settimanale |
| Precisione | 0.0000 | **0.0278** | Settimanale |
| F1 | 0.0000 | **0.0394** | Settimanale |
| Brier score | **0.1867** | 0.1887 | Giornaliero |

**L'approccio settimanale vince su 5 delle 6 metriche.** Questo differisce
dal risultato dell'articolo secondo cui l'approccio giornaliero supera quello
settimanale — probabilmente perché l'ensemble a 100 bag dell'articolo sfrutta
meglio le feature giornaliere a grana fine, mentre un singolo modello XGBoost
beneficia maggiormente dalle feature aggregate e meno rumorose dell'approccio
settimanale e dai rapporti acuto-cronico.

L'approccio giornaliero raggiunge zero recall alla soglia 0.63, il che
significa che non classifica nessun campione del test come infortunato — i
punteggi di probabilità appresi dal modello non superano la soglia elevata
per nessun caso vero positivo. Questo evidenzia l'estrema difficoltà del
task giornaliero con solo l'1.21% di tasso positivo e un'architettura a
singolo modello.

![Confronto metriche](figures/comparison/08_metrics_comparison_day_vs_week.png)

### 4.2 Curve ROC e PR

![Curve ROC](figures/comparison/08_roc_curves_combined.png)

![Curve PR](figures/comparison/08_pr_curves_combined.png)

---

## 5. Confronto con i Benchmark dell'Articolo

| Approccio | Nostro (singolo XGBoost) | Articolo (XGBoost con bagging) | % dell'articolo |
|---|---|---|---|
| Giornaliero | 0.5878 | 0.7240 | 81.2% |
| Settimanale | 0.6237 | 0.6780 | 92.0% |

![Confronto benchmark articolo](figures/comparison/08_paper_benchmark_comparison.png)

### Perché i nostri risultati differiscono

1. **Nessun ensemble con bagging**: l'articolo media le previsioni su 100
   modelli XGBoost addestrati indipendentemente, riducendo significativamente
   la varianza. Il nostro singolo modello è più suscettibile all'overfitting
   e al rumore.

2. **Separazione più rigorosa degli atleti (ADR-006)**: applichiamo GroupKFold e
   GroupShuffleSplit per ID Atleta lungo l'intera pipeline. La strategia esatta
   di split dell'articolo potrebbe differire — una separazione più rigorosa
   tipicamente riduce le performance apparenti prevenendo il leakage
   di informazione.

3. **Budget di iperparametri**: RandomizedSearchCV con 30 iterazioni vs.
   la ricerca probabilmente più esaustiva dell'articolo o valori predefiniti
   informati dal dominio.

4. **Gestione sentinel (ADR-007)**: sostituire -0.01 con 0.0 modifica le
   soglie apprese sulle feature rispetto al mantenimento del sentinel originale.

5. **Binarizzazione target settimanale (ADR-002)**: i dettagli implementativi
   nella conversione da continuo a binario possono introdurre piccole differenze.

Nonostante questi divari, l'approccio settimanale raggiunge il **92% del
benchmark dell'articolo**, validando la metodologia principale di questa replica.

---

## 6. Interpretabilità (SHAP)

### 6.1 Feature principali per approccio

L'analisi SHAP rivela quali feature di allenamento guidano le previsioni
del rischio di infortunio:

**Approccio giornaliero** — dominato dalle metriche giornaliere recenti:
- Km totali e sforzo percepito del Giorno 0 e Giorno 1, successo allenamento
- Cattura i picchi di carico acuto e le risposte soggettive immediate

**Approccio settimanale** — dominato dalle aggregazioni settimanali:
- Sforzo massimo della Settimana 0 e Settimana 1, km totali ad alta intensità (z3-z5)
- I rapporti acuto-cronico dei km forniscono contesto sulla progressione del carico

![Importanza feature SHAP](figures/comparison/08_shap_importance_day_vs_week.png)

### 6.2 Sovrapposizione a livello di concetto

Rimuovendo i prefissi temporali si rivela che entrambi gli approcci si basano
sugli stessi concetti sottostanti di scienza dello sport:

- **Volume di allenamento** (km totali) — costantemente importante in entrambi
- **Indicatori soggettivi** (sforzo percepito, successo allenamento, recupero) —
  segnale forte in entrambi gli approcci
- **Carico ad alta intensità** (distribuzioni per zone) — rilevante in entrambi

Questa coerenza valida il fatto che i modelli catturano fattori reali di
rischio di infortunio, non artefatti temporali.

![SHAP summary — giornaliero](figures/interpretability/06_shap_summary_day.png)

![SHAP summary — settimanale](figures/interpretability/06_shap_summary_week.png)

### 6.3 Previsioni individuali

I waterfall plot per veri positivi, veri negativi e falsi negativi rivelano
come il modello pesa le feature per singoli atleti:

| | Giornaliero | Settimanale |
|---|---|---|
| Vero positivo | ![](figures/interpretability/06_waterfall_day_tp.png) | ![](figures/interpretability/06_waterfall_week_tp.png) |
| Falso negativo | ![](figures/interpretability/06_waterfall_day_fn.png) | ![](figures/interpretability/06_waterfall_week_fn.png) |

---

## 7. Analisi di Equità

### 7.1 Approccio

Senza attributi demografici (età, sesso, nazionalità), abbiamo costruito tre
strategie di raggruppamento proxy per verificare l'equità del modello:

1. **Volume di allenamento**: atleti raggruppati per km totali (basso / medio / alto)
2. **Storia infortuni**: atleti raggruppati per conteggio storico degli infortuni (basso / alto)
3. **Densità dati**: atleti raggruppati per numero di osservazioni (sparso / denso)

Per ogni gruppo, abbiamo calcolato AUC-ROC, recall, precisione e F1, poi
calcolato i rapporti di disparità (metrica del gruppo / metrica complessiva)
per individuare divari sistematici nelle performance.

### 7.2 Risultati

- Entrambi gli approcci mostrano **profili di equità simili** per tutte e tre
  le strategie di raggruppamento
- Nessun bias sistematico verso atleti ad alto volume o frequentemente infortunati
- Una certa variazione nelle performance tra i gruppi è attesa date le piccole
  dimensioni dei sottogruppi nel test set

![Confronto equità](figures/fairness/07_day_vs_week_disparity_comparison.png)

### 7.3 Limitazioni dell'audit di equità

- I gruppi proxy non sono attributi protetti — bias sistematici per età,
  sesso o etnia non possono essere rilevati né esclusi
- La piccola coorte di atleti (74 totali, ~15 nel test set) limita la potenza
  statistica per l'analisi dei sottogruppi
- I risultati vanno interpretati come esplorativi, non come valutazione di conformità

---

## 8. Limitazioni

1. **Nessun ensemble con bagging**: l'architettura a 100 bag dell'articolo
   raggiunge AUC-ROC più elevati tramite riduzione della varianza. Il nostro
   design a singolo modello è più semplice ma meno performante.

2. **Piccola coorte di atleti**: 74 atleti totali, con circa 15 nel test set.
   I risultati potrebbero non generalizzarsi a popolazioni più ampie o diverse.

3. **Nessuna validazione esterna**: la valutazione utilizza uno split held-out
   dello stesso dataset. La vera generalizzabilità richiede test su dati
   indipendenti.

4. **Grave sbilanciamento delle classi**: ~1.2% di tasso positivo rende la
   calibrazione inaffidabile e le metriche dipendenti dalla soglia (precisione,
   recall, F1) altamente sensibili alla scelta della soglia.

5. **Nessuna validazione temporale**: GroupKFold previene il leakage tra atleti
   ma non testa la capacità del modello di prevedere infortuni futuri
   utilizzando rigorosamente dati passati (TimeSeriesSplit).

6. **Nessuna equità demografica**: senza attributi protetti, l'analisi per
   gruppi proxy fornisce garanzie limitate di equità.

7. **Singolo dataset, singolo sport**: i risultati sono specifici per runner
   olandesi d'élite (2012–2019). La trasferibilità ad altre popolazioni,
   sport o sistemi di monitoraggio è ignota.

---

## 9. Sviluppi Futuri

1. **Ensemble XGBoost con bagging**: implementare l'architettura a 100 bag
   per colmare il divario con i benchmark dell'articolo e quantificare la
   riduzione della varianza
2. **Deep learning**: modelli LSTM o Transformer per pattern in serie temporali,
   catturando dipendenze temporali a lungo raggio
3. **Raccolta dati demografici**: abilitare audit di equità adeguati sugli
   attributi protetti
4. **Validazione esterna**: testare su popolazioni di runner indipendenti
   (diversi paesi, livelli agonistici, sistemi di monitoraggio)
5. **Design di studio prospettico**: addestrare su dati storici (2012–2017),
   validare su stagioni future (2018–2019) per generalizzazione temporale
6. **Miglioramento della calibrazione**: Platt scaling o regressione isotonica
   per produrre probabilità di infortunio affidabili
7. **Dashboard interattiva**: deployment per monitoraggio in tempo reale da
   parte di allenatori e scienziati dello sport

---

## 10. Riferimenti

- Lovdal, S. S., Den Hartigh, R. J. R., & Azzopardi, G. (2021).
  Injury prediction in competitive runners with machine learning.
  *International Journal of Sports Physiology and Performance*, 16(10),
  1522–1531. [DOI: 10.1123/ijspp.2020-0518](https://doi.org/10.1123/ijspp.2020-0518)

---

## Appendice — Indice delle Figure

Tutte le figure sono salvate in `reports/figures/` organizzate per fase di analisi:

| Fase | Cartella | Conteggio |
|---|---|---|
| EDA | `figures/eda/` | 16 |
| Preprocessing | `figures/preprocessing/` | 4 |
| Modellazione | `figures/modeling/` | 8 |
| Interpretabilità | `figures/interpretability/` | 21 |
| Equità | `figures/fairness/` | 15 |
| Confronto | `figures/comparison/` | 5 |
| **Totale** | | **69** |

### Architecture Decision Records

Le decisioni progettuali chiave sono riassunte di seguito; il documento ADR dedicato è disponibile in [docs/ADR_IT.md](../docs/ADR_IT.md):

| ADR | Decisione |
|---|---|
| ADR-001 | Solo Pandas, niente DuckDB |
| ADR-002 | Target settimanale binarizzato a 0.5 |
| ADR-003 | Class weighting come strategia primaria per lo sbilanciamento |
| ADR-004 | Nessuna variabile d'ambiente |
| ADR-005 | Strategia bilingue English-first |
| ADR-006 | GroupKFold (non TimeSeriesSplit) |
| ADR-007 | Sentinel -0.01 sostituito con 0.0 |
