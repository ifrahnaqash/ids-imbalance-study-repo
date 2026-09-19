# Methodology

This document summarises the experimental pipeline. For full derivations, statistical
validation, and discussion, see the accompanying paper (citation in the root README).

## Datasets

Eight publicly available IDS benchmark datasets, spanning 2009–2023 and both enterprise
and IoT domains, totalling over 67 million network flows:

| Dataset | Year | Domain | Flows |
|---|---|---|---|
| NSL-KDD | 2009 | Enterprise | 148,517 |
| UNSW-NB15 | 2015 | Enterprise | 257,673 |
| CIC-IDS2017 | 2017 | Enterprise | 2,830,743 |
| HIKARI-2021 | 2021 | Mixed/Encrypted | 555,278 |
| BoT-IoT | 2019 | IoT | 3,668,522 |
| ToN-IoT | 2021 | IoT/IIoT | 211,043 |
| CSE-CIC-IDS2018 | 2018 | Enterprise | 16,232,943 |
| CICIoT2023 | 2023 | IoT | 45,019,234 |

Raw dataset files are **not** included in this repository (see root README for sourcing
and licensing). Scripts expect a local `DATA_ROOT` directory configured in
`src/phase1_imbalance_characterization.py`.

## Phase 1 — Imbalance Characterization (`phase1_imbalance_characterization.py`)

For each dataset, computes a structured *Imbalance Profile*:

- **Distribution metrics:** Imbalance Ratio (IR), normalized Shannon entropy, Gini
  impurity, benign traffic percentage.
- **Minority class metrics:** Global Deviation Ratio (GDR — a class-vs-global
  separability measure), within-class feature variance.
- **Temporal spread:** fraction of the capture window occupied by each class, used to
  detect scenario-level labelling artefacts.

Also applies leakage-column removal and a standardized label taxonomy (mapped to
MITRE ATT&CK-aligned categories) across all eight datasets.

## Phase 2 — Evaluation Distortion Study

**`phase2_evaluation_distortion.py`** constructs five controlled imbalance scenarios per
dataset (Native, IR=100, IR=20, IR=5, IR=1) under a **fixed-N experimental design**: the
total training budget is held constant across all five scenarios so that the Metric
Sensitivity Index (MSI) measures the effect of class ratio in isolation from sample size.
Trains Random Forest and XGBoost, 5 repeats per scenario.

**MSI** is defined as the range-based sensitivity:

```
MSI(m) = (max_s[m_s] − min_s[m_s]) / mean_s[m_s]
```

where `m_s` is a metric's mean value at scenario `s`. Computed separately for macro-F1
and MCC.

**`phase2_analysis.py`** aggregates results into paper-ready tables, figures, and
auto-generated interpretation notes.

**`phase2_dl_baseline.py`** extends the same fixed-N pipeline with an MLP classifier,
testing whether the RF/XGBoost findings generalize beyond tree ensembles.

**`robustness_analysis.py`** provides three statistical robustness checks:
1. Bootstrap confidence intervals and permutation tests on MSI (replacing a fixed
   threshold with a per-dataset significance criterion).
2. A sensitivity analysis excluding the Native scenario, to isolate any residual
   sample-size confound for the highest-native-IR datasets.
3. KL-divergence bin-count sensitivity for the Phase 3 fidelity results (below).

## Phase 3 — Correction Strategy Evaluation (`phase3_correction_strategies.py`)

Evaluates seven imbalance correction strategies (baseline, ROS, SMOTE, BorderlineSMOTE,
RUS, class-weighted RF, class-weighted XGBoost) on:

- **Performance:** macro-F1, MCC (5 repeats).
- **Minority class fidelity:** mean KL divergence between real and synthetic minority
  class feature distributions, via 20-bin histograms per feature.

## Phase 4 — Design Principles

Six evidence-based design principles (P1–P6) synthesised from Phases 1–3. Not a
standalone script — derived and presented in the paper, each principle traceable to a
specific empirical finding.

## Phase 5 — Design Principle Validation

**`phase5_principle_validation.py`** tests P1 (IR ceiling), P2 (minimum class size), and
P3 (benign proportion) empirically:

- **Part A:** a synthetic dataset constructed to satisfy all three principles
  simultaneously (via `sklearn.datasets.make_classification`), evaluated under the same
  fixed-N MSI pipeline.
- **Part B:** a real-data-derived P1–P3-compliant subset of CSE-CIC-IDS2018 (the study's
  most evaluation-unreliable dataset), constructed by excluding the one class that cannot
  satisfy P2 without new capture and subsampling the majority class to satisfy P3.

**`phase5_partA_only.py`** is a fast standalone re-run of Part A alone (no large file
dependency), useful for iterating on the synthetic dataset design without repeating
Part B's expensive real-data pipeline.

## Reproducibility notes

- All stratified sampling in the pipeline (500K-row caps for large datasets) uses
  `sklearn.model_selection.StratifiedShuffleSplit` with a fixed `random_state=42` to
  preserve class proportions exactly under capping.
- A `MIN_CLASS_SAMPLES` floor (default 20) is applied throughout to prevent classes from
  being silently reduced to zero-sample or single-sample groups during scenario
  construction; scenarios where a class falls below this floor are logged and that class
  is excluded from that scenario only.
- Every script that depends on shared dataset configuration imports
  `phase1_imbalance_characterization.py` directly (via `importlib`) rather than
  duplicating the `DATASET_CONFIG` dictionary, so a single source of truth is maintained
  for dataset paths, leakage columns, and label taxonomies.
