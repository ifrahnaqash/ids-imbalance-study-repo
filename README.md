# Imbalance as a Dataset Quality Dimension

Code accompanying the paper *"Imbalance as a Dataset Quality Dimension: Evaluation
Distortion and Design Principles for IDS Benchmark Datasets"* (Sanober & Mir).

This repository treats class imbalance in network intrusion detection (IDS) benchmark
datasets as a measurable dimension of dataset quality rather than a training-time
nuisance. Across eight widely used benchmarks and over 68 million network flows, it
measures how far reported evaluation metrics shift as the evaluation class distribution
changes, assesses seven correction strategies on both predictive performance and a
distributional fidelity criterion, and derives six construction principles for more
reliable benchmarks, three of which are tested empirically.

## Key findings

- **CSE-CIC-IDS2018 is the least evaluation-reliable dataset** in the study
  (MSI_MCC = 0.70). The ranking holds across three model families (Random Forest,
  XGBoost, MLP) and survives a Native-scenario sensitivity check, though a second
  cross-dataset training-budget confound is not fully resolved.
- **HIKARI-2021's near-perfect scores reflect trivial class separability**, not model
  strength. It cannot discriminate between weak and strong classifiers.
- **Macro-F1 and MCC diverge on CICIoT2023**, so reporting only F1 understates
  sensitivity to the evaluation protocol.
- **Class-weighted learning matches or exceeds data-level resampling** on 4 of 6 tested
  datasets with no distributional modification. BorderlineSMOTE produces the worst
  minority-class fidelity in every tested case (KL divergence up to 8.44), and can
  generate no synthetic samples at all for classes below roughly 20 real samples.
- **Cross-dataset transfer is infeasible for most dataset pairs.** Three-level column
  matching finds real multi-feature overlap in only 8 of 28 pairs; 20 of 28 share no
  feature beyond the label. The most compatible pair (CIC-IDS2017 and CSE-CIC-IDS2018,
  both CICFlowMeter-derived) shares 28 columns, still only about a third of each
  dataset's feature space.
- **Principles P1-P3 are necessary but not sufficient.** A real benchmark brought into
  full compliance remains at chance-level performance across all three model families.
  The reason is not established by the present evidence.
- **CSE-CIC-IDS2018 contains non-finite feature values** (36,039 in `Flow Byts/s`,
  95,760 in `Flow Pkts/s`) produced by CICFlowMeter dividing by a zero flow duration.
  Because they are confined to two of seven classes, they corrupt the class-conditional
  statistics of the classes that do *not* contain them, while the affected classes
  appear well behaved. Non-finite values must be excluded before any such statistic is
  computed.

## Repository layout

```
src/              Four-phase experimental pipeline
analysis/         Supporting analyses and robustness checks
notebooks/        Exploratory notebooks
docs/             Methodology notes
sample_outputs/   Example result files
```

### `src/` — main pipeline

| Script | Purpose |
|---|---|
| `phase1_imbalance_characterization.py` | Imbalance profiles: IR, normalised entropy, Gini, GDR, temporal spread |
| `phase2_evaluation_distortion.py` | Fixed-N scenario sweep and MSI computation (RF, XGBoost) |
| `phase2_dl_baseline.py` | MLP baseline under the identical fixed-N design |
| `phase2_analysis.py` | Phase 2 aggregation and figures |
| `phase3_correction_strategies.py` | Seven correction strategies, performance and KL fidelity |
| `phase5_principle_validation.py` | Principle validation on synthetic and real-derived data |
| `phase5_partA_only.py` | Synthetic-only validation run |
| `robustness_analysis.py` | Bootstrap confidence intervals and permutation tests |

### `analysis/` — supporting analyses

| Script | Purpose |
|---|---|
| `compute_bh_correction.py` | Benjamini-Hochberg FDR correction across the permutation tests |
| `compute_p1_scenario_summary.py` | Mean macro-F1 by scenario across valid dataset-model pairs |
| `compute_constant_nfixed_msi.py` | MSI under one shared training budget for all datasets |
| `compute_feature_overlap.py` | Three-level feature-schema overlap between dataset pairs |
| `diagnose_gdr_infinity.py` | Locates columns producing non-finite GDR values |
| `recompute_gdr_all_datasets.py` | GDR for all datasets with explicit non-finite handling |
| `e1_msi_vs_cv.py` | Compares MSI against a coefficient of variation |
| `e2_fidelity_predicts_performance.py` | Tests whether KL fidelity predicts downstream performance |

## Requirements

```bash
pip install -r requirements.txt
# On HPC or externally managed environments:
pip install -r requirements.txt --break-system-packages
```

Python 3.9 or later. The larger datasets (CSE-CIC-IDS2018, CICIoT2023) need roughly
32 GB of RAM to profile; the pipeline caps them at 500,000 rows via minority-class-aware
stratified sampling for the Phase 2 and Phase 3 experiments.

## Datasets

The eight benchmarks are third-party public releases and are not redistributed here.
Original sources are listed in `docs/METHODOLOGY.md`. Dataset paths are configured in
`DATASET_CONFIG` at the top of `src/phase1_imbalance_characterization.py`.

| Dataset | Flows | Classes |
|---|---|---|
| NSL-KDD | 148,517 | 5 |
| UNSW-NB15 | 257,673 | 9 |
| CIC-IDS2017 | 2,830,743 | 11 |
| HIKARI-2021 | 555,278 | 5 |
| BoT-IoT | 3,668,522 | 5 |
| ToN-IoT | 211,043 | 9 |
| CSE-CIC-IDS2018 | 16,232,943 | 7 |
| CICIoT2023 | 45,019,234 | 9 |

## Reproducing the results

Run the phases in order; each writes to `outputs/` and later phases read earlier output.

```bash
python src/phase1_imbalance_characterization.py
python src/phase2_evaluation_distortion.py
python src/phase2_dl_baseline.py
python src/phase3_correction_strategies.py
python src/robustness_analysis.py
python src/phase5_principle_validation.py
```

Supporting analyses can then be run in any order, for example:

```bash
python analysis/compute_bh_correction.py
python analysis/e1_msi_vs_cv.py
python analysis/recompute_gdr_all_datasets.py
```

Stochastic operations use a fixed seed of 42, with each of the five repeated runs per
scenario additionally seeded by its run index.

## A note on GDR and non-finite values

Global Deviation Ratio compares each class mean against the global mean. Non-finite
feature values propagate through it asymmetrically, and the effect differs by how they
are distributed:

- Confined to **some** classes, the unaffected classes report infinity while the affected
  ones return a finite value, because infinite minus infinite is undefined and is skipped.
- Present in **all** classes, no infinity appears in the output at all, but the affected
  columns have been silently dropped from every class average.

Both computation sites in `phase1_imbalance_characterization.py` exclude non-finite
values before computing any mean or variance. `diagnose_gdr_infinity.py` and
`recompute_gdr_all_datasets.py` report which columns are affected in which datasets.

## License

MIT. See `LICENSE`.

## Citation

See `CITATION.cff`.
