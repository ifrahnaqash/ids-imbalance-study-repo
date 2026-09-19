#!/usr/bin/env python3
"""
phase2_dl_baseline.py
IDS Imbalance Study — Deep Learning Baseline Extension

The MSI-based instability findings (CSE-CIC-IDS2018, CICIoT2023) and the
trivial-separability finding (HIKARI-2021) are established using tree-based
classifiers (RF, XGBoost), which are comparatively robust to class imbalance.
This script adds a Multi-Layer Perceptron, a different inductive bias, under
the identical fixed-N design used in Phase 2, to test whether those findings
are model-class-dependent or generalise beyond tree ensembles.

This script does NOT re-run RF/XGBoost (already computed in Phase 2). It runs
only the MLP scenarios and saves results separately; the analysis step then
merges with the existing all_scenario_results.csv for a three-way comparison.

Requires:
  - outputs/phase2/scenario_results/all_scenario_results.csv (for merge/comparison)
  - Raw datasets accessible via DATASET_CONFIG from phase1

Outputs saved to ./outputs/phase2_dl/

HPC Usage:
    screen -S phase2_dl
    python phase2_dl_baseline.py 2>&1 | tee phase2_dl_run.log
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, matthews_corrcoef

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Import Phase 1 shared config ─────────────────────────────────────────────
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "phase1", Path(__file__).parent / "phase1_imbalance_characterization.py")
_phase1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_phase1)

DATASET_CONFIG                = _phase1.DATASET_CONFIG
load_dataset                  = _phase1.load_dataset
compute_distribution_metrics  = _phase1.compute_distribution_metrics

print("Shared config imported from phase1_imbalance_characterization.py")

# ── Paths ─────────────────────────────────────────────────────────────────────
P2_OUT = Path('./outputs/phase2')
OUT    = Path('./outputs/phase2_dl')
for sub in ['scenario_results', 'tables', 'figures']:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

# ── Fixed-N design parameters — IDENTICAL to phase2_evaluation_distortion.py ──
MAX_IR_RATIO      = 100
MIN_FIXED_N       = 5000
MIN_CLASS_SAMPLES = 20
N_REPEATS         = 5

IR_TARGETS      = ['native', 100, 20, 5, 1]
SCENARIO_LABELS = ['Native (fixed-N)', 'IR=100', 'IR=20', 'IR=5', 'IR=1 (Balanced)']


def create_imbalance_scenario(df, label_col, target_ir, fixed_n, minority_count,
                              random_state=42):
    """Identical logic to Phase 2's fixed-N scenario constructor."""
    counts       = df[label_col].value_counts()
    majority_cls = counts.index[0]
    minority_cls = counts.index[-1]

    if target_ir == 'native':
        if len(df) <= fixed_n:
            return df.copy().sample(frac=1, random_state=random_state)
        scale = fixed_n / len(df)
        parts = []
        for cls, cnt in counts.items():
            n_keep = max(minority_count, int(cnt * scale))
            n_keep = min(n_keep, cnt)
            parts.append(df[df[label_col] == cls].sample(n=n_keep, random_state=random_state))
        result = pd.concat(parts, ignore_index=True)
        return result.sample(frac=1, random_state=random_state)

    n_majority_target = min(int(minority_count * target_ir), counts[majority_cls])
    n_minority_target = minority_count
    n_remaining = fixed_n - n_majority_target - n_minority_target
    intermediate_classes = [c for c in counts.index if c not in (majority_cls, minority_cls)]

    parts = [df[df[label_col] == majority_cls].sample(n=n_majority_target, random_state=random_state),
             df[df[label_col] == minority_cls]]

    if intermediate_classes and n_remaining > 0:
        inter_total = sum(counts[c] for c in intermediate_classes)
        for cls in intermediate_classes:
            n_inter = max(1, int(n_remaining * counts[cls] / max(inter_total, 1)))
            n_inter = min(n_inter, counts[cls])
            parts.append(df[df[label_col] == cls].sample(n=n_inter, random_state=random_state))

    result = pd.concat(parts, ignore_index=True)
    return result.sample(frac=1, random_state=random_state)


def compute_msi(scenario_means):
    arr = np.array(scenario_means)
    m = arr.mean()
    if m < 1e-9:
        return np.nan
    return float((arr.max() - arr.min()) / m)


def get_mlp():
    """
    Simple MLP — modest architecture chosen for tractability across dataset
    sizes without GPU infrastructure, consistent with this study's compute
    constraints. Early stopping avoids excessive training time.
    """
    return MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42,
    )


# ── Main experiment loop — MLP only, all 8 datasets, all 5 scenarios ────────
all_results = []
load_errors = []

for ds_name, ds_config in DATASET_CONFIG.items():
    print(f"\n{'='*60}\nMLP Baseline: {ds_name}\n{'='*60}")

    try:
        df = load_dataset(ds_name, ds_config)
    except Exception as e:
        print(f"  [ERROR] {e}")
        load_errors.append(ds_name)
        continue

    if len(df) > 500_000:
        # Stratified cap — matches Phase 2's cap exactly (StratifiedShuffleSplit,
        # not plain .sample()), ensuring the fixed-N budget computed here is
        # based on the SAME post-cap class composition as the RF/XGBoost results
        # in outputs/phase2/, making the three-way MSI comparison valid.
        counts = df['label_std'].value_counts()
        rare   = counts[counts < MIN_CLASS_SAMPLES].index.tolist()
        large  = counts[counts >= MIN_CLASS_SAMPLES].index.tolist()
        rare_df  = df[df['label_std'].isin(rare)]
        large_df = df[df['label_std'].isin(large)]
        budget   = 500_000 - len(rare_df)
        if budget > 0 and len(large_df) > budget:
            from sklearn.model_selection import StratifiedShuffleSplit as _SSS
            _sss = _SSS(n_splits=1, train_size=budget, random_state=42)
            idx, _ = next(_sss.split(large_df, large_df['label_std']))
            large_df = large_df.iloc[idx].reset_index(drop=True)
        df = pd.concat([rare_df, large_df], ignore_index=True).sample(
            frac=1, random_state=42).reset_index(drop=True)

    ds_counts        = df['label_std'].value_counts()
    ds_minority_n    = int(ds_counts.iloc[-1])
    ds_n_classes     = len(ds_counts)
    ds_n_intermediate = max(0, ds_n_classes - 2)
    ds_fixed_n = max(
        ds_minority_n * MAX_IR_RATIO + ds_n_intermediate * MIN_CLASS_SAMPLES + ds_minority_n,
        MIN_FIXED_N)
    ds_fixed_n = min(ds_fixed_n, len(df))
    print(f"  Fixed-N budget: {ds_fixed_n:,} rows | Minority: {ds_minority_n:,}")

    for scenario_label, target_ir in zip(SCENARIO_LABELS, IR_TARGETS):
        print(f"  Scenario: {scenario_label}")
        df_scenario = create_imbalance_scenario(
            df, 'label_std', target_ir, fixed_n=ds_fixed_n,
            minority_count=ds_minority_n, random_state=42)

        class_counts = df_scenario['label_std'].value_counts()
        rare_in_scenario = class_counts[class_counts < MIN_CLASS_SAMPLES].index.tolist()
        if rare_in_scenario:
            df_scenario = df_scenario[~df_scenario['label_std'].isin(rare_in_scenario)]

        if df_scenario['label_std'].nunique() < 2:
            print(f"    [SKIP] Fewer than 2 classes remain.")
            continue

        drop_cols    = ['label', 'label_std']
        feature_cols = [c for c in df_scenario.select_dtypes(include=[np.number]).columns
                        if c not in drop_cols]
        actual_ir = compute_distribution_metrics(df_scenario['label_std'])['IR']

        le = LabelEncoder()
        X_s = df_scenario[feature_cols].fillna(0).values
        y_s = le.fit_transform(df_scenario['label_std'].values)

        f1_scores, mcc_scores = [], []
        for rep in range(N_REPEATS):
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=rep)
            train_idx, test_idx = next(sss.split(X_s, y_s))
            X_tr, X_te = X_s[train_idx], X_s[test_idx]
            y_tr, y_te = y_s[train_idx], y_s[test_idx]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

            model = get_mlp()
            t0 = time.time()
            model.fit(X_tr, y_tr)
            train_time = time.time() - t0

            y_pred = model.predict(X_te)
            f1  = f1_score(y_te, y_pred, average='macro', zero_division=0)
            mcc = matthews_corrcoef(y_te, y_pred)
            f1_scores.append(f1)
            mcc_scores.append(mcc)

            all_results.append({
                'dataset': ds_name, 'scenario': scenario_label, 'IR': actual_ir,
                'model': 'MLP', 'repeat': rep,
                'macro_f1': round(f1, 4), 'mcc': round(mcc, 4),
                'train_time_s': round(train_time, 3),
                'n_samples': len(df_scenario),
            })

        print(f"    MLP: macro-F1={np.mean(f1_scores):.4f} \u00b1{np.std(f1_scores):.4f} "
              f"| MCC={np.mean(mcc_scores):.4f}")

        pd.DataFrame(all_results).to_csv(
            OUT / 'scenario_results' / 'mlp_results_partial.csv', index=False)

    del df

mlp_df = pd.DataFrame(all_results)
mlp_df.to_csv(OUT / 'scenario_results' / 'mlp_results.csv', index=False)
print(f"\nMLP results saved: {OUT / 'scenario_results' / 'mlp_results.csv'}")

# ── Compute MSI for MLP ────────────────────────────────────────────────────────
msi_records = []
for ds in mlp_df['dataset'].unique():
    subset = mlp_df[mlp_df['dataset'] == ds]
    for metric in ['macro_f1', 'mcc']:
        means = subset.groupby('scenario')[metric].mean()
        msi = compute_msi(means.values)
        msi_records.append({'dataset': ds, 'model': 'MLP', 'metric': metric,
                            'MSI': round(msi, 4) if not np.isnan(msi) else None})

msi_mlp_df = pd.DataFrame(msi_records)
msi_mlp_df.to_csv(OUT / 'tables' / 'mlp_msi.csv', index=False)
print(f"\nMLP MSI results:")
print(msi_mlp_df.to_string(index=False))

# ── Merge with existing RF/XGBoost results for 3-way comparison ─────────────
existing_path = P2_OUT / 'scenario_results' / 'all_scenario_results.csv'
if existing_path.exists():
    existing_df = pd.read_csv(existing_path)

    comparison_records = []
    for ds in mlp_df['dataset'].unique():
        for metric_col, metric_name in [('macro_f1', 'macro_f1'), ('mcc', 'mcc')]:
            row = {'dataset': ds, 'metric': metric_name}
            for model in ['RandomForest', 'XGBoost']:
                existing_subset = existing_df[(existing_df['dataset'] == ds) &
                                              (existing_df['model'] == model)]
                if 'valid' in existing_subset.columns:
                    existing_subset = existing_subset[existing_subset['valid'] == True]
                if not existing_subset.empty:
                    means = existing_subset.groupby('scenario')[metric_col].mean()
                    row[f'MSI_{model}'] = round(compute_msi(means.values), 4)
                else:
                    row[f'MSI_{model}'] = None

            mlp_subset = mlp_df[mlp_df['dataset'] == ds]
            if not mlp_subset.empty:
                means = mlp_subset.groupby('scenario')[metric_col].mean()
                row['MSI_MLP'] = round(compute_msi(means.values), 4)
            else:
                row['MSI_MLP'] = None

            comparison_records.append(row)

    comparison_df = pd.DataFrame(comparison_records)
    comparison_df.to_csv(OUT / 'tables' / 'three_way_msi_comparison.csv', index=False)
    print(f"\n{'='*60}\nThree-way MSI comparison (RF vs XGBoost vs MLP)\n{'='*60}")
    print(comparison_df.to_string(index=False))

    # Highlight the two key questions
    print(f"\n--- Key check 1: HIKARI-2021 trivial separability (model-agnostic?) ---")
    hikari_row = comparison_df[(comparison_df['dataset'] == 'HIKARI-2021') &
                               (comparison_df['metric'] == 'macro_f1')]
    print(hikari_row.to_string(index=False))

    print(f"\n--- Key check 2: CSE-CIC-IDS2018 / CICIoT2023 instability (model-agnostic?) ---")
    instab_rows = comparison_df[comparison_df['dataset'].isin(['CSE-CIC-IDS2018', 'CICIoT2023'])]
    print(instab_rows.to_string(index=False))
else:
    print(f"\n[WARN] Existing Phase 2 results not found at {existing_path}; "
          f"skipping 3-way comparison.")

print(f"\n\u2713 MLP baseline extension complete. Outputs saved to {OUT}/")
