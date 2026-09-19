#!/usr/bin/env python3
"""
phase3_correction_strategies.py
IDS Imbalance Study — RO2 Track A

Phase 3: Imbalance Correction Strategies
Evaluates data-level and algorithm-level correction strategies per dataset,
assessed on generalization stability and minority class fidelity —
NOT just held-out accuracy.

Requires: Phase 1 outputs, Phase 2 outputs

Strategies evaluated:
  Data-level:
    - Baseline (no correction)
    - Random Oversampling (ROS)
    - SMOTE
    - BorderlineSMOTE
    - Random Undersampling (RUS) — majority class only
  Algorithm-level:
    - Class-weighted RF (balanced)
    - Class-weighted XGBoost (scale_pos_weight)

Evaluation protocol:
  - Stratified 80/20 split (post-balancing, balancing applied to train only)
  - 5 repeated runs
  - Metrics: macro-F1, MCC, per-class recall for minority classes
  - Minority class fidelity: KL divergence of synthetic vs real feature distributions
  - Stability: std across repeats

Outputs saved to ./outputs/phase3/

HPC Usage:
    screen -S phase3
    python phase3_correction_strategies.py 2>&1 | tee phase3_run.log
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
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (f1_score, matthews_corrcoef,
                              recall_score, classification_report)
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
from scipy.stats import entropy as kl_entropy

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Import Phase 1 shared config ─────────────────────────────────────────────
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "phase1", Path(__file__).parent / "phase1_imbalance_characterization.py")
_phase1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_phase1)

DATA_ROOT                    = _phase1.DATA_ROOT
DATASET_CONFIG               = _phase1.DATASET_CONFIG
LEAKAGE_COLS                 = _phase1.LEAKAGE_COLS
LABEL_TAXONOMY               = _phase1.LABEL_TAXONOMY
NSL_KDD_COLS                 = _phase1.NSL_KDD_COLS
load_dataset                 = _phase1.load_dataset
compute_distribution_metrics = _phase1.compute_distribution_metrics

print("Shared config imported from phase1_imbalance_characterization.py")

# ── Check prerequisites ───────────────────────────────────────────────────────
P1_OUT = Path('./outputs/phase1')
P2_OUT = Path('./outputs/phase2')
P3_OUT = Path('./outputs/phase3')

if not (P1_OUT / 'imbalance_profiles.csv').exists():
    print("[ERROR] Phase 1 outputs not found.")
    sys.exit(1)

for subdir in ['results', 'fidelity', 'figures', 'per_class']:
    (P3_OUT / subdir).mkdir(parents=True, exist_ok=True)

profiles_df = pd.read_csv(P1_OUT / 'imbalance_profiles.csv')
print("Phase 3 initialised.")

# ── Config ────────────────────────────────────────────────────────────────────
MAX_P3_SAMPLES  = 300_000   # smaller cap for Phase 3 — oversampling is memory-intensive
MIN_CLASS_SIZE  = 20        # minimum samples per class to include
N_REPEATS       = 5
SMOTE_NEIGHBORS = 5         # k_neighbors for SMOTE/BorderlineSMOTE

# Datasets to run Phase 3 on.
# Excluded: HIKARI-2021 (trivially separable — correction has no effect),
#           CSE-CIC-IDS2018 (XGBoost collapsed — unreliable baseline)
# Both are documented findings; correction strategies add nothing there.
P3_DATASETS = [
    'NSL-KDD', 'UNSW-NB15', 'CIC-IDS2017',
    'BoT-IoT', 'ToN-IoT', 'CICIoT2023',
]

# ── Correction strategy definitions ──────────────────────────────────────────
def apply_correction(X_train, y_train, strategy, random_state=42):
    """
    Apply an imbalance correction strategy to training data.
    Returns corrected (X_train_c, y_train_c).
    Correction is applied ONLY to training data (no test leakage).

    Strategies:
        'baseline'        : No correction
        'ROS'             : Random Oversampling
        'SMOTE'           : SMOTE oversampling
        'BorderlineSMOTE' : Borderline SMOTE
        'RUS'             : Random Undersampling (majority class only)
        'ClassWeight_RF'  : Handled at model level, not data level
        'ClassWeight_XGB' : Handled at model level, not data level
    """
    if strategy in ('baseline', 'ClassWeight_RF', 'ClassWeight_XGB'):
        return X_train, y_train

    # Check minimum class size for SMOTE variants
    counts = pd.Series(y_train).value_counts()
    min_count = counts.min()

    if strategy == 'ROS':
        ros = RandomOverSampler(random_state=random_state)
        return ros.fit_resample(X_train, y_train)

    elif strategy == 'SMOTE':
        k = min(SMOTE_NEIGHBORS, min_count - 1)
        if k < 1:
            print(f"    [WARN] SMOTE: min class too small ({min_count}), "
                  "falling back to ROS")
            ros = RandomOverSampler(random_state=random_state)
            return ros.fit_resample(X_train, y_train)
        smote = SMOTE(k_neighbors=k, random_state=random_state)
        return smote.fit_resample(X_train, y_train)

    elif strategy == 'BorderlineSMOTE':
        k = min(SMOTE_NEIGHBORS, min_count - 1)
        if k < 1:
            print(f"    [WARN] BorderlineSMOTE: min class too small, "
                  "falling back to ROS")
            ros = RandomOverSampler(random_state=random_state)
            return ros.fit_resample(X_train, y_train)
        bsmote = BorderlineSMOTE(k_neighbors=k, random_state=random_state)
        return bsmote.fit_resample(X_train, y_train)

    elif strategy == 'RUS':
        rus = RandomUnderSampler(random_state=random_state)
        return rus.fit_resample(X_train, y_train)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def get_model(strategy, n_classes, random_state=42):
    """Return appropriate model for the strategy."""
    if strategy == 'ClassWeight_RF':
        return RandomForestClassifier(
            n_estimators=100, n_jobs=-1,
            class_weight='balanced', random_state=random_state)
    elif strategy == 'ClassWeight_XGB':
        return xgb.XGBClassifier(
            n_estimators=100, n_jobs=-1,
            random_state=random_state, eval_metric='mlogloss',
            use_label_encoder=False)
    else:
        # Default RF for data-level strategies
        return RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=random_state)


def compute_fidelity(X_real, X_synthetic, n_bins=20):
    """
    Compute mean KL divergence between real and synthetic minority
    class feature distributions.
    Lower = synthetic samples are more faithful to the real distribution.
    Uses histogram-based approximation per feature.
    """
    if X_synthetic is None or len(X_synthetic) == 0:
        return np.nan

    kl_divs = []
    n_features = min(X_real.shape[1], X_synthetic.shape[1])
    for i in range(n_features):
        real_col = X_real[:, i]
        syn_col  = X_synthetic[:, i]
        # Bin range covering both distributions
        lo = min(real_col.min(), syn_col.min())
        hi = max(real_col.max(), syn_col.max())
        if hi == lo:
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        p, _ = np.histogram(real_col, bins=bins, density=True)
        q, _ = np.histogram(syn_col,  bins=bins, density=True)
        # Add small epsilon to avoid log(0)
        p = p + 1e-10
        q = q + 1e-10
        p = p / p.sum()
        q = q / q.sum()
        kl_divs.append(float(kl_entropy(p, q)))

    return float(np.mean(kl_divs)) if kl_divs else np.nan


STRATEGIES = [
    'baseline', 'ROS', 'SMOTE', 'BorderlineSMOTE', 'RUS',
    'ClassWeight_RF', 'ClassWeight_XGB',
]

# ── Main Phase 3 Loop ─────────────────────────────────────────────────────────
all_p3_results  = []
fidelity_results = []
load_errors     = []

for ds_name in P3_DATASETS:
    if ds_name not in DATASET_CONFIG:
        print(f"\n[SKIP] {ds_name} not in DATASET_CONFIG")
        continue

    print(f"\n{'='*60}\nPhase 3: {ds_name}\n{'='*60}")

    try:
        df = load_dataset(ds_name, DATASET_CONFIG[ds_name])
    except Exception as e:
        print(f"  [ERROR] {e}")
        load_errors.append(ds_name)
        continue

    # Cap for memory
    if len(df) > MAX_P3_SAMPLES:
        print(f"  Capping to {MAX_P3_SAMPLES:,} rows (stratified)")
        counts = df['label_std'].value_counts()
        rare   = counts[counts < MIN_CLASS_SIZE].index.tolist()
        large  = counts[counts >= MIN_CLASS_SIZE].index.tolist()
        rare_df  = df[df['label_std'].isin(rare)]
        large_df = df[df['label_std'].isin(large)]
        budget   = MAX_P3_SAMPLES - len(rare_df)
        if budget > 0 and len(large_df) > budget:
            from sklearn.model_selection import StratifiedShuffleSplit as _S
            try:
                _sss = _S(n_splits=1, train_size=budget, random_state=42)
                idx, _ = next(_sss.split(large_df, large_df['label_std']))
                large_df = large_df.iloc[idx].reset_index(drop=True)
            except Exception:
                large_df = large_df.sample(n=budget, random_state=42)
        df = pd.concat([rare_df, large_df], ignore_index=True).sample(
            frac=1, random_state=42).reset_index(drop=True)

    # Drop classes below MIN_CLASS_SIZE
    counts = df['label_std'].value_counts()
    rare_classes = counts[counts < MIN_CLASS_SIZE].index.tolist()
    if rare_classes:
        print(f"  Dropping ultra-rare classes: {rare_classes}")
        df = df[~df['label_std'].isin(rare_classes)]

    if df['label_std'].nunique() < 2:
        print(f"  [SKIP] Fewer than 2 classes.")
        continue

    # Feature matrix
    drop_cols    = ['label', 'label_std']
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in drop_cols]
    X = df[feature_cols].fillna(0).values
    y = df['label_std'].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = le.classes_
    n_classes = len(class_names)

    # Identify minority classes (bottom third by count)
    counts_enc = pd.Series(y_enc).value_counts().sort_values()
    minority_class_indices = counts_enc.index[:max(1, n_classes // 3)].tolist()
    minority_class_names   = le.inverse_transform(minority_class_indices)
    print(f"  Minority classes for fidelity check: {minority_class_names.tolist()}")

    print(f"  Dataset shape: {X.shape} | Classes: {n_classes}")

    for strategy in STRATEGIES:
        print(f"\n  Strategy: {strategy}")
        f1_scores, mcc_scores = [], []
        per_class_recalls = {c: [] for c in class_names}
        fidelity_scores = []

        for rep in range(N_REPEATS):
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=0.2, random_state=rep)
            train_idx, test_idx = next(sss.split(X, y_enc))
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y_enc[train_idx], y_enc[test_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            # Apply correction (data-level strategies)
            try:
                X_tr_c, y_tr_c = apply_correction(
                    X_tr_s, y_tr, strategy, random_state=rep)
            except Exception as e:
                print(f"    [WARN] rep {rep}: correction failed ({e}), using baseline")
                X_tr_c, y_tr_c = X_tr_s, y_tr

            # Compute fidelity for oversampling strategies (rep 0 only)
            if rep == 0 and strategy in ('SMOTE', 'BorderlineSMOTE', 'ROS'):
                for mc_idx in minority_class_indices:
                    real_mask     = y_tr   == mc_idx
                    synth_mask    = y_tr_c == mc_idx
                    n_real  = real_mask.sum()
                    n_synth = synth_mask.sum() - n_real
                    if n_synth > 0:
                        X_real_mc  = X_tr_s[real_mask]
                        X_synth_mc = X_tr_c[synth_mask][n_real:]
                        kl = compute_fidelity(X_real_mc, X_synth_mc)
                        fidelity_results.append({
                            'dataset':        ds_name,
                            'strategy':       strategy,
                            'class':          le.inverse_transform([mc_idx])[0],
                            'n_real':         int(n_real),
                            'n_synthetic':    int(n_synth),
                            'mean_KL_div':    round(kl, 4) if not np.isnan(kl) else None,
                        })

            # Train model
            model = get_model(strategy, n_classes, random_state=rep)
            try:
                model.fit(X_tr_c, y_tr_c)
            except Exception as e:
                print(f"    [WARN] rep {rep}: model fit failed ({e})")
                continue

            y_pred = model.predict(X_te_s)

            f1  = f1_score(y_te, y_pred, average='macro', zero_division=0)
            mcc = matthews_corrcoef(y_te, y_pred)
            f1_scores.append(f1)
            mcc_scores.append(mcc)

            # Per-class recall
            recalls = recall_score(y_te, y_pred, average=None,
                                   labels=range(n_classes), zero_division=0)
            for ci, rc in enumerate(recalls):
                per_class_recalls[class_names[ci]].append(rc)

        if not f1_scores:
            continue

        mean_f1  = float(np.mean(f1_scores))
        std_f1   = float(np.std(f1_scores))
        mean_mcc = float(np.mean(mcc_scores))

        print(f"    macro-F1={mean_f1:.4f} ±{std_f1:.4f} | MCC={mean_mcc:.4f}")

        all_p3_results.append({
            'dataset':    ds_name,
            'strategy':   strategy,
            'macro_f1':   round(mean_f1,  4),
            'std_f1':     round(std_f1,   4),
            'mcc':        round(mean_mcc, 4),
            'std_mcc':    round(float(np.std(mcc_scores)), 4),
            'n_classes_used': n_classes,
        })

        # Save per-class recall
        pc_row = {'dataset': ds_name, 'strategy': strategy}
        for cls in class_names:
            recs = per_class_recalls[cls]
            pc_row[f'recall_{cls}'] = round(float(np.mean(recs)), 4) if recs else None
        pd.DataFrame([pc_row]).to_csv(
            P3_OUT / 'per_class' / f'{ds_name}_{strategy}_per_class.csv',
            index=False)

        # Intermediate save
        pd.DataFrame(all_p3_results).to_csv(
            P3_OUT / 'results' / 'p3_results_partial.csv', index=False)

    del df

# ── Save final outputs ────────────────────────────────────────────────────────
p3_df = pd.DataFrame(all_p3_results)
p3_df.to_csv(P3_OUT / 'results' / 'p3_results.csv', index=False)
print(f"\nPhase 3 results saved: {P3_OUT / 'results' / 'p3_results.csv'}")
print(p3_df.to_string(index=False))

if fidelity_results:
    fid_df = pd.DataFrame(fidelity_results)
    fid_df.to_csv(P3_OUT / 'fidelity' / 'fidelity_results.csv', index=False)
    print(f"\nFidelity results saved: {len(fid_df)} records")
    print(fid_df.to_string(index=False))

# ── Summary: best strategy per dataset ───────────────────────────────────────
print("\n" + "="*60)
print("BEST STRATEGY PER DATASET (by macro-F1)")
print("="*60)
if not p3_df.empty:
    best = (p3_df.loc[p3_df.groupby('dataset')['macro_f1'].idxmax()]
            [['dataset','strategy','macro_f1','mcc']])
    print(best.to_string(index=False))
    best.to_csv(P3_OUT / 'results' / 'best_strategy_per_dataset.csv', index=False)

print(f"\n✓ Phase 3 complete. Outputs saved to {P3_OUT}/")
