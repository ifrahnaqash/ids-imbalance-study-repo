#!/usr/bin/env python3
"""
phase5_partA_only.py
IDS Imbalance Study — Design Principle Validation, Part A (corrected)

Standalone re-run of Part A only (synthetic P1-P3-compliant dataset).
Part B (CSE-CIC-IDS2018 compliant subset) is unaffected by this fix and
does not need to be re-run; its results in outputs/phase5_validation/
remain valid.

Fix applied: the original Part A specified class sizes that produced an
IR=100 (P1 satisfied) but a majority class proportion of ~84% -- well
outside the P3 target range (40-65%) -- because the weights were computed
from class counts that did not actually correspond to the ~55% majority
proportion the script's own comments claimed. This version explicitly
verifies P1, P2, and P3 are satisfied by the computed class sizes BEFORE
generating data, and again AFTER generation using the realized profile.

Requires: no external data (synthetic only).

Outputs saved to ./outputs/phase5_validation/ (overwrites Part A files only).
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, matthews_corrcoef
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "phase1", Path(__file__).parent / "phase1_imbalance_characterization.py")
_phase1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_phase1)
compute_distribution_metrics = _phase1.compute_distribution_metrics

OUT = Path('./outputs/phase5_validation')
(OUT / 'tables').mkdir(parents=True, exist_ok=True)

MAX_IR_RATIO      = 100
MIN_FIXED_N       = 5000
MIN_CLASS_SAMPLES = 20
N_REPEATS         = 5
IR_TARGETS      = ['native', 100, 20, 5, 1]
SCENARIO_LABELS = ['Native (fixed-N)', 'IR=100', 'IR=20', 'IR=5', 'IR=1 (Balanced)']

MODELS = {
    'RandomForest': lambda: RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    'XGBoost':      lambda: xgb.XGBClassifier(n_estimators=100, n_jobs=-1, random_state=42,
                                              eval_metric='mlogloss', use_label_encoder=False),
    'MLP':          lambda: MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=200,
                                          early_stopping=True, n_iter_no_change=10, random_state=42),
}


def create_imbalance_scenario(df, label_col, target_ir, fixed_n, minority_count, random_state=42):
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


def run_msi_pipeline(df, label_col, dataset_tag):
    counts        = df[label_col].value_counts()
    minority_n    = int(counts.iloc[-1])
    n_classes     = len(counts)
    n_intermediate = max(0, n_classes - 2)
    fixed_n = max(minority_n * MAX_IR_RATIO + n_intermediate * MIN_CLASS_SAMPLES + minority_n,
                  MIN_FIXED_N)
    fixed_n = min(fixed_n, len(df))
    print(f"  [{dataset_tag}] Fixed-N budget: {fixed_n:,} | Minority: {minority_n:,} | Classes: {n_classes}")

    results = []
    for scenario_label, target_ir in zip(SCENARIO_LABELS, IR_TARGETS):
        df_scenario = create_imbalance_scenario(df, label_col, target_ir, fixed_n, minority_n, 42)
        class_counts = df_scenario[label_col].value_counts()
        rare = class_counts[class_counts < MIN_CLASS_SAMPLES].index.tolist()
        if rare:
            df_scenario = df_scenario[~df_scenario[label_col].isin(rare)]
        if df_scenario[label_col].nunique() < 2:
            print(f"    [SKIP] {scenario_label}: fewer than 2 classes")
            continue

        drop_cols = [label_col]
        feature_cols = [c for c in df_scenario.select_dtypes(include=[np.number]).columns
                        if c not in drop_cols]
        actual_ir = compute_distribution_metrics(df_scenario[label_col])['IR']

        le = LabelEncoder()
        X_s = df_scenario[feature_cols].fillna(0).values
        y_s = le.fit_transform(df_scenario[label_col].values)

        for model_name in MODELS:
            f1_scores, mcc_scores = [], []
            for rep in range(N_REPEATS):
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=rep)
                train_idx, test_idx = next(sss.split(X_s, y_s))
                X_tr, X_te = X_s[train_idx], X_s[test_idx]
                y_tr, y_te = y_s[train_idx], y_s[test_idx]
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)
                model = MODELS[model_name]()
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                f1  = f1_score(y_te, y_pred, average='macro', zero_division=0)
                mcc = matthews_corrcoef(y_te, y_pred)
                f1_scores.append(f1)
                mcc_scores.append(mcc)
                results.append({'dataset': dataset_tag, 'scenario': scenario_label, 'IR': actual_ir,
                                'model': model_name, 'repeat': rep,
                                'macro_f1': round(f1, 4), 'mcc': round(mcc, 4)})
            print(f"    {scenario_label} | {model_name}: "
                  f"F1={np.mean(f1_scores):.4f}\u00b1{np.std(f1_scores):.4f} "
                  f"MCC={np.mean(mcc_scores):.4f}")

    results_df = pd.DataFrame(results)
    msi_records = []
    for model_name in MODELS:
        subset = results_df[results_df['model'] == model_name]
        for metric in ['macro_f1', 'mcc']:
            means = subset.groupby('scenario')[metric].mean()
            msi = compute_msi(means.values) if len(means) >= 2 else np.nan
            msi_records.append({'dataset': dataset_tag, 'model': model_name, 'metric': metric,
                                'MSI': round(msi, 4) if not np.isnan(msi) else None})
    return results_df, pd.DataFrame(msi_records)


print("="*70)
print("PART A (CORRECTED): Controlled Synthetic Dataset (P1-P3 compliant)")
print("="*70)

MINORITY_N = 1500
MAJORITY_N = MINORITY_N * 100
INTER_1_N  = 60000
INTER_2_N  = 38000
INTER_3_N  = 23200
TOTAL_N    = MAJORITY_N + INTER_1_N + INTER_2_N + INTER_3_N + MINORITY_N
weights = [MAJORITY_N/TOTAL_N, INTER_1_N/TOTAL_N, INTER_2_N/TOTAL_N,
          INTER_3_N/TOTAL_N, MINORITY_N/TOTAL_N]

target_ir = MAJORITY_N / MINORITY_N
target_majority_pct = weights[0] * 100

print(f"Target sample sizes: Majority={MAJORITY_N:,}, Inter1={INTER_1_N:,}, "
      f"Inter2={INTER_2_N:,}, Inter3={INTER_3_N:,}, Minority={MINORITY_N:,}")
print(f"Total N={TOTAL_N:,} | Target IR={target_ir:.1f} | "
      f"Target majority proportion={target_majority_pct:.2f}%")
print(f"\nPre-generation principle check:")
print(f"  P1 (IR<=1000, target 50-100): target IR={target_ir:.1f} -> "
      f"{'PASS' if target_ir <= 1000 else 'FAIL'}")
print(f"  P2 (minority>=1000):          target minority={MINORITY_N:,} -> "
      f"{'PASS' if MINORITY_N >= 1000 else 'FAIL'}")
print(f"  P3 (majority 40-65%):         target majority%={target_majority_pct:.1f}% -> "
      f"{'PASS' if 40 <= target_majority_pct <= 65 else 'FAIL'}")

X_synth, y_synth = make_classification(
    n_samples=TOTAL_N, n_features=40, n_informative=25, n_redundant=5,
    n_classes=5, n_clusters_per_class=2, weights=weights,
    flip_y=0.02, class_sep=1.0, random_state=42,
)

synth_df = pd.DataFrame(X_synth, columns=[f'feat_{i}' for i in range(40)])
synth_df['label_std'] = y_synth

actual_profile = compute_distribution_metrics(synth_df['label_std'])
actual_majority_pct = actual_profile['majority_count']/actual_profile['n_samples']*100

print(f"\nActual (realized) synthetic dataset profile:")
print(f"  IR={actual_profile['IR']} | n_classes={actual_profile['n_classes']} | "
      f"n_samples={actual_profile['n_samples']:,}")
print(f"  Majority class proportion: {actual_majority_pct:.2f}%")
print(f"  Minority class count: {actual_profile['minority_count']:,}")

print(f"\nPost-generation principle check (using REALIZED values):")
print(f"  P1: realized IR={actual_profile['IR']} -> "
      f"{'PASS' if actual_profile['IR'] <= 1000 else 'FAIL'}")
print(f"  P2: realized minority={actual_profile['minority_count']:,} -> "
      f"{'PASS' if actual_profile['minority_count'] >= 1000 else 'FAIL'}")
print(f"  P3: realized majority%={actual_majority_pct:.2f}% -> "
      f"{'PASS' if 40 <= actual_majority_pct <= 65 else 'FAIL'}")

synth_results_df, synth_msi_df = run_msi_pipeline(synth_df, 'label_std', 'Synthetic_P1P3_Compliant')
synth_results_df.to_csv(OUT / 'tables' / 'part_a_synthetic_results.csv', index=False)
synth_msi_df.to_csv(OUT / 'tables' / 'part_a_synthetic_msi.csv', index=False)

profile_summary = pd.DataFrame([{
    'target_IR': round(target_ir, 1), 'realized_IR': actual_profile['IR'],
    'target_majority_pct': round(target_majority_pct, 2), 'realized_majority_pct': round(actual_majority_pct, 2),
    'target_minority_n': MINORITY_N, 'realized_minority_n': actual_profile['minority_count'],
    'n_samples': actual_profile['n_samples'], 'n_classes': actual_profile['n_classes'],
}])
profile_summary.to_csv(OUT / 'tables' / 'part_a_profile_verification.csv', index=False)

print(f"\nPart A MSI summary:")
print(synth_msi_df.to_string(index=False))

print(f"\n\u2713 Part A (corrected) complete. Outputs saved to {OUT}/")
print(f"  Part B results (from previous run) remain valid and unchanged.")
