#!/usr/bin/env python3
"""
Recompute MSI with a single shared training budget across all datasets.

The main fixed-N design holds the training budget constant across scenarios
within a dataset, but lets it vary between datasets because N_fixed scales with
each dataset's own minority class count. This script instead applies one shared
budget to every dataset, so that MSI magnitudes are directly comparable across
the collection.

The shared budget defaults to 5,000 rows, the floor already used natively by
CSE-CIC-IDS2018, CICIoT2023, CIC-IDS2017 and BoT-IoT; those four are therefore
unaffected, and only the remaining datasets are re-run at a smaller budget than
their native N_fixed. Datasets whose native budget differs substantially from
the shared value may show scenario-construction distortion as a result, which
should be taken into account when interpreting the output.

Usage:
    python compute_constant_nfixed_msi.py [--n-fixed 5000]

Output: outputs/phase5_nfixed_check/constant_nfixed_results.csv
        outputs/phase5_nfixed_check/constant_nfixed_msi.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, matthews_corrcoef
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "phase1", Path(__file__).parent / "phase1_imbalance_characterization.py")
_phase1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_phase1)
DATASET_CONFIG               = _phase1.DATASET_CONFIG
load_dataset                 = _phase1.load_dataset
compute_distribution_metrics = _phase1.compute_distribution_metrics

MIN_CLASS_SAMPLES = 20
N_REPEATS = 5
IR_TARGETS      = ['native', 100, 20, 5, 1]
SCENARIO_LABELS = ['Native (fixed-N)', 'IR=100', 'IR=20', 'IR=5', 'IR=1 (Balanced)']

MODELS = {
    'RandomForest': lambda: RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    'XGBoost':      lambda: xgb.XGBClassifier(n_estimators=100, n_jobs=-1, random_state=42,
                                              eval_metric='mlogloss', use_label_encoder=False),
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
            n_keep = max(min(minority_count, cnt), int(cnt * scale))
            n_keep = min(n_keep, cnt)
            parts.append(df[df[label_col] == cls].sample(n=n_keep, random_state=random_state))
        result = pd.concat(parts, ignore_index=True)
        return result.sample(frac=1, random_state=random_state)
    n_majority_target = min(int(minority_count * target_ir), counts[majority_cls])
    n_minority_target = min(minority_count, counts[minority_cls])
    n_remaining = fixed_n - n_majority_target - n_minority_target
    intermediate_classes = [c for c in counts.index if c not in (majority_cls, minority_cls)]
    parts = [df[df[label_col] == majority_cls].sample(n=n_majority_target, random_state=random_state),
             df[df[label_col] == minority_cls].sample(n=n_minority_target, random_state=random_state)]
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


def main(shared_n_fixed):
    OUT = Path('./outputs/phase5_nfixed_check')
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"Shared N_fixed for ALL datasets: {shared_n_fixed:,} rows")
    print("(This intentionally shrinks the budget for NSL-KDD, UNSW-NB15, ToN-IoT, "
          "and HIKARI-2021 relative to their native N_fixed; CIC-IDS2017, BoT-IoT, "
          "CSE-CIC-IDS2018, and CICIoT2023 already use this value natively and are "
          "therefore unaffected by this change.)\n")

    all_results = []
    for ds_name, ds_config in DATASET_CONFIG.items():
        print(f"{'='*60}\n{ds_name}\n{'='*60}")
        try:
            df = load_dataset(ds_name, ds_config)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        if len(df) > 500_000:
            counts = df['label_std'].value_counts()
            rare  = counts[counts < MIN_CLASS_SAMPLES].index.tolist()
            large = counts[counts >= MIN_CLASS_SAMPLES].index.tolist()
            rare_df  = df[df['label_std'].isin(rare)]
            large_df = df[df['label_std'].isin(large)]
            budget = 500_000 - len(rare_df)
            if budget > 0 and len(large_df) > budget:
                from sklearn.model_selection import StratifiedShuffleSplit as _SSS
                _sss = _SSS(n_splits=1, train_size=budget, random_state=42)
                idx, _ = next(_sss.split(large_df, large_df['label_std']))
                large_df = large_df.iloc[idx].reset_index(drop=True)
            df = pd.concat([rare_df, large_df], ignore_index=True).sample(
                frac=1, random_state=42).reset_index(drop=True)

        ds_counts = df['label_std'].value_counts()
        ds_minority_n = int(ds_counts.iloc[-1])
        native_n_fixed = shared_n_fixed  # <-- the whole point of this script

        for scenario_label, target_ir in zip(SCENARIO_LABELS, IR_TARGETS):
            df_scenario = create_imbalance_scenario(
                df, 'label_std', target_ir, fixed_n=native_n_fixed,
                minority_count=ds_minority_n, random_state=42)
            class_counts = df_scenario['label_std'].value_counts()
            rare_in_scenario = class_counts[class_counts < MIN_CLASS_SAMPLES].index.tolist()
            if rare_in_scenario:
                df_scenario = df_scenario[~df_scenario['label_std'].isin(rare_in_scenario)]
            if df_scenario['label_std'].nunique() < 2:
                print(f"  [SKIP] {scenario_label}: fewer than 2 classes")
                continue

            feature_cols = [c for c in df_scenario.select_dtypes(include=[np.number]).columns
                            if c not in ['label', 'label_std']]
            le = LabelEncoder()
            X_s = df_scenario[feature_cols].fillna(0).values
            y_s = le.fit_transform(df_scenario['label_std'].values)

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
                    f1_scores.append(f1_score(y_te, y_pred, average='macro', zero_division=0))
                    mcc_scores.append(matthews_corrcoef(y_te, y_pred))
                    all_results.append({
                        'dataset': ds_name, 'scenario': scenario_label, 'model': model_name,
                        'repeat': rep, 'macro_f1': round(f1_scores[-1], 4),
                        'mcc': round(mcc_scores[-1], 4), 'shared_n_fixed': shared_n_fixed,
                    })
                print(f"  {scenario_label} | {model_name}: "
                      f"F1={np.mean(f1_scores):.4f} MCC={np.mean(mcc_scores):.4f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT / 'constant_nfixed_results.csv', index=False)

    msi_records = []
    for ds in results_df['dataset'].unique():
        for model_name in MODELS:
            subset = results_df[(results_df['dataset'] == ds) & (results_df['model'] == model_name)]
            for metric in ['macro_f1', 'mcc']:
                means = subset.groupby('scenario')[metric].mean()
                msi = compute_msi(means.values) if len(means) >= 2 else np.nan
                msi_records.append({'dataset': ds, 'model': model_name, 'metric': metric,
                                    'MSI_constant_Nfixed': round(msi, 4) if not np.isnan(msi) else None})
    msi_df = pd.DataFrame(msi_records)
    msi_df.to_csv(OUT / 'constant_nfixed_msi.csv', index=False)

    print(f"\n{'='*60}\nRanking under CONSTANT N_fixed={shared_n_fixed:,} (mean MSI_F1 across RF+XGB)\n{'='*60}")
    ranking = (msi_df[msi_df['metric'] == 'macro_f1']
               .groupby('dataset')['MSI_constant_Nfixed'].mean()
               .sort_values(ascending=False))
    print(ranking.to_string())
    print(f"\nCompare this ranking against Table 3's native-N_fixed ranking. If "
          f"CSE-CIC-IDS2018 and CICIoT2023 remain #1/#2 here, the N_fixed confound "
          f"is NOT the primary driver of Finding 1.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-fixed', type=int, default=5000,
                        help='Shared N_fixed budget applied to ALL datasets (default: 5000, '
                             'the floor already used natively by 4 of 8 datasets)')
    args = parser.parse_args()
    main(args.n_fixed)
