#!/usr/bin/env python3
"""
phase5_principle_validation.py
IDS Imbalance Study — Design Principle Validation (Proof of Concept)

The six design principles (P1-P6) are derived inductively from the same eight
datasets used to diagnose the imbalance pathologies they address. This script
provides two complementary tests of whether following them actually reduces
MSI:

  PART A — Controlled synthetic dataset
    A fully synthetic classification dataset built explicitly to satisfy
    P1 (IR=100), P2 (minimum class size >=1,500), and P3 (~55% majority
    class), using sklearn.datasets.make_classification. This isolates
    whether the principles work in a noise-free, fully controlled setting.

  PART B — Real-data-derived compliant subset of CSE-CIC-IDS2018
    CSE-CIC-IDS2018 is the paper's most evaluation-unreliable dataset
    (MSI_MCC_RF=0.70). This part constructs a P1-P3-compliant SUBSET of
    the same real network traffic by:
      (i)  excluding WebAttack (317 samples) — the one class that cannot
           satisfy P2 (>=1,000 samples) without new capture, and
      (ii) subsampling Benign to achieve ~55-60% benign proportion (P3),
    then re-running the identical fixed-N 5-scenario MSI pipeline (RF,
    XGBoost, MLP) on this compliant subset and comparing MSI against the
    original (non-compliant) CSE-CIC-IDS2018 results already reported in
    the paper.

  If MSI drops substantially on both A and B relative to the native/
  non-compliant baselines, this constitutes empirical evidence that P1-P3
  are not merely descriptive of what went wrong in existing datasets, but
  prescriptive of what produces evaluation-stable ones.

Requires:
  - outputs/phase2/scenario_results/all_scenario_results.csv (for baseline comparison)
  - Raw CSE-CIC-IDS2018 data accessible via DATASET_CONFIG from phase1

Outputs saved to ./outputs/phase5_validation/

HPC Usage:
    screen -S phase5_validation
    python phase5_principle_validation.py 2>&1 | tee phase5_validation_run.log
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
OUT    = Path('./outputs/phase5_validation')
for sub in ['tables', 'figures']:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

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
    """
    Run the full fixed-N 5-scenario MSI pipeline (RF, XGBoost, MLP) on a
    given DataFrame, identical to Phase 2's protocol. Returns a results
    DataFrame and an MSI summary DataFrame.
    """
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

                results.append({
                    'dataset': dataset_tag, 'scenario': scenario_label, 'IR': actual_ir,
                    'model': model_name, 'repeat': rep,
                    'macro_f1': round(f1, 4), 'mcc': round(mcc, 4),
                })

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
            msi_records.append({'dataset': dataset_tag, 'model': model_name,
                                'metric': metric,
                                'MSI': round(msi, 4) if not np.isnan(msi) else None})

    return results_df, pd.DataFrame(msi_records)


# ══════════════════════════════════════════════════════════════════════════════
# PART A — Controlled Synthetic Dataset (P1, P2, P3 compliant by construction)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART A: Controlled Synthetic Dataset (P1-P3 compliant)")
print("="*70)

# Design: 5 classes, majority ~55% (P3), IR=100 relative to smallest class (P1),
# smallest class >=1,500 samples (P2, comfortably above the 1,000 floor).
# Class sizes verified below to actually produce ~55% majority proportion
# (an earlier version specified sizes that produced ~84% majority --
# satisfying P1/P2 but violating P3 -- this version corrects that).
MINORITY_N   = 1500
MAJORITY_N   = MINORITY_N * 100   # 150,000 -> IR=100 exactly
INTER_1_N    = 60000
INTER_2_N    = 38000
INTER_3_N    = 23200
TOTAL_N      = MAJORITY_N + INTER_1_N + INTER_2_N + INTER_3_N + MINORITY_N
weights = [MAJORITY_N/TOTAL_N, INTER_1_N/TOTAL_N, INTER_2_N/TOTAL_N,
          INTER_3_N/TOTAL_N, MINORITY_N/TOTAL_N]
assert 40 <= weights[0]*100 <= 65, f"P3 violated: majority={weights[0]*100:.1f}% not in [40,65]"
assert MAJORITY_N/MINORITY_N <= 1000, "P1 violated"
assert MINORITY_N >= 1000, "P2 violated" 

print(f"Target sample sizes: Majority={MAJORITY_N:,}, Inter1={INTER_1_N:,}, "
      f"Inter2={INTER_2_N:,}, Inter3={INTER_3_N:,}, Minority={MINORITY_N:,}")
print(f"Total N={TOTAL_N:,} | Target IR={MAJORITY_N/MINORITY_N:.1f} | "
      f"Target majority proportion={weights[0]*100:.1f}%")

X_synth, y_synth = make_classification(
    n_samples=TOTAL_N,
    n_features=40,          # comparable to typical NIDS feature counts
    n_informative=25,
    n_redundant=5,
    n_classes=5,
    n_clusters_per_class=2,
    weights=weights,
    flip_y=0.02,             # modest label noise, avoids trivial separability
    class_sep=1.0,           # moderate separability (not trivially separable like HIKARI)
    random_state=42,
)

synth_df = pd.DataFrame(X_synth, columns=[f'feat_{i}' for i in range(40)])
synth_df['label_std'] = y_synth

actual_profile = compute_distribution_metrics(synth_df['label_std'])
print(f"\nActual synthetic dataset profile:")
print(f"  IR={actual_profile['IR']} | n_classes={actual_profile['n_classes']} | "
      f"n_samples={actual_profile['n_samples']:,}")
print(f"  Majority class proportion: {actual_profile['majority_count']/actual_profile['n_samples']*100:.2f}%")
print(f"  Minority class count: {actual_profile['minority_count']:,}")

synth_results_df, synth_msi_df = run_msi_pipeline(synth_df, 'label_std', 'Synthetic_P1P3_Compliant')
synth_results_df.to_csv(OUT / 'tables' / 'part_a_synthetic_results.csv', index=False)
synth_msi_df.to_csv(OUT / 'tables' / 'part_a_synthetic_msi.csv', index=False)

print(f"\nPart A MSI summary:")
print(synth_msi_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# PART B — Real-Data-Derived Compliant Subset of CSE-CIC-IDS2018
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PART B: P1-P3-Compliant Subset of CSE-CIC-IDS2018")
print("="*70)

try:
    df_cse = load_dataset('CSE-CIC-IDS2018', DATASET_CONFIG['CSE-CIC-IDS2018'])
except Exception as e:
    print(f"[ERROR] Could not load CSE-CIC-IDS2018: {e}")
    df_cse = None

if df_cse is not None:
    native_profile = compute_distribution_metrics(df_cse['label_std'])
    print(f"\nOriginal CSE-CIC-IDS2018 profile (non-compliant):")
    print(f"  IR={native_profile['IR']} | classes={native_profile['n_classes']} | "
          f"benign_pct={native_profile['benign_pct']}")

    # Step (i): exclude WebAttack — cannot satisfy P2 (>=1,000 samples) without new capture
    counts = df_cse['label_std'].value_counts()
    print(f"\nClass counts before compliance adjustment:")
    print(counts.to_string())

    excluded_classes = counts[counts < 1000].index.tolist()
    print(f"\nExcluding classes below P2 threshold (<1,000 samples): {excluded_classes}")
    df_compliant = df_cse[~df_cse['label_std'].isin(excluded_classes)].copy()

    # Step (ii): subsample Benign to hit ~55-60% benign proportion (P3)
    counts_after_exclusion = df_compliant['label_std'].value_counts()
    non_benign_total = counts_after_exclusion.drop('Benign').sum()
    TARGET_BENIGN_PCT = 0.58  # midpoint of the 40-65% recommended range... but note
                              # P3 is now framed as provisional; 58% sits safely within it
    target_benign_n = int((TARGET_BENIGN_PCT / (1 - TARGET_BENIGN_PCT)) * non_benign_total)
    target_benign_n = min(target_benign_n, counts_after_exclusion['Benign'])

    print(f"\nNon-benign total after exclusion: {non_benign_total:,}")
    print(f"Target Benign count for {TARGET_BENIGN_PCT*100:.0f}% proportion: {target_benign_n:,}")

    benign_df = df_compliant[df_compliant['label_std'] == 'Benign'].sample(
        n=target_benign_n, random_state=42)
    non_benign_df = df_compliant[df_compliant['label_std'] != 'Benign']
    df_compliant_final = pd.concat([benign_df, non_benign_df], ignore_index=True).sample(
        frac=1, random_state=42).reset_index(drop=True)

    compliant_profile = compute_distribution_metrics(df_compliant_final['label_std'])
    print(f"\nCompliant subset profile (after P1-P3 adjustments):")
    print(f"  IR={compliant_profile['IR']} | classes={compliant_profile['n_classes']} | "
          f"benign_pct={compliant_profile['benign_pct']} | "
          f"n_samples={compliant_profile['n_samples']:,}")
    print(f"  Minority class: {compliant_profile['minority_class']} "
          f"({compliant_profile['minority_count']:,} samples)")

    # Save the compliance transformation summary
    compliance_summary = pd.DataFrame([{
        'metric': 'IR', 'native': native_profile['IR'], 'compliant': compliant_profile['IR'],
    }, {
        'metric': 'benign_pct', 'native': native_profile['benign_pct'],
        'compliant': compliant_profile['benign_pct'],
    }, {
        'metric': 'n_classes', 'native': native_profile['n_classes'],
        'compliant': compliant_profile['n_classes'],
    }, {
        'metric': 'minority_count', 'native': native_profile['minority_count'],
        'compliant': compliant_profile['minority_count'],
    }])
    compliance_summary.to_csv(OUT / 'tables' / 'part_b_compliance_transformation.csv', index=False)
    print(f"\nCompliance transformation summary:")
    print(compliance_summary.to_string(index=False))

    # Cap for tractability (same cap as Phase 2 for large datasets)
    if len(df_compliant_final) > 500_000:
        # Stratified cap — same fix as phase2_dl_baseline.py, ensures consistency
        # with Phase 2's capping protocol.
        counts_c = df_compliant_final['label_std'].value_counts()
        rare_c   = counts_c[counts_c < MIN_CLASS_SAMPLES].index.tolist()
        large_c  = counts_c[counts_c >= MIN_CLASS_SAMPLES].index.tolist()
        rare_df_c  = df_compliant_final[df_compliant_final['label_std'].isin(rare_c)]
        large_df_c = df_compliant_final[df_compliant_final['label_std'].isin(large_c)]
        budget_c = 500_000 - len(rare_df_c)
        if budget_c > 0 and len(large_df_c) > budget_c:
            from sklearn.model_selection import StratifiedShuffleSplit as _SSS2
            _sss2 = _SSS2(n_splits=1, train_size=budget_c, random_state=42)
            idx_c, _ = next(_sss2.split(large_df_c, large_df_c['label_std']))
            large_df_c = large_df_c.iloc[idx_c].reset_index(drop=True)
        df_compliant_final = pd.concat([rare_df_c, large_df_c], ignore_index=True).sample(
            frac=1, random_state=42).reset_index(drop=True)
        print(f"\nCapped to {len(df_compliant_final):,} rows for tractability "
              f"(stratified, same protocol as Phase 2)")

    compliant_results_df, compliant_msi_df = run_msi_pipeline(
        df_compliant_final, 'label_std', 'CSE-CIC-IDS2018_Compliant')
    compliant_results_df.to_csv(OUT / 'tables' / 'part_b_compliant_results.csv', index=False)
    compliant_msi_df.to_csv(OUT / 'tables' / 'part_b_compliant_msi.csv', index=False)

    print(f"\nPart B MSI summary (compliant subset):")
    print(compliant_msi_df.to_string(index=False))

    # ── Compare against original CSE-CIC-IDS2018 MSI from Phase 2 ──────────────
    existing_path = P2_OUT / 'scenario_results' / 'all_scenario_results.csv'
    if existing_path.exists():
        existing_df = pd.read_csv(existing_path)
        cse_existing = existing_df[existing_df['dataset'] == 'CSE-CIC-IDS2018']
        if 'valid' in cse_existing.columns:
            cse_existing = cse_existing[cse_existing['valid'] == True]

        print(f"\n{'='*60}\nBEFORE/AFTER COMPARISON: CSE-CIC-IDS2018 MSI\n{'='*60}")
        comparison_rows = []
        for model_name in ['RandomForest', 'XGBoost']:
            model_subset = cse_existing[cse_existing['model'] == model_name]
            if model_subset.empty:
                continue
            for metric in ['macro_f1', 'mcc']:
                means = model_subset.groupby('scenario')[metric].mean()
                native_msi = compute_msi(means.values) if len(means) >= 2 else np.nan

                compliant_row = compliant_msi_df[
                    (compliant_msi_df['model'] == model_name) &
                    (compliant_msi_df['metric'] == metric)]
                compliant_msi_val = compliant_row['MSI'].values[0] if not compliant_row.empty else None

                comparison_rows.append({
                    'model': model_name, 'metric': metric,
                    'MSI_native_CSE': round(native_msi, 4) if not np.isnan(native_msi) else None,
                    'MSI_compliant_CSE': compliant_msi_val,
                })

        # Add MLP (no existing baseline, compliant only)
        for metric in ['macro_f1', 'mcc']:
            compliant_row = compliant_msi_df[
                (compliant_msi_df['model'] == 'MLP') & (compliant_msi_df['metric'] == metric)]
            comparison_rows.append({
                'model': 'MLP', 'metric': metric,
                'MSI_native_CSE': None,  # no Phase 2 MLP baseline for CSE specifically in this script
                'MSI_compliant_CSE': compliant_row['MSI'].values[0] if not compliant_row.empty else None,
            })

        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df.to_csv(OUT / 'tables' / 'before_after_comparison.csv', index=False)
        print(comparison_df.to_string(index=False))
    else:
        print(f"\n[WARN] Existing Phase 2 results not found; skipping before/after comparison.")

print(f"\n\u2713 Principle validation complete. Outputs saved to {OUT}/")
