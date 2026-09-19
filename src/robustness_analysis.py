#!/usr/bin/env python3
"""
robustness_analysis.py
IDS Imbalance Study — Statistical Robustness Checks

Provides three statistical robustness checks supporting the MSI results:

  SECTION A — MSI threshold statistical foundation
    Replaces the informal "2x noise floor" heuristic for the MSI > 0.10
    threshold with:
      (i)  Bootstrap confidence intervals on MSI per dataset/model/metric
      (ii) A permutation test: null hypothesis = "scenario has no effect on
           the metric" (i.e., macro-F1/MCC values are exchangeable across
           scenarios). Rejecting the null at p<0.05 gives a statistically
           grounded criterion for declaring a dataset "evaluation-sensitive",
           replacing the eyeballed 0.10 cutoff.

  SECTION B — N_fixed / Native-scenario confound sensitivity analysis
    The Native scenario uses a smaller absolute majority-class sample count
    for high-native-IR datasets (CSE-CIC-IDS2018, CICIoT2023) than for
    low-IR datasets, even though N_fixed is held constant across the other
    four scenarios. This section recomputes MSI using ONLY the four
    genuinely N-matched scenarios (IR=100, 20, 5, 1), excluding Native, and
    checks whether CSE-CIC-IDS2018 and CICIoT2023 remain the two most
    unstable datasets under this restricted, fully-deconfounded computation.

  SECTION C — KL divergence bin-count sensitivity
    The headline KL=8.44 (ToN-IoT Ransomware, BorderlineSMOTE) and KL=5.85
    (CICIoT2023 Backdoor) figures were computed via 20-bin histograms.
    For classes with very few real samples (e.g., CICIoT2023 Backdoor: 17
    real samples), a 20-bin histogram is likely to be noisy. This section
    recomputes KL divergence for the four most extreme cases using:
      - Multiple bin counts (5, 10, 15, 20, 30)
      - A KDE-based continuous estimate (bin-free) as a robustness check
    If the qualitative ranking (BorderlineSMOTE >> SMOTE >> ROS) holds
    across all bin counts and the KDE estimate, the original finding is
    confirmed as robust rather than a binning artefact.

Requires:
  - outputs/phase2/scenario_results/all_scenario_results.csv (Sections A, B)
  - Raw datasets accessible via DATASET_CONFIG from phase1 (Section C)

Outputs saved to ./outputs/robustness/
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import gaussian_kde, entropy as kl_entropy
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Import Phase 1 shared config ─────────────────────────────────────────────
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "phase1", Path(__file__).parent / "phase1_imbalance_characterization.py")
_phase1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_phase1)

DATASET_CONFIG = _phase1.DATASET_CONFIG
load_dataset    = _phase1.load_dataset

print("Shared config imported from phase1_imbalance_characterization.py")

# ── Paths ─────────────────────────────────────────────────────────────────────
P2_OUT  = Path('./outputs/phase2')
OUT     = Path('./outputs/robustness')
for sub in ['tables', 'figures']:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP  = 5000
N_PERMUTE    = 5000
ALPHA        = 0.05

SCENARIO_ORDER   = ['Native (fixed-N)', 'IR=100', 'IR=20', 'IR=5', 'IR=1 (Balanced)']
FIXED_N_SCENARIOS = ['IR=100', 'IR=20', 'IR=5', 'IR=1 (Balanced)']  # excludes Native

# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — Bootstrap CI + Permutation Test on MSI
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION A: MSI Statistical Foundation (Bootstrap CI + Permutation Test)")
print("="*70)

results_df = pd.read_csv(P2_OUT / 'scenario_results' / 'all_scenario_results.csv')


def compute_msi(scenario_means):
    """MSI = (max - min) / mean across scenario means."""
    arr = np.array(scenario_means)
    m = arr.mean()
    if m < 1e-9:
        return np.nan
    return float((arr.max() - arr.min()) / m)


def bootstrap_msi_ci(df_subset, metric_col, scenario_col='scenario',
                      n_boot=N_BOOTSTRAP, ci=0.95):
    """
    Bootstrap CI for MSI: resample within each scenario (with replacement),
    recompute scenario means, recompute MSI. Repeat n_boot times.
    """
    scenarios = df_subset[scenario_col].unique()
    scenario_vals = {s: df_subset[df_subset[scenario_col] == s][metric_col].values
                     for s in scenarios}

    boot_msis = []
    for _ in range(n_boot):
        means = []
        for s in scenarios:
            vals = scenario_vals[s]
            if len(vals) == 0:
                continue
            resampled = np.random.choice(vals, size=len(vals), replace=True)
            means.append(resampled.mean())
        if len(means) >= 2:
            boot_msis.append(compute_msi(means))

    boot_msis = np.array([m for m in boot_msis if not np.isnan(m)])
    if len(boot_msis) == 0:
        return np.nan, np.nan, np.nan
    lo = np.percentile(boot_msis, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_msis, (1 + ci) / 2 * 100)
    return float(np.median(boot_msis)), float(lo), float(hi)


def permutation_test_msi(df_subset, metric_col, scenario_col='scenario',
                          n_perm=N_PERMUTE):
    """
    Permutation test: null = scenario has no effect on metric (values
    exchangeable across scenarios). Shuffle scenario labels, recompute
    MSI under the null, repeat n_perm times. p-value = fraction of null
    MSIs >= observed MSI.
    """
    scenarios = df_subset[scenario_col].values
    values    = df_subset[metric_col].values
    unique_scenarios = np.unique(scenarios)

    # Observed MSI
    obs_means = [values[scenarios == s].mean() for s in unique_scenarios]
    obs_msi = compute_msi(obs_means)

    if np.isnan(obs_msi):
        return obs_msi, np.nan

    null_msis = []
    for _ in range(n_perm):
        shuffled = np.random.permutation(values)
        means = [shuffled[scenarios == s].mean() for s in unique_scenarios]
        null_msis.append(compute_msi(means))

    null_msis = np.array([m for m in null_msis if not np.isnan(m)])
    p_value = float(np.mean(null_msis >= obs_msi))
    return obs_msi, p_value


datasets = results_df['dataset'].unique()
models   = ['RandomForest', 'XGBoost']
metrics  = ['macro_f1', 'mcc']

section_a_records = []
for ds in datasets:
    for model in models:
        subset = results_df[(results_df['dataset'] == ds) &
                            (results_df['model'] == model) &
                            results_df.get('valid', True)]
        if isinstance(subset, pd.DataFrame) and 'valid' in results_df.columns:
            subset = subset[subset['valid'] == True] if 'valid' in subset.columns else subset

        if subset.empty:
            continue

        for metric in metrics:
            metric_subset = subset[['scenario', metric]].dropna()
            if metric_subset['scenario'].nunique() < 2:
                continue

            med_msi, ci_lo, ci_hi = bootstrap_msi_ci(metric_subset, metric)
            obs_msi, p_val = permutation_test_msi(metric_subset, metric)

            section_a_records.append({
                'dataset': ds, 'model': model, 'metric': metric,
                'observed_MSI': round(obs_msi, 4) if not np.isnan(obs_msi) else None,
                'bootstrap_median_MSI': round(med_msi, 4) if not np.isnan(med_msi) else None,
                'CI_95_lower': round(ci_lo, 4) if not np.isnan(ci_lo) else None,
                'CI_95_upper': round(ci_hi, 4) if not np.isnan(ci_hi) else None,
                'permutation_p_value': round(p_val, 4) if not np.isnan(p_val) else None,
                'significant_at_0.05': (p_val < ALPHA) if not np.isnan(p_val) else None,
            })

section_a_df = pd.DataFrame(section_a_records)
section_a_df.to_csv(OUT / 'tables' / 'msi_statistical_validation.csv', index=False)
print(f"\nSaved: {OUT / 'tables' / 'msi_statistical_validation.csv'}")
print(section_a_df.to_string(index=False))

# Derive data-driven threshold: find the MSI value at p=0.05 boundary
# across all datasets, for macro_f1 specifically (the metric used for P1)
f1_significant = section_a_df[(section_a_df['metric'] == 'macro_f1') &
                              (section_a_df['significant_at_0.05'] == True)]
f1_not_significant = section_a_df[(section_a_df['metric'] == 'macro_f1') &
                                  (section_a_df['significant_at_0.05'] == False)]

print(f"\n--- Data-driven MSI threshold derivation ---")
if not f1_significant.empty and not f1_not_significant.empty:
    min_sig_msi = f1_significant['observed_MSI'].min()
    max_nonsig_msi = f1_not_significant['observed_MSI'].max()
    print(f"Minimum MSI among statistically significant (p<0.05) results: {min_sig_msi}")
    print(f"Maximum MSI among non-significant results: {max_nonsig_msi}")
    print(f"This empirically brackets the significance boundary, replacing "
          f"the heuristic 0.10 cutoff with a permutation-test-derived criterion.")
else:
    print("Insufficient split between significant/non-significant results "
          "to derive an empirical boundary; report per-dataset p-values directly.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — N_fixed Confound Sensitivity Analysis (exclude Native)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION B: N_fixed Confound Sensitivity Analysis")
print("="*70)
print("Recomputing MSI using ONLY the four N-matched scenarios "
      "(excluding Native), to test whether CSE-CIC-IDS2018 and CICIoT2023 "
      "remain the most unstable datasets when the Native-scenario confound "
      "is fully removed.\n")

section_b_records = []
for ds in datasets:
    for model in models:
        subset = results_df[(results_df['dataset'] == ds) &
                            (results_df['model'] == model)]
        if 'valid' in subset.columns:
            subset = subset[subset['valid'] == True]
        if subset.empty:
            continue

        for metric in metrics:
            # Full 5-scenario MSI (as reported in paper)
            full_subset = subset[subset['scenario'].isin(SCENARIO_ORDER)]
            full_means = (full_subset.groupby('scenario')[metric].mean())
            full_msi = compute_msi(full_means.values) if len(full_means) >= 2 else np.nan

            # Restricted 4-scenario MSI (excludes Native — fully N-matched)
            restricted_subset = subset[subset['scenario'].isin(FIXED_N_SCENARIOS)]
            restricted_means = (restricted_subset.groupby('scenario')[metric].mean())
            restricted_msi = compute_msi(restricted_means.values) if len(restricted_means) >= 2 else np.nan

            section_b_records.append({
                'dataset': ds, 'model': model, 'metric': metric,
                'MSI_with_Native': round(full_msi, 4) if not np.isnan(full_msi) else None,
                'MSI_without_Native': round(restricted_msi, 4) if not np.isnan(restricted_msi) else None,
                'delta': round(full_msi - restricted_msi, 4)
                         if not (np.isnan(full_msi) or np.isnan(restricted_msi)) else None,
            })

section_b_df = pd.DataFrame(section_b_records)
section_b_df.to_csv(OUT / 'tables' / 'n_fixed_sensitivity.csv', index=False)
print(f"Saved: {OUT / 'tables' / 'n_fixed_sensitivity.csv'}")
print(section_b_df.to_string(index=False))

# Check ranking stability for macro_f1, RF (the metric/model used for P1)
rf_f1 = section_b_df[(section_b_df['metric'] == 'macro_f1')]
rf_f1_mean = rf_f1.groupby('dataset').agg(
    MSI_with_Native=('MSI_with_Native', 'mean'),
    MSI_without_Native=('MSI_without_Native', 'mean')
).reset_index()

rank_with    = rf_f1_mean.sort_values('MSI_with_Native', ascending=False)['dataset'].tolist()
rank_without = rf_f1_mean.sort_values('MSI_without_Native', ascending=False)['dataset'].tolist()

print(f"\n--- Ranking stability check ---")
print(f"Top-2 most unstable WITH Native scenario:    {rank_with[:2]}")
print(f"Top-2 most unstable WITHOUT Native scenario: {rank_without[:2]}")
print(f"Ranking {'STABLE' if rank_with[:2] == rank_without[:2] else 'CHANGES'} "
      f"when the Native-scenario confound is removed.")

rf_f1_mean.to_csv(OUT / 'tables' / 'ranking_stability_check.csv', index=False)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — KL Divergence Bin-Count Sensitivity
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION C: KL Divergence Bin-Count Sensitivity")
print("="*70)

# The four most extreme KL cases reported in the paper, requiring
# re-derivation of real vs synthetic samples with the SAME random_state
# used in Phase 3, to enable bin-count and KDE-based re-estimation.
EXTREME_CASES = [
    {'dataset': 'ToN-IoT',    'class_name': 'Ransomware', 'strategy': 'BorderlineSMOTE'},
    {'dataset': 'CICIoT2023', 'class_name': 'Backdoor',   'strategy': 'BorderlineSMOTE'},
    {'dataset': 'BoT-IoT',    'class_name': 'Benign',     'strategy': 'BorderlineSMOTE'},
    # CIC-IDS2017's XSS class name is corrupted by an encoding fallback
    # (unmapped mojibake label). Matched dynamically at runtime via substring
    # search rather than hardcoded, since the exact byte sequence is
    # environment/encoding-dependent.
    {'dataset': 'CIC-IDS2017','class_name': None, 'class_substring': 'XSS',
     'strategy': 'BorderlineSMOTE'},
]

MAX_P3_SAMPLES = 300_000
MIN_CLASS_SIZE = 20
SMOTE_NEIGHBORS = 5


def compute_kl_at_bins(X_real, X_synthetic, n_bins):
    """KL divergence via n_bins-bin histogram, averaged over features."""
    kl_divs = []
    n_features = min(X_real.shape[1], X_synthetic.shape[1])
    for i in range(n_features):
        real_col = X_real[:, i]
        syn_col  = X_synthetic[:, i]
        lo = min(real_col.min(), syn_col.min())
        hi = max(real_col.max(), syn_col.max())
        if hi == lo:
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        p, _ = np.histogram(real_col, bins=bins, density=True)
        q, _ = np.histogram(syn_col,  bins=bins, density=True)
        p = p + 1e-10; q = q + 1e-10
        p = p / p.sum(); q = q / q.sum()
        kl_divs.append(float(kl_entropy(p, q)))
    return float(np.mean(kl_divs)) if kl_divs else np.nan


def compute_kl_kde(X_real, X_synthetic, n_grid=200):
    """
    KDE-based KL divergence: fit Gaussian KDE to real and synthetic samples
    per feature, evaluate both on a common grid, compute KL numerically.
    Bin-free — serves as a robustness check against histogram binning artefacts.
    """
    kl_divs = []
    n_features = min(X_real.shape[1], X_synthetic.shape[1])
    for i in range(n_features):
        real_col = X_real[:, i]
        syn_col  = X_synthetic[:, i]
        if len(np.unique(real_col)) < 2 or len(np.unique(syn_col)) < 2:
            continue
        try:
            kde_real = gaussian_kde(real_col)
            kde_syn  = gaussian_kde(syn_col)
        except Exception:
            continue
        lo = min(real_col.min(), syn_col.min())
        hi = max(real_col.max(), syn_col.max())
        grid = np.linspace(lo, hi, n_grid)
        p = kde_real(grid) + 1e-10
        q = kde_syn(grid) + 1e-10
        p = p / p.sum(); q = q / q.sum()
        kl_divs.append(float(kl_entropy(p, q)))
    return float(np.mean(kl_divs)) if kl_divs else np.nan


section_c_records = []

for case in EXTREME_CASES:
    ds_name = case['dataset']
    print(f"\nProcessing: {ds_name} — {case['class_name']} ({case['strategy']})")

    try:
        df = load_dataset(ds_name, DATASET_CONFIG[ds_name])
    except Exception as e:
        print(f"  [ERROR] {e}")
        continue

    # Resolve dynamic class name (substring match) if class_name is None
    if case.get('class_name') is None and 'class_substring' in case:
        candidates = [c for c in df['label_std'].unique()
                     if case['class_substring'].lower() in str(c).lower()]
        if not candidates:
            print(f"  [WARN] No class matching substring '{case['class_substring']}' found, skipping.")
            continue
        # Prefer the smallest matching class (most likely to be the fallback/rare one)
        counts_check = df['label_std'].value_counts()
        candidates_sorted = sorted(candidates, key=lambda c: counts_check.get(c, 0))
        resolved_name = candidates_sorted[0]
        case = dict(case)  # avoid mutating the shared EXTREME_CASES entry
        case['class_name'] = resolved_name
        print(f"  Resolved class name: {repr(resolved_name)} "
              f"(n={counts_check.get(resolved_name, 0)})")

    if len(df) > MAX_P3_SAMPLES:
        counts = df['label_std'].value_counts()
        rare   = counts[counts < MIN_CLASS_SIZE].index.tolist()
        large  = counts[counts >= MIN_CLASS_SIZE].index.tolist()
        rare_df  = df[df['label_std'].isin(rare)]
        large_df = df[df['label_std'].isin(large)]
        budget   = MAX_P3_SAMPLES - len(rare_df)
        if budget > 0 and len(large_df) > budget:
            large_df = large_df.sample(n=budget, random_state=42)
        df = pd.concat([rare_df, large_df], ignore_index=True).sample(
            frac=1, random_state=42).reset_index(drop=True)

    counts = df['label_std'].value_counts()
    rare_classes = counts[counts < MIN_CLASS_SIZE].index.tolist()

    # Match Phase 3's exact protocol: drop ALL classes below MIN_CLASS_SIZE
    # unconditionally. This preserves the same class set (and therefore the
    # same BorderlineSMOTE boundary/neighbor computation) as the original
    # Phase 3 run. If the target class itself is below MIN_CLASS_SIZE after
    # the stratified cap, it is reported and skipped rather than
    # artificially protected, which would distort the resampler's behavior.
    if case['class_name'] in rare_classes:
        print(f"  [WARN] Target class '{case['class_name']}' has "
              f"{counts.get(case['class_name'], 0)} samples after capping "
              f"(< MIN_CLASS_SIZE={MIN_CLASS_SIZE}); would be dropped under "
              f"Phase 3's exact protocol. Retaining it alone as a documented "
              f"deviation, since this is the specific small-n regime under test.")

    df_use = df[~df['label_std'].isin([c for c in rare_classes if c != case['class_name']])]

    if case['class_name'] not in df_use['label_std'].unique():
        print(f"  [WARN] Class '{case['class_name']}' not found after filtering, skipping.")
        continue

    drop_cols = ['label', 'label_std']
    feature_cols = [c for c in df_use.select_dtypes(include=[np.number]).columns
                    if c not in drop_cols]
    X = df_use[feature_cols].fillna(0).values
    y = df_use['label_std'].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_idx = list(le.classes_).index(case['class_name'])

    # Same split protocol as Phase 3 (rep 0, random_state=0)
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, _ = next(sss.split(X, y_enc))
    X_tr, y_tr = X[train_idx], y_enc[train_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)

    real_mask = y_tr == class_idx
    n_real = real_mask.sum()
    X_real = X_tr_s[real_mask]

    if n_real < 2:
        print(f"  [WARN] Fewer than 2 real samples, skipping.")
        continue

    # Regenerate synthetic samples with same protocol as Phase 3
    counts_tr = pd.Series(y_tr).value_counts()
    min_count = counts_tr.min()
    k = min(SMOTE_NEIGHBORS, min_count - 1)

    try:
        if k < 1:
            sampler = RandomOverSampler(random_state=0)
        else:
            sampler = BorderlineSMOTE(k_neighbors=k, random_state=0)
        X_res, y_res = sampler.fit_resample(X_tr_s, y_tr)
    except Exception as e:
        print(f"  [ERROR] Resampling failed: {e}")
        continue

    synth_mask = y_res == class_idx
    n_synth_total = synth_mask.sum()
    X_synth_only = X_res[synth_mask][n_real:]  # exclude the original real samples

    if len(X_synth_only) == 0:
        # BorderlineSMOTE can legitimately produce zero synthetic samples for
        # a class if none of its samples are classified as "danger" (boundary)
        # points — this is a known behavior of the borderline variant, distinct
        # from a fidelity problem. Retry with vanilla SMOTE to confirm the
        # target class is otherwise resamplable, and report this explicitly
        # as an additional BorderlineSMOTE failure mode rather than skipping silently.
        print(f"  [INFO] BorderlineSMOTE generated 0 synthetic samples for "
              f"'{case['class_name']}' (n_real={n_real}). This occurs when no "
              f"samples of this class are classified as boundary/'danger' points. "
              f"Retrying with vanilla SMOTE to confirm resamplability...")
        try:
            smote_sampler = SMOTE(k_neighbors=k, random_state=0) if k >= 1 \
                            else RandomOverSampler(random_state=0)
            X_res_alt, y_res_alt = smote_sampler.fit_resample(X_tr_s, y_tr)
            synth_mask_alt = y_res_alt == class_idx
            X_synth_alt = X_res_alt[synth_mask_alt][n_real:]
            if len(X_synth_alt) > 0:
                print(f"  [INFO] SMOTE (vanilla) successfully generated "
                      f"{len(X_synth_alt)} synthetic samples for the same class, "
                      f"confirming this is a BorderlineSMOTE-specific behavior, "
                      f"not a data issue. Recording as a documented failure mode.")
                section_c_records.append({
                    'dataset': ds_name, 'class': case['class_name'],
                    'strategy': 'BorderlineSMOTE',
                    'n_real': int(n_real), 'n_synthetic': 0,
                    'note': 'BorderlineSMOTE produced 0 synthetic samples '
                            '(no boundary/danger points identified); '
                            'vanilla SMOTE confirmed resamplable with '
                            f'{len(X_synth_alt)} synthetic samples.',
                })
        except Exception as e:
            print(f"  [WARN] SMOTE retry also failed: {e}")
        continue

    print(f"  Real samples: {n_real} | Synthetic samples: {len(X_synth_only)}")

    # Compute KL at multiple bin counts
    bin_results = {}
    for n_bins in [5, 10, 15, 20, 30]:
        kl_val = compute_kl_at_bins(X_real, X_synth_only, n_bins)
        bin_results[f'KL_bins_{n_bins}'] = round(kl_val, 4) if not np.isnan(kl_val) else None
        print(f"    {n_bins} bins: KL={kl_val:.4f}" if not np.isnan(kl_val) else f"    {n_bins} bins: KL=NaN")

    # Compute KDE-based KL (bin-free)
    kl_kde = compute_kl_kde(X_real, X_synth_only)
    print(f"    KDE (bin-free): KL={kl_kde:.4f}" if not np.isnan(kl_kde) else "    KDE: KL=NaN")

    record = {
        'dataset': ds_name,
        'class': case['class_name'],
        'strategy': case['strategy'],
        'n_real': int(n_real),
        'n_synthetic': int(len(X_synth_only)),
        **bin_results,
        'KL_KDE_binfree': round(kl_kde, 4) if not np.isnan(kl_kde) else None,
    }
    section_c_records.append(record)

    del df, df_use

section_c_df = pd.DataFrame(section_c_records)
section_c_df.to_csv(OUT / 'tables' / 'kl_bin_sensitivity.csv', index=False)
print(f"\nSaved: {OUT / 'tables' / 'kl_bin_sensitivity.csv'}")
print(section_c_df.to_string(index=False))

# ── Figure: KL vs bin count for each case ────────────────────────────────────
if not section_c_df.empty:
    fig, ax = plt.subplots(figsize=(9, 6))
    bin_counts = [5, 10, 15, 20, 30]
    for _, row in section_c_df.iterrows():
        y_vals = [row.get(f'KL_bins_{b}') for b in bin_counts]
        label = f"{row['dataset']} \u2014 {row['class']}"
        ax.plot(bin_counts, y_vals, marker='o', label=label, linewidth=2)
        if row.get('KL_KDE_binfree') is not None:
            ax.axhline(y=row['KL_KDE_binfree'], linestyle=':', alpha=0.4)
    ax.set_xlabel('Number of histogram bins', fontsize=11)
    ax.set_ylabel('KL Divergence', fontsize=11)
    ax.set_title('KL Divergence Sensitivity to Bin Count\n(dotted lines = KDE bin-free estimate)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / 'figures' / 'kl_bin_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {OUT / 'figures' / 'kl_bin_sensitivity.png'}")

print("\n" + "="*70)
print("✓ Robustness analysis complete. Outputs saved to ./outputs/robustness/")
print("="*70)
