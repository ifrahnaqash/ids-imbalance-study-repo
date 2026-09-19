#!/usr/bin/env python3
"""
Compare MSI against a coefficient of variation.

MSI is a relative range, (max - min) / mean, computed over the scenario means.
It therefore belongs to the same family as the coefficient of variation, and
this script quantifies how far the two differ in practice.

For every dataset x model x metric combination it computes:

    MSI       (max - min) / mean      relative range
    CV        std / mean              coefficient of variation
    IQR_norm  (q75 - q25) / mean      robust alternative

and then compares the dataset ORDERINGS each produces, since ordering is what
MSI is used for when ranking datasets by evaluation reliability. Rank agreement,
not agreement between raw values, is the decision-relevant quantity.

Usage:
    python e1_msi_vs_cv.py

Input:  outputs/phase2/scenario_results/all_scenario_results.csv
Output: outputs/e1_msi_vs_cv/msi_vs_cv_full.csv
        outputs/e1_msi_vs_cv/ranking_comparison.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

IN   = Path('./outputs/phase2/scenario_results/all_scenario_results.csv')
OUT  = Path('./outputs/e1_msi_vs_cv')
OUT.mkdir(parents=True, exist_ok=True)

if not IN.exists():
    raise SystemExit(f"[ERROR] Not found: {IN}\n"
                     f"Run phase2_evaluation_distortion.py first, or edit IN "
                     f"to point at your saved all_scenario_results.csv.")

df = pd.read_csv(IN)
if 'valid' in df.columns:
    df = df[df['valid'] == True]

# Exclude the already-established invalid combinations, matching Section 5.4.1
invalid = (df['dataset'] == 'CSE-CIC-IDS2018') & (df['model'] == 'XGBoost')
df = df[~invalid]
print(f"Rows: {len(df):,} | combinations: "
      f"{df.groupby(['dataset','model']).ngroups}")

rows = []
for (ds, model), g in df.groupby(['dataset', 'model']):
    for metric in ['macro_f1', 'mcc']:
        means = g.groupby('scenario')[metric].mean()
        if len(means) < 2:
            continue
        v = means.values
        mu = v.mean()
        if abs(mu) < 1e-9:
            continue          # degenerate; MSI undefined (majority-class collapse)
        rows.append({
            'dataset':  ds,
            'model':    model,
            'metric':   metric,
            'MSI':      (v.max() - v.min()) / mu,
            'CV':       v.std(ddof=1) / mu,
            'IQR_norm': (np.percentile(v, 75) - np.percentile(v, 25)) / mu,
        })

res = pd.DataFrame(rows).round(4)
res.to_csv(OUT / 'msi_vs_cv_full.csv', index=False)
print(f"\nSaved: {OUT/'msi_vs_cv_full.csv'}  ({len(res)} combinations)\n")

# ── Correlation between the raw statistics ────────────────────────────────
print("=" * 68)
print("Raw statistic correlation (expected to be high -- not the real question)")
print("=" * 68)
for metric in ['macro_f1', 'mcc']:
    s = res[res.metric == metric]
    if len(s) > 2:
        r_cv, p_cv = spearmanr(s.MSI, s.CV)
        r_iq, _    = spearmanr(s.MSI, s.IQR_norm)
        print(f"  {metric:>9}: MSI~CV rho={r_cv:.4f} (p={p_cv:.4g}) | "
              f"MSI~IQR rho={r_iq:.4f}")

# ── The decision-relevant test: do the RANKINGS agree? ───────────────────
print("\n" + "=" * 68)
print("Dataset ranking agreement (this is what the paper actually relies on)")
print("=" * 68)

rank_rows = []
for metric in ['macro_f1', 'mcc']:
    for model in sorted(res.model.unique()):
        s = res[(res.metric == metric) & (res.model == model)]
        if len(s) < 3:
            continue
        by_msi = s.sort_values('MSI', ascending=False)['dataset'].tolist()
        by_cv  = s.sort_values('CV',  ascending=False)['dataset'].tolist()
        rho, p = spearmanr(s.MSI.rank(), s.CV.rank())
        identical = by_msi == by_cv
        moved = [d for i, d in enumerate(by_msi) if by_cv.index(d) != i]

        print(f"\n  {model} / {metric}:  Spearman rho = {rho:.4f}   "
              f"{'IDENTICAL ordering' if identical else 'ORDERING DIFFERS'}")
        print(f"    by MSI: {' > '.join(by_msi)}")
        print(f"    by CV : {' > '.join(by_cv)}")
        if moved:
            print(f"    datasets changing position: {moved}")

        rank_rows.append({'model': model, 'metric': metric,
                          'spearman_rho': round(float(rho), 4),
                          'identical_ordering': identical,
                          'datasets_moved': '; '.join(moved) if moved else '',
                          'top_by_MSI': by_msi[0], 'top_by_CV': by_cv[0]})

rank_df = pd.DataFrame(rank_rows)
rank_df.to_csv(OUT / 'ranking_comparison.csv', index=False)

# ── Verdict ───────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("VERDICT")
print("=" * 68)
if rank_df.empty:
    print("  Insufficient data to compare rankings.")
else:
    n_ident = int(rank_df.identical_ordering.sum())
    n_tot   = len(rank_df)
    print(f"  Identical dataset ordering in {n_ident} of {n_tot} "
          f"model x metric slices.")
    print(f"  Mean Spearman rho between MSI and CV rankings: "
          f"{rank_df.spearman_rho.mean():.4f}")
    if n_ident == n_tot:
        print("\n  MSI and CV rank the datasets identically in every slice, so\n"
              "  MSI carries no discriminative value over CV for ranking\n"
              "  purposes. The distinguishing element is the fixed-N protocol\n"
              "  under which the statistic is computed, not the statistic.")
    else:
        print("\n  Rankings diverge in at least one slice, so the two are not\n"
              "  interchangeable here. The peak-to-peak form reflects the\n"
              "  worst-case deviation across plausible evaluation\n"
              "  distributions rather than the average deviation.")
print(f"\nSaved: {OUT/'ranking_comparison.csv'}")
