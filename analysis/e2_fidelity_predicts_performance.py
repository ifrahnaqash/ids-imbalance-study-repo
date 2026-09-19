#!/usr/bin/env python3
"""
Test whether minority-class distributional fidelity predicts downstream
performance.

Phase 3 measures the KL divergence between real and synthetic minority samples
as a fidelity criterion. This script tests whether that criterion has predictive
value: does poorer fidelity correspond to worse classification performance?

Fidelity results are joined to the Phase 3 performance results, each dataset's
baseline score is subtracted to give the performance change attributable to the
correction strategy, and Spearman correlations are computed between KL
divergence and that change, pooled and per strategy. Spearman rather than
Pearson because the KL values span orders of magnitude and the sample is small.

Correlations are reported both with and without ROS. ROS duplicates existing
points, so its KL is near zero by construction; including it can manufacture an
apparent association from a definitional artefact rather than a real effect.

The sample is small (roughly 6 datasets x 3 strategies), so any correlation is
indicative rather than conclusive, and n is printed alongside every result.

Usage:
    python e2_fidelity_predicts_performance.py

Input:  outputs/phase3/fidelity/fidelity_results.csv
        outputs/phase3/results/p3_results.csv
Output: outputs/e2_fidelity/fidelity_vs_performance.csv
        outputs/e2_fidelity/correlation_summary.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

FID = Path('./outputs/phase3/fidelity/fidelity_results.csv')
PERF = Path('./outputs/phase3/results/p3_results.csv')
OUT = Path('./outputs/e2_fidelity')
OUT.mkdir(parents=True, exist_ok=True)

for p in (FID, PERF):
    if not p.exists():
        raise SystemExit(f"[ERROR] Not found: {p}\n"
                         f"Run phase3_correction_strategies.py first, or edit "
                         f"the path at the top of this script.")

fid = pd.read_csv(FID)     # dataset, strategy, class, n_real, n_synthetic, mean_KL_div
perf = pd.read_csv(PERF)   # dataset, strategy, macro_f1, std_f1, mcc, std_mcc, n_classes_used
print(f"Fidelity rows: {len(fid)} | Performance rows: {len(perf)}")

fid = fid.dropna(subset=['mean_KL_div'])

# Aggregate fidelity across minority classes within each dataset x strategy
agg = (fid.groupby(['dataset', 'strategy'])
          .agg(mean_KL=('mean_KL_div', 'mean'),
               max_KL=('mean_KL_div', 'max'),
               n_classes_scored=('mean_KL_div', 'size'),
               total_synthetic=('n_synthetic', 'sum'))
          .reset_index())

# Performance delta relative to each dataset's own Baseline
base = (perf[perf.strategy.str.lower() == 'baseline']
        .set_index('dataset')[['macro_f1', 'mcc']]
        .rename(columns={'macro_f1': 'base_f1', 'mcc': 'base_mcc'}))
if base.empty:
    raise SystemExit("[ERROR] No 'Baseline' strategy rows found in p3_results.csv; "
                     "cannot compute deltas. Check the strategy naming.")

m = agg.merge(perf, on=['dataset', 'strategy'], how='inner').join(base, on='dataset')
m['delta_f1']  = m['macro_f1'] - m['base_f1']
m['delta_mcc'] = m['mcc']      - m['base_mcc']
m = m.dropna(subset=['delta_f1'])
m.to_csv(OUT / 'fidelity_vs_performance.csv', index=False)

print(f"\nPaired observations: {len(m)}  "
      f"({m.dataset.nunique()} datasets x {m.strategy.nunique()} strategies)\n")
print(m[['dataset', 'strategy', 'mean_KL', 'max_KL',
         'delta_f1', 'delta_mcc']].round(4).to_string(index=False))

# ── Correlations ──────────────────────────────────────────────────────────
def corr(x, y, label, rows):
    ok = (~pd.isna(x)) & (~pd.isna(y))
    n = int(ok.sum())
    if n < 4:
        print(f"  {label:<34} n={n}  (too few points to test)")
        rows.append({'comparison': label, 'n': n, 'spearman_rho': None,
                     'p_value': None, 'significant_0.05': None})
        return
    xv, yv = np.asarray(x[ok], float), np.asarray(y[ok], float)
    if np.ptp(xv) == 0 or np.ptp(yv) == 0:
        # One side is constant (e.g. ROS, whose KL is ~0 for every dataset by
        # construction). Correlation is undefined, not zero -- say so.
        print(f"  {label:<34} n={n}  undefined (one variable is constant)")
        rows.append({'comparison': label, 'n': n, 'spearman_rho': None,
                     'p_value': None, 'significant_0.05': None})
        return
    rho, p = spearmanr(xv, yv)
    if np.isnan(rho):
        print(f"  {label:<34} n={n}  undefined (degenerate input)")
        rows.append({'comparison': label, 'n': n, 'spearman_rho': None,
                     'p_value': None, 'significant_0.05': None})
        return
    flag = '  <-- significant' if p < 0.05 else ''
    print(f"  {label:<34} n={n}  rho={rho:+.4f}  p={p:.4f}{flag}")
    rows.append({'comparison': label, 'n': n, 'spearman_rho': round(float(rho), 4),
                 'p_value': round(float(p), 4), 'significant_0.05': bool(p < 0.05)})

print("\n" + "=" * 70)
print("Correlation: KL divergence vs performance delta (all strategies pooled)")
print("=" * 70)
rows = []
corr(m.mean_KL, m.delta_f1,  'mean_KL vs delta macro-F1',  rows)
corr(m.max_KL,  m.delta_f1,  'max_KL  vs delta macro-F1',  rows)
corr(m.mean_KL, m.delta_mcc, 'mean_KL vs delta MCC',       rows)
corr(m.max_KL,  m.delta_mcc, 'max_KL  vs delta MCC',       rows)

# Oversampling strategies only: ROS has KL ~ 0 by construction (it duplicates
# real points), so including it can manufacture correlation from a definitional
# artefact rather than a real effect. Reported separately for that reason.
print("\n" + "=" * 70)
print("Synthetic-generating strategies only (ROS excluded: KL ~ 0 by construction)")
print("=" * 70)
syn = m[~m.strategy.str.upper().str.contains('ROS')]
corr(syn.mean_KL, syn.delta_f1,  'mean_KL vs delta macro-F1 (no ROS)', rows)
corr(syn.max_KL,  syn.delta_f1,  'max_KL  vs delta macro-F1 (no ROS)', rows)

print("\n" + "=" * 70)
print("Per strategy")
print("=" * 70)
for s, g in m.groupby('strategy'):
    corr(g.mean_KL, g.delta_f1, f'{s}: mean_KL vs delta F1', rows)

summary = pd.DataFrame(rows)
summary.to_csv(OUT / 'correlation_summary.csv', index=False)

# ── Verdict ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
tested = summary.dropna(subset=['spearman_rho'])
sig = tested[tested['significant_0.05'] == True]
if tested.empty:
    print("  No comparison had enough paired points to test.")
elif sig.empty:
    print("  No statistically significant association between fidelity and")
    print("  downstream performance at this sample size.")
    print("  At this sample size the criterion bears on the distributional")
    print("  integrity of augmented training data rather than on predicted")
    print("  classification accuracy.")
else:
    neg = sig[sig.spearman_rho < 0]
    if not neg.empty:
        print("  Significant negative association: higher KL divergence")
        print("  corresponds to weaker downstream performance, so the")
        print("  criterion carries predictive value.")
        print(neg[['comparison', 'n', 'spearman_rho', 'p_value']].to_string(index=False))
    else:
        print("  Significant positive association: higher KL divergence")
        print("  corresponds to stronger downstream performance. Fidelity and")
        print("  utility are in tension in this sample.")
        print(sig[['comparison', 'n', 'spearman_rho', 'p_value']].to_string(index=False))

print(f"\n  NOTE: n is small throughout; any association is indicative")
print(f"  rather than established.")
print(f"\nSaved: {OUT/'fidelity_vs_performance.csv'}")
print(f"Saved: {OUT/'correlation_summary.csv'}")
