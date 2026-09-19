#!/usr/bin/env python3
"""
Benjamini-Hochberg false discovery rate correction for the permutation tests.

The permutation tests report one p-value per dataset x model x metric
combination. Because those tests are reported together, raw p-values overstate
significance; this script applies a Benjamini-Hochberg correction across the
full set and reports raw and adjusted values side by side.

The two combinations already established as invalid (CSE-CIC-IDS2018 with
XGBoost, for both metrics) are excluded, since a permutation test of metric
sensitivity is uninformative for a metric that is itself degenerate.

Usage:
    python compute_bh_correction.py

Input:  outputs/robustness/tables/msi_statistical_validation.csv
        (written by robustness_analysis.py)
Output: outputs/robustness/tables/bh_corrected_pvalues.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Matches robustness_analysis.py's actual output exactly (verified against
#    that script's source, not assumed) ─────────────────────────────────────
INPUT_PATH = Path('./outputs/robustness/tables/msi_statistical_validation.csv')
COLUMN_MAP = {
    'permutation_p_value': 'p_value',
}

OUT = Path('./outputs/robustness/tables')
OUT.mkdir(parents=True, exist_ok=True)


def benjamini_hochberg(pvals, alpha=0.05):
    """
    Standard BH step-up procedure, implemented directly (no extra dependency
    on statsmodels, in case it isn't installed in the run environment).
    Returns (adjusted_pvals, reject_flags) aligned to the ORIGINAL order of
    the input pvals array.
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked_pvals = pvals[order]

    # BH adjusted p-values: p_adj[i] = min_{j>=i} (n/j) * p_(j), enforced monotone
    adjusted_sorted = ranked_pvals * n / (np.arange(n) + 1)
    # Enforce monotonicity from the largest rank down
    for i in range(n - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)

    adjusted = np.empty(n)
    adjusted[order] = adjusted_sorted
    reject = adjusted <= alpha
    return adjusted, reject


if not INPUT_PATH.exists():
    print(f"[ERROR] Input file not found: {INPUT_PATH}")
    print("This script expects msi_statistical_validation.csv, saved by")
    print("robustness_analysis.py's Section A (bootstrap CI + permutation test).")
    print("If you ran robustness_analysis.py with a different OUT path than")
    print("./outputs/robustness, update INPUT_PATH at the top of this script")
    print("to match wherever your run actually wrote")
    print("tables/msi_statistical_validation.csv.")
    raise SystemExit(1)

df = pd.read_csv(INPUT_PATH)
if COLUMN_MAP:
    df = df.rename(columns=COLUMN_MAP)

required = {'dataset', 'model', 'metric', 'p_value'}
missing = required - set(df.columns)
if missing:
    print(f"[ERROR] Missing expected columns: {missing}")
    print(f"Found columns: {list(df.columns)}")
    print("Update COLUMN_MAP at the top of this script to map your column "
          "names to: dataset, model, metric, p_value")
    raise SystemExit(1)

# Exclude the 2 already-established invalid combinations (CSE-CIC-IDS2018 x
# XGBoost, both metrics), matching the reconciled 30-combination count in
# the manuscript's Section 5.4.1.
invalid_mask = (df['dataset'] == 'CSE-CIC-IDS2018') & (df['model'] == 'XGBoost')
valid_df = df[~invalid_mask].copy()
print(f"Total rows in input: {len(df)} | Excluded (invalid): {invalid_mask.sum()} "
      f"| Valid combinations: {len(valid_df)}")

if len(valid_df) != 30:
    print(f"[WARNING] Expected exactly 30 valid combinations to match the "
          f"manuscript's Section 5.4.1 count; found {len(valid_df)}. Check "
          f"that this input file covers the same dataset x model x metric "
          f"grid as the manuscript (8 datasets x 2 models x 2 metrics = 32, "
          f"minus 2 invalid = 30).")

adjusted, reject = benjamini_hochberg(valid_df['p_value'].values, alpha=0.05)
valid_df['p_adjusted_BH'] = np.round(adjusted, 4)
valid_df['significant_raw'] = valid_df['p_value'] <= 0.05
valid_df['significant_BH'] = reject

valid_df = valid_df.sort_values('p_value').reset_index(drop=True)
valid_df.to_csv(OUT / 'bh_corrected_pvalues.csv', index=False)

n_sig_raw = valid_df['significant_raw'].sum()
n_sig_bh = valid_df['significant_BH'].sum()

print(f"\n{'='*60}\nBenjamini-Hochberg correction results\n{'='*60}")
print(f"Significant at raw alpha=0.05:       {n_sig_raw} / {len(valid_df)}")
print(f"Significant after BH correction:     {n_sig_bh} / {len(valid_df)}")
print(f"Combinations that LOSE significance under BH correction:")
lost = valid_df[valid_df['significant_raw'] & ~valid_df['significant_BH']]
if lost.empty:
    print("  (none)")
else:
    print(lost[['dataset', 'model', 'metric', 'p_value', 'p_adjusted_BH']].to_string(index=False))

print(f"\nFull table saved: {OUT / 'bh_corrected_pvalues.csv'}")

print(f"\n{'='*60}\nSuggested paper text (paste into Section 5.4.1, adjust numbers to match your actual run):\n{'='*60}")
print(f"Applying Benjamini--Hochberg FDR correction across the 30 tests, "
      f"{n_sig_bh} of {len(valid_df)} combinations remain significant at the "
      f"adjusted level (vs.\\ {n_sig_raw} at uncorrected $\\alpha$=0.05)"
      f"{', with no change to the qualitative conclusion' if n_sig_bh == n_sig_raw else ''}.")
