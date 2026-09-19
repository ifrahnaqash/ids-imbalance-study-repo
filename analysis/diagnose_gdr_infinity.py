#!/usr/bin/env python3
"""
Diagnose non-finite values in the Global Deviation Ratio computation.

GDR compares each class mean against the global mean, normalised by global
variance. Two conditions can make the result non-finite:

  1. A column contains +/-inf. CICFlowMeter emits Infinity for throughput
     columns such as 'Flow Byts/s' when flow duration is zero. Any infinity
     makes the global mean infinite, so a class containing no infinities gives
     finite minus infinite, which is itself infinite, while a class containing
     them gives infinite minus infinite, which is undefined and is silently
     skipped. The classes free of the defect are therefore the ones whose
     statistics it corrupts.

  2. A column has zero variance in the global subsample. Zero variance is
     replaced by a 1e-9 floor before division, so any non-zero numerator
     produces a very large value. Because global statistics come from a
     subsample while class means use the full class, a near-constant column
     can yield a small non-zero numerator and hence a large ratio.

This script reports which columns fall into each category and recomputes GDR
under four exclusion regimes so that the contribution of each can be seen.

Usage:
    python diagnose_gdr_infinity.py

Output: outputs/gdr_diagnosis/csecic_column_diagnosis.csv
        outputs/gdr_diagnosis/csecic_gdr_recomputed.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util
import warnings

warnings.filterwarnings('ignore')

_spec = importlib.util.spec_from_file_location(
    "phase1", Path(__file__).parent / "phase1_imbalance_characterization.py")
_phase1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_phase1)
DATASET_CONFIG = _phase1.DATASET_CONFIG
load_dataset   = _phase1.load_dataset

OUT = Path('./outputs/gdr_diagnosis')
OUT.mkdir(parents=True, exist_ok=True)

DATASET = 'CSE-CIC-IDS2018'
MAX_GLOBAL_SAMPLE = 500_000   # must match phase1's constant
EPS_FLOOR = 1e-9              # must match phase1's zero-variance replacement

print("=" * 72)
print(f"GDR = infinity diagnosis for {DATASET}")
print("=" * 72)

df = load_dataset(DATASET, DATASET_CONFIG[DATASET])
label_col = 'label_std'
print(f"Loaded: {len(df):,} rows, {len(df.columns)} columns\n")

# Reproduce phase1's numeric column selection EXACTLY, including its
# pd.to_numeric coercion step. CSE-CIC-IDS2018 stores most feature columns as
# object dtype (numeric strings with occasional non-numeric entries), so a
# plain select_dtypes(include=[np.number]) finds nothing and every downstream
# statistic silently becomes empty.
exclude = {label_col, 'label', 'label_std'}
candidate_cols = [c for c in df.columns if c not in exclude]

numeric_cols = []
n_coerced = 0
for c in candidate_cols:
    if pd.api.types.is_numeric_dtype(df[c]):
        numeric_cols.append(c)
    else:
        coerced = pd.to_numeric(df[c], errors='coerce')
        if coerced.notna().mean() >= 0.8:
            df[c] = coerced
            numeric_cols.append(c)
            n_coerced += 1

numeric_cols = [c for c in numeric_cols if df[c].notna().mean() > 0.5]

print(f"Numeric columns used in GDR: {len(numeric_cols)} "
      f"({n_coerced} recovered by pd.to_numeric coercion)\n")

if not numeric_cols:
    print("[ERROR] No usable numeric columns found. This diagnostic cannot run.")
    print("Check that load_dataset returned the expected feature columns.")
    raise SystemExit(1)

# ── STEP 1: per-column pathology census (H1 and H2 candidates) ──────────────
print("-" * 72)
print("STEP 1: Per-column pathology census")
print("-" * 72)

records = []
for c in numeric_cols:
    col = df[c]
    n_posinf = int(np.isposinf(col).sum())
    n_neginf = int(np.isneginf(col).sum())
    n_nan    = int(col.isna().sum())
    finite   = col.replace([np.inf, -np.inf], np.nan).dropna()
    var_full = float(finite.var()) if len(finite) > 1 else np.nan
    nuniq    = int(finite.nunique())
    records.append({
        'column': c,
        'n_posinf': n_posinf, 'n_neginf': n_neginf, 'n_nan': n_nan,
        'n_unique_finite': nuniq,
        'variance_full_finite': var_full,
        'is_constant': nuniq <= 1,
        'has_nonfinite': (n_posinf + n_neginf) > 0,
    })

col_df = pd.DataFrame(records)
if col_df.empty or 'column' not in col_df.columns:
    print("[ERROR] Per-column census produced no rows despite non-empty "
          "numeric_cols. This should not happen; inspect load_dataset output.")
    raise SystemExit(1)

# Variance of the 500k subsample -- this is what phase1 actually divides by
sample = (df[numeric_cols].sample(MAX_GLOBAL_SAMPLE, random_state=42)
          if len(df) > MAX_GLOBAL_SAMPLE else df[numeric_cols])
sub_var = sample.var()
col_df['variance_in_500k_subsample'] = col_df['column'].map(sub_var)
col_df['zero_var_in_subsample'] = col_df['variance_in_500k_subsample'].fillna(0) == 0

col_df = col_df.sort_values(
    ['has_nonfinite', 'zero_var_in_subsample', 'variance_in_500k_subsample'],
    ascending=[False, False, True])
col_df.to_csv(OUT / 'csecic_column_diagnosis.csv', index=False)

h1 = col_df[col_df['has_nonfinite']]
h2 = col_df[col_df['zero_var_in_subsample'] & ~col_df['has_nonfinite']]
h3 = col_df[col_df['is_constant'] & ~col_df['has_nonfinite']]

print(f"H1 -- columns containing +/-inf ({len(h1)}):")
print(h1[['column', 'n_posinf', 'n_neginf']].to_string(index=False)
      if not h1.empty else "  (none)")
print(f"\nH2 -- columns with zero variance in the 500k subsample, no inf ({len(h2)}):")
print(h2[['column', 'n_unique_finite', 'variance_full_finite']].to_string(index=False)
      if not h2.empty else "  (none)")
print(f"\nH3 -- genuinely constant across the full dataset ({len(h3)}):")
print(h3[['column', 'n_unique_finite']].to_string(index=False)
      if not h3.empty else "  (none)")

# ── STEP 2: recompute GDR under four column-exclusion regimes ──────────────
print("\n" + "-" * 72)
print("STEP 2: Recomputing GDR per class under different exclusions")
print("-" * 72)

def compute_gdr(frame, cols, tag):
    """Replicates phase1's GDR computation exactly, over a given column set."""
    if not cols:
        return {}
    smp = (frame[cols].sample(MAX_GLOBAL_SAMPLE, random_state=42)
           if len(frame) > MAX_GLOBAL_SAMPLE else frame[cols])
    g_mean = smp.mean()
    g_var  = smp.var().replace(0, np.nan)
    out = {}
    for cls, cnt in frame[label_col].value_counts().items():
        if cnt < 2:
            out[cls] = np.nan
            continue
        c_mean = frame.loc[frame[label_col] == cls, cols].mean()
        common = c_mean.index.intersection(g_mean.index)
        per_feat = ((c_mean[common] - g_mean[common]) ** 2
                    / g_var[common].replace(np.nan, EPS_FLOOR))
        out[cls] = float(per_feat.mean())
    return out

regimes = {
    'as_published (all numeric cols)': numeric_cols,
    'drop_inf_cols (H1 removed)':      [c for c in numeric_cols if c not in set(h1['column'])],
    'drop_zerovar_cols (H2 removed)':  [c for c in numeric_cols if c not in set(h2['column'])],
    'drop_both (H1+H2 removed)':       [c for c in numeric_cols
                                        if c not in set(h1['column']) | set(h2['column'])],
}

results = {}
for tag, cols in regimes.items():
    print(f"\n{tag}  [{len(cols)} columns]")
    gdr = compute_gdr(df, cols, tag)
    results[tag] = gdr
    for cls, v in sorted(gdr.items(), key=lambda kv: str(kv[0])):
        shown = 'inf' if np.isinf(v) else ('nan' if np.isnan(v) else f'{v:,.4f}')
        print(f"    {cls:<18} GDR = {shown}")

res_df = pd.DataFrame(results)
res_df.index.name = 'class'
res_df.to_csv(OUT / 'csecic_gdr_recomputed.csv')

# ── STEP 3: verdict ────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("VERDICT")
print("=" * 72)

def n_inf(d):
    return sum(1 for v in d.values() if isinstance(v, float) and np.isinf(v))

base_inf = n_inf(results['as_published (all numeric cols)'])
print(f"Classes with GDR=inf as published:        {base_inf}")
print(f"  after removing inf-valued columns (H1): {n_inf(results['drop_inf_cols (H1 removed)'])}")
print(f"  after removing zero-var columns (H2):   {n_inf(results['drop_zerovar_cols (H2 removed)'])}")
print(f"  after removing both:                    {n_inf(results['drop_both (H1+H2 removed)'])}")

print("""
INTERPRETATION

* If removing the inf-valued columns alone eliminates the infinities, the
  non-finite result is a data-cleaning artefact of the throughput columns
  rather than evidence of degenerate class boundaries.

* If removing the zero-variance columns is what eliminates them, the result
  follows from dividing by the 1e-9 variance floor while comparing full-class
  means against subsampled global means, and is likewise not a property of the
  data.

* If infinities persist after removing both, the effect is intrinsic to the
  feature distributions; the surviving columns and their variances are then the
  quantity of interest.

The per-column CSV lists every candidate with its non-finite counts and
variances.
""")
print(f"Saved: {OUT / 'csecic_column_diagnosis.csv'}")
print(f"Saved: {OUT / 'csecic_gdr_recomputed.csv'}")
