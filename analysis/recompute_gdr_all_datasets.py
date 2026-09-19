#!/usr/bin/env python3
"""
Compute GDR for all datasets with explicit non-finite handling.

Non-finite feature values propagate into GDR in two different ways depending on
how they are distributed across classes:

  Infinities confined to SOME classes
      The global mean of the affected column is infinite. A class containing no
      infinities gives finite minus infinite, which is infinite; a class
      containing them gives infinite minus infinite, which is undefined and is
      skipped. A subset of classes therefore reports inf.

  Infinities present in ALL classes
      Every class mean for that column is infinite, every numerator is
      undefined, and all are skipped. No infinity appears in the output, but the
      affected columns have been dropped from the average for every class, with
      no visible warning.

The second case is why this script covers all eight datasets rather than only
those that visibly report inf. A secondary effect is also reported: zero global
variance is replaced by a 1e-9 floor before division, so results are given both
with and without constant columns to make that contribution visible.

GDR is a descriptive statistic and is not consumed by the Phase 2, Phase 3 or
Phase 5 pipelines, so re-running this script does not affect any MSI,
correction-strategy or validation result.

Usage:
    python recompute_gdr_all_datasets.py

Output: outputs/gdr_recompute/gdr_corrected_all_datasets.csv
        outputs/gdr_recompute/gdr_old_vs_new_summary.csv
        outputs/gdr_recompute/gdr_affected_columns.csv
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

OUT = Path('./outputs/gdr_recompute')
OUT.mkdir(parents=True, exist_ok=True)

MAX_GLOBAL_SAMPLE = 500_000   # matches phase1
EPS_FLOOR         = 1e-9      # matches phase1's zero-variance replacement
LABEL_COL         = 'label_std'


def select_numeric_cols(df):
    """Replicates phase1's numeric column selection, including pd.to_numeric
    coercion (CSE-CIC-IDS2018 stores most features as object dtype)."""
    exclude = {LABEL_COL, 'label', 'label_std'}
    numeric_cols = []
    for c in [c for c in df.columns if c not in exclude]:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            coerced = pd.to_numeric(df[c], errors='coerce')
            if coerced.notna().mean() >= 0.8:
                df[c] = coerced
                numeric_cols.append(c)
    return [c for c in numeric_cols if df[c].notna().mean() > 0.5]


def compute_gdr(df, cols, strip_inf):
    """
    GDR per class, replicating phase1's formula exactly.

    strip_inf=False -> original behaviour (the bug).
    strip_inf=True  -> replace +/-inf with NaN before computing any mean or
                       variance, so infinities are excluded from the statistics
                       rather than poisoning them. NaNs are skipped per-column
                       by pandas, which is the intended behaviour for missing
                       values.
    """
    if not cols:
        return {}
    block = df[cols]
    if strip_inf:
        block = block.replace([np.inf, -np.inf], np.nan)

    smp = (block.sample(MAX_GLOBAL_SAMPLE, random_state=42)
           if len(block) > MAX_GLOBAL_SAMPLE else block)
    g_mean = smp.mean()
    g_var  = smp.var().replace(0, np.nan)

    out = {}
    for cls, cnt in df[LABEL_COL].value_counts().items():
        if cnt < 2:
            out[cls] = float('nan')
            continue
        c_mean = block.loc[df[LABEL_COL] == cls].mean()
        common = c_mean.index.intersection(g_mean.index)
        per_feat = ((c_mean[common] - g_mean[common]) ** 2
                    / g_var[common].replace(np.nan, EPS_FLOOR))
        out[cls] = float(per_feat.mean())
    return out


all_rows, col_rows = [], []

for ds_name, ds_cfg in DATASET_CONFIG.items():
    print("=" * 72)
    print(ds_name)
    print("=" * 72)
    try:
        df = load_dataset(ds_name, ds_cfg)
    except Exception as e:
        print(f"  [ERROR] {e}\n")
        continue

    cols = select_numeric_cols(df)
    if not cols:
        print("  [ERROR] No usable numeric columns.\n")
        continue

    # Census: which columns carry infinities, which are constant
    inf_cols, zerovar_cols = [], []
    for c in cols:
        col = df[c]
        n_inf = int(np.isposinf(col).sum() + np.isneginf(col).sum())
        if n_inf > 0:
            inf_cols.append(c)
            # How many classes contain at least one infinity? This distinguishes
            # Symptom A (subset of classes) from Symptom B (all classes).
            aff = df.loc[np.isinf(col), LABEL_COL].nunique()
            col_rows.append({'dataset': ds_name, 'column': c, 'issue': 'non-finite',
                             'n_nonfinite': n_inf,
                             'n_classes_containing': int(aff),
                             'n_classes_total': int(df[LABEL_COL].nunique())})
        finite = col.replace([np.inf, -np.inf], np.nan).dropna()
        if len(finite) > 1 and finite.nunique() <= 1:
            zerovar_cols.append(c)
            col_rows.append({'dataset': ds_name, 'column': c, 'issue': 'constant',
                             'n_nonfinite': 0, 'n_classes_containing': None,
                             'n_classes_total': int(df[LABEL_COL].nunique())})

    n_cls = df[LABEL_COL].nunique()
    if inf_cols:
        aff_counts = [df.loc[np.isinf(df[c]), LABEL_COL].nunique() for c in inf_cols]
        symptom = ('A (subset of classes -> visible inf)'
                   if min(aff_counts) < n_cls
                   else 'B (all classes -> silent column drop, no visible inf)')
        print(f"  Non-finite columns: {inf_cols}  --> Symptom {symptom}")
    else:
        print("  Non-finite columns: none")
    print(f"  Constant (zero-variance) columns: {len(zerovar_cols)}")

    variants = {
        'original_buggy':       compute_gdr(df, cols, strip_inf=False),
        'corrected_strip_inf':  compute_gdr(df, cols, strip_inf=True),
        'corrected_strip_inf_and_constants': compute_gdr(
            df, [c for c in cols if c not in set(zerovar_cols)], strip_inf=True),
    }

    print(f"\n  {'class':<20}{'original':>14}{'corrected':>14}{'corr.(-const)':>16}")
    for cls in sorted(variants['original_buggy'], key=str):
        def fmt(v):
            return 'inf' if np.isinf(v) else ('nan' if np.isnan(v) else f'{v:,.4f}')
        o = variants['original_buggy'][cls]
        c1 = variants['corrected_strip_inf'][cls]
        c2 = variants['corrected_strip_inf_and_constants'][cls]
        flag = '  <-- CHANGED' if (np.isinf(o) != np.isinf(c1)
                                   or (np.isfinite(o) and np.isfinite(c1)
                                       and abs(o - c1) > 0.01 * max(abs(o), 1e-12))) else ''
        print(f"  {str(cls):<20}{fmt(o):>14}{fmt(c1):>14}{fmt(c2):>16}{flag}")
        all_rows.append({'dataset': ds_name, 'class': cls,
                         'gdr_original_buggy': o,
                         'gdr_corrected': c1,
                         'gdr_corrected_no_constants': c2})
    print()
    del df

res = pd.DataFrame(all_rows)
res.to_csv(OUT / 'gdr_corrected_all_datasets.csv', index=False)
pd.DataFrame(col_rows).to_csv(OUT / 'gdr_affected_columns.csv', index=False)

# ── Summary of values the paper currently cites ────────────────────────────
print("=" * 72)
print("VALUES CITED IN THE PAPER -- CHECK EACH AGAINST ITS CORRECTED FIGURE")
print("=" * 72)

cited = [
    ('BoT-IoT', 'Benign', 314.17,
     "Sec 4.2: 'highest of any class in any dataset'"),
    ('CSE-CIC-IDS2018', 'WebAttack', float('inf'),
     "Sec 4.2 / 7.6 / 7.7.2: 'GDR = inf for five of seven classes'"),
    ('CSE-CIC-IDS2018', 'Botnet', float('inf'), "same claim"),
    ('CSE-CIC-IDS2018', 'BruteForce', float('inf'), "same claim"),
    ('CSE-CIC-IDS2018', 'DoS', float('inf'), "same claim"),
    ('CSE-CIC-IDS2018', 'DDoS', float('inf'), "same claim"),
]
summary = []
for ds, cls, old, where in cited:
    row = res[(res.dataset == ds) & (res['class'] == cls)]
    new = float(row['gdr_corrected'].iloc[0]) if not row.empty else float('nan')
    changed = row.empty or np.isinf(old) != np.isinf(new) or (
        np.isfinite(old) and np.isfinite(new) and abs(old - new) > 0.01 * max(abs(old), 1e-12))
    summary.append({'dataset': ds, 'class': cls, 'paper_value': old,
                    'corrected_value': new, 'changed': bool(changed), 'cited_in': where})
    o = 'inf' if np.isinf(old) else f'{old:,.4f}'
    n = 'inf' if np.isinf(new) else ('nan' if np.isnan(new) else f'{new:,.4f}')
    print(f"  {ds} / {cls}: paper={o}  corrected={n}  "
          f"{'** CHANGED **' if changed else '(unchanged)'}")
    print(f"      {where}")

pd.DataFrame(summary).to_csv(OUT / 'gdr_old_vs_new_summary.csv', index=False)

# Highest-GDR class overall, to re-check the BoT-IoT superlative in Sec 4.2
fin = res[np.isfinite(res['gdr_corrected'])]
if not fin.empty:
    top = fin.loc[fin['gdr_corrected'].idxmax()]
    print(f"\n  Highest corrected GDR across all datasets: "
          f"{top['dataset']} / {top['class']} = {top['gdr_corrected']:,.4f}")
    print("  (Section 4.2 claims this is BoT-IoT Benign -- verify it still holds.)")

print(f"\nSaved: {OUT}/gdr_corrected_all_datasets.csv")
print(f"Saved: {OUT}/gdr_old_vs_new_summary.csv")
print(f"Saved: {OUT}/gdr_affected_columns.csv")
