#!/usr/bin/env python3
"""
Mean macro-F1 by imbalance scenario, averaged across valid dataset-model
combinations.

Aggregates the Phase 2 scenario results to show how mean macro-F1 varies with
the evaluation class ratio across the dataset collection. Combinations already
established as degenerate (CSE-CIC-IDS2018 with XGBoost, which collapses to
majority-class prediction) are excluded so that they do not distort the mean.

Usage:
    python compute_p1_scenario_summary.py

Input:  outputs/phase2/scenario_results/all_scenario_results.csv
Output: outputs/phase2/tables/p1_scenario_summary.csv
        A LaTeX table fragment printed to stdout.
"""

import pandas as pd
import numpy as np
from pathlib import Path

IN_PATH  = Path('./outputs/phase2/scenario_results/all_scenario_results.csv')
OUT_DIR  = Path('./outputs/phase2/tables')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIO_ORDER = ['Native (fixed-N)', 'IR=100', 'IR=20', 'IR=5', 'IR=1 (Balanced)']

df = pd.read_csv(IN_PATH)

# Exclude combinations already established as invalid/degenerate
# (CSE-CIC-IDS2018 x XGBoost -- MCC collapse, see Table 3's own N/A markers).
# Keep this exclusion consistent with Section 5.4.1's reconciled 30-combination count.
invalid_mask = (df['dataset'] == 'CSE-CIC-IDS2018') & (df['model'] == 'XGBoost')
valid_df = df[~invalid_mask].copy()

print(f"Total rows: {len(df):,} | Excluded (invalid) rows: {invalid_mask.sum():,} "
      f"| Valid rows used: {len(valid_df):,}")

# Mean macro-F1 per scenario, averaged first within each dataset-model
# combination (across the 5 repeated runs), then across all valid
# dataset-model combinations (so every dataset-model pair contributes
# equally regardless of how many repeated runs it has).
per_combo = (valid_df.groupby(['dataset', 'model', 'scenario'])['macro_f1']
             .mean().reset_index())
scenario_summary = (per_combo.groupby('scenario')['macro_f1']
                    .agg(['mean', 'std', 'count']).reindex(SCENARIO_ORDER))
scenario_summary.columns = ['mean_macro_f1', 'std_macro_f1', 'n_dataset_model_combos']

scenario_summary.to_csv(OUT_DIR / 'p1_scenario_summary.csv')
print(f"\nSaved: {OUT_DIR / 'p1_scenario_summary.csv'}\n")
print(scenario_summary.round(4).to_string())

best_scenario = scenario_summary['mean_macro_f1'].idxmax()
print(f"\nHighest mean macro-F1 scenario: {best_scenario} "
      f"({scenario_summary.loc[best_scenario, 'mean_macro_f1']:.4f})")

# ── LaTeX table fragment ──────────────────────────────────────────────────
print("\n" + "="*70)
print("LaTeX table fragment (paste in place of Table~\\ref{tblIR}):")
print("="*70)
print(r"""
\begin{table}[htbp]
\centering
\caption{Mean Macro-F1 by Imbalance Scenario, Averaged Across Valid Dataset-Model Combinations}
\label{tblIR}
\begin{tabular*}{\linewidth}{@{}lccc@{}}
\toprule
\textbf{Scenario} & \textbf{Mean macro-F1} & \textbf{Std.\ dev.} & \textbf{N combinations} \\
\midrule""")
for scen in SCENARIO_ORDER:
    row = scenario_summary.loc[scen]
    print(f"{scen} & {row['mean_macro_f1']:.4f} & {row['std_macro_f1']:.4f} "
          f"& {int(row['n_dataset_model_combos'])} \\\\")
print(r"""\bottomrule
\end{tabular*}
\end{table}
""")
