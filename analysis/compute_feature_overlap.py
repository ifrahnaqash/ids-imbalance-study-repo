#!/usr/bin/env python3
"""
Feature-schema overlap between IDS benchmark datasets.

Compares the column sets of every dataset pair at three levels of strictness:

  Level 1  exact    raw column-name equality
  Level 2  normalised   lowercase with whitespace, underscores and hyphens
                        stripped, so that "Flow Duration" matches "flow_duration"
  Level 3  fuzzy    token-set Jaccard similarity above a threshold; reported
                    for manual review but NOT counted as a match, since
                    lexically similar names can denote different quantities

Only column headers are read, so the script runs in seconds regardless of
dataset size.

Usage:
    python compute_feature_overlap.py

Output: outputs/phase2/tables/feature_overlap_matrix.csv
        outputs/phase2/tables/feature_near_misses.csv
        A LaTeX table fragment printed to stdout.
"""

import re
import itertools
import pandas as pd
from pathlib import Path

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "phase1", Path(__file__).parent / "phase1_imbalance_characterization.py")
_phase1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_phase1)
DATASET_CONFIG = _phase1.DATASET_CONFIG

OUT = Path('./outputs/phase2/tables')
OUT.mkdir(parents=True, exist_ok=True)

FUZZY_THRESHOLD = 0.5  # Jaccard similarity on tokenized column names


def normalize(col):
    """Level 2: lowercase, strip whitespace/underscores/hyphens entirely."""
    return re.sub(r'[\s_\-]+', '', col.lower().strip())


def tokenize(col):
    """Split a column name into lowercase word tokens for fuzzy comparison.
    Handles snake_case, space-separated, and simple CamelCase."""
    # Insert a space before capital letters preceded by lowercase (CamelCase split)
    s = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', col)
    s = re.sub(r'[\s_\-/]+', ' ', s).lower().strip()
    return set(t for t in s.split() if t)


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def get_raw_columns(ds_name, ds_config):
    """Read only the header row of each dataset -- fast, no full data load."""
    import pandas as pd
    files = ds_config.get('files')
    if files:
        # Multi-file dataset: read header of the first file only
        first_file = files[0] if isinstance(files, list) else files
        cols = pd.read_csv(first_file, nrows=0).columns.tolist()
    else:
        path = ds_config.get('path')
        cols = pd.read_csv(path, nrows=0).columns.tolist()
    return [str(c).strip() for c in cols]


print("Loading column headers for all 8 datasets...\n")
dataset_columns = {}
for ds_name, ds_config in DATASET_CONFIG.items():
    try:
        cols = get_raw_columns(ds_name, ds_config)
        dataset_columns[ds_name] = cols
        print(f"  {ds_name}: {len(cols)} columns")
    except Exception as e:
        print(f"  [ERROR] {ds_name}: {e}")

print(f"\n{'='*70}\nPairwise Feature Overlap (Level 1: exact, Level 2: normalized)\n{'='*70}")

overlap_records = []
near_miss_records = []

for ds_a, ds_b in itertools.combinations(dataset_columns.keys(), 2):
    cols_a = dataset_columns[ds_a]
    cols_b = dataset_columns[ds_b]

    exact_match = set(cols_a) & set(cols_b)

    norm_a = {normalize(c): c for c in cols_a}
    norm_b = {normalize(c): c for c in cols_b}
    normalized_match = set(norm_a.keys()) & set(norm_b.keys())

    overlap_records.append({
        'dataset_A': ds_a, 'dataset_B': ds_b,
        'n_cols_A': len(cols_a), 'n_cols_B': len(cols_b),
        'exact_matches': len(exact_match),
        'normalized_matches': len(normalized_match),
        'exact_match_names': '; '.join(sorted(exact_match))[:200],
        'normalized_match_names': '; '.join(
            f"{norm_a[k]}=={norm_b[k]}" for k in sorted(normalized_match))[:300],
    })

    print(f"\n{ds_a} <-> {ds_b}: exact={len(exact_match)}, normalized={len(normalized_match)}")
    if normalized_match:
        print(f"  Normalized matches: {sorted(normalized_match)[:10]}")

    # Level 3: fuzzy near-misses among columns NOT already caught by Level 2
    unmatched_a = [c for c in cols_a if normalize(c) not in normalized_match]
    unmatched_b = [c for c in cols_b if normalize(c) not in normalized_match]
    tokens_a = {c: tokenize(c) for c in unmatched_a}
    tokens_b = {c: tokenize(c) for c in unmatched_b}

    for ca in unmatched_a:
        best_match, best_score = None, 0.0
        for cb in unmatched_b:
            score = jaccard(tokens_a[ca], tokens_b[cb])
            if score > best_score:
                best_score, best_match = score, cb
        if best_score >= FUZZY_THRESHOLD:
            near_miss_records.append({
                'dataset_A': ds_a, 'column_A': ca,
                'dataset_B': ds_b, 'column_B': best_match,
                'jaccard_similarity': round(best_score, 3),
            })

overlap_df = pd.DataFrame(overlap_records)
overlap_df.to_csv(OUT / 'feature_overlap_matrix.csv', index=False)
print(f"\nSaved: {OUT / 'feature_overlap_matrix.csv'}")

near_miss_df = pd.DataFrame(near_miss_records).sort_values(
    'jaccard_similarity', ascending=False) if near_miss_records else pd.DataFrame()
near_miss_df.to_csv(OUT / 'feature_near_misses.csv', index=False)
print(f"Saved: {OUT / 'feature_near_misses.csv'} ({len(near_miss_df)} candidates)")

print(f"\n{'='*70}\nCIC-IDS2017 vs CSE-CIC-IDS2018 (both CICFlowMeter-derived) -- detail\n{'='*70}")
key_pair = overlap_df[
    ((overlap_df['dataset_A'] == 'CIC-IDS2017') & (overlap_df['dataset_B'] == 'CSE-CIC-IDS2018')) |
    ((overlap_df['dataset_A'] == 'CSE-CIC-IDS2018') & (overlap_df['dataset_B'] == 'CIC-IDS2017'))
]
if not key_pair.empty:
    print(key_pair.to_string(index=False))
else:
    print("  Pair not found in results -- check dataset naming in DATASET_CONFIG.")

key_near_misses = near_miss_df[
    ((near_miss_df.get('dataset_A') == 'CIC-IDS2017') & (near_miss_df.get('dataset_B') == 'CSE-CIC-IDS2018')) |
    ((near_miss_df.get('dataset_A') == 'CSE-CIC-IDS2018') & (near_miss_df.get('dataset_B') == 'CIC-IDS2017'))
] if not near_miss_df.empty else pd.DataFrame()

print(f"\nNear-miss candidates for this pair ({len(key_near_misses)}):")
if not key_near_misses.empty:
    print(key_near_misses.to_string(index=False))

# ── LaTeX table fragment for near-misses (top 10 by similarity) ────────────
print(f"\n{'='*70}\nLaTeX table fragment (near-miss examples, paste into MC3 response):\n{'='*70}")
print(r"""
\begin{table}[htbp]
\centering
\caption{Feature-Name Near-Misses Under Normalized Matching (examples)}
\label{tblFeatureNearMiss}
\begin{tabular*}{\linewidth}{@{}lll c@{}}
\toprule
\textbf{Dataset A / Column} & \textbf{Dataset B / Column} & \textbf{Jaccard} \\
\midrule""")
for _, row in near_miss_df.head(10).iterrows():
    print(f"{row['dataset_A']}: {row['column_A']} & {row['dataset_B']}: {row['column_B']} "
          f"& {row['jaccard_similarity']} \\\\")
print(r"""\bottomrule
\end{tabular*}
\end{table}
""")

print("\nNote: a non-zero 'normalized_matches' count for any pair indicates genuine "
      "feature-schema overlap between those datasets, which bears directly on whether "
      "cross-dataset transfer is possible at the feature level. The CIC-IDS2017 / "
      "CSE-CIC-IDS2018 pair is the most likely to show overlap, since both derive from "
      "the CICFlowMeter family.")
