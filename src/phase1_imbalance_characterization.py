#!/usr/bin/env python3
"""
phase1_imbalance_characterization.py
IDS Imbalance Study — RO2 Track A

Phase 1: Imbalance Characterization
Produces a quantitative Imbalance Profile for each of the 8 IDS benchmark datasets.

Usage:
    python phase1_imbalance_characterization.py

HPC Usage (run inside screen to survive SSH disconnect):
    screen -S phase1
    python phase1_imbalance_characterization.py 2>&1 | tee phase1_run.log
    # Detach: Ctrl+A then D
    # Reattach: screen -r phase1

Outputs (saved to ./outputs/phase1/):
    imbalance_profiles.csv
    class_distributions/   — per-dataset class count + temporal spread CSVs
    minority_profiles/     — per-dataset FDR + within-class variance CSVs
    figures/               — distribution charts, FDR charts, heatmap

Datasets: NSL-KDD, UNSW-NB15, CIC-IDS2017, HIKARI-2021,
          BoT-IoT, ToN-IoT, CSE-CIC-IDS2018, CICIoT2023
"""


# ── SECTION 1: Imports ─────────────────────────────────────────────────────────

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — safe for HPC
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from scipy.stats import entropy as scipy_entropy
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif

warnings.filterwarnings('ignore')
np.random.seed(42)


# ── SECTION 2: Configuration ───────────────────────────────────────────────────

# ── PATH CONFIGURATION ──────────────────────────────────────────────────────
# Update DATA_ROOT to your dataset directory.
# HPC example:   DATA_ROOT = Path('/scratch/ifrah/datasets')
# Local example: DATA_ROOT = Path('D:/Datasets')
DATA_ROOT = Path('/path/to/your/datasets')  # <-- UPDATE THIS

OUTPUT_ROOT = Path('../outputs/phase1')
for subdir in ['class_distributions', 'minority_profiles', 'figures']:
    (OUTPUT_ROOT / subdir).mkdir(parents=True, exist_ok=True)

# ── DATASET FILE PATHS ───────────────────────────────────────────────────────
# Each entry: dataset_name -> dict with file paths and label column.
# CIC-IDS2017 and CSE-CIC-IDS2018 are multi-file; list all CSVs.
DATASET_CONFIG = {
    'NSL-KDD': {
        'files': [DATA_ROOT / 'NSL-KDD' / 'KDDTrain+.txt'],
        'label_col': 'label',
        'has_header': False,
        'separator': ',',
    },
    'UNSW-NB15': {
        'files': [DATA_ROOT / 'UNSW-NB15' / 'UNSW_NB15_training-set.csv',
                  DATA_ROOT / 'UNSW-NB15' / 'UNSW_NB15_testing-set.csv'],
        'label_col': 'label',
        'has_header': True,
        'separator': ',',
    },
    'CIC-IDS2017': {
        'files': sorted((DATA_ROOT / 'CIC-IDS2017').glob('*.csv')),
        'label_col': ' Label',
        'has_header': True,
        'separator': ',',
    },
    'HIKARI-2021': {
        'files': [DATA_ROOT / 'HIKARI-2021' / 'HIKARI_2021.csv'],
        'label_col': 'Attack',
        'has_header': True,
        'separator': ',',
    },
    'BoT-IoT': {
        'files': sorted((DATA_ROOT / 'BoT-IoT').glob('*.csv')),
        'label_col': 'category',
        'has_header': True,
        'separator': ',',
    },
    'ToN-IoT': {
        'files': [DATA_ROOT / 'ToN-IoT' / 'Train_Test_Network.csv'],
        'label_col': 'type',
        'has_header': True,
        'separator': ',',
    },
    'CSE-CIC-IDS2018': {
        'files': sorted((DATA_ROOT / 'CSE-CIC-IDS2018').glob('*.csv')),
        'label_col': 'Label',
        'has_header': True,
        'separator': ',',
    },
    'CICIoT2023': {
        # Use the MERGED_CSV folder — 63 files, all labels present in each file.
        # Features are packet-window aggregates (window=10 or 100 packets),
        # not pure flow-level. Note this when comparing profiles across datasets.
        'files': sorted((DATA_ROOT / 'CICIoT2023' / 'MERGED_CSV').glob('*.csv')),
        'label_col': 'Label',
        'has_header': True,
        'separator': ',',
    },
}

# ── NSL-KDD COLUMN NAMES (no header in raw file) ────────────────────────────
NSL_KDD_COLS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]

# ── LEAKAGE-CORRECTED COLUMN DROP LISTS (ported from IDS-ReComm) ────────────
# Columns identified as leakage sources — dropped before any analysis.
LEAKAGE_COLS = {
    'NSL-KDD':        [],  # No leakage columns identified in IDS-ReComm
    'UNSW-NB15':      ['id', 'attack_cat'],
    'CIC-IDS2017':    ['Flow ID', ' Source IP', ' Source Port',
                       ' Destination IP', ' Destination Port', ' Timestamp'],
    'HIKARI-2021':    ['uid', 'originh', 'originp', 'resph', 'respp'],
    'BoT-IoT':        ['pkSeqID', 'stime', 'ltime', 'srcip', 'dstip',
                       'srcport', 'dstport', 'seq', 'attack'],
    'ToN-IoT':        ['ts', 'src_ip', 'dst_ip', 'src_port', 'dst_port'],
    'CSE-CIC-IDS2018':['Timestamp', 'Dst Port'],
    # CICIoT2023: no leakage columns identified in the MERGED_CSV feature set.
    # Features are aggregate statistics over packet windows — no raw IPs/ports/timestamps.
    'CICIoT2023':   [],
}

# ── STANDARDIZED LABEL TAXONOMY (RO1 mapping) ────────────────────────────────
# Maps each dataset's raw label values to a standardized category.
# 'normal' maps to 'Benign'; everything else maps to an attack family.
# Extend as needed when loading each dataset.
LABEL_TAXONOMY = {
    'NSL-KDD': {
        'normal': 'Benign',
        'neptune': 'DoS', 'smurf': 'DoS', 'pod': 'DoS', 'teardrop': 'DoS',
        'land': 'DoS', 'back': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS',
        'processtable': 'DoS', 'mailbomb': 'DoS',
        'ipsweep': 'Reconnaissance', 'portsweep': 'Reconnaissance',
        'nmap': 'Reconnaissance', 'satan': 'Reconnaissance',
        'mscan': 'Reconnaissance', 'saint': 'Reconnaissance',
        'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L',
        'multihop': 'R2L', 'named': 'R2L', 'phf': 'R2L', 'sendmail': 'R2L',
        'snmpgetattack': 'R2L', 'snmpguess': 'R2L', 'spy': 'R2L',
        'warezclient': 'R2L', 'warezmaster': 'R2L', 'worm': 'R2L', 'xlock': 'R2L',
        'xsnoop': 'R2L', 'httptunnel': 'R2L',
        'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R',
        'ps': 'U2R', 'rootkit': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R',
    },
    'UNSW-NB15': {
        # Named labels (training/testing CSVs with attack_cat column removed)
        'Normal': 'Benign', 'normal': 'Benign',
        'Fuzzers': 'Fuzzing', 'Analysis': 'Reconnaissance',
        'Backdoor': 'Backdoor', 'DoS': 'DoS', 'Exploits': 'Exploitation',
        'Generic': 'Generic', 'Reconnaissance': 'Reconnaissance',
        'Shellcode': 'Shellcode', 'Worms': 'Worm',
        # Binary integer labels (present in some UNSW-NB15 file variants)
        # 0 = Normal/Benign, 1 = Attack (generic)
        '0': 'Benign', '1': 'Attack',
        0: 'Benign', 1: 'Attack',
    },
    'CIC-IDS2017': {
        'BENIGN': 'Benign',
        'FTP-Patator': 'BruteForce', 'SSH-Patator': 'BruteForce',
        'DoS slowloris': 'DoS', 'DoS Slowhttptest': 'DoS',
        'DoS Hulk': 'DoS', 'DoS GoldenEye': 'DoS', 'Heartbleed': 'Exploitation',
        # Correct UTF-8 em-dash form
        'Web Attack – Brute Force': 'WebAttack',
        'Web Attack – XSS': 'WebAttack',
        'Web Attack – Sql Injection': 'WebAttack',
        # Mojibake form (â) produced when CSVs read without encoding='utf-8'
        'Web Attack â Brute Force': 'WebAttack',
        'Web Attack â XSS': 'WebAttack',
        'Web Attack â Sql Injection': 'WebAttack',
        'Infiltration': 'Infiltration',
        'Bot': 'Botnet',
        'DDoS': 'DDoS',
        'PortScan': 'Reconnaissance',
    },
    'HIKARI-2021': {
        'Benign': 'Benign', 'benign': 'Benign',
        'DoS': 'DoS', 'DDoS': 'DDoS',
        'Ransomware': 'Ransomware', 'Phishing': 'Phishing',
        'SQL Injection': 'WebAttack', 'XSS': 'WebAttack',
        'Insider Threats': 'InsiderThreat',
        # Variants present in actual HIKARI-2021 files
        'Bruteforce-XML': 'BruteForce',
        'Bruteforce': 'BruteForce',
        'Background': 'Background',   # non-attack background traffic
        'Probing': 'Reconnaissance',
        'XMRIGCC CryptoMiner': 'CryptoMining',
    },
    'BoT-IoT': {
        'Normal': 'Benign', 'normal': 'Benign',
        'DDoS': 'DDoS', 'DoS': 'DoS',
        'Reconnaissance': 'Reconnaissance',
        'Theft': 'DataExfiltration',
    },
    'ToN-IoT': {
        'normal': 'Benign',
        'backdoor': 'Backdoor', 'ddos': 'DDoS', 'dos': 'DoS',
        'injection': 'WebAttack', 'mitm': 'MITM', 'password': 'BruteForce',
        'ransomware': 'Ransomware', 'scanning': 'Reconnaissance',
        'xss': 'WebAttack',
    },
    'CICIoT2023': {
        'BENIGN': 'Benign',
        # DDoS variants
        'DDOS-ICMP_FLOOD': 'DDoS', 'DDOS-UDP_FLOOD': 'DDoS',
        'DDOS-TCP_FLOOD': 'DDoS', 'DDOS-PSHACK_FLOOD': 'DDoS',
        'DDOS-RSTFINFLOOD': 'DDoS', 'DDOS-SYN_FLOOD': 'DDoS',
        'DDOS-SYNONYMOUSIP_FLOOD': 'DDoS', 'DDOS-HTTP_FLOOD': 'DDoS',
        'DDOS-SLOWLORIS': 'DDoS', 'DDOS-ICMP_FRAGMENTATION': 'DDoS',
        'DDOS-ACK_FRAGMENTATION': 'DDoS', 'DDOS-UDP_FRAGMENTATION': 'DDoS',
        # DoS variants
        'DOS-UDP_FLOOD': 'DoS', 'DOS-TCP_FLOOD': 'DoS',
        'DOS-SYN_FLOOD': 'DoS', 'DOS-HTTP_FLOOD': 'DoS',
        # Mirai botnet variants
        'MIRAI-GREETH_FLOOD': 'Botnet', 'MIRAI-UDPPLAIN': 'Botnet',
        'MIRAI-GREIP_FLOOD': 'Botnet',
        # Reconnaissance variants
        'VULNERABILITYSCAN': 'Reconnaissance', 'RECON-HOSTDISCOVERY': 'Reconnaissance',
        'RECON-OSSCAN': 'Reconnaissance', 'RECON-PORTSCAN': 'Reconnaissance',
        'RECON-PINGSWEEP': 'Reconnaissance',
        # Spoofing → MITM (closest RO1 category)
        'MITM-ARPSPOOFING': 'MITM', 'DNS_SPOOFING': 'MITM',
        # Brute force
        'DICTIONARYBRUTEFORCE': 'BruteForce',
        # Web-based attacks
        'SQLINJECTION': 'WebAttack', 'COMMANDINJECTION': 'WebAttack',
        'XSS': 'WebAttack', 'UPLOADING_ATTACK': 'WebAttack',
        'BROWSERHIJACKING': 'WebAttack',
        # Backdoor
        'BACKDOOR_MALWARE': 'Backdoor',
    },
    'CSE-CIC-IDS2018': {
        'Benign': 'Benign', 'benign': 'Benign',
        # Brute force variants
        'FTP-BruteForce': 'BruteForce', 'SSH-BruteForce': 'BruteForce',
        'FTP-Bruteforce': 'BruteForce', 'SSH-Bruteforce': 'BruteForce',
        'Brute Force -Web': 'BruteForce', 'Brute Force -XSS': 'WebAttack',
        # DoS variants — two naming conventions present in raw files
        'DoS-GoldenEye': 'DoS', 'DoS-Slowloris': 'DoS',
        'DoS-SlowHTTPTest': 'DoS', 'DoS-Hulk': 'DoS',
        'DoS attacks-GoldenEye': 'DoS', 'DoS attacks-Slowloris': 'DoS',
        'DoS attacks-SlowHTTPTest': 'DoS', 'DoS attacks-Hulk': 'DoS',
        # Infiltration (note the typo present in raw files)
        'Infilteration': 'Infiltration', 'Infiltration': 'Infiltration',
        # Botnet
        'Bot': 'Botnet',
        # DDoS variants
        'DDoS attacks-LOIC-HTTP': 'DDoS',
        'DDOS attack-HOIC': 'DDoS', 'DDOS attack-LOIC-UDP': 'DDoS',
        # Web attacks
        'SQL Injection': 'WebAttack',
    },
}

print(f"Datasets configured: {list(DATASET_CONFIG.keys())}")
print(f"Output root: {OUTPUT_ROOT.resolve()}")

# ── SECTION 3: Dataset Loader ──────────────────────────────────────────────────

def load_dataset(name: str, config: dict) -> pd.DataFrame:
    """
    Load a dataset from one or more CSV files.
    Applies leakage column removal and label taxonomy mapping.
    Returns a DataFrame with a clean 'label' column (raw) and
    'label_std' column (standardized taxonomy).
    """
    dfs = []
    for fpath in config['files']:
        if not Path(fpath).exists():
            print(f"  [WARN] File not found, skipping: {fpath}")
            continue
        if name == 'NSL-KDD' and not config['has_header']:
            df = pd.read_csv(fpath, header=None, names=NSL_KDD_COLS,
                             sep=config['separator'])
            df.drop(columns=['difficulty'], inplace=True, errors='ignore')
        else:
            # Force utf-8 encoding to prevent mojibake on special characters
            # (e.g. em-dash in CIC-IDS2017 Web Attack label names).
            df = pd.read_csv(fpath, sep=config['separator'],
                             low_memory=False, encoding='utf-8',
                             on_bad_lines='skip')

        # CSE-CIC-IDS2018 CSV artifact: some rows contain the string "Label"
        # as a data value due to file concatenation. Drop them immediately.
        if name == 'CSE-CIC-IDS2018' and config['label_col'] in df.columns:
            artifact_mask = df[config['label_col']].astype(str).str.strip() == 'Label'
            n_artifact = artifact_mask.sum()
            if n_artifact > 0:
                df = df[~artifact_mask]
                print(f"  Dropped {n_artifact} CSV artifact rows (Label==\'Label\')")

        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No files loaded for dataset: {name}")

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {name}: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Strip column name whitespace early (CIC datasets)
    df.columns = df.columns.str.strip()

    # Rename label column to 'label' for consistency.
    # If the label_col differs from 'label' and a 'label' column already exists
    # (e.g. ToN-IoT has both 'type' and possibly 'label'), drop the stale one first.
    label_col = config['label_col'].strip()
    if label_col in df.columns and label_col != 'label':
        if 'label' in df.columns:
            df.drop(columns=['label'], inplace=True)
        df.rename(columns={label_col: 'label'}, inplace=True)
    elif label_col not in df.columns and 'label' not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in {name}. "
                       f"Available columns: {list(df.columns)}")

    # Drop leakage columns
    leakage = [c for c in LEAKAGE_COLS.get(name, []) if c in df.columns]
    if leakage:
        df.drop(columns=leakage, inplace=True)
        print(f"  Dropped {len(leakage)} leakage columns: {leakage}")

    # Ensure label column is string before stripping.
    # astype(str) converts actual NaN values to the string 'nan' —
    # replace these explicitly so they are caught by dropna below.
    df['label'] = df['label'].astype(str).str.strip()
    df['label'] = df['label'].replace({'nan': np.nan, 'NaN': np.nan, 'NULL': np.nan, '': np.nan})
    before_null = len(df)
    df.dropna(subset=['label'], inplace=True)
    if len(df) < before_null:
        print(f"  Dropped {before_null - len(df):,} rows with null/nan labels")

    # Apply standardized taxonomy mapping
    taxonomy = LABEL_TAXONOMY.get(name, {})
    df['label_std'] = df['label'].map(taxonomy)
    unmapped = df['label_std'].isna().sum()
    if unmapped > 0:
        unmapped_vals = df.loc[df['label_std'].isna(), 'label'].unique()
        print(f"  [WARN] {unmapped:,} rows with unmapped labels: {unmapped_vals}")
        # Fall back to raw label for unmapped values
        df['label_std'] = df['label_std'].fillna(df['label'])

    # Drop rows with inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(subset=['label'], inplace=True)
    if len(df) < before:
        print(f"  Dropped {before - len(df):,} rows with null labels")

    print(f"  Final shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"  Unique raw labels: {df['label'].nunique()} | "
          f"Standardized: {df['label_std'].nunique()}")
    return df


# ── SECTION 4: Imbalance Metric Functions ──────────────────────────────────────

def compute_distribution_metrics(label_series: pd.Series) -> dict:
    """
    Compute dataset-level imbalance distribution metrics.

    Returns:
        dict with keys:
          n_classes       : number of unique classes
          n_samples       : total sample count
          majority_class  : name of the largest class
          majority_count  : sample count of majority class
          minority_class  : name of the smallest class
          minority_count  : sample count of minority class
          IR              : Imbalance Ratio (majority / minority)
          shannon_entropy : Shannon entropy of class distribution (nats)
          entropy_norm    : Shannon entropy normalized by log(n_classes)
                            (0 = maximally imbalanced, 1 = perfectly balanced)
          gini_impurity   : Gini impurity of class distribution
          benign_pct      : percentage of samples labelled Benign/Normal
    """
    counts = label_series.value_counts()
    n = len(label_series)
    probs = counts / n

    sh_entropy = scipy_entropy(probs.values)  # natural log base
    max_entropy = np.log(len(counts)) if len(counts) > 1 else 1.0
    entropy_norm = sh_entropy / max_entropy if max_entropy > 0 else 0.0

    gini = 1.0 - np.sum(probs.values ** 2)

    # Benign percentage — look for common benign label names
    benign_keys = {'benign', 'normal', 'Benign', 'Normal', 'BENIGN'}
    benign_count = counts[counts.index.isin(benign_keys)].sum()
    benign_pct = 100.0 * benign_count / n

    return {
        'n_classes':       len(counts),
        'n_samples':       n,
        'majority_class':  counts.index[0],
        'majority_count':  int(counts.iloc[0]),
        'minority_class':  counts.index[-1],
        'minority_count':  int(counts.iloc[-1]),
        'IR':              round(counts.iloc[0] / counts.iloc[-1], 2),
        'shannon_entropy': round(sh_entropy, 4),
        'entropy_norm':    round(entropy_norm, 4),
        'gini_impurity':   round(gini, 4),
        'benign_pct':      round(benign_pct, 2),
    }


def compute_minority_profiles(df: pd.DataFrame,
                               label_col: str = 'label_std',
                               top_n_minority: int = 10) -> pd.DataFrame:
    """
    For each class (sorted by count ascending), compute:
      - count, pct
      - mean_within_class_variance: average variance of numeric features
        within that class (proxy for intra-class spread)
      - fisher_ratio: Fisher Discriminant Ratio vs. all other samples
        (between-class variance / within-class variance, averaged over features)
        Higher = more separable from the rest.

    Returns a DataFrame sorted by count ascending (rarest first).
    Only numeric columns are used for variance/FDR computations.
    """
    counts = df[label_col].value_counts().sort_values(ascending=True)
    n_total = len(df)

    # Robustly identify numeric columns using pd.to_numeric coercion.
    # This handles CSE-CIC-IDS2018 where some object-dtype columns contain
    # mostly numeric strings with occasional non-numeric entries.
    exclude = {label_col, 'label', 'label_std'}
    candidate_cols = [c for c in df.columns if c not in exclude]

    numeric_cols = []
    for c in candidate_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            # Try coercing — accept if >= 80% of values convert successfully
            coerced = pd.to_numeric(df[c], errors='coerce')
            if coerced.notna().mean() >= 0.8:
                df[c] = coerced
                numeric_cols.append(c)

    # Drop columns that are >50% NaN even after coercion
    numeric_cols = [c for c in numeric_cols
                    if df[c].notna().mean() > 0.5]

    if not numeric_cols:
        print("  [WARN] No usable numeric columns found for minority profiling.")
        return pd.DataFrame()

    # For very large datasets, compute global stats on a capped sample
    # to avoid materialising the full numeric block in memory.
    MAX_GLOBAL_SAMPLE = 500_000
    if len(df) > MAX_GLOBAL_SAMPLE:
        global_sample = df[numeric_cols].sample(MAX_GLOBAL_SAMPLE, random_state=42)
    else:
        global_sample = df[numeric_cols].copy()

    # Exclude non-finite values before computing any statistic. CICFlowMeter
    # emits Infinity in throughput columns when flow duration is zero, and a
    # single infinity makes the column mean infinite. Where such values are
    # confined to a subset of classes, the classes FREE of them yield
    # finite - infinite = infinite, while the classes containing them yield
    # infinite - infinite = NaN, which pandas skips. The classes unaffected by
    # the defect are therefore the ones whose statistics it corrupts.
    global_sample = global_sample.replace([np.inf, -np.inf], np.nan)

    global_mean = global_sample.mean()
    global_var  = global_sample.var().replace(0, np.nan)  # avoid /0
    del global_sample

    records = []
    for cls, cnt in counts.items():
        mask = df[label_col] == cls
        cls_df = df.loc[mask, numeric_cols]

        # For very small classes, skip if fewer than 2 samples (var undefined)
        if cnt < 2:
            records.append({
                'class':             cls,
                'count':             cnt,
                'pct':               round(100.0 * cnt / n_total, 4),
                'mean_within_var':   float('nan'),
                'mean_fisher_ratio': float('nan'),
            })
            continue

        within_var = cls_df.var()
        mean_within_var = float(within_var.mean())

        cls_mean = cls_df.replace([np.inf, -np.inf], np.nan).mean()
        common_cols = cls_mean.index.intersection(global_mean.index)
        fdr_per_feature = ((cls_mean[common_cols] - global_mean[common_cols]) ** 2
                           / global_var[common_cols].replace(np.nan, 1e-9))
        mean_fdr = float(fdr_per_feature.mean())

        records.append({
            'class':                cls,
            'count':                cnt,
            'pct':                  round(100.0 * cnt / n_total, 4),
            'mean_within_var':      round(mean_within_var, 4),
            'mean_fisher_ratio':    round(mean_fdr, 4),
        })

    return pd.DataFrame(records)


def compute_temporal_spread(df: pd.DataFrame,
                             label_col: str = 'label_std',
                             timestamp_col: str = None) -> pd.DataFrame:
    """
    If a timestamp column is available, compute for each class:
      - first_occurrence, last_occurrence
      - temporal_span_pct: fraction of total dataset time span occupied
        (0 = concentrated in one window, 1 = spread across entire capture)

    If no timestamp column is available or found, returns an empty DataFrame.
    Uses row index as a proxy for time order if no timestamp is provided.
    """
    # Try to find a timestamp column if not specified
    if timestamp_col is None:
        ts_candidates = [c for c in df.columns
                         if any(k in c.lower() for k in
                                ['time', 'timestamp', 'ts', 'stime', 'ltime'])]
        timestamp_col = ts_candidates[0] if ts_candidates else None

    if timestamp_col and timestamp_col in df.columns:
        try:
            ts = pd.to_numeric(df[timestamp_col], errors='coerce')
        except Exception:
            ts = None
    else:
        # Use row index as time proxy
        ts = pd.Series(range(len(df)), index=df.index)
        print("  [INFO] No timestamp column found — using row index as temporal proxy.")

    if ts is None or ts.isna().all():
        return pd.DataFrame()

    total_span = ts.max() - ts.min()
    if total_span == 0:
        return pd.DataFrame()

    records = []
    for cls in df[label_col].unique():
        mask = df[label_col] == cls
        cls_ts = ts[mask].dropna()
        if len(cls_ts) == 0:
            continue
        span = cls_ts.max() - cls_ts.min()
        records.append({
            'class':              cls,
            'first_occurrence':   cls_ts.min(),
            'last_occurrence':    cls_ts.max(),
            'temporal_span_pct':  round(100.0 * span / total_span, 2),
        })

    return pd.DataFrame(records).sort_values('temporal_span_pct')


# ── SECTION 5: Plotting Helpers ────────────────────────────────────────────────

def plot_class_distribution(counts: pd.Series, dataset_name: str,
                             save_dir: Path):
    """
    Horizontal bar chart of class counts (log scale).
    Saved to save_dir / f'{dataset_name}_class_dist.png'
    """
    fig, ax = plt.subplots(figsize=(10, max(4, len(counts) * 0.5)))
    colors = ['#2196F3' if 'benign' in str(c).lower() or 'normal' in str(c).lower()
              else '#E53935' for c in counts.index]
    counts.sort_values().plot(kind='barh', ax=ax, color=colors, edgecolor='white')
    ax.set_xscale('log')
    ax.set_xlabel('Sample Count (log scale)', fontsize=11)
    ax.set_title(f'{dataset_name} — Class Distribution', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    # Annotate with counts
    for i, (idx, val) in enumerate(counts.sort_values().items()):
        ax.text(val * 1.05, i, f'{val:,}', va='center', fontsize=8)
    plt.tight_layout()
    out = save_dir / f'{dataset_name}_class_dist.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {out.name}")


def plot_entropy_heatmap(profiles_df: pd.DataFrame, save_dir: Path):
    """
    Heatmap of key imbalance metrics across all datasets.
    """
    metrics = ['IR', 'entropy_norm', 'gini_impurity', 'benign_pct', 'n_classes']
    available = [m for m in metrics if m in profiles_df.columns]
    plot_df = profiles_df.set_index('dataset')[available].astype(float)

    # Normalize each column to 0-1 for visual comparability
    plot_norm = (plot_df - plot_df.min()) / (plot_df.max() - plot_df.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_norm) * 0.7)))
    sns.heatmap(plot_norm, annot=plot_df.round(2), fmt='g',
                cmap='RdYlGn_r', linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Normalized value (0=best, 1=worst)'})
    ax.set_title('Imbalance Profile Heatmap — All Datasets', fontsize=13,
                 fontweight='bold')
    ax.set_xlabel('')
    plt.tight_layout()
    out = save_dir / 'imbalance_heatmap_all_datasets.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Heatmap saved: {out.name}")


def plot_fdr_heatmap(minority_df: pd.DataFrame, dataset_name: str,
                     save_dir: Path):
    """
    Bar chart of Fisher Discriminant Ratio per class for a single dataset.
    """
    if minority_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(4, len(minority_df) * 0.5)))
    minority_df_sorted = minority_df.sort_values('mean_fisher_ratio')
    bars = ax.barh(minority_df_sorted['class'], minority_df_sorted['mean_fisher_ratio'],
                   color='#7B1FA2', edgecolor='white')
    ax.set_xlabel('Mean Fisher Discriminant Ratio', fontsize=11)
    ax.set_title(f'{dataset_name} — Separability by Class (FDR)', fontsize=13,
                 fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    out = save_dir / f'{dataset_name}_fdr.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  FDR figure saved: {out.name}")


# ── SECTION 6: Main Phase 1 Loop ───────────────────────────────────────────────

all_profiles   = []   # dataset-level summary rows
all_minority   = {}   # per-dataset minority profile DataFrames
all_temporal   = {}   # per-dataset temporal spread DataFrames
load_errors    = []   # datasets that failed to load

for ds_name, ds_config in DATASET_CONFIG.items():
    print(f"\n{'='*60}")
    print(f"Processing: {ds_name}")
    print(f"{'='*60}")

    # ── Load ────────────────────────────────────────────────────────────────
    try:
        df = load_dataset(ds_name, ds_config)
    except Exception as e:
        print(f"  [ERROR] Could not load {ds_name}: {e}")
        load_errors.append(ds_name)
        continue

    # ── Distribution metrics (using standardized labels) ─────────────────────
    print("\n  Computing distribution metrics...")
    dist_metrics = compute_distribution_metrics(df['label_std'])
    dist_metrics['dataset'] = ds_name
    all_profiles.append(dist_metrics)

    # Save class counts table
    counts_df = df['label_std'].value_counts().reset_index()
    counts_df.columns = ['class', 'count']
    counts_df['pct'] = (100.0 * counts_df['count'] / len(df)).round(4)
    counts_out = OUTPUT_ROOT / 'class_distributions' / f'{ds_name}_class_counts.csv'
    counts_df.to_csv(counts_out, index=False)
    print(f"  Class counts saved: {counts_out.name}")

    # Plot class distribution
    plot_class_distribution(
        df['label_std'].value_counts(),
        ds_name,
        OUTPUT_ROOT / 'figures'
    )

    # Print summary
    print(f"\n  Distribution Summary:")
    for k, v in dist_metrics.items():
        if k != 'dataset':
            print(f"    {k:25s}: {v}")

    # ── Minority class profiles ──────────────────────────────────────────────
    print("\n  Computing minority class profiles (FDR + within-class variance)...")
    try:
        min_df = compute_minority_profiles(df, label_col='label_std')
        all_minority[ds_name] = min_df
        min_out = OUTPUT_ROOT / 'minority_profiles' / f'{ds_name}_minority_profile.csv'
        min_df.to_csv(min_out, index=False)
        print(f"  Minority profiles saved: {min_out.name}")
        print(min_df.to_string(index=False))

        # Plot FDR
        plot_fdr_heatmap(min_df, ds_name, OUTPUT_ROOT / 'figures')
    except Exception as e:
        print(f"  [WARN] Minority profile failed: {e}")

    # ── Temporal spread ──────────────────────────────────────────────────────
    print("\n  Computing temporal spread...")
    try:
        temp_df = compute_temporal_spread(df, label_col='label_std')
        if not temp_df.empty:
            all_temporal[ds_name] = temp_df
            temp_out = OUTPUT_ROOT / 'class_distributions' / f'{ds_name}_temporal_spread.csv'
            temp_df.to_csv(temp_out, index=False)
            print(f"  Temporal spread saved: {temp_out.name}")
            print(temp_df.to_string(index=False))
    except Exception as e:
        print(f"  [WARN] Temporal spread failed: {e}")

    # Free memory before next dataset
    del df

print(f"\n{'='*60}")
print("Phase 1 loop complete.")
if load_errors:
    print(f"Datasets that failed to load: {load_errors}")

# ── SECTION 7: Aggregate Profile Table + Heatmap ───────────────────────────────

if not all_profiles:
    print("No profiles to aggregate — check dataset paths.")
else:
    profiles_df = pd.DataFrame(all_profiles)
    # Reorder columns for readability
    col_order = ['dataset', 'n_samples', 'n_classes', 'majority_class',
                 'majority_count', 'minority_class', 'minority_count',
                 'IR', 'shannon_entropy', 'entropy_norm',
                 'gini_impurity', 'benign_pct']
    profiles_df = profiles_df[[c for c in col_order if c in profiles_df.columns]]

    # Save master profile CSV
    profile_out = OUTPUT_ROOT / 'imbalance_profiles.csv'
    profiles_df.to_csv(profile_out, index=False)
    print(f"Master imbalance profile saved: {profile_out}")
    print("\n" + profiles_df.to_string(index=False))

    # Cross-dataset heatmap
    plot_entropy_heatmap(profiles_df, OUTPUT_ROOT / 'figures')

# ── SECTION 8: Auto-Interpretation Notes ───────────────────────────────────────

def interpret_profile(row: pd.Series) -> str:
    """
    Generate a one-paragraph plain-language interpretation of a dataset's
    imbalance profile for use in paper writing.
    """
    lines = []
    lines.append(f"**{row['dataset']}** ({row['n_samples']:,} samples, "
                 f"{row['n_classes']} classes):")

    # IR severity
    ir = row['IR']
    if ir < 10:
        ir_desc = "mild"
    elif ir < 100:
        ir_desc = "moderate"
    elif ir < 1000:
        ir_desc = "severe"
    else:
        ir_desc = "extreme"
    lines.append(f"  IR={ir} ({ir_desc} imbalance). "
                 f"Majority class: '{row['majority_class']}' "
                 f"({row['majority_count']:,} samples). "
                 f"Rarest class: '{row['minority_class']}' "
                 f"({row['minority_count']:,} samples).")

    # Entropy
    en = row['entropy_norm']
    en_desc = "near-uniform" if en > 0.8 else ("moderately spread" if en > 0.5
              else ("concentrated" if en > 0.2 else "highly concentrated"))
    lines.append(f"  Normalized entropy={en} → distribution is {en_desc}.")

    # Benign pct
    bp = row['benign_pct']
    if bp > 80:
        lines.append(f"  Benign traffic dominates ({bp}% of samples), "
                     f"typical of real-world capture but risks masking rare attacks.")
    elif bp < 20:
        lines.append(f"  Low benign proportion ({bp}%), suggesting a "
                     f"capture focused on attack traffic — reduced ecological validity.")
    else:
        lines.append(f"  Benign proportion is {bp}% — reasonable balance "
                     f"between normal and attack traffic.")

    return "\n".join(lines)


if 'profiles_df' in dir() and not profiles_df.empty:
    print("=" * 70)
    print("AUTO-GENERATED INTERPRETATION NOTES")
    print("=" * 70)
    for _, row in profiles_df.iterrows():
        print()
        print(interpret_profile(row))
    print()
    print("=" * 70)
else:
    print("Run Cell 7 first.")

print("\n✓ Phase 1 complete. Outputs saved to ./outputs/phase1/")
