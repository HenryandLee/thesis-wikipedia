"""
060_build_editor_cycle_dataset.py

Build editor-by-cycle datasets for EDA on editor geographic and partisan
portfolios.

Steps:
    1. Load all_revisions.csv, filter to House races, drop bots.
    2. Remap party edge cases (DFL, Democratic-NPL to Democratic).
    3. Compute editor-cycle geographic metrics (S, MS, N).
    4. Compute editor-cycle party metrics (N_D, N_R, PC, etc.).
    5. Assign revisions to windows (baseline vs. near-general).
    6. Output separate datasets for registered and IP editors.

Input:
    data/processed_html_parsed/all_revisions.csv

Output:
    data/analysis/editor_cycle_registered.parquet
    data/analysis/editor_cycle_ip.parquet
    data/analysis/editor_cycle_window_registered.parquet
    data/analysis/editor_cycle_window_ip.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta
import logging
import json

from election_cycles import (
    get_election_day, get_washout_end, get_analysis_window,
    get_week_start_tuesday, date_to_week_relative_to_election
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed_html_parsed"
REFERENCE_DIR = DATA_DIR / "reference"
ANALYSIS_DIR = DATA_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

ELECTION_CYCLES = [2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
NEAR_WINDOW_L = 18  # weeks before general election for near window
MIN_EDITS_THRESHOLD = 5  # m in the design doc


# =============================================================================
# PARTY MAPPING
# =============================================================================

# Map party variants to canonical labels
PARTY_MAP = {
    'DFL': 'Democratic',
    'Democratic-NPL': 'Democratic',
    'Democratic–NPL': 'Democratic',  # en-dash variant
}


def map_party(party: str) -> str:
    """Map party name to canonical label."""
    return PARTY_MAP.get(party, party)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_filter_revisions() -> pd.DataFrame:
    """Load all_revisions.csv, filter to House, drop bots, map parties."""
    logger.info("Loading all_revisions.csv...")
    df = pd.read_csv(
        PROCESSED_DIR / "all_revisions.csv",
        parse_dates=['timestamp'],
        dtype={'page_title': str, 'election_cycle': int, 'office': str,
               'state': str, 'party': str, 'user': str}
    )
    logger.info(f"Loaded {len(df):,} revisions")

    # Filter to House
    df = df[df['office'] == 'House'].copy()
    logger.info(f"House only: {len(df):,} revisions")

    # Drop bots
    df = df[~df['is_bot']].copy()
    logger.info(f"After dropping bots: {len(df):,} revisions")

    # Map party labels
    df['party_mapped'] = df['party'].apply(map_party)
    n_remapped = (df['party'] != df['party_mapped']).sum()
    logger.info(f"Remapped {n_remapped:,} party labels (DFL/D-NPL to Democratic)")

    # Extract date
    df['date'] = df['timestamp'].dt.date

    return df


# =============================================================================
# PRIMARY DATE LOADING & WINDOW ASSIGNMENT
# =============================================================================

def assign_windows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each revision to a window: 'baseline', 'near_general', or 'washout'.

    Windows are defined per cycle using general-election-relative weeks:
    - Washout: day after prior election through Monday after Congress convenes
    - Near-general: k_general in {-18, ..., -1} (18 weeks before general election)
    - Baseline: in analysis window, not washout, not near-general
    """
    logger.info("Assigning revisions to windows (near-general, vectorized)...")

    # Map cycle -> washout_end date
    washout_ends = {cycle: get_washout_end(cycle) for cycle in ELECTION_CYCLES}
    df['_washout_end'] = df['election_cycle'].map(washout_ends)

    # Identify washout revisions
    df['_in_washout'] = df['date'] <= df['_washout_end']

    # Compute k_general: weeks relative to general election (vectorized)
    # General election date is the same for all states within a cycle
    election_days = {cycle: get_election_day(cycle) for cycle in ELECTION_CYCLES}
    df['_election_day'] = df['election_cycle'].map(election_days)

    # Week start for each date and election day (Tuesday-based weeks)
    date_series = pd.to_datetime(df['date'].apply(str))
    election_series = pd.to_datetime(df['_election_day'].apply(str))

    date_weekday = date_series.dt.weekday
    election_weekday = election_series.dt.weekday

    date_week_start = date_series - pd.to_timedelta((date_weekday - 1) % 7, unit='D')
    election_week_start = election_series - pd.to_timedelta((election_weekday - 1) % 7, unit='D')

    df['_k_general'] = ((date_week_start - election_week_start).dt.days // 7).values

    # Near-general: k_general in [-18, -1]
    df['_in_near_general'] = (
        (df['_k_general'] >= -NEAR_WINDOW_L) &
        (df['_k_general'] <= -1)
    )

    # Assign windows
    df['window'] = 'baseline'  # default
    df.loc[df['_in_washout'], 'window'] = 'washout'
    df.loc[~df['_in_washout'] & df['_in_near_general'], 'window'] = 'near_general'

    # Clean up temp columns
    df.drop(columns=['_washout_end', '_in_washout', '_k_general',
                     '_in_near_general', '_election_day'],
            inplace=True)

    # Log window distribution
    window_counts = df['window'].value_counts()
    for w, c in window_counts.items():
        logger.info(f"  {w}: {c:,} revisions")

    return df


# =============================================================================
# EDITOR-CYCLE AGGREGATION
# =============================================================================

def build_editor_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build editor-cycle dataset with geographic and party metrics.

    Computes per editor-cycle:
    - N_ey: total edits (all parties)
    - S_ey: distinct states edited
    - MS_ey: modal-state share
    - N_D_ey, N_R_ey: edits to D/R pages
    - P_D_ey, P_R_ey: unique D/R pages edited
    - party_type, pi_ey, PC_ey
    - entry_cycle
    """
    logger.info("Building editor-cycle dataset...")

    # --- Geographic metrics ---
    geo = df.groupby(['user', 'election_cycle']).agg(
        N_ey=('revid', 'count'),
        S_ey=('state', 'nunique'),
    ).reset_index()

    # Modal-state share: max(edits in one state) / total edits
    state_counts = df.groupby(['user', 'election_cycle', 'state']).size().reset_index(name='n')
    modal_state = state_counts.groupby(['user', 'election_cycle']).agg(
        max_state_edits=('n', 'max'),
        total_edits=('n', 'sum')
    ).reset_index()
    modal_state['MS_ey'] = modal_state['max_state_edits'] / modal_state['total_edits']

    geo = geo.merge(
        modal_state[['user', 'election_cycle', 'MS_ey']],
        on=['user', 'election_cycle'],
        how='left'
    )

    # --- Party metrics (D/R only) ---
    dr_mask = df['party_mapped'].isin(['Democratic', 'Republican'])
    df_dr = df[dr_mask].copy()
    df_dr['is_D'] = (df_dr['party_mapped'] == 'Democratic').astype(int)
    df_dr['is_R'] = (df_dr['party_mapped'] == 'Republican').astype(int)

    party = df_dr.groupby(['user', 'election_cycle']).agg(
        N_D_ey=('is_D', 'sum'),
        N_R_ey=('is_R', 'sum'),
    ).reset_index()

    # Unique pages by party
    pages_D = df_dr[df_dr['is_D'] == 1].groupby(
        ['user', 'election_cycle']
    )['page_title'].nunique().reset_index(name='P_D_ey')

    pages_R = df_dr[df_dr['is_R'] == 1].groupby(
        ['user', 'election_cycle']
    )['page_title'].nunique().reset_index(name='P_R_ey')

    party = party.merge(pages_D, on=['user', 'election_cycle'], how='left')
    party = party.merge(pages_R, on=['user', 'election_cycle'], how='left')
    party[['P_D_ey', 'P_R_ey']] = party[['P_D_ey', 'P_R_ey']].fillna(0).astype(int)

    # --- Merge geographic + party ---
    ec = geo.merge(party, on=['user', 'election_cycle'], how='left')
    ec[['N_D_ey', 'N_R_ey', 'P_D_ey', 'P_R_ey']] = (
        ec[['N_D_ey', 'N_R_ey', 'P_D_ey', 'P_R_ey']].fillna(0).astype(int)
    )

    # --- Party type classification (only when N_D + N_R >= threshold) ---
    ec['N_DR_ey'] = ec['N_D_ey'] + ec['N_R_ey']

    def classify_party_type(row):
        if row['N_DR_ey'] < MIN_EDITS_THRESHOLD:
            return np.nan
        if row['N_D_ey'] > 0 and row['N_R_ey'] == 0:
            return 'D-specialist'
        if row['N_R_ey'] > 0 and row['N_D_ey'] == 0:
            return 'R-specialist'
        if row['N_D_ey'] > 0 and row['N_R_ey'] > 0:
            return 'Cross-party'
        return np.nan  # N_D == N_R == 0 but N_DR >= 5 shouldn't happen

    ec['party_type'] = ec.apply(classify_party_type, axis=1)

    # Smoothed party share and concentration index
    ec['pi_ey'] = (ec['N_D_ey'] + 1) / (ec['N_DR_ey'] + 2)
    ec['PC_ey'] = 2 * np.abs(ec['pi_ey'] - 0.5)

    # --- Entry cycle (first cycle with N >= threshold) ---
    qualified = ec[ec['N_ey'] >= MIN_EDITS_THRESHOLD][['user', 'election_cycle']]
    entry = qualified.groupby('user')['election_cycle'].min().reset_index(name='entry_cycle')
    ec = ec.merge(entry, on='user', how='left')

    # --- First cycle (earliest cycle with ANY activity, for newcomer classification) ---
    first = ec.groupby('user')['election_cycle'].min().reset_index(name='first_cycle')
    ec = ec.merge(first, on='user', how='left')

    # --- Add is_anonymous flag ---
    ec['is_anonymous'] = ec['user'].str.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')

    logger.info(f"Built editor-cycle dataset: {len(ec):,} editor-cycles")
    logger.info(f"  Registered: {(~ec['is_anonymous']).sum():,}")
    logger.info(f"  IP/anonymous: {ec['is_anonymous'].sum():,}")

    return ec


# =============================================================================
# WINDOW-LEVEL AGGREGATION (for Section C)
# =============================================================================

def build_editor_cycle_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build editor-cycle-window dataset for newcomer/regular analysis.

    Only considers baseline and near_general windows.
    Computes D/R edit counts per editor-cycle-window.
    """
    logger.info("Building editor-cycle-window dataset...")

    # Filter to baseline and near_general only
    df_win = df[df['window'].isin(['baseline', 'near_general'])].copy()

    # Flag D/R edits
    df_win['is_DR'] = df_win['party_mapped'].isin(['Democratic', 'Republican']).astype(int)
    df_win['is_D'] = (df_win['party_mapped'] == 'Democratic').astype(int)
    df_win['is_R'] = (df_win['party_mapped'] == 'Republican').astype(int)

    # Aggregate per editor-cycle-window
    ecw = df_win.groupby(['user', 'election_cycle', 'window']).agg(
        N_total=('revid', 'count'),
        N_DR=('is_DR', 'sum'),
        N_D=('is_D', 'sum'),
        N_R=('is_R', 'sum'),
        S=('state', 'nunique'),
        n_pages=('page_title', 'nunique'),
    ).reset_index()

    # Modal-state share per editor-cycle-window
    state_counts = df_win.groupby(
        ['user', 'election_cycle', 'window', 'state']
    ).size().reset_index(name='n')
    modal = state_counts.groupby(
        ['user', 'election_cycle', 'window']
    ).agg(max_state_edits=('n', 'max'), total_edits=('n', 'sum')).reset_index()
    modal['MS'] = modal['max_state_edits'] / modal['total_edits']
    ecw = ecw.merge(
        modal[['user', 'election_cycle', 'window', 'MS']],
        on=['user', 'election_cycle', 'window'],
        how='left',
    )

    # Add is_anonymous flag
    ecw['is_anonymous'] = ecw['user'].str.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')

    logger.info(f"Built editor-cycle-window dataset: {len(ecw):,} rows")

    return ecw


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("=" * 60)
    logger.info("BUILDING EDITOR-CYCLE DATASETS FOR EDA")
    logger.info("=" * 60)

    # Step 1: Load and filter revisions
    df = load_and_filter_revisions()

    # Step 2: Assign windows (baseline vs near-general-election)
    df = assign_windows(df)

    # Step 3: Build editor-cycle dataset
    ec = build_editor_cycle(df)

    # Step 4: Build editor-cycle-window dataset
    ecw = build_editor_cycle_window(df)

    # Step 5: Split by registered/IP and save
    logger.info("Saving datasets...")

    # Editor-cycle
    ec_reg = ec[~ec['is_anonymous']].copy()
    ec_ip = ec[ec['is_anonymous']].copy()
    ec_reg.to_parquet(ANALYSIS_DIR / "editor_cycle_registered.parquet", index=False)
    ec_ip.to_parquet(ANALYSIS_DIR / "editor_cycle_ip.parquet", index=False)
    logger.info(f"  editor_cycle_registered: {len(ec_reg):,} rows")
    logger.info(f"  editor_cycle_ip: {len(ec_ip):,} rows")

    # Editor-cycle-window
    ecw_reg = ecw[~ecw['is_anonymous']].copy()
    ecw_ip = ecw[ecw['is_anonymous']].copy()
    ecw_reg.to_parquet(ANALYSIS_DIR / "editor_cycle_window_registered.parquet", index=False)
    ecw_ip.to_parquet(ANALYSIS_DIR / "editor_cycle_window_ip.parquet", index=False)
    logger.info(f"  editor_cycle_window_registered: {len(ecw_reg):,} rows")
    logger.info(f"  editor_cycle_window_ip: {len(ecw_ip):,} rows")

    # Summary
    print("\n" + "=" * 60)
    print("EDITOR-CYCLE DATASET CONSTRUCTION COMPLETE")
    print("=" * 60)

    print(f"\nEditor-cycle dataset:")
    print(f"  Total editor-cycles: {len(ec):,}")
    print(f"  Registered: {len(ec_reg):,}")
    print(f"  IP/anonymous: {len(ec_ip):,}")

    print(f"\nWith N >= {MIN_EDITS_THRESHOLD}:")
    qual_reg = ec_reg[ec_reg['N_ey'] >= MIN_EDITS_THRESHOLD]
    qual_ip = ec_ip[ec_ip['N_ey'] >= MIN_EDITS_THRESHOLD]
    print(f"  Registered: {len(qual_reg):,}")
    print(f"  IP/anonymous: {len(qual_ip):,}")

    print(f"\nParty type distribution (registered, N_DR >= {MIN_EDITS_THRESHOLD}):")
    pt = ec_reg[ec_reg['N_DR_ey'] >= MIN_EDITS_THRESHOLD]['party_type'].value_counts()
    for ptype, count in pt.items():
        print(f"  {ptype}: {count:,}")

    print(f"\nBy cycle (registered, N >= {MIN_EDITS_THRESHOLD}):")
    for cycle in ELECTION_CYCLES:
        n = len(qual_reg[qual_reg['election_cycle'] == cycle])
        print(f"  {cycle}: {n:,}")


if __name__ == "__main__":
    main()
