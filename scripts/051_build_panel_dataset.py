"""
051_build_panel_dataset.py

Build the page-cycle-week panel dataset for econometric analysis.

Steps:
    1. Load all_revisions.csv and filter to House races.
    2. Apply washout period exclusion.
    3. Apply page creation censoring.
    4. Aggregate to Tuesday-Monday weeks.
    5. Create complete panel with zero-fill for missing weeks.
    6. Compute event time variables and baseline-centered outcomes.
    7. Merge state primary dates for Q2 analysis.

Input:
    data/processed_html_parsed/all_revisions.csv
    data/reference/state_primary_dates.csv

Output:
    data/analysis/panel_weekly.parquet
    data/analysis/panel_metadata.json
    data/analysis/censoring_report.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta
from typing import Tuple, Dict, List
import logging
import json
from tqdm import tqdm

# Import from election_cycles
from election_cycles import (
    get_election_day, get_washout_end, get_analysis_window,
    get_week_start_tuesday, date_to_week_relative_to_election,
    get_cycle_boundaries
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

# Election cycles to include
ELECTION_CYCLES = [2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]

# Near window length (L weeks before election)
NEAR_WINDOW_L = 18


def load_revisions_data() -> pd.DataFrame:
    """
    Load and filter revisions data to House races only.

    Returns:
        DataFrame with House revisions
    """
    logger.info("Loading all_revisions.csv...")
    revisions_path = PROCESSED_DIR / "all_revisions.csv"

    df = pd.read_csv(
        revisions_path,
        parse_dates=['timestamp'],
        dtype={
            'page_title': str,
            'election_cycle': int,
            'office': str,
            'state': str,
        }
    )

    logger.info(f"Loaded {len(df):,} revisions")

    # Filter to House only
    df = df[df['office'] == 'House'].copy()
    logger.info(f"Filtered to House: {len(df):,} revisions")

    # Extract date from timestamp
    df['date'] = df['timestamp'].dt.date

    return df


def get_page_first_edit(df: pd.DataFrame) -> pd.Series:
    """
    Get the global first edit date for each page (across all time).

    Args:
        df: Revisions DataFrame

    Returns:
        Series mapping page_title to first edit date
    """
    first_edits = df.groupby('page_title')['date'].min()
    return first_edits


def apply_washout_exclusion(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Exclude revisions in the washout period for each cycle.

    Washout period = from day after prior election through Monday after
    Congress convenes.

    Args:
        df: Revisions DataFrame

    Returns:
        Tuple of (filtered DataFrame, stats dictionary)
    """
    logger.info("Applying washout period exclusion...")

    stats = {
        'before': len(df),
        'excluded_by_cycle': {},
        'after': 0
    }

    mask = pd.Series(True, index=df.index)

    for cycle in ELECTION_CYCLES:
        # Get washout boundaries
        prior_election = get_election_day(cycle - 2)
        washout_start = prior_election + timedelta(days=1)  # Day after prior election
        washout_end = get_washout_end(cycle)

        # Find revisions in washout period for this cycle
        cycle_mask = df['election_cycle'] == cycle
        date_mask = (df['date'] >= washout_start) & (df['date'] <= washout_end)
        in_washout = cycle_mask & date_mask

        excluded_count = in_washout.sum()
        stats['excluded_by_cycle'][cycle] = excluded_count

        mask = mask & ~in_washout

        if excluded_count > 0:
            logger.info(f"  Cycle {cycle}: excluded {excluded_count:,} revisions in washout ({washout_start} to {washout_end})")

    df_filtered = df[mask].copy()
    stats['after'] = len(df_filtered)

    logger.info(f"Washout exclusion: {stats['before']:,} -> {stats['after']:,} revisions")

    return df_filtered, stats


def apply_page_creation_censoring(
    df: pd.DataFrame,
    global_first_edits: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply page creation censoring rule.

    Drop entire page-cycle if the page's first-ever edit falls within
    the analysis window (post-washout through election day).

    IMPORTANT: global_first_edits must be computed from the ORIGINAL data
    (before washout exclusion) to get the true first edit dates. Otherwise,
    pages whose first edits fall in the washout period will appear to have
    been "created" after washout ends, leading to spurious censoring.

    Args:
        df: Revisions DataFrame (post-washout exclusion)
        global_first_edits: Series mapping page_title to first edit date,
                           computed from ORIGINAL data before washout exclusion

    Returns:
        Tuple of (filtered DataFrame, censoring report DataFrame)
    """
    logger.info("Applying page creation censoring...")

    # Use the pre-computed global first edits (from original data)
    page_first_edit = global_first_edits

    # Create page-cycle pairs
    page_cycles = df[['page_title', 'election_cycle']].drop_duplicates()

    censoring_records = []

    for _, row in tqdm(page_cycles.iterrows(), total=len(page_cycles), desc="Checking censoring"):
        page = row['page_title']
        cycle = row['election_cycle']

        first_edit = page_first_edit.get(page)
        if first_edit is None:
            continue

        # Get analysis window boundaries
        analysis_start, election_day = get_analysis_window(cycle)

        # Check if first edit is within analysis window
        is_censored = (first_edit >= analysis_start) and (first_edit <= election_day)

        censoring_records.append({
            'page_title': page,
            'election_cycle': cycle,
            'first_edit': first_edit,
            'analysis_start': analysis_start,
            'election_day': election_day,
            'is_censored': is_censored
        })

    censoring_df = pd.DataFrame(censoring_records)

    # Create report
    censoring_summary = censoring_df.groupby('election_cycle').agg({
        'page_title': 'count',
        'is_censored': 'sum'
    }).rename(columns={
        'page_title': 'total_page_cycles',
        'is_censored': 'censored_page_cycles'
    })
    censoring_summary['retained_page_cycles'] = (
        censoring_summary['total_page_cycles'] - censoring_summary['censored_page_cycles']
    )

    logger.info("\nPage creation censoring by cycle:")
    for cycle in ELECTION_CYCLES:
        if cycle in censoring_summary.index:
            row = censoring_summary.loc[cycle]
            logger.info(f"  {cycle}: {row['total_page_cycles']} total, "
                       f"{row['censored_page_cycles']} censored, "
                       f"{row['retained_page_cycles']} retained")

    # Filter out censored page-cycles
    censored_pairs = set(
        censoring_df[censoring_df['is_censored']][['page_title', 'election_cycle']]
        .apply(tuple, axis=1)
    )

    mask = ~df.apply(
        lambda r: (r['page_title'], r['election_cycle']) in censored_pairs,
        axis=1
    )

    df_filtered = df[mask].copy()

    logger.info(f"Page creation censoring: {len(df):,} -> {len(df_filtered):,} revisions")

    return df_filtered, censoring_summary.reset_index()


def aggregate_to_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate revisions to weekly level using Tuesday-Monday weeks.

    Args:
        df: Revisions DataFrame

    Returns:
        DataFrame with weekly edit counts per page-cycle
    """
    logger.info("Aggregating to weekly level...")

    # Add week_start column
    df['week_start'] = df['date'].apply(get_week_start_tuesday)

    # Aggregate by page-cycle-week
    weekly = df.groupby([
        'page_title', 'election_cycle', 'state', 'week_start'
    ]).agg({
        'revid': 'count',           # Edit count
        'user': 'nunique',          # Unique editors
        'size_change': 'sum',       # Net size change
        'is_bot': 'sum',            # Bot edits
        'is_anonymous': 'sum',      # Anonymous edits
    }).reset_index()

    weekly.rename(columns={
        'revid': 'edit_count',
        'user': 'editor_count',
        'size_change': 'net_size_change',
        'is_bot': 'bot_edits',
        'is_anonymous': 'anon_edits'
    }, inplace=True)

    logger.info(f"Aggregated to {len(weekly):,} page-cycle-week observations")

    return weekly


def create_complete_panel(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Create complete panel by filling in zero-edit weeks.

    For each page-cycle, generate all valid weeks from analysis_start
    to election_day and merge with actual edit counts.

    Args:
        weekly: Weekly edit counts DataFrame

    Returns:
        Complete panel DataFrame
    """
    logger.info("Creating complete panel with zero-fill...")

    # Get unique page-cycles with their metadata
    page_cycles = weekly[['page_title', 'election_cycle', 'state']].drop_duplicates()

    all_rows = []

    for _, pc in tqdm(page_cycles.iterrows(), total=len(page_cycles), desc="Expanding panel"):
        page = pc['page_title']
        cycle = pc['election_cycle']
        state = pc['state']

        # Get analysis window
        analysis_start, election_day = get_analysis_window(cycle)

        # Generate all Tuesday week starts
        current_week = get_week_start_tuesday(analysis_start)
        election_week = get_week_start_tuesday(election_day)

        while current_week <= election_week:
            all_rows.append({
                'page_title': page,
                'election_cycle': cycle,
                'state': state,
                'week_start': current_week
            })
            current_week += timedelta(days=7)

    complete_df = pd.DataFrame(all_rows)

    # Merge with actual edit counts
    complete_df = complete_df.merge(
        weekly[['page_title', 'election_cycle', 'week_start',
                'edit_count', 'editor_count', 'net_size_change',
                'bot_edits', 'anon_edits']],
        on=['page_title', 'election_cycle', 'week_start'],
        how='left'
    )

    # Fill NaN with 0 for weeks with no edits
    fill_cols = ['edit_count', 'editor_count', 'net_size_change', 'bot_edits', 'anon_edits']
    complete_df[fill_cols] = complete_df[fill_cols].fillna(0).astype(int)

    logger.info(f"Complete panel: {len(complete_df):,} observations "
               f"({len(page_cycles):,} page-cycles)")

    return complete_df


def add_outcome_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add outcome variables Y_jt and P_jt.

    Args:
        df: Panel DataFrame

    Returns:
        DataFrame with outcome variables
    """
    logger.info("Adding outcome variables...")

    # E_jt: raw edit count (already present as edit_count)
    df['E_jt'] = df['edit_count']

    # Y_jt: log(1 + edit_count) - intensive margin
    df['Y_jt'] = np.log1p(df['edit_count'])

    # P_jt: indicator for any edits - extensive margin
    df['P_jt'] = (df['edit_count'] > 0).astype(int)

    return df


def add_event_time_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add event time variables for general election and primary.

    Args:
        df: Panel DataFrame

    Returns:
        DataFrame with event time variables
    """
    logger.info("Adding event time variables...")

    # k_general: weeks to general election (negative before)
    df['election_date'] = df['election_cycle'].apply(get_election_day)
    df['k_general'] = df.apply(
        lambda r: date_to_week_relative_to_election(r['week_start'], r['election_cycle']),
        axis=1
    )

    return df


def merge_primary_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge state primary dates and compute k_primary.

    Args:
        df: Panel DataFrame

    Returns:
        DataFrame with primary dates and k_primary
    """
    logger.info("Merging state primary dates...")

    primary_path = REFERENCE_DIR / "state_primary_dates.csv"

    if not primary_path.exists():
        logger.warning(f"Primary dates file not found: {primary_path}")
        logger.warning("Run 050_collect_primary_dates.py first to create this file")
        df['primary_date'] = pd.NaT
        df['k_primary'] = np.nan
        return df

    primary_df = pd.read_csv(primary_path)

    # Create state name to abbreviation mapping (reverse)
    # Note: The primary_dates file uses full state names, all_revisions uses abbreviations
    # Need to check which format is used
    sample_state = df['state'].iloc[0] if len(df) > 0 else None

    if sample_state and len(sample_state) == 2:
        # df uses abbreviations, need to convert primary_df state names to abbreviations
        STATE_ABBREV = {
            'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
            'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
            'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
            'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
            'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
            'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
            'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
            'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
            'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
            'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
            'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
            'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
            'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC'
        }
        primary_df['state_abbrev'] = primary_df['state'].map(STATE_ABBREV)
        merge_col = 'state_abbrev'
    else:
        merge_col = 'state'

    # Parse primary_date
    primary_df['primary_date'] = pd.to_datetime(primary_df['primary_date']).dt.date

    # Merge
    df = df.merge(
        primary_df[['state' if merge_col == 'state' else 'state_abbrev',
                    'election_year', 'primary_date', 'primary_type']].rename(
            columns={merge_col if merge_col != 'state' else 'state': 'state_merge',
                     'election_year': 'election_cycle'}
        ),
        left_on=['state', 'election_cycle'],
        right_on=['state_merge', 'election_cycle'],
        how='left'
    )

    if 'state_merge' in df.columns:
        df.drop(columns=['state_merge'], inplace=True)

    # Compute k_primary
    def compute_k_primary(row):
        if pd.isna(row['primary_date']):
            return np.nan
        primary = row['primary_date']
        week = row['week_start']
        # Weeks relative to primary (negative before)
        primary_week_start = get_week_start_tuesday(primary)
        return (week - primary_week_start).days // 7

    df['k_primary'] = df.apply(compute_k_primary, axis=1)

    # Count how many observations have primary dates
    has_primary = df['primary_date'].notna().sum()
    logger.info(f"Primary dates merged: {has_primary:,}/{len(df):,} observations have primary dates")

    return df


def compute_baseline_centered_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute baseline-centered outcomes for Q1 and Q2.

    Q1 (General Election Baseline): baseline is k_general <= -19
    Q2 (Primary Baseline): baseline is k_primary <= -19

    Args:
        df: Panel DataFrame

    Returns:
        DataFrame with baseline-centered outcomes
    """
    logger.info("Computing baseline-centered outcomes...")

    # Create page_cycle identifier
    df['page_cycle'] = df['page_title'] + '_' + df['election_cycle'].astype(str)

    # --- Q1: General Election Baseline ---
    # Baseline = k_general <= -(L+1) = -19 or earlier
    df['in_general_baseline'] = df['k_general'] <= -(NEAR_WINDOW_L + 1)

    # Compute baseline means per page-cycle
    baseline_means_G = df[df['in_general_baseline']].groupby('page_cycle').agg({
        'Y_jt': 'mean',
        'P_jt': 'mean'
    }).reset_index()
    baseline_means_G.columns = ['page_cycle', 'Y_bar_G', 'P_bar_G']

    df = df.merge(baseline_means_G, on='page_cycle', how='left')

    # Centered outcomes
    df['Y_tilde_G'] = df['Y_jt'] - df['Y_bar_G']
    df['P_tilde_G'] = df['P_jt'] - df['P_bar_G']

    # --- Q2: Primary Baseline ---
    # Baseline = k_primary <= -19
    df['in_primary_baseline'] = df['k_primary'] <= -(NEAR_WINDOW_L + 1)

    baseline_means_P = df[df['in_primary_baseline']].groupby('page_cycle').agg({
        'Y_jt': 'mean',
        'P_jt': 'mean'
    }).reset_index()
    baseline_means_P.columns = ['page_cycle', 'Y_bar_P', 'P_bar_P']

    df = df.merge(baseline_means_P, on='page_cycle', how='left')

    # Centered outcomes for primary
    df['Y_tilde_P'] = df['Y_jt'] - df['Y_bar_P']
    df['P_tilde_P'] = df['P_jt'] - df['P_bar_P']

    # Count page-cycles with valid baselines
    has_general_baseline = df.groupby('page_cycle')['in_general_baseline'].any().sum()
    has_primary_baseline = df.groupby('page_cycle')['in_primary_baseline'].any().sum()

    n_page_cycles = df['page_cycle'].nunique()
    logger.info(f"Page-cycles with general baseline: {has_general_baseline:,}/{n_page_cycles:,}")
    logger.info(f"Page-cycles with primary baseline: {has_primary_baseline:,}/{n_page_cycles:,}")

    return df


def add_calendar_week_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar week ID for fixed effects.

    Args:
        df: Panel DataFrame

    Returns:
        DataFrame with calendar_week_id
    """
    # Reference date: First Tuesday of 2006
    reference_date = date(2006, 1, 3)  # This was a Tuesday

    df['calendar_week_id'] = df['week_start'].apply(
        lambda w: (w - reference_date).days // 7
    )

    return df


def save_panel(df: pd.DataFrame, censoring_report: pd.DataFrame):
    """
    Save the panel dataset and metadata.

    Args:
        df: Complete panel DataFrame
        censoring_report: Censoring summary DataFrame
    """
    logger.info("Saving panel dataset...")

    # Save as parquet
    output_path = ANALYSIS_DIR / "panel_weekly.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved panel to {output_path}")

    # Save censoring report
    censoring_path = ANALYSIS_DIR / "censoring_report.csv"
    censoring_report.to_csv(censoring_path, index=False)
    logger.info(f"Saved censoring report to {censoring_path}")

    # Save metadata
    metadata = {
        'created': str(date.today()),
        'n_observations': len(df),
        'n_page_cycles': df['page_cycle'].nunique(),
        'election_cycles': ELECTION_CYCLES,
        'near_window_L': NEAR_WINDOW_L,
        'columns': list(df.columns),
        'k_general_range': [int(df['k_general'].min()), int(df['k_general'].max())],
    }

    metadata_path = ANALYSIS_DIR / "panel_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


def main():
    """Main entry point for panel construction."""
    logger.info("=" * 60)
    logger.info("BUILDING PANEL DATASET")
    logger.info("=" * 60)

    # Step 1: Load data
    df = load_revisions_data()

    # Step 1.5: Compute global first edits BEFORE washout exclusion
    # This is critical - if we compute first edits after washout exclusion,
    # pages whose true first edits are in the washout period will appear to
    # have been "created" after washout ends, leading to spurious censoring.
    logger.info("Computing global first edits (before washout exclusion)...")
    global_first_edits = get_page_first_edit(df)
    logger.info(f"  Earliest first edit in dataset: {global_first_edits.min()}")
    logger.info(f"  Pages with edits: {len(global_first_edits):,}")

    # Step 2: Apply washout exclusion
    df, washout_stats = apply_washout_exclusion(df)

    # Step 3: Apply page creation censoring (using pre-computed first edits)
    df, censoring_report = apply_page_creation_censoring(df, global_first_edits)

    # Step 4: Aggregate to weeks
    weekly = aggregate_to_weeks(df)

    # Step 5: Create complete panel
    panel = create_complete_panel(weekly)

    # Step 6: Add outcome variables
    panel = add_outcome_variables(panel)

    # Step 7: Add event time variables
    panel = add_event_time_variables(panel)

    # Step 8: Merge primary dates
    panel = merge_primary_dates(panel)

    # Step 9: Compute baseline-centered outcomes
    panel = compute_baseline_centered_outcomes(panel)

    # Step 10: Add calendar week ID
    panel = add_calendar_week_id(panel)

    # Step 11: Save
    save_panel(panel, censoring_report)

    # Summary
    print("\n" + "=" * 60)
    print("PANEL CONSTRUCTION COMPLETE")
    print("=" * 60)
    print(f"Observations: {len(panel):,}")
    print(f"Page-cycles: {panel['page_cycle'].nunique():,}")
    print(f"Weeks per page-cycle: ~{len(panel) / panel['page_cycle'].nunique():.1f}")
    print(f"k_general range: [{panel['k_general'].min()}, {panel['k_general'].max()}]")

    # Page-cycles by election cycle
    print("\nPage-cycles by election year:")
    pc_by_year = panel.groupby('election_cycle')['page_cycle'].nunique()
    for year, count in pc_by_year.items():
        print(f"  {year}: {count:,}")

    return panel


if __name__ == "__main__":
    main()
