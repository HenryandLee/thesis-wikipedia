"""
031_process_data.py

Process Wikipedia revision histories into analysis-ready datasets.
Joins revision data with candidate metadata, filters revisions to
cycles when the candidate was on the ballot, identifies bots, and
aggregates to daily and candidate-cycle summaries.

Key design decisions:
    1. Only includes revisions when the candidate was on the ballot
       for that cycle (cycle-specific filtering).
    2. All metadata is cycle-specific; no cross-cycle aggregation.
    3. Cycles begin with 2008 (2006-11-08 start date).
    4. PVI parsed to numeric: R+ positive, D+ negative, missing = NaN.

Input:
    data/raw/html_parsed_candidates/*.csv
    data/raw/revisions_html_parsed/*.json

Output:
    data/processed_html_parsed/all_revisions.csv
    data/processed_html_parsed/daily_edits.csv
    data/processed_html_parsed/daily_edits_house.csv
    data/processed_html_parsed/daily_edits_senate.csv
    data/processed_html_parsed/candidate_cycle_stats.csv
    data/processed_html_parsed/editor_profiles.csv
    data/processed_html_parsed/processing_summary.md
"""
import json
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
import logging
from collections import defaultdict
import re

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from wiki_api import WikipediaAPI, load_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# ELECTION CYCLE DEFINITIONS
# =============================================================================

# Federal election days (first Tuesday after first Monday in November)
ELECTION_DATES = {
    2008: date(2008, 11, 4),
    2010: date(2010, 11, 2),
    2012: date(2012, 11, 6),
    2014: date(2014, 11, 4),
    2016: date(2016, 11, 8),
    2018: date(2018, 11, 6),
    2020: date(2020, 11, 3),
    2022: date(2022, 11, 8),
    2024: date(2024, 11, 5),
}

# 2-year non-overlapping cycles (day after previous election to current election)
# Note: 2008 cycle starts from 2006-11-08 (no 2006 candidate data available)
CYCLES = {
    2008: (date(2006, 11, 8), date(2008, 11, 4)),
    2010: (date(2008, 11, 5), date(2010, 11, 2)),
    2012: (date(2010, 11, 3), date(2012, 11, 6)),
    2014: (date(2012, 11, 7), date(2014, 11, 4)),
    2016: (date(2014, 11, 5), date(2016, 11, 8)),
    2018: (date(2016, 11, 9), date(2018, 11, 6)),
    2020: (date(2018, 11, 7), date(2020, 11, 3)),
    2022: (date(2020, 11, 4), date(2022, 11, 8)),
    2024: (date(2022, 11, 9), date(2024, 11, 5)),
    2026: (date(2024, 11, 6), date(2026, 12, 31)),  # For post-2024 revisions
}


def get_cycle_for_date(dt):
    """
    Determine which election cycle a date belongs to.

    Args:
        dt: datetime.date or datetime.datetime object

    Returns:
        Election year (2008, 2010, ..., 2024, 2026) or None if before 2006-11-08
    """
    if isinstance(dt, datetime):
        dt = dt.date()

    # Exclude revisions before first cycle
    if dt < CYCLES[2008][0]:
        return None

    # Find matching cycle
    for year in sorted(CYCLES.keys()):
        start, end = CYCLES[year]
        if start <= dt <= end:
            return year

    # Should not happen if CYCLES is properly defined
    return None


# =============================================================================
# PVI PARSING
# =============================================================================

def parse_pvi(pvi_string):
    """
    Parse Cook Partisan Voting Index to numeric score.

    Convention:
    - Republican lean: positive (R+15 -> 15.0)
    - Democratic lean: negative (D+3 -> -3.0)
    - Even: 0.0
    - Missing/NaN: NaN (not 0!)

    Args:
        pvi_string: String like 'R+15', 'D+3', 'EVEN', or None

    Returns:
        Float value or np.nan
    """
    if pd.isna(pvi_string) or pvi_string == '':
        return np.nan

    pvi_string = str(pvi_string).strip().upper()

    if pvi_string in ['EVEN', 'R+0', 'D+0']:
        return 0.0

    # Match pattern like R+15 or D+3
    match = re.match(r'([RD])\+(\d+)', pvi_string)
    if match:
        party, value = match.groups()
        numeric_value = float(value)
        return numeric_value if party == 'R' else -numeric_value

    logger.warning(f"Could not parse PVI: '{pvi_string}'")
    return np.nan


# =============================================================================
# METADATA LOADING FROM CSVs
# =============================================================================

def extract_page_title_from_url(url):
    """Extract Wikipedia page title from URL."""
    if pd.isna(url) or not url:
        return None

    # URL format: https://en.wikipedia.org/wiki/Page_Title
    if '/wiki/' in url:
        return url.split('/wiki/')[-1]

    return None


def load_candidate_metadata_from_csvs(csv_dir: Path) -> dict:
    """
    Load candidate metadata from HTML-parsed CSV files.

    Builds a lookup structure: {page_title: {cycle_year: {metadata_dict}}}

    Each candidate-cycle has:
    - office: 'House' or 'Senate'
    - state: State name
    - district: District name (House) or None (Senate)
    - party: Party affiliation
    - pvi: Original PVI string (e.g., 'R+15')
    - pvi_numeric: Parsed numeric PVI
    - is_incumbent: Boolean
    - is_winner: Boolean
    - vote_percentage: Float
    - election_type: 'general' or 'primary'

    Args:
        csv_dir: Directory containing year_office_candidates.csv files

    Returns:
        Dictionary: {page_title: {year: metadata_dict}}
    """
    metadata_lookup = defaultdict(dict)

    csv_files = sorted(csv_dir.glob('*.csv'))
    logger.info(f"Loading metadata from {len(csv_files)} CSV files...")

    for csv_file in csv_files:
        # Parse filename: 2016_house_candidates.csv -> year=2016, office='House'
        filename = csv_file.stem
        parts = filename.split('_')

        if len(parts) < 2:
            logger.warning(f"Skipping file with unexpected name: {csv_file.name}")
            continue

        year = int(parts[0])
        office_type = parts[1].capitalize()  # 'house' -> 'House', 'senate' -> 'Senate'

        logger.info(f"  Loading {year} {office_type} candidates...")

        df = pd.read_csv(csv_file)

        # Extract page titles from wikipedia_url column
        df['page_title'] = df['wikipedia_url'].apply(extract_page_title_from_url)

        # Filter to rows with valid Wikipedia URLs
        df_valid = df[df['page_title'].notna()].copy()

        logger.info(f"    Found {len(df_valid)} candidates with Wikipedia pages")

        # Store metadata for each candidate
        for _, row in df_valid.iterrows():
            page_title = row['page_title']

            # Build metadata dict for this candidate-cycle
            metadata = {
                'office': office_type,
                'state': row.get('state'),
                'party': row.get('party'),
                'pvi': row.get('pvi'),
                'pvi_numeric': parse_pvi(row.get('pvi')),
                'is_incumbent': bool(row.get('is_incumbent', False)),
                'is_winner': bool(row.get('is_winner', False)),
                'vote_percentage': row.get('vote_percentage'),
                'election_type': row.get('election_type', 'general'),
            }

            # Add office-specific fields
            if office_type == 'House':
                metadata['district'] = row.get('district')
            else:
                metadata['district'] = None

            # Store in lookup
            metadata_lookup[page_title][year] = metadata

    logger.info(f"\nLoaded metadata for {len(metadata_lookup)} unique candidates")

    # Print distribution by cycle
    cycle_counts = defaultdict(int)
    for page_title, cycles in metadata_lookup.items():
        for year in cycles.keys():
            cycle_counts[year] += 1

    logger.info("\nCandidates per cycle:")
    for year in sorted(cycle_counts.keys()):
        logger.info(f"  {year}: {cycle_counts[year]} candidates")

    return dict(metadata_lookup)


# =============================================================================
# REVISION PROCESSING
# =============================================================================

def process_revisions_to_timeseries(revisions_dir: Path, metadata_lookup: dict) -> pd.DataFrame:
    """
    Process revision files and enrich with cycle-specific metadata.

    IMPORTANT: Only includes revisions where candidate was on ballot for that cycle.

    Example:
    - Candidate ran in 2016, 2020 (not 2018)
    - Revisions from 2017-05-10 fall in 2018 cycle
    - Since candidate NOT on ballot in 2018 -- revision excluded

    Args:
        revisions_dir: Directory containing revision JSON files
        metadata_lookup: Candidate metadata from load_candidate_metadata_from_csvs()

    Returns:
        Tuple of (DataFrame with all revisions + cycle-specific metadata, stats dict)
    """
    all_revisions = []

    revision_files = list(revisions_dir.glob('*.json'))
    logger.info(f"\nProcessing {len(revision_files)} revision files...")

    stats = {
        'total_revisions_read': 0,
        'revisions_before_2006': 0,
        'revisions_no_cycle': 0,
        'revisions_candidate_not_on_ballot': 0,
        'revisions_kept': 0,
    }

    for i, revision_file in enumerate(revision_files, 1):
        if i % 100 == 0:
            logger.info(f"  Processed {i}/{len(revision_files)} files...")

        try:
            data = load_json(revision_file)
            page_title = data['page_title']

            # Get candidate's cycles from metadata
            candidate_cycles = metadata_lookup.get(page_title, {})

            for rev in data['revisions']:
                stats['total_revisions_read'] += 1

                # Parse timestamp
                timestamp = pd.to_datetime(rev['timestamp'])

                # Determine cycle
                cycle_year = get_cycle_for_date(timestamp)

                # Filter 1: Exclude revisions before 2006-11-08
                if cycle_year is None:
                    stats['revisions_before_2006'] += 1
                    continue

                # Filter 2: Only keep if candidate was on ballot this cycle
                if cycle_year not in candidate_cycles:
                    stats['revisions_candidate_not_on_ballot'] += 1
                    continue

                # Get cycle-specific metadata
                cycle_metadata = candidate_cycles[cycle_year]

                # Build revision record with metadata
                revision_record = {
                    # Original revision data
                    'page_title': page_title,
                    'revid': rev['revid'],
                    'parentid': rev['parentid'],
                    'timestamp': timestamp,
                    'user': rev['user'],
                    'userid': rev['userid'],
                    'size': rev['size'],
                    'comment': rev['comment'],
                    'minor': rev['minor'],
                    'tags': ','.join(rev['tags']) if rev['tags'] else '',

                    # Cycle information
                    'election_cycle': cycle_year,

                    # Cycle-specific metadata
                    'office': cycle_metadata['office'],
                    'state': cycle_metadata['state'],
                    'district': cycle_metadata['district'],
                    'party': cycle_metadata['party'],
                    'pvi': cycle_metadata['pvi'],
                    'pvi_numeric': cycle_metadata['pvi_numeric'],
                    'candidate_was_incumbent': cycle_metadata['is_incumbent'],
                    'candidate_won_election': cycle_metadata['is_winner'],
                    'vote_percentage': cycle_metadata['vote_percentage'],
                    'election_type': cycle_metadata['election_type'],
                }

                all_revisions.append(revision_record)
                stats['revisions_kept'] += 1

        except Exception as e:
            logger.error(f"Error processing {revision_file.name}: {e}")

    # Create DataFrame
    df = pd.DataFrame(all_revisions)

    if len(df) > 0:
        # Add temporal features
        df['date'] = df['timestamp'].dt.date
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        df['weekday'] = df['timestamp'].dt.dayofweek

        # Calculate size changes (within each page)
        df = df.sort_values(['page_title', 'timestamp'])
        df['size_change'] = df.groupby('page_title')['size'].diff()

        # Identify anonymous users
        df['is_anonymous'] = df['userid'] == 0

        # Initialize bot flag (will be set by API query)
        df['is_bot'] = False

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

    # Log statistics
    logger.info("\n" + "="*60)
    logger.info("REVISION FILTERING STATISTICS")
    logger.info("="*60)
    logger.info(f"Total revisions read: {stats['total_revisions_read']:,}")
    logger.info(f"  Excluded (before 2006-11-08): {stats['revisions_before_2006']:,}")
    logger.info(f"  Excluded (candidate not on ballot): {stats['revisions_candidate_not_on_ballot']:,}")
    logger.info(f"  KEPT for analysis: {stats['revisions_kept']:,}")
    logger.info(f"  Keep rate: {100*stats['revisions_kept']/stats['total_revisions_read']:.1f}%")

    return df, stats


# =============================================================================
# BOT IDENTIFICATION
# =============================================================================

def identify_bots_from_api(usernames: list, language: str = 'en') -> set:
    """
    Query Wikipedia API to identify bot accounts.

    Args:
        usernames: List of unique usernames
        language: Wikipedia language code

    Returns:
        Set of usernames that are bots
    """
    logger.info(f"\nQuerying Wikipedia API to identify bots among {len(usernames)} unique users...")

    api = WikipediaAPI(language=language)
    bots = api.identify_bots(usernames)

    logger.info(f"Identified {len(bots)} bot accounts")
    if bots:
        logger.info(f"Bot accounts: {sorted(bots)[:10]}{'...' if len(bots) > 10 else ''}")

    return bots


# =============================================================================
# AGGREGATION FUNCTIONS
# =============================================================================

def create_daily_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create daily aggregated statistics with metadata.

    Returns one row per page-date with cycle-specific metadata.
    """
    daily_stats = df.groupby(['page_title', 'date']).agg({
        'revid': 'count',
        'user': 'nunique',
        'size_change': 'sum',
        'minor': 'sum',
        'is_anonymous': 'sum',
        'is_bot': 'sum',
        # Metadata (same for all revisions on this page-date)
        'election_cycle': 'first',
        'office': 'first',
        'state': 'first',
        'district': 'first',
        'party': 'first',
        'pvi': 'first',
        'pvi_numeric': 'first',
        'candidate_was_incumbent': 'first',
        'candidate_won_election': 'first',
    }).reset_index()

    daily_stats.columns = [
        'page_title', 'date', 'num_edits', 'unique_editors',
        'net_size_change', 'minor_edits', 'anonymous_edits', 'bot_edits',
        'election_cycle', 'office', 'state', 'district', 'party',
        'pvi', 'pvi_numeric', 'candidate_was_incumbent', 'candidate_won_election'
    ]

    return daily_stats


def create_candidate_cycle_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create statistics for each candidate-cycle combination.

    Returns one row per candidate-cycle with both metadata and aggregated stats.
    """
    cycle_stats = df.groupby(['page_title', 'election_cycle']).agg({
        'revid': 'count',
        'user': 'nunique',
        'size_change': 'sum',
        'minor': 'sum',
        'is_anonymous': 'sum',
        'is_bot': 'sum',
        'timestamp': ['min', 'max'],
        # Metadata (same for all revisions in this candidate-cycle)
        'office': 'first',
        'state': 'first',
        'district': 'first',
        'party': 'first',
        'pvi': 'first',
        'pvi_numeric': 'first',
        'candidate_was_incumbent': 'first',
        'candidate_won_election': 'first',
        'vote_percentage': 'first',
        'election_type': 'first',
    }).reset_index()

    cycle_stats.columns = [
        'page_title', 'election_cycle',
        'total_revisions', 'unique_editors', 'net_size_change',
        'minor_edits', 'anonymous_edits', 'bot_edits',
        'first_edit', 'last_edit',
        'office', 'state', 'district', 'party', 'pvi', 'pvi_numeric',
        'candidate_was_incumbent', 'candidate_won_election',
        'vote_percentage', 'election_type'
    ]

    # Calculate editing span
    cycle_stats['editing_span_days'] = (
        pd.to_datetime(cycle_stats['last_edit']) -
        pd.to_datetime(cycle_stats['first_edit'])
    ).dt.days

    # Sort by cycle and total revisions
    cycle_stats = cycle_stats.sort_values(
        ['election_cycle', 'total_revisions'],
        ascending=[True, False]
    )

    return cycle_stats


def create_editor_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create editor-level aggregated data.
    """
    editor_stats = df.groupby(['page_title', 'user']).agg({
        'revid': 'count',
        'size_change': 'sum',
        'minor': 'sum',
        'timestamp': ['min', 'max']
    }).reset_index()

    editor_stats.columns = [
        'page_title', 'user', 'num_edits', 'total_size_change',
        'minor_edits', 'first_edit', 'last_edit'
    ]

    # Calculate editing span
    editor_stats['editing_span_days'] = (
        pd.to_datetime(editor_stats['last_edit']) -
        pd.to_datetime(editor_stats['first_edit'])
    ).dt.days

    # Calculate average edit size
    editor_stats['avg_size_change'] = (
        editor_stats['total_size_change'] / editor_stats['num_edits']
    )

    return editor_stats


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""

    logger.info("=" * 80)
    logger.info("PROCESSING HTML-PARSED CANDIDATE REVISION DATA")
    logger.info("=" * 80)

    # Setup paths
    base_dir = Path(__file__).parent.parent
    csv_dir = base_dir / 'data' / 'raw' / 'html_parsed_candidates'
    revisions_dir = base_dir / 'data' / 'raw' / 'revisions_html_parsed'
    output_dir = base_dir / 'data' / 'processed_html_parsed'

    # Clear old processed data
    if output_dir.exists():
        import shutil
        logger.info(f"\nClearing old processed data from {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nInput directories:")
    logger.info(f"  CSVs: {csv_dir}")
    logger.info(f"  Revisions: {revisions_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Step 1: Load candidate metadata from CSVs
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Loading candidate metadata from CSVs")
    logger.info("="*60)
    metadata_lookup = load_candidate_metadata_from_csvs(csv_dir)

    # Step 2: Process revisions with metadata enrichment
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Processing revisions with metadata enrichment")
    logger.info("="*60)
    df_revisions, stats = process_revisions_to_timeseries(revisions_dir, metadata_lookup)

    if len(df_revisions) == 0:
        logger.error("No revisions found after filtering. Exiting.")
        return

    logger.info(f"\nKept {len(df_revisions):,} revisions for analysis")

    # Step 3: Identify bots via Wikipedia API
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Identifying bot accounts via Wikipedia API")
    logger.info("="*60)
    unique_users = df_revisions['user'].unique().tolist()
    bot_usernames = identify_bots_from_api(unique_users, language='en')

    df_revisions['is_bot'] = df_revisions['user'].isin(bot_usernames)
    logger.info(f"Marked {df_revisions['is_bot'].sum():,} edits as bot edits "
                f"({df_revisions['is_bot'].mean()*100:.1f}% of total)")

    # Step 4: Create output datasets
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Creating output datasets")
    logger.info("="*60)

    # 4.1: Save full revisions
    logger.info("\n  4.1: Saving all_revisions.csv...")
    revisions_csv = output_dir / 'all_revisions.csv'
    df_revisions.to_csv(revisions_csv, index=False, encoding='utf-8')
    logger.info(f"       Saved {len(df_revisions):,} revisions to {revisions_csv.name}")

    # 4.2: Create daily aggregates
    logger.info("\n  4.2: Creating daily aggregates...")
    df_daily = create_daily_aggregates(df_revisions)
    daily_csv = output_dir / 'daily_edits.csv'
    df_daily.to_csv(daily_csv, index=False, encoding='utf-8')
    logger.info(f"       Created {len(df_daily):,} daily records")
    logger.info(f"       Saved to {daily_csv.name}")

    # 4.3: Create daily aggregates by office
    logger.info("\n  4.3: Creating office-specific daily aggregates...")
    df_daily_house = df_daily[df_daily['office'] == 'House']
    df_daily_senate = df_daily[df_daily['office'] == 'Senate']

    house_csv = output_dir / 'daily_edits_house.csv'
    senate_csv = output_dir / 'daily_edits_senate.csv'

    df_daily_house.to_csv(house_csv, index=False, encoding='utf-8')
    df_daily_senate.to_csv(senate_csv, index=False, encoding='utf-8')

    logger.info(f"       House: {len(df_daily_house):,} records, saved to {house_csv.name}")
    logger.info(f"       Senate: {len(df_daily_senate):,} records, saved to {senate_csv.name}")

    # 4.4: Create candidate-cycle statistics
    logger.info("\n  4.4: Creating candidate-cycle statistics...")
    df_cycle_stats = create_candidate_cycle_stats(df_revisions)
    cycle_stats_csv = output_dir / 'candidate_cycle_stats.csv'
    df_cycle_stats.to_csv(cycle_stats_csv, index=False, encoding='utf-8')
    logger.info(f"       Created {len(df_cycle_stats):,} candidate-cycle records")
    logger.info(f"       Saved to {cycle_stats_csv.name}")

    # 4.5: Create editor profiles
    logger.info("\n  4.5: Creating editor profiles...")
    df_editors = create_editor_profiles(df_revisions)
    editors_csv = output_dir / 'editor_profiles.csv'
    df_editors.to_csv(editors_csv, index=False, encoding='utf-8')
    logger.info(f"       Created {len(df_editors):,} editor profiles")
    logger.info(f"       Saved to {editors_csv.name}")

    # Step 5: Generate summary statistics markdown
    logger.info("\n" + "="*80)
    logger.info("STEP 5: Generating summary statistics")
    logger.info("="*80)

    # Calculate all statistics
    office_counts = df_revisions.groupby('office')['revid'].count()
    cycle_counts = df_revisions.groupby('election_cycle')['revid'].count().sort_index()
    party_counts = df_revisions.groupby('party')['revid'].count().sort_values(ascending=False)
    state_counts = df_revisions.groupby('state')['revid'].count().sort_values(ascending=False)
    top_candidates = df_cycle_stats.groupby('page_title')['total_revisions'].sum().sort_values(ascending=False).head(10)

    # Candidate counts per cycle
    candidates_per_cycle = df_cycle_stats.groupby('election_cycle')['page_title'].nunique().sort_index()

    # Office and party breakdown per cycle
    cycle_office_counts = df_cycle_stats.groupby(['election_cycle', 'office'])['page_title'].nunique().unstack(fill_value=0)
    cycle_party_counts = df_cycle_stats.groupby(['election_cycle', 'party'])['page_title'].nunique().unstack(fill_value=0)

    # Save processing summary as markdown
    summary_path = output_dir / 'processing_summary.md'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Data Processing Summary\n\n")
        f.write(f"**Processing Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # Overview
        f.write("## Dataset Overview\n\n")
        f.write(f"- **Total Revisions**: {len(df_revisions):,}\n")
        f.write(f"- **Unique Candidates**: {df_revisions['page_title'].nunique():,}\n")
        f.write(f"- **Unique Editors**: {df_revisions['user'].nunique():,}\n")
        f.write(f"- **Date Range**: {df_revisions['timestamp'].min().strftime('%Y-%m-%d')} to {df_revisions['timestamp'].max().strftime('%Y-%m-%d')}\n")
        f.write(f"- **Election Cycles Covered**: {', '.join(map(str, sorted(cycle_counts.index)))}\n\n")

        # Filtering statistics
        f.write("---\n\n")
        f.write("## Data Filtering\n\n")
        f.write("The script applies strict filtering to only include revisions when candidates were on the ballot:\n\n")
        f.write("| Category | Count | Percentage |\n")
        f.write("|----------|------:|:----------:|\n")
        f.write(f"| Total revisions read | {stats['total_revisions_read']:,} | 100.0% |\n")
        f.write(f"| Excluded (before 2006-11-08) | {stats['revisions_before_2006']:,} | {100*stats['revisions_before_2006']/stats['total_revisions_read']:.1f}% |\n")
        f.write(f"| Excluded (not on ballot) | {stats['revisions_candidate_not_on_ballot']:,} | {100*stats['revisions_candidate_not_on_ballot']/stats['total_revisions_read']:.1f}% |\n")
        f.write(f"| **Kept for analysis** | **{stats['revisions_kept']:,}** | **{100*stats['revisions_kept']/stats['total_revisions_read']:.1f}%** |\n\n")

        # Edit types
        f.write("---\n\n")
        f.write("## Edit Types\n\n")
        f.write("| Type | Count | Percentage |\n")
        f.write("|------|------:|:----------:|\n")
        f.write(f"| Anonymous edits | {df_revisions['is_anonymous'].sum():,} | {df_revisions['is_anonymous'].mean()*100:.1f}% |\n")
        f.write(f"| Bot edits | {df_revisions['is_bot'].sum():,} | {df_revisions['is_bot'].mean()*100:.1f}% |\n")
        f.write(f"| Minor edits | {df_revisions['minor'].sum():,} | {df_revisions['minor'].mean()*100:.1f}% |\n\n")

        # By office
        f.write("---\n\n")
        f.write("## Distribution by Office\n\n")
        f.write("| Office | Revisions | Percentage | Candidates |\n")
        f.write("|--------|----------:|:----------:|:----------:|\n")
        for office in ['House', 'Senate']:
            if office in office_counts:
                count = office_counts[office]
                num_candidates = df_cycle_stats[df_cycle_stats['office'] == office]['page_title'].nunique()
                f.write(f"| {office} | {count:,} | {100*count/len(df_revisions):.1f}% | {num_candidates} |\n")
        f.write("\n")

        # By election cycle
        f.write("---\n\n")
        f.write("## Distribution by Election Cycle\n\n")
        f.write("| Cycle | Revisions | Percentage | Candidates | House | Senate |\n")
        f.write("|------:|----------:|:----------:|:----------:|------:|-------:|\n")
        for cycle, count in cycle_counts.items():
            num_candidates = candidates_per_cycle.get(cycle, 0)
            house = cycle_office_counts.loc[cycle, 'House'] if cycle in cycle_office_counts.index and 'House' in cycle_office_counts.columns else 0
            senate = cycle_office_counts.loc[cycle, 'Senate'] if cycle in cycle_office_counts.index and 'Senate' in cycle_office_counts.columns else 0
            f.write(f"| {cycle} | {count:,} | {100*count/len(df_revisions):.1f}% | {num_candidates} | {house} | {senate} |\n")
        f.write("\n")

        # By party
        f.write("---\n\n")
        f.write("## Distribution by Party\n\n")
        f.write("| Party | Revisions | Percentage | Candidates |\n")
        f.write("|-------|----------:|:----------:|:----------:|\n")
        for party, count in party_counts.head(5).items():
            num_candidates = df_cycle_stats[df_cycle_stats['party'] == party]['page_title'].nunique()
            f.write(f"| {party} | {count:,} | {100*count/len(df_revisions):.1f}% | {num_candidates} |\n")
        f.write("\n")

        # By state
        f.write("---\n\n")
        f.write("## Top 15 States by Revision Count\n\n")
        f.write("| Rank | State | Revisions | Percentage | Candidates |\n")
        f.write("|-----:|-------|----------:|:----------:|:----------:|\n")
        for i, (state, count) in enumerate(state_counts.head(15).items(), 1):
            num_candidates = df_cycle_stats[df_cycle_stats['state'] == state]['page_title'].nunique()
            f.write(f"| {i} | {state} | {count:,} | {100*count/len(df_revisions):.1f}% | {num_candidates} |\n")
        f.write("\n")

        # Top candidates
        f.write("---\n\n")
        f.write("## Top 15 Most Edited Candidates\n\n")
        f.write("| Rank | Candidate | Total Revisions | Unique Editors |\n")
        f.write("|-----:|-----------|----------------:|---------------:|\n")
        top_15_candidates = df_cycle_stats.groupby('page_title').agg({
            'total_revisions': 'sum',
            'unique_editors': 'sum'
        }).sort_values('total_revisions', ascending=False).head(15)

        for i, (candidate, row) in enumerate(top_15_candidates.iterrows(), 1):
            f.write(f"| {i} | {candidate.replace('_', ' ')} | {row['total_revisions']:,} | {row['unique_editors']:,} |\n")
        f.write("\n")

        # Files generated
        f.write("---\n\n")
        f.write("## Generated Files\n\n")
        f.write("| File | Rows | Description |\n")
        f.write("|------|-----:|:------------|\n")
        f.write(f"| `{revisions_csv.name}` | {len(df_revisions):,} | All revisions with metadata |\n")
        f.write(f"| `{daily_csv.name}` | {len(df_daily):,} | Daily aggregates |\n")
        f.write(f"| `{house_csv.name}` | {len(df_daily_house):,} | Daily aggregates (House only) |\n")
        f.write(f"| `{senate_csv.name}` | {len(df_daily_senate):,} | Daily aggregates (Senate only) |\n")
        f.write(f"| `{cycle_stats_csv.name}` | {len(df_cycle_stats):,} | Candidate-cycle statistics |\n")
        f.write(f"| `{editors_csv.name}` | {len(df_editors):,} | Editor profiles |\n")
        f.write("\n")

    logger.info(f"Processing summary saved to {summary_path.name}")

    logger.info("\n" + "="*80)
    logger.info("DATA PROCESSING COMPLETE")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"See detailed statistics in: {summary_path.name}")
    logger.info(f"Generated {len([revisions_csv, daily_csv, house_csv, senate_csv, cycle_stats_csv, editors_csv])} datasets.")


if __name__ == "__main__":
    main()
