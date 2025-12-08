"""
Script 3: Process raw revision data into analysis-ready datasets

Usage:
    python 03_process_data.py          # Default: Taiwan (Chinese Wikipedia)
    python 03_process_data.py --config us    # US (English Wikipedia)
"""
import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import logging
import pandas as pd
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from wiki_api import load_json, WikipediaAPI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def identify_bots_from_api(usernames: list, language: str = 'en') -> set:
    """
    Query Wikipedia API to definitively identify bot accounts.
    
    Args:
        usernames: List of unique usernames from the dataset
        language: Wikipedia language code ('en' for English, 'zh' for Chinese)
        
    Returns:
        Set of usernames that are confirmed bots
    """
    logger.info(f"Querying Wikipedia API to identify bots among {len(usernames)} unique users...")
    
    api = WikipediaAPI(language=language)
    bots = api.identify_bots(usernames)
    
    logger.info(f"Identified {len(bots)} bot accounts")
    if bots:
        logger.info(f"Bot accounts: {sorted(bots)[:10]}{'...' if len(bots) > 10 else ''}")
    
    return bots


def process_revisions_to_timeseries(revisions_dir: Path) -> pd.DataFrame:
    """
    Process all revision files into a time series dataset
    
    Args:
        revisions_dir: Directory containing revision JSON files
        
    Returns:
        DataFrame with time series of editing activity
    """
    all_revisions = []
    
    for revision_file in revisions_dir.glob('*.json'):
        try:
            data = load_json(revision_file)
            page_title = data['page_title']
            
            for rev in data['revisions']:
                all_revisions.append({
                    'page_title': page_title,
                    'revid': rev['revid'],
                    'parentid': rev['parentid'],
                    'timestamp': rev['timestamp'],
                    'user': rev['user'],
                    'userid': rev['userid'],
                    'size': rev['size'],
                    'comment': rev['comment'],
                    'minor': rev['minor'],
                    'tags': ','.join(rev['tags']) if rev['tags'] else ''
                })
        except Exception as e:
            logger.error(f"Error processing {revision_file}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(all_revisions)
    
    if len(df) > 0:
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        df['weekday'] = df['timestamp'].dt.dayofweek
        
        # Calculate size changes
        df = df.sort_values(['page_title', 'timestamp'])
        df['size_change'] = df.groupby('page_title')['size'].diff()
        
        # Identify anonymous users (bot detection done separately via API)
        df['is_anonymous'] = df['userid'] == 0
        df['is_bot'] = False  # Will be set by API query in main()
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df


def create_daily_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create daily aggregated statistics
    
    Args:
        df: Processed revisions DataFrame
        
    Returns:
        DataFrame with daily aggregates
    """
    daily_stats = df.groupby(['page_title', 'date']).agg({
        'revid': 'count',  # Number of edits
        'user': 'nunique',  # Unique editors
        'size_change': 'sum',  # Net size change
        'minor': 'sum',  # Number of minor edits
        'is_anonymous': 'sum',  # Anonymous edits
        'is_bot': 'sum'  # Bot edits
    }).reset_index()
    
    daily_stats.columns = [
        'page_title', 'date', 'num_edits', 'unique_editors', 
        'net_size_change', 'minor_edits', 'anonymous_edits', 'bot_edits'
    ]
    
    return daily_stats


def create_editor_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create editor-level aggregated data
    
    Args:
        df: Processed revisions DataFrame
        
    Returns:
        DataFrame with editor profiles
    """
    editor_stats = df.groupby(['page_title', 'user']).agg({
        'revid': 'count',  # Number of edits
        'size_change': 'sum',  # Total size contributed
        'minor': 'sum',  # Minor edits
        'timestamp': ['min', 'max']  # First and last edit
    }).reset_index()
    
    editor_stats.columns = [
        'page_title', 'user', 'num_edits', 'total_size_change', 
        'minor_edits', 'first_edit', 'last_edit'
    ]
    
    # Calculate editing span in days
    editor_stats['editing_span_days'] = (
        pd.to_datetime(editor_stats['last_edit']) - 
        pd.to_datetime(editor_stats['first_edit'])
    ).dt.days
    
    # Calculate average edit size
    editor_stats['avg_size_change'] = (
        editor_stats['total_size_change'] / editor_stats['num_edits']
    )
    
    return editor_stats


def create_page_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create page-level statistics
    
    Args:
        df: Processed revisions DataFrame
        
    Returns:
        DataFrame with page statistics
    """
    page_stats = df.groupby('page_title').agg({
        'revid': 'count',  # Total edits
        'user': 'nunique',  # Unique editors
        'size': 'last',  # Final size
        'size_change': 'sum',  # Net size change
        'timestamp': ['min', 'max'],  # Date range
        'minor': 'sum',  # Minor edits
        'is_anonymous': 'sum',  # Anonymous edits
        'is_bot': 'sum'  # Bot edits
    }).reset_index()
    
    page_stats.columns = [
        'page_title', 'total_edits', 'unique_editors', 'final_size',
        'net_size_change', 'first_edit', 'last_edit', 'minor_edits',
        'anonymous_edits', 'bot_edits'
    ]
    
    # Calculate editing intensity
    page_stats['editing_span_days'] = (
        pd.to_datetime(page_stats['last_edit']) - 
        pd.to_datetime(page_stats['first_edit'])
    ).dt.days + 1
    
    page_stats['edits_per_day'] = (
        page_stats['total_edits'] / page_stats['editing_span_days']
    )
    
    return page_stats


def load_page_metadata(page_list_path: Path) -> dict:
    """Load page metadata including on_ballot info from page list."""
    if not page_list_path.exists():
        return {}
    
    data = load_json(page_list_path)
    pages = data.get('pages', [])
    
    metadata = {}
    for page in pages:
        title = page.get('title')
        if title:
            metadata[title] = {
                'on_ballot': page.get('on_ballot', {}),
                'primary_office': page.get('primary_office'),
                'primary_party': page.get('primary_party'),
                'primary_state': page.get('primary_state')
            }
    
    return metadata


def main():
    """Main execution function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Process Wikipedia revision data')
    parser.add_argument('--config', choices=['tw', 'us', 'candidates'], default='tw',
                       help='Configuration: tw (Taiwan/Chinese), us (US/English), or candidates (FEC roster)')
    args = parser.parse_args()
    
    # Setup paths based on config
    base_dir = Path(__file__).parent.parent
    
    page_list_path = None
    if args.config == 'candidates':
        revisions_dir = base_dir / 'data' / 'raw' / 'revisions_candidates'
        output_dir = base_dir / 'data' / 'processed_candidates'
        page_list_path = base_dir / 'data' / 'raw' / 'page_list_candidates.json'
        logger.info("Using Candidates configuration (FEC roster)")
    elif args.config == 'us':
        revisions_dir = base_dir / 'data' / 'raw' / 'revisions_us'
        output_dir = base_dir / 'data' / 'processed_us'
        page_list_path = base_dir / 'data' / 'raw' / 'page_list_us.json'
        logger.info("Using US configuration (English Wikipedia)")
    else:
        revisions_dir = base_dir / 'data' / 'raw' / 'revisions'
        output_dir = base_dir / 'data' / 'processed'
        logger.info("Using Taiwan configuration (Chinese Wikipedia)")
    
    # Clear old processed data to ensure fresh analysis
    if output_dir.exists():
        import shutil
        logger.info(f"Clearing old processed data from {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DATA PROCESSING")
    logger.info("=" * 60)
    logger.info(f"Input directory: {revisions_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Check if revision files exist
    revision_files = list(revisions_dir.glob('*.json'))
    if not revision_files:
        logger.error(f"No revision files found in {revisions_dir}")
        logger.error("Please run '02_collect_revisions.py' first")
        return
    
    logger.info(f"Found {len(revision_files)} revision files")
    
    # Load page metadata if available
    page_metadata = {}
    if page_list_path:
        page_metadata = load_page_metadata(page_list_path)
        logger.info(f"Loaded metadata for {len(page_metadata)} pages")
    
    # Determine Wikipedia language for API queries
    if args.config == 'tw':
        wiki_language = 'zh'
    else:
        wiki_language = 'en'
    
    # Process revisions
    logger.info("\nStep 1: Loading and processing revisions...")
    df_revisions = process_revisions_to_timeseries(revisions_dir)
    logger.info(f"  Loaded {len(df_revisions)} total revisions")
    
    if len(df_revisions) == 0:
        logger.error("No revisions found. Exiting.")
        return
    
    # Step 1b: Identify bots via Wikipedia API (definitive method)
    logger.info("\nStep 1b: Identifying bot accounts via Wikipedia API...")
    unique_users = df_revisions['user'].unique().tolist()
    bot_usernames = identify_bots_from_api(unique_users, language=wiki_language)
    
    # Update is_bot column with definitive API results
    df_revisions['is_bot'] = df_revisions['user'].isin(bot_usernames)
    logger.info(f"  Marked {df_revisions['is_bot'].sum():,} edits as bot edits "
                f"({df_revisions['is_bot'].mean()*100:.1f}% of total)")
    
    # Save full revision dataset
    revisions_csv = output_dir / 'all_revisions.csv'
    df_revisions.to_csv(revisions_csv, index=False, encoding='utf-8')
    logger.info(f"  Saved to: {revisions_csv}")
    
    # Create daily aggregates
    logger.info("\nStep 2: Creating daily aggregates...")
    df_daily = create_daily_aggregates(df_revisions)
    daily_csv = output_dir / 'daily_edits.csv'
    df_daily.to_csv(daily_csv, index=False, encoding='utf-8')
    logger.info(f"  Created {len(df_daily)} daily records")
    logger.info(f"  Saved to: {daily_csv}")
    
    # Create editor profiles
    logger.info("\nStep 3: Creating editor profiles...")
    df_editors = create_editor_profiles(df_revisions)
    editors_csv = output_dir / 'editor_profiles.csv'
    df_editors.to_csv(editors_csv, index=False, encoding='utf-8')
    logger.info(f"  Created {len(df_editors)} editor profiles")
    logger.info(f"  Saved to: {editors_csv}")
    
    # Create page statistics
    logger.info("\nStep 4: Creating page statistics...")
    df_pages = create_page_statistics(df_revisions)
    
    # Add page metadata if available (on_ballot, office, party, state)
    if page_metadata:
        df_pages['primary_office'] = df_pages['page_title'].map(
            lambda x: page_metadata.get(x, {}).get('primary_office')
        )
        df_pages['primary_party'] = df_pages['page_title'].map(
            lambda x: page_metadata.get(x, {}).get('primary_party')
        )
        df_pages['primary_state'] = df_pages['page_title'].map(
            lambda x: page_metadata.get(x, {}).get('primary_state')
        )
        
        # Add on_ballot columns for each cycle
        cycles = [2016, 2018, 2020, 2022, 2024]
        for cycle in cycles:
            df_pages[f'on_ballot_{cycle}'] = df_pages['page_title'].map(
                lambda x, c=cycle: page_metadata.get(x, {}).get('on_ballot', {}).get(str(c), False)
            )
        
        logger.info(f"  Added on_ballot indicators for {len(cycles)} cycles")
    
    pages_csv = output_dir / 'page_statistics.csv'
    df_pages.to_csv(pages_csv, index=False, encoding='utf-8')
    logger.info(f"  Created statistics for {len(df_pages)} pages")
    logger.info(f"  Saved to: {pages_csv}")
    
    # Print summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total revisions: {len(df_revisions):,}")
    logger.info(f"Unique pages: {df_revisions['page_title'].nunique()}")
    logger.info(f"Unique editors: {df_revisions['user'].nunique()}")
    logger.info(f"Date range: {df_revisions['timestamp'].min()} to {df_revisions['timestamp'].max()}")
    logger.info(f"Anonymous edits: {df_revisions['is_anonymous'].sum():,} ({df_revisions['is_anonymous'].mean()*100:.1f}%)")
    logger.info(f"Bot edits: {df_revisions['is_bot'].sum():,} ({df_revisions['is_bot'].mean()*100:.1f}%)")
    logger.info(f"Minor edits: {df_revisions['minor'].sum():,} ({df_revisions['minor'].mean()*100:.1f}%)")
    
    logger.info("\nTop 5 most active editors:")
    top_editors = df_editors.groupby('user')['num_edits'].sum().sort_values(ascending=False).head()
    for i, (editor, edits) in enumerate(top_editors.items(), 1):
        logger.info(f"  {i}. {editor}: {edits} edits")
    
    logger.info("\nTop 5 most edited pages:")
    top_pages = df_pages.sort_values('total_edits', ascending=False).head()
    for i, (_, row) in enumerate(top_pages.iterrows(), 1):
        logger.info(f"  {i}. {row['page_title']}: {row['total_edits']} edits")
    
    logger.info("\n✓ Data processing complete!")
    logger.info("\nGenerated datasets:")
    logger.info(f"  - {revisions_csv}")
    logger.info(f"  - {daily_csv}")
    logger.info(f"  - {editors_csv}")
    logger.info(f"  - {pages_csv}")
    logger.info("\nYou can now begin your analysis!")


if __name__ == "__main__":
    main()

