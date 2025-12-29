"""
Script 021: Collect revision histories for candidates from HTML parsed roster

This script reads candidate data from CSV files in data/raw/html_parsed_candidates/
and collects Wikipedia revision histories for candidates with available Wikipedia URLs.

Usage:
    python scripts/021_collect_revisions.py
"""
import json
import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm
from urllib.parse import urlparse, unquote

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from wiki_api import WikipediaAPI, save_json, load_json
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_to_native_types(obj):
    """Convert pandas/numpy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_to_native_types(item) for item in obj.tolist()]
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif pd.isna(obj):
        return None
    else:
        return obj


def extract_page_title_from_url(url: str) -> str:
    """
    Extract Wikipedia page title from URL

    Args:
        url: Wikipedia URL (e.g., 'https://en.wikipedia.org/wiki/Adam_Schiff')

    Returns:
        Page title (e.g., 'Adam_Schiff')
    """
    if pd.isna(url) or not url:
        return None

    # Parse URL and extract the path
    parsed = urlparse(url)
    path_parts = parsed.path.split('/')

    # Wikipedia URLs are in format /wiki/Page_Title
    if len(path_parts) >= 3 and path_parts[1] == 'wiki':
        # Get the title part and decode URL encoding
        title = unquote(path_parts[2])
        return title

    return None


def load_candidates_from_csvs(csv_dir: Path) -> pd.DataFrame:
    """
    Load all candidate data from CSV files in the directory

    Args:
        csv_dir: Directory containing CSV files

    Returns:
        DataFrame with all candidates
    """
    all_candidates = []

    csv_files = sorted(csv_dir.glob('*.csv'))
    logger.info(f"Found {len(csv_files)} CSV files")

    for csv_file in csv_files:
        logger.info(f"Loading {csv_file.name}")
        df = pd.read_csv(csv_file)

        # Add source file information
        df['source_file'] = csv_file.name

        # Extract year from filename (e.g., '2024_senate_candidates.csv' -> 2024)
        year = csv_file.stem.split('_')[0]
        df['election_year'] = int(year)

        all_candidates.append(df)

    combined = pd.concat(all_candidates, ignore_index=True)
    logger.info(f"Loaded {len(combined)} total candidates")

    return combined


def collect_revisions_for_candidate(api: WikipediaAPI, page_title: str,
                                    candidate_name: str,
                                    start_date: str, end_date: str,
                                    output_dir: Path) -> dict:
    """
    Collect all revisions for a single candidate's Wikipedia page

    Args:
        api: WikipediaAPI instance
        page_title: Wikipedia page title (extracted from URL)
        candidate_name: Candidate name from CSV
        start_date: Start date for revisions
        end_date: End date for revisions
        output_dir: Directory to save revisions

    Returns:
        Dictionary with collection statistics
    """
    revisions = []

    try:
        for revision in api.get_revisions(page_title, start_date, end_date):
            revisions.append(revision)

        # Calculate statistics
        stats = {
            'total_revisions': len(revisions),
            'unique_editors': len(set(rev['user'] for rev in revisions)),
            'date_range': {
                'start': revisions[0]['timestamp'] if revisions else None,
                'end': revisions[-1]['timestamp'] if revisions else None
            }
        }

        # Save revisions to file (use page_title as filename)
        safe_filename = page_title.replace('/', '_').replace(':', '_')
        output_file = output_dir / f"{safe_filename}.json"

        # Reconstruct Wikipedia URL from page_title
        wikipedia_url = f"https://en.wikipedia.org/wiki/{page_title}"

        output_data = {
            'page_title': page_title,
            'candidate_name': candidate_name,
            'wikipedia_url': wikipedia_url,
            'collection_date': datetime.now().isoformat(),
            'date_filter': {
                'start': start_date,
                'end': end_date
            },
            'statistics': stats,
            'revisions': revisions
        }

        save_json(output_data, output_file)

        return stats

    except Exception as e:
        logger.error(f"Error collecting revisions for '{candidate_name}' ({page_title}): {e}")
        return {'total_revisions': 0, 'unique_editors': 0, 'error': str(e)}


def main():
    """Main execution function"""

    # Setup paths
    base_dir = Path(__file__).parent.parent
    csv_dir = base_dir / 'data' / 'raw' / 'html_parsed_candidates'
    output_dir = base_dir / 'data' / 'raw' / 'revisions_html_parsed'

    if not csv_dir.exists():
        logger.error(f"CSV directory not found at {csv_dir}")
        return

    # Load all candidates from CSVs
    logger.info("=" * 60)
    logger.info("LOADING CANDIDATE DATA")
    logger.info("=" * 60)

    candidates = load_candidates_from_csvs(csv_dir)

    # Filter to candidates with Wikipedia URLs
    candidates_with_wiki = candidates[candidates['wikipedia_url'].notna()].copy()
    logger.info(f"Candidates with Wikipedia URLs: {len(candidates_with_wiki)}")
    logger.info(f"Candidates without Wikipedia URLs: {len(candidates) - len(candidates_with_wiki)}")

    # Extract page titles from URLs
    candidates_with_wiki['page_title'] = candidates_with_wiki['wikipedia_url'].apply(
        extract_page_title_from_url
    )

    # Remove any that failed to parse
    valid_candidates = candidates_with_wiki[candidates_with_wiki['page_title'].notna()].copy()
    failed_parse = len(candidates_with_wiki) - len(valid_candidates)
    if failed_parse > 0:
        logger.warning(f"Failed to parse {failed_parse} Wikipedia URLs")

    logger.info(f"Valid candidates to process: {len(valid_candidates)}")

    # Remove duplicates (same page_title may appear multiple times)
    # Keep all unique candidate name variations and track metadata
    unique_pages = valid_candidates.groupby('page_title').agg({
        'candidate_name': lambda x: list(x.unique()),  # Keep all unique name variations
        'wikipedia_url': 'first',
        'election_year': lambda x: sorted(x.unique()),  # Sorted list of unique years
        'source_file': lambda x: list(x.unique()),  # Track which CSVs
        'party': lambda x: list(x.unique()) if 'party' in valid_candidates.columns else None,
        'state': lambda x: list(x.unique()) if 'state' in valid_candidates.columns else None
    }).reset_index()

    # Use the first name variation as primary, keep others as aliases
    unique_pages['primary_name'] = unique_pages['candidate_name'].apply(lambda x: x[0])
    unique_pages['name_variations'] = unique_pages['candidate_name'].apply(
        lambda x: x if len(x) > 1 else None
    )

    logger.info(f"Unique Wikipedia pages to process: {len(unique_pages)}")

    # Show statistics about duplicates
    total_appearances = len(valid_candidates)
    duplicate_appearances = total_appearances - len(unique_pages)
    logger.info(f"Total candidate appearances: {total_appearances}")
    logger.info(f"Duplicate appearances (same page): {duplicate_appearances}")

    # Show candidates with name variations
    name_variations = unique_pages[unique_pages['name_variations'].notna()]
    if len(name_variations) > 0:
        logger.info(f"\nFound {len(name_variations)} candidates with name variations:")
        for _, row in name_variations.head(10).iterrows():
            logger.info(f"  {row['page_title']}: {row['name_variations']}")

    # Initialize API (English Wikipedia)
    api = WikipediaAPI(language='en')

    # Setup output directory - clear old data first to avoid stale files
    if output_dir.exists():
        import shutil
        logger.info(f"Clearing old revision data from {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Date range: 2006-11-01 to 2024-12-31
    start_date = "2006-11-01T00:00:00Z"
    end_date = "2024-12-31T23:59:59Z"

    logger.info("=" * 60)
    logger.info("REVISION COLLECTION")
    logger.info("=" * 60)
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Total unique pages to process: {len(unique_pages)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)

    # Collect revisions for each unique page
    collection_summary = []

    for i, row in enumerate(tqdm(unique_pages.itertuples(),
                                  total=len(unique_pages),
                                  desc="Collecting revisions"), 1):
        page_title = row.page_title
        primary_name = row.primary_name

        logger.info(f"\n[{i}/{len(unique_pages)}] Processing: {primary_name} ({page_title})")

        stats = collect_revisions_for_candidate(
            api, page_title, primary_name,
            start_date, end_date, output_dir
        )

        # Convert all fields to native Python types for JSON serialization
        entry = {
            'page_title': page_title,
            'candidate_name': primary_name,
            'name_variations': convert_to_native_types(row.name_variations) if row.name_variations else None,
            'wikipedia_url': row.wikipedia_url,
            'election_years': convert_to_native_types(row.election_year if isinstance(row.election_year, list) else [row.election_year]),
            'source_files': convert_to_native_types(row.source_file if isinstance(row.source_file, list) else [row.source_file]),
            'parties': convert_to_native_types(row.party) if hasattr(row, 'party') and row.party else None,
            'states': convert_to_native_types(row.state) if hasattr(row, 'state') and row.state else None,
            'statistics': stats
        }
        collection_summary.append(entry)

        logger.info(f"  Collected {stats['total_revisions']} revisions, "
                   f"{stats['unique_editors']} unique editors")

    # Save collection summary
    summary_path = base_dir / 'data' / 'raw' / 'collection_summary_html_parsed.json'
    summary_data = {
        'collection_date': datetime.now().isoformat(),
        'date_range': {
            'start': start_date,
            'end': end_date
        },
        'total_unique_pages': len(unique_pages),
        'total_candidates': len(candidates),
        'candidates_with_wiki': len(candidates_with_wiki),
        'total_revisions': sum(p['statistics']['total_revisions'] for p in collection_summary),
        'pages': collection_summary
    }

    save_json(summary_data, summary_path)

    # Print final summary
    logger.info("\n" + "=" * 60)
    logger.info("COLLECTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Unique pages processed: {len(collection_summary)}")
    logger.info(f"Total candidates in roster: {len(candidates)}")
    logger.info(f"Candidates with Wikipedia: {len(candidates_with_wiki)}")
    logger.info(f"Total revisions collected: {summary_data['total_revisions']}")
    logger.info(f"Summary saved to: {summary_path}")

    # Show top pages by activity
    sorted_pages = sorted(collection_summary,
                         key=lambda x: x['statistics']['total_revisions'],
                         reverse=True)

    logger.info("\nTop 10 most edited pages:")
    for i, page in enumerate(sorted_pages[:10], 1):
        logger.info(f"  {i:2d}. {page['candidate_name']} ({page['page_title']}): "
                   f"{page['statistics']['total_revisions']} revisions, "
                   f"{page['statistics']['unique_editors']} editors")

    logger.info("\nCollection complete!")
    logger.info("Next step: Process revision data for analysis")


if __name__ == "__main__":
    main()
