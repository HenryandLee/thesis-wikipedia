"""
Script 2: Collect revision histories for selected Wikipedia pages

Usage:
    python 02_collect_revisions.py          # Default: Taiwan (Chinese Wikipedia)
    python 02_collect_revisions.py --config us    # US (English Wikipedia)
"""
import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from wiki_api import WikipediaAPI, save_json, load_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def collect_revisions_for_page(api: WikipediaAPI, page_title: str, 
                               start_date: str, end_date: str, 
                               output_dir: Path) -> dict:
    """
    Collect all revisions for a single page
    
    Args:
        api: WikipediaAPI instance
        page_title: Title of the page
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
        
        # Save revisions to file
        safe_filename = page_title.replace('/', '_').replace(':', '_')
        output_file = output_dir / f"{safe_filename}.json"
        
        output_data = {
            'page_title': page_title,
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
        logger.error(f"Error collecting revisions for '{page_title}': {e}")
        return {'total_revisions': 0, 'unique_editors': 0, 'error': str(e)}


def main():
    """Main execution function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Collect Wikipedia revision histories')
    parser.add_argument('--config', choices=['tw', 'us', 'candidates'], default='tw',
                       help='Configuration: tw (Taiwan/Chinese), us (US/English), or candidates (FEC roster)')
    args = parser.parse_args()
    
    # Determine paths based on config
    base_dir = Path(__file__).parent.parent
    
    if args.config == 'candidates':
        page_list_path = base_dir / 'data' / 'raw' / 'page_list_candidates.json'
        output_dir = base_dir / 'data' / 'raw' / 'revisions_candidates'
        summary_suffix = '_candidates'
        logger.info("Using Candidates configuration (FEC roster -> Wikipedia)")
    elif args.config == 'us':
        page_list_path = base_dir / 'data' / 'raw' / 'page_list_us.json'
        output_dir = base_dir / 'data' / 'raw' / 'revisions_us'
        summary_suffix = '_us'
        logger.info("Using US configuration (English Wikipedia)")
    else:
        page_list_path = base_dir / 'data' / 'raw' / 'page_list.json'
        output_dir = base_dir / 'data' / 'raw' / 'revisions'
        summary_suffix = ''
        logger.info("Using Taiwan configuration (Chinese Wikipedia)")
    
    if not page_list_path.exists():
        logger.error(f"Page list not found at {page_list_path}")
        if args.config == 'us':
            logger.error("Please run 'collect_comprehensive_us.py' first, then copy output to page_list_us.json")
        else:
            logger.error("Please run 'collect_comprehensive.py' first")
        return
    
    logger.info(f"Loading page list from {page_list_path}")
    page_data = load_json(page_list_path)
    
    pages = page_data['pages']
    metadata = page_data['metadata']
    
    # Get language from metadata (backwards compatible)
    language = metadata.get('language', 'zh' if args.config == 'tw' else 'en')
    
    # Get config (backwards compatible)
    if 'config' in metadata:
        config = metadata['config']
    else:
        config = metadata
    
    # Initialize API
    api = WikipediaAPI(language=language)

    # Setup output directory - clear old data first to avoid stale files
    if output_dir.exists():
        import shutil
        logger.info(f"Clearing old revision data from {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get date range from config (with fallback for different metadata structures)
    if 'date_range' in config:
        start_date = config['date_range']['start']
        end_date = config['date_range']['end']
    else:
        # Default date range covering all elections (including full 2016 cycle from Nov 2014)
        start_date = "2014-11-01T00:00:00Z"
        end_date = "2024-12-31T23:59:59Z"
        logger.info("No date_range in config, using default: 2014-11 to 2024-12")
    
    logger.info("=" * 60)
    logger.info("REVISION COLLECTION")
    logger.info("=" * 60)
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Total pages to process: {len(pages)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)
    
    # Collect revisions for each page
    collection_summary = []
    
    for i, page in enumerate(tqdm(pages, desc="Collecting revisions"), 1):
        page_title = page['title']
        logger.info(f"\n[{i}/{len(pages)}] Processing: {page_title}")
        
        stats = collect_revisions_for_page(
            api, page_title, start_date, end_date, output_dir
        )
        
        collection_summary.append({
            'title': page_title,
            'pageid': page['pageid'],
            'statistics': stats
        })
        
        logger.info(f"  Collected {stats['total_revisions']} revisions, "
                   f"{stats['unique_editors']} unique editors")
    
    # Save collection summary
    summary_path = Path(__file__).parent.parent / 'data' / 'raw' / f'collection_summary{summary_suffix}.json'
    summary_data = {
        'collection_date': datetime.now().isoformat(),
        'date_range': {
            'start': start_date,
            'end': end_date
        },
        'total_pages': len(pages),
        'total_revisions': sum(p['statistics']['total_revisions'] for p in collection_summary),
        'pages': collection_summary
    }
    
    save_json(summary_data, summary_path)
    
    # Print final summary
    logger.info("\n" + "=" * 60)
    logger.info("COLLECTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Pages processed: {len(collection_summary)}")
    logger.info(f"Total revisions collected: {summary_data['total_revisions']}")
    logger.info(f"Summary saved to: {summary_path}")
    
    # Show top pages by activity
    sorted_pages = sorted(collection_summary, 
                         key=lambda x: x['statistics']['total_revisions'], 
                         reverse=True)
    
    logger.info("\nTop 10 most edited pages:")
    for i, page in enumerate(sorted_pages[:10], 1):
        logger.info(f"  {i:2d}. {page['title']}: "
                   f"{page['statistics']['total_revisions']} revisions, "
                   f"{page['statistics']['unique_editors']} editors")
    
    logger.info("\n✓ Revision collection complete!")
    logger.info("Next step: Run '03_process_data.py' to create analysis-ready datasets")


if __name__ == "__main__":
    main()

