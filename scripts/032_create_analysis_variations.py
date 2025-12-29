"""
Script 032: Create Analysis Variations for Time Series

Creates 3 variations of House candidate datasets for comparative analysis:
1. Full dataset (all House candidates with Wikipedia pages)
2. Exclude special election cycles (remove idiosyncratic special election attention)
3. Also exclude presidential candidates (remove different editing patterns)

Time range: 2014-11-05 onwards (2016 cycle forward)
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
base_dir = Path(__file__).parent.parent
raw_dir = base_dir / 'data' / 'raw'
processed_dir = base_dir / 'data' / 'processed_html_parsed'
output_dir = processed_dir

# Date cutoff: 2014-11-05 (day after 2014 election, start of 2016 cycle)
CUTOFF_DATE = pd.Timestamp('2014-11-05', tz='UTC')

# =============================================================================
# STEP 1: Load Base Data
# =============================================================================

logger.info("Loading base datasets...")

# Load all revisions
df_revisions = pd.read_csv(processed_dir / 'all_revisions.csv', parse_dates=['timestamp'])
logger.info(f"  Loaded {len(df_revisions):,} revisions")

# Filter to House only and date range
df_house = df_revisions[
    (df_revisions['office'] == 'House') &
    (df_revisions['timestamp'] >= CUTOFF_DATE)
].copy()
logger.info(f"  Filtered to {len(df_house):,} House revisions from 2014-11-05 onwards")

# Load candidate metadata from CSVs
def load_candidate_metadata():
    """Load candidate metadata from raw CSV files"""
    metadata = []

    for year in [2016, 2018, 2020, 2022, 2024]:
        csv_path = raw_dir / 'html_parsed_candidates' / f'{year}_house_candidates.csv'
        if not csv_path.exists():
            logger.warning(f"  Missing CSV: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        df['election_cycle'] = year

        # Extract page title from wikipedia_url
        df['page_title'] = df['wikipedia_url'].apply(
            lambda url: url.split('/')[-1] if pd.notna(url) and url else None
        )

        metadata.append(df)

    df_meta = pd.concat(metadata, ignore_index=True)
    logger.info(f"  Loaded metadata for {len(df_meta):,} candidate-cycles")
    return df_meta

df_metadata = load_candidate_metadata()

# =============================================================================
# STEP 2: Identify Special Election Candidates
# =============================================================================

logger.info("\nIdentifying special election candidates...")

# Find all candidate-cycles that were special elections
special_elections = df_metadata[df_metadata['election_type'] == 'special'][
    ['page_title', 'election_cycle', 'candidate_name', 'state', 'district']
].drop_duplicates()

logger.info(f"  Found {len(special_elections)} special election candidate-cycles")
logger.info(f"  Unique candidates in special elections: {special_elections['page_title'].nunique()}")

# Show some examples
if len(special_elections) > 0:
    logger.info(f"\n  Examples of special election candidates:")
    for _, row in special_elections.head(10).iterrows():
        district = row['district'] if pd.notna(row['district']) else 'AL'
        logger.info(f"    - {row['candidate_name']:30s} | {row['state']}-{district} | Cycle {row['election_cycle']}")

# =============================================================================
# STEP 3: Identify Presidential Candidates
# =============================================================================

logger.info("\nIdentifying presidential candidates...")

# Load FEC roster
with open(raw_dir / 'fec_roster_raw.json') as f:
    fec_data = json.load(f)

# Extract all presidential candidates from FEC data
fec_presidential = [c for c in fec_data['roster'] if c['office'] == 'President']
logger.info(f"  Found {len(fec_presidential)} presidential candidates in FEC sample (5%)")

# Extract all name variants
fec_pres_names = set()
for candidate in fec_presidential:
    fec_pres_names.update(candidate['search_variants'])

# Manual supplement: Well-known House members who ran for president
# (These might not be in 5% FEC sample but we know they ran for president)
KNOWN_HOUSE_PRESIDENTIAL_CANDIDATES = {
    # Name as it appears in Wikipedia page titles
    'Ron_Paul',           # TX House → Presidential 2008, 2012
    'Dennis_Kucinich',    # OH House → Presidential 2004, 2008
    'Michele_Bachmann',   # MN House → Presidential 2012
    'Tulsi_Gabbard',      # HI House → Presidential 2020
    'Eric_Swalwell',      # CA House → Presidential 2020
    'Tim_Ryan_(Ohio_politician)',  # OH House → Presidential 2020
    'John_Delaney_(Maryland_politician)',  # MD House → Presidential 2020
    'Seth_Moulton',       # MA House → Presidential 2020
    'Joe_Sestak',         # PA House → Presidential 2020
    'Steve_Bullock_(American_politician)',  # Governor but included for completeness
}

logger.info(f"  Manual supplement: {len(KNOWN_HOUSE_PRESIDENTIAL_CANDIDATES)} known House→Presidential candidates")

# Create matching function
def normalize_name(name):
    """Normalize name for matching"""
    if pd.isna(name) or not name:
        return ""
    # Remove common suffixes and punctuation
    name = str(name).lower()
    name = name.replace(',', '').replace('.', '').replace(' jr', '').replace(' sr', '').replace(' iii', '').replace(' ii', '')
    name = name.strip()
    return name

# Match House candidates to presidential candidates
presidential_page_titles = set()

# Method 1: Direct match on known list
for page_title in KNOWN_HOUSE_PRESIDENTIAL_CANDIDATES:
    if page_title in df_house['page_title'].values:
        presidential_page_titles.add(page_title)
        logger.info(f"    ✓ Found: {page_title.replace('_', ' ')}")

# Method 2: Fuzzy match using FEC name variants
# Create lookup of House candidate page_title -> candidate_name
house_names = df_metadata[['page_title', 'candidate_name']].drop_duplicates()
house_names = house_names[house_names['page_title'].notna()]

for _, row in house_names.iterrows():
    page_title = row['page_title']
    candidate_name = row['candidate_name']

    # Normalize the House candidate name
    normalized_house = normalize_name(candidate_name)

    # Check against all FEC presidential name variants
    for fec_name in fec_pres_names:
        normalized_fec = normalize_name(fec_name)

        # Simple exact match on normalized names
        if normalized_house and normalized_fec and normalized_house == normalized_fec:
            presidential_page_titles.add(page_title)
            logger.info(f"    ✓ Matched: {candidate_name} → {fec_name}")
            break

logger.info(f"\n  Total presidential candidates identified in House data: {len(presidential_page_titles)}")

# =============================================================================
# STEP 4: Create Variation 1 - Full Dataset
# =============================================================================

logger.info("\n" + "="*80)
logger.info("VARIATION 1: Full House Dataset (2014-11-05 onwards)")
logger.info("="*80)

df_v1 = df_house.copy()

# Create daily aggregates
def create_daily_aggregates(df, name):
    """Create daily aggregates grouped by (date, page_title), then average across pages"""

    # First group by date and page_title
    daily_by_page = df.groupby(['date', 'page_title']).agg({
        'revid': 'count',          # num_edits per page
        'user': 'nunique',         # unique_editors per page
    }).reset_index()
    daily_by_page.columns = ['date', 'page_title', 'num_edits', 'unique_editors']

    # Then average across pages for each date
    daily_agg = daily_by_page.groupby('date').agg({
        'num_edits': 'mean',
        'unique_editors': 'mean',
        'page_title': 'nunique'
    }).reset_index()
    daily_agg.columns = ['date', 'mean_edits_per_page', 'mean_editors_per_page', 'num_pages']

    logger.info(f"  {name}: {len(daily_agg):,} days of data")
    logger.info(f"  {name}: {daily_by_page['page_title'].nunique():,} unique pages")
    logger.info(f"  {name}: {df['revid'].count():,} total revisions")

    return daily_agg

# Need to aggregate by date first (extract date from timestamp)
df_v1['date'] = df_v1['timestamp'].dt.date
df_v1['date'] = pd.to_datetime(df_v1['date'])

daily_v1 = create_daily_aggregates(df_v1, "Variation 1")

# Save
output_path = output_dir / 'daily_edits_house_v1_full.csv'
daily_v1.to_csv(output_path, index=False)
logger.info(f"  ✓ Saved: {output_path}")

# =============================================================================
# STEP 5: Create Variation 2 - Exclude Special Elections
# =============================================================================

logger.info("\n" + "="*80)
logger.info("VARIATION 2: Exclude Special Election Cycles")
logger.info("="*80)

# Create exclusion mask
excluded_v2 = 0
df_v2 = df_v1.copy()

for _, special in special_elections.iterrows():
    page_title = special['page_title']
    cycle = special['election_cycle']

    # Exclude all revisions for this candidate in this cycle
    mask = (df_v2['page_title'] == page_title) & (df_v2['election_cycle'] == cycle)
    excluded_v2 += mask.sum()
    df_v2 = df_v2[~mask]

logger.info(f"  Excluded {excluded_v2:,} revisions from {len(special_elections)} special election candidate-cycles")
logger.info(f"  Remaining: {len(df_v2):,} revisions ({100*len(df_v2)/len(df_v1):.1f}% of Variation 1)")

daily_v2 = create_daily_aggregates(df_v2, "Variation 2")

# Save
output_path = output_dir / 'daily_edits_house_v2_no_special.csv'
daily_v2.to_csv(output_path, index=False)
logger.info(f"  ✓ Saved: {output_path}")

# =============================================================================
# STEP 6: Create Variation 3 - Also Exclude Presidential Candidates
# =============================================================================

logger.info("\n" + "="*80)
logger.info("VARIATION 3: Exclude Special Elections + Presidential Candidates")
logger.info("="*80)

# Exclude presidential candidates entirely (all cycles)
excluded_v3 = 0
df_v3 = df_v2.copy()

for page_title in presidential_page_titles:
    mask = df_v3['page_title'] == page_title
    excluded_v3 += mask.sum()
    df_v3 = df_v3[~mask]

logger.info(f"  Excluded {excluded_v3:,} revisions from {len(presidential_page_titles)} presidential candidates")
logger.info(f"  Remaining: {len(df_v3):,} revisions ({100*len(df_v3)/len(df_v2):.1f}% of Variation 2)")

daily_v3 = create_daily_aggregates(df_v3, "Variation 3")

# Save
output_path = output_dir / 'daily_edits_house_v3_no_special_no_pres.csv'
daily_v3.to_csv(output_path, index=False)
logger.info(f"  ✓ Saved: {output_path}")

# =============================================================================
# STEP 7: Summary Statistics
# =============================================================================

logger.info("\n" + "="*80)
logger.info("SUMMARY COMPARISON")
logger.info("="*80)

summary_data = {
    'Variation': ['V1: Full', 'V2: No Special', 'V3: No Special/Pres'],
    'Total Revisions': [len(df_v1), len(df_v2), len(df_v3)],
    'Unique Pages': [
        df_v1['page_title'].nunique(),
        df_v2['page_title'].nunique(),
        df_v3['page_title'].nunique()
    ],
    'Days of Data': [len(daily_v1), len(daily_v2), len(daily_v3)],
    'Avg Edits/Page/Day': [
        daily_v1['mean_edits_per_page'].mean(),
        daily_v2['mean_edits_per_page'].mean(),
        daily_v3['mean_edits_per_page'].mean()
    ]
}

df_summary = pd.DataFrame(summary_data)
print("\n" + df_summary.to_string(index=False))

# Save summary
summary_path = output_dir / 'variation_summary.csv'
df_summary.to_csv(summary_path, index=False)
logger.info(f"\n✓ Summary saved: {summary_path}")

# Save exclusion lists
exclusions = {
    'special_elections': special_elections.to_dict('records'),
    'presidential_candidates': list(presidential_page_titles),
    'statistics': {
        'special_election_candidate_cycles': len(special_elections),
        'presidential_candidates': len(presidential_page_titles),
        'revisions_excluded_v2': int(excluded_v2),
        'revisions_excluded_v3': int(excluded_v3)
    }
}

with open(output_dir / 'exclusion_lists.json', 'w') as f:
    json.dump(exclusions, f, indent=2)

logger.info(f"✓ Exclusion lists saved: {output_dir / 'exclusion_lists.json'}")
logger.info("\n" + "="*80)
logger.info("DONE! Three variation datasets created successfully.")
logger.info("="*80)
