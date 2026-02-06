"""
Script 032: Create Analysis Variations for Time Series

Creates 3 variations of House candidate datasets for comparative analysis:
1. Full dataset (all House candidates with Wikipedia pages)
2. Exclude special election cycles (remove idiosyncratic special election attention)
3. Also exclude presidential candidates (remove different editing patterns)

Presidential candidate identification uses FEC Candidate Master bulk files
(downloaded directly for presidential election years only). Identification is
based on normalized name matching between FEC presidential filers and House
candidate metadata.

Exclusion is cycle-specific: a House candidate who ran for president in 2020
is only excluded from the 2020 cycle, not from other cycles.

Time range: 2006-11-08 onwards (2008 cycle forward)

See docs/032_create_analysis_variations.md for detailed documentation.
"""
import json
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO
from collections import defaultdict
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
base_dir = Path(__file__).parent.parent
raw_dir = base_dir / 'data' / 'raw'
processed_dir = base_dir / 'data' / 'processed_html_parsed'
cache_dir = base_dir / 'data' / 'cache' / 'fec'
output_dir = processed_dir

# Election cycles to cover
ALL_CYCLES = list(range(2008, 2026, 2))  # 2008, 2010, ..., 2024
PRESIDENTIAL_CYCLES = [2008, 2012, 2016, 2020, 2024]

# Date cutoff: day after 2006 election (start of 2008 cycle)
CUTOFF_DATE = pd.Timestamp('2006-11-08', tz='UTC')

# FEC Candidate Master File columns (pipe-delimited, no header)
FEC_COLUMNS = [
    'candidate_id', 'candidate_name', 'party_affiliation', 'election_year',
    'office_state', 'office', 'office_district', 'incumbent_challenger_status',
    'candidate_status', 'principal_committee_id', 'street_1', 'street_2',
    'city', 'state', 'zip',
]


# =============================================================================
# HELPERS
# =============================================================================

def normalize_name(name):
    """Normalize a name string for comparison.

    Handles FEC format 'LAST, FIRST MIDDLE' and plain 'First Last'.
    Returns lowercase 'first last' with suffixes stripped.
    """
    if not name or (isinstance(name, float) and np.isnan(name)):
        return ""
    name = str(name).strip().lower()
    # FEC names are "LAST, FIRST MIDDLE SUFFIX"
    if ',' in name:
        parts = name.split(',', 1)
        last = parts[0].strip()
        first_middle = parts[1].strip() if len(parts) > 1 else ''
        name = f"{first_middle} {last}"
    # Strip suffixes and punctuation
    name = re.sub(r'\b(jr|sr|iii|ii|iv)\b', '', name)
    name = re.sub(r'[.,"\']', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def download_fec_presidential_names(cycle):
    """Download FEC Candidate Master file for a presidential cycle and return
    a set of normalized names of presidential candidates."""
    yy = str(cycle)[-2:]
    cache_file = cache_dir / f'cn{yy}.txt'

    # Use cache if available
    if cache_file.exists():
        logger.info(f"  Using cached FEC file: cn{yy}.txt")
        text = cache_file.read_text(encoding='latin-1')
    else:
        url = f"https://www.fec.gov/files/bulk-downloads/{cycle}/cn{yy}.zip"
        logger.info(f"  Downloading FEC file: {url}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        cache_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            txt_name = f'cn{yy}.txt'
            # Some zips use cn.txt instead of cn{yy}.txt
            names = zf.namelist()
            txt_name = next((n for n in names if n.endswith('.txt')), names[0])
            text = zf.read(txt_name).decode('latin-1')
            cache_file.write_text(text, encoding='latin-1')

    # Parse pipe-delimited file
    lines = text.strip().split('\n')
    pres_names = set()
    for line in lines:
        fields = line.split('|')
        if len(fields) < 6:
            continue
        office = fields[5].strip()
        if office == 'P':
            raw_name = fields[1].strip()
            pres_names.add(normalize_name(raw_name))
    logger.info(f"  {cycle}: found {len(pres_names)} presidential filers in FEC data")
    return pres_names


# =============================================================================
# STEP 1: Load Base Data
# =============================================================================

logger.info("Loading base datasets...")

df_revisions = pd.read_csv(processed_dir / 'all_revisions.csv', parse_dates=['timestamp'])
logger.info(f"  Loaded {len(df_revisions):,} revisions")

# Filter to House only and date range
df_house = df_revisions[
    (df_revisions['office'] == 'House') &
    (df_revisions['timestamp'] >= CUTOFF_DATE)
].copy()
logger.info(f"  Filtered to {len(df_house):,} House revisions from {CUTOFF_DATE.date()} onwards")

# Load candidate metadata from CSVs
def load_candidate_metadata():
    metadata = []
    for year in ALL_CYCLES:
        csv_path = raw_dir / 'html_parsed_candidates' / f'{year}_house_candidates.csv'
        if not csv_path.exists():
            logger.warning(f"  Missing CSV: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df['election_cycle'] = year
        df['page_title'] = df['wikipedia_url'].apply(
            lambda url: url.split('/')[-1] if pd.notna(url) and url else None
        )
        metadata.append(df)
    df_meta = pd.concat(metadata, ignore_index=True)
    logger.info(f"  Loaded metadata for {len(df_meta):,} candidate-cycles across {len(ALL_CYCLES)} cycles")
    return df_meta

df_metadata = load_candidate_metadata()

# =============================================================================
# STEP 2: Identify Special Election Candidates
# =============================================================================

logger.info("\nIdentifying special election candidates...")

special_elections = df_metadata[df_metadata['election_type'] == 'special'][
    ['page_title', 'election_cycle', 'candidate_name', 'state', 'district']
].drop_duplicates()

logger.info(f"  Found {len(special_elections)} special election candidate-cycles")

# =============================================================================
# STEP 3: Identify Presidential Candidates (cycle-specific)
# =============================================================================

logger.info("\nIdentifying House candidates who also ran for president...")

# Build normalized name lookup for House candidates: (normalized_name) -> [(page_title, cycle)]
house_name_lookup = defaultdict(list)
for _, row in df_metadata[df_metadata['page_title'].notna()].iterrows():
    norm = normalize_name(row['candidate_name'])
    if norm:
        house_name_lookup[norm].append((row['page_title'], row['election_cycle']))

# Collect (page_title, cycle) pairs for exclusion
presidential_exclusions = []  # list of dicts for reporting
presidential_exclusion_pairs = set()  # (page_title, cycle) for filtering

for cycle in PRESIDENTIAL_CYCLES:
    logger.info(f"\n  --- {cycle} Presidential Election ---")

    # Method 1: FEC data matching
    try:
        fec_pres_names = download_fec_presidential_names(cycle)
    except Exception as e:
        logger.warning(f"  Could not download FEC data for {cycle}: {e}")
        fec_pres_names = set()

    matched_this_cycle = set()

    # Match FEC presidential names to House candidates in this cycle
    for norm_name, entries in house_name_lookup.items():
        if norm_name in fec_pres_names:
            for page_title, entry_cycle in entries:
                if entry_cycle == cycle:
                    matched_this_cycle.add(page_title)

    # Record results
    for pt in sorted(matched_this_cycle):
        presidential_exclusion_pairs.add((pt, cycle))
        # Find candidate name from metadata
        name_rows = df_metadata[
            (df_metadata['page_title'] == pt) & (df_metadata['election_cycle'] == cycle)
        ]
        cand_name = name_rows['candidate_name'].iloc[0] if len(name_rows) > 0 else pt.replace('_', ' ')
        state = name_rows['state'].iloc[0] if len(name_rows) > 0 else ''
        district = name_rows['district'].iloc[0] if len(name_rows) > 0 and pd.notna(name_rows['district'].iloc[0]) else ''

        presidential_exclusions.append({
            'election_cycle': cycle,
            'candidate_name': cand_name,
            'page_title': pt,
            'state': state,
            'district': district,
        })
        logger.info(f"    Identified: {cand_name} ({state} {district})")

    logger.info(f"  {cycle}: {len(matched_this_cycle)} House candidates identified as presidential candidates")

logger.info(f"\nTotal presidential exclusion pairs (candidate x cycle): {len(presidential_exclusion_pairs)}")

# Save report on house candidates who also ran for presidential office
df_pres_report = pd.DataFrame(presidential_exclusions)
pres_report_path = output_dir / 'house_presidential_candidates.csv'
df_pres_report.to_csv(pres_report_path, index=False)
logger.info(f"  Saved presidential candidate report: {pres_report_path}")

# =============================================================================
# STEP 4: Create Variation 1 - Full Dataset
# =============================================================================

logger.info("\n" + "="*80)
logger.info("VARIATION 1: Full House Dataset (2006-11-08 onwards)")
logger.info("="*80)

df_v1 = df_house.copy()
df_v1['date'] = pd.to_datetime(df_v1['timestamp'].dt.date)

def create_daily_aggregates(df, name):
    """Create daily aggregates grouped by (date, page_title), then average across pages."""
    daily_by_page = df.groupby(['date', 'page_title']).agg({
        'revid': 'count',
        'user': 'nunique',
    }).reset_index()
    daily_by_page.columns = ['date', 'page_title', 'num_edits', 'unique_editors']

    daily_agg = daily_by_page.groupby('date').agg({
        'num_edits': 'mean',
        'unique_editors': 'mean',
        'page_title': 'nunique'
    }).reset_index()
    daily_agg.columns = ['date', 'mean_edits_per_page', 'mean_editors_per_page', 'num_pages']

    logger.info(f"  {name}: {len(daily_agg):,} days, {daily_by_page['page_title'].nunique():,} pages, {df['revid'].count():,} revisions")
    return daily_agg

daily_v1 = create_daily_aggregates(df_v1, "Variation 1")
daily_v1.to_csv(output_dir / 'daily_edits_house_v1_full.csv', index=False)
logger.info(f"  Saved: daily_edits_house_v1_full.csv")

# =============================================================================
# STEP 5: Create Variation 2 - Exclude Special Elections
# =============================================================================

logger.info("\n" + "="*80)
logger.info("VARIATION 2: Exclude Special Election Cycles")
logger.info("="*80)

df_v2 = df_v1.copy()
excluded_v2 = 0

for _, special in special_elections.iterrows():
    mask = (df_v2['page_title'] == special['page_title']) & (df_v2['election_cycle'] == special['election_cycle'])
    excluded_v2 += mask.sum()
    df_v2 = df_v2[~mask]

logger.info(f"  Excluded {excluded_v2:,} revisions from {len(special_elections)} special election candidate-cycles")

daily_v2 = create_daily_aggregates(df_v2, "Variation 2")
daily_v2.to_csv(output_dir / 'daily_edits_house_v2_no_special.csv', index=False)
logger.info(f"  Saved: daily_edits_house_v2_no_special.csv")

# =============================================================================
# STEP 6: Create Variation 3 - Also Exclude Presidential Candidates (cycle-specific)
# =============================================================================

logger.info("\n" + "="*80)
logger.info("VARIATION 3: Exclude Special Elections + Presidential Candidates (cycle-specific)")
logger.info("="*80)

df_v3 = df_v2.copy()
excluded_v3 = 0

for page_title, cycle in presidential_exclusion_pairs:
    mask = (df_v3['page_title'] == page_title) & (df_v3['election_cycle'] == cycle)
    excluded_v3 += mask.sum()
    df_v3 = df_v3[~mask]

logger.info(f"  Excluded {excluded_v3:,} revisions from {len(presidential_exclusion_pairs)} presidential candidate-cycles")

daily_v3 = create_daily_aggregates(df_v3, "Variation 3")
daily_v3.to_csv(output_dir / 'daily_edits_house_v3_no_special_no_pres.csv', index=False)
logger.info(f"  Saved: daily_edits_house_v3_no_special_no_pres.csv")

# =============================================================================
# STEP 7: Summary
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

summary_path = output_dir / 'variation_summary.csv'
df_summary.to_csv(summary_path, index=False)
logger.info(f"\nSummary saved: {summary_path}")

# Save exclusion lists
exclusions = {
    'special_elections': special_elections.to_dict('records'),
    'presidential_candidates': presidential_exclusions,
    'statistics': {
        'special_election_candidate_cycles': len(special_elections),
        'presidential_candidate_cycles': len(presidential_exclusion_pairs),
        'revisions_excluded_v2': int(excluded_v2),
        'revisions_excluded_v3': int(excluded_v3)
    }
}

with open(output_dir / 'exclusion_lists.json', 'w') as f:
    json.dump(exclusions, f, indent=2, default=str)

logger.info(f"Exclusion lists saved: {output_dir / 'exclusion_lists.json'}")
logger.info("\nDONE! Three variation datasets created successfully.")
