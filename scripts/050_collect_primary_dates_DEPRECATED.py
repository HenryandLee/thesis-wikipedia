"""
050_collect_primary_dates.py

Collect state primary election dates for US House races.

This script:
1. Scrapes NCSL (National Conference of State Legislatures) for recent years (2020-2024)
2. Provides a template for manual compilation of older years (2008-2018)
3. Handles edge cases (Louisiana jungle primary, California/Washington top-two)
4. Outputs a CSV with state, election_year, primary_date, and metadata

Output: data/reference/state_primary_dates.csv
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
from datetime import datetime, date
import logging
import re
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

# NCSL URLs for primary date pages
NCSL_URLS = {
    2024: "https://www.ncsl.org/elections-and-campaigns/2024-state-primary-election-dates",
    2022: "https://www.ncsl.org/elections-and-campaigns/2022-state-primary-election-dates-and-filing-deadlines",
    2020: "https://www.ncsl.org/elections-and-campaigns/2020-state-primary-election-dates",
}

# State name to abbreviation mapping
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

ABBREV_TO_STATE = {v: k for k, v in STATE_ABBREV.items()}

# Special primary types
SPECIAL_PRIMARY_STATES = {
    # Louisiana uses jungle/blanket primary on general election day
    'Louisiana': {
        'primary_type': 'jungle',
        'notes': 'Jungle primary on general election day - excluded from Q2 analysis'
    },
    # California top-two (since 2012)
    'California': {
        'primary_type': 'top_two',
        'notes': 'Top-two primary since 2012'
    },
    # Washington top-two (since 2008)
    'Washington': {
        'primary_type': 'top_two',
        'notes': 'Top-two primary since 2008'
    },
    # Alaska uses blanket primary (since 2022 for general elections)
    'Alaska': {
        'primary_type': 'blanket',
        'notes': 'Uses top-four/blanket primary since 2022'
    }
}

# States with runoff primaries (Southern states primarily)
RUNOFF_STATES = ['Alabama', 'Arkansas', 'Georgia', 'Louisiana', 'Mississippi',
                 'North Carolina', 'Oklahoma', 'South Carolina', 'Texas']


def parse_date_string(date_str: str, year: int) -> date:
    """
    Parse various date string formats into a date object.

    Args:
        date_str: Date string like "March 5", "Mar. 5", "3/5", etc.
        year: The election year to use

    Returns:
        Parsed date object
    """
    date_str = date_str.strip()

    # Try various formats
    formats = [
        "%B %d",      # March 5
        "%b %d",      # Mar 5
        "%b. %d",     # Mar. 5
        "%m/%d",      # 3/5
        "%B %d, %Y",  # March 5, 2024
        "%b %d, %Y",  # Mar 5, 2024
    ]

    for fmt in formats:
        try:
            parsed = datetime.strptime(date_str, fmt)
            # If year not in format, use provided year
            if "%Y" not in fmt:
                return date(year, parsed.month, parsed.day)
            return parsed.date()
        except ValueError:
            continue

    # Try to extract month and day with regex
    match = re.search(r'(\w+\.?)\s+(\d+)', date_str)
    if match:
        month_str, day = match.groups()
        month_map = {
            'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
            'mar': 3, 'march': 3, 'apr': 4, 'april': 4,
            'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
            'aug': 8, 'august': 8, 'sep': 9, 'september': 9, 'sept': 9,
            'oct': 10, 'october': 10, 'nov': 11, 'november': 11,
            'dec': 12, 'december': 12
        }
        month_str_clean = month_str.lower().rstrip('.')
        if month_str_clean in month_map:
            return date(year, month_map[month_str_clean], int(day))

    raise ValueError(f"Could not parse date: {date_str}")


def scrape_ncsl_primary_dates(year: int) -> pd.DataFrame:
    """
    Scrape primary dates from NCSL website for a given election year.

    Args:
        year: Election year (2020, 2022, or 2024)

    Returns:
        DataFrame with columns: state, primary_date, runoff_date, notes
    """
    if year not in NCSL_URLS:
        raise ValueError(f"NCSL URL not available for year {year}")

    url = NCSL_URLS[year]
    logger.info(f"Scraping NCSL for {year}: {url}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch NCSL page: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the primary dates table
    tables = soup.find_all('table')

    records = []
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 2:
                state_cell = cells[0].get_text(strip=True)

                # Check if this looks like a state name
                state_name = None
                for name in STATE_ABBREV.keys():
                    if name.lower() in state_cell.lower():
                        state_name = name
                        break

                if state_name and len(cells) >= 2:
                    date_cell = cells[1].get_text(strip=True)

                    # Skip header rows
                    if 'primary' in date_cell.lower() or 'date' in date_cell.lower():
                        continue

                    try:
                        primary_date = parse_date_string(date_cell, year)

                        # Check for runoff date in additional columns
                        runoff_date = None
                        if len(cells) >= 3:
                            runoff_cell = cells[2].get_text(strip=True)
                            if runoff_cell and runoff_cell not in ['-', 'N/A', '']:
                                try:
                                    runoff_date = parse_date_string(runoff_cell, year)
                                except ValueError:
                                    pass

                        records.append({
                            'state': state_name,
                            'primary_date': primary_date,
                            'runoff_date': runoff_date
                        })

                    except ValueError as e:
                        logger.warning(f"Could not parse date for {state_name}: {date_cell}")

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.drop_duplicates(subset=['state'], keep='first')
        logger.info(f"Scraped {len(df)} states for {year}")

    return df


def create_manual_template() -> pd.DataFrame:
    """
    Create a template DataFrame for manual entry of primary dates.

    Returns:
        DataFrame with all states and election years, ready for manual filling
    """
    years = [2008, 2010, 2012, 2014, 2016, 2018]
    states = list(STATE_ABBREV.keys())

    records = []
    for year in years:
        for state in states:
            records.append({
                'state': state,
                'election_year': year,
                'primary_date': None,
                'runoff_date': None,
                'primary_type': SPECIAL_PRIMARY_STATES.get(state, {}).get('primary_type', 'traditional'),
                'notes': SPECIAL_PRIMARY_STATES.get(state, {}).get('notes', ''),
                'source': 'MANUAL_ENTRY_REQUIRED'
            })

    return pd.DataFrame(records)


def compile_all_primary_dates() -> pd.DataFrame:
    """
    Compile primary dates from all sources into a single DataFrame.

    Returns:
        Complete DataFrame with primary dates for all states and years
    """
    all_records = []

    # Scrape available NCSL years
    for year in [2024, 2022, 2020]:
        try:
            df = scrape_ncsl_primary_dates(year)
            if not df.empty:
                df['election_year'] = year
                df['source'] = 'NCSL'
                all_records.append(df)
        except Exception as e:
            logger.error(f"Failed to scrape {year}: {e}")

    # Create manual template for older years
    manual_template = create_manual_template()
    all_records.append(manual_template)

    # Combine all records
    combined = pd.concat(all_records, ignore_index=True)

    # Add metadata for special states
    for idx, row in combined.iterrows():
        state = row['state']
        if state in SPECIAL_PRIMARY_STATES:
            meta = SPECIAL_PRIMARY_STATES[state]
            combined.loc[idx, 'primary_type'] = meta.get('primary_type', 'traditional')
            if not combined.loc[idx, 'notes']:
                combined.loc[idx, 'notes'] = meta.get('notes', '')

    # Fill default values
    combined['primary_type'] = combined['primary_type'].fillna('traditional')
    combined['notes'] = combined['notes'].fillna('')

    # Sort by year and state
    combined = combined.sort_values(['election_year', 'state']).reset_index(drop=True)

    return combined


def load_or_create_primary_dates() -> pd.DataFrame:
    """
    Load existing primary dates CSV or create new one with scraped data.

    Returns:
        DataFrame with primary dates
    """
    output_path = REFERENCE_DIR / "state_primary_dates.csv"

    if output_path.exists():
        logger.info(f"Loading existing primary dates from {output_path}")
        df = pd.read_csv(output_path, parse_dates=['primary_date', 'runoff_date'])
        return df

    logger.info("Creating new primary dates file...")
    df = compile_all_primary_dates()

    # Convert dates to string format for CSV
    df['primary_date'] = pd.to_datetime(df['primary_date']).dt.strftime('%Y-%m-%d')
    df['runoff_date'] = pd.to_datetime(df['runoff_date']).dt.strftime('%Y-%m-%d')
    df['runoff_date'] = df['runoff_date'].replace('NaT', '')

    df.to_csv(output_path, index=False)
    logger.info(f"Saved primary dates to {output_path}")

    return df


def validate_primary_dates(df: pd.DataFrame) -> dict:
    """
    Validate the primary dates DataFrame for completeness and consistency.

    Args:
        df: Primary dates DataFrame

    Returns:
        Dictionary with validation results
    """
    validation = {
        'total_records': len(df),
        'years_covered': sorted(df['election_year'].unique().tolist()),
        'states_covered': len(df['state'].unique()),
        'missing_dates': [],
        'issues': []
    }

    # Check for missing dates
    for year in [2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]:
        year_data = df[df['election_year'] == year]

        # Check each state
        for state in STATE_ABBREV.keys():
            state_data = year_data[year_data['state'] == state]

            if state_data.empty:
                validation['missing_dates'].append(f"{state} {year}: No record")
            elif pd.isna(state_data['primary_date'].iloc[0]) or state_data['primary_date'].iloc[0] in ['', 'NaT']:
                # Louisiana is expected to have no traditional primary
                if state != 'Louisiana':
                    validation['missing_dates'].append(f"{state} {year}: No primary date")

    # Check for Louisiana (should be marked as jungle primary)
    la_data = df[df['state'] == 'Louisiana']
    if not la_data.empty:
        if not all(la_data['primary_type'] == 'jungle'):
            validation['issues'].append("Louisiana should have primary_type='jungle'")

    logger.info(f"Validation: {len(validation['missing_dates'])} missing dates, {len(validation['issues'])} issues")

    return validation


# =============================================================================
# MANUAL PRIMARY DATE DATA
# =============================================================================
# These are compiled from historical sources. Fill in as needed.

MANUAL_PRIMARY_DATES = {
    # Format: (state, year): (primary_date, runoff_date, notes)
    # 2008 primaries
    ('Alabama', 2008): ('2008-06-03', '2008-07-15', ''),
    ('Alaska', 2008): ('2008-08-26', None, ''),
    ('Arizona', 2008): ('2008-09-02', None, ''),
    ('Arkansas', 2008): ('2008-05-20', '2008-06-10', ''),
    ('California', 2008): ('2008-06-03', None, ''),
    ('Colorado', 2008): ('2008-08-12', None, ''),
    ('Connecticut', 2008): ('2008-08-12', None, ''),
    ('Delaware', 2008): ('2008-09-09', None, ''),
    ('Florida', 2008): ('2008-08-26', None, ''),
    ('Georgia', 2008): ('2008-07-15', '2008-08-05', ''),
    ('Hawaii', 2008): ('2008-09-20', None, ''),
    ('Idaho', 2008): ('2008-05-27', None, ''),
    ('Illinois', 2008): ('2008-02-05', None, 'Super Tuesday'),
    ('Indiana', 2008): ('2008-05-06', None, ''),
    ('Iowa', 2008): ('2008-06-03', None, ''),
    ('Kansas', 2008): ('2008-08-05', None, ''),
    ('Kentucky', 2008): ('2008-05-20', None, ''),
    ('Louisiana', 2008): (None, None, 'Jungle primary on general election day'),
    ('Maine', 2008): ('2008-06-10', None, ''),
    ('Maryland', 2008): ('2008-02-12', None, ''),
    ('Massachusetts', 2008): ('2008-09-16', None, ''),
    ('Michigan', 2008): ('2008-08-05', None, ''),
    ('Minnesota', 2008): ('2008-09-09', None, ''),
    ('Mississippi', 2008): ('2008-03-11', '2008-04-01', ''),
    ('Missouri', 2008): ('2008-08-05', None, ''),
    ('Montana', 2008): ('2008-06-03', None, ''),
    ('Nebraska', 2008): ('2008-05-13', None, ''),
    ('Nevada', 2008): ('2008-08-12', None, ''),
    ('New Hampshire', 2008): ('2008-09-09', None, ''),
    ('New Jersey', 2008): ('2008-06-03', None, ''),
    ('New Mexico', 2008): ('2008-06-03', None, ''),
    ('New York', 2008): ('2008-09-09', None, ''),
    ('North Carolina', 2008): ('2008-05-06', '2008-06-24', ''),
    ('North Dakota', 2008): ('2008-06-10', None, ''),
    ('Ohio', 2008): ('2008-03-04', None, ''),
    ('Oklahoma', 2008): ('2008-07-29', '2008-08-26', ''),
    ('Oregon', 2008): ('2008-05-20', None, ''),
    ('Pennsylvania', 2008): ('2008-04-22', None, ''),
    ('Rhode Island', 2008): ('2008-09-09', None, ''),
    ('South Carolina', 2008): ('2008-06-10', '2008-06-24', ''),
    ('South Dakota', 2008): ('2008-06-03', None, ''),
    ('Tennessee', 2008): ('2008-08-07', None, ''),
    ('Texas', 2008): ('2008-03-04', '2008-04-08', ''),
    ('Utah', 2008): ('2008-06-24', None, ''),
    ('Vermont', 2008): ('2008-09-09', None, ''),
    ('Virginia', 2008): ('2008-06-10', None, ''),
    ('Washington', 2008): ('2008-08-19', None, 'Top-two primary'),
    ('West Virginia', 2008): ('2008-05-13', None, ''),
    ('Wisconsin', 2008): ('2008-09-09', None, ''),
    ('Wyoming', 2008): ('2008-08-19', None, ''),

    # Add more years as needed...
    # The script will use scraped data for 2020-2024 and this manual data for earlier years
}


def apply_manual_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply manually compiled primary dates to the DataFrame.

    Args:
        df: Primary dates DataFrame with potentially missing dates

    Returns:
        Updated DataFrame with manual dates filled in
    """
    for (state, year), (primary, runoff, notes) in MANUAL_PRIMARY_DATES.items():
        mask = (df['state'] == state) & (df['election_year'] == year)

        if mask.any():
            if primary:
                df.loc[mask, 'primary_date'] = primary
            if runoff:
                df.loc[mask, 'runoff_date'] = runoff
            if notes:
                df.loc[mask, 'notes'] = notes
            df.loc[mask, 'source'] = 'manual'

    return df


def main():
    """Main entry point for primary date collection."""
    logger.info("=" * 60)
    logger.info("COLLECTING STATE PRIMARY ELECTION DATES")
    logger.info("=" * 60)

    # Compile primary dates from all sources
    df = compile_all_primary_dates()

    # Apply manual dates
    df = apply_manual_dates(df)

    # Validate
    validation = validate_primary_dates(df)

    # Save
    output_path = REFERENCE_DIR / "state_primary_dates.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")

    # Save validation report
    validation_path = REFERENCE_DIR / "primary_dates_validation.json"
    with open(validation_path, 'w') as f:
        json.dump(validation, f, indent=2, default=str)
    logger.info(f"Validation report saved to {validation_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total records: {len(df)}")
    print(f"Years covered: {validation['years_covered']}")
    print(f"States covered: {validation['states_covered']}")
    print(f"Missing dates: {len(validation['missing_dates'])}")

    if validation['missing_dates']:
        print("\nMissing dates (first 20):")
        for item in validation['missing_dates'][:20]:
            print(f"  - {item}")
        if len(validation['missing_dates']) > 20:
            print(f"  ... and {len(validation['missing_dates']) - 20} more")

    print(f"\nOutput: {output_path}")

    return df


if __name__ == "__main__":
    main()
