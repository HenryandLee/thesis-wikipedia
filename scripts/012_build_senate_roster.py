"""
US Senate Elections Scraper
Scrapes candidate data from Wikipedia Senate election pages

Usage:
    python scripts/012_build_senate_roster.py 2024
    python scripts/012_build_senate_roster.py all
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import sys
from typing import List, Dict, Optional


def parse_candidate_from_li(li, state: str, incumbent_name: str, incumbent_party: str,
                            electoral_history: str, result_text: str, pvi: str,
                            last_election_result: str, election_type: str) -> Optional[Dict]:
    """
    Parse a single candidate from an <li> element
    Format: ▌YName(Party) percentage%
    """
    text = li.get_text(strip=True)

    # Check if winner (has ▌Y marker)
    is_winner = '▌Y' in text or text.startswith('Y')

    # Extract party first - in parentheses
    party_match = re.search(r'\(([^)]+)\)', text)
    party = party_match.group(1).strip() if party_match else ''

    # Extract candidate name - look for <a> tag first
    candidate_link = li.find('a', href=lambda x: x and '/wiki/' in x and ':' not in x)

    if candidate_link:
        link_text = candidate_link.get_text(strip=True)
        # Check if this link is the party (inside parentheses) or the candidate name
        # If the link text matches the party, it's a party link, not a candidate link
        if link_text == party:
            # This is a party link, parse candidate name from text
            match = re.search(r'([^(▌Y]+)\s*\(', text)
            if match:
                candidate_name = match.group(1).strip()
                wikipedia_url = None
            else:
                return None
        else:
            # This is a candidate link
            candidate_name = link_text
            wikipedia_url = 'https://en.wikipedia.org' + candidate_link['href']
    else:
        # No link, try to parse from text
        # Pattern: Name(Party) percentage%
        match = re.search(r'([^(▌Y]+)\s*\(', text)
        if match:
            candidate_name = match.group(1).strip()
            wikipedia_url = None
        else:
            return None

    # Extract vote percentage
    percent_match = re.search(r'([\d.]+)%', text)
    if not percent_match:
        return None

    vote_percentage = float(percent_match.group(1))

    return {
        'state': state,
        'incumbent_name': incumbent_name,
        'incumbent_party': incumbent_party,
        'electoral_history': electoral_history,
        'result_text': result_text,
        'candidate_name': candidate_name,
        'party': party,
        'vote_percentage': vote_percentage,
        'is_winner': is_winner,
        'pvi': pvi,
        'last_election_result': last_election_result,
        'wikipedia_url': wikipedia_url,
        'election_type': election_type
    }


def parse_pvi_table(table, table_format: str = '2024') -> Dict[str, Dict[str, str]]:
    """
    Parse PVI table to extract PVI and last election result by state
    Returns: {state_name: {'pvi': 'R+2', 'last_election_result': '49.96% D'}}

    table_format: '2024', '2022', or '2012' (different column structures)
    """
    pvi_data = {}

    rows = table.find_all('tr')
    data_rows = [row for row in rows if row.find('td')]

    for row in data_rows:
        cells = row.find_all(['td', 'th'])

        if table_format == '2024':
            # 2024 format: Cell[0]=State, Cell[1]=PVI, Cell[3]=Last election
            if len(cells) < 4:
                continue

            # Cell[0]: State (may have footnotes like "California[l]")
            state = cells[0].get_text(strip=True)
            # Remove footnote markers
            state = re.sub(r'\[.*?\]', '', state).strip()

            # Cell[1]: PVI
            pvi = cells[1].get_text(strip=True)

            # Cell[3]: Last election result
            last_election = cells[3].get_text(strip=True)

        elif table_format == '2012':
            # 2012 format: Cell[0]=State, Cell[1]=Incumbent, Cell[2]=Last election
            # No PVI data available for 2012
            if len(cells) < 3:
                continue

            # Cell[0]: State
            state = cells[0].get_text(strip=True)
            # Remove footnote markers
            state = re.sub(r'\[.*?\]', '', state).strip()

            # No PVI data for 2012
            pvi = ''

            # Cell[2]: Last election result (e.g., "53.3% R")
            last_election = cells[2].get_text(strip=True)

        else:  # 2022 format
            # 2022 format: Cell[0]=State, Cell[1]=PVI, Cell[3]=Last election
            if len(cells) < 4:
                continue

            # Cell[0]: State
            state = cells[0].get_text(strip=True)
            # Remove footnote markers and class info
            state = re.sub(r'\[.*?\]', '', state).strip()

            # Cell[1]: PVI
            pvi = cells[1].get_text(strip=True)

            # Cell[3]: Last election result (e.g., "64.0% R")
            last_election = cells[3].get_text(strip=True)

        pvi_data[state] = {
            'pvi': pvi,
            'last_election_result': last_election
        }

    return pvi_data


def parse_race_table(table, pvi_data: Dict[str, Dict[str, str]], election_type: str) -> List[Dict]:
    """
    Parse a race results table (either special or general elections)
    Table structure:
    - Cell[0]: State
    - Cell[1]: Incumbent name
    - Cell[2]: Incumbent party
    - Cell[3]: Electoral history
    - Cell[4]: Result text
    - Cell[5]: Candidates (with <li> elements)
    """
    candidates = []

    rows = table.find_all('tr')
    data_rows = [row for row in rows if row.find('td')]

    for row in data_rows:
        cells = row.find_all(['td', 'th'])

        if len(cells) < 6:
            continue

        # Extract race information
        state = cells[0].get_text(strip=True)
        # Remove class info like "(Class 1)" and footnotes
        state = re.sub(r'\(Class \d\)', '', state)
        state = re.sub(r'\[.*?\]', '', state).strip()

        incumbent_name = cells[1].get_text(strip=True)
        incumbent_party = cells[2].get_text(strip=True)
        # Use separator to convert line breaks to commas
        electoral_history = cells[3].get_text(separator=', ', strip=True)
        # Clean up footnote markers that got separated (e.g., "2018, [, k, ]" -> "2018")
        electoral_history = re.sub(r',\s*\[.*?\]', '', electoral_history)
        electoral_history = re.sub(r',\s*\[', '', electoral_history)  # Remove orphaned [
        electoral_history = re.sub(r',\s*\]', '', electoral_history)  # Remove orphaned ]
        electoral_history = re.sub(r',\s*[a-z](?=\s*,|\s*$)', '', electoral_history)  # Remove single letters
        result_text = cells[4].get_text(strip=True)

        # Get PVI data for this state
        # Try exact match first, then try with election_type suffix (for 2022 Oklahoma case)
        state_pvi_data = pvi_data.get(state)
        if not state_pvi_data:
            # Try with election type suffix (e.g., "Oklahoma(special)")
            state_pvi_data = pvi_data.get(f"{state}({election_type})")
        if not state_pvi_data and election_type == 'general':
            # Wikipedia might use "(regular)" instead of "(general)"
            state_pvi_data = pvi_data.get(f"{state}(regular)")
        if not state_pvi_data:
            # Flexible fallback: find any key that starts with state and contains election_type
            # Handles typos like "South Carolinaspecial)" matching "South Carolina" + "special"
            for key in pvi_data.keys():
                if key.startswith(state) and election_type in key.lower():
                    state_pvi_data = pvi_data[key]
                    break
        if not state_pvi_data:
            state_pvi_data = {}

        pvi = state_pvi_data.get('pvi', '')
        last_election_result = state_pvi_data.get('last_election_result', '')

        # Parse candidates from Cell[5]
        candidates_cell = cells[5]
        list_items = candidates_cell.find_all('li')

        if list_items:
            for li in list_items:
                candidate_data = parse_candidate_from_li(
                    li, state, incumbent_name, incumbent_party,
                    electoral_history, result_text, pvi,
                    last_election_result, election_type
                )
                if candidate_data:
                    candidates.append(candidate_data)

    return candidates


def scrape_senate_elections(year: int) -> pd.DataFrame:
    """
    Scrape Senate election data for a given year
    """
    config = YEAR_CONFIGS.get(year)
    if not config:
        raise ValueError(f"No configuration found for year {year}")

    url = config['url']
    pvi_table_index = config['pvi_table_index']
    special_table_index = config.get('special_table_index')
    general_table_index = config['general_table_index']
    table_format = config.get('table_format', '2024')  # Default to 2024 format

    print(f"Scraping year: {year}")
    print("="*80)
    print(f"{year} US SENATE ELECTIONS - CANDIDATE SCRAPER")
    print("="*80)

    # Fetch page
    print("\n1. Fetching Wikipedia page...")
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    print("   ✓ Page fetched successfully")

    # Find all tables
    wikitables = soup.find_all('table', class_='wikitable')
    print(f"\n2. Found {len(wikitables)} total tables on page")

    # Parse PVI table (Table 4)
    print(f"\n3. Parsing PVI/ratings table (Table {pvi_table_index})...")
    pvi_table = wikitables[pvi_table_index]
    pvi_data = parse_pvi_table(pvi_table, table_format=table_format)
    print(f"   ✓ Extracted PVI data for {len(pvi_data)} states")

    # Parse special elections table if exists
    all_candidates = []

    if special_table_index is not None:
        print(f"\n4. Parsing special elections table (Table {special_table_index})...")
        special_table = wikitables[special_table_index]
        special_candidates = parse_race_table(special_table, pvi_data, 'special')
        all_candidates.extend(special_candidates)
        print(f"   ✓ Extracted {len(special_candidates)} candidates from special elections")

    # Parse general elections table
    table_num = 5 if special_table_index is not None else 4
    print(f"\n{table_num}. Parsing general elections table (Table {general_table_index})...")
    general_table = wikitables[general_table_index]
    general_candidates = parse_race_table(general_table, pvi_data, 'general')
    all_candidates.extend(general_candidates)
    print(f"   ✓ Extracted {len(general_candidates)} candidates from general elections")

    # Create DataFrame
    print(f"\n{table_num + 1}. Creating DataFrame...")
    df = pd.DataFrame(all_candidates)

    print(f"   ✓ Created DataFrame with {len(df)} total candidates")
    print(f"\n   Summary:")
    print(f"   - States: {df['state'].nunique()}")
    print(f"   - Races: {df.groupby(['state', 'election_type']).ngroups}")
    print(f"   - Candidates with Wikipedia pages: {df['wikipedia_url'].notna().sum()}")
    print(f"   - Winners: {df['is_winner'].sum()}")

    if 'election_type' in df.columns:
        print(f"\n   Election Type Breakdown:")
        print(df['election_type'].value_counts().to_string())

    # Show sample
    print(f"\n{table_num + 2}. Sample of data (first 10 rows):")
    sample_cols = ['state', 'candidate_name', 'party', 'vote_percentage', 'is_winner', 'pvi', 'election_type']
    print(df[sample_cols].head(10).to_string(index=False))

    # Save to CSV
    output_path = f'data/raw/html_parsed_candidates/{year}_senate_candidates.csv'
    print(f"\n{table_num + 3}. Saving to CSV: {output_path}")
    df.to_csv(output_path, index=False)
    print("   ✓ Saved successfully!")

    print("\n" + "="*80)
    print("SCRAPING COMPLETE!")
    print("="*80)

    return df


# Configuration for different election years
YEAR_CONFIGS = {
    2024: {
        'url': 'https://en.wikipedia.org/wiki/2024_United_States_Senate_elections',
        'pvi_table_index': 4,           # Table with PVI and ratings
        'special_table_index': 7,       # Special elections (CA, NE)
        'general_table_index': 8,       # Regular elections (33 races)
        'table_format': '2024',
    },
    2022: {
        'url': 'https://en.wikipedia.org/wiki/2022_United_States_Senate_elections',
        'pvi_table_index': 7,           # Table with PVI and ratings
        'special_table_index': 10,      # Special elections (CA, OK)
        'general_table_index': 11,      # Regular elections (34 races)
        'table_format': '2022',
    },
    2020: {
        'url': 'https://en.wikipedia.org/wiki/2020_United_States_Senate_elections',
        'pvi_table_index': 5,           # Table with PVI and ratings
        'special_table_index': 10,      # Special elections (AZ, GA)
        'general_table_index': 11,      # Regular elections (33 races)
        'table_format': '2022',         # Same format as 2022
    },
    2018: {
        'url': 'https://en.wikipedia.org/wiki/2018_United_States_Senate_elections',
        'pvi_table_index': 4,           # Table with PVI and ratings
        'special_table_index': 9,       # Special elections (MN, MS)
        'general_table_index': 10,      # Regular elections (33 races)
        'table_format': '2022',         # Same format as 2022
    },
    2016: {
        'url': 'https://en.wikipedia.org/wiki/2016_United_States_Senate_elections',
        'pvi_table_index': 4,           # Table with PVI and ratings
        'special_table_index': None,    # No special elections in 2016
        'general_table_index': 9,       # Regular elections (34 races)
        'table_format': '2022',         # Same format as 2022
    },
    2014: {
        'url': 'https://en.wikipedia.org/wiki/2014_United_States_Senate_elections',
        'pvi_table_index': 12,          # Table with PVI and ratings
        'special_table_index': 8,       # Special elections (HI, OK, SC)
        'general_table_index': 9,       # Regular elections (33 races)
        'table_format': '2022',         # Same format as 2022
    },
    2012: {
        'url': 'https://en.wikipedia.org/wiki/2012_United_States_Senate_elections',
        'pvi_table_index': 6,           # Table with ratings (no PVI, but has last election)
        'special_table_index': None,    # No special elections in 2012
        'general_table_index': 4,       # Regular elections (33 races)
        'table_format': '2012',         # 2012 has different table structure (no PVI)
    },
    2010: {
        'url': 'https://en.wikipedia.org/wiki/2010_United_States_Senate_elections',
        'pvi_table_index': 5,           # Table with ratings (no PVI, but has last election)
        'special_table_index': 9,       # Special elections (5 races: MA, DE, IL, NY, WV)
        'general_table_index': 10,      # Regular elections (34 races)
        'table_format': '2012',         # Same structure as 2012 (no PVI)
    },
    2008: {
        'url': 'https://en.wikipedia.org/wiki/2008_United_States_Senate_elections',
        'pvi_table_index': 8,           # Table with ratings (no PVI, but has last election)
        'special_table_index': 5,       # Special elections (2 races: MS, WY)
        'general_table_index': 6,       # Regular elections (33 races)
        'table_format': '2012',         # Same structure as 2012 (no PVI)
    },
}


def main():
    """Main execution function"""
    if len(sys.argv) < 2:
        print("Usage: python scripts/012_build_senate_roster.py <year>")
        print("       python scripts/012_build_senate_roster.py all")
        print(f"\nAvailable years: {', '.join(map(str, YEAR_CONFIGS.keys()))}")
        sys.exit(1)

    year_arg = sys.argv[1]

    if year_arg.lower() == 'all':
        # Scrape all configured years
        for year in sorted(YEAR_CONFIGS.keys()):
            try:
                scrape_senate_elections(year)
                print("\n")
            except Exception as e:
                print(f"Error scraping {year}: {e}")
                import traceback
                traceback.print_exc()
    else:
        # Scrape specific year
        try:
            year = int(year_arg)
            scrape_senate_elections(year)
        except ValueError:
            print(f"Error: '{year_arg}' is not a valid year")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == '__main__':
    main()
