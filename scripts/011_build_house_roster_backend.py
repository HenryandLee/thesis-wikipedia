"""
011_build_house_roster_backend.py

Scrape US House election candidate data from Wikipedia for 2008-2024.
Collects candidates from state-by-state tables (general elections) and
special elections. Uses year-specific parsers to handle format differences
across cycles.

Input:
    Wikipedia election pages (fetched at runtime):
        https://en.wikipedia.org/wiki/{year}_United_States_House_Representatives_elections

Output:
    data/raw/html_parsed_candidates/{year}_house_candidates.csv  (one per year)

Usage:
    python scripts/011_build_house_roster_backend.py [year]
        year: 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, or 2024
        If year is omitted, all years are scraped.

Notes:
    2016: some states use plain text instead of <li> elements
    2012: table header is one line instead of two
    2008: some states lack the CPVI column
    election_type values: general, special, general_rcv_round1,
        general_rcv_runoff, special_rcv_round1, special_rcv_runoff
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Dict, Optional
import re
import time

def fetch_wikipedia_page(url: str) -> BeautifulSoup:
    """Fetch Wikipedia page with proper headers"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return BeautifulSoup(response.content, 'html.parser')


def determine_election_type(li_element, source_table: str) -> str:
    """
    Determine the type of election based on source table and context

    Returns one of:
    - 'special' - Special election
    - 'general' - General election
    - 'special_rcv_round1' - Special election ranked-choice round 1
    - 'special_rcv_runoff' - Special election ranked-choice runoff
    - 'general_rcv_round1' - General election ranked-choice round 1
    - 'general_rcv_runoff' - General election ranked-choice runoff
    """

    # Get surrounding context by looking at all previous siblings
    is_first_round = False
    is_runoff = False

    # Check all previous <li> siblings to find section markers
    current = li_element
    while current:
        prev_sibling = current.find_previous_sibling('li')
        if prev_sibling:
            prev_text = prev_sibling.get_text(strip=True)
            # Check if we've hit a section marker
            if 'First round:' in prev_text:
                is_first_round = True
                break
            elif 'Instant runoff' in prev_text or prev_text.startswith('Instant runoff'):
                is_runoff = True
                break
            current = prev_sibling
        else:
            break

    # If no marker found in siblings, we're in a non-RCV election
    # (RCV elections always have the "First round:" or "Instant runoff" markers)

    # Determine base type from source table
    if source_table == 'special':
        if is_runoff:
            return 'special_rcv_runoff'
        elif is_first_round:
            return 'special_rcv_round1'
        else:
            return 'special'
    else:  # general election
        if is_runoff:
            return 'general_rcv_runoff'
        elif is_first_round:
            return 'general_rcv_round1'
        else:
            return 'general'


def parse_candidate_from_li(li_element, district_name: str, state: str, incumbent_name: str,
                            pvi: str, source_table: str, context_lis: list = None) -> Optional[Dict]:
    """
    Parse a single <li> element to extract candidate information
    Format: "▌Y[Name]([Party]) [percentage]%"

    Args:
        context_lis: All <li> elements in the same list for context
    """
    text = li_element.get_text(strip=True)

    # Skip empty or header lines
    if not text or 'First round:' in text or 'Instant runoff' in text:
        return None

    # Extract party first (in parentheses)
    party_match = re.search(r'\(([^)]+)\)', text)
    party = party_match.group(1) if party_match else None

    # Extract candidate name with link if available
    link = li_element.find('a', href=True)
    if link and '/wiki/' in link['href'] and not link['href'].startswith('#'):
        link_text = link.get_text(strip=True)

        # Check if this link is the party (inside parentheses) or the candidate name
        if link_text == party:
            # This is a party link, parse candidate name from text
            match = re.search(r'[▌Y]*(.*?)\s*\(', text)
            if match:
                candidate_name = match.group(1).strip()
                wikipedia_url = None
            else:
                return None
        else:
            # This is the candidate link
            candidate_name = link_text
            wikipedia_url = f"https://en.wikipedia.org{link['href']}"
    else:
        # No link - extract name from text pattern
        # Pattern: "▌Y[Name]([Party]) [percentage]%"
        match = re.search(r'[▌Y]*(.*?)\s*\(', text)
        if match:
            candidate_name = match.group(1).strip()
            wikipedia_url = None
        else:
            return None

    # Extract vote percentage
    pct_match = re.search(r'([\d.]+)%', text)
    vote_percentage = float(pct_match.group(1)) if pct_match else None

    # Check if incumbent
    is_incumbent = (candidate_name.lower() == incumbent_name.lower()) if incumbent_name else False

    # Check if winner (has ▌Y marker)
    is_winner = '▌Y' in text or text.startswith('Y')

    # Determine election type
    election_type = determine_election_type(li_element, source_table)

    return {
        'state': state,
        'district': district_name,
        'candidate_name': candidate_name,
        'party': party,
        'vote_percentage': vote_percentage,
        'is_incumbent': is_incumbent,
        'is_winner': is_winner,
        'pvi': pvi,
        'wikipedia_url': wikipedia_url,
        'election_type': election_type
    }


def extract_state_from_district(district_text: str) -> str:
    """Extract state name from district text like 'Alabama 1' or 'Alaska at-large'"""
    # Remove district number and extra info
    state = re.sub(r'\s*\d+.*', '', district_text)
    state = re.sub(r'\s*at-large.*', '', state, flags=re.IGNORECASE)
    return state.strip()


def parse_candidates_from_plain_text(text: str, district_name: str, state: str,
                                     incumbent_name: str, pvi: str, source_table: str) -> List[Dict]:
    """
    Parse candidates from plain text (for 2016 format without <li> elements)
    Format: "▌YName(Party) percentage%▌YName(Party) percentage%"

    DEPRECATED: This function only parses text and loses embedded Wikipedia URLs.
    Use parse_candidates_from_plain_text_cell() instead for better URL extraction.
    """
    candidates = []

    # Split by the ▌ or Y markers
    # Pattern: Capture marker, name, party, percentage
    pattern = r'([▌Y]*)([^▌]+?)\s*\(([^)]+)\)\s*([\d.]+)%'
    matches = re.findall(pattern, text)

    for match in matches:
        marker = match[0]
        candidate_name = match[1].strip()
        party = match[2].strip()
        vote_percentage = float(match[3])

        # Check if incumbent
        is_incumbent = (candidate_name.lower() == incumbent_name.lower()) if incumbent_name else False

        # Check if winner (has ▌Y marker)
        is_winner = '▌Y' in marker or marker.startswith('Y')

        candidates.append({
            'state': state,
            'district': district_name,
            'candidate_name': candidate_name,
            'party': party,
            'vote_percentage': vote_percentage,
            'is_incumbent': is_incumbent,
            'is_winner': is_winner,
            'pvi': pvi,
            'wikipedia_url': None,  # Plain text format doesn't have links
            'election_type': 'special' if source_table == 'special' else 'general'
        })

    return candidates


def parse_candidates_from_plain_text_cell(cell_element, district_name: str, state: str,
                                          incumbent_name: str, pvi: str, source_table: str) -> List[Dict]:
    """
    Parse candidates from plain text cell HTML (for 2016 format without <li> elements)
    Extracts embedded Wikipedia URLs from <a> tags while parsing candidate data

    Format: "▌YName(Party) percentage%▌YName(Party) percentage%"
    HTML may contain: <a href="/wiki/Name">Name</a> for candidates with Wikipedia pages

    Args:
        cell_element: BeautifulSoup element (the candidates cell)
        district_name, state, incumbent_name, pvi: Candidate metadata
        source_table: 'general' or 'special'

    Returns:
        List of candidate dictionaries with wikipedia_url extracted from links
    """
    candidates = []

    # Extract all Wikipedia links from the cell
    # Map normalized candidate names to URLs for matching
    name_to_url = {}
    for link in cell_element.find_all('a', href=True):
        href = link['href']
        if '/wiki/' in href and not href.startswith('#'):
            link_text = link.get_text(strip=True)
            # Normalize name for matching (lowercase, remove extra spaces)
            normalized = link_text.lower().strip()
            name_to_url[normalized] = f"https://en.wikipedia.org{href}"

    # Get plain text for pattern matching
    text = cell_element.get_text(strip=True)

    # Pattern: Capture marker, name, party, percentage
    pattern = r'([▌Y]*)([^▌]+?)\s*\(([^)]+)\)\s*([\d.]+)%'
    matches = re.findall(pattern, text)

    for match in matches:
        marker = match[0]
        candidate_name = match[1].strip()
        party = match[2].strip()
        vote_percentage = float(match[3])

        # Check if incumbent
        is_incumbent = (candidate_name.lower() == incumbent_name.lower()) if incumbent_name else False

        # Check if winner (has ▌Y marker)
        is_winner = '▌Y' in marker or marker.startswith('Y')

        # Try to find Wikipedia URL for this candidate
        normalized_name = candidate_name.lower().strip()
        wikipedia_url = name_to_url.get(normalized_name)

        candidates.append({
            'state': state,
            'district': district_name,
            'candidate_name': candidate_name,
            'party': party,
            'vote_percentage': vote_percentage,
            'is_incumbent': is_incumbent,
            'is_winner': is_winner,
            'pvi': pvi,
            'wikipedia_url': wikipedia_url,
            'election_type': 'special' if source_table == 'special' else 'general'
        })

    return candidates


def parse_state_table_2022(table, state_name: str = None) -> List[Dict]:
    """
    Parse state table specifically for 2022 elections
    2022 has structure with 7 or 5 cells per row:
    - 7 cells: Normal district (District, PVI, Incumbent, Party, First elected, Status, Candidates)
    - 5 cells: New/redistricted district (District, PVI, Incumbent, Status, Candidates)
    """
    candidates = []
    all_rows = table.find_all('tr')
    data_rows = [row for row in all_rows if row.find('td')]

    for row in data_rows:
        cells = row.find_all(['td', 'th'])

        # Need at least 5 cells for valid data
        if len(cells) < 5:
            continue

        # Extract based on cell count
        if len(cells) == 7:
            # Normal district
            district = cells[0].get_text(strip=True)
            pvi = cells[1].get_text(strip=True)
            incumbent_name = cells[2].get_text(strip=True)
            candidates_cell = cells[6]
        elif len(cells) == 5:
            # New/redistricted district
            district = cells[0].get_text(strip=True)
            pvi = cells[1].get_text(strip=True)
            incumbent_name = cells[2].get_text(strip=True)
            candidates_cell = cells[4]
        else:
            continue

        # Clean PVI (remove footnotes)
        pvi = re.sub(r'\[.*?\]', '', pvi).strip()

        # Get state from district if not provided
        if not state_name:
            state_name = extract_state_from_district(district)

        # Parse candidates from candidates_cell
        list_items = candidates_cell.find_all('li')

        if list_items:
            # Candidates in <li> elements (contested races)
            for li in list_items:
                candidate_data = parse_candidate_from_li(
                    li, district, state_name, incumbent_name, pvi,
                    source_table='general', context_lis=list_items
                )
                if candidate_data:
                    candidates.append(candidate_data)
        else:
            # No <li> elements - check for plain text (uncontested 100% races)
            text = candidates_cell.get_text(strip=True)
            if text and '%' in text:
                plain_text_candidates = parse_candidates_from_plain_text_cell(
                    candidates_cell, district, state_name, incumbent_name, pvi, 'general'
                )
                candidates.extend(plain_text_candidates)

    return candidates


def parse_state_table_2008(table, state_name: str = None) -> List[Dict]:
    """
    Parse state table specifically for 2008 elections
    2008 has THREE different structures on the same page:
    - 7 cells (AL, AK, AZ, AR): District, PVI, Incumbent, Party, First elected, Status, Candidates
    - 6 cells (CA onwards): District, Incumbent, Party, First elected, Status, Candidates (NO PVI)
    - 4 cells (vacant seats): District, Incumbent (colspan=3), Status, Candidates (NO PVI)
    Excludes non-voting delegate seats from territories
    """
    candidates = []
    all_rows = table.find_all('tr')
    data_rows = [row for row in all_rows if row.find('td')]

    for row in data_rows:
        cells = row.find_all(['td', 'th'])

        # Need at least 4 cells for valid data
        if len(cells) < 4:
            continue

        # Extract based on cell count
        if len(cells) == 7:
            # Format with PVI (Alabama, Alaska, Arizona, Arkansas)
            district = cells[0].get_text(strip=True)
            pvi = cells[1].get_text(strip=True)
            incumbent_name = cells[2].get_text(strip=True)
            candidates_cell = cells[6]
        elif len(cells) == 6:
            # Format without PVI (California onwards)
            district = cells[0].get_text(strip=True)
            pvi = ''  # No PVI column
            incumbent_name = cells[1].get_text(strip=True)
            candidates_cell = cells[5]
        elif len(cells) == 4:
            # Vacant seat format (e.g., Ohio 11)
            # Cell[0]: District, Cell[1]: Vacant (colspan=3), Cell[2]: Status, Cell[3]: Candidates
            district = cells[0].get_text(strip=True)
            pvi = ''  # No PVI column
            incumbent_name = cells[1].get_text(strip=True)
            candidates_cell = cells[3]
        else:
            continue

        # Skip non-voting delegate seats (territories)
        # Normalize district name (replace non-breaking spaces with regular spaces)
        district_normalized = district.replace('\xa0', ' ')
        non_voting_territories = [
            'American Samoa at-large',
            'District of Columbia at-large',
            'Guam at-large',
            'Northern Mariana Islands at-large',
            'Puerto Rico at-large',
            'United States Virgin Islands at-large',
            'U.S. Virgin Islands at-large'
        ]
        if district_normalized in non_voting_territories:
            continue

        # Clean PVI (remove footnotes)
        pvi = re.sub(r'\[.*?\]', '', pvi).strip()

        # Get state from district if not provided
        if not state_name:
            state_name = extract_state_from_district(district)

        # Parse candidates from candidates_cell
        list_items = candidates_cell.find_all('li')

        if list_items:
            # Candidates in <li> elements (contested races)
            for li in list_items:
                candidate_data = parse_candidate_from_li(
                    li, district, state_name, incumbent_name, pvi,
                    source_table='general', context_lis=list_items
                )
                if candidate_data:
                    candidates.append(candidate_data)
        else:
            # No <li> elements - check for plain text (uncontested 100% races)
            text = candidates_cell.get_text(strip=True)
            if text and '%' in text:
                plain_text_candidates = parse_candidates_from_plain_text_cell(
                    candidates_cell, district, state_name, incumbent_name, pvi, 'general'
                )
                candidates.extend(plain_text_candidates)

    return candidates


def parse_state_table_2010(table, state_name: str = None) -> List[Dict]:
    """
    Parse state table specifically for 2010 elections
    2010 has structure with 7 or 5 cells per row:
    - 7 cells: Normal district (District, PVI, Incumbent, Party, First elected, Status, Candidates)
    - 5 cells: Vacant district (District, PVI, Incumbent, Status, Candidates)
    Excludes non-voting delegate seats from territories
    """
    candidates = []
    all_rows = table.find_all('tr')
    data_rows = [row for row in all_rows if row.find('td')]

    for row in data_rows:
        cells = row.find_all(['td', 'th'])

        # Need at least 5 cells for valid data
        if len(cells) < 5:
            continue

        # Extract based on cell count
        if len(cells) == 7:
            # Normal district
            district = cells[0].get_text(strip=True)
            pvi = cells[1].get_text(strip=True)
            incumbent_name = cells[2].get_text(strip=True)
            candidates_cell = cells[6]
        elif len(cells) == 5:
            # Vacant district
            district = cells[0].get_text(strip=True)
            pvi = cells[1].get_text(strip=True)
            incumbent_name = cells[2].get_text(strip=True)
            candidates_cell = cells[4]
        else:
            continue

        # Skip non-voting delegate seats (territories)
        # Normalize district name (replace non-breaking spaces with regular spaces)
        district_normalized = district.replace('\xa0', ' ')
        non_voting_territories = [
            'American Samoa at-large',
            'District of Columbia at-large',
            'Guam at-large',
            'Northern Mariana Islands at-large',
            'Puerto Rico at-large',
            'United States Virgin Islands at-large',
            'U.S. Virgin Islands at-large'
        ]
        if district_normalized in non_voting_territories:
            continue

        # Clean PVI (remove footnotes)
        pvi = re.sub(r'\[.*?\]', '', pvi).strip()

        # Get state from district if not provided
        if not state_name:
            state_name = extract_state_from_district(district)

        # Parse candidates from candidates_cell
        list_items = candidates_cell.find_all('li')

        if list_items:
            # Candidates in <li> elements (contested races)
            for li in list_items:
                candidate_data = parse_candidate_from_li(
                    li, district, state_name, incumbent_name, pvi,
                    source_table='general', context_lis=list_items
                )
                if candidate_data:
                    candidates.append(candidate_data)
        else:
            # No <li> elements - check for plain text (uncontested 100% races)
            text = candidates_cell.get_text(strip=True)
            if text and '%' in text:
                plain_text_candidates = parse_candidates_from_plain_text_cell(
                    candidates_cell, district, state_name, incumbent_name, pvi, 'general'
                )
                candidates.extend(plain_text_candidates)

    return candidates


def parse_state_table_2012(table, state_name: str = None) -> List[Dict]:
    """
    Parse state table specifically for 2012 elections
    2012 has structure with 7, 5, or 4 cells per row:
    - 7 cells: Normal district (District, PVI, Incumbent, Party, First elected, Status, Candidates)
    - 5 cells: New seat from redistricting (District, PVI, Incumbent, Status, Candidates)
    - 4 cells: Continuation row from rowspan (SKIP these)
    Excludes non-voting delegate seats from territories
    """
    candidates = []
    all_rows = table.find_all('tr')
    data_rows = [row for row in all_rows if row.find('td')]

    for row in data_rows:
        cells = row.find_all(['td', 'th'])

        # Skip continuation rows (4 cells)
        if len(cells) == 4:
            continue

        # Need at least 5 cells for valid data
        if len(cells) < 5:
            continue

        # Extract based on cell count
        if len(cells) == 7:
            # Normal district
            district = cells[0].get_text(strip=True)
            pvi = cells[1].get_text(strip=True)
            incumbent_name = cells[2].get_text(strip=True)
            candidates_cell = cells[6]
        elif len(cells) == 5:
            # New seat from redistricting
            district = cells[0].get_text(strip=True)
            pvi = cells[1].get_text(strip=True)
            incumbent_name = cells[2].get_text(strip=True)
            candidates_cell = cells[4]
        else:
            continue

        # Skip non-voting delegate seats (territories)
        # Normalize district name (replace non-breaking spaces with regular spaces)
        district_normalized = district.replace('\xa0', ' ')
        non_voting_territories = [
            'American Samoa at-large',
            'District of Columbia at-large',
            'Guam at-large',
            'Northern Mariana Islands at-large',
            'Puerto Rico at-large',
            'United States Virgin Islands at-large',
            'U.S. Virgin Islands at-large'
        ]
        if district_normalized in non_voting_territories:
            continue

        # Clean PVI (remove footnotes)
        pvi = re.sub(r'\[.*?\]', '', pvi).strip()

        # Get state from district if not provided
        if not state_name:
            state_name = extract_state_from_district(district)

        # Parse candidates from candidates_cell
        list_items = candidates_cell.find_all('li')

        if list_items:
            # Candidates in <li> elements (contested races)
            for li in list_items:
                candidate_data = parse_candidate_from_li(
                    li, district, state_name, incumbent_name, pvi,
                    source_table='general', context_lis=list_items
                )
                if candidate_data:
                    candidates.append(candidate_data)
        else:
            # No <li> elements - check for plain text (uncontested 100% races)
            text = candidates_cell.get_text(strip=True)
            if text and '%' in text:
                plain_text_candidates = parse_candidates_from_plain_text_cell(
                    candidates_cell, district, state_name, incumbent_name, pvi, 'general'
                )
                candidates.extend(plain_text_candidates)

    return candidates


def parse_state_table_2014(table, state_name: str = None) -> List[Dict]:
    """
    Parse state table specifically for 2014 elections
    2014 has structure with 7 or 5 cells per row:
    - 7 cells: Normal district (District, PVI, Incumbent, Party, First elected, Status, Candidates)
    - 5 cells: Vacant district (District, PVI, Incumbent, Status, Candidates)
    Excludes non-voting delegate seats from territories
    """
    candidates = []
    all_rows = table.find_all('tr')
    data_rows = [row for row in all_rows if row.find('td')]

    for row in data_rows:
        cells = row.find_all(['td', 'th'])

        # Need at least 5 cells for valid data
        if len(cells) < 5:
            continue

        # Extract based on cell count
        if len(cells) == 7:
            # Normal district
            district = cells[0].get_text(strip=True)
            pvi = cells[1].get_text(strip=True)
            incumbent_name = cells[2].get_text(strip=True)
            candidates_cell = cells[6]
        elif len(cells) == 5:
            # Vacant district
            district = cells[0].get_text(strip=True)
            pvi = cells[1].get_text(strip=True)
            incumbent_name = cells[2].get_text(strip=True)
            candidates_cell = cells[4]
        else:
            continue

        # Skip non-voting delegate seats (territories)
        # Normalize district name (replace non-breaking spaces with regular spaces)
        district_normalized = district.replace('\xa0', ' ')
        non_voting_territories = [
            'American Samoa at-large',
            'District of Columbia at-large',
            'Guam at-large',
            'Northern Mariana Islands at-large',
            'Puerto Rico at-large',
            'United States Virgin Islands at-large',
            'U.S. Virgin Islands at-large'
        ]
        if district_normalized in non_voting_territories:
            continue

        # Clean PVI (remove footnotes)
        pvi = re.sub(r'\[.*?\]', '', pvi).strip()

        # Get state from district if not provided
        if not state_name:
            state_name = extract_state_from_district(district)

        # Parse candidates from candidates_cell
        list_items = candidates_cell.find_all('li')

        if list_items:
            # Candidates in <li> elements (contested races)
            for li in list_items:
                candidate_data = parse_candidate_from_li(
                    li, district, state_name, incumbent_name, pvi,
                    source_table='general', context_lis=list_items
                )
                if candidate_data:
                    candidates.append(candidate_data)
        else:
            # No <li> elements - check for plain text (uncontested 100% races)
            text = candidates_cell.get_text(strip=True)
            if text and '%' in text:
                plain_text_candidates = parse_candidates_from_plain_text_cell(
                    candidates_cell, district, state_name, incumbent_name, pvi, 'general'
                )
                candidates.extend(plain_text_candidates)

    return candidates


def parse_state_table_2016(table, state_name: str = None) -> List[Dict]:
    """
    Parse state table specifically for 2016 elections
    2016 has structure with 7, 5, or 4 cells per row:
    - 7 cells: Normal district (District, PVI, Incumbent, Party, First elected, Status, Candidates)
    - 5 cells: New/redistricted/vacant district (District, PVI, Incumbent, Status, Candidates)
    - 4 cells: Continuation row from rowspan (SKIP these)
    """
    candidates = []
    all_rows = table.find_all('tr')
    data_rows = [row for row in all_rows if row.find('td')]

    for row in data_rows:
        cells = row.find_all(['td', 'th'])

        # Skip continuation rows (4 cells)
        if len(cells) == 4:
            continue

        # Need at least 5 cells for valid data
        if len(cells) < 5:
            continue

        # Extract based on cell count
        if len(cells) == 7:
            # Normal district
            district = cells[0].get_text(strip=True)
            pvi = cells[1].get_text(strip=True)
            incumbent_name = cells[2].get_text(strip=True)
            candidates_cell = cells[6]
        elif len(cells) == 5:
            # New/redistricted/vacant district
            district = cells[0].get_text(strip=True)
            pvi = cells[1].get_text(strip=True)
            incumbent_name = cells[2].get_text(strip=True)
            candidates_cell = cells[4]
        else:
            continue

        # Clean PVI (remove footnotes)
        pvi = re.sub(r'\[.*?\]', '', pvi).strip()

        # Get state from district if not provided
        if not state_name:
            state_name = extract_state_from_district(district)

        # Parse candidates from candidates_cell
        list_items = candidates_cell.find_all('li')

        if list_items:
            # Candidates in <li> elements (contested races)
            for li in list_items:
                candidate_data = parse_candidate_from_li(
                    li, district, state_name, incumbent_name, pvi,
                    source_table='general', context_lis=list_items
                )
                if candidate_data:
                    candidates.append(candidate_data)
        else:
            # No <li> elements - check for plain text (uncontested 100% races)
            text = candidates_cell.get_text(strip=True)
            if text and '%' in text:
                plain_text_candidates = parse_candidates_from_plain_text_cell(
                    candidates_cell, district, state_name, incumbent_name, pvi, 'general'
                )
                candidates.extend(plain_text_candidates)

    return candidates


def parse_state_table_2018(table, state_name: str = None) -> List[Dict]:
    """
    Parse state table specifically for 2018 elections
    2018 has structure with 7 or 5 cells per row:
    - 7 cells: Normal district (District, PVI, Incumbent, Party, First elected, Status, Candidates)
    - 5 cells: New/redistricted district (District, PVI, Incumbent, Status, Candidates)
    """
    candidates = []
    all_rows = table.find_all('tr')
    data_rows = [row for row in all_rows if row.find('td')]

    for row in data_rows:
        cells = row.find_all(['td', 'th'])

        # Need at least 5 cells for valid data
        if len(cells) < 5:
            continue

        # Extract based on cell count
        if len(cells) == 7:
            # Normal district
            district = cells[0].get_text(strip=True)
            pvi = cells[1].get_text(strip=True)
            incumbent_name = cells[2].get_text(strip=True)
            candidates_cell = cells[6]
        elif len(cells) == 5:
            # New/redistricted district
            district = cells[0].get_text(strip=True)
            pvi = cells[1].get_text(strip=True)
            incumbent_name = cells[2].get_text(strip=True)
            candidates_cell = cells[4]
        else:
            continue

        # Clean PVI (remove footnotes)
        pvi = re.sub(r'\[.*?\]', '', pvi).strip()

        # Get state from district if not provided
        if not state_name:
            state_name = extract_state_from_district(district)

        # Parse candidates from candidates_cell
        list_items = candidates_cell.find_all('li')

        if list_items:
            # Candidates in <li> elements (contested races)
            for li in list_items:
                candidate_data = parse_candidate_from_li(
                    li, district, state_name, incumbent_name, pvi,
                    source_table='general', context_lis=list_items
                )
                if candidate_data:
                    candidates.append(candidate_data)
        else:
            # No <li> elements - check for plain text (uncontested 100% races)
            text = candidates_cell.get_text(strip=True)
            if text and '%' in text:
                plain_text_candidates = parse_candidates_from_plain_text_cell(
                    candidates_cell, district, state_name, incumbent_name, pvi, 'general'
                )
                candidates.extend(plain_text_candidates)

    return candidates


def parse_state_table_2020(table, state_name: str = None) -> List[Dict]:
    """
    Parse state table specifically for 2020 elections
    2020 has structure with 7 or 5 cells per row:
    - 7 cells: Normal district (District, PVI, Incumbent, Party, First elected, Status, Candidates)
    - 5 cells: New/redistricted district (District, PVI, Incumbent, Status, Candidates)
    """
    candidates = []
    all_rows = table.find_all('tr')
    data_rows = [row for row in all_rows if row.find('td')]

    for row in data_rows:
        cells = row.find_all(['td', 'th'])

        # Need at least 5 cells for valid data
        if len(cells) < 5:
            continue

        # Extract based on cell count
        if len(cells) == 7:
            # Normal district
            district = cells[0].get_text(strip=True)
            pvi = cells[1].get_text(strip=True)
            incumbent_name = cells[2].get_text(strip=True)
            candidates_cell = cells[6]
        elif len(cells) == 5:
            # New/redistricted district
            district = cells[0].get_text(strip=True)
            pvi = cells[1].get_text(strip=True)
            incumbent_name = cells[2].get_text(strip=True)
            candidates_cell = cells[4]
        else:
            continue

        # Clean PVI (remove footnotes)
        pvi = re.sub(r'\[.*?\]', '', pvi).strip()

        # Get state from district if not provided
        if not state_name:
            state_name = extract_state_from_district(district)

        # Parse candidates from candidates_cell
        list_items = candidates_cell.find_all('li')

        if list_items:
            # Candidates in <li> elements (contested races)
            for li in list_items:
                candidate_data = parse_candidate_from_li(
                    li, district, state_name, incumbent_name, pvi,
                    source_table='general', context_lis=list_items
                )
                if candidate_data:
                    candidates.append(candidate_data)
        else:
            # No <li> elements - check for plain text (uncontested 100% races)
            text = candidates_cell.get_text(strip=True)
            if text and '%' in text:
                plain_text_candidates = parse_candidates_from_plain_text_cell(
                    candidates_cell, district, state_name, incumbent_name, pvi, 'general'
                )
                candidates.extend(plain_text_candidates)

    return candidates


def parse_state_table_2024(table, state_name: str = None) -> List[Dict]:
    """
    Parse state table specifically for 2024 elections
    2024 has unique structure with rowspan for redistricting:
    - 7 cells: Normal district (District, PVI, Incumbent, Party, First elected, Status, Candidates)
    - 5 cells: New district (District, PVI, "None (new district)", Status, Candidates)
    - 4 cells: Continuation row from rowspan (SKIP - shows new incumbent info)
    """
    candidates = []
    all_rows = table.find_all('tr')
    data_rows = [row for row in all_rows if row.find('td')]

    for row in data_rows:
        cells = row.find_all(['td', 'th'])

        # Skip continuation rows (rowspan second row)
        if len(cells) == 4:
            continue

        # Need at least 5 cells for valid data
        if len(cells) < 5:
            continue

        # Extract based on cell count
        if len(cells) == 7:
            # Normal district: Cell[0]=District, Cell[1]=PVI, Cell[2]=Incumbent, Cell[6]=Candidates
            district = cells[0].get_text(strip=True)
            pvi = cells[1].get_text(strip=True)
            incumbent_name = cells[2].get_text(strip=True)
            candidates_cell = cells[6]
        elif len(cells) == 5:
            # New district: Cell[0]=District, Cell[1]=PVI, Cell[2]=Incumbent, Cell[4]=Candidates
            district = cells[0].get_text(strip=True)
            pvi = cells[1].get_text(strip=True)
            incumbent_name = cells[2].get_text(strip=True)
            candidates_cell = cells[4]
        else:
            # Unexpected format, skip
            continue

        # Clean PVI (remove footnotes)
        pvi = re.sub(r'\[.*?\]', '', pvi).strip()

        # Get state from district if not provided
        if not state_name:
            state_name = extract_state_from_district(district)

        # Parse candidates from candidates_cell
        list_items = candidates_cell.find_all('li')

        if list_items:
            # Candidates in <li> elements (contested races)
            for li in list_items:
                candidate_data = parse_candidate_from_li(
                    li, district, state_name, incumbent_name, pvi,
                    source_table='general', context_lis=list_items
                )
                if candidate_data:
                    candidates.append(candidate_data)
        else:
            # No <li> elements - check for plain text (uncontested 100% races)
            text = candidates_cell.get_text(strip=True)
            if text and '%' in text:
                # Parse plain text format: "▌YName(Party) 100%"
                plain_text_candidates = parse_candidates_from_plain_text(
                    text, district, state_name, incumbent_name, pvi, 'general'
                )
                candidates.extend(plain_text_candidates)

    return candidates

# fallback parser for state tables that don't have a dedicated parser (2008-2016)
def parse_state_table(table, state_name: str = None) -> List[Dict]:
    """
    Parse a state table to extract all candidates
    State tables have structure (varies by year):
    - With CPVI (2010-2024 most states, 2008 some states):
      Cell[0]: District, Cell[1]: CPVI, Cell[2]: Incumbent, Cell[3]: Party,
      Cell[4]: First elected, Cell[5]: Results, Cell[6]: Candidates
    - Without CPVI (2008 some states):
      Cell[0]: District, Cell[1]: Incumbent, Cell[2]: Party,
      Cell[3]: First elected, Cell[4]: Results, Cell[5]: Candidates
    """
    candidates = []
    all_rows = table.find_all('tr')

    # Find column indices from headers
    headers = [th.get_text(strip=True) for th in table.find_all('th')]
    candidates_col_idx = None
    incumbent_col_idx = None
    pvi_col_idx = None

    for idx, header in enumerate(headers):
        if 'Candidates' in header:
            candidates_col_idx = idx
        elif 'Incumbent' in header:
            incumbent_col_idx = idx
        elif 'CPVI' in header or 'PVI' in header:
            pvi_col_idx = idx

    # If we couldn't find candidates column in headers, use defaults
    if candidates_col_idx is None:
        # Try to infer based on cell count - last cell is usually candidates
        candidates_col_idx = -1  # Will use last cell

    # Determine incumbent and PVI indices based on structure
    if pvi_col_idx is not None:
        # Has CPVI column (structure: District, CPVI, Incumbent, ...)
        if incumbent_col_idx is None:
            incumbent_col_idx = 2
    else:
        # No CPVI column (structure: District, Incumbent, ...)
        if incumbent_col_idx is None:
            incumbent_col_idx = 1

    # Skip header rows (rows with only th elements, no td elements)
    data_rows = [row for row in all_rows if row.find('td')]

    for row in data_rows:
        cells = row.find_all(['td', 'th'])

        # Need at least 6 cells for a valid data row (minimum without CPVI)
        if len(cells) < 6:
            continue

        # Extract row information
        district = cells[0].get_text(strip=True)
        pvi = cells[pvi_col_idx].get_text(strip=True) if pvi_col_idx is not None and len(cells) > pvi_col_idx else ''
        incumbent_name = cells[incumbent_col_idx].get_text(strip=True) if len(cells) > incumbent_col_idx else ''

        # Get state from district if not provided
        if not state_name:
            state_name = extract_state_from_district(district)

        # Parse candidates cell (always the last cell in all years)
        candidates_cell = cells[-1]

        list_items = candidates_cell.find_all('li')

        if list_items:
            # New format with <li> elements (2008+)
            for li in list_items:
                candidate_data = parse_candidate_from_li(
                    li, district, state_name, incumbent_name, pvi,
                    source_table='general', context_lis=list_items
                )
                if candidate_data:
                    candidates.append(candidate_data)
        else:
            # Old format with plain text (2016)
            text = candidates_cell.get_text(strip=True)
            if text:
                plain_text_candidates = parse_candidates_from_plain_text_cell(
                    candidates_cell, district, state_name, incumbent_name, pvi, 'general'
                )
                candidates.extend(plain_text_candidates)

    return candidates


def parse_special_elections_table(table) -> List[Dict]:
    """
    Parse the special elections table (Table 3)
    This has a different structure from state tables
    """
    candidates = []
    all_rows = table.find_all('tr')

    # Skip header rows (rows with only th elements, no td elements)
    data_rows = [row for row in all_rows if row.find('td')]

    for row in data_rows:
        cells = row.find_all(['td', 'th'])

        # Need at least 5 cells
        if len(cells) < 5:
            continue

        # Extract district from first cell
        district_cell = cells[0]
        district = district_cell.get_text(strip=True)
        state_name = extract_state_from_district(district)

        # Get incumbent info
        incumbent_name = cells[1].get_text(strip=True) if len(cells) > 1 else ''

        # Candidates should be in the last cell
        candidates_cell = cells[-1]
        list_items = candidates_cell.find_all('li')

        # PVI might not be available in special elections
        pvi = ''

        if list_items:
            # New format with <li> elements (2018+)
            for li in list_items:
                candidate_data = parse_candidate_from_li(
                    li, district, state_name, incumbent_name, pvi,
                    source_table='special', context_lis=list_items
                )
                if candidate_data:
                    candidates.append(candidate_data)
        else:
            # Old format with plain text (2016)
            text = candidates_cell.get_text(strip=True)
            if text:
                plain_text_candidates = parse_candidates_from_plain_text_cell(
                    candidates_cell, district, state_name, incumbent_name, pvi, 'special'
                )
                candidates.extend(plain_text_candidates)

    return candidates

def scrape_house_elections(
    url: str,
    year: int,
    special_table_index: int,
    state_tables_start_index: int,
    output_path: str
) -> pd.DataFrame:
    """
    Main scraping function for US House elections

    Args:
        url: Wikipedia URL to scrape
        year: Election year (e.g., 2022, 2024)
        special_table_index: Table index for special elections
        state_tables_start_index: Starting index for state tables
        output_path: Path to save CSV output

    Returns:
        DataFrame with all candidates
    """
    print("="*80)
    print(f"{year} US HOUSE ELECTIONS - CANDIDATE SCRAPER")
    print("="*80)

    print("\n1. Fetching Wikipedia page...")
    soup = fetch_wikipedia_page(url)
    print("   Page fetched successfully")

    # Get all tables
    tables = soup.find_all('table', class_='wikitable')
    print(f"\n2. Found {len(tables)} total tables on page")

    all_candidates = []

    # Parse special elections
    print(f"\n3. Parsing Special Elections table (Table {special_table_index})...")
    if len(tables) > special_table_index:
        special_candidates = parse_special_elections_table(tables[special_table_index])
        all_candidates.extend(special_candidates)
        print(f"   Extracted {len(special_candidates)} candidates from special elections")

    # Identify and parse all state tables
    print(f"\n4. Identifying state tables (starting at index {state_tables_start_index})...")
    state_tables = identify_state_tables(soup, start_index=state_tables_start_index)
    print(f"   Found {len(state_tables)} state tables")

    print("\n5. Parsing state tables...")
    for idx, (table, state_name) in enumerate(state_tables, 1):
        # Use year-specific parsers
        if year == 2024:
            state_candidates = parse_state_table_2024(table, state_name)
        elif year == 2022:
            state_candidates = parse_state_table_2022(table, state_name)
        elif year == 2020:
            state_candidates = parse_state_table_2020(table, state_name)
        elif year == 2018:
            state_candidates = parse_state_table_2018(table, state_name)
        elif year == 2016:
            state_candidates = parse_state_table_2016(table, state_name)
        elif year == 2014:
            state_candidates = parse_state_table_2014(table, state_name)
        elif year == 2012:
            state_candidates = parse_state_table_2012(table, state_name)
        elif year == 2010:
            state_candidates = parse_state_table_2010(table, state_name)
        elif year == 2008:
            state_candidates = parse_state_table_2008(table, state_name)
        else:
            state_candidates = parse_state_table(table, state_name)
        all_candidates.extend(state_candidates)

        if state_name:
            print(f"   [{idx}/{len(state_tables)}] {state_name:20s} - {len(state_candidates):3d} candidates")
        else:
            print(f"   [{idx}/{len(state_tables)}] Unknown State      - {len(state_candidates):3d} candidates")

    # Create DataFrame
    print("\n6. Creating DataFrame...")
    df = pd.DataFrame(all_candidates)

    # Reorder columns
    column_order = [
        'state', 'district', 'candidate_name', 'party',
        'vote_percentage', 'is_incumbent', 'is_winner', 'pvi', 'election_type', 'wikipedia_url'
    ]
    df = df[column_order]

    # Sort by state, district, election_type, vote percentage
    df = df.sort_values(['state', 'district', 'election_type', 'vote_percentage'],
                        ascending=[True, True, True, False])

    print(f"   Created DataFrame with {len(df)} total candidates")
    print(f"\n   Summary:")
    print(f"   - States: {df['state'].nunique()}")
    print(f"   - Districts: {df['district'].nunique()}")
    print(f"   - Candidates with Wikipedia pages: {df['wikipedia_url'].notna().sum()}")
    print(f"   - Incumbents: {df['is_incumbent'].sum()}")

    print(f"\n   Election Type Breakdown:")
    print(df['election_type'].value_counts().to_string())

    # Display sample
    print("\n7. Sample of data (first 15 rows):")
    print(df.head(15).to_string(index=False))

    # Save to CSV
    print(f"\n8. Saving to CSV: {output_path}")
    df.to_csv(output_path, index=False)
    print("   Saved successfully.")

    print("\n" + "="*80)
    print("SCRAPING COMPLETE")
    print("="*80)

    return df


def identify_state_tables(soup: BeautifulSoup, start_index: int = 5) -> List[tuple]:
    """
    Identify all state tables in the page
    Returns list of (table, state_name) tuples

    Args:
        start_index: Table index to start searching from (5 for 2022, 7 for 2024)
    """
    tables = soup.find_all('table', class_='wikitable')
    state_tables = []

    # State tables start at start_index and have 'Candidates' column
    for idx, table in enumerate(tables[start_index:], start=start_index):
        headers = [th.get_text(strip=True) for th in table.find_all('th')]

        # Check if this is a state table
        has_candidates = any('Candidates' in h for h in headers)
        has_district = any('District' in h for h in headers)

        if has_candidates and has_district:
            # Try to identify state name from headers
            # State headers often appear as "Alabama 1", "Alabama 2", etc.
            state_name = None
            for h in headers:
                # Look for state names with district numbers
                match = re.match(r'([A-Za-z\s]+)\s*\d+', h)
                if match:
                    state_name = match.group(1).strip()
                    break
                # Also check for "at-large" pattern
                match = re.match(r'([A-Za-z\s]+)\s*at-large', h, re.IGNORECASE)
                if match:
                    state_name = match.group(1).strip()
                    break

            state_tables.append((table, state_name))

    return state_tables


# Configuration for different election years
YEAR_CONFIGS = {
    2008: {
        'url': 'https://en.wikipedia.org/wiki/2008_United_States_House_of_Representatives_elections',
        'special_table_index': 3,
        'state_tables_start_index': 4,
    },
    2010: {
        'url': 'https://en.wikipedia.org/wiki/2010_United_States_House_of_Representatives_elections',
        'special_table_index': 3,
        'state_tables_start_index': 4,
    },
    2012: {
        'url': 'https://en.wikipedia.org/wiki/2012_United_States_House_of_Representatives_elections',
        'special_table_index': 3,
        'state_tables_start_index': 4,
    },
    2014: {
        'url': 'https://en.wikipedia.org/wiki/2014_United_States_House_of_Representatives_elections',
        'special_table_index': 4,
        'state_tables_start_index': 5,
    },
    2016: {
        'url': 'https://en.wikipedia.org/wiki/2016_United_States_House_of_Representatives_elections',
        'special_table_index': 4,
        'state_tables_start_index': 5,
    },
    2018: {
        'url': 'https://en.wikipedia.org/wiki/2018_United_States_House_of_Representatives_elections',
        'special_table_index': 3,
        'state_tables_start_index': 6,
    },
    2020: {
        'url': 'https://en.wikipedia.org/wiki/2020_United_States_House_of_Representatives_elections',
        'special_table_index': 5,
        'state_tables_start_index': 7,
    },
    2022: {
        'url': 'https://en.wikipedia.org/wiki/2022_United_States_House_of_Representatives_elections',
        'special_table_index': 3,
        'state_tables_start_index': 5,
    },
    2024: {
        'url': 'https://en.wikipedia.org/wiki/2024_United_States_House_of_Representatives_elections',
        'special_table_index': 6,
        'state_tables_start_index': 7,
    },
}


def scrape_year(year: int, output_dir: str = '/Users/haiqili/Desktop/ThesisData/data/raw/html_parsed_candidates'):
    """
    Scrape House elections for a specific year

    Args:
        year: Election year to scrape
        output_dir: Directory to save CSV output

    Returns:
        DataFrame with candidates
    """
    if year not in YEAR_CONFIGS:
        raise ValueError(f"Year {year} not configured. Available years: {list(YEAR_CONFIGS.keys())}")

    config = YEAR_CONFIGS[year]
    output_path = f"{output_dir}/{year}_house_candidates.csv"

    return scrape_house_elections(
        url=config['url'],
        year=year,
        special_table_index=config['special_table_index'],
        state_tables_start_index=config['state_tables_start_index'],
        output_path=output_path
    )


def main():
    """Main execution function - scrapes all configured years or specific year from command line"""
    import sys

    if len(sys.argv) > 1:
        # Scrape specific year from command line
        try:
            year = int(sys.argv[1])
            print(f"\nScraping year: {year}")
            scrape_year(year)
        except ValueError as e:
            print(f"Error: {e}")
            print(f"Usage: python {sys.argv[0]} [year]")
            print(f"Available years: {list(YEAR_CONFIGS.keys())}")
            sys.exit(1)
    else:
        # Scrape all configured years
        print(f"\nScraping all configured years: {list(YEAR_CONFIGS.keys())}\n")
        for year in sorted(YEAR_CONFIGS.keys()):
            print(f"\n{'='*80}")
            print(f"Starting scrape for {year}")
            print('='*80)
            scrape_year(year)
            print("\n")


if __name__ == "__main__":
    main()
