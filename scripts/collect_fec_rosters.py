"""
FEC Candidate Roster Collection

Downloads and parses FEC Candidate Master files for federal elections.
Creates structured roster data with:
- Candidate ID, name, party
- Office (President/Senate/House)
- State, district (for House)
- Election cycle

Data source: https://www.fec.gov/data/browse-data/?tab=bulk-data
"""
import os
import sys
import re
import csv
import zipfile
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging
from io import BytesIO, StringIO
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# FEC DATA URLS AND SCHEMAS
# =============================================================================

# FEC Bulk Data URLs for Candidate Master files
# Format: https://www.fec.gov/files/bulk-downloads/{YEAR}/cn{YY}.zip
def get_fec_candidate_url(cycle_year: int) -> str:
    """Get URL for FEC candidate master file for a given cycle."""
    yy = str(cycle_year)[-2:]
    return f"https://www.fec.gov/files/bulk-downloads/{cycle_year}/cn{yy}.zip"


# FEC Candidate Master File Schema (pipe-delimited)
# Reference: https://www.fec.gov/campaign-finance-data/candidate-master-file-description/
FEC_CANDIDATE_COLUMNS = [
    'candidate_id',        # CAND_ID
    'candidate_name',      # CAND_NAME (Last, First Middle Suffix)
    'party_affiliation',   # CAND_PTY_AFFILIATION
    'election_year',       # CAND_ELECTION_YR
    'office_state',        # CAND_OFFICE_ST
    'office',              # CAND_OFFICE (H/S/P)
    'office_district',     # CAND_OFFICE_DISTRICT
    'incumbent_challenger_status',  # CAND_ICI (I/C/O)
    'candidate_status',    # CAND_STATUS (C/N/P/R/W)
    'principal_committee_id',  # CAND_PCC
    'street_1',            # CAND_ST1
    'street_2',            # CAND_ST2
    'city',                # CAND_CITY
    'state',               # CAND_ST
    'zip',                 # CAND_ZIP
]


# =============================================================================
# NAME PARSING AND NORMALIZATION
# =============================================================================

def parse_fec_name(fec_name: str) -> Dict[str, str]:
    """
    Parse FEC-style name (LAST, FIRST MIDDLE SUFFIX) into components.
    
    Examples:
        "BIDEN, JOSEPH R JR" -> {'last': 'Biden', 'first': 'Joseph', 'middle': 'R', 'suffix': 'Jr'}
        "OCASIO-CORTEZ, ALEXANDRIA" -> {'last': 'Ocasio-Cortez', 'first': 'Alexandria', ...}
        
    Returns dictionary with: last, first, middle, suffix, full_name
    """
    if not fec_name:
        return {'last': '', 'first': '', 'middle': '', 'suffix': '', 'full_name': ''}
    
    # Handle comma separation
    parts = fec_name.split(',', 1)
    
    if len(parts) == 1:
        # No comma - try to parse as "First Last"
        words = parts[0].strip().split()
        if len(words) >= 2:
            last = words[-1].title()
            first = words[0].title()
            middle = ' '.join(words[1:-1]).title() if len(words) > 2 else ''
            suffix = ''
        else:
            last = parts[0].strip().title()
            first = ''
            middle = ''
            suffix = ''
    else:
        last = parts[0].strip().title()
        rest = parts[1].strip().split()
        
        # Known suffixes
        suffixes = {'JR', 'JR.', 'SR', 'SR.', 'II', 'III', 'IV', 'V', 'MD', 'PHD', 'ESQ'}
        
        first = ''
        middle = ''
        suffix = ''
        
        remaining = []
        for word in rest:
            if word.upper() in suffixes:
                suffix = word.title().replace('.', '')
            else:
                remaining.append(word)
        
        if remaining:
            first = remaining[0].title()
            middle = ' '.join(remaining[1:]).title() if len(remaining) > 1 else ''
    
    # Build full name
    full_parts = [first, middle, last]
    if suffix:
        full_parts.append(suffix)
    full_name = ' '.join(p for p in full_parts if p)
    
    return {
        'last': last,
        'first': first,
        'middle': middle,
        'suffix': suffix,
        'full_name': full_name
    }


def normalize_name_for_search(name_parts: Dict[str, str]) -> List[str]:
    """
    Generate search variants for Wikipedia matching.
    
    Returns list of search strings to try, in order of preference.
    """
    variants = []
    
    first = name_parts['first']
    last = name_parts['last']
    middle = name_parts['middle']
    suffix = name_parts['suffix']
    
    # Primary: "First Last" (most common Wikipedia format)
    if first and last:
        variants.append(f"{first} {last}")
    
    # With suffix
    if first and last and suffix:
        variants.append(f"{first} {last} {suffix}")
        variants.append(f"{first} {last}, {suffix}")
    
    # With middle initial
    if first and last and middle:
        middle_initial = middle[0] if len(middle) == 1 else middle.split()[0][0] if middle.split() else ''
        if middle_initial:
            variants.append(f"{first} {middle_initial}. {last}")
    
    # Full name with middle
    if first and middle and last:
        variants.append(f"{first} {middle} {last}")
    
    return variants


# =============================================================================
# FEC DATA DOWNLOAD AND PARSING
# =============================================================================

class FECRosterCollector:
    """Collect and parse FEC candidate roster data."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize collector.
        
        Args:
            cache_dir: Directory to cache downloaded files (optional)
        """
        self.cache_dir = cache_dir or Path(__file__).parent.parent / 'data' / 'cache' / 'fec'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WikiElectionResearch/1.0 (Educational Research)'
        })
    
    def download_candidate_file(self, cycle_year: int, force: bool = False) -> Path:
        """
        Download FEC candidate master file for a cycle.
        
        Args:
            cycle_year: Election cycle year (e.g., 2024)
            force: Force re-download even if cached
            
        Returns:
            Path to the downloaded/cached file
        """
        cache_file = self.cache_dir / f"cn{str(cycle_year)[-2:]}.txt"
        
        if cache_file.exists() and not force:
            logger.info(f"Using cached FEC data for {cycle_year}")
            return cache_file
        
        url = get_fec_candidate_url(cycle_year)
        logger.info(f"Downloading FEC candidate data: {url}")
        
        try:
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            
            # Extract from zip
            with zipfile.ZipFile(BytesIO(response.content)) as zf:
                # Find the .txt file in the zip
                txt_files = [n for n in zf.namelist() if n.endswith('.txt')]
                if not txt_files:
                    raise ValueError(f"No .txt file found in FEC zip for {cycle_year}")
                
                # Extract the first .txt file
                with zf.open(txt_files[0]) as f:
                    content = f.read()
                    with open(cache_file, 'wb') as out:
                        out.write(content)
            
            logger.info(f"Downloaded and cached FEC data for {cycle_year}")
            return cache_file
            
        except Exception as e:
            logger.error(f"Failed to download FEC data for {cycle_year}: {e}")
            raise
    
    def parse_candidate_file(self, filepath: Path) -> List[Dict]:
        """
        Parse FEC candidate master file.
        
        Args:
            filepath: Path to the .txt file
            
        Returns:
            List of candidate dictionaries
        """
        candidates = []
        
        with open(filepath, 'r', encoding='latin-1') as f:
            for line in f:
                # Pipe-delimited
                fields = line.strip().split('|')
                
                if len(fields) < len(FEC_CANDIDATE_COLUMNS):
                    # Pad with empty strings
                    fields.extend([''] * (len(FEC_CANDIDATE_COLUMNS) - len(fields)))
                
                record = dict(zip(FEC_CANDIDATE_COLUMNS, fields))
                
                # Parse name
                name_parts = parse_fec_name(record['candidate_name'])
                record['name_last'] = name_parts['last']
                record['name_first'] = name_parts['first']
                record['name_middle'] = name_parts['middle']
                record['name_suffix'] = name_parts['suffix']
                record['name_full'] = name_parts['full_name']
                record['search_variants'] = normalize_name_for_search(name_parts)
                
                # Normalize office code
                office = record['office'].upper() if record['office'] else ''
                record['office_normalized'] = {
                    'H': 'House',
                    'S': 'Senate',
                    'P': 'President'
                }.get(office, office)
                
                # Normalize party
                party = record['party_affiliation'].upper() if record['party_affiliation'] else ''
                record['party_normalized'] = {
                    'DEM': 'Democratic',
                    'REP': 'Republican',
                    'LIB': 'Libertarian',
                    'GRE': 'Green',
                    'IND': 'Independent',
                    'CON': 'Constitution',
                    'NPA': 'No Party Affiliation',
                    'UNK': 'Unknown'
                }.get(party, party)
                
                candidates.append(record)
        
        return candidates
    
    def collect_cycle(self, cycle_year: int) -> List[Dict]:
        """
        Collect and parse candidates for a single cycle.
        
        Args:
            cycle_year: Election cycle (e.g., 2024)
            
        Returns:
            List of candidate records
        """
        filepath = self.download_candidate_file(cycle_year)
        return self.parse_candidate_file(filepath)
    
    def collect_all_cycles(self, cycles: List[int] = None) -> Dict[int, List[Dict]]:
        """
        Collect candidates for multiple cycles.
        
        Args:
            cycles: List of cycle years (default: 2016, 2018, 2020, 2022, 2024)
            
        Returns:
            Dictionary mapping cycle year to candidate list
        """
        if cycles is None:
            cycles = [2016, 2018, 2020, 2022, 2024]
        
        all_candidates = {}
        
        for year in cycles:
            logger.info(f"\nCollecting FEC data for cycle {year}...")
            try:
                candidates = self.collect_cycle(year)
                all_candidates[year] = candidates
                logger.info(f"  Found {len(candidates)} candidates")
            except Exception as e:
                logger.error(f"  Failed: {e}")
                all_candidates[year] = []
        
        return all_candidates
    
    def filter_general_election_candidates(
        self,
        candidates: List[Dict],
        office: Optional[str] = None,  # 'H', 'S', 'P', or None for all
        major_parties_only: bool = False
    ) -> List[Dict]:
        """
        Filter candidates for general election roster.
        
        Args:
            candidates: List of candidate records
            office: Filter to specific office (H/S/P) or None for all
            major_parties_only: Only include Democrats and Republicans
            
        Returns:
            Filtered list of candidates
        """
        filtered = []
        
        major_parties = {'DEM', 'REP'}
        
        for c in candidates:
            # Filter by office if specified
            if office and c['office'].upper() != office.upper():
                continue
            
            # Filter by party if specified
            if major_parties_only and c['party_affiliation'].upper() not in major_parties:
                continue
            
            # Filter out withdrawn candidates (status W)
            # Keep: C (statutory candidate), N (not yet candidate), P (statutory candidate pending)
            status = c['candidate_status'].upper() if c['candidate_status'] else ''
            if status == 'W':
                continue
            
            filtered.append(c)
        
        return filtered


# =============================================================================
# ROSTER BUILDING
# =============================================================================

def build_federal_roster(
    cycles: List[int] = None,
    include_president: bool = True,
    include_senate: bool = True,
    include_house: bool = True,
    major_parties_only: bool = False
) -> Dict:
    """
    Build comprehensive federal candidate roster from FEC data.
    
    Args:
        cycles: List of cycle years (default: 2016-2024)
        include_president: Include presidential candidates
        include_senate: Include Senate candidates
        include_house: Include House candidates
        major_parties_only: Only major party candidates
        
    Returns:
        Dictionary with roster data and metadata
    """
    if cycles is None:
        cycles = [2016, 2018, 2020, 2022, 2024]
    
    collector = FECRosterCollector()
    all_candidates = collector.collect_all_cycles(cycles)
    
    # Build unified roster
    roster = []
    seen_candidates = set()  # Track by (candidate_id, cycle)
    
    for cycle_year, candidates in all_candidates.items():
        offices_to_include = []
        if include_president:
            offices_to_include.append('P')
        if include_senate:
            offices_to_include.append('S')
        if include_house:
            offices_to_include.append('H')
        
        for office in offices_to_include:
            filtered = collector.filter_general_election_candidates(
                candidates, 
                office=office,
                major_parties_only=major_parties_only
            )
            
            for c in filtered:
                key = (c['candidate_id'], cycle_year)
                if key in seen_candidates:
                    continue
                seen_candidates.add(key)
                
                roster.append({
                    'candidate_id': c['candidate_id'],
                    'name_full': c['name_full'],
                    'name_last': c['name_last'],
                    'name_first': c['name_first'],
                    'name_middle': c['name_middle'],
                    'name_suffix': c['name_suffix'],
                    'search_variants': c['search_variants'],
                    'party': c['party_normalized'],
                    'party_code': c['party_affiliation'],
                    'office': c['office_normalized'],
                    'office_code': c['office'],
                    'state': c['office_state'],
                    'district': c['office_district'] if c['office'] == 'H' else None,
                    'election_cycle': cycle_year,
                    'incumbent_status': c['incumbent_challenger_status'],
                    'candidate_status': c['candidate_status'],
                })
    
    # Sort roster
    roster.sort(key=lambda x: (x['election_cycle'], x['office'], x['state'] or '', x['name_last']))
    
    # Summary statistics
    stats = {
        'total_candidates': len(roster),
        'by_cycle': defaultdict(int),
        'by_office': defaultdict(int),
        'by_party': defaultdict(int),
    }
    
    for r in roster:
        stats['by_cycle'][r['election_cycle']] += 1
        stats['by_office'][r['office']] += 1
        stats['by_party'][r['party']] += 1
    
    return {
        'metadata': {
            'source': 'FEC Candidate Master Files',
            'cycles': cycles,
            'filters': {
                'president': include_president,
                'senate': include_senate,
                'house': include_house,
                'major_parties_only': major_parties_only
            },
            'statistics': dict(stats)
        },
        'roster': roster
    }


def save_roster(roster_data: Dict, output_path: Path):
    """Save roster to JSON file."""
    # Convert defaultdicts to regular dicts for JSON serialization
    if 'metadata' in roster_data and 'statistics' in roster_data['metadata']:
        stats = roster_data['metadata']['statistics']
        for key in ['by_cycle', 'by_office', 'by_party']:
            if key in stats:
                stats[key] = dict(stats[key])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(roster_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved roster to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution - build and save FEC roster."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect FEC candidate rosters')
    parser.add_argument('--cycles', nargs='+', type=int, default=[2016, 2018, 2020, 2022, 2024],
                       help='Election cycles to collect')
    parser.add_argument('--major-only', action='store_true',
                       help='Only include major party candidates (D/R)')
    parser.add_argument('--office', choices=['P', 'S', 'H', 'all'], default='all',
                       help='Office to include (P=President, S=Senate, H=House)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path')
    args = parser.parse_args()
    
    print("=" * 70)
    print("FEC CANDIDATE ROSTER COLLECTION")
    print("=" * 70)
    print(f"Cycles: {args.cycles}")
    print(f"Major parties only: {args.major_only}")
    print(f"Office: {args.office}")
    print("=" * 70)
    
    # Build roster
    roster_data = build_federal_roster(
        cycles=args.cycles,
        include_president=(args.office in ['P', 'all']),
        include_senate=(args.office in ['S', 'all']),
        include_house=(args.office in ['H', 'all']),
        major_parties_only=args.major_only
    )
    
    # Print summary
    stats = roster_data['metadata']['statistics']
    print("\n" + "=" * 70)
    print("COLLECTION SUMMARY")
    print("=" * 70)
    print(f"Total candidates: {stats['total_candidates']}")
    print("\nBy cycle:")
    for cycle, count in sorted(stats['by_cycle'].items()):
        print(f"  {cycle}: {count}")
    print("\nBy office:")
    for office, count in sorted(stats['by_office'].items()):
        print(f"  {office}: {count}")
    print("\nBy party (top 5):")
    for party, count in sorted(stats['by_party'].items(), key=lambda x: -x[1])[:5]:
        print(f"  {party}: {count}")
    
    # Save output
    output_dir = Path(__file__).parent.parent / 'data' / 'raw'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = f"_{'_'.join(str(c) for c in args.cycles)}"
        if args.major_only:
            suffix += "_major"
        output_path = output_dir / f'fec_roster{suffix}.json'
    
    save_roster(roster_data, output_path)
    
    print("\n" + "=" * 70)
    print(f"Roster saved to: {output_path}")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Run match_wikipedia_pages.py to match candidates to Wikipedia pages")
    print("2. Review matches and manually verify low-confidence matches")


if __name__ == '__main__':
    main()

