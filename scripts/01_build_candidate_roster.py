"""
Build Complete Candidate Roster with On-Ballot Indicators

This is the main pipeline script that:
1. Collects FEC candidate data
2. Matches to Wikipedia pages
3. Adds on_ballot indicators per cycle
4. Creates analysis-ready roster

Output is a comprehensive roster ready for event-study analysis.
"""
import sys
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import logging
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from election_cycles import (
    get_election_day, get_on_ballot_indicator, get_study_cycles,
    STATE_SENATE_CLASSES, is_presidential_year
)
from collect_fec_rosters import FECRosterCollector, build_federal_roster, save_roster
from match_wikipedia_pages import (
    WikipediaPageMatcher, match_roster_to_wikipedia,
    create_page_list_from_matched_roster, US_STATE_NAMES
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# SENATE CLASS ASSIGNMENT
# =============================================================================

# Known senators with their class (for lookup)
# This helps when FEC data doesn't include class info
KNOWN_SENATORS = {
    # Format: 'name_normalized': {'class': int, 'state': str, 'party': str}
    # Will be populated from external sources or manually curated
}


def infer_senate_class(candidate: Dict, election_year: int) -> Optional[int]:
    """
    Infer Senate class from state and election year.
    
    Logic: If a senator is running in year Y, they must be in the class
    that has elections in year Y.
    
    Args:
        candidate: Candidate dictionary with 'state' field
        election_year: The year of the election
        
    Returns:
        Senate class (1, 2, or 3) or None if cannot determine
    """
    state = candidate.get('state')
    if not state or state not in STATE_SENATE_CLASSES:
        return None
    
    classes = STATE_SENATE_CLASSES[state]
    
    # Check which class is up this year
    from election_cycles import is_senate_seat_on_ballot
    
    for cls in classes:
        if is_senate_seat_on_ballot(cls, election_year):
            return cls
    
    return None


# =============================================================================
# ON-BALLOT INDICATOR COMPUTATION
# =============================================================================

def compute_on_ballot_matrix(
    candidate: Dict,
    cycles: List[int]
) -> Dict[int, bool]:
    """
    Compute on_ballot indicator for each cycle.
    
    For a candidate who appeared in cycle Y:
    - House: on_ballot=1 only in year Y (must file each cycle)
    - Senate: on_ballot=1 only in years matching their class
    - President: on_ballot=1 only in presidential years they filed
    
    Args:
        candidate: Candidate dictionary with office, state, election_cycle
        cycles: List of all study cycles
        
    Returns:
        Dictionary mapping cycle year to on_ballot status
    """
    office = candidate.get('office_code', candidate.get('office', ''))
    if isinstance(office, str) and len(office) > 1:
        office_map = {'House': 'H', 'Senate': 'S', 'President': 'P'}
        office = office_map.get(office, office[0].upper())
    
    filed_cycle = candidate.get('election_cycle')
    state = candidate.get('state')
    
    on_ballot = {}
    
    for cycle in cycles:
        if office == 'H':
            # House: on ballot only in the cycle they filed
            on_ballot[cycle] = (cycle == filed_cycle)
            
        elif office == 'S':
            # Senate: need to track by class
            senate_class = infer_senate_class(candidate, filed_cycle) if filed_cycle else None
            
            if senate_class:
                # Check if this class is up in this cycle
                on_ballot[cycle] = get_on_ballot_indicator('S', cycle, senate_class=senate_class)
            else:
                # Fallback: only mark on ballot in filed cycle
                on_ballot[cycle] = (cycle == filed_cycle)
                
        elif office == 'P':
            # President: on ballot only in presidential years they filed
            on_ballot[cycle] = (cycle == filed_cycle) and is_presidential_year(cycle)
            
        else:
            on_ballot[cycle] = False
    
    return on_ballot


# =============================================================================
# ROSTER BUILDING PIPELINE
# =============================================================================

class RosterBuilder:
    """Build complete candidate roster with Wikipedia matching and on_ballot indicators."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cycles = [2016, 2018, 2020, 2022, 2024]
        self.fec_roster = None
        self.matched_roster = None
        self.final_roster = None
    
    def step1_collect_fec_data(
        self,
        include_president: bool = True,
        include_senate: bool = True,
        include_house: bool = True,
        major_parties_only: bool = False,
        sample_fraction: Optional[float] = None,
        random_seed: int = 42
    ) -> Path:
        """
        Step 1: Collect FEC candidate data.
        
        Args:
            include_president: Include presidential candidates
            include_senate: Include Senate candidates
            include_house: Include House candidates
            major_parties_only: Only D/R candidates
            sample_fraction: If set (0.0-1.0), randomly sample this fraction of candidates
            random_seed: Random seed for reproducible sampling
        
        Returns path to saved roster.
        """
        logger.info("=" * 70)
        logger.info("STEP 1: Collecting FEC Candidate Data")
        logger.info("=" * 70)
        
        roster_data = build_federal_roster(
            cycles=self.cycles,
            include_president=include_president,
            include_senate=include_senate,
            include_house=include_house,
            major_parties_only=major_parties_only
        )
        
        # Apply sampling if requested
        if sample_fraction is not None and 0 < sample_fraction < 1:
            original_count = len(roster_data['roster'])
            
            # Set seed for reproducibility
            random.seed(random_seed)
            
            # Sample the roster
            sample_size = int(original_count * sample_fraction)
            roster_data['roster'] = random.sample(roster_data['roster'], sample_size)
            
            # Update metadata
            roster_data['metadata']['sampling'] = {
                'original_count': original_count,
                'sample_fraction': sample_fraction,
                'sample_size': sample_size,
                'random_seed': random_seed
            }
            roster_data['metadata']['statistics']['total_candidates'] = sample_size
            
            # Recompute by_cycle, by_office, by_party stats
            stats = roster_data['metadata']['statistics']
            stats['by_cycle'] = defaultdict(int)
            stats['by_office'] = defaultdict(int)
            stats['by_party'] = defaultdict(int)
            for c in roster_data['roster']:
                stats['by_cycle'][c['election_cycle']] += 1
                stats['by_office'][c['office']] += 1
                stats['by_party'][c['party']] += 1
            stats['by_cycle'] = dict(stats['by_cycle'])
            stats['by_office'] = dict(stats['by_office'])
            stats['by_party'] = dict(stats['by_party'])
            
            logger.info(f"Sampled {sample_size} candidates from {original_count} "
                       f"({sample_fraction*100:.1f}%, seed={random_seed})")
        
        self.fec_roster = roster_data
        
        # Save
        fec_path = self.output_dir / 'fec_roster_raw.json'
        save_roster(roster_data, fec_path)
        
        stats = roster_data['metadata']['statistics']
        logger.info(f"Final roster: {stats['total_candidates']} candidates")
        
        return fec_path
    
    def step2_match_wikipedia(
        self,
        fec_roster_path: Path = None,
        confidence_threshold: float = 0.6
    ) -> Path:
        """
        Step 2: Match candidates to Wikipedia pages.
        
        Returns path to matched roster.
        """
        logger.info("=" * 70)
        logger.info("STEP 2: Matching to Wikipedia Pages")
        logger.info("=" * 70)
        
        if fec_roster_path is None:
            fec_roster_path = self.output_dir / 'fec_roster_raw.json'
        
        matched_path = self.output_dir / 'fec_roster_matched.json'
        
        # Run matching
        from tqdm import tqdm
        
        with open(fec_roster_path, 'r') as f:
            roster_data = json.load(f)
        
        total = len(roster_data['roster'])
        pbar = tqdm(total=total, desc="Matching Wikipedia pages")
        
        def progress(current, total, name):
            pbar.update(1)
            pbar.set_postfix_str(name[:25] if name else '')
        
        results = match_roster_to_wikipedia(
            fec_roster_path,
            matched_path,
            confidence_threshold=confidence_threshold,
            progress_callback=progress
        )
        
        pbar.close()
        
        logger.info(f"Matched: {results['matched']}, "
                   f"Low confidence: {results['low_confidence']}, "
                   f"Unmatched: {results['unmatched']}")
        
        # Load matched roster
        with open(matched_path, 'r') as f:
            self.matched_roster = json.load(f)
        
        return matched_path
    
    def step3_build_final_roster(
        self,
        matched_roster_path: Path = None,
        include_unmatched: bool = False,
        include_low_confidence: bool = False,
        min_confidence: float = 0.6,
        exclude_election_pages: bool = True
    ) -> Path:
        """
        Step 3: Build final roster with on_ballot indicators.

        Creates analysis-ready roster with:
        - Wikipedia page info
        - on_ballot[cycle] for each cycle
        - Candidate metadata

        Args:
            matched_roster_path: Path to matched roster JSON
            include_unmatched: Include unmatched candidates
            include_low_confidence: Include low-confidence matches (overrides min_confidence)
            min_confidence: Minimum confidence threshold (default: 0.6)
            exclude_election_pages: Filter out election/informational pages (default: True)

        Returns path to final roster.
        """
        logger.info("=" * 70)
        logger.info("STEP 3: Building Final Roster with On-Ballot Indicators")
        logger.info("=" * 70)
        
        if matched_roster_path is None:
            matched_roster_path = self.output_dir / 'fec_roster_matched.json'
        
        with open(matched_roster_path, 'r') as f:
            roster_data = json.load(f)
        
        # Group by Wikipedia page (deduplicate)
        pages_data = defaultdict(lambda: {
            'candidates': [],
            'cycles_on_ballot': {},
            'wikipedia': None
        })
        
        # Statistics tracking
        stats_filtered = {
            'unmatched': 0,
            'low_confidence': 0,
            'election_pages': 0,
            'included': 0
        }

        for candidate in roster_data['roster']:
            # Skip unmatched unless requested
            if candidate['match_status'] == 'unmatched':
                stats_filtered['unmatched'] += 1
                if not include_unmatched:
                    continue

            # Get Wikipedia match
            wiki_match = candidate.get('wikipedia_match')
            if wiki_match is None:
                continue

            # Filter by confidence
            confidence = wiki_match.get('confidence', 0)
            if not include_low_confidence and confidence < min_confidence:
                stats_filtered['low_confidence'] += 1
                continue

            # Filter out election/informational pages
            page_title = wiki_match.get('title', '')
            if exclude_election_pages and self._is_election_page(page_title):
                stats_filtered['election_pages'] += 1
                continue

            # Filter out non-politician pages (athletes, entertainers, etc.)
            if self._is_non_politician_page(page_title):
                stats_filtered['election_pages'] += 1  # Count as "bad pages"
                logger.warning(f"Filtered non-politician page: {page_title} (matched to {candidate['name_full']})")
                continue

            stats_filtered['included'] += 1
            pageid = wiki_match['pageid']
            
            # Store Wikipedia info (first occurrence)
            if pages_data[pageid]['wikipedia'] is None:
                pages_data[pageid]['wikipedia'] = {
                    'pageid': pageid,
                    'title': wiki_match['title'],
                    'url': wiki_match['url']
                }
            
            # Compute on_ballot for this candidate
            on_ballot = compute_on_ballot_matrix(candidate, self.cycles)
            
            # Store candidate info
            pages_data[pageid]['candidates'].append({
                'candidate_id': candidate['candidate_id'],
                'name': candidate['name_full'],
                'party': candidate['party'],
                'party_code': candidate['party_code'],
                'office': candidate['office'],
                'office_code': candidate.get('office_code', candidate['office'][0] if candidate['office'] else ''),
                'state': candidate.get('state'),
                'district': candidate.get('district'),
                'election_cycle': candidate['election_cycle'],
                'match_confidence': wiki_match['confidence']
            })
            
            # Merge on_ballot indicators
            for cycle, is_on_ballot in on_ballot.items():
                if is_on_ballot:
                    pages_data[pageid]['cycles_on_ballot'][cycle] = {
                        'office': candidate['office'],
                        'party': candidate['party'],
                        'state': candidate.get('state')
                    }
        
        # Build final page list
        final_pages = []
        for pageid, data in pages_data.items():
            # Create on_ballot array for each cycle
            on_ballot_by_cycle = {
                cycle: cycle in data['cycles_on_ballot']
                for cycle in self.cycles
            }
            
            final_pages.append({
                **data['wikipedia'],
                'on_ballot': on_ballot_by_cycle,
                'cycles_on_ballot_details': data['cycles_on_ballot'],
                'candidates': data['candidates'],
                'primary_office': self._get_primary_office(data['candidates']),
                'primary_party': self._get_primary_party(data['candidates']),
                'primary_state': self._get_primary_state(data['candidates']),
            })
        
        # Sort by number of on-ballot cycles (descending)
        final_pages.sort(key=lambda x: (
            -sum(x['on_ballot'].values()),
            x['title']
        ))
        
        # Build output
        final_roster = {
            'metadata': {
                'description': 'Federal candidates roster with Wikipedia pages and on-ballot indicators',
                'source': 'FEC Candidate Master + Wikipedia',
                'cycles': self.cycles,
                'election_days': {
                    cycle: get_election_day(cycle).isoformat()
                    for cycle in self.cycles
                },
                'total_pages': len(final_pages),
                'statistics': self._compute_stats(final_pages),
                'created': datetime.now().isoformat(),
                'date_range': {
                    'start': '2014-11-01T00:00:00Z',  # Before 2014 midterm to cover full 2016 cycle
                    'end': '2024-12-31T23:59:59Z'
                },
                'language': 'en'
            },
            'pages': final_pages
        }
        
        self.final_roster = final_roster

        # Save
        final_path = self.output_dir / 'candidate_roster_final.json'
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(final_roster, f, indent=2, ensure_ascii=False)

        # Log filtering statistics
        logger.info("\nFiltering Summary:")
        logger.info(f"  Included: {stats_filtered['included']} candidates")
        logger.info(f"  Filtered out:")
        logger.info(f"    - Unmatched: {stats_filtered['unmatched']}")
        logger.info(f"    - Low confidence (< {min_confidence}): {stats_filtered['low_confidence']}")
        logger.info(f"    - Election pages: {stats_filtered['election_pages']}")
        logger.info(f"  Unique Wikipedia pages: {len(final_pages)}")
        
        # Also save as page_list format for revision collection
        page_list_path = self.output_dir / 'page_list_candidates.json'
        self._save_as_page_list(final_roster, page_list_path)
        
        return final_path
    
    def _is_election_page(self, title: str) -> bool:
        """
        Check if a Wikipedia page is an election/informational page rather than a candidate page.

        Args:
            title: Wikipedia page title

        Returns:
            True if this is an election/informational page
        """
        title_lower = title.lower()

        # Election pages
        election_patterns = [
            'election in',
            'elections in',
            'united states house of representatives elections',
            'united states senate election',
            ' primary',
            ' caucus',
            'electoral district',
            'congressional district',
            'state senate district',
            'municipal election',
            'gubernatorial election'
        ]

        for pattern in election_patterns:
            if pattern in title_lower:
                return True

        return False

    def _is_non_politician_page(self, title: str) -> bool:
        """
        Check if a Wikipedia page is clearly about a non-politician (athlete, entertainer, etc.).

        This catches common mismatches where candidates with common names get matched to
        famous people in other fields.

        Args:
            title: Wikipedia page title

        Returns:
            True if this is definitely not a politician page
        """
        title_lower = title.lower()

        # Occupation disambiguators that indicate non-politicians
        non_politician_patterns = [
            'footballer', 'soccer player', 'baseball', 'basketball', 'hockey',
            'cricketer', 'rugby', 'tennis', 'golfer', 'athlete',
            'darts player', 'boxer', 'wrestler',
            'actor', 'actress', 'musician', 'singer', 'rapper', 'dj',
            'artist', 'painter', 'sculptor',
            'writer', 'author', 'poet', 'journalist', 'reporter',
            'director', 'producer', 'screenwriter',
            'scientist', 'physicist', 'chemist', 'biologist',
            'doctor', 'physician', 'surgeon',
            'businessman', 'entrepreneur', 'ceo',
            'chef', 'architect', 'engineer'
        ]

        # Check for occupation in parentheses (e.g., "John Smith (footballer)")
        if '(' in title_lower:
            # Extract content in parentheses
            paren_content = title_lower.split('(')[-1].split(')')[0]
            for pattern in non_politician_patterns:
                if pattern in paren_content:
                    return True

        return False

    def _get_primary_office(self, candidates: List[Dict]) -> str:
        """Get most common/highest office for this person."""
        offices = [c['office'] for c in candidates]
        # Prioritize: President > Senate > House
        if 'President' in offices:
            return 'President'
        if 'Senate' in offices:
            return 'Senate'
        return offices[0] if offices else 'Unknown'
    
    def _get_primary_party(self, candidates: List[Dict]) -> str:
        """Get most common party for this person."""
        from collections import Counter
        parties = [c['party'] for c in candidates if c['party']]
        if not parties:
            return 'Unknown'
        return Counter(parties).most_common(1)[0][0]
    
    def _get_primary_state(self, candidates: List[Dict]) -> Optional[str]:
        """Get most common state for this person."""
        from collections import Counter
        states = [c['state'] for c in candidates if c['state']]
        if not states:
            return None
        return Counter(states).most_common(1)[0][0]
    
    def _compute_stats(self, pages: List[Dict]) -> Dict:
        """Compute summary statistics."""
        stats = {
            'total_pages': len(pages),
            'by_office': defaultdict(int),
            'by_party': defaultdict(int),
            'on_ballot_counts': {cycle: 0 for cycle in self.cycles}
        }
        
        for page in pages:
            stats['by_office'][page['primary_office']] += 1
            stats['by_party'][page['primary_party']] += 1
            for cycle, on_ballot in page['on_ballot'].items():
                if on_ballot:
                    stats['on_ballot_counts'][cycle] += 1
        
        return {
            **stats,
            'by_office': dict(stats['by_office']),
            'by_party': dict(stats['by_party'])
        }
    
    def _save_as_page_list(self, roster: Dict, output_path: Path):
        """Save in page_list format compatible with existing revision collection."""
        page_list = {
            'metadata': roster['metadata'],
            'pages': [
                {
                    'title': p['title'],
                    'pageid': p['pageid'],
                    'url': p['url'],
                    'on_ballot': p['on_ballot'],
                    'primary_office': p['primary_office'],
                    'primary_party': p['primary_party'],
                    'primary_state': p['primary_state']
                }
                for p in roster['pages']
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(page_list, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved page list to {output_path}")
    
    def run_full_pipeline(
        self,
        include_president: bool = True,
        include_senate: bool = True,
        include_house: bool = True,
        major_parties_only: bool = False,
        confidence_threshold: float = 0.6,
        sample_fraction: Optional[float] = None,
        random_seed: int = 42,
        include_low_confidence: bool = False,
        exclude_election_pages: bool = True
    ) -> Path:
        """
        Run the full pipeline: FEC -> Wikipedia matching -> Final roster.

        Args:
            sample_fraction: If set (0.0-1.0), randomly sample this fraction of candidates
            random_seed: Random seed for reproducible sampling
            include_low_confidence: Include low-confidence matches in final roster
            exclude_election_pages: Exclude election/informational pages from final roster

        Returns path to final roster.
        """
        logger.info("=" * 70)
        logger.info("RUNNING FULL CANDIDATE ROSTER PIPELINE")
        logger.info("=" * 70)
        
        if sample_fraction is not None:
            logger.info(f"SAMPLING: {sample_fraction*100:.1f}% of candidates (seed={random_seed})")
        
        # Step 1: FEC data
        self.step1_collect_fec_data(
            include_president=include_president,
            include_senate=include_senate,
            include_house=include_house,
            major_parties_only=major_parties_only,
            sample_fraction=sample_fraction,
            random_seed=random_seed
        )
        
        # Step 2: Wikipedia matching
        self.step2_match_wikipedia(confidence_threshold=confidence_threshold)

        # Step 3: Final roster with on_ballot
        final_path = self.step3_build_final_roster(
            include_low_confidence=include_low_confidence,
            min_confidence=confidence_threshold,
            exclude_election_pages=exclude_election_pages
        )
        
        # Print summary
        self._print_summary()
        
        return final_path
    
    def _print_summary(self):
        """Print final summary."""
        if self.final_roster is None:
            return
        
        stats = self.final_roster['metadata']['statistics']
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\nTotal Wikipedia pages: {stats['total_pages']}")
        
        print("\nBy office:")
        for office, count in sorted(stats['by_office'].items()):
            print(f"  {office}: {count}")
        
        print("\nBy party (top 5):")
        by_party = sorted(stats['by_party'].items(), key=lambda x: -x[1])
        for party, count in by_party[:5]:
            print(f"  {party}: {count}")
        
        print("\nOn-ballot counts by cycle:")
        for cycle, count in sorted(stats['on_ballot_counts'].items()):
            print(f"  {cycle}: {count} pages")
        
        print("\n" + "=" * 70)
        print("OUTPUT FILES:")
        print("=" * 70)
        print(f"  FEC roster (raw): {self.output_dir / 'fec_roster_raw.json'}")
        print(f"  FEC roster (matched): {self.output_dir / 'fec_roster_matched.json'}")
        print(f"  Final roster: {self.output_dir / 'candidate_roster_final.json'}")
        print(f"  Page list: {self.output_dir / 'page_list_candidates.json'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Build candidate roster with on-ballot indicators')
    
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: data/raw)')
    parser.add_argument('--major-only', action='store_true',
                       help='Only include major party candidates (D/R)')
    parser.add_argument('--no-president', action='store_true',
                       help='Exclude presidential candidates')
    parser.add_argument('--no-senate', action='store_true',
                       help='Exclude Senate candidates')
    parser.add_argument('--no-house', action='store_true',
                       help='Exclude House candidates')
    parser.add_argument('--sample', type=float, default=None,
                       help='Randomly sample X%% of candidates (e.g., --sample 10 for 10%%)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling (default: 42)')
    parser.add_argument('--confidence', type=float, default=0.6,
                       help='Wikipedia match confidence threshold (default: 0.6)')
    parser.add_argument('--include-low-confidence', action='store_true',
                       help='Include low-confidence matches (< threshold)')
    parser.add_argument('--include-election-pages', action='store_true',
                       help='Include election/informational pages (not recommended)')
    parser.add_argument('--step', choices=['fec', 'match', 'final', 'all'], default='all',
                       help='Which step to run')
    
    args = parser.parse_args()
    
    # Parse sample fraction (accept both 10 for 10% and 0.1 for 10%)
    sample_fraction = None
    if args.sample is not None:
        if args.sample > 1:
            sample_fraction = args.sample / 100.0  # Treat as percentage
        else:
            sample_fraction = args.sample  # Treat as fraction
        
        if sample_fraction <= 0 or sample_fraction >= 1:
            parser.error("--sample must be between 0 and 100 (percentage) or 0 and 1 (fraction)")
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / 'data' / 'raw'
    
    builder = RosterBuilder(output_dir)
    
    if args.step == 'all':
        builder.run_full_pipeline(
            include_president=not args.no_president,
            include_senate=not args.no_senate,
            include_house=not args.no_house,
            major_parties_only=args.major_only,
            confidence_threshold=args.confidence,
            sample_fraction=sample_fraction,
            random_seed=args.seed,
            include_low_confidence=args.include_low_confidence,
            exclude_election_pages=not args.include_election_pages
        )
    elif args.step == 'fec':
        builder.step1_collect_fec_data(
            include_president=not args.no_president,
            include_senate=not args.no_senate,
            include_house=not args.no_house,
            major_parties_only=args.major_only,
            sample_fraction=sample_fraction,
            random_seed=args.seed
        )
    elif args.step == 'match':
        builder.step2_match_wikipedia(confidence_threshold=args.confidence)
    elif args.step == 'final':
        builder.step3_build_final_roster(
            include_low_confidence=args.include_low_confidence,
            min_confidence=args.confidence,
            exclude_election_pages=not args.include_election_pages
        )
    
    print("\nDone!")
    print("\nNext steps:")
    print("1. Review fec_roster_matched.json for low-confidence matches")
    print("2. Run: python 02_collect_revisions.py --config candidates")
    print("3. Run: python 03_process_data.py --config candidates")
    print("4. Open the updated analysis notebook")


if __name__ == '__main__':
    main()

