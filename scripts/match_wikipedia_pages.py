"""
Wikipedia Page Matching for Candidate Rosters

Matches FEC candidate roster to Wikipedia pages using:
1. Wikipedia search API with disambiguators
2. Redirect resolution
3. Fuzzy name matching with confidence scores

Outputs matched roster with Wikipedia page info ready for revision collection.
"""
import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging
from difflib import SequenceMatcher

sys.path.append(str(Path(__file__).parent))
from wiki_api import WikipediaAPI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# FUZZY MATCHING UTILITIES
# =============================================================================

def normalize_string(s: str) -> str:
    """Normalize string for comparison (lowercase, remove punctuation)."""
    s = s.lower()
    s = re.sub(r'[^\w\s]', '', s)  # Remove punctuation
    s = re.sub(r'\s+', ' ', s)     # Normalize whitespace
    return s.strip()


def string_similarity(s1: str, s2: str) -> float:
    """
    Calculate similarity between two strings using SequenceMatcher.
    Returns value between 0 (no match) and 1 (exact match).
    """
    s1_norm = normalize_string(s1)
    s2_norm = normalize_string(s2)
    return SequenceMatcher(None, s1_norm, s2_norm).ratio()


def name_match_score(candidate_name: str, wiki_title: str, 
                     state: str = None, office: str = None) -> float:
    """
    Calculate match confidence between candidate name and Wikipedia title.
    
    Considers:
    - Name similarity
    - Disambiguation hints (state, office)
    - Common Wikipedia title patterns
    
    Returns confidence score 0.0 to 1.0
    """
    # Clean Wikipedia title (remove disambiguation)
    wiki_clean = re.sub(r'\s*\([^)]+\)\s*$', '', wiki_title).strip()
    
    # Base similarity
    base_score = string_similarity(candidate_name, wiki_clean)
    
    # Bonus for state match in disambiguation
    state_bonus = 0.0
    if state and state.lower() in wiki_title.lower():
        state_bonus = 0.1
    
    # Bonus for office match in disambiguation
    office_bonus = 0.0
    office_keywords = {
        'House': ['representative', 'congressman', 'congresswoman'],
        'Senate': ['senator'],
        'President': ['president', 'presidential']
    }
    if office and office in office_keywords:
        for keyword in office_keywords[office]:
            if keyword in wiki_title.lower():
                office_bonus = 0.1
                break
    
    # Politician disambiguation is good signal
    if '(politician)' in wiki_title.lower() or '(u.s.' in wiki_title.lower():
        office_bonus = max(office_bonus, 0.05)
    
    final_score = min(1.0, base_score + state_bonus + office_bonus)
    return final_score


# =============================================================================
# WIKIPEDIA API EXTENSIONS
# =============================================================================

class WikipediaPageMatcher:
    """Match candidates to Wikipedia pages with disambiguation handling."""
    
    def __init__(self):
        self.api = WikipediaAPI(language='en')
        self.cache = {}  # Cache search results
    
    def resolve_redirects(self, title: str) -> Optional[Dict]:
        """
        Resolve redirects and get canonical page info.
        
        Returns:
            Dictionary with pageid, title, url, is_redirect, is_disambiguation
        """
        try:
            params = {
                'action': 'query',
                'titles': title,
                'redirects': 1,  # Follow redirects
                'prop': 'info|pageprops',
                'inprop': 'url',
            }
            
            data = self.api._make_request(params)
            
            # Check for redirects
            redirects = data.get('query', {}).get('redirects', [])
            
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                if page_id == '-1':
                    return None
                
                # Check if disambiguation page
                pageprops = page_data.get('pageprops', {})
                is_disambiguation = 'disambiguation' in pageprops
                
                return {
                    'pageid': page_data.get('pageid'),
                    'title': page_data.get('title'),
                    'url': page_data.get('fullurl'),
                    'is_redirect': len(redirects) > 0,
                    'redirect_from': redirects[0]['from'] if redirects else None,
                    'is_disambiguation': is_disambiguation
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error resolving redirects for '{title}': {e}")
            return None
    
    def search_candidate_pages(
        self, 
        name: str, 
        state: str = None, 
        office: str = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search for Wikipedia pages matching a candidate.
        
        Args:
            name: Candidate full name
            state: State code (e.g., 'CA')
            office: Office type (House/Senate/President)
            limit: Max results to return
            
        Returns:
            List of potential matches with confidence scores
        """
        results = []
        seen_pageids = set()
        
        # Build search queries
        queries = [f'"{name}"']  # Exact name match
        
        # Add disambiguators
        if state:
            # Convert state code to name if needed
            state_name = US_STATE_NAMES.get(state, state)
            queries.append(f'"{name}" {state_name}')
        
        if office:
            office_terms = {
                'House': 'representative',
                'Senate': 'senator', 
                'President': 'president'
            }
            if office in office_terms:
                queries.append(f'"{name}" {office_terms[office]}')
        
        # Run searches
        for query in queries:
            if query in self.cache:
                search_results = self.cache[query]
            else:
                search_results = self.api.search_pages(query, limit=limit)
                self.cache[query] = search_results
                time.sleep(0.1)  # Rate limiting
            
            for title in search_results:
                # Resolve redirects
                page_info = self.resolve_redirects(title)
                
                if page_info is None:
                    continue
                    
                # Skip disambiguation pages
                if page_info['is_disambiguation']:
                    continue
                
                # Skip if already seen
                if page_info['pageid'] in seen_pageids:
                    continue
                seen_pageids.add(page_info['pageid'])
                
                # Calculate match score
                confidence = name_match_score(
                    name, page_info['title'], state, office
                )
                
                results.append({
                    'pageid': page_info['pageid'],
                    'title': page_info['title'],
                    'url': page_info['url'],
                    'search_query': query,
                    'confidence': confidence,
                    'is_redirect': page_info['is_redirect'],
                    'redirect_from': page_info['redirect_from']
                })
        
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return results[:limit]
    
    def best_match(
        self, 
        name: str, 
        state: str = None, 
        office: str = None,
        confidence_threshold: float = 0.6
    ) -> Optional[Dict]:
        """
        Get best Wikipedia match for a candidate.
        
        Args:
            name: Candidate full name
            state: State code
            office: Office type
            confidence_threshold: Minimum confidence to accept
            
        Returns:
            Best match or None if no good match found
        """
        matches = self.search_candidate_pages(name, state, office, limit=5)
        
        if not matches:
            return None
        
        best = matches[0]
        
        if best['confidence'] >= confidence_threshold:
            return best
        
        return None


# US State code to name mapping
US_STATE_NAMES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
    'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
    'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia',
    'PR': 'Puerto Rico', 'GU': 'Guam', 'VI': 'Virgin Islands', 'AS': 'American Samoa'
}


# =============================================================================
# ROSTER MATCHING
# =============================================================================

def match_roster_to_wikipedia(
    roster_path: Path,
    output_path: Path,
    confidence_threshold: float = 0.6,
    progress_callback=None
):
    """
    Match entire roster to Wikipedia pages.
    
    Args:
        roster_path: Path to FEC roster JSON
        output_path: Path to save matched roster
        confidence_threshold: Minimum confidence for auto-match
        progress_callback: Optional callback(current, total, candidate_name)
    """
    # Load roster
    with open(roster_path, 'r', encoding='utf-8') as f:
        roster_data = json.load(f)
    
    roster = roster_data['roster']
    logger.info(f"Loaded roster with {len(roster)} candidates")
    
    # Initialize matcher
    matcher = WikipediaPageMatcher()
    
    # Track results
    matched = []
    unmatched = []
    low_confidence = []
    
    # Match each candidate
    for i, candidate in enumerate(roster):
        name = candidate['name_full']
        state = candidate.get('state')
        office = candidate.get('office')
        
        if progress_callback:
            progress_callback(i + 1, len(roster), name)
        
        # Search for matches
        matches = matcher.search_candidate_pages(name, state, office, limit=5)
        
        if matches:
            best = matches[0]
            
            candidate_result = {
                **candidate,
                'wikipedia_match': {
                    'pageid': best['pageid'],
                    'title': best['title'],
                    'url': best['url'],
                    'confidence': best['confidence'],
                    'is_redirect': best['is_redirect'],
                    'redirect_from': best['redirect_from'],
                    'search_query': best['search_query']
                },
                'all_matches': matches[:3],  # Keep top 3 for review
                'match_status': 'matched' if best['confidence'] >= confidence_threshold else 'low_confidence'
            }
            
            if best['confidence'] >= confidence_threshold:
                matched.append(candidate_result)
            else:
                low_confidence.append(candidate_result)
        else:
            candidate_result = {
                **candidate,
                'wikipedia_match': None,
                'all_matches': [],
                'match_status': 'unmatched'
            }
            unmatched.append(candidate_result)
        
        # Log progress
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(roster)} candidates")
    
    # Combine results
    all_results = matched + low_confidence + unmatched
    
    # Create output data
    output_data = {
        'metadata': {
            **roster_data['metadata'],
            'wikipedia_matching': {
                'confidence_threshold': confidence_threshold,
                'matched': len(matched),
                'low_confidence': len(low_confidence),
                'unmatched': len(unmatched),
                'total': len(roster)
            }
        },
        'roster': all_results
    }
    
    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved matched roster to {output_path}")
    
    return {
        'matched': len(matched),
        'low_confidence': len(low_confidence),
        'unmatched': len(unmatched)
    }


def create_page_list_from_matched_roster(
    matched_roster_path: Path,
    output_path: Path,
    include_low_confidence: bool = False
) -> Dict:
    """
    Create page list for revision collection from matched roster.
    
    Only includes candidates with Wikipedia matches.
    Deduplicates pages (same person may appear in multiple cycles).
    
    Args:
        matched_roster_path: Path to matched roster JSON
        output_path: Path to save page list
        include_low_confidence: Include low confidence matches
        
    Returns:
        Summary statistics
    """
    with open(matched_roster_path, 'r', encoding='utf-8') as f:
        roster_data = json.load(f)
    
    roster = roster_data['roster']
    
    # Deduplicate pages
    pages_by_id = {}  # pageid -> page_info
    page_candidates = defaultdict(list)  # pageid -> list of candidate info
    
    for candidate in roster:
        if candidate['match_status'] == 'unmatched':
            continue
        
        if candidate['match_status'] == 'low_confidence' and not include_low_confidence:
            continue
        
        match = candidate['wikipedia_match']
        pageid = match['pageid']
        
        if pageid not in pages_by_id:
            pages_by_id[pageid] = {
                'title': match['title'],
                'pageid': pageid,
                'url': match['url'],
            }
        
        # Track which candidates map to this page
        page_candidates[pageid].append({
            'candidate_id': candidate['candidate_id'],
            'name': candidate['name_full'],
            'party': candidate['party'],
            'office': candidate['office'],
            'state': candidate.get('state'),
            'district': candidate.get('district'),
            'election_cycle': candidate['election_cycle'],
            'confidence': match['confidence']
        })
    
    # Build page list with candidate info
    pages = []
    for pageid, page_info in pages_by_id.items():
        candidates = page_candidates[pageid]
        
        # Determine on_ballot status for each cycle
        cycles_on_ballot = {}
        for c in candidates:
            cycle = c['election_cycle']
            cycles_on_ballot[cycle] = {
                'office': c['office'],
                'party': c['party'],
                'state': c.get('state'),
                'district': c.get('district')
            }
        
        pages.append({
            **page_info,
            'candidates': candidates,
            'cycles_on_ballot': cycles_on_ballot,
            'num_cycles': len(cycles_on_ballot)
        })
    
    # Sort by number of cycles (most cycles first = major politicians)
    pages.sort(key=lambda x: (-x['num_cycles'], x['title']))
    
    # Create output
    output_data = {
        'metadata': {
            'description': 'Wikipedia pages for US federal candidates (FEC roster)',
            'source': 'FEC + Wikipedia matching',
            'total_pages': len(pages),
            'date_range': {
                'start': '2015-01-01T00:00:00Z',
                'end': '2024-12-31T23:59:59Z'
            },
            'language': 'en'
        },
        'pages': pages
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created page list with {len(pages)} unique pages")
    
    return {
        'total_pages': len(pages),
        'total_candidate_entries': sum(len(p['candidates']) for p in pages)
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser(description='Match FEC roster to Wikipedia pages')
    parser.add_argument('--roster', type=str, required=True,
                       help='Path to FEC roster JSON')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for matched roster')
    parser.add_argument('--threshold', type=float, default=0.6,
                       help='Confidence threshold for auto-match (0.0-1.0)')
    parser.add_argument('--create-page-list', action='store_true',
                       help='Also create page list for revision collection')
    args = parser.parse_args()
    
    roster_path = Path(args.roster)
    if not roster_path.exists():
        logger.error(f"Roster file not found: {roster_path}")
        return
    
    output_path = Path(args.output) if args.output else roster_path.with_suffix('.matched.json')
    
    print("=" * 70)
    print("WIKIPEDIA PAGE MATCHING")
    print("=" * 70)
    print(f"Input roster: {roster_path}")
    print(f"Output path: {output_path}")
    print(f"Confidence threshold: {args.threshold}")
    print("=" * 70)
    
    # Create progress bar
    pbar = None
    
    def progress_callback(current, total, name):
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(total=total, desc="Matching")
        pbar.update(1)
        pbar.set_postfix_str(name[:30])
    
    # Run matching
    results = match_roster_to_wikipedia(
        roster_path,
        output_path,
        confidence_threshold=args.threshold,
        progress_callback=progress_callback
    )
    
    if pbar:
        pbar.close()
    
    print("\n" + "=" * 70)
    print("MATCHING RESULTS")
    print("=" * 70)
    print(f"Matched (confidence >= {args.threshold}): {results['matched']}")
    print(f"Low confidence (needs review): {results['low_confidence']}")
    print(f"Unmatched (no Wikipedia page found): {results['unmatched']}")
    
    # Create page list if requested
    if args.create_page_list:
        page_list_path = output_path.parent / 'page_list_fec.json'
        page_stats = create_page_list_from_matched_roster(
            output_path, 
            page_list_path,
            include_low_confidence=False
        )
        print(f"\nPage list created: {page_list_path}")
        print(f"  Unique pages: {page_stats['total_pages']}")
        print(f"  Candidate-cycle entries: {page_stats['total_candidate_entries']}")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Review low_confidence matches in the output file")
    print("2. Manually verify/correct matches as needed")
    print("3. Run collect_revisions.py with the page list")


if __name__ == '__main__':
    main()

