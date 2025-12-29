"""
Election Cycle Utilities for US Federal Elections

Defines:
- Election Day calculations (first Tuesday after first Monday in November)
- 2-year federal election cycles
- Cycle-safe baseline windows with post-election washout
- Senate class staggering mapping
"""
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional
import calendar


# =============================================================================
# ELECTION DAY CALCULATION
# =============================================================================

def get_election_day(year: int) -> date:
    """
    Calculate federal Election Day for a given year.
    Election Day = First Tuesday after the first Monday in November.
    
    Args:
        year: Election year (even years only for federal elections)
        
    Returns:
        date object for Election Day
    """
    # Find the first Monday in November
    nov_1 = date(year, 11, 1)
    
    # If Nov 1 is a Monday, first Monday is Nov 1
    # Otherwise, find the first Monday
    days_until_monday = (7 - nov_1.weekday()) % 7
    if nov_1.weekday() == 0:  # Nov 1 is already Monday
        first_monday = nov_1
    else:
        first_monday = nov_1 + timedelta(days=days_until_monday)
    
    # Election Day is the day after (Tuesday)
    election_day = first_monday + timedelta(days=1)
    
    return election_day


def get_all_election_days(start_year: int = 2014, end_year: int = 2024) -> Dict[int, date]:
    """
    Get Election Days for all federal election years in range.
    
    Args:
        start_year: Start year (will be adjusted to even year)
        end_year: End year (inclusive)
        
    Returns:
        Dictionary mapping year to Election Day
    """
    # Adjust to even years only
    start = start_year if start_year % 2 == 0 else start_year + 1
    
    return {year: get_election_day(year) for year in range(start, end_year + 1, 2)}


# =============================================================================
# ELECTION CYCLE DEFINITIONS
# =============================================================================

def get_cycle_boundaries(election_year: int) -> Tuple[date, date]:
    """
    Get the start and end dates for a 2-year election cycle.
    
    Cycle y: (E_{y-2}, E_y]
    - Starts day after previous election
    - Ends on current election day (inclusive)
    
    Args:
        election_year: The target election year (even year)
        
    Returns:
        Tuple of (cycle_start, cycle_end) dates
    """
    if election_year % 2 != 0:
        raise ValueError(f"Election year must be even, got {election_year}")
    
    prev_election = get_election_day(election_year - 2)
    curr_election = get_election_day(election_year)
    
    cycle_start = prev_election + timedelta(days=1)  # Day after previous election
    cycle_end = curr_election
    
    return cycle_start, cycle_end


def get_cycle_windows(
    election_year: int,
    pre_election_days: int = 60,
    washout_days: int = 30
) -> Dict[str, Tuple[date, date]]:
    """
    Get analysis windows within a cycle (cycle-safe, no baseline contamination).
    
    Within each cycle (E_{y-2}, E_y]:
    - Washout: [E_{y-2}, E_{y-2} + W] - excluded from analysis
    - Baseline: (E_{y-2} + W, E_y - L] - "normal" editing period
    - Pre-election: (E_y - L, E_y) - event window of interest
    
    Args:
        election_year: Target election year
        pre_election_days: Days before election for event window (L)
        washout_days: Days after previous election to exclude (W)
        
    Returns:
        Dictionary with window definitions (start, end dates)
    """
    prev_election = get_election_day(election_year - 2)
    curr_election = get_election_day(election_year)
    
    washout_start = prev_election
    washout_end = prev_election + timedelta(days=washout_days)
    
    baseline_start = washout_end + timedelta(days=1)
    baseline_end = curr_election - timedelta(days=pre_election_days)
    
    pre_election_start = curr_election - timedelta(days=pre_election_days) + timedelta(days=1)
    pre_election_end = curr_election - timedelta(days=1)  # Day before election
    
    return {
        'washout': (washout_start, washout_end),
        'baseline': (baseline_start, baseline_end),
        'pre_election': (pre_election_start, pre_election_end),
        'election_day': (curr_election, curr_election),
        'cycle_start': prev_election + timedelta(days=1),
        'cycle_end': curr_election
    }


def date_to_event_time(d: date, election_year: int) -> int:
    """
    Convert a date to event time (days until Election Day, negative before).
    
    Args:
        d: Date to convert
        election_year: Reference election year
        
    Returns:
        Integer days from election (negative = before, 0 = Election Day)
    """
    election_day = get_election_day(election_year)
    return (d - election_day).days


# =============================================================================
# SENATE CLASS STAGGERING
# =============================================================================

# Senate classes and which years they're up for election
# Class 1: 2018, 2024, 2030...
# Class 2: 2020, 2026, 2032...
# Class 3: 2016, 2022, 2028...

SENATE_CLASS_BASE_YEARS = {
    1: 2018,  # Class 1 was up in 2018
    2: 2020,  # Class 2 was up in 2020
    3: 2016,  # Class 3 was up in 2016
}


def get_senate_class_election_years(senate_class: int, start_year: int = 2014, end_year: int = 2024) -> List[int]:
    """
    Get all election years when a Senate class is up for election.
    
    Args:
        senate_class: Senate class (1, 2, or 3)
        start_year: Start of range
        end_year: End of range (inclusive)
        
    Returns:
        List of years when this class has elections
    """
    if senate_class not in [1, 2, 3]:
        raise ValueError(f"Senate class must be 1, 2, or 3, got {senate_class}")
    
    base_year = SENATE_CLASS_BASE_YEARS[senate_class]
    
    years = []
    year = base_year
    while year >= start_year:
        year -= 6
    year += 6
    
    while year <= end_year:
        if year >= start_year:
            years.append(year)
        year += 6
    
    return years


def is_senate_seat_on_ballot(senate_class: int, election_year: int) -> bool:
    """
    Check if a Senate seat (by class) is on the ballot in a given year.
    
    Args:
        senate_class: Senate class (1, 2, or 3)
        election_year: Year to check
        
    Returns:
        True if this class has an election that year
    """
    base_year = SENATE_CLASS_BASE_YEARS[senate_class]
    return (election_year - base_year) % 6 == 0


# State -> Senate class mapping (both seats)
# Format: state_code -> (class_of_seat_1, class_of_seat_2)
STATE_SENATE_CLASSES = {
    'AL': (2, 3), 'AK': (2, 3), 'AZ': (1, 3), 'AR': (2, 3), 'CA': (1, 3),
    'CO': (2, 3), 'CT': (1, 3), 'DE': (1, 2), 'FL': (1, 3), 'GA': (2, 3),
    'HI': (1, 3), 'ID': (2, 3), 'IL': (2, 3), 'IN': (1, 3), 'IA': (2, 3),
    'KS': (2, 3), 'KY': (2, 3), 'LA': (2, 3), 'ME': (1, 2), 'MD': (1, 3),
    'MA': (1, 2), 'MI': (1, 2), 'MN': (1, 2), 'MS': (1, 2), 'MO': (1, 3),
    'MT': (1, 2), 'NE': (1, 2), 'NV': (1, 3), 'NH': (2, 3), 'NJ': (1, 2),
    'NM': (1, 2), 'NY': (1, 3), 'NC': (2, 3), 'ND': (1, 3), 'OH': (1, 3),
    'OK': (2, 3), 'OR': (2, 3), 'PA': (1, 3), 'RI': (1, 2), 'SC': (2, 3),
    'SD': (2, 3), 'TN': (1, 2), 'TX': (1, 2), 'UT': (1, 3), 'VT': (1, 3),
    'VA': (1, 2), 'WA': (1, 3), 'WV': (1, 2), 'WI': (1, 3), 'WY': (1, 2),
}


# =============================================================================
# PRESIDENTIAL ELECTION YEARS
# =============================================================================

def is_presidential_year(year: int) -> bool:
    """Check if a year is a presidential election year (every 4 years)."""
    return year % 4 == 0


def get_presidential_years(start_year: int = 2014, end_year: int = 2024) -> List[int]:
    """Get all presidential election years in range."""
    return [y for y in range(start_year, end_year + 1) if is_presidential_year(y)]


# =============================================================================
# ON-BALLOT INDICATOR
# =============================================================================

def get_on_ballot_indicator(
    office: str,  # 'P', 'S', 'H'
    election_year: int,
    senate_class: Optional[int] = None,
    state: Optional[str] = None
) -> bool:
    """
    Determine if a candidate is on the ballot in a given cycle.
    
    Args:
        office: 'P' (President), 'S' (Senate), 'H' (House)
        election_year: The election year
        senate_class: For Senate, the class (1, 2, or 3)
        state: State code (used for Senate class lookup if senate_class not provided)
        
    Returns:
        True if on ballot, False otherwise
    """
    if office == 'P':
        return is_presidential_year(election_year)
    
    elif office == 'H':
        # House is up every 2 years
        return election_year % 2 == 0
    
    elif office == 'S':
        if senate_class is not None:
            return is_senate_seat_on_ballot(senate_class, election_year)
        elif state is not None and state in STATE_SENATE_CLASSES:
            # Check if either seat in the state is up
            classes = STATE_SENATE_CLASSES[state]
            return any(is_senate_seat_on_ballot(c, election_year) for c in classes)
        else:
            raise ValueError("For Senate, must provide senate_class or state")
    
    else:
        raise ValueError(f"Unknown office: {office}")


# =============================================================================
# ELECTION METADATA
# =============================================================================

ELECTION_METADATA = {
    2008: {
        'type': 'presidential',
        'president_on_ballot': True,
        'candidates': {'D': 'Barack Obama', 'R': 'John McCain'},
        'winner': 'D',
        'description': '56th Presidential Election'
    },
    2010: {
        'type': 'midterm',
        'president_on_ballot': False,
        'description': 'Midterm elections for 112th Congress'
    },
    2012: {
        'type': 'presidential',
        'president_on_ballot': True,
        'candidates': {'D': 'Barack Obama', 'R': 'Mitt Romney'},
        'winner': 'D',
        'description': '57th Presidential Election'
    },
    2014: {
        'type': 'midterm',
        'president_on_ballot': False,
        'description': 'Midterm elections for 114th Congress'
    },
    2016: {
        'type': 'presidential',
        'president_on_ballot': True,
        'candidates': {'D': 'Hillary Clinton', 'R': 'Donald Trump'},
        'winner': 'R',
        'description': '58th Presidential Election'
    },
    2018: {
        'type': 'midterm',
        'president_on_ballot': False,
        'description': 'Midterm elections for 116th Congress'
    },
    2020: {
        'type': 'presidential',
        'president_on_ballot': True,
        'candidates': {'D': 'Joe Biden', 'R': 'Donald Trump'},
        'winner': 'D',
        'description': '59th Presidential Election'
    },
    2022: {
        'type': 'midterm',
        'president_on_ballot': False,
        'description': 'Midterm elections for 118th Congress'
    },
    2024: {
        'type': 'presidential',
        'president_on_ballot': True,
        'candidates': {'D': 'Kamala Harris', 'R': 'Donald Trump'},
        'winner': 'R',
        'description': '60th Presidential Election'
    },
}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_study_cycles(start_year: int = 2008, end_year: int = 2024) -> List[Dict]:
    """
    Get all election cycles for the study period with metadata.
    
    Returns list of dictionaries with cycle info.
    """
    cycles = []
    
    for year in range(start_year, end_year + 1, 2):
        if year % 2 != 0:
            continue
            
        cycle_start, cycle_end = get_cycle_boundaries(year)
        windows = get_cycle_windows(year)
        meta = ELECTION_METADATA.get(year, {})
        
        cycles.append({
            'election_year': year,
            'election_day': get_election_day(year),
            'cycle_start': cycle_start,
            'cycle_end': cycle_end,
            'type': meta.get('type', 'unknown'),
            'president_on_ballot': meta.get('president_on_ballot', False),
            'windows': windows
        })
    
    return cycles


if __name__ == '__main__':
    # Test the utilities
    print("=" * 70)
    print("ELECTION CYCLE UTILITIES TEST")
    print("=" * 70)
    
    # Test Election Day calculation
    print("\nElection Days:")
    for year in range(2014, 2026, 2):
        ed = get_election_day(year)
        print(f"  {year}: {ed.strftime('%A, %B %d, %Y')}")
    
    # Test cycle windows
    print("\n2024 Cycle Windows (L=60, W=30):")
    windows = get_cycle_windows(2024, pre_election_days=60, washout_days=30)
    for name, (start, end) in windows.items():
        if start == end:
            print(f"  {name}: {start}")
        else:
            print(f"  {name}: {start} to {end}")
    
    # Test Senate class staggering
    print("\nSenate Class Elections (2014-2024):")
    for cls in [1, 2, 3]:
        years = get_senate_class_election_years(cls, 2014, 2024)
        print(f"  Class {cls}: {years}")
    
    # Test on_ballot indicator
    print("\nOn-Ballot Tests:")
    print(f"  President 2024: {get_on_ballot_indicator('P', 2024)}")
    print(f"  President 2022: {get_on_ballot_indicator('P', 2022)}")
    print(f"  House 2024: {get_on_ballot_indicator('H', 2024)}")
    print(f"  Senate Class 1 in 2024: {get_on_ballot_indicator('S', 2024, senate_class=1)}")
    print(f"  Senate Class 1 in 2022: {get_on_ballot_indicator('S', 2022, senate_class=1)}")
    print(f"  Senate Class 3 in 2022: {get_on_ballot_indicator('S', 2022, senate_class=3)}")
    
    print("\n" + "=" * 70)
    print("All tests complete!")

