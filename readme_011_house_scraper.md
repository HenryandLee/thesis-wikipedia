# US House Elections Scraper (2008-2024)

**File:** `scripts/011_build_house_roster_backend.py`

Scrapes US House election candidate data from Wikipedia with year-specific parsers for 2008-2024.

---

## Usage

**Scrape a specific year:**
```bash
python scripts/011_build_house_roster_backend.py 2024
```

**Scrape all years (2008-2024):**
```bash
python scripts/011_build_house_roster_backend.py
```

**Output location:** `data/raw/html_parsed_candidates/{year}_house_candidates.csv`

---

## Summary Statistics

| Year | Total Candidates | General Election | Special Election | Districts | Winners |
|------|-----------------|------------------|------------------|-----------|---------|
| 2008 | 1,149 | 1,124 | 25 | 435 | 443 |
| 2010 | 1,298 | 1,282 | 16 | 435 | 442 |
| 2012 | 1,195 | 1,179 | 16 | 435 | 441 |
| 2014 | 1,128 | 1,111 | 17 | 435 | 445 |
| 2016 | 1,131 | 1,122 | 9 | 435 | 443 |
| 2018 | 1,125 | 1,103 | 22 | 435 | 447 |
| 2020 | 1,109 | 1,099 | 10 | 435 | 441 |
| 2022 | 1,086 | 1,057 | 29 | 435 | 450 |
| 2024 | 1,089 | 1,068 | 21 | 435 | 449 |

**Notes:**
- All years have exactly 435 districts (50 states, excludes non-voting territories)
- Winners > 435 due to special elections and RCV rounds
- Each year uses a custom parser to handle Wikipedia table format variations

---

## Output Data Schema

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `state` | string | State name | "Alaska" |
| `district` | string | District identifier | "Alaska at-large" |
| `candidate_name` | string | Full candidate name | "Mary Peltola" |
| `party` | string | Political party | "Democratic" |
| `vote_percentage` | float | Vote percentage | 55.0 |
| `is_incumbent` | boolean | Whether candidate is incumbent | True |
| `is_winner` | boolean | Whether candidate won | True |
| `pvi` | string | Cook Partisan Voting Index | "R+8" |
| `election_type` | string | Election classification | "general_rcv_runoff" |
| `wikipedia_url` | string | Candidate Wikipedia page URL | "https://en.wikipedia.org/wiki/..." |

---

## Election Type Values

| Value | Meaning |
|-------|---------|
| `general` | Regular November general election |
| `special` | Special election (non-RCV) |
| `general_rcv_round1` | General election RCV first round |
| `general_rcv_runoff` | General election RCV final runoff |
| `special_rcv_round1` | Special election RCV first round |
| `special_rcv_runoff` | Special election RCV final runoff |

**Example - Alaska 2022 (RCV Special + General):**
```csv
Alaska,Alaska at-large,Mary Peltola,Democratic,39.6,False,False,R+8,special_rcv_round1,...
Alaska,Alaska at-large,Mary Peltola,Democratic,51.5,False,True,R+8,special_rcv_runoff,...
Alaska,Alaska at-large,Mary Peltola,Democratic,48.8,True,False,R+8,general_rcv_round1,...
Alaska,Alaska at-large,Mary Peltola,Democratic,55.0,True,True,R+8,general_rcv_runoff,...
```

---

## Year-Specific Parser Details

Each year has a custom parser to handle Wikipedia table format variations:

- **2008**: Dual format (7-cell with PVI for AL/AK/AZ/AR, 6-cell without PVI for others, 4-cell for redistricted districts)
- **2010**: 7-cell and 5-cell (redistricted districts)
- **2012**: 7-cell, 5-cell (redistricted districts), 4-cell continuation rows
- **2014**: 7-cell, 5-cell (redistricted districts), 4-cell continuation rows
- **2016**: 7-cell, 5-cell (redistricted districts), 4-cell continuation rows
- **2018**: 7-cell and 5-cell (redistricted districts)
- **2020**: 7-cell and 5-cell (redistricted districts)
- **2022**: 7-cell and 5-cell (redistricted districts)
- **2024**: 7-cell and 5-cell (redistricted districts)
