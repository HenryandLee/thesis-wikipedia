# US Senate Elections Scraper (2008-2024)

**File:** `scripts/012_build_senate_roster.py`

Scrapes US Senate election candidate data from Wikipedia for all election cycles from 2008-2024.

---

## Usage

**Scrape a specific year:**
```bash
python scripts/012_build_senate_roster.py 2024
```

**Scrape all years (2008-2024):**
```bash
python scripts/012_build_senate_roster.py all
```

**Output location:** `data/raw/html_parsed_candidates/{year}_senate_candidates.csv`

---

## Summary Statistics

| Year | Total Candidates | General Election | Special Election | States | Races | With PVI |
|------|-----------------|------------------|------------------|--------|-------|----------|
| 2008 | 117 | 113 | 4 | 33 | 35 | 0 |
| 2010 | 138 | 118 | 20 | 36 | 38 | 0 |
| 2012 | 115 | 115 | 0 | 33 | 33 | 0 |
| 2014 | 134 | 125 | 9 | 34 | 36 | 134 |
| 2016 | 106 | 106 | 0 | 34 | 34 | 106 |
| 2018 | 119 | 113 | 6 | 33 | 35 | 119 |
| 2020 | 136 | 132 | 4 | 34 | 35 | 136 |
| 2022 | 137 | 131 | 6 | 34 | 36 | 137 |
| 2024 | 118 | 114 | 4 | 33 | 35 | 118 |

**Important Statistical Notes:**
- **PVI data availability:** Only 2014-2024 have PVI (Cook Partisan Voting Index) data directly from Wikipedia; not 2008-2012
- **Election cycle variation:** Senate races vary by year (33-36 races) due to staggered 6-year terms and Class rotation
- **Special elections:** Occur when incumbents resign/die; can happen in same states as regular elections (e.g., 2022 Oklahoma had both)
- **Winners > Races:** Some states use ranked-choice voting (Alaska) creating multiple winner entries per race
- **Class system:** Senate seats divided into Class 1 (33 seats), Class 2 (33 seats), Class 3 (34 seats), each up for election every 6 years

---

## Output Data Schema

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `state` | string | State name | "Arizona" |
| `incumbent_name` | string | Current incumbent senator | "Kyrsten Sinema" |
| `incumbent_party` | string | Incumbent's party | "Democratic" |
| `electoral_history` | string | Years incumbent was elected | "2018" |
| `result_text` | string | Race outcome description | "Incumbent re-elected." |
| `candidate_name` | string | Full candidate name | "Ruben Gallego" |
| `party` | string | Candidate's political party | "Democratic" |
| `vote_percentage` | float | Vote percentage | 50.06 |
| `is_winner` | boolean | Whether candidate won | True |
| `pvi` | string | Cook Partisan Voting Index | "R+2" |
| `last_election_result` | string | Previous election result | "51.2% D" |
| `wikipedia_url` | string | Candidate Wikipedia page URL | "https://en.wikipedia.org/..." |
| `election_type` | string | "general" or "special" | "general" |

---

## Election Type Values

| Value | Meaning |
|-------|---------|
| `general` | Regular November general election for the scheduled Senate class |
| `special` | Special election to fill vacancy from resignation, death, or appointment |

**Example - 2022 Oklahoma (Both Special and General):**
```csv
Oklahoma,James Lankford,Republican,"2014, (special), 2016",Incumbent re-elected.,James Lankford,Republican,64.3,True,R+20,67.7% R,...,general
Oklahoma,Jim Inhofe,Republican,"1994, (special), 1996, 2002, 2008, 2014, 2020","Incumbent resigned...",Markwayne Mullin,Republican,61.8,True,R+20,62.9% R(2020),...,special
```

---

## Technical Implementation

### Table Format Variations by Year

The scraper handles Wikipedia table structure variations across years:

- **2014-2024**: Standard format with PVI column
  - Table structure: State, PVI, Incumbent, Last Election, Ratings, Candidates
  - Parse format: `'2022'` (2014, 2016, 2018, 2020, 2022 use this)
  - Parse format: `'2024'` (2024 uses this)

- **2008-2012**: No PVI data
  - Table structure: State, Incumbent, Last Election, Ratings, Candidates
  - Parse format: `'2012'` (2008, 2010, 2012 use this)
  - PVI column left empty in output

### Flexible PVI Matching

The scraper includes robust PVI matching logic to handle Wikipedia inconsistencies:
- Exact state name match
- State + election type suffix (e.g., "Oklahoma(special)")
- Handles "regular" vs "general" naming variations
- Fuzzy fallback for typos (e.g., "South Carolinaspecial)" missing opening parenthesis)

---

## Data Coverage

**Total:** 1,120 Senate candidates across 9 election cycles (2008-2024)

**Geographic coverage:** All 50 states (varies by election year based on Senate class rotation)

**Special elections:**
| Year | Number of Special Races | States                      |
|------|------------------------|-----------------------------|
| 2008 | 2                      | MS, WY                      |
| 2010 | 5                      | MA, DE, IL, NY, WV          |
| 2012 | 0                      |                             |
| 2014 | 3                      | HI, OK, SC                  |
| 2016 | 0                      |                             |
| 2018 | 2                      | MN, MS                      |
| 2020 | 2                      | AZ, GA                      |
| 2022 | 2                      | CA, OK                      |
| 2024 | 2                      | CA, NE                      |
