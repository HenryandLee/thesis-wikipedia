## Script 1: 01_build_candidate_roster.py
Output: data/raw/
Files created:
  * fec_roster_raw.json          (overwrites)
  * fec_roster_matched.json      (overwrites)
  * candidate_roster_final.json  (overwrites)
  * page_list_candidates.json    (overwrites)

Cleanup: Automatically overwrites JSON files

## Script 2: 02_collect_revisions.py --config candidates
Output directory: data/raw/revisions_candidates/
Files created:
  * One .json file per Wikipedia page
  * Summary file: 
    collection_summary_candidates.json

Cleanup: Clears directory first

## Script 3: 03_process_data.py --config candidates
Output: data/processed_candidates/
Files created:
  * daily_edits.csv
  * page_statistics.csv
  * all_revisions.csv
  * editor_profiles.csv

Cleanup: Clears directory first

## Procedure
Run: 
```bash
python scripts/01_build_candidate_roster.py --major-only --sample 0.1 --seed [int]
python scripts/02_collect_revisions.py --config candidates
python scripts/03_process_data.py --config candidates
```
