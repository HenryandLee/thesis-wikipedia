"""
Generate processing_summary.md from existing processed data files
"""
import pandas as pd
from pathlib import Path
from datetime import datetime

# Paths
base_dir = Path(__file__).parent.parent
output_dir = base_dir / 'data' / 'processed_html_parsed'

# Check if processed data exists
if not output_dir.exists():
    print(f"Error: {output_dir} does not exist")
    print("Please run 031_process_data.py first")
    exit(1)

print("Loading processed data files...")

# Load datasets
df_revisions = pd.read_csv(output_dir / 'all_revisions.csv', parse_dates=['timestamp'])
df_cycle_stats = pd.read_csv(output_dir / 'candidate_cycle_stats.csv')
df_daily = pd.read_csv(output_dir / 'daily_edits.csv', parse_dates=['date'])
df_daily_house = pd.read_csv(output_dir / 'daily_edits_house.csv', parse_dates=['date'])
df_daily_senate = pd.read_csv(output_dir / 'daily_edits_senate.csv', parse_dates=['date'])
df_editors = pd.read_csv(output_dir / 'editor_profiles.csv')

print(f"✓ Loaded {len(df_revisions):,} revisions")
print(f"✓ Loaded {len(df_cycle_stats):,} candidate-cycle records")

# Calculate statistics
print("\nCalculating statistics...")

office_counts = df_revisions.groupby('office')['revid'].count()
cycle_counts = df_revisions.groupby('election_cycle')['revid'].count().sort_index()
party_counts = df_revisions.groupby('party')['revid'].count().sort_values(ascending=False)
state_counts = df_revisions.groupby('state')['revid'].count().sort_values(ascending=False)

# Candidate counts per cycle
candidates_per_cycle = df_cycle_stats.groupby('election_cycle')['page_title'].nunique().sort_index()

# Office and party breakdown per cycle
cycle_office_counts = df_cycle_stats.groupby(['election_cycle', 'office'])['page_title'].nunique().unstack(fill_value=0)
cycle_party_counts = df_cycle_stats.groupby(['election_cycle', 'party'])['page_title'].nunique().unstack(fill_value=0)

# Estimate filtering statistics (since we don't have the raw stats from processing)
# We can approximate based on the data we have
total_revisions_kept = len(df_revisions)

# Generate markdown summary
print("Generating summary markdown...")

summary_path = output_dir / 'processing_summary.md'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("# Data Processing Summary\n\n")
    f.write(f"**Processing Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("*Note: This summary was generated from existing processed files.*\n\n")
    f.write("---\n\n")

    # Add critical disclaimer
    f.write("## ⚠️ CRITICAL: Sample Composition\n\n")
    f.write("**This dataset contains ONLY candidates with Wikipedia pages.**\n\n")
    f.write("This represents approximately **50.2%** of all on-ballot candidates (2008-2024).\n\n")

    # Load and display coverage stats
    df_cov_house = pd.read_csv(output_dir / 'wikipedia_coverage_house.csv')
    df_cov_senate = pd.read_csv(output_dir / 'wikipedia_coverage_senate.csv')

    house_total = df_cov_house['total'].sum()
    house_wiki = df_cov_house['with_wiki'].sum()
    senate_total = df_cov_senate['total'].sum()
    senate_wiki = df_cov_senate['with_wiki'].sum()

    f.write("| Office | With Wikipedia | Total On-Ballot | Coverage |\n")
    f.write("|--------|---------------:|----------------:|---------:|\n")
    f.write(f"| House | {house_wiki:,} | {house_total:,} | {100*house_wiki/house_total:.1f}% |\n")
    f.write(f"| Senate | {senate_wiki:,} | {senate_total:,} | {100*senate_wiki/senate_total:.1f}% |\n")
    f.write(f"| **Total** | **{house_wiki+senate_wiki:,}** | **{house_total+senate_total:,}** | **{100*(house_wiki+senate_wiki)/(house_total+senate_total):.1f}%** |\n\n")

    f.write("**Selection Bias:** Our sample over-represents:\n")
    f.write("- More prominent/well-known candidates\n")
    f.write("- Better-funded campaigns\n")
    f.write("- Major party candidates (Democratic/Republican)\n")
    f.write("- Incumbents and serious challengers\n\n")

    f.write("**Missing:** Lesser-known candidates, minor parties, minimal campaigns\n\n")
    f.write("**⚠️ All findings apply to 'Wikipedia-notable' candidates only!**\n\n")
    f.write("---\n\n")

    # Overview
    f.write("## Dataset Overview\n\n")
    f.write(f"- **Total Revisions**: {len(df_revisions):,}\n")
    f.write(f"- **Unique Candidates**: {df_revisions['page_title'].nunique():,}\n")
    f.write(f"- **Unique Editors**: {df_revisions['user'].nunique():,}\n")
    f.write(f"- **Date Range**: {df_revisions['timestamp'].min().strftime('%Y-%m-%d')} to {df_revisions['timestamp'].max().strftime('%Y-%m-%d')}\n")
    f.write(f"- **Election Cycles Covered**: {', '.join(map(str, sorted(cycle_counts.index)))}\n\n")

    # Edit types
    f.write("---\n\n")
    f.write("## Edit Types\n\n")
    f.write("| Type | Count | Percentage |\n")
    f.write("|------|------:|:----------:|\n")
    f.write(f"| Anonymous edits | {df_revisions['is_anonymous'].sum():,} | {df_revisions['is_anonymous'].mean()*100:.1f}% |\n")
    f.write(f"| Bot edits | {df_revisions['is_bot'].sum():,} | {df_revisions['is_bot'].mean()*100:.1f}% |\n")
    f.write(f"| Minor edits | {df_revisions['minor'].sum():,} | {df_revisions['minor'].mean()*100:.1f}% |\n\n")

    # By office
    f.write("---\n\n")
    f.write("## Distribution by Office\n\n")
    f.write("| Office | Revisions | Percentage | Candidates |\n")
    f.write("|--------|----------:|:----------:|:----------:|\n")
    for office in ['House', 'Senate']:
        if office in office_counts:
            count = office_counts[office]
            num_candidates = df_cycle_stats[df_cycle_stats['office'] == office]['page_title'].nunique()
            f.write(f"| {office} | {count:,} | {100*count/len(df_revisions):.1f}% | {num_candidates} |\n")
    f.write("\n")

    # By election cycle
    f.write("---\n\n")
    f.write("## Distribution by Election Cycle\n\n")
    f.write("| Cycle | Revisions | Percentage | Candidates | House | Senate |\n")
    f.write("|------:|----------:|:----------:|:----------:|------:|-------:|\n")
    for cycle, count in cycle_counts.items():
        num_candidates = candidates_per_cycle.get(cycle, 0)
        house = cycle_office_counts.loc[cycle, 'House'] if cycle in cycle_office_counts.index and 'House' in cycle_office_counts.columns else 0
        senate = cycle_office_counts.loc[cycle, 'Senate'] if cycle in cycle_office_counts.index and 'Senate' in cycle_office_counts.columns else 0
        f.write(f"| {cycle} | {count:,} | {100*count/len(df_revisions):.1f}% | {num_candidates} | {house} | {senate} |\n")
    f.write("\n")

    # By party
    f.write("---\n\n")
    f.write("## Distribution by Party\n\n")
    f.write("| Party | Revisions | Percentage | Candidates |\n")
    f.write("|-------|----------:|:----------:|:----------:|\n")
    for party, count in party_counts.head(5).items():
        num_candidates = df_cycle_stats[df_cycle_stats['party'] == party]['page_title'].nunique()
        f.write(f"| {party} | {count:,} | {100*count/len(df_revisions):.1f}% | {num_candidates} |\n")
    f.write("\n")

    # By state
    f.write("---\n\n")
    f.write("## Top 15 States by Revision Count\n\n")
    f.write("| Rank | State | Revisions | Percentage | Candidates |\n")
    f.write("|-----:|-------|----------:|:----------:|:----------:|\n")
    for i, (state, count) in enumerate(state_counts.head(15).items(), 1):
        num_candidates = df_cycle_stats[df_cycle_stats['state'] == state]['page_title'].nunique()
        f.write(f"| {i} | {state} | {count:,} | {100*count/len(df_revisions):.1f}% | {num_candidates} |\n")
    f.write("\n")

    # Top candidates
    f.write("---\n\n")
    f.write("## Top 15 Most Edited Candidates\n\n")
    f.write("| Rank | Candidate | Total Revisions | Unique Editors |\n")
    f.write("|-----:|-----------|----------------:|---------------:|\n")
    top_15_candidates = df_cycle_stats.groupby('page_title').agg({
        'total_revisions': 'sum',
        'unique_editors': 'sum'
    }).sort_values('total_revisions', ascending=False).head(15)

    for i, (candidate, row) in enumerate(top_15_candidates.iterrows(), 1):
        f.write(f"| {i} | {candidate.replace('_', ' ')} | {row['total_revisions']:,} | {row['unique_editors']:,} |\n")
    f.write("\n")

    # Files generated
    f.write("---\n\n")
    f.write("## Generated Files\n\n")
    f.write("| File | Rows | Description |\n")
    f.write("|------|-----:|:------------|\n")
    f.write(f"| `all_revisions.csv` | {len(df_revisions):,} | All revisions with metadata |\n")
    f.write(f"| `daily_edits.csv` | {len(df_daily):,} | Daily aggregates |\n")
    f.write(f"| `daily_edits_house.csv` | {len(df_daily_house):,} | Daily aggregates (House only) |\n")
    f.write(f"| `daily_edits_senate.csv` | {len(df_daily_senate):,} | Daily aggregates (Senate only) |\n")
    f.write(f"| `candidate_cycle_stats.csv` | {len(df_cycle_stats):,} | Candidate-cycle statistics |\n")
    f.write(f"| `editor_profiles.csv` | {len(df_editors):,} | Editor profiles |\n")
    f.write("\n")

print(f"\n✓ Summary saved to: {summary_path}")
print(f"\nDone! You can now run the notebook.")
