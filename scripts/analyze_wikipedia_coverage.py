"""
Analyze Wikipedia coverage - what percentage of on-ballot candidates have Wikipedia pages
"""
import pandas as pd
import glob
from pathlib import Path

base_dir = Path(__file__).parent.parent
raw_dir = base_dir / 'data' / 'raw' / 'html_parsed_candidates'

print("=" * 80)
print("WIKIPEDIA COVERAGE ANALYSIS")
print("=" * 80)
print("\nIMPORTANT: Our dataset contains only candidates WITH Wikipedia pages.")
print("This represents a subset of all on-ballot candidates.\n")

# Analyze House coverage
print("\n" + "=" * 80)
print("HOUSE CANDIDATES - Wikipedia Coverage by Election Cycle")
print("=" * 80)
print(f"{'Year':<6} {'With Wiki':>10} {'Total':>10} {'Coverage':>12} {'Missing':>10}")
print("-" * 80)

house_stats = []
for csv_file in sorted(glob.glob(str(raw_dir / '*house*.csv'))):
    year = int(csv_file.split('/')[-1].split('_')[0])
    df = pd.read_csv(csv_file)
    total = len(df)
    with_wiki = df['wikipedia_url'].notna().sum()
    without_wiki = total - with_wiki
    coverage = 100 * with_wiki / total if total > 0 else 0

    house_stats.append({
        'year': year,
        'with_wiki': with_wiki,
        'total': total,
        'without_wiki': without_wiki,
        'coverage_pct': coverage
    })

    print(f"{year:<6} {with_wiki:>10,} {total:>10,} {coverage:>11.1f}% {without_wiki:>10,}")

df_house = pd.DataFrame(house_stats)
print("-" * 80)
print(f"{'TOTAL':<6} {df_house['with_wiki'].sum():>10,} {df_house['total'].sum():>10,} "
      f"{100*df_house['with_wiki'].sum()/df_house['total'].sum():>11.1f}% "
      f"{df_house['without_wiki'].sum():>10,}")

# Analyze Senate coverage
print("\n" + "=" * 80)
print("SENATE CANDIDATES - Wikipedia Coverage by Election Cycle")
print("=" * 80)
print(f"{'Year':<6} {'With Wiki':>10} {'Total':>10} {'Coverage':>12} {'Missing':>10}")
print("-" * 80)

senate_stats = []
for csv_file in sorted(glob.glob(str(raw_dir / '*senate*.csv'))):
    year = int(csv_file.split('/')[-1].split('_')[0])
    df = pd.read_csv(csv_file)
    total = len(df)
    with_wiki = df['wikipedia_url'].notna().sum()
    without_wiki = total - with_wiki
    coverage = 100 * with_wiki / total if total > 0 else 0

    senate_stats.append({
        'year': year,
        'with_wiki': with_wiki,
        'total': total,
        'without_wiki': without_wiki,
        'coverage_pct': coverage
    })

    print(f"{year:<6} {with_wiki:>10,} {total:>10,} {coverage:>11.1f}% {without_wiki:>10,}")

df_senate = pd.DataFrame(senate_stats)
print("-" * 80)
print(f"{'TOTAL':<6} {df_senate['with_wiki'].sum():>10,} {df_senate['total'].sum():>10,} "
      f"{100*df_senate['with_wiki'].sum()/df_senate['total'].sum():>11.1f}% "
      f"{df_senate['without_wiki'].sum():>10,}")

# Combined statistics
print("\n" + "=" * 80)
print("OVERALL STATISTICS")
print("=" * 80)

total_all = df_house['total'].sum() + df_senate['total'].sum()
with_wiki_all = df_house['with_wiki'].sum() + df_senate['with_wiki'].sum()
without_wiki_all = total_all - with_wiki_all

print(f"\nTotal on-ballot candidates (2008-2024): {total_all:,}")
print(f"  - With Wikipedia pages: {with_wiki_all:,} ({100*with_wiki_all/total_all:.1f}%)")
print(f"  - Without Wikipedia pages: {without_wiki_all:,} ({100*without_wiki_all/total_all:.1f}%)")
print(f"\nHouse: {df_house['with_wiki'].sum():,} / {df_house['total'].sum():,} "
      f"({100*df_house['with_wiki'].sum()/df_house['total'].sum():.1f}%)")
print(f"Senate: {df_senate['with_wiki'].sum():,} / {df_senate['total'].sum():,} "
      f"({100*df_senate['with_wiki'].sum()/df_senate['total'].sum():.1f}%)")

# Save coverage data
output_dir = base_dir / 'data' / 'processed_html_parsed'
df_house.to_csv(output_dir / 'wikipedia_coverage_house.csv', index=False)
df_senate.to_csv(output_dir / 'wikipedia_coverage_senate.csv', index=False)

print(f"\n✓ Coverage statistics saved to:")
print(f"  - {output_dir / 'wikipedia_coverage_house.csv'}")
print(f"  - {output_dir / 'wikipedia_coverage_senate.csv'}")

# Key insights
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print("\n1. SAMPLE REPRESENTATION:")
print(f"   Our dataset captures approximately HALF of all on-ballot candidates")
print(f"   ({100*with_wiki_all/total_all:.1f}% overall coverage)")

print("\n2. COVERAGE VARIATION:")
print(f"   House coverage ranges: {df_house['coverage_pct'].min():.1f}% - {df_house['coverage_pct'].max():.1f}%")
print(f"   Senate coverage ranges: {df_senate['coverage_pct'].min():.1f}% - {df_senate['coverage_pct'].max():.1f}%")
print(f"   Senate has slightly better coverage than House")

print("\n3. SELECTION BIAS:")
print("   Candidates WITH Wikipedia pages are likely:")
print("   - More prominent/well-known")
print("   - Better-funded campaigns")
print("   - Major party candidates (vs. minor parties)")
print("   - Incumbents or serious challengers")
print("   - NOT representative of all candidates")

print("\n4. IMPLICATION FOR ANALYSIS:")
print("   All findings apply to 'Wikipedia-notable' candidates only")
print("   Results cannot be generalized to all on-ballot candidates")
print("   This is a selective sample of more prominent campaigns")

print("\n" + "=" * 80)
