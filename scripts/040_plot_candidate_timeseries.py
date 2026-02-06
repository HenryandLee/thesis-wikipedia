"""
Script 040: Plot weekly edit time series for individual House candidates.

Produces two figures (4 subplots each, 2x2 grid):
1. 4 randomly selected House candidates (with ≥600 total edits)
2. 4 House candidates with the most total edits (ranked by edit counts only for 
   periods when they were on the ballot. NOT true all-time ranking.)

Each subplot:
- Green shaded regions: election cycles when candidate ran for House
- Weekly edit counts (Tuesday-Monday weeks), all edits 2006-2024
- Election days (vertical red dashed lines)
- Y-axis starts at 0

Week aggregation: Tuesday-Monday so election day (always Tuesday) starts a new week
and pre-election editing is cleanly separated.

Data source: Raw JSON files from data/raw/revisions_html_parsed/
(This shows ALL edits, not just during on-ballot periods)
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import numpy as np
import json
from datetime import timedelta

base_dir = Path(__file__).parent.parent
raw_revisions_dir = base_dir / 'data' / 'raw' / 'revisions_html_parsed'
processed_dir = base_dir / 'data' / 'processed_html_parsed'
figures_dir = base_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

SEED = 42
MIN_EDITS_FOR_RANDOM = 600

# -----------------------------------------------------------------------------
# Load candidate metadata (to know which are House candidates and their cycles)
# -----------------------------------------------------------------------------
print("Loading candidate metadata from processed data...")
df_meta = pd.read_csv(
    processed_dir / 'all_revisions.csv',
    usecols=['page_title', 'office', 'election_cycle'],
    low_memory=False,
)
df_meta = df_meta[df_meta['office'] == 'House'].copy()

# Map candidate to their election cycles (when they ran for House)
candidate_cycles = df_meta.groupby('page_title')['election_cycle'].apply(lambda x: sorted(x.unique())).to_dict()

# Get list of House candidates
house_candidates = set(candidate_cycles.keys())
print(f"Found {len(house_candidates)} House candidates")

# -----------------------------------------------------------------------------
# Get total edit counts from processed data (for candidate selection)
# -----------------------------------------------------------------------------
print("Computing total edits per candidate from processed data...")
df_processed = pd.read_csv(
    processed_dir / 'all_revisions.csv',
    usecols=['page_title', 'office'],
    low_memory=False,
)
df_processed = df_processed[df_processed['office'] == 'House']
totals_processed = df_processed.groupby('page_title').size().sort_values(ascending=False)

# Determine which candidates to plot (4 random + 4 top + 10 more random)
rng = np.random.default_rng(SEED)
eligible = totals_processed[totals_processed >= MIN_EDITS_FOR_RANDOM].index.tolist()
print(f"  {len(eligible)} candidates with >= {MIN_EDITS_FOR_RANDOM} edits (from processed data)")
random_4 = list(rng.choice(eligible, size=4, replace=False))
top_4 = totals_processed.head(4).index.tolist()
# Select 10 more random candidates (different from random_4)
remaining_eligible = [c for c in eligible if c not in random_4]
random_10 = list(rng.choice(remaining_eligible, size=10, replace=False))
candidates_to_plot = list(set(random_4 + top_4 + random_10))
print(f"  Will plot {len(candidates_to_plot)} unique candidates")

# -----------------------------------------------------------------------------
# Load raw JSON revisions only for candidates we'll plot
# -----------------------------------------------------------------------------
print("Loading ALL revisions from raw JSON files (for selected candidates only)...")

def load_raw_revisions(page_title: str) -> pd.DataFrame:
    """Load all revisions for a candidate from raw JSON file."""
    json_path = raw_revisions_dir / f"{page_title}.json"
    if not json_path.exists():
        return pd.DataFrame()
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data.get('revisions'):
        return pd.DataFrame()
    
    records = []
    for rev in data['revisions']:
        records.append({
            'page_title': page_title,
            'timestamp': pd.to_datetime(rev['timestamp']),
        })
    
    return pd.DataFrame(records)

all_revisions = []
for pt in candidates_to_plot:
    df_cand = load_raw_revisions(pt)
    if len(df_cand) > 0:
        all_revisions.append(df_cand)
        print(f"  Loaded {pt}: {len(df_cand):,} revisions")

df = pd.concat(all_revisions, ignore_index=True)
print(f"Loaded {len(df):,} total revisions from {len(candidates_to_plot)} candidates")

df['date'] = df['timestamp'].dt.tz_localize(None).dt.normalize()

# -----------------------------------------------------------------------------
# Weekly aggregation (Tuesday-Monday weeks)
# -----------------------------------------------------------------------------
def get_week_start_tuesday(date):
    """Get the Tuesday that starts the week containing this date.
    Weeks run Tuesday-Monday so election day (Tuesday) starts a new week."""
    # Monday=0, Tuesday=1, ..., Sunday=6
    days_since_tuesday = (date.weekday() - 1) % 7
    return date - timedelta(days=days_since_tuesday)

df['week_start'] = df['date'].apply(get_week_start_tuesday)

# Weekly edit counts per candidate
weekly = df.groupby(['page_title', 'week_start']).size().reset_index(name='edits')

# Total edits per candidate (from raw data - for labeling)
totals = weekly.groupby('page_title')['edits'].sum().sort_values(ascending=False)

# Readable name from page title
def readable(pt):
    return pt.replace('_', ' ').split(' (')[0]

# Election days for vertical reference lines
ELECTION_DAYS = pd.to_datetime([
    '2006-11-07', '2008-11-04', '2010-11-02', '2012-11-06',
    '2014-11-04', '2016-11-08', '2018-11-06', '2020-11-03',
    '2022-11-08', '2024-11-05',
])

# Cycle date ranges: each cycle runs from day after previous election to this election
CYCLE_RANGES = {
    2008: (pd.Timestamp('2006-11-08'), pd.Timestamp('2008-11-04')),
    2010: (pd.Timestamp('2008-11-05'), pd.Timestamp('2010-11-02')),
    2012: (pd.Timestamp('2010-11-03'), pd.Timestamp('2012-11-06')),
    2014: (pd.Timestamp('2012-11-07'), pd.Timestamp('2014-11-04')),
    2016: (pd.Timestamp('2014-11-05'), pd.Timestamp('2016-11-08')),
    2018: (pd.Timestamp('2016-11-09'), pd.Timestamp('2018-11-06')),
    2020: (pd.Timestamp('2018-11-07'), pd.Timestamp('2020-11-03')),
    2022: (pd.Timestamp('2020-11-04'), pd.Timestamp('2022-11-08')),
    2024: (pd.Timestamp('2022-11-09'), pd.Timestamp('2024-11-05')),
}

# Generate all Tuesday week starts for the full date range
all_weeks = pd.date_range('2006-10-31', '2024-12-31', freq='W-TUE')  # W-TUE = weeks starting Tuesday

# -----------------------------------------------------------------------------
# Plotting helper (static)
# -----------------------------------------------------------------------------
def plot_grid(page_titles, suptitle, filename):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    axes = axes.flatten()

    for ax, pt in zip(axes, page_titles[:4]):
        # Get candidate weekly data
        cand = weekly[weekly['page_title'] == pt][['week_start', 'edits']].set_index('week_start').sort_index()
        edits_series = cand['edits'].reindex(all_weeks, fill_value=0)

        # Shade cycles when candidate ran for House
        cycles_ran = candidate_cycles.get(pt, [])
        for cycle in cycles_ran:
            if cycle in CYCLE_RANGES:
                start, end = CYCLE_RANGES[cycle]
                ax.axvspan(start, end, alpha=0.08, color='green', zorder=0)

        # Plot weekly data
        ax.plot(edits_series.index, edits_series.values,
                color='steelblue', linewidth=1.2, label='Weekly edits', zorder=2)

        # Election day markers (prominent)
        for ed in ELECTION_DAYS:
            ax.axvline(ed, color='red', alpha=0.5, linewidth=1.2, linestyle='--', zorder=3)

        # Formatting
        total = int(totals.get(pt, 0))
        ax.set_title(f"{readable(pt)}  ({total:,} edits)", fontsize=11, fontweight='bold')
        ax.set_ylabel('Edits / week', fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(labelsize=8)

        # Legend only on first subplot
        if ax == axes[0]:
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    fig.suptitle(suptitle, fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = figures_dir / filename
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")

# -----------------------------------------------------------------------------
# Figure 1: Random 4
# -----------------------------------------------------------------------------
print("Plotting random 4 candidates...")
plot_grid(random_4, "Weekly Wikipedia Edits (All Time, Tue-Mon) — 4 Random House Candidates (≥600 edits)", "candidate_timeseries_random4.png")

# -----------------------------------------------------------------------------
# Figure 2: Top 4
# -----------------------------------------------------------------------------
print("Plotting top 4 candidates...")
plot_grid(top_4, "Weekly Wikipedia Edits (All Time, Tue-Mon) — 4 Most-Edited House Candidates", "candidate_timeseries_top4.png")

# -----------------------------------------------------------------------------
# Interactive plots (Plotly) - zoomable HTML files
# -----------------------------------------------------------------------------
print("\nCreating interactive plots...")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    def plot_interactive(page_titles, title, filename):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f"{readable(pt)} ({int(totals.get(pt, 0)):,} edits)" for pt in page_titles[:4]],
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )
        
        # Axis references for each subplot position
        # (row, col) -> (xref, yref)
        axis_refs = {
            (1, 1): ('x', 'y'),
            (1, 2): ('x2', 'y2'),
            (2, 1): ('x3', 'y3'),
            (2, 2): ('x4', 'y4'),
        }
        
        for i, pt in enumerate(page_titles[:4]):
            row = i // 2 + 1
            col = i % 2 + 1
            xref, yref = axis_refs[(row, col)]
            
            # Get candidate weekly data
            cand = weekly[weekly['page_title'] == pt][['week_start', 'edits']].set_index('week_start').sort_index()
            edits_series = cand['edits'].reindex(all_weeks, fill_value=0)
            
            # Compute week end dates for hover
            week_ends = edits_series.index + timedelta(days=6)
            
            # Create hover text with week range
            hover_text = [
                f"Week: {ws.strftime('%b %d')} - {we.strftime('%b %d, %Y')}<br>Edits: {edits}"
                for ws, we, edits in zip(edits_series.index, week_ends, edits_series.values)
            ]
            
            # Weekly edits (plot first so shading goes on top layer behind)
            fig.add_trace(
                go.Scatter(
                    x=edits_series.index,
                    y=edits_series.values,
                    mode='lines',
                    name='Weekly edits',
                    line=dict(color='steelblue', width=1.5),
                    showlegend=(i == 0),
                    text=hover_text,
                    hovertemplate='%{text}<extra></extra>',
                ),
                row=row, col=col,
            )
            
            # Shade cycles when candidate ran (using shapes for reliability)
            cycles_ran = candidate_cycles.get(pt, [])
            for cycle in cycles_ran:
                if cycle in CYCLE_RANGES:
                    start, end = CYCLE_RANGES[cycle]
                    fig.add_shape(
                        type="rect",
                        x0=start, x1=end,
                        y0=0, y1=1,
                        xref=xref,
                        yref=yref + " domain",
                        fillcolor="green",
                        opacity=0.15,
                        layer="below",
                        line_width=0,
                    )
            
            # Election day lines (red dashed)
            for ed in ELECTION_DAYS:
                fig.add_shape(
                    type="line",
                    x0=ed, x1=ed,
                    y0=0, y1=1,
                    xref=xref,
                    yref=yref + " domain",
                    line=dict(color="red", width=1.5, dash="dash"),
                    opacity=0.6,
                )
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            height=800,
            width=1200,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        )
        
        fig.update_xaxes(
            tickformat='%b %Y',
            dtick='M12',
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor='gray',
            spikethickness=1,
        )
        fig.update_yaxes(rangemode='tozero', title_text='Edits / week')
        
        out = figures_dir / filename
        fig.write_html(out)
        print(f"Saved interactive: {out}")
    
    def plot_interactive_10(page_titles, title, filename):
        """Plot 10 candidates in a 5x2 grid."""
        fig = make_subplots(
            rows=5, cols=2,
            subplot_titles=[f"{readable(pt)} ({int(totals.get(pt, 0)):,} edits)" for pt in page_titles[:10]],
            vertical_spacing=0.06,
            horizontal_spacing=0.08,
        )
        
        # Axis references for each subplot position (5x2 grid)
        axis_refs = {}
        for idx in range(10):
            r = idx // 2 + 1
            c = idx % 2 + 1
            if idx == 0:
                axis_refs[(r, c)] = ('x', 'y')
            else:
                axis_refs[(r, c)] = (f'x{idx+1}', f'y{idx+1}')
        
        for i, pt in enumerate(page_titles[:10]):
            row = i // 2 + 1
            col = i % 2 + 1
            xref, yref = axis_refs[(row, col)]
            
            # Get candidate weekly data
            cand = weekly[weekly['page_title'] == pt][['week_start', 'edits']].set_index('week_start').sort_index()
            edits_series = cand['edits'].reindex(all_weeks, fill_value=0)
            
            # Compute week end dates for hover
            week_ends = edits_series.index + timedelta(days=6)
            
            # Create hover text with week range
            hover_text = [
                f"Week: {ws.strftime('%b %d')} - {we.strftime('%b %d, %Y')}<br>Edits: {edits}"
                for ws, we, edits in zip(edits_series.index, week_ends, edits_series.values)
            ]
            
            # Weekly edits (plot first so shading goes on top layer behind)
            fig.add_trace(
                go.Scatter(
                    x=edits_series.index,
                    y=edits_series.values,
                    mode='lines',
                    name='Weekly edits',
                    line=dict(color='steelblue', width=1.5),
                    showlegend=(i == 0),
                    text=hover_text,
                    hovertemplate='%{text}<extra></extra>',
                ),
                row=row, col=col,
            )
            
            # Shade cycles when candidate ran (using shapes for reliability)
            cycles_ran = candidate_cycles.get(pt, [])
            for cycle in cycles_ran:
                if cycle in CYCLE_RANGES:
                    start, end = CYCLE_RANGES[cycle]
                    fig.add_shape(
                        type="rect",
                        x0=start, x1=end,
                        y0=0, y1=1,
                        xref=xref,
                        yref=yref + " domain",
                        fillcolor="green",
                        opacity=0.15,
                        layer="below",
                        line_width=0,
                    )
            
            # Election day lines (red dashed)
            for ed in ELECTION_DAYS:
                fig.add_shape(
                    type="line",
                    x0=ed, x1=ed,
                    y0=0, y1=1,
                    xref=xref,
                    yref=yref + " domain",
                    line=dict(color="red", width=1.5, dash="dash"),
                    opacity=0.6,
                )
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            height=1600,
            width=1200,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='center', x=0.5),
        )
        
        fig.update_xaxes(
            tickformat='%b %Y',
            dtick='M12',
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor='gray',
            spikethickness=1,
        )
        fig.update_yaxes(rangemode='tozero')
        
        out = figures_dir / filename
        fig.write_html(out)
        print(f"Saved interactive: {out}")
    
    plot_interactive(random_4, "Weekly Wikipedia Edits (Tue-Mon) — 4 Random House Candidates", "candidate_timeseries_random4_interactive.html")
    plot_interactive(top_4, "Weekly Wikipedia Edits (Tue-Mon) — 4 Most-Edited House Candidates", "candidate_timeseries_top4_interactive.html")
    plot_interactive_10(random_10, "Weekly Wikipedia Edits (Tue-Mon) — 10 Random House Candidates", "candidate_timeseries_random10_interactive.html")

except ImportError:
    print("Plotly not installed. Skipping interactive plots.")
    print("Install with: pip install plotly")

print("\nDone.")
