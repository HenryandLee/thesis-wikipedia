"""
052_q1_descriptive_analysis.py

Q1 Descriptive Analysis: Establish the shared national "election-season" pattern.

This script:
1. Creates Figure 1 (Panels A & B): Baseline-centered outcomes vs weeks to election
2. Creates Table 1: Summary statistics for the near window
3. Uses two-stage bootstrap for confidence intervals

Output:
  - data/analysis/q1_results/figure1_panels.png
  - data/analysis/q1_results/table1_summary.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_DIR = DATA_DIR / "analysis"
Q1_RESULTS_DIR = ANALYSIS_DIR / "q1_results"
Q1_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Analysis parameters
NEAR_WINDOW_L = 18  # Near window is k = -18 to -1
K_RANGE = (-94, -1)  # Plot from -94 weeks before election (post-washout)
WASHOUT_END_APPROX = -95  # Approximate end of washout period
N_BOOTSTRAP = 100  # Number of bootstrap iterations (reduced for speed)


def load_panel() -> pd.DataFrame:
    """
    Load the panel dataset.

    Returns:
        Panel DataFrame
    """
    panel_path = ANALYSIS_DIR / "panel_weekly.parquet"

    if not panel_path.exists():
        raise FileNotFoundError(
            f"Panel dataset not found: {panel_path}\n"
            "Run 051_build_panel_dataset.py first."
        )

    logger.info(f"Loading panel from {panel_path}")
    df = pd.read_parquet(panel_path)
    logger.info(f"Loaded {len(df):,} observations")

    return df


def two_stage_bootstrap(
    panel_df: pd.DataFrame,
    outcome_col: str,
    group_col: str = 'k_general',
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 42
) -> Tuple[pd.Series, pd.Series]:
    """
    Two-stage bootstrap for election-level clustering (optimized).

    Stage 1: Sample election years with replacement
    Stage 2: Within each sampled year, sample page-cycles with replacement

    Args:
        panel_df: Panel DataFrame
        outcome_col: Column name for the outcome variable
        group_col: Column to group by (e.g., k_general)
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed for reproducibility

    Returns:
        Tuple of (lower_ci, upper_ci) as Series indexed by group_col
    """
    np.random.seed(seed)

    years = panel_df['election_cycle'].unique()

    # Pre-compute: group data by year and page_cycle for fast lookup
    # Create a mapping: (year, page_cycle) -> list of row indices
    logger.info("Pre-computing indices for bootstrap...")

    # Get unique page-cycles per year
    year_page_cycles = {}
    year_data_dict = {}
    for y in years:
        year_mask = panel_df['election_cycle'] == y
        year_data_dict[y] = panel_df[year_mask]
        year_page_cycles[y] = year_data_dict[y]['page_cycle'].unique()

    # Pre-aggregate: compute mean outcome by (year, page_cycle, k_general)
    # This is the key optimization - we work with pre-aggregated data
    logger.info("Pre-aggregating data...")
    agg_df = panel_df.groupby(['election_cycle', 'page_cycle', group_col])[outcome_col].mean().reset_index()

    # Create lookup dict: year -> DataFrame of (page_cycle, k, mean_outcome)
    year_agg = {y: agg_df[agg_df['election_cycle'] == y] for y in years}

    bootstrap_means = []

    for b in tqdm(range(n_bootstrap), desc=f"Bootstrap ({outcome_col})"):
        # Stage 1: Sample years with replacement
        sampled_years = np.random.choice(years, size=len(years), replace=True)

        # Stage 2: Within each sampled year, sample page-cycles with replacement
        sampled_agg_dfs = []
        for y in sampled_years:
            y_agg = year_agg[y]
            if len(y_agg) == 0:
                continue

            pcs = year_page_cycles[y]
            sampled_pcs = np.random.choice(pcs, size=len(pcs), replace=True)

            # Use isin for vectorized filtering (handles duplicates via concat)
            # For true bootstrap with replacement, we need to handle duplicates
            pc_counts = pd.Series(sampled_pcs).value_counts()

            for pc, count in pc_counts.items():
                pc_data = y_agg[y_agg['page_cycle'] == pc]
                for _ in range(count):
                    sampled_agg_dfs.append(pc_data)

        if not sampled_agg_dfs:
            continue

        boot_agg = pd.concat(sampled_agg_dfs, ignore_index=True)

        # Compute mean by group (now working with pre-aggregated data)
        group_mean = boot_agg.groupby(group_col)[outcome_col].mean()
        bootstrap_means.append(group_mean)

    # Combine bootstrap samples
    boot_results = pd.DataFrame(bootstrap_means)

    # Compute 2.5% and 97.5% percentiles
    ci_lower = boot_results.quantile(0.025)
    ci_upper = boot_results.quantile(0.975)

    return ci_lower, ci_upper


def two_stage_bootstrap_scalar(
    panel_df: pd.DataFrame,
    outcome_col: str,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Two-stage bootstrap for a scalar statistic (overall mean) - optimized.

    Args:
        panel_df: Panel DataFrame
        outcome_col: Column name for the outcome variable
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed

    Returns:
        Tuple of (lower_ci, upper_ci)
    """
    np.random.seed(seed)

    years = panel_df['election_cycle'].unique()

    # Pre-aggregate: compute mean outcome per page-cycle
    pc_means = panel_df.groupby(['election_cycle', 'page_cycle'])[outcome_col].mean().reset_index()
    pc_means.columns = ['election_cycle', 'page_cycle', 'pc_mean']

    # Create lookup: year -> array of page-cycle means
    year_pc_means = {}
    year_page_cycles = {}
    for y in years:
        y_data = pc_means[pc_means['election_cycle'] == y]
        year_page_cycles[y] = y_data['page_cycle'].values
        year_pc_means[y] = y_data['pc_mean'].values

    bootstrap_means = []

    for _ in tqdm(range(n_bootstrap), desc=f"Bootstrap scalar ({outcome_col})"):
        # Stage 1: Sample years with replacement
        sampled_years = np.random.choice(years, size=len(years), replace=True)

        # Stage 2: Sample page-cycles within each year
        sampled_values = []
        for y in sampled_years:
            pcs = year_page_cycles[y]
            means = year_pc_means[y]
            if len(pcs) == 0:
                continue

            # Sample indices with replacement
            sampled_idx = np.random.choice(len(pcs), size=len(pcs), replace=True)
            sampled_values.extend(means[sampled_idx])

        if sampled_values:
            bootstrap_means.append(np.nanmean(sampled_values))

    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)

    return ci_lower, ci_upper


def create_figure1(panel_df: pd.DataFrame):
    """
    Create Figure 1: Election Season Pattern.

    Panel A: Mean baseline-centered log edits (Y_tilde_G) vs weeks to election
    Panel B: Mean baseline-centered activity probability (P_tilde_G) vs weeks to election

    Args:
        panel_df: Panel DataFrame
    """
    logger.info("Creating Figure 1...")

    # Filter to k range
    plot_data = panel_df[
        (panel_df['k_general'] >= K_RANGE[0]) &
        (panel_df['k_general'] <= K_RANGE[1]) &
        panel_df['Y_tilde_G'].notna()
    ].copy()

    logger.info(f"Plot data: {len(plot_data):,} observations")

    # Compute point estimates
    means_Y = plot_data.groupby('k_general')['Y_tilde_G'].mean()
    means_P = plot_data.groupby('k_general')['P_tilde_G'].mean()

    # Compute bootstrap CIs
    logger.info("Computing bootstrap CIs for Panel A (Y_tilde_G)...")
    ci_lower_Y, ci_upper_Y = two_stage_bootstrap(plot_data, 'Y_tilde_G')

    logger.info("Computing bootstrap CIs for Panel B (P_tilde_G)...")
    ci_lower_P, ci_upper_P = two_stage_bootstrap(plot_data, 'P_tilde_G')

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel A: Intensive Margin ---
    ax = axes[0]

    k_values = means_Y.index.values
    y_values = means_Y.values

    ax.plot(k_values, y_values, 'b-', linewidth=1.5, label='Mean')

    # CI band
    ci_lower_aligned = ci_lower_Y.reindex(k_values).values
    ci_upper_aligned = ci_upper_Y.reindex(k_values).values
    ax.fill_between(k_values, ci_lower_aligned, ci_upper_aligned, alpha=0.2, color='blue')

    # Reference lines
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(-NEAR_WINDOW_L, color='red', linestyle=':', linewidth=1.5,
               label=f'Near window (k=-{NEAR_WINDOW_L})')
    ax.axvline(WASHOUT_END_APPROX, color='gray', linestyle=':', linewidth=1,
               label=f'Washout end (k≈{WASHOUT_END_APPROX})')

    ax.set_xlabel('Weeks to General Election (k)', fontsize=11)
    ax.set_ylabel('Mean Baseline-Centered Log(1+Edits)', fontsize=11)
    ax.set_title('Panel A: Intensive Margin', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(-106, K_RANGE[1] + 0.5)  # x-axis from -104 to -1
    ax.set_xticks([-100, -80, -60, -40, -20, -1])  # Integer ticks only

    # --- Panel B: Extensive Margin ---
    ax = axes[1]

    k_values = means_P.index.values
    y_values = means_P.values

    ax.plot(k_values, y_values, 'b-', linewidth=1.5, label='Mean')

    # CI band
    ci_lower_aligned = ci_lower_P.reindex(k_values).values
    ci_upper_aligned = ci_upper_P.reindex(k_values).values
    ax.fill_between(k_values, ci_lower_aligned, ci_upper_aligned, alpha=0.2, color='blue')

    # Reference lines
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(-NEAR_WINDOW_L, color='red', linestyle=':', linewidth=1.5,
               label=f'Near window (k=-{NEAR_WINDOW_L})')
    ax.axvline(WASHOUT_END_APPROX, color='gray', linestyle=':', linewidth=1,
               label=f'Washout end (k≈{WASHOUT_END_APPROX})')

    ax.set_xlabel('Weeks to General Election (k)', fontsize=11)
    ax.set_ylabel('Mean Baseline-Centered P(Edits > 0)', fontsize=11)
    ax.set_title('Panel B: Extensive Margin', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(-106, K_RANGE[1] + 0.5)  # x-axis from -104 to -1
    ax.set_xticks([-100, -80, -60, -40, -20, -1])  # Integer ticks only

    fig.tight_layout()

    # Save
    output_path = Q1_RESULTS_DIR / "figure1_panels.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved Figure 1 to {output_path}")


def create_figure1a_aggregated(panel_df: pd.DataFrame, weeks_per_bin: int = 4):
    """
    Create Figure 1A: Aggregated (smoothed) Election Season Pattern.

    Same as Figure 1 but with data aggregated into multi-week bins.
    This is an experimental visualization to see smoothed patterns.

    Args:
        panel_df: Panel DataFrame
        weeks_per_bin: Number of weeks to aggregate (default 4)
    """
    logger.info(f"Creating Figure 1A (aggregated, {weeks_per_bin}-week bins)...")

    # Filter to k range
    plot_data = panel_df[
        (panel_df['k_general'] >= K_RANGE[0]) &
        (panel_df['k_general'] <= K_RANGE[1]) &
        panel_df['Y_tilde_G'].notna()
    ].copy()

    # Create bin variable: group k values into bins
    # Bins are aligned to end at -1 (e.g., -4 to -1, -8 to -5, etc.)
    # Formula: bin = floor((k + 1) / weeks_per_bin) * weeks_per_bin
    plot_data['k_bin'] = ((plot_data['k_general'] + 1) // weeks_per_bin) * weeks_per_bin

    # For labeling, use the midpoint of each bin
    # e.g., bin -4 (covering -4 to -1) has midpoint -2.5
    plot_data['k_bin_mid'] = plot_data['k_bin'] + (weeks_per_bin - 1) / 2

    logger.info(f"Plot data: {len(plot_data):,} observations, "
                f"{plot_data['k_bin'].nunique()} bins")

    # Compute point estimates by bin
    means_Y = plot_data.groupby('k_bin')['Y_tilde_G'].mean()
    means_P = plot_data.groupby('k_bin')['P_tilde_G'].mean()

    # For bootstrap CIs on aggregated data, we bootstrap on the binned data
    logger.info("Computing bootstrap CIs for aggregated Panel A...")
    ci_lower_Y, ci_upper_Y = two_stage_bootstrap(plot_data, 'Y_tilde_G', group_col='k_bin')

    logger.info("Computing bootstrap CIs for aggregated Panel B...")
    ci_lower_P, ci_upper_P = two_stage_bootstrap(plot_data, 'P_tilde_G', group_col='k_bin')

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel A: Intensive Margin ---
    ax = axes[0]

    k_values = means_Y.index.values
    y_values = means_Y.values

    # Use midpoint for x-axis positioning
    k_mid = k_values + (weeks_per_bin - 1) / 2

    ax.plot(k_mid, y_values, 'b-o', linewidth=2, markersize=8, label='Mean')

    # CI band
    ci_lower_aligned = ci_lower_Y.reindex(k_values).values
    ci_upper_aligned = ci_upper_Y.reindex(k_values).values
    ax.fill_between(k_mid, ci_lower_aligned, ci_upper_aligned, alpha=0.2, color='blue')

    # Reference lines
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(-NEAR_WINDOW_L, color='red', linestyle=':', linewidth=1.5,
               label=f'Near window (k=-{NEAR_WINDOW_L})')
    ax.axvline(WASHOUT_END_APPROX, color='gray', linestyle=':', linewidth=1,
               label=f'Washout end (k≈{WASHOUT_END_APPROX})')

    ax.set_xlabel(f'Weeks to General Election (k, {weeks_per_bin}-week bins)', fontsize=11)
    ax.set_ylabel('Mean Baseline-Centered Log(1+Edits)', fontsize=11)
    ax.set_title(f'Panel A: Intensive Margin ({weeks_per_bin}-week aggregated)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(-106, K_RANGE[1] + 0.5)

    # --- Panel B: Extensive Margin ---
    ax = axes[1]

    k_values = means_P.index.values
    y_values = means_P.values
    k_mid = k_values + (weeks_per_bin - 1) / 2

    ax.plot(k_mid, y_values, 'b-o', linewidth=2, markersize=8, label='Mean')

    # CI band
    ci_lower_aligned = ci_lower_P.reindex(k_values).values
    ci_upper_aligned = ci_upper_P.reindex(k_values).values
    ax.fill_between(k_mid, ci_lower_aligned, ci_upper_aligned, alpha=0.2, color='blue')

    # Reference lines
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(-NEAR_WINDOW_L, color='red', linestyle=':', linewidth=1.5,
               label=f'Near window (k=-{NEAR_WINDOW_L})')
    ax.axvline(WASHOUT_END_APPROX, color='gray', linestyle=':', linewidth=1,
               label=f'Washout end (k≈{WASHOUT_END_APPROX})')

    ax.set_xlabel(f'Weeks to General Election (k, {weeks_per_bin}-week bins)', fontsize=11)
    ax.set_ylabel('Mean Baseline-Centered P(Edits > 0)', fontsize=11)
    ax.set_title(f'Panel B: Extensive Margin ({weeks_per_bin}-week aggregated)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(-106, K_RANGE[1] + 0.5)

    fig.tight_layout()

    # Save
    output_path = Q1_RESULTS_DIR / f"figure1a_aggregated_{weeks_per_bin}week.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved Figure 1A to {output_path}")


def create_table1(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Table 1: Summary Statistics for Near Window.

    For each outcome:
    1. Average Y_tilde_G over k=-18 to -1 for each page-cycle
    2. Average across all page-cycles
    3. Report 95% bootstrap CI

    Args:
        panel_df: Panel DataFrame

    Returns:
        Summary table DataFrame
    """
    logger.info("Creating Table 1...")

    # Filter to near window
    near_window = panel_df[
        (panel_df['k_general'] >= -NEAR_WINDOW_L) &
        (panel_df['k_general'] <= -1) &
        panel_df['Y_tilde_G'].notna()
    ].copy()

    logger.info(f"Near window data: {len(near_window):,} observations")

    # Step 1: Average within each page-cycle
    pc_means = near_window.groupby('page_cycle').agg({
        'Y_tilde_G': 'mean',
        'P_tilde_G': 'mean',
        'election_cycle': 'first'
    }).reset_index()

    # Step 2: Grand mean
    grand_mean_Y = pc_means['Y_tilde_G'].mean()
    grand_mean_P = pc_means['P_tilde_G'].mean()

    # Step 3: Bootstrap CIs
    logger.info("Computing bootstrap CI for Y_tilde_G mean...")
    ci_Y = two_stage_bootstrap_scalar(near_window, 'Y_tilde_G')

    logger.info("Computing bootstrap CI for P_tilde_G mean...")
    ci_P = two_stage_bootstrap_scalar(near_window, 'P_tilde_G')

    # Create table
    table1 = pd.DataFrame({
        'Outcome': ['Log(1 + Edits) - Baseline', 'P(Edits > 0) - Baseline'],
        'Mean': [grand_mean_Y, grand_mean_P],
        'CI_Lower': [ci_Y[0], ci_P[0]],
        'CI_Upper': [ci_Y[1], ci_P[1]],
        'N_PageCycles': [len(pc_means), len(pc_means)]
    })

    # Format CI as string
    table1['95% Bootstrap CI'] = table1.apply(
        lambda r: f"[{r['CI_Lower']:.4f}, {r['CI_Upper']:.4f}]",
        axis=1
    )

    # Save
    output_path = Q1_RESULTS_DIR / "table1_summary.csv"
    table1.to_csv(output_path, index=False)
    logger.info(f"Saved Table 1 to {output_path}")

    # Also save page-cycle breakdown by year
    pc_by_year = pc_means.groupby('election_cycle').size().reset_index(name='n_page_cycles')
    breakdown_path = Q1_RESULTS_DIR / "page_cycles_by_year.csv"
    pc_by_year.to_csv(breakdown_path, index=False)
    logger.info(f"Saved page-cycle breakdown to {breakdown_path}")

    return table1


def main():
    """Main entry point for Q1 analysis."""
    logger.info("=" * 60)
    logger.info("Q1 DESCRIPTIVE ANALYSIS")
    logger.info("=" * 60)

    # Load panel
    panel = load_panel()

    # Create Figure 1
    create_figure1(panel)

    # Create Figure 1A (experimental: 4-week aggregated)
    create_figure1a_aggregated(panel, weeks_per_bin=4)

    # Create Table 1
    table1 = create_table1(panel)

    # Print results
    print("\n" + "=" * 60)
    print("Q1 RESULTS SUMMARY")
    print("=" * 60)

    print("\nTable 1: Near Window Summary Statistics")
    print("-" * 50)
    for _, row in table1.iterrows():
        print(f"{row['Outcome']}:")
        print(f"  Mean: {row['Mean']:.4f}")
        print(f"  95% CI: {row['95% Bootstrap CI']}")
        print(f"  N page-cycles: {row['N_PageCycles']}")
        print()

    # Page-cycles by year
    pc_by_year = panel.groupby('election_cycle')['page_cycle'].nunique()
    print("Page-cycles by election year:")
    for year, count in pc_by_year.items():
        print(f"  {year}: {count:,}")

    print(f"\nOutputs saved to: {Q1_RESULTS_DIR}")

    return table1


if __name__ == "__main__":
    main()
