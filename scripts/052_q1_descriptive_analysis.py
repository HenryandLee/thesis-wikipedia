"""
052_q1_descriptive_analysis.py

Q1 analysis: establish the shared national election-season editing pattern.

Steps:
    1. Figure 1 (3 panels): raw mean editing trajectories with reference lines.
       Panel A: full two-year cycle (k = -96 to 0), natural scale.
       Panel B: pre-election zoom, intensive margin (k = -52 to -1), no CI.
       Panel C: pre-election zoom, extensive margin (k = -52 to -1), no CI.
    2. Q1 event-study regression (page-cycle FE + election-year FE) with
       dummies D_k for k = -34 to -1, reference category k in {-70,...,-36}.
       Observations with k < -70 excluded due to post-election decay tail.
    3. Figure 1b: regression coefficient profile.
    4. Table 1: NI scalar test (k in {-8,...,-1}) and sensitivity window
       (k in {-10,...,-1}).
    5. Appendix figure: year-specific mean trajectories.

Reference period rationale:
    k=-96 to k=-84: mean Y=0.534 (post-election decay, elevated)
    k=-83 to k=-71: mean Y=0.372 (transition zone)
    k=-70 to k=-36: mean Y=0.326 (clean mid-cycle plateau, used as reference)
    Old reference (k<=-35): mean Y=0.381 (17% above plateau, contaminated)

CI bands on Panels B/C are omitted:
    The data covers near-complete population (9 cycles x all House candidates);
    a two-stage bootstrap with 9 year-level clusters is not well-calibrated.
    Inference is delegated to Figure 1b and Table 1.

Input:
    data/analysis/panel_weekly.parquet

Output:
    data/analysis/q1_results/figure1_panels.png
    data/analysis/q1_results/figure1b_regression.png
    data/analysis/q1_results/table1_regression.csv
    data/analysis/q1_results/table1_summary.csv
    data/analysis/q1_results/figure_appendix_year_trajectories.png
    data/analysis/q1_results/table_clustering_robustness.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict
import logging

try:
    import pyfixest as pf
except ImportError:
    raise ImportError(
        "pyfixest is required for this script. "
        "Install with: pip install pyfixest"
    )

from scipy import stats

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

# ============================================================
# Analysis constants
# ============================================================
FULL_K_RANGE = (-96, -1)         # Panel A full cycle (k=0 excluded: election week)
ZOOM_K_RANGE = (-52, -1)         # Panels B & C pre-election zoom
OFFSEASON_CUTOFF = -52           # k <= this marks Panel A shading boundary
REF_PERIOD_LOWER = -70           # Drop k < -70 from regression (post-election decay tail)
REF_PERIOD_UPPER = -36           # Reference window upper bound for descriptive stats / Figure 1 line
PRIMARY_START_K = -35            # Earliest March primaries
PRIMARY_END_K = -8               # Latest August primaries (post-updating cleared)
REG_WINDOW_START = -34           # Regression dummies start here (k = -34 to -1)
# Actual regression reference category: k in {-70,...,-35}  (no dummy for k <= -35)
# Note: REF_PERIOD_UPPER = -36 is used only for the descriptive reference line in Figure 1;
# the regression reference extends one week further to k = -35 because dummies start at -34.
REG_REFERENCE_K = -70            # Lower bound of reference category
BIN_WIDTH = 4                    # 4-week bins for descriptive panels B/C

# 4-week bin edges for Panels B & C: covers k = -52 to -1 in 13 bins
_BIN_EDGES = list(range(ZOOM_K_RANGE[0], 1, BIN_WIDTH))          # [-52,-48,...,0]
_BIN_CENTERS = [e + BIN_WIDTH // 2 for e in _BIN_EDGES[:-1]]     # [-50,-46,...,-2]


def _aggregate_4week(data: pd.DataFrame, outcome_col: str) -> pd.Series:
    """
    Aggregate weekly k_general means to 4-week bins.
    Returns a Series indexed by bin center (integer).
    Bins: [-52,-48), [-48,-44), ..., [-4,0) with centers -50,-46,...,-2.
    """
    binned = pd.cut(data['k_general'], bins=_BIN_EDGES,
                    labels=_BIN_CENTERS, right=False)
    return data.groupby(binned, observed=True)[outcome_col].mean().rename(int)


def load_panel() -> pd.DataFrame:
    """Load the panel dataset."""
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



# ============================================================
# Figure 1: Three-panel raw means (redesigned)
# ============================================================

def create_figure1(panel_df: pd.DataFrame):
    """
    Create Figure 1: Three-panel election-cycle trajectory.

    Panel A — Full two-year cycle (k = -96 to 0), intensive margin, no CI.
    Panel B — Pre-election zoom (k = -52 to -1), intensive margin, no CI [Option A].
    Panel C — Pre-election zoom (k = -52 to -1), extensive margin, no CI [Option A].

    Uses raw Y_jt and P_jt (no baseline-centering).
    Reference line = equal-weight mean of page-cycle means for k in {-70,...,-36}
    (clean mid-cycle window, excluding post-election decay tail).
    """
    logger.info("Creating Figure 1 (3-panel, raw means, no CI bands)...")

    # ---- Reference lines: clean mid-cycle window k in {-70,...,-36} ----
    ref_data = panel_df[
        (panel_df['k_general'] >= REF_PERIOD_LOWER) &
        (panel_df['k_general'] <= REF_PERIOD_UPPER)
    ]
    offseason_mean_Y = ref_data.groupby('page_cycle')['Y_jt'].mean().mean()
    offseason_mean_P = ref_data.groupby('page_cycle')['P_jt'].mean().mean()
    logger.info(f"Reference period mean: Y = {offseason_mean_Y:.4f}, P = {offseason_mean_P:.4f}")

    # ---- Panel A: Full range means ----
    full_data = panel_df[
        (panel_df['k_general'] >= FULL_K_RANGE[0]) &
        (panel_df['k_general'] <= FULL_K_RANGE[1]) &
        panel_df['Y_jt'].notna()
    ]
    means_A = full_data.groupby('k_general')['Y_jt'].mean()

    # ---- Panels B & C: Zoom range means, 4-week bins ----
    zoom_data = panel_df[
        (panel_df['k_general'] >= ZOOM_K_RANGE[0]) &
        (panel_df['k_general'] <= ZOOM_K_RANGE[1]) &
        panel_df['Y_jt'].notna()
    ]
    # Aggregate to 4-week bins (reduces visual noise while preserving trend)
    means_B = _aggregate_4week(zoom_data, 'Y_jt')
    means_C = _aggregate_4week(zoom_data, 'P_jt')

    # ---- Build figure: A on top (slim), B & C side-by-side below ----
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], hspace=0.35, wspace=0.30)
    ax_A = fig.add_subplot(gs[0, :])   # full width top
    ax_B = fig.add_subplot(gs[1, 0])   # left bottom
    ax_C = fig.add_subplot(gs[1, 1])   # right bottom

    # --- Panel A ---
    k_A = means_A.index.values
    ax_A.plot(k_A, means_A.values, color='steelblue', linewidth=1.2)

    # Shade pre-election region (zoom range)
    ax_A.axvspan(ZOOM_K_RANGE[0], ZOOM_K_RANGE[1], alpha=0.12, color='steelblue')
    ax_A.axvline(ZOOM_K_RANGE[0], color='gray', linestyle='--', linewidth=0.8)

    # Annotate post-election burst (leftmost spike region)
    y_max = means_A.values.max()
    spike_k = k_A[np.argmax(means_A.values)]
    ax_A.annotate(
        "Post-election\npage updating",
        xy=(spike_k, y_max),
        xytext=(spike_k + 10, y_max * 0.85),
        fontsize=8,
        arrowprops=dict(arrowstyle='->', color='dimgray', lw=1.0),
        color='dimgray',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
    )

    # Label the shaded region
    ax_A.text(ZOOM_K_RANGE[0] + 1, ax_A.get_ylim()[1] * 0.97,
              "expanded in B & C (right)",
              va='top', ha='left', fontsize=8, color='steelblue')

    ax_A.set_xlabel("Weeks to General Election", fontsize=10)
    ax_A.set_ylabel("Mean log(1 + edits)", fontsize=10)
    ax_A.set_title("(A) Full election cycle, 2008–2024 average", fontsize=11, fontweight='bold')
    ax_A.set_xlim(FULL_K_RANGE[0] - 1, FULL_K_RANGE[1] + 1)
    ax_A.tick_params(labelsize=9)

    for ax, means, ref_val, ylabel, title, color in [
        (ax_B, means_B, offseason_mean_Y,
         "Mean log(1 + edits)", "(B) Pre-election: intensive margin", "steelblue"),
        (ax_C, means_C, offseason_mean_P,
         "Fraction of pages with ≥1 edit", "(C) Pre-election: extensive margin", "#d6604d"),
    ]:
        k_vals = means.index.values
        ax.plot(k_vals, means.values, 'o-', color=color, linewidth=1.5, markersize=5)

        # Reference period mean line
        ax.axhline(ref_val, color='gray', linestyle='--', linewidth=1.0)
        ax.text(ZOOM_K_RANGE[1] + 1, ref_val,
                "Ref. period\navg.", va='center', ha='left', fontsize=7.5, color='gray')

        # Primary season vertical lines and shading
        ax.axvline(PRIMARY_START_K, color='gray', linestyle=':', linewidth=1.0)
        ax.axvline(PRIMARY_END_K, color='gray', linestyle=':', linewidth=1.0)
        ax.axvspan(PRIMARY_START_K, PRIMARY_END_K, alpha=0.06, color='gray')
        ylo, yhi = ax.get_ylim()
        ax.text(PRIMARY_START_K - 1.5, yhi - 0.01 * (yhi - ylo), "Mar.\nprimaries",
                va='top', ha='right', fontsize=7.5, color='gray')
        ax.text(PRIMARY_END_K - 1.5, yhi - 0.01 * (yhi - ylo), "Aug.\nprimaries",
                va='top', ha='right', fontsize=7.5, color='gray')

        ax.set_xlabel("Weeks to General Election (4-week bins)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(ZOOM_K_RANGE[0] - 1, ZOOM_K_RANGE[1] + 6)
        ax.set_xticks(_BIN_CENTERS[::2])   # every other bin center for readability
        ax.tick_params(labelsize=9)

    fig.tight_layout()

    output_path = Q1_RESULTS_DIR / "figure1_panels.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Figure 1 to {output_path}")


# ============================================================
# Q1 Event-Study Regression
# ============================================================

def run_q1_regression(panel_df: pd.DataFrame, outcome_col: str) -> Tuple[pd.DataFrame, dict]:
    """
    Run Q1 event-study regression:

        Y_jt = alpha_j + lambda_y + sum_{k=-34}^{-1} beta_k^(G) D_k^(G)(j,t) + epsilon_jt

    Sample: k in {-70,...,-1}.
      - k < -70 excluded: post-election decay contamination from the prior election.
      - k = 0  excluded: election week has no dummy and would otherwise enter the
        reference category, biasing the baseline upward.
    Reference category: k in {-70,...,-35} (no dummy; defines the zero baseline).
    Event dummies:       k in {-34,...,-1} (one dummy per week).
    Fixed effects: page_cycle (alpha_j) + election_cycle (lambda_y).
    Two-way clustered SEs: page_cycle + election_cycle.

    Args:
        panel_df: Panel DataFrame.
        outcome_col: 'Y_jt' or 'P_jt'.

    Returns:
        Tuple of (coef_df, test_results_dict).
    """
    logger.info(f"Running Q1 regression for outcome: {outcome_col}")

    # Filter: keep k in {-70,...,-1} only.
    # Lower bound: drop k < -70 (post-election decay contamination from prior election).
    # Upper bound: drop k = 0 (election week); it has no dummy and would silently
    #   enter the reference category, biasing the baseline upward (election-week
    #   editing is elevated due to results-announcement updating).
    # Reference category = k in {-70,...,-35}; dummies = k in {-34,...,-1}.
    df = panel_df[
        (panel_df['k_general'] >= REF_PERIOD_LOWER) &
        (panel_df['k_general'] <= -1) &
        panel_df['k_general'].notna() &
        panel_df[outcome_col].notna()
    ].copy()

    # Create dummies for k = -34 to -1
    for k in range(REG_WINDOW_START, 0):
        col_name = f'D_k_gm{abs(k)}'
        df[col_name] = (df['k_general'] == k).astype(int)

    # Ensure FE columns are categorical
    df['page_cycle'] = df['page_cycle'].astype('category')
    df['election_cycle'] = df['election_cycle'].astype('category')

    dummy_terms = ' + '.join([f'D_k_gm{abs(k)}' for k in range(REG_WINDOW_START, 0)])
    formula = f"{outcome_col} ~ {dummy_terms} | page_cycle + election_cycle"

    logger.info(f"Fitting: {formula[:80]}...")
    model = pf.feols(formula, data=df, vcov={'CRV1': 'page_cycle+election_cycle'})

    coefs = model.coef()
    ses = model.se()
    pvals = model.pvalue()

    results = []
    for k in range(REG_WINDOW_START, 0):
        var_name = f'D_k_gm{abs(k)}'
        if var_name in coefs.index:
            results.append({
                'k': k,
                'beta': coefs[var_name],
                'se': ses[var_name],
                'pvalue': pvals[var_name],
                'ci_lower': coefs[var_name] - 1.96 * ses[var_name],
                'ci_upper': coefs[var_name] + 1.96 * ses[var_name]
            })

    coef_df = pd.DataFrame(results)

    # ---- NI scalar test (k in {-8,...,-1}) ----
    coef_names = list(coefs.index)
    vcov = model._vcov

    def wald_test(window_ks):
        n_K = len(window_ks)
        R = np.zeros(len(coefs))
        for k in window_ks:
            var = f'D_k_gm{abs(k)}'
            if var in coef_names:
                idx = coef_names.index(var)
                R[idx] = 1.0 / n_K
        NI = R @ coefs.values
        var_NI = R @ vcov @ R
        se_NI = np.sqrt(var_NI)
        wald_stat = (NI ** 2) / var_NI
        p_value = 1 - stats.chi2.cdf(wald_stat, df=1)
        return {'NI': NI, 'se': se_NI, 'wald': wald_stat, 'pvalue': p_value}

    window_8 = list(range(-8, 0))   # k in {-8,...,-1}
    window_10 = list(range(-10, 0)) # k in {-10,...,-1}

    test_results = {
        'outcome': outcome_col,
        'N': model._N,
        'NI_8': wald_test(window_8),
        'NI_10': wald_test(window_10),
    }

    logger.info(f"  NI (k=-8...-1): {test_results['NI_8']['NI']:.4f} "
                f"(SE={test_results['NI_8']['se']:.4f}, p={test_results['NI_8']['pvalue']:.4f})")
    logger.info(f"  NI₁₀ (k=-10...-1): {test_results['NI_10']['NI']:.4f} "
                f"(SE={test_results['NI_10']['se']:.4f}, p={test_results['NI_10']['pvalue']:.4f})")

    return coef_df, test_results


def create_figure1b_regression(results_Y: Tuple, results_P: Tuple):
    """
    Create Figure 1b: Q1 regression coefficient profiles for both margins.

    Args:
        results_Y: (coef_df, test_results) for Y_jt
        results_P: (coef_df, test_results) for P_jt
    """
    logger.info("Creating Figure 1b (regression coefficient profiles)...")

    coef_Y, tests_Y = results_Y
    coef_P, tests_P = results_P

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, coef_df, label, color, outcome_label in [
        (axes[0], coef_Y, "Intensive margin", "steelblue", "log(1 + edits)"),
        (axes[1], coef_P, "Extensive margin", "#d6604d", "P(edits > 0)")
    ]:
        k_vals = coef_df['k'].values
        betas = coef_df['beta'].values
        ci_lo = coef_df['ci_lower'].values
        ci_hi = coef_df['ci_upper'].values

        ax.plot(k_vals, betas, 'o-', color=color, linewidth=1.5, markersize=4)
        ax.fill_between(k_vals, ci_lo, ci_hi, alpha=0.15, color=color)

        # Reference line
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)

        # Post-primary window shading and boundary label
        ax.axvline(PRIMARY_END_K, color='gray', linestyle=':', linewidth=1.0)
        ymax = ax.get_ylim()[1]
        ax.text(PRIMARY_END_K, ymax, "Post-primary\nwindow starts\n(k=−8)", va='bottom', ha='center',
                fontsize=7.5, color='gray')
        ax.axvspan(PRIMARY_END_K, -1, alpha=0.06, color='gray')

        # Fix x-axis to the dummy range; reference period k in {-70,...,-36}
        # noted in caption — no need for an off-plot axvline
        ax.set_xlim(REG_WINDOW_START - 1, 0)
        ax.set_xlabel("Weeks to General Election (k)", fontsize=10)
        ax.set_ylabel(rf"$\hat\beta_k^{{(G)}}$ ({outcome_label})", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=9)

    fig.tight_layout()

    output_path = Q1_RESULTS_DIR / "figure1b_regression.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Figure 1b to {output_path}")


def save_table1_regression(tests_Y: dict, tests_P: dict):
    """
    Save Table 1: NI scalar test results for both margins and both windows.
    """
    rows = []
    for tests, outcome_label in [(tests_Y, 'Intensive (log edits)'),
                                  (tests_P, 'Extensive (P>0)')]:
        for window_key, window_label in [('NI_8', 'k∈{-8,...,-1}'),
                                          ('NI_10', 'k∈{-10,...,-1}')]:
            t = tests[window_key]
            rows.append({
                'Outcome': outcome_label,
                'Window': window_label,
                'NI': t['NI'],
                'SE': t['se'],
                'Wald': t['wald'],
                'p-value': t['pvalue'],
                'N': tests['N'],
            })

    table1 = pd.DataFrame(rows)
    output_path = Q1_RESULTS_DIR / "table1_regression.csv"
    table1.to_csv(output_path, index=False)
    logger.info(f"Saved Table 1 (regression) to {output_path}")
    return table1


# ============================================================
# Robustness: One-way clustering (page-cycle only)
# ============================================================

def run_q1_regression_oneway(panel_df: pd.DataFrame, outcome_col: str) -> dict:
    """
    Re-run Q1 event-study regression with one-way clustering (page-cycle only)
    for robustness comparison against the two-way clustered baseline.

    Returns:
        Dictionary with NI test results under one-way clustering.
    """
    logger.info(f"Running Q1 regression (one-way cluster) for outcome: {outcome_col}")

    df = panel_df[
        (panel_df['k_general'] >= REF_PERIOD_LOWER) &
        (panel_df['k_general'] <= -1) &
        panel_df['k_general'].notna() &
        panel_df[outcome_col].notna()
    ].copy()

    for k in range(REG_WINDOW_START, 0):
        col_name = f'D_k_gm{abs(k)}'
        df[col_name] = (df['k_general'] == k).astype(int)

    df['page_cycle'] = df['page_cycle'].astype('category')
    df['election_cycle'] = df['election_cycle'].astype('category')

    dummy_terms = ' + '.join([f'D_k_gm{abs(k)}' for k in range(REG_WINDOW_START, 0)])
    formula = f"{outcome_col} ~ {dummy_terms} | page_cycle + election_cycle"

    model = pf.feols(formula, data=df, vcov={'CRV1': 'page_cycle'})

    coefs = model.coef()
    coef_names = list(coefs.index)
    vcov = model._vcov

    def wald_test(window_ks):
        n_K = len(window_ks)
        R = np.zeros(len(coefs))
        for k in window_ks:
            var = f'D_k_gm{abs(k)}'
            if var in coef_names:
                idx = coef_names.index(var)
                R[idx] = 1.0 / n_K
        NI = R @ coefs.values
        var_NI = R @ vcov @ R
        se_NI = np.sqrt(var_NI)
        wald_stat = (NI ** 2) / var_NI
        p_value = 1 - stats.chi2.cdf(wald_stat, df=1)
        return {'NI': NI, 'se': se_NI, 'wald': wald_stat, 'pvalue': p_value}

    return {
        'outcome': outcome_col,
        'N': model._N,
        'NI_8': wald_test(list(range(-8, 0))),
        'NI_10': wald_test(list(range(-10, 0))),
    }


def save_clustering_robustness(tests_2way_Y, tests_2way_P, tests_1way_Y, tests_1way_P):
    """
    Save Appendix Table: side-by-side comparison of two-way vs one-way clustered SEs.
    """
    rows = []
    for t2, t1, label in [
        (tests_2way_Y, tests_1way_Y, 'Intensive (log edits)'),
        (tests_2way_P, tests_1way_P, 'Extensive (P>0)'),
    ]:
        for wk, wl in [('NI_8', 'k∈{-8,...,-1}'), ('NI_10', 'k∈{-10,...,-1}')]:
            rows.append({
                'Outcome': label,
                'Window': wl,
                'NI': t2[wk]['NI'],
                'SE_twoway': t2[wk]['se'],
                'p_twoway': t2[wk]['pvalue'],
                'SE_oneway': t1[wk]['se'],
                'p_oneway': t1[wk]['pvalue'],
            })

    df = pd.DataFrame(rows)
    output_path = Q1_RESULTS_DIR / "table_clustering_robustness.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved clustering robustness table to {output_path}")

    print("\nClustering Robustness: Two-way vs. One-way (page-cycle only)")
    print("-" * 75)
    for _, row in df.iterrows():
        print(f"  {row['Outcome']}, {row['Window']}:")
        print(f"    NI={row['NI']:.4f}  |  2-way SE={row['SE_twoway']:.4f} (p={row['p_twoway']:.4f})  "
              f"|  1-way SE={row['SE_oneway']:.4f} (p={row['p_oneway']:.4f})")

    return df


# ============================================================
# Appendix Figure: Year-specific trajectories (Option B)
# ============================================================

def create_appendix_year_trajectories(panel_df: pd.DataFrame):
    """
    Appendix Figure: Year-specific mean editing trajectories (Option B).

    Shows 9 thin election-year-specific mean lines behind the bold overall
    mean, for intensive and extensive margins. No CI bands.
    Illustrates cross-year heterogeneity underlying the pooled mean in Figure 1.
    """
    logger.info("Creating appendix year-trajectory figure...")

    zoom_data = panel_df[
        (panel_df['k_general'] >= ZOOM_K_RANGE[0]) &
        (panel_df['k_general'] <= ZOOM_K_RANGE[1])
    ].copy()

    years = sorted(zoom_data['election_cycle'].unique())
    year_means_Y = {}
    year_means_P = {}
    for y in years:
        ydata = zoom_data[zoom_data['election_cycle'] == y]
        year_means_Y[y] = _aggregate_4week(ydata, 'Y_jt')
        year_means_P[y] = _aggregate_4week(ydata, 'P_jt')

    overall_Y = _aggregate_4week(zoom_data, 'Y_jt')
    overall_P = _aggregate_4week(zoom_data, 'P_jt')

    # Reference lines from clean period
    ref_data = panel_df[
        (panel_df['k_general'] >= REF_PERIOD_LOWER) &
        (panel_df['k_general'] <= REF_PERIOD_UPPER)
    ]
    ref_Y = ref_data.groupby('page_cycle')['Y_jt'].mean().mean()
    ref_P = ref_data.groupby('page_cycle')['P_jt'].mean().mean()

    cmap = plt.get_cmap('tab10')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, year_means, overall, ref_val, ylabel, title in [
        (axes[0], year_means_Y, overall_Y, ref_Y,
         "Mean log(1 + edits)", "(A) Intensive margin"),
        (axes[1], year_means_P, overall_P, ref_P,
         "Fraction of pages with ≥1 edit", "(B) Extensive margin"),
    ]:
        for i, y in enumerate(years):
            k_vals = year_means[y].index.values
            ax.plot(k_vals, year_means[y].values,
                    color=cmap(i), linewidth=0.9, alpha=0.55, label=str(y))

        ax.plot(overall.index.values, overall.values,
                color='black', linewidth=2.2, label='Overall mean', zorder=5)

        ax.axhline(ref_val, color='gray', linestyle='--', linewidth=1.0)
        ax.text(ZOOM_K_RANGE[1] + 0.3, ref_val,
                "Ref. period\navg.", va='center', ha='left', fontsize=7.5, color='gray')

        ax.axvline(PRIMARY_START_K, color='gray', linestyle=':', linewidth=1.0)
        ax.axvline(PRIMARY_END_K, color='gray', linestyle=':', linewidth=1.0)
        ax.axvspan(PRIMARY_START_K, PRIMARY_END_K, alpha=0.06, color='gray')

        ax.set_xlabel("Weeks to General Election (4-week bins)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(ZOOM_K_RANGE[0] - 1, ZOOM_K_RANGE[1] + 6)
        ax.set_xticks(_BIN_CENTERS[::2])
        ax.legend(fontsize=7.5, loc='upper left', ncol=2)
        ax.tick_params(labelsize=9)

    fig.tight_layout()

    output_path = Q1_RESULTS_DIR / "figure_appendix_year_trajectories.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved appendix year-trajectory figure to {output_path}")


# ============================================================
# DEPRECATED: Old baseline-centered analysis (kept for reproducibility)
# ============================================================

# DEPRECATED — original create_figure1() and create_table1() below.
# These produced figure1_panels.png using baseline-centered Y_tilde_G / P_tilde_G
# and bootstrap summary statistics. Now superseded by the raw-mean 3-panel figure
# and the Q1 event-study regression above. Kept verbatim for reproducibility.

def _deprecated_create_figure1(panel_df: pd.DataFrame):
    """
    [DEPRECATED] Create old Figure 1: Baseline-centered two-panel figure.
    """
    OLD_K_RANGE = (-94, -1)
    NEAR_WINDOW_L = 18
    WASHOUT_END_APPROX = -95

    plot_data = panel_df[
        (panel_df['k_general'] >= OLD_K_RANGE[0]) &
        (panel_df['k_general'] <= OLD_K_RANGE[1]) &
        panel_df['Y_tilde_G'].notna()
    ].copy()

    means_Y = plot_data.groupby('k_general')['Y_tilde_G'].mean()
    means_P = plot_data.groupby('k_general')['P_tilde_G'].mean()
    ci_lower_Y, ci_upper_Y = two_stage_bootstrap(plot_data, 'Y_tilde_G')
    ci_lower_P, ci_upper_P = two_stage_bootstrap(plot_data, 'P_tilde_G')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, means, ci_lo, ci_hi, ylabel, title in [
        (axes[0], means_Y, ci_lower_Y, ci_upper_Y,
         'Mean Baseline-Centered Log(1+Edits)', 'Panel A: Intensive Margin'),
        (axes[1], means_P, ci_lower_P, ci_upper_P,
         'Mean Baseline-Centered P(Edits > 0)', 'Panel B: Extensive Margin'),
    ]:
        k_values = means.index.values
        ax.plot(k_values, means.values, 'b-', linewidth=1.5, label='Mean')
        ax.fill_between(k_values,
                        ci_lo.reindex(k_values).values,
                        ci_hi.reindex(k_values).values,
                        alpha=0.2, color='blue')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(-NEAR_WINDOW_L, color='red', linestyle=':', linewidth=1.5,
                   label=f'Near window (k=-{NEAR_WINDOW_L})')
        ax.axvline(WASHOUT_END_APPROX, color='gray', linestyle=':', linewidth=1,
                   label=f'Washout end (k≈{WASHOUT_END_APPROX})')
        ax.set_xlabel('Weeks to General Election (k)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim(-106, OLD_K_RANGE[1] + 0.5)
        ax.set_xticks([-100, -80, -60, -40, -20, -1])

    fig.tight_layout()
    output_path = Q1_RESULTS_DIR / "figure1_panels_deprecated.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"[DEPRECATED] Saved old Figure 1 to {output_path}")


def _deprecated_create_table1(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    [DEPRECATED] Create old Table 1: Bootstrap summary statistics.
    """
    NEAR_WINDOW_L = 18
    near_window = panel_df[
        (panel_df['k_general'] >= -NEAR_WINDOW_L) &
        (panel_df['k_general'] <= -1) &
        panel_df['Y_tilde_G'].notna()
    ].copy()

    pc_means = near_window.groupby('page_cycle').agg({
        'Y_tilde_G': 'mean',
        'P_tilde_G': 'mean',
        'election_cycle': 'first'
    }).reset_index()

    grand_mean_Y = pc_means['Y_tilde_G'].mean()
    grand_mean_P = pc_means['P_tilde_G'].mean()
    ci_Y = two_stage_bootstrap_scalar(near_window, 'Y_tilde_G')
    ci_P = two_stage_bootstrap_scalar(near_window, 'P_tilde_G')

    table1 = pd.DataFrame({
        'Outcome': ['Log(1 + Edits) - Baseline', 'P(Edits > 0) - Baseline'],
        'Mean': [grand_mean_Y, grand_mean_P],
        'CI_Lower': [ci_Y[0], ci_P[0]],
        'CI_Upper': [ci_Y[1], ci_P[1]],
        'N_PageCycles': [len(pc_means), len(pc_means)]
    })
    table1['95% Bootstrap CI'] = table1.apply(
        lambda r: f"[{r['CI_Lower']:.4f}, {r['CI_Upper']:.4f}]", axis=1
    )
    output_path = Q1_RESULTS_DIR / "table1_summary.csv"
    table1.to_csv(output_path, index=False)
    logger.info(f"[DEPRECATED] Saved old Table 1 to {output_path}")
    return table1


# ============================================================
# Main
# ============================================================

def main():
    """Main entry point for Q1 analysis."""
    logger.info("=" * 60)
    logger.info("Q1 DESCRIPTIVE ANALYSIS")
    logger.info("=" * 60)

    panel = load_panel()

    # ---- Step 1: Figure 1 (3-panel, raw means, no CI bands) ----
    create_figure1(panel)

    # ---- Step 2: Q1 Event-Study Regression (reference k in {-70,...,-36}) ----
    results_Y = run_q1_regression(panel, 'Y_jt')
    results_P = run_q1_regression(panel, 'P_jt')

    # ---- Step 3: Figure 1b (regression coefficient profiles) ----
    create_figure1b_regression(results_Y, results_P)

    # ---- Step 4: Table 1 (regression NI tests) ----
    _, tests_Y = results_Y
    _, tests_P = results_P
    table1_reg = save_table1_regression(tests_Y, tests_P)

    # ---- Step 5: Robustness — one-way clustering (page-cycle only) ----
    tests_1way_Y = run_q1_regression_oneway(panel, 'Y_jt')
    tests_1way_P = run_q1_regression_oneway(panel, 'P_jt')
    robustness_df = save_clustering_robustness(tests_Y, tests_P, tests_1way_Y, tests_1way_P)

    # ---- Step 6: Appendix Figure (Option B — year-specific trajectories) ----
    create_appendix_year_trajectories(panel)

    # ---- Print summary ----
    print("\n" + "=" * 60)
    print("Q1 RESULTS SUMMARY")
    print("=" * 60)

    print("\nTable 1 (Regression): NI Scalar Tests")
    print("-" * 55)
    for _, row in table1_reg.iterrows():
        sig = '***' if row['p-value'] < 0.01 else ('**' if row['p-value'] < 0.05 else
              ('*' if row['p-value'] < 0.10 else ''))
        print(f"  {row['Outcome']}, {row['Window']}: "
              f"NI={row['NI']:.4f} (SE={row['SE']:.4f}), p={row['p-value']:.4f} {sig}")

    print(f"\nOutputs saved to: {Q1_RESULTS_DIR}")

    # Page-cycles by year
    pc_by_year = panel.groupby('election_cycle')['page_cycle'].nunique()
    print("\nPage-cycles by election year:")
    for year, count in pc_by_year.items():
        print(f"  {year}: {count:,}")

    return table1_reg


if __name__ == "__main__":
    main()
