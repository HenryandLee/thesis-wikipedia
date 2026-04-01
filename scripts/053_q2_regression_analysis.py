"""
053_q2_regression_analysis.py

Q2 regression analysis: measure locality vs. national synchronization
using cross-state variation in primary dates.

Steps:
    1. Prepare regression data (exclude Louisiana, create dummies).
    2. Estimate Model 1: basic event study with page-cycle and
       calendar-week fixed effects.
    3. Estimate Model 2: trend interactions for nationalization test.
    4. Figure 2: primary timing profile (beta_k coefficients).
    5. Figure 3: trend comparison (2008 vs. 2024 implied profiles).

Uses pyfixest for high-dimensional fixed effects with two-way clustering.

Input:
    data/analysis/panel_weekly.parquet

Output:
    data/analysis/q2_results/figure2_beta_k.png
    data/analysis/q2_results/figure3_trend_comparison.png
    data/analysis/q2_results/model1_estimates.csv
    data/analysis/q2_results/model2_estimates.csv
    data/analysis/q2_results/sample_balance_diagnostic.csv
    data/analysis/q2_results/h1_test_model1.csv
    data/analysis/q2_results/h2_test_model2.csv
    data/analysis/q2_results/appendix_year_locality.csv
    data/analysis/q2_results/locality_robustness_windows.csv
    data/analysis/q2_results/regression_summary.txt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, List
import logging
from scipy import stats

try:
    import pyfixest as pf
except ImportError:
    raise ImportError(
        "pyfixest is required for this script. "
        "Install with: pip install pyfixest"
    )

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_DIR = DATA_DIR / "analysis"
Q2_RESULTS_DIR = ANALYSIS_DIR / "q2_results"
Q2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Analysis parameters
NEAR_WINDOW_L = 18  # k = -18 to -1
REFERENCE_YEAR = 2008  # For trend calculation (Trend=0 in 2008, Trend=8 in 2024)
LOCALITY_WINDOW = [-4, -3, -2, -1]  # K = {-4, -3, -2, -1} for locality index
ELECTION_YEARS = [2008, 2010, 2012, 2014, 2016, 2018, 2022, 2024]


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


def prepare_regression_data(panel_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for Q2 regression analysis.

    Steps:
    1. Filter to observations with valid primary dates
    2. Exclude Louisiana (jungle primary)
    3. Filter to relevant k_primary range (near window + baseline)
    4. Create dummy variables D_k_mXX
    5. Create trend variable
    6. Create interaction terms

    Note on sample balance:
    ---------------------------------
    I check whether the sample contributing to each k value is the same.

    Potential issue: If a page-cycle's primary is so early that k_primary = -19
    (the baseline cutoff) falls before the analysis window starts, that page-cycle
    will have no baseline observations. When computing Y_tilde_P = Y_jt - Y_bar_P,
    Y_bar_P will be NaN, and the entire page-cycle gets dropped.

    So the sample at each k value should be balanced (same page-cycles
    contribute to all k from -18 to -1), because
    1. The panel is constructed as a complete panel (all weeks for each page-cycle)
    2. Any page-cycle without a valid baseline is dropped entirely
    3. Remaining page-cycles have observations for all k values in [-18, -1]

    I verify this below by checking observation counts per k value.
    If counts are identical, the sample is balanced. If they differ, there may
    be compositional effects that complicate interpretation of β_k comparisons.

    Args:
        panel_df: Panel DataFrame

    Returns:
        Tuple of (regression-ready DataFrame, balance diagnostic DataFrame)
    """
    logger.info("Preparing regression data...")

    # Start with valid primary dates
    df = panel_df[panel_df['k_primary'].notna()].copy()
    logger.info(f"Observations with primary dates: {len(df):,}")

    # Exclude Louisiana
    df = df[df['state'] != 'LA'].copy()
    logger.info(f"After excluding Louisiana: {len(df):,}")

    # Exclude 2020 election cycle — COVID caused widespread primary postponements,
    # making k_primary systematically mis-measured for that year.
    df = df[df['election_cycle'] != 2020].copy()
    logger.info(f"After excluding 2020 (COVID primary disruptions): {len(df):,}")

    # Filter to relevant k_primary range
    # Near window: k = -18 to -1
    # Baseline (reference): k <= -19
    df = df[
        ((df['k_primary'] >= -NEAR_WINDOW_L) & (df['k_primary'] <= -1)) |
        (df['k_primary'] <= -(NEAR_WINDOW_L + 1))
    ].copy()
    logger.info(f"After k_primary filtering: {len(df):,}")

    # Filter to observations with valid Y_tilde_P
    df = df[df['Y_tilde_P'].notna()].copy()
    logger.info(f"After dropping missing Y_tilde_P: {len(df):,}")

    # =========================================================================
    # DIAGNOSTIC: Check sample balance across k values
    # =========================================================================
    # Verify that the same set of page-cycles contributes to each k value.
    # If observation counts differ across k, there are compositional differences
    # that could affect interpretation (comparing β_k across different samples).
    #
    # Expected: All k values from -18 to -1 should have identical counts,
    # because each page-cycle in the sample should have exactly one observation
    # per k value (the panel is complete, and page-cycles without valid baselines
    # have already been dropped).
    # =========================================================================

    near_window_obs = df[(df['k_primary'] >= -NEAR_WINDOW_L) & (df['k_primary'] <= -1)]
    k_counts = near_window_obs.groupby('k_primary').size().sort_index()

    logger.info("\n" + "=" * 50)
    logger.info("SAMPLE BALANCE DIAGNOSTIC")
    logger.info("=" * 50)
    logger.info("Observations per k_primary value in near window:")
    for k, count in k_counts.items():
        logger.info(f"  k = {int(k):3d}: {count:,} observations")

    # Check if balanced
    unique_counts = k_counts.unique()
    if len(unique_counts) == 1:
        logger.info(f"BALANCED: All k values have exactly {unique_counts[0]:,} observations")
        logger.info("  Same page-cycles contribute to all k values from -18 to -1")
    else:
        logger.warning(f"UNBALANCED: Observation counts vary across k values")
        logger.warning(f"  Range: {k_counts.min():,} to {k_counts.max():,}")
        logger.warning("  This indicates compositional differences across k values.")
        logger.warning("  Interpretation: beta_k coefficients may reflect different samples,")
        logger.warning("  not just different timing relative to primary.")

    # Also check unique page-cycles per k
    pc_per_k = near_window_obs.groupby('k_primary')['page_cycle'].nunique().sort_index()
    unique_pc_counts = pc_per_k.unique()
    if len(unique_pc_counts) == 1:
        logger.info(f"  Page-cycles per k: {unique_pc_counts[0]:,} (constant)")
    else:
        logger.warning(f"  Page-cycles per k vary: {pc_per_k.min():,} to {pc_per_k.max():,}")

    logger.info("=" * 50 + "\n")

    # Save diagnostic to DataFrame for later output
    balance_diagnostic = pd.DataFrame({
        'k_primary': k_counts.index.astype(int),
        'n_observations': k_counts.values,
        'n_page_cycles': pc_per_k.values
    })

    # Create dummy variables D_k_mXX for k = -18, ..., -1
    # Using 'm' prefix for minus to avoid formula parsing issues
    for k in range(-NEAR_WINDOW_L, 0):
        col_name = f'D_k_m{abs(k)}'
        df[col_name] = (df['k_primary'] == k).astype(int)

    # Create trend variable: Trend = (year - 2008) / 2
    # This centers Trend at 2008 (Trend=0), so β_k in Model 2 represents the 2008 profile
    # and γ_k represents the change per 2-year cycle step
    # 2008->0, 2010->1, 2012->2, ..., 2024->8
    df['Trend'] = (df['election_cycle'] - REFERENCE_YEAR) / 2

    # Create interaction terms for Model 2
    for k in range(-NEAR_WINDOW_L, 0):
        base_col = f'D_k_m{abs(k)}'
        interaction_col = f'{base_col}_trend'
        df[interaction_col] = df[base_col] * df['Trend']

    # Ensure page_cycle and calendar_week_id are categorical for FE
    df['page_cycle'] = df['page_cycle'].astype('category')
    df['calendar_week_id'] = df['calendar_week_id'].astype('category')

    logger.info(f"Final regression sample: {len(df):,} observations, "
               f"{df['page_cycle'].nunique():,} page-cycles")

    return df, balance_diagnostic


def estimate_model1(df: pd.DataFrame) -> Tuple[object, pd.DataFrame]:
    """
    Estimate Model 1: Basic event study.

    Y_tilde_P = alpha_j + delta_t + sum(beta_k * D_k) + epsilon

    Args:
        df: Regression-ready DataFrame

    Returns:
        Tuple of (pyfixest model object, coefficient DataFrame)
    """
    logger.info("Estimating Model 1...")

    # Build formula
    dummy_vars = ' + '.join([f'D_k_m{abs(k)}' for k in range(-NEAR_WINDOW_L, 0)])
    formula = f"Y_tilde_P ~ {dummy_vars} | page_cycle + calendar_week_id"

    logger.info(f"Formula: {formula[:100]}...")

    # Estimate with two-way clustering
    model = pf.feols(
        formula,
        data=df,
        vcov={'CRV1': 'page_cycle+calendar_week_id'}
    )

    # Extract coefficients
    coefs = model.coef()
    ses = model.se()
    pvals = model.pvalue()

    results = []
    for k in range(-NEAR_WINDOW_L, 0):
        var_name = f'D_k_m{abs(k)}'
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

    logger.info(f"Model 1 estimated. N = {model._N}")

    return model, coef_df


def estimate_model2(df: pd.DataFrame) -> Tuple[object, pd.DataFrame, pd.DataFrame]:
    """
    Estimate Model 2: Trend interactions for nationalization test.

    Y_tilde_P = alpha_j + delta_t + sum((beta_k + gamma_k * Trend) * D_k) + epsilon

    With Trend centered at 2008 (Trend=0 in 2008, Trend=8 in 2024):
    - β_k represents the primary-timing profile in 2008 (the baseline year)
    - γ_k represents the change in the profile per 2-year cycle step

    Args:
        df: Regression-ready DataFrame

    Returns:
        Tuple of (pyfixest model object, beta DataFrame, gamma DataFrame)
    """
    logger.info("Estimating Model 2...")

    # Build formula with base dummies and trend interactions
    base_terms = ' + '.join([f'D_k_m{abs(k)}' for k in range(-NEAR_WINDOW_L, 0)])
    interaction_terms = ' + '.join([f'D_k_m{abs(k)}_trend' for k in range(-NEAR_WINDOW_L, 0)])

    formula = f"Y_tilde_P ~ {base_terms} + {interaction_terms} | page_cycle + calendar_week_id"

    logger.info(f"Formula: {formula[:100]}...")

    # Estimate with two-way clustering
    model = pf.feols(
        formula,
        data=df,
        vcov={'CRV1': 'page_cycle+calendar_week_id'}
    )

    # Extract coefficients
    coefs = model.coef()
    ses = model.se()
    pvals = model.pvalue()

    # Beta coefficients (base dummies)
    beta_results = []
    for k in range(-NEAR_WINDOW_L, 0):
        var_name = f'D_k_m{abs(k)}'
        if var_name in coefs.index:
            beta_results.append({
                'k': k,
                'beta': coefs[var_name],
                'se': ses[var_name],
                'pvalue': pvals[var_name],
                'ci_lower': coefs[var_name] - 1.96 * ses[var_name],
                'ci_upper': coefs[var_name] + 1.96 * ses[var_name]
            })

    beta_df = pd.DataFrame(beta_results)

    # Gamma coefficients (trend interactions)
    gamma_results = []
    for k in range(-NEAR_WINDOW_L, 0):
        var_name = f'D_k_m{abs(k)}_trend'
        if var_name in coefs.index:
            gamma_results.append({
                'k': k,
                'gamma': coefs[var_name],
                'se': ses[var_name],
                'pvalue': pvals[var_name],
                'ci_lower': coefs[var_name] - 1.96 * ses[var_name],
                'ci_upper': coefs[var_name] + 1.96 * ses[var_name]
            })

    gamma_df = pd.DataFrame(gamma_results)

    logger.info(f"Model 2 estimated. N = {model._N}")

    return model, beta_df, gamma_df


def run_h1_test(model1: object) -> pd.DataFrame:
    """
    Test H1: Is there average locality near primaries? (Model 1)

    For each window K, test H_0: beta_bar_K = (1/|K|) * sum_{k in K} beta_k = 0
    using a 1-df Wald test with two-way clustered CRV1 standard errors.

    Runs across three windows:
      K = {-4,-3,-2,-1}  (main)
      K = {-2,-1}        (narrower)
      K = {-8,...,-1}    (wider)

    Args:
        model1: pyfixest model object from Model 1 estimation

    Returns:
        DataFrame with H1 test results (3 rows, one per window)
    """
    logger.info("Running H1 test (beta_bar_K = 0) from Model 1...")

    coefs = model1.coef()
    vcov = model1._vcov
    coef_names = list(coefs.index)

    windows = {
        'K={-4,-3,-2,-1}': [-4, -3, -2, -1],
        'K={-2,-1}':        [-2, -1],
        'K={-8,...,-1}':    list(range(-8, 0)),
    }

    rows = []
    for window_label, K in windows.items():
        n_K = len(K)
        beta_names = [f'D_k_m{abs(k)}' for k in K]
        missing = [n for n in beta_names if n not in coef_names]
        if missing:
            logger.warning(f"Window {window_label}: missing {missing}; skipping.")
            continue

        R = np.zeros(len(coefs))
        for k in K:
            idx = coef_names.index(f'D_k_m{abs(k)}')
            R[idx] = 1.0 / n_K

        beta_bar = R @ coefs.values
        var_bb   = R @ vcov @ R
        se_bb    = np.sqrt(var_bb)
        wald     = (beta_bar ** 2) / var_bb
        pval     = 1 - stats.chi2.cdf(wald, df=1)

        rows.append({
            'window':    window_label,
            'beta_bar_K': beta_bar,
            'se':          se_bb,
            'wald':        wald,
            'p_value':     pval,
        })
        logger.info(f"  {window_label}: beta_bar={beta_bar:.4f} (SE={se_bb:.4f}), p={pval:.4f}")

    h1_df = pd.DataFrame(rows)
    output_path = Q2_RESULTS_DIR / "h1_test_model1.csv"
    h1_df.to_csv(output_path, index=False)
    logger.info(f"Saved H1 test results to {output_path}")
    return h1_df


def run_h2_test(model2: object) -> pd.DataFrame:
    """
    Test H2: Is locality changing over time? (Model 2)

    For each window K, test H_0: gamma_bar_K = (1/|K|) * sum_{k in K} gamma_k = 0
    using a 1-df Wald test with two-way clustered CRV1 standard errors.

    Runs across three windows:
      K = {-4,-3,-2,-1}  (main)
      K = {-2,-1}        (narrower)
      K = {-8,...,-1}    (wider)

    Args:
        model2: pyfixest model object from Model 2 estimation

    Returns:
        DataFrame with H2 test results (3 rows, one per window)
    """
    logger.info("Running H2 test (gamma_bar_K = 0) from Model 2...")

    coefs = model2.coef()
    vcov = model2._vcov
    coef_names = list(coefs.index)

    windows = {
        'K={-4,-3,-2,-1}': [-4, -3, -2, -1],
        'K={-2,-1}':        [-2, -1],
        'K={-8,...,-1}':    list(range(-8, 0)),
    }

    rows = []
    for window_label, K in windows.items():
        n_K = len(K)
        gamma_names = [f'D_k_m{abs(k)}_trend' for k in K]
        missing = [n for n in gamma_names if n not in coef_names]
        if missing:
            logger.warning(f"Window {window_label}: missing {missing}; skipping.")
            continue

        R = np.zeros(len(coefs))
        for k in K:
            idx = coef_names.index(f'D_k_m{abs(k)}_trend')
            R[idx] = 1.0 / n_K

        gamma_bar = R @ coefs.values
        var_g     = R @ vcov @ R
        se_g      = np.sqrt(var_g)
        wald      = (gamma_bar ** 2) / var_g
        pval      = 1 - stats.chi2.cdf(wald, df=1)

        rows.append({
            'window':      window_label,
            'gamma_bar_K': gamma_bar,
            'se':           se_g,
            'wald':         wald,
            'p_value':      pval,
        })
        logger.info(f"  {window_label}: gamma_bar={gamma_bar:.4f} (SE={se_g:.4f}), p={pval:.4f}")

    h2_df = pd.DataFrame(rows)
    output_path = Q2_RESULTS_DIR / "h2_test_model2.csv"
    h2_df.to_csv(output_path, index=False)
    logger.info(f"Saved H2 test results to {output_path}")
    return h2_df


def run_year_locality_appendix(model2: object) -> pd.DataFrame:
    """
    Compute year-by-year LI_y from Model 2 for the main window K={-4,-3,-2,-1}.

    Produces the transparency appendix table (Appendix D1) showing
    LI_y = (1/|K|) * sum_{k in K} (beta_k + gamma_k * Trend_y) for each of the 8 cycles.

    Note: These 8 values are arithmetic projections from the two scalar parameters
    (beta_bar_K, gamma_bar_K); they add no new information beyond Tables 2 and 3,
    but aid interpretation cycle by cycle.

    Args:
        model2: pyfixest model object from Model 2 estimation

    Returns:
        DataFrame with year-by-year LI_y results (8 rows)
    """
    logger.info("Computing year-by-year LI_y for Appendix D1...")

    coefs = model2.coef()
    vcov = model2._vcov
    coef_names = list(coefs.index)

    K = LOCALITY_WINDOW  # [-4, -3, -2, -1]
    n_K = len(K)

    rows = []
    for year in ELECTION_YEARS:
        trend_y = (year - REFERENCE_YEAR) / 2
        R = np.zeros(len(coefs))
        for k in K:
            beta_idx  = coef_names.index(f'D_k_m{abs(k)}')
            gamma_idx = coef_names.index(f'D_k_m{abs(k)}_trend')
            R[beta_idx]  = 1.0 / n_K
            R[gamma_idx] = trend_y / n_K

        LI_y   = R @ coefs.values
        var_LI = R @ vcov @ R
        se_LI  = np.sqrt(var_LI)
        wald   = (LI_y ** 2) / var_LI
        pval   = 1 - stats.chi2.cdf(wald, df=1)

        rows.append({
            'year':    year,
            'trend':   trend_y,
            'LI_y':    LI_y,
            'se':      se_LI,
            'wald':    wald,
            'p_value': pval,
        })
        logger.info(f"  {year}: LI = {LI_y:.4f} (SE={se_LI:.4f}), p={pval:.4f}")

    appendix_df = pd.DataFrame(rows)
    output_path = Q2_RESULTS_DIR / "appendix_year_locality.csv"
    appendix_df.to_csv(output_path, index=False)
    logger.info(f"Saved year-by-year locality appendix to {output_path}")
    return appendix_df


def create_figure2(coef_df: pd.DataFrame):
    """
    Create Figure 2: Primary timing profile (beta_k plot).

    Args:
        coef_df: DataFrame with beta coefficients from Model 1
    """
    logger.info("Creating Figure 2...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot coefficients with error bars
    ax.errorbar(
        coef_df['k'],
        coef_df['beta'],
        yerr=[coef_df['beta'] - coef_df['ci_lower'],
              coef_df['ci_upper'] - coef_df['beta']],
        fmt='o-',
        capsize=3,
        color='navy',
        markersize=6,
        linewidth=1.5
    )

    # Reference line at 0
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    # Labels and title
    ax.set_xlabel('Weeks to State Primary (k)', fontsize=11)
    ax.set_ylabel(r'$\hat{\beta}_k$', fontsize=12)
    ax.set_xlim(-NEAR_WINDOW_L - 0.5, -0.5)
    ax.set_xticks(range(-NEAR_WINDOW_L, 0))

    # Add grid
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    # Save
    output_path = Q2_RESULTS_DIR / "figure2_beta_k.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved Figure 2 to {output_path}")


def create_figure3(beta_df: pd.DataFrame, gamma_df: pd.DataFrame):
    """
    Create Figure 3: Trend comparison (2008 vs 2024).

    Predicted f(k) = beta_k + gamma_k * Trend

    With Trend centered at 2008 (Trend=0):
    - f_2008 = β_k (the 2008 profile is directly the beta coefficients)
    - f_2024 = β_k + 8 * γ_k

    Args:
        beta_df: DataFrame with beta coefficients from Model 2
        gamma_df: DataFrame with gamma coefficients from Model 2
    """
    logger.info("Creating Figure 3...")

    # Merge beta and gamma
    merged = beta_df.merge(gamma_df[['k', 'gamma']], on='k')

    # Compute predicted f(k) for early (2008) and late (2024)
    # Trend = (year - 2008) / 2, so:
    # 2008 => Trend = 0
    # 2024 => Trend = 8
    trend_2008 = (2008 - REFERENCE_YEAR) / 2  # = 0
    trend_2024 = (2024 - REFERENCE_YEAR) / 2  # = 8

    # f_2008 = beta (since Trend=0, the beta coefficients are just the 2008 profile)
    merged['f_2008'] = merged['beta'] + merged['gamma'] * trend_2008
    merged['f_2024'] = merged['beta'] + merged['gamma'] * trend_2024

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot both lines
    ax.plot(merged['k'], merged['f_2008'], 'b-o',
            label='2008 (Trend=0)', linewidth=2, markersize=6)
    ax.plot(merged['k'], merged['f_2024'], 'r-s',
            label='2024 (Trend=8)', linewidth=2, markersize=6)

    # Reference line at 0
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    # Labels and title
    ax.set_xlabel('Weeks to State Primary (k)', fontsize=11)
    ax.set_ylabel(r'State-primary timing effect $f(k,y)$', fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(-NEAR_WINDOW_L - 0.5, -0.5)
    ax.set_xticks(range(-NEAR_WINDOW_L, 0))

    # Add grid
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    # Save
    output_path = Q2_RESULTS_DIR / "figure3_trend_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved Figure 3 to {output_path}")




def save_results(
    model1: object,
    model1_coef: pd.DataFrame,
    model2: object,
    model2_beta: pd.DataFrame,
    model2_gamma: pd.DataFrame,
    balance_diagnostic: pd.DataFrame,
    h1_results: pd.DataFrame = None,
    h2_results: pd.DataFrame = None,
    appendix_year_results: pd.DataFrame = None
):
    """
    Save regression results to CSV files.

    Args:
        model1: Model 1 object
        model1_coef: Model 1 coefficients
        model2: Model 2 object
        model2_beta: Model 2 beta coefficients
        model2_gamma: Model 2 gamma coefficients
        balance_diagnostic: Sample balance diagnostic DataFrame
        h1_results: H1 test results (beta_bar_K from Model 1, 3 windows)
        h2_results: H2 test results (gamma_bar_K from Model 2, 3 windows)
        appendix_year_results: Year-by-year LI_y from Model 2 (main window)
    """
    logger.info("Saving regression results...")

    # Model 1 estimates
    model1_path = Q2_RESULTS_DIR / "model1_estimates.csv"
    model1_coef.to_csv(model1_path, index=False)
    logger.info(f"Saved Model 1 estimates to {model1_path}")

    # Model 2 estimates (combined beta and gamma)
    model2_combined = model2_beta.merge(
        model2_gamma[['k', 'gamma', 'se', 'pvalue', 'ci_lower', 'ci_upper']].rename(
            columns={'se': 'gamma_se', 'pvalue': 'gamma_pvalue',
                     'ci_lower': 'gamma_ci_lower', 'ci_upper': 'gamma_ci_upper'}
        ),
        on='k'
    )
    model2_path = Q2_RESULTS_DIR / "model2_estimates.csv"
    model2_combined.to_csv(model2_path, index=False)
    logger.info(f"Saved Model 2 estimates to {model2_path}")

    # Save sample balance diagnostic
    balance_path = Q2_RESULTS_DIR / "sample_balance_diagnostic.csv"
    balance_diagnostic.to_csv(balance_path, index=False)
    logger.info(f"Saved sample balance diagnostic to {balance_path}")

    # Save hypothesis test results if provided
    if h1_results is not None:
        h1_path = Q2_RESULTS_DIR / "h1_test_model1.csv"
        h1_results.to_csv(h1_path, index=False)
        logger.info(f"Saved H1 test results to {h1_path}")

    if h2_results is not None:
        h2_path = Q2_RESULTS_DIR / "h2_test_model2.csv"
        h2_results.to_csv(h2_path, index=False)
        logger.info(f"Saved H2 test results to {h2_path}")

    if appendix_year_results is not None:
        app_path = Q2_RESULTS_DIR / "appendix_year_locality.csv"
        appendix_year_results.to_csv(app_path, index=False)
        logger.info(f"Saved year-by-year locality appendix to {app_path}")

    # Save model summaries as text
    summary_path = Q2_RESULTS_DIR / "regression_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MODEL 1: BASIC EVENT STUDY\n")
        f.write("=" * 60 + "\n")
        f.write(f"N observations: {model1._N}\n")
        f.write(f"R-squared: {model1._r2:.4f}\n")
        f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("MODEL 2: TREND INTERACTIONS\n")
        f.write("=" * 60 + "\n")
        f.write(f"N observations: {model2._N}\n")
        f.write(f"R-squared: {model2._r2:.4f}\n")
        f.write("\n")
        f.write("Trend variable: Trend = (cycle_year - 2008) / 2\n")
        f.write("  2008 -> Trend = 0\n")
        f.write("  2024 -> Trend = 8\n")
        f.write("\n")
        f.write("Coefficient interpretation:\n")
        f.write("  beta_k  = primary-timing profile in 2008 (baseline year)\n")
        f.write("  gamma_k = change in profile per 2-year cycle step\n")
        f.write("  f(k, year) = beta_k + gamma_k * Trend\n")
        f.write("\n")

        # Key coefficients
        f.write("Key coefficients (k = -1, week immediately before primary):\n")
        k_m1_beta = model2_beta[model2_beta['k'] == -1]['beta'].values[0]
        k_m1_gamma = model2_gamma[model2_gamma['k'] == -1]['gamma'].values[0]
        f.write(f"  beta_(-1)  = {k_m1_beta:.4f}  (2008 effect)\n")
        f.write(f"  gamma_(-1) = {k_m1_gamma:.4f}  (change per cycle)\n")
        f.write(f"  f(-1, 2024) = {k_m1_beta + 8 * k_m1_gamma:.4f}  (2024 effect)\n")
        f.write("\n")

        # Interpretation
        f.write("Interpretation:\n")
        if k_m1_gamma < 0:
            f.write("  gamma_(-1) < 0: Local primary-timed component is SHRINKING over time\n")
            f.write("  This is consistent with increasing nationalization.\n")
        else:
            f.write("  gamma_(-1) > 0: Local primary-timed component is STRENGTHENING over time\n")
            f.write("  This suggests locality is persisting or increasing.\n")

        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write("SAMPLE BALANCE DIAGNOSTIC\n")
        f.write("=" * 60 + "\n")
        f.write("Observations per k_primary value in near window:\n")
        f.write("-" * 40 + "\n")
        for _, row in balance_diagnostic.iterrows():
            f.write(f"  k = {int(row['k_primary']):3d}: "
                   f"{int(row['n_observations']):,} obs, "
                   f"{int(row['n_page_cycles']):,} page-cycles\n")

        # Check balance
        unique_obs = balance_diagnostic['n_observations'].unique()
        unique_pcs = balance_diagnostic['n_page_cycles'].unique()

        f.write("\n")
        if len(unique_obs) == 1 and len(unique_pcs) == 1:
            f.write("STATUS: BALANCED\n")
            f.write(f"  All k values have exactly {int(unique_obs[0]):,} observations\n")
            f.write(f"  All k values have exactly {int(unique_pcs[0]):,} page-cycles\n")
            f.write("  Same sample contributes to all k values from -18 to -1.\n")
        else:
            f.write("STATUS: UNBALANCED\n")
            f.write(f"  Observations range: {int(balance_diagnostic['n_observations'].min()):,} "
                   f"to {int(balance_diagnostic['n_observations'].max()):,}\n")
            f.write(f"  Page-cycles range: {int(balance_diagnostic['n_page_cycles'].min()):,} "
                   f"to {int(balance_diagnostic['n_page_cycles'].max()):,}\n")
            f.write("  WARNING: Different samples contribute to different k values.\n")
            f.write("  This may affect interpretation of beta_k comparisons.\n")

        # Hypothesis tests
        if h1_results is not None:
            f.write("\n")
            f.write("=" * 60 + "\n")
            f.write("Q2 HYPOTHESIS TESTS\n")
            f.write("=" * 60 + "\n")
            f.write("\n")
            f.write("-" * 60 + "\n")
            f.write("H1 TEST: beta_bar_K = 0 (locality exists?) — from Model 1\n")
            f.write("         beta_bar_K = (1/|K|) * sum_{k in K} beta_k\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Window':<22} {'beta_bar':>10} {'SE':>10} {'Wald':>10} {'p-value':>10}\n")
            f.write("-" * 60 + "\n")
            for _, row in h1_results.iterrows():
                sig = '***' if row['p_value'] < 0.01 else ('**' if row['p_value'] < 0.05 else ('*' if row['p_value'] < 0.1 else ''))
                f.write(f"{row['window']:<22} {row['beta_bar_K']:>10.4f} {row['se']:>10.4f} "
                       f"{row['wald']:>10.4f} {row['p_value']:>10.4f} {sig}\n")
            f.write("\n")
            f.write("Significance: *** p<0.01, ** p<0.05, * p<0.1\n")

        if h2_results is not None:
            f.write("\n")
            f.write("-" * 60 + "\n")
            f.write("H2 TEST: gamma_bar_K = 0 (nationalization?) — from Model 2\n")
            f.write("         gamma_bar_K = (1/|K|) * sum_{k in K} gamma_k\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Window':<22} {'gamma_bar':>10} {'SE':>10} {'Wald':>10} {'p-value':>10}\n")
            f.write("-" * 60 + "\n")
            for _, row in h2_results.iterrows():
                sig = '***' if row['p_value'] < 0.01 else ('**' if row['p_value'] < 0.05 else ('*' if row['p_value'] < 0.1 else ''))
                f.write(f"{row['window']:<22} {row['gamma_bar_K']:>10.4f} {row['se']:>10.4f} "
                       f"{row['wald']:>10.4f} {row['p_value']:>10.4f} {sig}\n")
            f.write("\n")
            f.write("Significance: *** p<0.01, ** p<0.05, * p<0.1\n")

        if appendix_year_results is not None:
            f.write("\n")
            f.write("-" * 60 + "\n")
            f.write("APPENDIX D1: Year-by-year LI_y (main window K={-4,-3,-2,-1})\n")
            f.write("             from Model 2\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Year':<8} {'LI_y':>10} {'SE':>10} {'Wald':>10} {'p-value':>10}\n")
            f.write("-" * 60 + "\n")
            for _, row in appendix_year_results.iterrows():
                sig = '***' if row['p_value'] < 0.01 else ('**' if row['p_value'] < 0.05 else ('*' if row['p_value'] < 0.1 else ''))
                f.write(f"{int(row['year']):<8} {row['LI_y']:>10.4f} {row['se']:>10.4f} "
                       f"{row['wald']:>10.4f} {row['p_value']:>10.4f} {sig}\n")

    logger.info(f"Saved summary to {summary_path}")


def main():
    """Main entry point for Q2 regression analysis."""
    logger.info("=" * 60)
    logger.info("Q2 REGRESSION ANALYSIS")
    logger.info("=" * 60)

    panel = load_panel()
    reg_data, balance_diagnostic = prepare_regression_data(panel)
    model1, model1_coef = estimate_model1(reg_data)
    model2, model2_beta, model2_gamma = estimate_model2(reg_data)
    create_figure2(model1_coef)
    create_figure3(model2_beta, model2_gamma)

    # Run hypothesis tests
    h1_results            = run_h1_test(model1)
    h2_results            = run_h2_test(model2)
    appendix_year_results = run_year_locality_appendix(model2)

    # Save results
    save_results(model1, model1_coef, model2, model2_beta, model2_gamma, balance_diagnostic,
                 h1_results, h2_results, appendix_year_results)

    # Print summary
    print("\n" + "=" * 60)
    print("Q2 RESULTS SUMMARY")
    print("=" * 60)

    print("\nModel 1: Basic Event Study")
    print("-" * 40)
    print(f"N observations: {model1._N:,}")
    print(f"R-squared: {model1._r2:.4f}")
    print("\nCoefficients (selected k values):")
    for k in [-1, -4, -8, -12, -18]:
        row = model1_coef[model1_coef['k'] == k]
        if not row.empty:
            beta = row['beta'].values[0]
            se = row['se'].values[0]
            print(f"  k={k:3d}: beta = {beta:7.4f} (se = {se:.4f})")

    print("\nModel 2: Trend Interactions")
    print("-" * 40)
    print(f"N observations: {model2._N:,}")
    print(f"R-squared: {model2._r2:.4f}")
    print("\nTrend coefficients (gamma) for selected k values:")
    for k in [-1, -4, -8, -12, -18]:
        row = model2_gamma[model2_gamma['k'] == k]
        if not row.empty:
            gamma = row['gamma'].values[0]
            se = row['se'].values[0]
            pval = row['pvalue'].values[0]
            sig = '*' if pval < 0.05 else ''
            print(f"  k={k:3d}: gamma = {gamma:7.4f} (se = {se:.4f}) {sig}")

    # Hypothesis tests summary
    print("\n" + "=" * 60)
    print("Q2 HYPOTHESIS TESTS")
    print("=" * 60)

    print("\nTest H1: beta_bar_K = 0 (locality exists?) — from Model 1")
    print("-" * 55)
    for _, row in h1_results.iterrows():
        sig = '***' if row['p_value'] < 0.01 else ('**' if row['p_value'] < 0.05 else '')
        print(f"  {row['window']}: beta_bar = {row['beta_bar_K']:.4f} "
              f"(SE = {row['se']:.4f}), p = {row['p_value']:.4f} {sig}")

    print("\nTest H2: gamma_bar_K = 0 (nationalization trend?) — from Model 2")
    print("-" * 55)
    for _, row in h2_results.iterrows():
        sig = '***' if row['p_value'] < 0.01 else ('**' if row['p_value'] < 0.05 else '')
        print(f"  {row['window']}: gamma_bar = {row['gamma_bar_K']:.4f} "
              f"(SE = {row['se']:.4f}), p = {row['p_value']:.4f} {sig}")

    print("\nYear-by-year LI_y (main window, for Appendix D1):")
    print("-" * 40)
    for _, row in appendix_year_results.iterrows():
        sig = '***' if row['p_value'] < 0.01 else ('**' if row['p_value'] < 0.05 else '')
        print(f"  {int(row['year'])}: LI = {row['LI_y']:.4f} "
              f"(SE = {row['se']:.4f}), p = {row['p_value']:.4f} {sig}")

    print(f"\nOutputs saved to: {Q2_RESULTS_DIR}")


if __name__ == "__main__":
    main()
