"""
053_q2_regression_analysis.py

Q2 Regression Analysis: Measure locality vs synchronization using staggered primaries.

Functionality:
1. Prepares data for regression (excludes Louisiana, creates dummies)
2. Estimates Model 1: Basic event study with page-cycle and calendar-week FE
3. Estimates Model 2: Trend interactions for nationalization test
4. Creates Figure 2: Primary timing profile (beta_k plot)
5. Creates Figure 3: Trend comparison (2008 vs 2024)

Uses pyfixest for high-dimensional FE with two-way clustering.

Output:
  - data/analysis/q2_results/figure2_beta_k.png
  - data/analysis/q2_results/figure3_trend_comparison.png
  - data/analysis/q2_results/model1_estimates.csv
  - data/analysis/q2_results/model2_estimates.csv
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
ELECTION_YEARS = [2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]


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
        logger.info(f"\n✓ BALANCED: All k values have exactly {unique_counts[0]:,} observations")
        logger.info("  Same page-cycles contribute to all k values from -18 to -1")
    else:
        logger.warning(f"\n⚠ UNBALANCED: Observation counts vary across k values")
        logger.warning(f"  Range: {k_counts.min():,} to {k_counts.max():,}")
        logger.warning("  This indicates compositional differences across k values.")
        logger.warning("  Interpretation: β_k coefficients may reflect different samples,")
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
    # 2008→0, 2010→1, 2012→2, ..., 2024→8
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


def run_locality_tests(model2: object) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run locality hypothesis tests from further_tests_for_q2.md.

    Test 1: "Is there any locality near primaries in year y?"
        For each year y, test H_0: LI_y = 0
        where LI_y = (1/|K|) * Σ_{k∈K} f_y(k)
        and f_y(k) = β_k + γ_k * Trend_y

    Test 2: "Is locality getting weaker/stronger over time?"
        Test H_0: γ̄_K = 0
        where γ̄_K = (1/|K|) * Σ_{k∈K} γ_k

    Uses 1-df Wald tests with two-way clustered CRV1 standard errors.

    Args:
        model2: pyfixest model object from Model 2 estimation

    Returns:
        Tuple of (test1_results DataFrame, test2_results DataFrame)
    """
    logger.info("Running locality hypothesis tests...")

    # Get coefficients and variance-covariance matrix
    # The vcov is already computed with two-way clustering during estimation
    coefs = model2.coef()
    vcov = model2._vcov  # Access the stored vcov matrix

    # Get coefficient names for the near-primary locality window
    # K = {-4, -3, -2, -1}
    K = LOCALITY_WINDOW
    n_K = len(K)

    # Identify beta and gamma coefficient names for k in K
    beta_names = [f'D_k_m{abs(k)}' for k in K]
    gamma_names = [f'D_k_m{abs(k)}_trend' for k in K]

    # Verify all coefficients exist
    all_names = list(coefs.index)
    for name in beta_names + gamma_names:
        if name not in all_names:
            raise ValueError(f"Coefficient {name} not found in model")

    # Get indices for constructing contrast vectors
    coef_names = list(coefs.index)

    # =========================================================================
    # Test 1: LI_y = 0 for each year y
    # =========================================================================
    # LI_y = (1/|K|) * Σ_{k∈K} (β_k + γ_k * Trend_y)
    #      = (1/|K|) * Σ_{k∈K} β_k + Trend_y * (1/|K|) * Σ_{k∈K} γ_k
    #
    # This is a linear combination: R'θ where θ is the coefficient vector
    # R has weights 1/|K| for each β_k in K, and Trend_y/|K| for each γ_k in K
    # =========================================================================

    test1_results = []

    for year in ELECTION_YEARS:
        # Compute Trend for this year
        trend_y = (year - REFERENCE_YEAR) / 2

        # Construct contrast vector R
        R = np.zeros(len(coefs))
        for k in K:
            beta_idx = coef_names.index(f'D_k_m{abs(k)}')
            gamma_idx = coef_names.index(f'D_k_m{abs(k)}_trend')
            R[beta_idx] = 1.0 / n_K
            R[gamma_idx] = trend_y / n_K

        # Compute LI_y = R'θ
        LI_y = R @ coefs.values

        # Compute Var(LI_y) = R' V R
        var_LI_y = R @ vcov @ R

        # Standard error
        se_LI_y = np.sqrt(var_LI_y)

        # Wald test statistic: W = (R'θ)^2 / (R' V R) ~ χ²(1)
        wald_stat = (LI_y ** 2) / var_LI_y

        # p-value from chi-squared distribution with 1 df
        p_value = 1 - stats.chi2.cdf(wald_stat, df=1)

        # Also compute t-statistic and two-sided p-value (for comparison)
        t_stat = LI_y / se_LI_y

        test1_results.append({
            'year': year,
            'trend': trend_y,
            'LI_y': LI_y,
            'se': se_LI_y,
            't_stat': t_stat,
            'wald_stat': wald_stat,
            'p_value': p_value,
            'significant_05': p_value < 0.05,
            'significant_01': p_value < 0.01
        })

    test1_df = pd.DataFrame(test1_results)

    logger.info("Test 1 (LI_y = 0 for each year) completed:")
    for _, row in test1_df.iterrows():
        sig = '***' if row['p_value'] < 0.01 else ('**' if row['p_value'] < 0.05 else '')
        logger.info(f"  {int(row['year'])}: LI = {row['LI_y']:.4f} (SE = {row['se']:.4f}), "
                   f"p = {row['p_value']:.4f} {sig}")

    # =========================================================================
    # Test 2: γ̄_K = 0 (nationalization trend)
    # =========================================================================
    # γ̄_K = (1/|K|) * Σ_{k∈K} γ_k
    #
    # Construct contrast vector R with weights 1/|K| for each γ_k in K
    # =========================================================================

    R_gamma = np.zeros(len(coefs))
    for k in K:
        gamma_idx = coef_names.index(f'D_k_m{abs(k)}_trend')
        R_gamma[gamma_idx] = 1.0 / n_K

    # Compute γ̄_K = R'θ
    gamma_bar = R_gamma @ coefs.values

    # Compute Var(γ̄_K) = R' V R
    var_gamma_bar = R_gamma @ vcov @ R_gamma

    # Standard error
    se_gamma_bar = np.sqrt(var_gamma_bar)

    # Wald test statistic
    wald_stat_gamma = (gamma_bar ** 2) / var_gamma_bar

    # p-value
    p_value_gamma = 1 - stats.chi2.cdf(wald_stat_gamma, df=1)

    # t-statistic
    t_stat_gamma = gamma_bar / se_gamma_bar

    test2_results = pd.DataFrame([{
        'parameter': 'gamma_bar_K',
        'window': str(K),
        'estimate': gamma_bar,
        'se': se_gamma_bar,
        't_stat': t_stat_gamma,
        'wald_stat': wald_stat_gamma,
        'p_value': p_value_gamma,
        'significant_05': p_value_gamma < 0.05,
        'significant_01': p_value_gamma < 0.01
    }])

    logger.info("Test 2 (γ̄_K = 0, nationalization trend) completed:")
    sig = '***' if p_value_gamma < 0.01 else ('**' if p_value_gamma < 0.05 else '')
    logger.info(f"  γ̄_K = {gamma_bar:.4f} (SE = {se_gamma_bar:.4f}), p = {p_value_gamma:.4f} {sig}")

    # Interpretation
    if gamma_bar < 0 and p_value_gamma < 0.05:
        logger.info("  Interpretation: Locality is WEAKENING over time (nationalization).")
    elif gamma_bar > 0 and p_value_gamma < 0.05:
        logger.info("  Interpretation: Locality is STRENGTHENING over time.")
    else:
        logger.info("  Interpretation: No significant change in locality over time.")

    return test1_df, test2_results


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
    ax.set_title('Figure 2. Local (state-timed) change in Wikipedia editing\n'
                 'before primaries, net of national synchronization.',
                 fontsize=11, fontweight='bold')

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
    ax.set_ylabel('Predicted f(k)', fontsize=11)
    ax.set_title('Figure 3. Change in the primary-timed locality profile\n'
                 'over time (2008 vs 2024).',
                 fontsize=11, fontweight='bold')

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
    test1_results: pd.DataFrame = None,
    test2_results: pd.DataFrame = None
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
        test1_results: Locality Index test results by year (optional)
        test2_results: Nationalization trend test results (optional)
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

    # Save locality test results if provided
    if test1_results is not None:
        test1_path = Q2_RESULTS_DIR / "locality_test1_by_year.csv"
        test1_results.to_csv(test1_path, index=False)
        logger.info(f"Saved Test 1 results to {test1_path}")

    if test2_results is not None:
        test2_path = Q2_RESULTS_DIR / "locality_test2_trend.csv"
        test2_results.to_csv(test2_path, index=False)
        logger.info(f"Saved Test 2 results to {test2_path}")

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
        f.write("  2008 → Trend = 0\n")
        f.write("  2024 → Trend = 8\n")
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

        # Locality hypothesis tests
        if test1_results is not None:
            f.write("\n")
            f.write("=" * 60 + "\n")
            f.write("LOCALITY HYPOTHESIS TESTS\n")
            f.write("=" * 60 + "\n")
            f.write("\n")
            f.write("Near-primary window K = {-4, -3, -2, -1}\n")
            f.write("LI_y = (1/|K|) * Σ_{k∈K} f_y(k)\n")
            f.write("where f_y(k) = β_k + γ_k * Trend_y\n")
            f.write("\n")
            f.write("-" * 60 + "\n")
            f.write("TEST 1: Is there any locality near primaries in year y?\n")
            f.write("        H_0: LI_y = 0 (Wald test, χ²(1))\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Year':<8} {'LI_y':>10} {'SE':>10} {'Wald':>10} {'p-value':>10} {'Sig':>6}\n")
            f.write("-" * 60 + "\n")
            for _, row in test1_results.iterrows():
                sig = '***' if row['p_value'] < 0.01 else ('**' if row['p_value'] < 0.05 else ('*' if row['p_value'] < 0.1 else ''))
                f.write(f"{int(row['year']):<8} {row['LI_y']:>10.4f} {row['se']:>10.4f} "
                       f"{row['wald_stat']:>10.4f} {row['p_value']:>10.4f} {sig:>6}\n")
            f.write("\n")
            f.write("Significance: *** p<0.01, ** p<0.05, * p<0.1\n")

        if test2_results is not None:
            f.write("\n")
            f.write("-" * 60 + "\n")
            f.write("TEST 2: Is locality changing over time (nationalization)?\n")
            f.write("        H_0: γ̄_K = 0 where γ̄_K = (1/|K|) * Σ_{k∈K} γ_k\n")
            f.write("-" * 60 + "\n")
            row = test2_results.iloc[0]
            sig = '***' if row['p_value'] < 0.01 else ('**' if row['p_value'] < 0.05 else ('*' if row['p_value'] < 0.1 else ''))
            f.write(f"  γ̄_K estimate: {row['estimate']:.4f}\n")
            f.write(f"  Standard error: {row['se']:.4f}\n")
            f.write(f"  Wald statistic: {row['wald_stat']:.4f}\n")
            f.write(f"  p-value: {row['p_value']:.4f} {sig}\n")
            f.write("\n")
            f.write("Interpretation:\n")
            if row['estimate'] < 0 and row['p_value'] < 0.05:
                f.write("  γ̄_K < 0 and significant: Locality is WEAKENING over time.\n")
                f.write("  This is consistent with increasing nationalization of elections.\n")
            elif row['estimate'] > 0 and row['p_value'] < 0.05:
                f.write("  γ̄_K > 0 and significant: Locality is STRENGTHENING over time.\n")
                f.write("  This suggests local primary timing effects are becoming more pronounced.\n")
            else:
                f.write("  No significant trend: Locality appears stable over time.\n")
                f.write("  The local primary-timing effect is neither strengthening nor weakening.\n")

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

    # Run locality hypothesis tests
    test1_results, test2_results = run_locality_tests(model2)

    # Save results
    save_results(model1, model1_coef, model2, model2_beta, model2_gamma, balance_diagnostic,
                 test1_results, test2_results)

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

    # Locality hypothesis tests summary
    print("\n" + "=" * 60)
    print("LOCALITY HYPOTHESIS TESTS")
    print("=" * 60)
    print(f"\nNear-primary window K = {LOCALITY_WINDOW}")

    print("\nTest 1: LI_y = 0 (is there locality in year y?)")
    print("-" * 40)
    for _, row in test1_results.iterrows():
        sig = '***' if row['p_value'] < 0.01 else ('**' if row['p_value'] < 0.05 else '')
        print(f"  {int(row['year'])}: LI = {row['LI_y']:.4f} (SE = {row['se']:.4f}), "
              f"p = {row['p_value']:.4f} {sig}")

    print("\nTest 2: γ̄_K = 0 (nationalization trend)")
    print("-" * 40)
    row = test2_results.iloc[0]
    sig = '***' if row['p_value'] < 0.01 else ('**' if row['p_value'] < 0.05 else '')
    print(f"  γ̄_K = {row['estimate']:.4f} (SE = {row['se']:.4f}), p = {row['p_value']:.4f} {sig}")
    if row['estimate'] < 0 and row['p_value'] < 0.05:
        print("  → Locality is WEAKENING over time (nationalization)")
    elif row['estimate'] > 0 and row['p_value'] < 0.05:
        print("  → Locality is STRENGTHENING over time")
    else:
        print("  → No significant trend in locality over time")

    print(f"\nOutputs saved to: {Q2_RESULTS_DIR}")


if __name__ == "__main__":
    main()
