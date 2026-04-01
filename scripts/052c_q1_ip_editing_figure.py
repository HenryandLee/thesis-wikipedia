"""
052c_q1_ip_editing_figure.py

Appendix figure: IP (anonymous) editor trajectory over the election cycle,
analogous to Figure 1 Panel A but decomposing total edits into IP vs. registered.

Panel A: all edits (same as Figure 1 Panel A, shown for reference).
Panel B: IP-only edits (log(1 + anon_edits), fraction with at least 1 anon edit).

Uses the same k-axis, reference window (k in {-70,...,-36}), and annotation
style as Figure 1 for direct comparability.

Input:
    data/analysis/panel_weekly.parquet

Output:
    data/analysis/q1_results/figure_appendix_ip_editing.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
Q1_RESULTS_DIR = ANALYSIS_DIR / "q1_results"

# Match Figure 1 constants exactly
FULL_K_RANGE     = (-96, -1)  # k=0 excluded: election week
ZOOM_K_RANGE     = (-52, -1)
REF_PERIOD_LOWER = -70
REF_PERIOD_UPPER = -36   # descriptive reference line upper bound (matches Figure 1)


def load_panel() -> pd.DataFrame:
    path = ANALYSIS_DIR / "panel_weekly.parquet"
    logger.info(f"Loading panel from {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} observations")
    return df


def build_ip_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Add IP-specific outcome columns to the panel."""
    df = df.copy()
    df['Y_ip'] = np.log1p(df['anon_edits'])          # intensive: log(1 + anon_edits)
    df['P_ip'] = (df['anon_edits'] > 0).astype(int)  # extensive: any anon edit?
    return df


def ref_mean(df: pd.DataFrame, col: str) -> float:
    """Equal-weight mean of page-cycle means over the clean reference window."""
    ref = df[(df['k_general'] >= REF_PERIOD_LOWER) & (df['k_general'] <= REF_PERIOD_UPPER)]
    return ref.groupby('page_cycle')[col].mean().mean()


def full_cycle_means(df: pd.DataFrame, col: str) -> pd.Series:
    """Weekly means over the full two-year cycle."""
    sub = df[
        (df['k_general'] >= FULL_K_RANGE[0]) &
        (df['k_general'] <= FULL_K_RANGE[1]) &
        df[col].notna()
    ]
    return sub.groupby('k_general')[col].mean()


def _draw_panel(ax, means, ref_val, ylabel, title, color):
    """Draw a single Panel-A–style trajectory panel."""
    k = means.index.values

    ax.plot(k, means.values, color=color, linewidth=1.2)

    # Shade pre-election window
    ax.axvspan(ZOOM_K_RANGE[0], ZOOM_K_RANGE[1], alpha=0.12, color=color)
    ax.axvline(ZOOM_K_RANGE[0], color='gray', linestyle='--', linewidth=0.8)

    # Reference period dashed line
    ax.axhline(ref_val, color='gray', linestyle=':', linewidth=1.0, label=f'Ref. avg. = {ref_val:.3f}')
    ax.text(FULL_K_RANGE[0] - 0.5, ref_val,
            f'ref.\navg.', va='center', ha='right', fontsize=7.5, color='gray')

    # Annotate post-election burst
    y_max = means.values.max()
    spike_k = k[np.argmax(means.values)]
    ax.annotate(
        "Post-election\npage updating",
        xy=(spike_k, y_max),
        xytext=(spike_k + 12, y_max * 0.85),
        fontsize=8,
        arrowprops=dict(arrowstyle='->', color='dimgray', lw=1.0),
        color='dimgray',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
    )

    ax.set_xlabel("Weeks to General Election", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(FULL_K_RANGE[0] - 3, FULL_K_RANGE[1] + 3)
    ax.tick_params(labelsize=9)


def create_figure(df: pd.DataFrame):
    logger.info("Building IP editing appendix figure...")

    # Outcomes
    means_all = full_cycle_means(df, 'Y_jt')
    means_ip  = full_cycle_means(df, 'Y_ip')

    ref_all = ref_mean(df, 'Y_jt')
    ref_ip  = ref_mean(df, 'Y_ip')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    _draw_panel(
        axes[0], means_all, ref_all,
        ylabel="Mean $\\log(1+\\mathrm{edits})$",
        title="(A) All edits — 2008–2024 pooled average",
        color='steelblue'
    )
    _draw_panel(
        axes[1], means_ip, ref_ip,
        ylabel="Mean $\\log(1+\\mathrm{IP\\ edits})$",
        title="(B) IP (anonymous) edits only — 2008–2024 pooled average",
        color='#b5651d'
    )

    fig.tight_layout()

    out = Q1_RESULTS_DIR / "figure_appendix_ip_editing.png"
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved to {out}")

    # Report summary statistics
    ref_all_val = ref_mean(df, 'Y_jt')
    ref_ip_val  = ref_mean(df, 'Y_ip')
    peak_all = means_all.max()
    peak_ip  = means_ip.max()
    # Pre-election window peak (k in {-8,...,-1})
    pre_all = means_all[(means_all.index >= -8) & (means_all.index <= -1)].mean()
    pre_ip  = means_ip[(means_ip.index  >= -8) & (means_ip.index  <= -1)].mean()

    logger.info("--- Summary ---")
    logger.info(f"All edits  — ref avg: {ref_all_val:.4f}, cycle peak: {peak_all:.4f}, "
                f"pre-election avg (k=-8..-1): {pre_all:.4f}, "
                f"relative rise: {(pre_all/ref_all_val - 1)*100:.1f}%")
    logger.info(f"IP edits   — ref avg: {ref_ip_val:.4f},  cycle peak: {peak_ip:.4f},  "
                f"pre-election avg (k=-8..-1): {pre_ip:.4f}, "
                f"relative rise: {(pre_ip/ref_ip_val - 1)*100:.1f}%")


def main():
    df = load_panel()
    df = build_ip_outcomes(df)
    create_figure(df)
    logger.info("Done.")


if __name__ == "__main__":
    main()
