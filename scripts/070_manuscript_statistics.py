"""
070_manuscript_statistics.py

Compute all in-text statistics needed for the manuscript that are not
already reported in regression output or figure captions.

Input:
    data/raw/html_parsed_candidates/{year}_house_candidates.csv  (2008-2024)
    data/processed_html_parsed/all_revisions.csv
    data/analysis/panel_weekly.parquet
    data/analysis/editor_cycle_registered.parquet

Output:
    data/analysis/manuscript_stats.json
    data/analysis/manuscript_stats.md
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_DIR = DATA_DIR / "analysis"
RAW_DIR = DATA_DIR / "raw" / "html_parsed_candidates"

CYCLES = [2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
THRESHOLD = 5  # editor activity threshold


# =============================================================================
# 1. Wikipedia coverage rate: fraction of on-ballot House candidates with pages
# =============================================================================

def compute_coverage_rate():
    """
    For each cycle, count total on-ballot House candidates and those
    with Wikipedia URLs. wikipedia_url is non-null iff the candidate
    was matched to a Wikipedia page via the election-result-page scraper.
    """
    rows = []
    for cycle in CYCLES:
        path = RAW_DIR / f"{cycle}_house_candidates.csv"
        if not path.exists():
            print(f"  WARNING: missing {path.name}")
            continue
        df = pd.read_csv(path)
        # Keep general-election candidates only (exclude special elections)
        general = df[df["election_type"].str.startswith("general", na=False)].copy()
        total = len(general)
        with_page = general["wikipedia_url"].notna().sum()
        # Incumbents vs non-incumbents
        inc = general[general["is_incumbent"] == True]
        chl = general[general["is_incumbent"] == False]
        rows.append({
            "cycle": cycle,
            "total_candidates": int(total),
            "with_wiki_page": int(with_page),
            "coverage_pct": round(100 * with_page / total, 1) if total > 0 else None,
            "incumbents_total": int(len(inc)),
            "incumbents_with_page": int(inc["wikipedia_url"].notna().sum()),
            "challengers_total": int(len(chl)),
            "challengers_with_page": int(chl["wikipedia_url"].notna().sum()),
        })
    df_cov = pd.DataFrame(rows)
    total_all = df_cov["total_candidates"].sum()
    total_with = df_cov["with_wiki_page"].sum()
    overall_pct = round(100 * total_with / total_all, 1)
    inc_all = df_cov["incumbents_total"].sum()
    inc_with = df_cov["incumbents_with_page"].sum()
    chl_all = df_cov["challengers_total"].sum()
    chl_with = df_cov["challengers_with_page"].sum()
    return {
        "by_cycle": df_cov.to_dict(orient="records"),
        "overall_pct": overall_pct,
        "overall_total": int(total_all),
        "overall_with_page": int(total_with),
        "incumbents_pct": round(100 * inc_with / inc_all, 1),
        "challengers_pct": round(100 * chl_with / chl_all, 1),
    }


# =============================================================================
# 2. Q2 regression sample sizes and centering-window statistics
# =============================================================================

def compute_q2_sample_stats():
    """
    Load the weekly panel and replicate the Q2 data preparation filters
    to report: total observations, near-window observations, baseline
    observations, number of unique calendar weeks, and per-page-cycle
    distribution of centering-window lengths.
    """
    panel = pd.read_parquet(ANALYSIS_DIR / "panel_weekly.parquet")
    print(f"  Full panel: {len(panel):,} rows")

    # Replicate Q2 filters (from 053_q2_regression_analysis.py)
    df = panel[panel["k_primary"].notna()].copy()
    df = df[df["state"] != "LA"].copy()
    df = df[df["election_cycle"] != 2020].copy()
    df = df[
        ((df["k_primary"] >= -18) & (df["k_primary"] <= -1)) |
        (df["k_primary"] <= -19)
    ].copy()
    df = df[df["Y_tilde_P"].notna()].copy()
    print(f"  Q2 regression sample: {len(df):,} rows")

    near = df[(df["k_primary"] >= -18) & (df["k_primary"] <= -1)]
    baseline = df[df["k_primary"] <= -19]

    # Unique calendar weeks in the full Q2 sample
    n_cal_weeks = df["week_start"].nunique()

    # Per-page-cycle centering-window lengths
    # The centering window is k_primary <= -19; its length varies by primary date
    baseline_lengths = (
        baseline.groupby(["page_title", "election_cycle"])["k_primary"].count()
    )

    return {
        "q2_total_obs": int(len(df)),
        "q2_near_window_obs": int(len(near)),
        "q2_baseline_obs": int(len(baseline)),
        "q2_n_page_cycles": int(near.groupby(["page_title", "election_cycle"]).ngroups),
        "q2_n_unique_calendar_weeks": int(n_cal_weeks),
        "centering_window_median_weeks": float(baseline_lengths.median()),
        "centering_window_min_weeks": int(baseline_lengths.min()),
        "centering_window_max_weeks": int(baseline_lengths.max()),
        "centering_window_mean_weeks": round(float(baseline_lengths.mean()), 1),
    }


# =============================================================================
# 3. Effect-size translation: NI = 0.091 log-units to edit counts
# =============================================================================

def compute_effect_size_translation():
    """
    NI_K = avg beta_k for k in {-8,...,-1}.
    Y = log(1 + edits), so edits = exp(Y) - 1.
    Reference mean Ybar = 0.33 (quoted in manuscript).
    NI = 0.091 => pre-election mean = 0.33 + 0.091 = 0.421.
    Translate both to expected edit counts.
    """
    panel = pd.read_parquet(ANALYSIS_DIR / "panel_weekly.parquet")
    # Reference window: k_general in {-70,...,-35}
    ref = panel[(panel["k_general"] >= -70) & (panel["k_general"] <= -35)]
    baseline_mean_Y = float(ref["Y_jt"].mean())

    # Pre-election window: k_general in {-8,...,-1}
    pre_el = panel[(panel["k_general"] >= -8) & (panel["k_general"] <= -1)]
    pre_el_mean_Y = float(pre_el["Y_jt"].mean())

    # NI scalar from regression (quoted in manuscript)
    NI = 0.0911
    # Expected edits at baseline and pre-election
    edits_baseline = np.exp(baseline_mean_Y) - 1
    edits_pre_el = np.exp(baseline_mean_Y + NI) - 1
    pct_increase = round(100 * (edits_pre_el - edits_baseline) / edits_baseline, 1)

    return {
        "baseline_mean_Y": round(baseline_mean_Y, 4),
        "pre_el_mean_Y": round(pre_el_mean_Y, 4),
        "NI_regression": NI,
        "edits_per_week_baseline": round(edits_baseline, 3),
        "edits_per_week_pre_election": round(edits_pre_el, 3),
        "pct_increase_edits": pct_increase,
    }


# =============================================================================
# 4. IP editor effective sample sizes per cycle
# =============================================================================

def compute_ip_sample_sizes():
    ec_ip = pd.read_parquet(ANALYSIS_DIR / "editor_cycle_ip.parquet")
    ec_reg = pd.read_parquet(ANALYSIS_DIR / "editor_cycle_registered.parquet")
    rows = []
    for cycle in CYCLES:
        n_ip = int((ec_ip[(ec_ip["election_cycle"] == cycle) &
                          (ec_ip["N_ey"] >= THRESHOLD)]).shape[0])
        n_reg = int((ec_reg[(ec_reg["election_cycle"] == cycle) &
                            (ec_reg["N_ey"] >= THRESHOLD)]).shape[0])
        rows.append({"cycle": cycle, "n_ip": n_ip, "n_reg": n_reg})
    df = pd.DataFrame(rows)
    return {
        "by_cycle": df.to_dict(orient="records"),
        "ip_min": int(df["n_ip"].min()),
        "ip_max": int(df["n_ip"].max()),
        "ip_min_cycle": int(df.loc[df["n_ip"].idxmin(), "cycle"]),
        "ip_max_cycle": int(df.loc[df["n_ip"].idxmax(), "cycle"]),
    }


# =============================================================================
# 5. Newcomer and regular counts (pooled and by cycle)
# =============================================================================

def compute_newcomer_regular_counts():
    ecw_reg = pd.read_parquet(ANALYSIS_DIR / "editor_cycle_window_registered.parquet")
    ecw_ip = pd.read_parquet(ANALYSIS_DIR / "editor_cycle_window_ip.parquet")

    def classify(ecw):
        baseline = (ecw[ecw["window"] == "baseline"]
                    .set_index(["user", "election_cycle"])[["N_DR"]]
                    .rename(columns={"N_DR": "N_DR_base"}))
        near = (ecw[ecw["window"] == "near_primary"]
                .set_index(["user", "election_cycle"])[["N_DR"]]
                .rename(columns={"N_DR": "N_DR_near"}))
        merged = near.join(baseline, how="left").fillna(0).reset_index()
        merged["editor_type"] = np.where(merged["N_DR_base"] == 0, "Newcomer", "Regular")
        return merged

    nc_reg = classify(ecw_reg)
    nc_ip = classify(ecw_ip)

    return {
        "registered_newcomers": int((nc_reg["editor_type"] == "Newcomer").sum()),
        "registered_regulars": int((nc_reg["editor_type"] == "Regular").sum()),
        "ip_newcomers": int((nc_ip["editor_type"] == "Newcomer").sum()),
        "ip_regulars": int((nc_ip["editor_type"] == "Regular").sum()),
        "registered_total_near_primary": int(len(nc_reg)),
        "ip_total_near_primary": int(len(nc_ip)),
    }


# =============================================================================
# 6. Q2 minimum detectable effect (power analysis)
# =============================================================================

def compute_mde():
    """
    Approximate MDE for the Q2 H1 test (main window K={-4,-3,-2,-1}).
    Using SE from Table 2: SE(beta_bar) = 0.0132 (main window).
    MDE at 80% power, alpha=0.05 (two-sided):
        MDE = (z_alpha/2 + z_beta) * SE = (1.96 + 0.842) * SE
    """
    SE_main = 0.0132   # from Table 2 in manuscript
    z_power = 0.842    # 80% power
    z_alpha = 1.960    # 5% two-sided
    mde = (z_alpha + z_power) * SE_main
    return {
        "se_main_window": SE_main,
        "mde_80pct_5pct": round(mde, 4),
        "note": (
            "MDE = (z_alpha/2 + z_beta) * SE = (1.960 + 0.842) * 0.0132. "
            "At 80% power and alpha=0.05, effects smaller than this MDE "
            "would be undetectable with the current sample."
        ),
    }


# =============================================================================
# 7. Q1 baseline mean and post-election burst for context
# =============================================================================

def compute_q1_context():
    """Compute the mean Y in post-election burst to provide scale context."""
    panel = pd.read_parquet(ANALYSIS_DIR / "panel_weekly.parquet")
    # Post-election burst: k_general > 0 (first ~4 weeks after election)
    post = panel[(panel["k_general"] > 0) & (panel["k_general"] <= 4)]
    post_mean_Y = float(post["Y_jt"].mean())
    ref = panel[(panel["k_general"] >= -70) & (panel["k_general"] <= -35)]
    ref_mean_Y = float(ref["Y_jt"].mean())
    pre_el = panel[(panel["k_general"] >= -8) & (panel["k_general"] <= -1)]
    pre_el_mean_Y = float(pre_el["Y_jt"].mean())
    return {
        "post_election_burst_mean_Y": round(post_mean_Y, 4),
        "reference_mean_Y": round(ref_mean_Y, 4),
        "pre_election_8wk_mean_Y": round(pre_el_mean_Y, 4),
        "post_burst_vs_ref_pct": round(
            100 * (np.exp(post_mean_Y) - np.exp(ref_mean_Y)) / (np.exp(ref_mean_Y) - 1), 1
        ),
        "pre_el_vs_ref_pct": round(
            100 * (np.exp(pre_el_mean_Y) - np.exp(ref_mean_Y)) / (np.exp(ref_mean_Y) - 1), 1
        ),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    stats = {}

    print("1. Computing coverage rates...")
    stats["coverage"] = compute_coverage_rate()

    print("2. Computing Q2 sample sizes...")
    stats["q2_sample"] = compute_q2_sample_stats()

    print("3. Computing effect-size translation...")
    stats["effect_size"] = compute_effect_size_translation()

    print("4. Computing IP sample sizes...")
    stats["ip_samples"] = compute_ip_sample_sizes()

    print("5. Computing newcomer/regular counts...")
    stats["newcomers"] = compute_newcomer_regular_counts()

    print("6. Computing MDE...")
    stats["mde"] = compute_mde()

    print("7. Computing Q1 context statistics...")
    stats["q1_context"] = compute_q1_context()

    # Save JSON
    out_json = ANALYSIS_DIR / "manuscript_stats.json"
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved: {out_json}")

    # Save human-readable markdown
    out_md = ANALYSIS_DIR / "manuscript_stats.md"
    with open(out_md, "w") as f:
        f.write("# Manuscript Statistics\n\n")
        f.write("Generated by `scripts/070_manuscript_statistics.py`.\n\n")

        f.write("## 1. Wikipedia Coverage Rate\n\n")
        cov = stats["coverage"]
        f.write(f"- Overall: {cov['overall_with_page']:,} of {cov['overall_total']:,} "
                f"on-ballot candidates have Wikipedia pages ({cov['overall_pct']}%)\n")
        f.write(f"- Incumbents: {cov['incumbents_pct']}%\n")
        f.write(f"- Challengers/non-incumbents: {cov['challengers_pct']}%\n\n")
        f.write("| Cycle | Total candidates | With Wikipedia page | Coverage |\n")
        f.write("|------:|-----------------:|--------------------:|---------:|\n")
        for row in cov["by_cycle"]:
            f.write(f"| {row['cycle']} | {row['total_candidates']:,} | "
                    f"{row['with_wiki_page']:,} | {row['coverage_pct']}% |\n")

        f.write("\n## 2. Q2 Sample Sizes\n\n")
        q2 = stats["q2_sample"]
        f.write(f"- Total observations (near window + baseline): {q2['q2_total_obs']:,}\n")
        f.write(f"- Near-window observations (k∈[-18,-1]): {q2['q2_near_window_obs']:,}\n")
        f.write(f"- Baseline observations (k≤-19): {q2['q2_baseline_obs']:,}\n")
        f.write(f"- Unique page-cycles: {q2['q2_n_page_cycles']:,}\n")
        f.write(f"- Unique calendar weeks: {q2['q2_n_unique_calendar_weeks']:,}\n")
        f.write(f"- Centering window (weeks per page-cycle): "
                f"median {q2['centering_window_median_weeks']:.0f}, "
                f"range [{q2['centering_window_min_weeks']}, {q2['centering_window_max_weeks']}]\n")

        f.write("\n## 3. Effect-Size Translation\n\n")
        es = stats["effect_size"]
        f.write(f"- Baseline mean Y (log edits): {es['baseline_mean_Y']}\n")
        f.write(f"- Baseline edits/page-week: {es['edits_per_week_baseline']:.3f}\n")
        f.write(f"- NI = 0.091 => pre-election edits/page-week: {es['edits_per_week_pre_election']:.3f}\n")
        f.write(f"- Percentage increase: ~{es['pct_increase_edits']}%\n")

        f.write("\n## 4. IP Editor Effective Sample Sizes\n\n")
        ip = stats["ip_samples"]
        f.write(f"- Range: {ip['ip_min']} ({ip['ip_min_cycle']}) to "
                f"{ip['ip_max']} ({ip['ip_max_cycle']})\n\n")
        f.write("| Cycle | IP editors (N≥5) | Registered editors (N≥5) |\n")
        f.write("|------:|-----------------:|-------------------------:|\n")
        for row in ip["by_cycle"]:
            f.write(f"| {row['cycle']} | {row['n_ip']:,} | {row['n_reg']:,} |\n")

        f.write("\n## 5. Newcomer / Regular Counts\n\n")
        nc = stats["newcomers"]
        f.write(f"- Registered: {nc['registered_newcomers']:,} newcomers, "
                f"{nc['registered_regulars']:,} regulars "
                f"(total near-primary: {nc['registered_total_near_primary']:,})\n")
        f.write(f"- IP: {nc['ip_newcomers']:,} newcomers, "
                f"{nc['ip_regulars']:,} regulars "
                f"(total near-primary: {nc['ip_total_near_primary']:,})\n")

        f.write("\n## 6. Q2 Minimum Detectable Effect\n\n")
        mde = stats["mde"]
        f.write(f"- SE (main window K={{-4,-3,-2,-1}}): {mde['se_main_window']}\n")
        f.write(f"- MDE at 80% power, α=0.05: {mde['mde_80pct_5pct']} log-units\n")
        f.write(f"- Note: {mde['note']}\n")

        f.write("\n## 7. Q1 Context: Scale of Pre-Election vs Post-Election Activity\n\n")
        ctx = stats["q1_context"]
        f.write(f"- Reference mean Y: {ctx['reference_mean_Y']}\n")
        f.write(f"- Pre-election (k∈[-8,-1]) mean Y: {ctx['pre_election_8wk_mean_Y']}\n")
        f.write(f"- Post-election burst (k∈[1,4]) mean Y: {ctx['post_election_burst_mean_Y']}\n")

    print(f"Saved: {out_md}")
    print("\nDone. Key statistics:")
    print(f"  Coverage overall: {stats['coverage']['overall_pct']}%")
    print(f"  Coverage incumbents: {stats['coverage']['incumbents_pct']}%")
    print(f"  Coverage challengers: {stats['coverage']['challengers_pct']}%")
    print(f"  Q2 total obs: {stats['q2_sample']['q2_total_obs']:,}")
    print(f"  Q2 page-cycles: {stats['q2_sample']['q2_n_page_cycles']:,}")
    print(f"  Q2 calendar weeks: {stats['q2_sample']['q2_n_unique_calendar_weeks']:,}")
    print(f"  Centering window median: {stats['q2_sample']['centering_window_median_weeks']:.0f} wks")
    print(f"  Effect size +{stats['effect_size']['pct_increase_edits']}% edits")
    print(f"  IP sample: {stats['ip_samples']['ip_min']}–{stats['ip_samples']['ip_max']}")
    print(f"  Newcomers (reg): {stats['newcomers']['registered_newcomers']:,}")
    print(f"  Regulars (reg): {stats['newcomers']['registered_regulars']:,}")
    print(f"  MDE: {stats['mde']['mde_80pct_5pct']} log-units")


if __name__ == "__main__":
    main()
