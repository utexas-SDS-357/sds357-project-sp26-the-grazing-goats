"""
Reproduce all EDA visualizations for the NC Traffic Stops project.
Reads: data/nc_traffic_stops_cleaned.parquet (or .csv fallback)
Writes: 17 PNG plots, 2 CSV tables, and eda_summary.md into eda/
"""

import os
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ── Configuration ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "eda"
OUT_DIR.mkdir(exist_ok=True)

PARQUET = DATA_DIR / "nc_traffic_stops_cleaned.parquet"
CSV = DATA_DIR / "nc_traffic_stops_cleaned.csv"

MAIN_RACES = ["black", "white", "hispanic", "asian/pacific islander"]
ALL_RACES = ["black", "white", "hispanic", "asian/pacific islander", "other", "unknown"]
RACE_PALETTE = {
    "black": sns.color_palette()[0],
    "white": sns.color_palette()[1],
    "hispanic": sns.color_palette()[2],
    "asian/pacific islander": sns.color_palette()[3],
    "other": sns.color_palette()[4],
    "unknown": sns.color_palette()[5],
}
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CITY_ORDER = ["Charlotte", "Raleigh", "Greensboro", "Fayetteville", "Winston-Salem", "Durham"]

DPI = 150
FIGSIZE = (10, 6)

plt.rcParams.update({
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})


def load_data() -> pd.DataFrame:
    if PARQUET.exists():
        print(f"Loading {PARQUET} ...")
        df = pd.read_parquet(PARQUET)
    elif CSV.exists():
        print(f"Loading {CSV} ...")
        df = pd.read_csv(CSV, low_memory=False)
    else:
        raise FileNotFoundError("No cleaned dataset found in data/")
    print(f"  {len(df):,} rows, {len(df.columns)} columns")
    return df


def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1e6:.1f}M"
    if n >= 1_000:
        return f"{n/1e3:.0f}K"
    return str(n)


def _binomial_ci(k, n, z=1.96):
    """Wilson score 95% confidence interval for a proportion."""
    if n == 0:
        return 0, 0, 0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return p, max(0, centre - margin), min(1, centre + margin)


# ── Plot functions ───────────────────────────────────────────────────────────

def plot_01_outcome_distribution(df):
    valid = df[~df["outcome_missing"]].copy()
    vc = valid["outcome"].value_counts()
    order = ["citation", "warning", "arrest"]
    counts = [vc.get(o, 0) for o in order]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(order, counts, color=sns.color_palette()[:3])
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{c:,}", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Number of Stops")
    ax.set_title("Overall Stop Outcome Distribution")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt_count(int(x))))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "01_outcome_distribution.png", dpi=DPI)
    plt.close(fig)
    print("  01_outcome_distribution.png")
    return vc


def plot_02_outcome_by_race(df):
    valid = df[(~df["outcome_missing"]) & (df["subject_race"].isin(MAIN_RACES))].copy()
    ct = pd.crosstab(valid["subject_race"], valid["outcome"], normalize="index") * 100
    ct = ct.reindex(index=MAIN_RACES, columns=["arrest", "citation", "warning"])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ct.plot(kind="bar", stacked=True, ax=ax,
            color=[sns.color_palette()[0], sns.color_palette()[1], sns.color_palette()[2]],
            width=0.7)
    ax.set_ylabel("Percentage of Stops")
    ax.set_xlabel("Race")
    ax.set_title("Stop Outcome Distribution by Race (Proportional)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Outcome")
    ax.set_ylim(0, 100)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "02_outcome_by_race.png", dpi=DPI)
    plt.close(fig)
    print("  02_outcome_by_race.png")


def plot_03_arrest_rate_by_race(df):
    sub = df[df["subject_race"].isin(MAIN_RACES)].copy()
    rates, lows, highs = [], [], []
    for race in MAIN_RACES:
        g = sub[sub["subject_race"] == race]
        p, lo, hi = _binomial_ci(g["arrested"].sum(), len(g))
        rates.append(p * 100)
        lows.append((p - lo) * 100)
        highs.append((hi - p) * 100)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(MAIN_RACES, rates, yerr=[lows, highs], capsize=5,
                  color=sns.color_palette()[:4])
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                f"{r:.2f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Arrest Rate (%)")
    ax.set_xlabel("Race")
    ax.set_title("Arrest Rate by Race (with 95% CI)")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "03_arrest_rate_by_race.png", dpi=DPI)
    plt.close(fig)
    print("  03_arrest_rate_by_race.png")


def plot_04_arrest_rate_by_city(df):
    rates, lows, highs = [], [], []
    for city in CITY_ORDER:
        g = df[df["city"] == city]
        p, lo, hi = _binomial_ci(g["arrested"].sum(), len(g))
        rates.append(p * 100)
        lows.append((p - lo) * 100)
        highs.append((hi - p) * 100)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(CITY_ORDER, rates, yerr=[lows, highs], capsize=5,
                  color=sns.color_palette()[:6])
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{r:.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Arrest Rate (%)")
    ax.set_xlabel("City")
    ax.set_title("Arrest Rate by City (with 95% CI)")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "04_arrest_rate_by_city.png", dpi=DPI)
    plt.close(fig)
    print("  04_arrest_rate_by_city.png")


def plot_05_heatmap_race_city(df):
    sub = df[df["subject_race"].isin(MAIN_RACES)].copy()
    pivot = sub.groupby(["city", "subject_race"])["arrested"].mean().unstack() * 100
    pivot = pivot.reindex(index=CITY_ORDER, columns=MAIN_RACES)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "Arrest Rate (%)"}, linewidths=0.5)
    ax.grid(False)
    ax.set_title("Arrest Rate (%) by Race × City")
    ax.set_xlabel("Race")
    ax.set_ylabel("City")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "05_arrest_rate_race_city_heatmap.png", dpi=DPI)
    plt.close(fig)
    print("  05_arrest_rate_race_city_heatmap.png")


def plot_06_outcome_by_reason(df):
    valid = df[~df["outcome_missing"]].copy()
    top10 = valid["reason_for_stop"].value_counts().head(10).index.tolist()
    sub = valid[valid["reason_for_stop"].isin(top10)]
    ct = pd.crosstab(sub["reason_for_stop"], sub["outcome"], normalize="index") * 100
    ct = ct.reindex(index=top10, columns=["arrest", "citation", "warning"])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ct.plot(kind="barh", stacked=True, ax=ax,
            color=[sns.color_palette()[0], sns.color_palette()[1], sns.color_palette()[2]])
    ax.set_xlabel("Percentage of Stops")
    ax.set_title("Stop Outcome by Reason for Stop (Top 10)")
    ax.legend(title="Outcome")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "06_outcome_by_reason.png", dpi=DPI)
    plt.close(fig)
    print("  06_outcome_by_reason.png")


def plot_07_arrest_rate_reason_race(df):
    sub = df[df["subject_race"].isin(MAIN_RACES)].copy()
    top6 = sub["reason_for_stop"].value_counts().head(6).index.tolist()
    sub = sub[sub["reason_for_stop"].isin(top6)]
    rates = sub.groupby(["reason_for_stop", "subject_race"])["arrested"].mean() * 100
    rates = rates.unstack()
    rates = rates.reindex(index=top6, columns=MAIN_RACES)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    rates.plot(kind="bar", ax=ax, color=[RACE_PALETTE[r] for r in MAIN_RACES], width=0.75)
    ax.set_ylabel("Arrest Rate (%)")
    ax.set_xlabel("Reason for Stop")
    ax.set_title("Arrest Rate by Reason for Stop, Colored by Race (Top 6 Reasons)")
    ax.legend(title="Race")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "07_arrest_rate_reason_race.png", dpi=DPI)
    plt.close(fig)
    print("  07_arrest_rate_reason_race.png")


def plot_08_stops_per_year_city(df):
    counts = df.groupby(["year", "city"]).size().unstack()
    counts = counts.reindex(columns=CITY_ORDER)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for city in CITY_ORDER:
        ax.plot(counts.index, counts[city], marker="o", markersize=4, label=city)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Stops")
    ax.set_title("Number of Stops per Year by City")
    ax.legend(title="City")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt_count(int(x))))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "08_stops_per_year_city.png", dpi=DPI)
    plt.close(fig)
    print("  08_stops_per_year_city.png")


def plot_09_arrest_rate_time_race(df):
    sub = df[df["subject_race"].isin(MAIN_RACES)].copy()
    rates = sub.groupby(["year", "subject_race"])["arrested"].mean().unstack() * 100
    rates = rates.reindex(columns=MAIN_RACES)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for race in MAIN_RACES:
        ax.plot(rates.index, rates[race], marker="o", markersize=4,
                label=race, color=RACE_PALETTE[race])
    ax.set_xlabel("Year")
    ax.set_ylabel("Arrest Rate (%)")
    ax.set_title("Arrest Rate Over Time by Race")
    ax.legend(title="Race")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "09_arrest_rate_time_race.png", dpi=DPI)
    plt.close(fig)
    print("  09_arrest_rate_time_race.png")


def plot_10_stops_by_hour(df):
    sub = df[df["subject_race"].isin(MAIN_RACES) & df["hour"].notna()].copy()
    sub["hour_int"] = sub["hour"].astype(int)
    counts = sub.groupby(["hour_int", "subject_race"]).size().unstack()
    counts = counts.reindex(columns=MAIN_RACES)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for race in MAIN_RACES:
        ax.plot(counts.index, counts[race], label=race, color=RACE_PALETTE[race])
    ax.set_xlabel("Hour (0 = midnight)")
    ax.set_ylabel("Number of Stops")
    ax.set_title("Stop Volume by Hour of Day, Split by Race")
    ax.set_xticks(range(24))
    ax.legend(title="Race")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt_count(int(x))))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "10_stops_by_hour.png", dpi=DPI)
    plt.close(fig)
    print("  10_stops_by_hour.png")


def plot_10b_stops_by_hour_weighted(df):
    sub = df[df["subject_race"].isin(MAIN_RACES) & df["hour"].notna()].copy()
    sub["hour_int"] = sub["hour"].astype(int)
    counts = sub.groupby(["hour_int", "subject_race"]).size().unstack()
    counts = counts.reindex(columns=MAIN_RACES)
    pcts = counts.div(counts.sum(axis=0), axis=1) * 100

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for race in MAIN_RACES:
        ax.plot(pcts.index, pcts[race], label=race, color=RACE_PALETTE[race])
    ax.set_xlabel("Hour (0 = midnight)")
    ax.set_ylabel("% of Race's Total Stops")
    ax.set_title("Share of Each Race's Stops by Hour of Day (Population-Weighted)")
    ax.set_xticks(range(24))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(title="Race")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "10b_stops_by_hour_race_weighted.png", dpi=DPI)
    plt.close(fig)
    print("  10b_stops_by_hour_race_weighted.png")


def plot_11_stops_by_day_of_week(df):
    vc = df["day_of_week"].value_counts().reindex(DAY_ORDER)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(DAY_ORDER, vc.values, color=sns.color_palette()[:7])
    for bar, c in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                _fmt_count(c), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Number of Stops")
    ax.set_title("Stop Volume by Day of Week")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt_count(int(x))))
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "11_stops_by_day_of_week.png", dpi=DPI)
    plt.close(fig)
    print("  11_stops_by_day_of_week.png")


def plot_12_race_distribution_by_city(df):
    ct = pd.crosstab(df["city"], df["subject_race"], normalize="index") * 100
    ct = ct.reindex(index=CITY_ORDER, columns=ALL_RACES)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ct.plot(kind="bar", stacked=True, ax=ax,
            color=[RACE_PALETTE[r] for r in ALL_RACES], width=0.7)
    ax.set_ylabel("Percentage")
    ax.set_xlabel("City")
    ax.set_title("Race Distribution of Stopped Drivers by City")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Race", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(0, 100)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "12_race_distribution_by_city.png", dpi=DPI)
    plt.close(fig)
    print("  12_race_distribution_by_city.png")


def plot_13_age_distribution_race(df):
    sub = df[df["subject_race"].isin(MAIN_RACES) & df["subject_age"].notna()].copy()
    sub = sub[(sub["subject_age"] >= 10) & (sub["subject_age"] <= 85)]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for race in MAIN_RACES:
        data = sub.loc[sub["subject_race"] == race, "subject_age"]
        data.plot.kde(ax=ax, label=race, color=RACE_PALETTE[race], bw_method=0.15)
    ax.set_xlabel("Age")
    ax.set_ylabel("Density")
    ax.set_title("Age Distribution of Stopped Drivers by Race")
    ax.set_xlim(10, 85)
    ax.legend(title="Race")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "13_age_distribution_race.png", dpi=DPI)
    plt.close(fig)
    print("  13_age_distribution_race.png")


def plot_14_sex_distribution_outcome(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)
    fig.suptitle("Sex Distribution Overall and by Outcome", fontsize=13, fontweight="bold")

    sex_vc = df["subject_sex"].value_counts().reindex(["male", "female", "unknown"])
    ax1.bar(sex_vc.index, sex_vc.values, color=sns.color_palette()[:3])
    ax1.set_title("Overall Sex Distribution")
    ax1.set_ylabel("Number of Stops")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt_count(int(x))))

    valid = df[~df["outcome_missing"]].copy()
    ct = pd.crosstab(valid["outcome"], valid["subject_sex"], normalize="index") * 100
    outcomes = ["arrest", "citation", "warning"]
    sexes = ["female", "male", "unknown"]
    ct = ct.reindex(index=outcomes, columns=sexes)
    ct.plot(kind="bar", ax=ax2, width=0.7)
    ax2.set_title("Sex Distribution by Outcome")
    ax2.set_ylabel("Percentage")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    ax2.set_xlabel("outcome")
    ax2.legend(title="Sex")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "14_sex_distribution_outcome.png", dpi=DPI)
    plt.close(fig)
    print("  14_sex_distribution_outcome.png")


def plot_15_search_rate_by_race(df):
    sub = df[df["subject_race"].isin(MAIN_RACES)].copy()
    rates = sub.groupby("subject_race")["search_conducted"].mean() * 100
    rates = rates.reindex(MAIN_RACES)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(MAIN_RACES, rates.values, color=sns.color_palette()[:4])
    for bar, r in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{r:.2f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Search Rate (%)")
    ax.set_xlabel("Race")
    ax.set_title("Search Rate by Race")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "15_search_rate_by_race.png", dpi=DPI)
    plt.close(fig)
    print("  15_search_rate_by_race.png")


def plot_16_arrest_rate_search_status_race(df):
    sub = df[df["subject_race"].isin(MAIN_RACES)].copy()
    sub["searched_label"] = sub["search_conducted"].map({True: "Searched", False: "Not Searched"})
    rates = sub.groupby(["subject_race", "searched_label"])["arrested"].mean() * 100
    rates = rates.unstack()
    rates = rates.reindex(index=MAIN_RACES, columns=["Not Searched", "Searched"])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(MAIN_RACES))
    w = 0.35
    bars1 = ax.bar(x - w / 2, rates["Not Searched"], w, label="Not Searched", color="skyblue")
    bars2 = ax.bar(x + w / 2, rates["Searched"], w, label="Searched", color="indianred")
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(MAIN_RACES)
    ax.set_ylabel("Arrest Rate (%)")
    ax.set_xlabel("Race")
    ax.set_title("Arrest Rate: Searched vs. Not Searched, by Race")
    ax.legend(title="Search Conducted")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "16_arrest_rate_search_status_race.png", dpi=DPI)
    plt.close(fig)
    print("  16_arrest_rate_search_status_race.png")


def plot_17_correlation_heatmap(df):
    num_cols = ["arrested", "search_conducted", "subject_age", "hour", "year"]
    sub = df[num_cols].copy()
    sub["search_conducted"] = sub["search_conducted"].astype(float)
    corr = sub.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                mask=mask, square=True, ax=ax,
                vmin=-0.3, vmax=0.3,
                cbar_kws={"label": "Pearson r"}, linewidths=0.5)
    ax.grid(False)
    ax.set_title("Correlation Heatmap of Numeric/Binary Variables")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "17_correlation_heatmap.png", dpi=DPI)
    plt.close(fig)
    print("  17_correlation_heatmap.png")


def export_csv_tables(df):
    sub = df[df["subject_race"].isin(MAIN_RACES)].copy()

    race_sex = sub.groupby(["subject_race", "subject_sex"])["arrested"].agg(["sum", "count"])
    race_sex.columns = ["arrests", "total"]
    race_sex["arrest_rate"] = (race_sex["arrests"] / race_sex["total"] * 100).round(2)
    race_sex.to_csv(OUT_DIR / "arrest_rates_race_sex.csv")
    print("  arrest_rates_race_sex.csv")

    race_reason = sub.groupby(["subject_race", "reason_for_stop"])["arrested"].agg(["sum", "count"])
    race_reason.columns = ["arrests", "total"]
    race_reason["arrest_rate"] = (race_reason["arrests"] / race_reason["total"] * 100).round(2)
    race_reason.to_csv(OUT_DIR / "arrest_rates_race_reason.csv")
    print("  arrest_rates_race_reason.csv")


# ── Summary statistics for eda_summary.md ────────────────────────────────────

def compute_summary_stats(df):
    """Compute all statistics referenced in the EDA summary."""
    s = {}
    total = len(df)
    s["total_rows"] = total

    valid = df[~df["outcome_missing"]]
    vc = valid["outcome"].value_counts()
    for o in ["citation", "warning", "arrest"]:
        s[f"outcome_{o}_n"] = vc.get(o, 0)
        s[f"outcome_{o}_pct"] = vc.get(o, 0) / len(valid) * 100
    s["outcome_missing_n"] = df["outcome_missing"].sum()
    s["outcome_missing_pct"] = s["outcome_missing_n"] / total * 100

    main = df[df["subject_race"].isin(MAIN_RACES)]

    # arrest rates by race
    for race in MAIN_RACES:
        g = main[main["subject_race"] == race]
        s[f"arrest_rate_{race}"] = g["arrested"].mean() * 100

    # arrest rates by city
    for city in CITY_ORDER:
        g = df[df["city"] == city]
        s[f"arrest_rate_{city}"] = g["arrested"].mean() * 100

    # search rates by race
    for race in MAIN_RACES:
        g = main[main["subject_race"] == race]
        s[f"search_rate_{race}"] = g["search_conducted"].mean() * 100

    # arrest rate searched vs not by race
    for race in MAIN_RACES:
        g = main[main["subject_race"] == race]
        searched = g[g["search_conducted"] == True]
        not_searched = g[g["search_conducted"] == False]
        s[f"arrest_searched_{race}"] = searched["arrested"].mean() * 100 if len(searched) else 0
        s[f"arrest_not_searched_{race}"] = not_searched["arrested"].mean() * 100 if len(not_searched) else 0

    # time-based
    time_race = main.groupby(["year", "subject_race"])["arrested"].mean() * 100
    s["hisp_2001_arrest"] = time_race.get((2001, "hispanic"), np.nan)

    # sex
    sex_vc = df["subject_sex"].value_counts()
    s["male_pct"] = sex_vc.get("male", 0) / total * 100

    # correlation
    num_cols = ["arrested", "search_conducted", "subject_age", "hour", "year"]
    sub = df[num_cols].copy()
    sub["search_conducted"] = sub["search_conducted"].astype(float)
    corr = sub.corr()
    s["corr_search_arrested"] = corr.loc["search_conducted", "arrested"]

    # missing time
    s["missing_time_pct"] = df["hour"].isna().mean() * 100

    # cities
    n_cities = len(df["city"].unique())
    s["n_cities"] = n_cities

    # year range
    s["year_min"] = int(df["year"].min())
    s["year_max"] = int(df["year"].max())

    return s


def generate_summary_md(df, stats):
    r = lambda k: f"{stats.get(k, 0):.1f}"
    r2 = lambda k: f"{stats.get(k, 0):.2f}"

    md = textwrap.dedent(f"""\
    # Exploratory Data Analysis Summary — NC Traffic Stops

    > **Dataset**: `nc_traffic_stops_cleaned.csv` — {stats['total_rows']:,} traffic stops across {stats['n_cities']} NC cities ({stats['year_min']}–{stats['year_max']})
    > **Generated by**: `eda_analysis.py`

    ---

    ## Output Files

    | # | File | Description |
    |---|---|---|
    | 1 | `01_outcome_distribution.png` | Overall stop outcome bar chart |
    | 2 | `02_outcome_by_race.png` | Proportional outcome breakdown by race |
    | 3 | `03_arrest_rate_by_race.png` | Arrest rate per racial group with 95% CI |
    | 4 | `04_arrest_rate_by_city.png` | Arrest rate per city with 95% CI |
    | 5 | `05_arrest_rate_race_city_heatmap.png` | Arrest rate heatmap: race × city |
    | 6 | `06_outcome_by_reason.png` | Outcome distribution by top 10 stop reasons |
    | 7 | `07_arrest_rate_reason_race.png` | Arrest rate by reason × race (top 6 reasons) |
    | 8 | `08_stops_per_year_city.png` | Stop volume per year by city |
    | 9 | `09_arrest_rate_time_race.png` | Arrest rate trend over time by race |
    | 10 | `10_stops_by_hour.png` | Hourly stop volume split by race |
    | 11 | `11_stops_by_day_of_week.png` | Stop volume by day of week |
    | 12 | `12_race_distribution_by_city.png` | Proportional race composition of stops per city |
    | 13 | `13_age_distribution_race.png` | KDE of stopped driver age by race |
    | 14 | `14_sex_distribution_outcome.png` | Sex distribution overall and by outcome |
    | 15 | `15_search_rate_by_race.png` | Search rate per racial group |
    | 16 | `16_arrest_rate_search_status_race.png` | Arrest rate by search status × race |
    | 17 | `17_correlation_heatmap.png` | Pearson correlation of numeric/binary variables |
    | 18a | `arrest_rates_race_sex.csv` | Arrest rates disaggregated by race × sex |
    | 18b | `arrest_rates_race_reason.csv` | Arrest rates disaggregated by race × stop reason |

    ---

    ## Key Findings by Plot

    ### Plot 1 — Overall Outcome Distribution

    Citations are the most common outcome ({r('outcome_citation_pct')}%), followed by warnings ({r('outcome_warning_pct')}%) and arrests ({r('outcome_arrest_pct')}%). About {stats['outcome_missing_n']:,} stops (~{r('outcome_missing_pct')}%) have missing outcome data. The low arrest rate means the outcome variable is heavily imbalanced — relevant for any predictive modeling.

    ### Plot 2 — Outcome Distribution by Race

    Hispanic drivers have the highest arrest proportion and the highest citation rate, but the lowest warning rate — suggesting officers are less likely to let them off with warnings. Black drivers have a moderate arrest share and the highest warning rate. Asian/Pacific Islander drivers have the lowest arrest proportion.

    ### Plot 3 — Arrest Rate by Race (with 95% CI)

    Hispanic drivers face the highest arrest rate at {r2('arrest_rate_hispanic')}%, followed by Black drivers at {r2('arrest_rate_black')}%, white drivers at {r2('arrest_rate_white')}%, and Asian/Pacific Islander drivers at {r2('arrest_rate_asian/pacific islander')}%. Confidence intervals are extremely narrow due to sample sizes in the hundreds of thousands to millions, making all pairwise differences statistically significant. Hispanic drivers are arrested at {stats['arrest_rate_hispanic']/stats['arrest_rate_white']:.1f}× the rate of white drivers; Black drivers at {stats['arrest_rate_black']/stats['arrest_rate_white']:.1f}× the white rate.

    ### Plot 4 — Arrest Rate by City

    Charlotte has the highest arrest rate ({r2('arrest_rate_Charlotte')}%), followed by Raleigh ({r2('arrest_rate_Raleigh')}%), Durham ({r2('arrest_rate_Durham')}%), Greensboro ({r2('arrest_rate_Greensboro')}%), and Fayetteville ({r2('arrest_rate_Fayetteville')}%). Winston-Salem has the lowest rate ({r2('arrest_rate_Winston-Salem')}%). Charlotte's elevated rate may reflect its status as the largest city with the most diverse stop portfolio.

    ### Plot 5 — Arrest Rate Heatmap: Race × City

    The racial hierarchy (Hispanic > Black > White > Asian/PI) holds in every city, though magnitudes vary. Charlotte shows the widest absolute gaps. Winston-Salem has the narrowest gaps but the same ranking. The consistency of the pattern across six independent police departments is notable.

    ### Plot 6 — Outcome by Reason for Stop (Top 10)

    Speed limit violations are the most common reason, followed by regulatory and equipment violations. "Other Motor Vehicle Violation" and investigation-type stops have the highest arrest rates among common stop reasons. Equipment and regulatory violations mostly result in citations. Safe movement violations produce a relatively high share of warnings.

    ### Plot 7 — Arrest Rate by Reason for Stop × Race

    The racial disparity appears across every stop reason category. Even for the lowest-arrest stop reasons, the racial gap is present, indicating it is not an artifact of differential stop reasons.

    ### Plot 8 — Stops per Year by City

    Charlotte dominates in volume. Most cities show declining stop volumes after 2010–2012. Durham and Raleigh have data starting from 2002–2003, not 2000 — earlier years show near-zero counts, reflecting data availability rather than policing changes.

    ### Plot 9 — Arrest Rate Over Time by Race

    After 2004, all groups show a gradual downward trend in arrest rates. The Hispanic > Black > White ranking persists across the entire period. By 2015, the gap has narrowed somewhat but remains.

    ### Plot 10 — Stop Volume by Hour of Day

    Stops peak during daytime hours (10 AM – 4 PM) across all racial groups, consistent with higher traffic volume. There is a secondary evening peak around 8–10 PM. Late-night stops (midnight–5 AM) are lower in volume. The racial distribution of stops is roughly constant across hours, with Black drivers forming the largest group at most hours.

    ### Plot 11 — Stop Volume by Day of Week

    Weekdays see significantly higher stop volumes than weekends, consistent with commuting patterns. Monday through Thursday are roughly equal; Friday is slightly elevated. Saturday and Sunday drop substantially, with Sunday being the lowest-volume day.

    ### Plot 12 — Race Distribution of Stopped Drivers by City

    Durham and Fayetteville have the highest proportion of Black drivers stopped (>55%). Charlotte has the largest Hispanic share, likely reflecting local demographics. Comparing these proportions to Census data would be necessary to determine if any group is over- or under-represented relative to their population.

    ### Plot 13 — Age Distribution by Race

    All groups show a right-skewed age distribution peaking in the mid-20s. Hispanic drivers skew younger, with a sharper peak around age 22–25. White drivers have a flatter, wider distribution with relatively more stops among 40–60 year-olds.

    ### Plot 14 — Sex Distribution Overall and by Outcome

    Males constitute ~{r('male_pct')}% of all stops and are overrepresented in arrests. Female drivers are more likely to receive warnings relative to their share of stops.

    ### Plot 15 — Search Rate by Race

    Black drivers are searched at {r2('search_rate_black')}% of stops — more than double the rate for white drivers ({r2('search_rate_white')}%). Hispanic drivers are searched at {r2('search_rate_hispanic')}%. Asian/Pacific Islander drivers have the lowest search rate at {r2('search_rate_asian/pacific islander')}%. This large disparity in search rates is a critical factor: if searches drive arrests, then disparate search rates mechanically produce disparate arrest rates.

    ### Plot 16 — Arrest Rate: Searched vs. Not Searched, by Race

    Being searched dramatically increases the probability of arrest: from <2% to 34–54%. Among searched drivers, Hispanic drivers have the highest post-search arrest rate ({r('arrest_searched_hispanic')}%), followed by White ({r('arrest_searched_white')}%), Asian/PI ({r('arrest_searched_asian/pacific islander')}%), then Black ({r('arrest_searched_black')}%). The lower post-search arrest rate for Black drivers suggests searches of Black drivers may have a lower evidentiary threshold — a pattern consistent with the "outcome test" literature on racial profiling.

    ### Plot 17 — Correlation Heatmap

    `search_conducted` has the strongest correlation with `arrested` (~{stats['corr_search_arrested']:.1f}), confirming searches as the most important individual predictor. Age has a slight negative correlation with arrest. Year and hour show near-zero correlations, suggesting temporal features alone are weak linear predictors but may contribute through interactions.

    ### Plot 18 — Arrest Rate Tables (CSV)

    The race × sex CSV shows male drivers of every race have higher arrest rates than female counterparts. The race × reason CSV provides the fully disaggregated rates needed for pre-modeling feature selection and confirms that disparities persist within each stop reason.

    ---

    ## Preliminary Takeaways for Project Objectives

    ### Objective 1: Describe patterns in stop outcomes across race, location, and stop context

    - **Racial disparities are clear and consistent**: Hispanic drivers face the highest arrest rate ({r2('arrest_rate_hispanic')}%), followed by Black ({r2('arrest_rate_black')}%), White ({r2('arrest_rate_white')}%), and Asian/PI ({r2('arrest_rate_asian/pacific islander')}%). This ranking holds across all six cities and virtually all stop reasons.
    - **City-level variation exists** but does not eliminate the racial gap — Charlotte has the highest rates and widest gaps, Winston-Salem the lowest, but the ranking is the same everywhere (Plot 5).
    - **Stop reason matters but doesn't explain the gap**: The arrest rate varies across stop reasons, but racial gaps persist within each category (Plot 7).
    - **Search rates are highly disparate**: Black drivers are searched at {stats['search_rate_black']/stats['search_rate_white']:.1f}× the White rate ({r2('search_rate_black')}% vs. {r2('search_rate_white')}%). Since search is the strongest predictor of arrest, this is a key mechanism.

    ### Objective 2: Predict arrest escalation

    - **Class imbalance**: Only {r('outcome_arrest_pct')}% of stops result in arrest. Models will need SMOTE, cost-sensitive learning, or evaluation via AUC-PR / F1 rather than accuracy.
    - **Strongest single predictor**: `search_conducted` — arrest probability jumps from ~1% to ~40% when a search occurs.
    - **Useful features**: Race, city, reason for stop, age, sex, and search status all show meaningful variation in arrest rates. Hour and year are weaker individually but may contribute as interactions.
    - **Feature engineering ideas**: Race × search interaction, reason × race, time-of-day buckets (night vs. day), age bins.

    ### Objective 3: Infer whether demographic disparities persist after controlling for situational factors

    - **Unadjusted disparities are large and consistent**: The descriptive EDA establishes the baseline racial gap.
    - **Within-reason analysis (Plot 7)**: Disparities hold within each stop reason, suggesting they are not fully explained by stop context.
    - **Search as a mediator**: The dramatic effect of search on arrest (Plot 16) combined with disparate search rates (Plot 15) points to search decisions as a key pathway. The "outcome test" finding (lower post-search arrest rates for Black drivers) suggests possible over-searching of Black drivers.
    - **Recommended modeling approaches**: Logistic regression with race interactions, propensity score matching on situational factors, or Blinder-Oaxaca decomposition. A two-stage model (search decision → arrest | search) could disentangle direct and indirect race effects on arrest outcomes.

    ---

    ## Data Quality Notes

    1. **Missing time data** (~{r('missing_time_pct')}%): Hour-based analyses under-represent cities with higher missingness.
    2. **Missing outcomes** (~{r('outcome_missing_pct')}%): {stats['outcome_missing_n']:,} rows lack outcome data. Flagged via `outcome_missing` and retained, but should be excluded from outcome-based models.
    3. **Contraband data unavailable**: Dropped due to >95% missingness. This prevents the canonical "hit rate" test for racial profiling.
    4. **Early-year data sparsity**: Some cities have near-zero data before 2002. Consider filtering to 2002+ for temporal analyses.
    5. **No Census benchmarking**: We cannot assess whether stop *volumes* are disproportionate without population denominators per city.

    ---

    *This summary accompanies the EDA visualizations in the `eda/` folder and feeds into the modeling phase of the SDS 357 project.*
    """)
    md_path = OUT_DIR / "eda_summary.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  eda_summary.md")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("EDA Analysis — NC Traffic Stops")
    print("=" * 60)

    df = load_data()

    print("\nGenerating plots...")
    plot_01_outcome_distribution(df)
    plot_02_outcome_by_race(df)
    plot_03_arrest_rate_by_race(df)
    plot_04_arrest_rate_by_city(df)
    plot_05_heatmap_race_city(df)
    plot_06_outcome_by_reason(df)
    plot_07_arrest_rate_reason_race(df)
    plot_08_stops_per_year_city(df)
    plot_09_arrest_rate_time_race(df)
    plot_10_stops_by_hour(df)
    plot_10b_stops_by_hour_weighted(df)
    plot_11_stops_by_day_of_week(df)
    plot_12_race_distribution_by_city(df)
    plot_13_age_distribution_race(df)
    plot_14_sex_distribution_outcome(df)
    plot_15_search_rate_by_race(df)
    plot_16_arrest_rate_search_status_race(df)
    plot_17_correlation_heatmap(df)

    print("\nExporting CSV tables...")
    export_csv_tables(df)

    print("\nGenerating eda_summary.md...")
    stats = compute_summary_stats(df)
    generate_summary_md(df, stats)

    print("\n" + "=" * 60)
    print("Done! All outputs saved to eda/")
    print("=" * 60)


if __name__ == "__main__":
    main()
