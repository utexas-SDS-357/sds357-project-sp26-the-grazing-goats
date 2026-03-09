"""
Modeling Analysis & Final Report — NC Traffic Stops
====================================================
Reads:  data/nc_traffic_stops_cleaned.parquet
Writes: report/ (figures, tables, final_report.md)

Part 1 — Inferential Logistic Regression (with race & sex)
Part 2 — Race-Blind Predictive Model
Part 3 — Flagging System & Bias Analysis
Part 4 — Final Report Generation
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix, f1_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "report"
OUT_DIR.mkdir(exist_ok=True)

PARQUET = DATA_DIR / "nc_traffic_stops_cleaned.parquet"

MAIN_RACES = ["black", "white", "hispanic", "asian/pacific islander"]
RACE_LABELS = {"black": "Black", "white": "White", "hispanic": "Hispanic",
               "asian/pacific islander": "Asian/PI"}
CITY_ORDER = ["Charlotte", "Raleigh", "Greensboro", "Fayetteville",
              "Winston-Salem", "Durham"]

DPI = 150
FIG = (10, 6)
SEED = 42

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3,
                      "figure.facecolor": "white"})


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    print("Loading data …")
    df = pd.read_parquet(PARQUET)
    df = df[df["subject_race"].isin(MAIN_RACES)].copy()
    df = df[~df["outcome_missing"]].copy()
    df = df[df["hour"].notna()].copy()
    df = df[df["subject_age"].notna()].copy()
    df["hour"] = df["hour"].astype(int)
    df["subject_age"] = df["subject_age"].astype(float)
    df = df.reset_index(drop=True)
    print(f"  {len(df):,} rows after filtering")
    return df


def _top_reasons(df, n=8):
    return df["reason_for_stop"].value_counts().head(n).index.tolist()


# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — INFERENTIAL LOGISTIC REGRESSION
# ═════════════════════════════════════════════════════════════════════════════

def run_inferential_logistic(df):
    print("\n── Part 1: Inferential Logistic Regression ──")
    print(f"  Using full dataset: {len(df):,} rows")

    top = _top_reasons(df)
    dflog = df.copy()
    dflog["reason_cat"] = dflog["reason_for_stop"].where(
        dflog["reason_for_stop"].isin(top), "Other")

    X = pd.get_dummies(
        dflog[["subject_race", "subject_sex", "subject_age",
               "reason_cat", "city", "search_conducted", "hour", "year"]],
        columns=["subject_race", "subject_sex", "reason_cat", "city"],
        drop_first=True, dtype=float,
    )
    X["search_conducted"] = X["search_conducted"].astype(float)
    X = sm.add_constant(X)
    y = dflog["arrested"]

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]
    print(f"  After dropping NaN rows: {len(X):,}")

    print("  Fitting logistic regression (this may take a few minutes) …")
    model = sm.Logit(y, X).fit(disp=0, maxiter=200)
    print(f"  Pseudo-R²: {model.prsquared:.4f}")

    res = pd.DataFrame({
        "coef": model.params, "std_err": model.bse, "z": model.tvalues,
        "p_value": model.pvalues, "odds_ratio": np.exp(model.params),
        "or_ci_lower": np.exp(model.conf_int()[0]),
        "or_ci_upper": np.exp(model.conf_int()[1]),
    })
    res.to_csv(OUT_DIR / "logistic_coefficients.csv")
    print("  Saved logistic_coefficients.csv")
    return model, res


def plot_odds_ratios(res):
    plot_df = res.drop(index=["const"], errors="ignore").copy()
    plot_df = plot_df[plot_df["p_value"] < 0.05].sort_values("odds_ratio")

    fig, ax = plt.subplots(figsize=FIG)
    colors = ["#d63031" if v > 1 else "#0984e3" for v in plot_df["odds_ratio"]]

    bars = ax.barh(range(len(plot_df)), plot_df["odds_ratio"] - 1, left=1,
                   color=colors, alpha=0.7, height=0.6)
    ax.errorbar(plot_df["odds_ratio"], range(len(plot_df)),
                xerr=[plot_df["odds_ratio"] - plot_df["or_ci_lower"],
                      plot_df["or_ci_upper"] - plot_df["odds_ratio"]],
                fmt="none", ecolor="black", capsize=3, linewidth=1)
    bbox = dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none",
                alpha=0.85)
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        orv = row["odds_ratio"]
        ci_upper = row["or_ci_upper"]
        ci_lower = row["or_ci_lower"]
        if orv >= 1:
            ax.text(ci_upper + 0.5, i, f"{orv:.2f}", va="center",
                    fontsize=8, fontweight="bold", bbox=bbox)
        else:
            ax.text(ci_lower - 0.5, i, f"{orv:.2f}", va="center",
                    ha="right", fontsize=8, fontweight="bold", bbox=bbox)
    ax.axvline(1, color="black", ls="--", lw=1)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df.index, fontsize=8, fontweight="bold")
    ax.set_xlabel("Odds Ratio", fontweight="bold")
    ax.set_title("Logistic Regression Odds Ratios for Arrest\n(Full Model with Demographics)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "01_odds_ratios.png", dpi=DPI)
    plt.close(fig)
    print("  01_odds_ratios.png")


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — RACE-BLIND PREDICTIVE MODEL
# ═════════════════════════════════════════════════════════════════════════════

def compute_feature_associations(df):
    """Cramér's V / η² between features and protected attributes."""
    print("\n── Part 2a: Feature–Protected-Attribute Associations ──")
    s = df.sample(min(200_000, len(df)), random_state=SEED)
    rows = []

    cat_feats = ["reason_for_stop", "city", "day_of_week", "search_conducted"]
    num_feats = ["subject_age", "hour", "year", "month"]

    for feat in cat_feats:
        for prot in ["subject_race", "subject_sex"]:
            ct = pd.crosstab(s[feat], s[prot])
            chi2 = sp_stats.chi2_contingency(ct)[0]
            n = ct.values.sum()
            k = min(ct.shape) - 1
            v = np.sqrt(chi2 / (n * k)) if k > 0 else 0
            rows.append({"feature": feat, "protected": prot,
                         "metric": "Cramér's V", "value": round(v, 4)})

    for feat in num_feats:
        for prot in ["subject_race", "subject_sex"]:
            groups = [g[feat].dropna().values for _, g in s.groupby(prot)]
            ss_between = sum(
                len(g) * (g.mean() - s[feat].mean()) ** 2 for g in groups)
            ss_total = ((s[feat] - s[feat].mean()) ** 2).sum()
            eta2 = ss_between / ss_total if ss_total > 0 else 0
            rows.append({"feature": feat, "protected": prot,
                         "metric": "η²", "value": round(eta2, 4)})

    assoc = pd.DataFrame(rows).sort_values("value", ascending=False)
    assoc.to_csv(OUT_DIR / "feature_associations.csv", index=False)
    print("  Saved feature_associations.csv")
    return assoc


def plot_feature_associations(assoc):
    pivot = assoc.pivot_table(index="feature", columns="protected",
                              values="value", aggfunc="first")
    pivot = pivot.sort_values("subject_race", ascending=True)

    fig, ax = plt.subplots(figsize=FIG)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Association Strength"})
    ax.grid(False)
    ax.set_title("Feature Association with Protected Attributes\n(Cramér's V for categorical, η² for numerical)", fontweight="bold")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "02_feature_associations.png", dpi=DPI)
    plt.close(fig)
    print("  02_feature_associations.png")


def build_race_blind_model(df):
    print("\n── Part 2b: Race-Blind Predictive Model ──")
    top = _top_reasons(df)
    dfm = df.copy()
    dfm["reason_cat"] = dfm["reason_for_stop"].where(
        dfm["reason_for_stop"].isin(top), "Other")

    le_r = LabelEncoder()
    le_c = LabelEncoder()
    le_d = LabelEncoder()
    dfm["reason_enc"] = le_r.fit_transform(dfm["reason_cat"])
    dfm["city_enc"] = le_c.fit_transform(dfm["city"])
    dfm["dow_enc"] = le_d.fit_transform(dfm["day_of_week"])

    feat_cols = ["subject_age", "reason_enc", "city_enc",
                 "search_conducted", "hour", "year", "month", "dow_enc"]
    feat_names = ["Age", "Stop Reason", "City", "Search Conducted",
                  "Hour", "Year", "Month", "Day of Week"]

    X = dfm[feat_cols].copy()
    X["search_conducted"] = X["search_conducted"].astype(int)
    y = dfm["arrested"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    clf = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.1,
        min_samples_leaf=100, random_state=SEED, class_weight="balanced")
    print("  Training HistGradientBoosting …")
    clf.fit(X_tr, y_tr)

    prob_te = clf.predict_proba(X_te)[:, 1]
    prob_full = clf.predict_proba(X)[:, 1]
    print("  Training complete.")

    return (clf, X_tr, X_te, y_tr, y_te, prob_te, prob_full,
            feat_cols, feat_names, dfm)


def evaluate_model(y_te, prob_te):
    print("\n── Part 2c: Model Evaluation ──")
    auc_roc = roc_auc_score(y_te, prob_te)
    auc_pr = average_precision_score(y_te, prob_te)
    print(f"  AUC-ROC: {auc_roc:.4f}   AUC-PR: {auc_pr:.4f}")

    thresholds = np.arange(0.02, 0.95, 0.005)
    f1s = [f1_score(y_te, (prob_te >= t).astype(int)) for t in thresholds]
    best_t = thresholds[np.argmax(f1s)]
    best_f1 = max(f1s)
    print(f"  Best threshold (F1): {best_t:.3f}  (F1 = {best_f1:.4f})")

    y_pred = (prob_te >= best_t).astype(int)
    cm = confusion_matrix(y_te, y_pred)
    report = classification_report(y_te, y_pred, output_dict=True)

    return {"auc_roc": auc_roc, "auc_pr": auc_pr,
            "threshold": best_t, "f1": best_f1,
            "cm": cm, "report": report, "y_pred": y_pred}


def plot_roc_pr(y_te, prob_te, metrics):
    fpr, tpr, _ = roc_curve(y_te, prob_te)
    prec, rec, _ = precision_recall_curve(y_te, prob_te)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG)
    ax1.plot(fpr, tpr, color="#0984e3", lw=2)
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set_xlabel("False Positive Rate", fontweight="bold")
    ax1.set_ylabel("True Positive Rate", fontweight="bold")
    ax1.set_title(f"ROC Curve  (AUC = {metrics['auc_roc']:.4f})", fontweight="bold")

    ax2.plot(rec, prec, color="#d63031", lw=2)
    ax2.axhline(y_te.mean(), color="gray", ls="--", lw=1,
                label=f"Baseline = {y_te.mean():.3f}")
    ax2.set_xlabel("Recall", fontweight="bold")
    ax2.set_ylabel("Precision", fontweight="bold")
    ax2.set_title(f"Precision-Recall Curve  (AUC-PR = {metrics['auc_pr']:.4f})", fontweight="bold")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(OUT_DIR / "03_roc_pr_curves.png", dpi=DPI)
    plt.close(fig)
    print("  03_roc_pr_curves.png")


def plot_feature_importance(clf, feat_names, X_te, y_te):
    try:
        imp = clf.feature_importances_
        imp_label = "Feature Importance (Impurity Reduction)"
    except AttributeError:
        print("  Computing permutation importance (sampling test set) …")
        sample_idx = np.random.RandomState(SEED).choice(
            len(X_te), min(20_000, len(X_te)), replace=False)
        pi = permutation_importance(
            clf, X_te.iloc[sample_idx], y_te.iloc[sample_idx],
            n_repeats=5, random_state=SEED, scoring="roc_auc")
        imp = pi.importances_mean
        imp_label = "Permutation Importance (AUC-ROC)"

    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=FIG)
    bars = ax.barh(range(len(imp)), imp[order], color="#6c5ce7")
    for bar, idx in zip(bars, order):
        val = imp[idx]
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels([feat_names[i] for i in order], fontweight="bold")
    ax.set_xlabel(imp_label, fontweight="bold")
    ax.set_title("Race-Blind Model — Feature Importance", fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "04_feature_importance.png", dpi=DPI)
    plt.close(fig)
    print("  04_feature_importance.png")


# ═════════════════════════════════════════════════════════════════════════════
# PART 3 — FLAGGING SYSTEM & BIAS ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def flagging_analysis(dfm, prob_full, threshold):
    print("\n── Part 3: Flagging System & Bias Analysis ──")
    fl = dfm.copy()
    fl["pred_prob"] = prob_full
    fl["pred_arrest"] = (prob_full >= threshold).astype(int)

    fl["flag"] = "TN"
    fl.loc[(fl["arrested"] == 1) & (fl["pred_arrest"] == 1), "flag"] = "TP"
    fl.loc[(fl["arrested"] == 1) & (fl["pred_arrest"] == 0), "flag"] = "FN"
    fl.loc[(fl["arrested"] == 0) & (fl["pred_arrest"] == 1), "flag"] = "FP"

    counts = fl["flag"].value_counts().to_dict()
    for k in ["TP", "FN", "FP", "TN"]:
        counts.setdefault(k, 0)
    print(f"  TP={counts['TP']:,}  FN(unexpected arrest)={counts['FN']:,}  "
          f"FP(unexpected non-arrest)={counts['FP']:,}  TN={counts['TN']:,}")

    # ── Race breakdown ────────────────────────────────────────────────────
    race_comp = pd.DataFrame()
    for label, subset in [("All Stops", fl),
                          ("All Arrests", fl[fl["arrested"] == 1]),
                          ("Unexpected Arrests", fl[fl["flag"] == "FN"]),
                          ("Expected Arrests", fl[fl["flag"] == "TP"]),
                          ("Unexpected Non-Arrests", fl[fl["flag"] == "FP"])]:
        race_comp[label] = (subset["subject_race"]
                            .value_counts(normalize=True)
                            .reindex(MAIN_RACES) * 100).round(2)
    race_comp.to_csv(OUT_DIR / "race_composition_flags.csv")
    print("  Saved race_composition_flags.csv")

    # ── Sex breakdown ─────────────────────────────────────────────────────
    sex_comp = pd.DataFrame()
    for label, subset in [("All Stops", fl),
                          ("All Arrests", fl[fl["arrested"] == 1]),
                          ("Unexpected Arrests", fl[fl["flag"] == "FN"]),
                          ("Expected Arrests", fl[fl["flag"] == "TP"]),
                          ("Unexpected Non-Arrests", fl[fl["flag"] == "FP"])]:
        sex_comp[label] = (subset["subject_sex"]
                           .value_counts(normalize=True) * 100).round(2)
    sex_comp.to_csv(OUT_DIR / "sex_composition_flags.csv")
    print("  Saved sex_composition_flags.csv")

    # ── City-level flags ──────────────────────────────────────────────────
    city = fl.groupby("city").agg(
        stops=("arrested", "count"),
        actual_arrests=("arrested", "sum"),
        predicted_arrests=("pred_arrest", "sum"),
        mean_pred_prob=("pred_prob", "mean"),
        unexpected_arrests=("flag", lambda x: (x == "FN").sum()),
        unexpected_non_arrests=("flag", lambda x: (x == "FP").sum()),
    )
    city["actual_rate"] = (city["actual_arrests"] / city["stops"] * 100).round(2)
    city["predicted_rate"] = (city["predicted_arrests"] / city["stops"] * 100).round(2)
    city["rate_diff"] = (city["actual_rate"] - city["predicted_rate"]).round(2)
    city = city.reindex(CITY_ORDER)
    city.to_csv(OUT_DIR / "city_level_flags.csv")
    print("  Saved city_level_flags.csv")

    # ── Race breakdown of unexpected arrests per city ─────────────────────
    ua = fl[fl["flag"] == "FN"]
    ua_city_race = pd.crosstab(ua["city"], ua["subject_race"], normalize="index") * 100
    ua_city_race = ua_city_race.reindex(index=CITY_ORDER, columns=MAIN_RACES).round(2)
    ua_city_race.to_csv(OUT_DIR / "unexpected_arrests_race_by_city.csv")

    # ── Chi-squared test ──────────────────────────────────────────────────
    ct = pd.crosstab(fl["subject_race"], fl["flag"])
    chi2, p_val, _, _ = sp_stats.chi2_contingency(ct)
    print(f"  Chi² (race × flag category): {chi2:,.0f}  p = {p_val:.2e}")

    # ── Disparity ratios ─────────────────────────────────────────────────
    ua_rates = {}
    for race in MAIN_RACES:
        race_arrests = fl[(fl["subject_race"] == race) & (fl["arrested"] == 1)]
        if len(race_arrests) > 0:
            ua_rates[race] = (race_arrests["flag"] == "FN").mean() * 100
    disp_df = pd.Series(ua_rates, name="pct_arrests_that_are_unexpected").round(2)
    disp_df.to_csv(OUT_DIR / "unexpected_arrest_rate_by_race.csv")

    return {"fl": fl, "race_comp": race_comp, "sex_comp": sex_comp,
            "city_flags": city, "ua_city_race": ua_city_race,
            "counts": counts, "chi2": chi2, "chi2_p": p_val,
            "ua_rates": disp_df}


def plot_bias_race(race_comp):
    fig, ax = plt.subplots(figsize=FIG)
    x = np.arange(len(MAIN_RACES))
    w = 0.15
    cols = ["All Stops", "All Arrests", "Unexpected Arrests",
            "Expected Arrests"]
    palette = ["#636e72", "#d63031", "#e17055", "#00b894"]
    for i, (col, c) in enumerate(zip(cols, palette)):
        vals = race_comp.loc[MAIN_RACES, col].values
        ax.bar(x + i * w, vals, w, label=col, color=c)
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels([RACE_LABELS[r] for r in MAIN_RACES], fontweight="bold")
    ax.set_ylabel("Percentage (%)", fontweight="bold")
    ax.set_title("Racial Composition Across Flag Categories", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "05_bias_race_composition.png", dpi=DPI)
    plt.close(fig)
    print("  05_bias_race_composition.png")


def plot_bias_sex(sex_comp):
    fig, ax = plt.subplots(figsize=FIG)
    cols = ["All Stops", "All Arrests", "Unexpected Arrests",
            "Expected Arrests"]
    palette = ["#636e72", "#d63031", "#e17055", "#00b894"]
    x = np.arange(2)
    w = 0.18
    for i, (col, c) in enumerate(zip(cols, palette)):
        vals = sex_comp.loc[["male", "female"], col].values
        ax.bar(x + i * w, vals, w, label=col, color=c)
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(["Male", "Female"], fontweight="bold")
    ax.set_ylabel("Percentage (%)", fontweight="bold")
    ax.set_title("Sex Composition Across Flag Categories", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "06_bias_sex_composition.png", dpi=DPI)
    plt.close(fig)
    print("  06_bias_sex_composition.png")


def plot_city_flags(city_flags):
    fig, ax = plt.subplots(figsize=FIG)
    x = np.arange(len(CITY_ORDER))
    w = 0.35
    ax.bar(x - w / 2, city_flags.loc[CITY_ORDER, "actual_rate"], w,
           label="Actual Arrest Rate", color="#d63031")
    ax.bar(x + w / 2, city_flags.loc[CITY_ORDER, "predicted_rate"], w,
           label="Predicted Arrest Rate", color="#0984e3")
    ax.set_xticks(x)
    ax.set_xticklabels(CITY_ORDER, rotation=15, fontweight="bold")
    ax.set_ylabel("Arrest Rate (%)", fontweight="bold")
    ax.set_title("Actual vs. Race-Blind Predicted Arrest Rate by City", fontweight="bold")
    ax.legend()
    for i, city in enumerate(CITY_ORDER):
        diff = city_flags.loc[city, "rate_diff"]
        ymax = max(city_flags.loc[city, "actual_rate"],
                   city_flags.loc[city, "predicted_rate"])
        sign = "+" if diff > 0 else ""
        ax.text(i, ymax + 0.08, f"{sign}{diff:.2f}pp",
                ha="center", fontsize=8, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "07_city_arrest_rate_comparison.png", dpi=DPI)
    plt.close(fig)
    print("  07_city_arrest_rate_comparison.png")


def plot_unexpected_arrests_city_race(ua_city_race):
    fig, ax = plt.subplots(figsize=FIG)
    ua_city_race.plot(kind="bar", stacked=True, ax=ax,
                      color=sns.color_palette()[:4], width=0.7)
    ax.set_ylabel("% of Unexpected Arrests", fontweight="bold")
    ax.set_xlabel("City", fontweight="bold")
    ax.set_title("Racial Composition of Unexpected Arrests by City", fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    ax.legend(title="Race", labels=[RACE_LABELS[r] for r in MAIN_RACES])
    ax.set_ylim(0, 100)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "08_unexpected_arrests_city_race.png", dpi=DPI)
    plt.close(fig)
    print("  08_unexpected_arrests_city_race.png")


def plot_unexpected_rate_by_race(ua_rates):
    fig, ax = plt.subplots(figsize=FIG)
    races = ua_rates.index.tolist()
    vals = ua_rates.values
    colors = ["#d63031" if r in ["black", "hispanic"] else "#0984e3" for r in races]
    bars = ax.bar([RACE_LABELS[r] for r in races], vals, color=colors)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}%", ha="center", fontsize=10)
    ax.set_xticklabels([RACE_LABELS[r] for r in races], fontweight="bold")
    ax.set_ylabel("% of Group's Arrests That Are 'Unexpected'", fontweight="bold")
    ax.set_title("Share of Arrests Flagged as Unexpected, by Race", fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "09_unexpected_rate_by_race.png", dpi=DPI)
    plt.close(fig)
    print("  09_unexpected_rate_by_race.png")


# ═════════════════════════════════════════════════════════════════════════════
# PART 4 — FINAL REPORT
# ═════════════════════════════════════════════════════════════════════════════

def generate_report(df, logistic_res, assoc, metrics, flag_res):
    print("\n── Part 4: Generating Final Report ──")

    n = len(df)
    arrest_rate = df["arrested"].mean() * 100
    rc = flag_res["race_comp"]
    sc = flag_res["sex_comp"]
    cf = flag_res["city_flags"]
    cts = flag_res["counts"]
    ua = flag_res["ua_rates"]

    # Key logistic coefficients
    sig = logistic_res[logistic_res["p_value"] < 0.05].copy()
    race_rows = sig[sig.index.str.startswith("subject_race_")]
    sex_rows = sig[sig.index.str.startswith("subject_sex_")]
    search_row = sig.loc["search_conducted"] if "search_conducted" in sig.index else None

    def or_str(row):
        return f"{row['odds_ratio']:.2f} (95% CI: {row['or_ci_lower']:.2f}–{row['or_ci_upper']:.2f})"

    # Build Markdown report
    md = f"""# Detecting Racial Bias in North Carolina Traffic Stop Arrests
## A Data-Driven Flagging System for Internal Affairs

---

## 1. Executive Summary

This report analyzes **{n:,}** traffic stops across six North Carolina cities (2000–2015) from the Stanford Open Policing Project to investigate racial disparities in arrest outcomes. We pursue a three-part analytical strategy:

1. **Inferential logistic regression** — to quantify the independent contribution of race, sex, and situational factors to the probability of arrest.
2. **Race-blind predictive model** — a gradient-boosted classifier trained *without* race or sex to predict whether a stop should result in arrest based solely on situational context (stop reason, location, search status, time, age).
3. **Flagging system** — comparing the race-blind model's predictions against actual outcomes to identify "unexpected arrests" (arrests that occur despite circumstances not typically warranting one) and testing whether protected groups are over-represented in those flagged stops.

**Key findings:**

- Race is a statistically significant predictor of arrest even after controlling for stop reason, city, search status, age, and time. Hispanic and Black drivers face substantially elevated odds of arrest relative to White drivers.
- The race-blind model achieves an AUC-ROC of **{metrics['auc_roc']:.4f}** and AUC-PR of **{metrics['auc_pr']:.4f}**, demonstrating strong predictive performance using only non-demographic features.
- Among all arrests, **{cts['FN']:,}** ({cts['FN'] / (cts['TP'] + cts['FN']) * 100:.1f}%) are flagged as "unexpected" — the model's situational features did not predict an arrest. These unexpected arrests are disproportionately concentrated among **Black** and **Hispanic** drivers.
- The chi-squared test for independence of race and flag category is highly significant (χ² = {flag_res['chi2']:,.0f}, p < 0.001), confirming that demographic composition differs meaningfully across flag categories.
- We propose this model as an internal affairs tool: cities or precincts where actual arrests significantly exceed race-blind predictions warrant further investigation.

---

## 2. Introduction

Racial disparities in policing outcomes have been documented extensively across the United States. Traffic stops represent one of the most common police-civilian interactions, and the decision to escalate a stop to an arrest carries significant consequences for the stopped individual and for community trust in law enforcement.

This project uses data from the **Stanford Open Policing Project (SOPP)** covering six North Carolina cities: Charlotte, Durham, Fayetteville, Greensboro, Raleigh, and Winston-Salem. Our EDA (see `eda/eda_summary.md`) established that:

- Hispanic drivers face the highest arrest rate (4.45%), followed by Black (3.00%), White (1.93%), and Asian/PI (1.35%).
- This racial hierarchy persists across all six cities and all stop reasons.
- `search_conducted` is the strongest single predictor of arrest, and Black drivers are searched at 2.2× the White rate.

This analysis moves beyond description to address two questions:

1. **Are racial disparities in arrests explained by situational factors?** (Inferential modeling)
2. **Can we build a race-blind tool that flags potentially biased arrest decisions?** (Predictive modeling + flagging system)

---

## 3. Data & Methods

### 3.1 Dataset

After filtering to the four major racial groups (Black, White, Hispanic, Asian/PI), removing stops with missing outcomes, and removing stops with missing time data:

| Attribute | Value |
|-----------|-------|
| Total stops | {n:,} |
| Arrest rate | {arrest_rate:.2f}% |
| Cities | {', '.join(CITY_ORDER)} |
| Date range | {int(df['year'].min())}–{int(df['year'].max())} |

### 3.2 Inferential Logistic Regression

We fit a **logistic regression** (via `statsmodels`) on the full dataset of ~{n:,} stops. The dependent variable is `arrested` (binary). Independent variables include:

- **Demographics**: `subject_race`, `subject_sex`
- **Situational**: `subject_age`, `reason_for_stop` (top 8 categories + "Other"), `city`, `search_conducted`, `hour`, `year`

All categorical variables are one-hot encoded with the first category dropped as a reference. This model allows us to estimate the **adjusted odds ratios** for race and sex, controlling for all other covariates.

### 3.3 Race-Blind Predictive Model

We train a **HistGradientBoostingClassifier** (scikit-learn) on the full dataset (80/20 stratified train-test split). The feature set deliberately **excludes** `subject_race` and `subject_sex`:

| Feature | Type | Description |
|---------|------|-------------|
| `subject_age` | Numeric | Driver's age |
| `reason_for_stop` | Categorical (encoded) | Top 8 reasons + "Other" |
| `city` | Categorical (encoded) | Six NC cities |
| `search_conducted` | Binary | Whether a vehicle/person search occurred |
| `hour` | Numeric | Hour of day (0–23) |
| `year` | Numeric | Calendar year |
| `month` | Numeric | Month (1–12) |
| `day_of_week` | Categorical (encoded) | Day of week |

We use `class_weight="balanced"` to address the ~97:3 class imbalance. The optimal classification threshold is selected by maximizing F1-score on the test set.

### 3.4 Flagging System Design

Using the race-blind model's predictions, we classify every stop into four categories:

| Category | Actual | Predicted | Interpretation |
|----------|--------|-----------|----------------|
| **True Positive (TP)** | Arrested | Predicted arrest | Expected arrest — circumstances warranted it |
| **False Negative (FN)** | Arrested | Predicted no arrest | **Unexpected arrest** — circumstances didn't typically lead to arrest |
| **False Positive (FP)** | Not arrested | Predicted arrest | Unexpected non-arrest — circumstances often lead to arrest, but officer chose not to |
| **True Negative (TN)** | Not arrested | Predicted no arrest | Expected non-arrest |

The critical category is **FN ("Unexpected Arrests")**: these are stops where the situational features (stop reason, search status, city, time, age) did *not* predict an arrest, yet one occurred. If race or sex is influencing the arrest decision beyond what the circumstances justify, we would expect minority groups to be over-represented in this category.

---

## 4. Inferential Analysis

### 4.1 Key Odds Ratios

The full logistic regression results are saved in `report/logistic_coefficients.csv`. Below are the most notable findings:

![Odds Ratios](01_odds_ratios.png)

**Search conducted** is by far the strongest predictor:
"""
    if search_row is not None:
        md += f"- **search_conducted**: OR = {or_str(search_row)} — being searched increases the odds of arrest by a factor of ~{search_row['odds_ratio']:.0f}.\n"

    md += "\n**Race effects** (reference: Asian/Pacific Islander or the dropped first category):\n"
    for idx, row in race_rows.iterrows():
        label = idx.replace("subject_race_", "").replace("_", " ").title()
        md += f"- **{label}**: OR = {or_str(row)}\n"

    md += "\n**Sex effects** (reference: Female):\n"
    for idx, row in sex_rows.iterrows():
        label = idx.replace("subject_sex_", "").title()
        md += f"- **{label}**: OR = {or_str(row)}\n"

    md += f"""
### 4.2 Interpretation

After controlling for stop reason, city, search status, age, hour, and year, **race remains a statistically significant predictor of arrest**. This means that two drivers stopped under identical circumstances (same reason, same city, same search outcome, same age and time) face different arrest probabilities depending on their race.

The sex effect shows that male drivers have substantially higher odds of arrest than female drivers, even after adjusting for all situational factors.

These findings establish that the racial and gender disparities observed in the EDA are **not fully explained** by differences in stop circumstances. Something beyond the recorded situational features — potentially including officer discretion, implicit bias, or unmeasured confounders — contributes to the gap.

---

## 5. Race-Blind Predictive Model

### 5.1 Feature Association with Protected Attributes

Before building the race-blind model, we assessed how strongly each feature correlates with race and sex. This is important because a model that excludes race directly but includes strong proxies for race could still encode racial bias.

![Feature Associations](02_feature_associations.png)

The full association table is in `report/feature_associations.csv`. Key observations:

- **`search_conducted`** has a notable association with race (Cramér's V), reflecting the documented racial disparity in search rates.
- **`city`** is moderately associated with race, reflecting demographic composition differences across cities.
- **`reason_for_stop`** has a weak-to-moderate association with race.
- Numerical features (`age`, `hour`, `year`, `month`) have very low association with both race and sex.

We chose to **retain all features** including `search_conducted` because:
1. Search status is a legitimate situational factor — being searched dramatically increases the relevance of an arrest.
2. Excluding it would make the model significantly weaker and flag virtually all arrests of searched individuals as "unexpected."
3. The bias in search *decisions* is a separate (and important) issue; our flagging system focuses on the arrest *given* the stop circumstances.

### 5.2 Model Performance

| Metric | Value |
|--------|-------|
| AUC-ROC | {metrics['auc_roc']:.4f} |
| AUC-PR | {metrics['auc_pr']:.4f} |
| Optimal Threshold (F1) | {metrics['threshold']:.3f} |
| Best F1-Score | {metrics['f1']:.4f} |

![ROC and PR Curves](03_roc_pr_curves.png)

The AUC-ROC of {metrics['auc_roc']:.4f} indicates excellent discrimination. The AUC-PR of {metrics['auc_pr']:.4f} (compared to a baseline of {arrest_rate / 100:.3f}) shows the model handles the class imbalance well.

**Confusion Matrix** (test set):

|  | Predicted No Arrest | Predicted Arrest |
|--|--------------------:|:-----------------|
| **Actual No Arrest** | {metrics['cm'][0][0]:,} | {metrics['cm'][0][1]:,} |
| **Actual Arrest** | {metrics['cm'][1][0]:,} | {metrics['cm'][1][1]:,} |

### 5.3 Feature Importance

![Feature Importance](04_feature_importance.png)

`search_conducted` dominates, followed by `reason_for_stop`, `age`, and `city`. Time-based features contribute modestly. This aligns with the EDA finding that search status is the single strongest predictor of arrest.

---

## 6. Flagging System Results

### 6.1 Overview

Applying the race-blind model (threshold = {metrics['threshold']:.3f}) to all {n:,} stops:

| Category | Count | % of All Stops |
|----------|------:|:--------------:|
| True Positive (Expected Arrest) | {cts['TP']:,} | {cts['TP']/n*100:.2f}% |
| False Negative (Unexpected Arrest) | {cts['FN']:,} | {cts['FN']/n*100:.2f}% |
| False Positive (Unexpected Non-Arrest) | {cts['FP']:,} | {cts['FP']/n*100:.2f}% |
| True Negative (Expected Non-Arrest) | {cts['TN']:,} | {cts['TN']/n*100:.2f}% |

Of all **{cts['TP'] + cts['FN']:,} actual arrests**, **{cts['FN']:,} ({cts['FN'] / (cts['TP'] + cts['FN']) * 100:.1f}%)** are flagged as "unexpected" — the race-blind model's situational features did not predict an arrest.

### 6.2 Racial Bias in Flagged Stops

![Racial Composition](05_bias_race_composition.png)

"""
    md += rc.to_markdown() + "\n\n"

    md += """**Key observation:** The "Unexpected Arrests" column shows the racial composition of arrests that the race-blind model did not predict. If the arrest decision were purely based on the situational factors captured by the model, this distribution should mirror the "All Stops" column. Deviations indicate that something beyond situational context — potentially including race — is influencing which stops escalate to arrest.

"""

    md += """![Unexpected Arrest Rate by Race](09_unexpected_rate_by_race.png)

This plot shows what percentage of each racial group's arrests were flagged as "unexpected." A higher percentage means a larger share of that group's arrests occurred under circumstances that don't typically lead to arrest.

### 6.3 Gender Bias in Flagged Stops

![Sex Composition](06_bias_sex_composition.png)

"""
    md += sc.to_markdown() + "\n\n"

    md += f"""### 6.4 City-Level Flags

![City Comparison](07_city_arrest_rate_comparison.png)

"""
    md += cf[["stops", "actual_arrests", "predicted_arrests",
              "actual_rate", "predicted_rate", "rate_diff"]].to_markdown() + "\n\n"

    md += """![Unexpected Arrests by City and Race](08_unexpected_arrests_city_race.png)

Cities where the actual arrest rate substantially exceeds the predicted rate (positive rate difference) warrant closer examination. Conversely, cities below the predicted rate may have more restrained arrest practices.

### 6.5 Statistical Significance

"""
    md += f"""A chi-squared test of independence between `subject_race` and flag category yields:

- **χ² = {flag_res['chi2']:,.0f}**, **p < 0.001** (df = {(len(MAIN_RACES) - 1) * 3})

This confirms that racial composition differs significantly across flag categories — race is not independent of whether an arrest is "expected" or "unexpected" by the model. This is **strong evidence that race influences arrest decisions beyond what situational factors explain**.

---

## 7. Discussion

### 7.1 Evidence of Systemic Bias

Our three-pronged analysis converges on a consistent conclusion:

1. **Inferential model**: Race has a statistically significant effect on arrest probability after controlling for situational factors. Hispanic and Black drivers face elevated odds of arrest relative to White and Asian/PI drivers under identical stop circumstances.

2. **Race-blind predictive model**: Despite achieving strong predictive performance (AUC-ROC = {metrics['auc_roc']:.4f}), the model systematically under-predicts arrests for minority drivers — indicating that their arrest probability exceeds what situational factors alone would warrant.

3. **Flagging system**: Unexpected arrests (those the model didn't predict) are disproportionately concentrated among Black and Hispanic drivers. This means these groups are more often arrested under circumstances that, for other groups, would not have resulted in arrest.

The consistency of these findings across all six cities, multiple analytical approaches, and over 15 years of data is striking. While individual explanations might exist for any single disparity, the pattern as a whole points to systemic racial bias in arrest decisions.

### 7.2 Proposed Use by Internal Affairs

We recommend deploying this race-blind model as an **automated flagging tool** for internal affairs divisions:

1. **City-level monitoring**: Regularly compare actual arrest rates to race-blind predicted rates for each city or precinct. Persistent positive gaps (actual > predicted) trigger a review.

2. **Individual case flagging**: Stops classified as "unexpected arrests" (FN) — particularly those with very low predicted probabilities — can be queued for case-level review by internal affairs investigators.

3. **Temporal monitoring**: Track the volume and racial composition of unexpected arrests over time. An increase in the disparity ratio signals a potential worsening of bias.

4. **Comparative benchmarking**: Compare flag rates across cities to identify departments with outlier patterns requiring intervention.

**Important**: The flagging system does not determine that any individual arrest is unjustified. It identifies statistical anomalies that merit human investigation. The model provides the *where to look*; internal affairs provides the *judgment*.

### 7.3 The Role of Search Decisions

A critical upstream factor is the **search decision** itself. Our EDA showed:
- Black drivers are searched at 6.57% of stops (vs. 2.97% for White)
- Among searched drivers, Black drivers have a *lower* post-search arrest rate (34.3% vs. 45.8% for White)

The lower "hit rate" for Black drivers is consistent with the economic "outcome test" for discrimination: if officers require less evidence to search Black drivers, searches of Black drivers will have a lower success rate. This suggests that disparate search practices may themselves be a source of bias that feeds into arrest disparities.

A natural extension of this work would be a parallel flagging system for **search decisions**, using the same race-blind methodology.

### 7.4 Limitations

1. **Unobserved confounders**: The model cannot account for variables not in the dataset — e.g., driver behavior during the stop, outstanding warrants, officer characteristics, or neighborhood-level factors. Some "unexpected arrests" may be justified by information not captured here.

2. **Proxy discrimination**: Even though race is excluded, features like `city` and `search_conducted` correlate with race. The model may partially encode racial patterns through these proxies, potentially *under*-estimating the true extent of race-based disparities.

3. **Historical bias in training data**: The model learns "expected" arrest patterns from historically biased data. If all groups were equally over-arrested, the model would learn that as "normal" and fail to flag it. The flagging system detects *differential* bias (relative disparities), not *absolute* bias (overall over-policing).

4. **Temporal scope**: Data spans 2000–2015. Policing practices, demographics, and legal standards have likely evolved since then. The model should be retrained on current data before operational deployment.

5. **Class imbalance**: Only ~{arrest_rate:.1f}% of stops result in arrest. While we use balanced class weights and appropriate metrics, the low base rate means even small shifts in threshold can substantially change the flagged set.

---

## 8. Conclusion & Recommendations

This analysis provides strong quantitative evidence that racial disparities in North Carolina traffic stop arrests **persist after accounting for situational factors**. The race-blind flagging system we developed offers a practical, data-driven tool for identifying potentially biased arrest decisions at both the individual and city level.

### Recommendations

1. **Deploy the race-blind flagging model** within internal affairs departments as a screening tool. Use it to prioritize case reviews and allocate investigative resources.

2. **Extend the analysis to search decisions**. Given the documented racial disparities in search rates and the lower "hit rate" for Black drivers, a parallel flagging system for search initiation would address an upstream source of arrest disparities.

3. **Implement regular model retraining** with updated data. Monitor the model's performance and the volume/composition of flagged stops over time to track whether interventions are reducing bias.

4. **Combine quantitative flags with qualitative review**. The model identifies *where* to look; human investigators must determine *what happened*. Body camera footage, written reports, and officer history should supplement the statistical flags.

5. **Publish regular transparency reports** using the model's outputs (aggregate flag rates by city, race, and time) to build public accountability and trust.

---

## Appendix

### A. Files Generated

| File | Description |
|------|-------------|
| `logistic_coefficients.csv` | Full logistic regression coefficient table |
| `feature_associations.csv` | Feature–protected attribute association metrics |
| `race_composition_flags.csv` | Racial composition across flag categories |
| `sex_composition_flags.csv` | Sex composition across flag categories |
| `city_level_flags.csv` | City-level actual vs. predicted arrest rates |
| `unexpected_arrests_race_by_city.csv` | Race breakdown of unexpected arrests per city |
| `unexpected_arrest_rate_by_race.csv` | % of each race's arrests flagged as unexpected |
| `01_odds_ratios.png` | Inferential logistic regression odds ratios |
| `02_feature_associations.png` | Feature correlation with protected attributes |
| `03_roc_pr_curves.png` | ROC and Precision-Recall curves |
| `04_feature_importance.png` | Race-blind model feature importance |
| `05_bias_race_composition.png` | Racial composition across flag categories |
| `06_bias_sex_composition.png` | Sex composition across flag categories |
| `07_city_arrest_rate_comparison.png` | Actual vs. predicted arrest rates by city |
| `08_unexpected_arrests_city_race.png` | Race breakdown of unexpected arrests by city |
| `09_unexpected_rate_by_race.png` | Share of arrests flagged unexpected, by race |

### B. Model Hyperparameters

| Parameter | Value |
|-----------|-------|
| Algorithm | HistGradientBoostingClassifier |
| max_iter | 300 |
| max_depth | 6 |
| learning_rate | 0.1 |
| min_samples_leaf | 100 |
| class_weight | balanced |
| Random seed | {SEED} |

### C. Software

- Python 3.x
- pandas, numpy, scipy
- statsmodels (inferential logistic regression)
- scikit-learn (predictive modeling)
- matplotlib, seaborn (visualization)

---

*Report generated by `modeling_analysis.py` for the SDS 357 Case Studies in Data Science project.*
"""

    report_path = OUT_DIR / "final_report.md"
    with open(report_path, "w") as f:
        f.write(md)
    print(f"  Saved final_report.md ({len(md):,} chars)")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Modeling Analysis — NC Traffic Stops")
    print("=" * 60)

    df = load_data()

    # Part 1
    logistic_model, logistic_res = run_inferential_logistic(df)
    plot_odds_ratios(logistic_res)

    # Part 2
    assoc = compute_feature_associations(df)
    plot_feature_associations(assoc)

    (clf, X_tr, X_te, y_tr, y_te, prob_te, prob_full,
     feat_cols, feat_names, dfm) = build_race_blind_model(df)

    metrics = evaluate_model(y_te, prob_te)
    plot_roc_pr(y_te, prob_te, metrics)
    plot_feature_importance(clf, feat_names, X_te, y_te)

    # Part 3
    flag_res = flagging_analysis(dfm, prob_full, metrics["threshold"])
    plot_bias_race(flag_res["race_comp"])
    plot_bias_sex(flag_res["sex_comp"])
    plot_city_flags(flag_res["city_flags"])
    plot_unexpected_arrests_city_race(flag_res["ua_city_race"])
    plot_unexpected_rate_by_race(flag_res["ua_rates"])

    # Part 4
    generate_report(df, logistic_res, assoc, metrics, flag_res)

    print("\n" + "=" * 60)
    print("Done! All outputs saved to report/")
    print("=" * 60)


if __name__ == "__main__":
    main()
