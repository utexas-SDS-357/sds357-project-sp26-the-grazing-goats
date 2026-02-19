"""
Clean and preprocess SOPP North Carolina traffic stops data.
Produces: nc_traffic_stops_cleaned.csv and cleaning_summary.md
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = Path("/Users/harsha/SDS357 project /Stanford Data")
OUTPUT_DIR = Path("/Users/harsha/SDS357 project ")

FILES = {
    "Charlotte": "nc_charlotte_2020_04_01.csv",
    "Durham": "nc_durham_2020_04_01.csv",
    "Fayetteville": "nc_fayetteville_2020_04_01.csv",
    "Greensboro": "nc_greensboro_2020_04_01.csv",
    "Raleigh": "nc_raleigh_2020_04_01.csv",
    "Winston-Salem": "nc_winston-salem_2020_04_01.csv",
}

# ── 1. Load and combine ───────────────────────────────────────────────────
print("Loading city files...")
raw_info = {}  # store per-city raw metadata for the summary
frames = []

for city, filename in FILES.items():
    fp = DATA_DIR / filename
    df = pd.read_csv(fp, low_memory=False)
    raw_info[city] = {"rows": len(df), "columns": list(df.columns)}
    df["city"] = city
    frames.append(df)
    print(f"  {city}: {len(df):,} rows, {len(df.columns)} columns")

raw_combined = pd.concat(frames, ignore_index=True)
total_raw_rows = len(raw_combined)
raw_columns = list(raw_combined.columns)
print(f"\nCombined raw dataset: {total_raw_rows:,} rows, {len(raw_columns)} columns")

# ── 2. Standardize column names ──────────────────────────────────────────
# All files already share the same schema, but enforce lowercase snake_case
rename_map = {c: c.lower().replace(" ", "_") for c in raw_combined.columns}
raw_combined.rename(columns=rename_map, inplace=True)
print(f"Columns after rename: {list(raw_combined.columns)}")

# ── 3. Parse dates and times / engineer temporal features ────────────────
print("\nParsing dates and times...")
raw_combined["date"] = pd.to_datetime(raw_combined["date"], errors="coerce")
raw_combined["time"] = pd.to_datetime(raw_combined["time"], format="%H:%M:%S", errors="coerce").dt.time

raw_combined["year"] = raw_combined["date"].dt.year
raw_combined["month"] = raw_combined["date"].dt.month
raw_combined["day_of_week"] = raw_combined["date"].dt.day_name()
raw_combined["hour"] = pd.to_datetime(
    raw_combined["time"].astype(str), format="%H:%M:%S", errors="coerce"
).dt.hour

# ── 4. Standardize categorical variables ─────────────────────────────────
print("Standardizing categorical variables...")

# --- subject_race ---
race_map = {
    "white": "white",
    "black": "black",
    "hispanic": "hispanic",
    "asian/pacific islander": "asian/pacific islander",
    "asian": "asian/pacific islander",
    "pacific islander": "asian/pacific islander",
    "other": "other",
    "unknown": "unknown",
    "other/unknown": "unknown",
    "na": "unknown",
}
raw_combined["subject_race"] = (
    raw_combined["subject_race"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(race_map)
    .fillna("unknown")
)

# --- subject_sex ---
sex_map = {
    "male": "male",
    "female": "female",
    "m": "male",
    "f": "female",
    "na": "unknown",
    "nan": "unknown",
}
raw_combined["subject_sex"] = (
    raw_combined["subject_sex"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(sex_map)
    .fillna("unknown")
)

# --- reason_for_stop ---
raw_combined["reason_for_stop"] = (
    raw_combined["reason_for_stop"]
    .astype(str)
    .str.strip()
    .str.title()
)
raw_combined.loc[
    raw_combined["reason_for_stop"].isin(["Nan", "Na", ""]), "reason_for_stop"
] = "Unknown"

# --- outcome ---
raw_combined["outcome"] = (
    raw_combined["outcome"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# --- boolean columns: normalize TRUE/FALSE/NA strings to bool/NaN ---
bool_cols = [
    "arrest_made", "citation_issued", "warning_issued",
    "contraband_found", "contraband_drugs", "contraband_weapons",
    "frisk_performed", "search_conducted", "search_person", "search_vehicle",
]
for col in bool_cols:
    raw_combined[col] = (
        raw_combined[col]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({"TRUE": True, "FALSE": False})
    )

# ── 5. Missing-value analysis ────────────────────────────────────────────
print("\nComputing missingness rates...")

# Per-city per-column missingness
missingness_city = {}
for city in FILES:
    subset = raw_combined[raw_combined["city"] == city]
    miss = subset.isnull().mean().round(4) * 100
    missingness_city[city] = miss

miss_df = pd.DataFrame(missingness_city).T  # cities as rows, columns as cols
miss_df.index.name = "city"

# Overall missingness
overall_miss = raw_combined.isnull().mean().round(4) * 100

# Identify columns to drop (>50% missing overall)
cols_to_drop = overall_miss[overall_miss > 50].index.tolist()
print(f"Columns dropped (>50% missing): {cols_to_drop}")

raw_combined.drop(columns=cols_to_drop, inplace=True, errors="ignore")

# Flag rows with missing outcome data
raw_combined["outcome_missing"] = raw_combined["outcome"].isin(["nan", "na", ""]) | raw_combined["outcome"].isna()
n_outcome_missing = raw_combined["outcome_missing"].sum()
print(f"Rows flagged with missing outcome: {n_outcome_missing:,}")

# ── 6. Create binary arrested variable ───────────────────────────────────
print("Creating arrested binary variable...")

# Primary: use arrest_made boolean column (most reliable)
# Fallback: check outcome == "arrest"
raw_combined["arrested"] = np.where(
    raw_combined["arrest_made"] == True, 1,
    np.where(
        raw_combined["outcome"].str.contains("arrest", case=False, na=False), 1, 0
    )
)

arrest_counts = raw_combined["arrested"].value_counts()
print(f"Arrested distribution:\n{arrest_counts}")

# ── 7. Filter to relevant columns ────────────────────────────────────────
keep_cols = [
    "date", "time", "city",
    "subject_race", "subject_sex", "subject_age",
    "reason_for_stop", "outcome",
    "search_conducted", "contraband_found",
    "arrested",
    "year", "month", "day_of_week", "hour",
    "outcome_missing",
]
# Only keep columns that actually exist after the drop step
keep_cols = [c for c in keep_cols if c in raw_combined.columns]

cleaned = raw_combined[keep_cols].copy()

# ── 8. Save cleaned CSV ──────────────────────────────────────────────────
out_csv = OUTPUT_DIR / "nc_traffic_stops_cleaned.csv"
cleaned.to_csv(out_csv, index=False)
print(f"\nSaved cleaned data to {out_csv}")

# ── 9. Print summary ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CLEANED DATASET SUMMARY")
print("=" * 60)
print(f"Shape: {cleaned.shape}")
print(f"\nDtypes:\n{cleaned.dtypes}")
print(f"\nMissingness (%):\n{cleaned.isnull().mean().round(4) * 100}")
print(f"\nRace value counts:\n{cleaned['subject_race'].value_counts()}")
print(f"\nCity value counts:\n{cleaned['city'].value_counts()}")
print(f"\nOutcome value counts:\n{cleaned['outcome'].value_counts()}")
print(f"\nSex value counts:\n{cleaned['subject_sex'].value_counts()}")
print(f"\nArrested value counts:\n{cleaned['arrested'].value_counts()}")

# ══════════════════════════════════════════════════════════════════════════
# 10. Generate cleaning_summary.md
# ══════════════════════════════════════════════════════════════════════════
print("\nGenerating cleaning_summary.md...")

# Build missingness table (markdown)
miss_table_cols = list(miss_df.columns)
miss_header = "| City | " + " | ".join(miss_table_cols) + " |"
miss_sep = "|---" * (len(miss_table_cols) + 1) + "|"
miss_rows = []
for city_name in miss_df.index:
    vals = " | ".join(f"{v:.1f}" for v in miss_df.loc[city_name])
    miss_rows.append(f"| {city_name} | {vals} |")
miss_table = "\n".join([miss_header, miss_sep] + miss_rows)

# Race value counts table
race_vc = cleaned["subject_race"].value_counts()
race_table = "| Race | Count | Percent |\n|---|---|---|\n"
for race_val, cnt in race_vc.items():
    pct = cnt / len(cleaned) * 100
    race_table += f"| {race_val} | {cnt:,} | {pct:.1f}% |\n"

# City value counts table
city_vc = cleaned["city"].value_counts()
city_table = "| City | Count | Percent |\n|---|---|---|\n"
for city_val, cnt in city_vc.items():
    pct = cnt / len(cleaned) * 100
    city_table += f"| {city_val} | {cnt:,} | {pct:.1f}% |\n"

# Outcome value counts table
outcome_vc = cleaned["outcome"].value_counts()
outcome_table = "| Outcome | Count | Percent |\n|---|---|---|\n"
for outcome_val, cnt in outcome_vc.items():
    pct = cnt / len(cleaned) * 100
    outcome_table += f"| {outcome_val} | {cnt:,} | {pct:.1f}% |\n"

# Sex value counts table
sex_vc = cleaned["subject_sex"].value_counts()
sex_table = "| Sex | Count | Percent |\n|---|---|---|\n"
for sex_val, cnt in sex_vc.items():
    pct = cnt / len(cleaned) * 100
    sex_table += f"| {sex_val} | {cnt:,} | {pct:.1f}% |\n"

# Arrested distribution table
arr_vc = cleaned["arrested"].value_counts()
arr_table = "| Arrested | Count | Percent |\n|---|---|---|\n"
for arr_val, cnt in arr_vc.items():
    pct = cnt / len(cleaned) * 100
    arr_table += f"| {arr_val} | {cnt:,} | {pct:.1f}% |\n"

# Dtypes table
dtypes_table = "| Column | Dtype |\n|---|---|\n"
for col_name, dtype in cleaned.dtypes.items():
    dtypes_table += f"| {col_name} | {dtype} |\n"

# Final missingness table
final_miss = cleaned.isnull().mean().round(4) * 100
final_miss_table = "| Column | Missing % |\n|---|---|\n"
for col_name, pct in final_miss.items():
    final_miss_table += f"| {col_name} | {pct:.1f}% |\n"

# Raw info summary
raw_summary = ""
for city_name, info in raw_info.items():
    raw_summary += f"- **{city_name}**: {info['rows']:,} rows, {len(info['columns'])} columns\n"

# Columns dropped
if cols_to_drop:
    dropped_text = "\n".join(f"- `{c}` — {overall_miss[c]:.1f}% missing overall" for c in cols_to_drop)
else:
    dropped_text = "No columns were dropped (all had ≤50% missingness)."

md = f"""# Data Cleaning Summary — NC Traffic Stops (SOPP)

> Generated by `clean_data.py` on the Stanford Open Policing Project data for six North Carolina cities (2000–2015).

---

## 1. Overview of Raw Data

**Source**: Stanford Open Policing Project (SOPP), North Carolina city-level files (downloaded 2020-04-01 release).

**Files loaded**: 6 CSV files from `Stanford Data/` directory.

{raw_summary}

**Combined raw dataset**: {total_raw_rows:,} rows, {len(raw_columns)} columns.

All six files shared an identical schema with these 29 columns:

```
{', '.join(raw_columns)}
```

---

## 2. Standardization Decisions

### Column Names
All column names were converted to lowercase snake_case. Since all six city files already shared the same SOPP schema, no cross-city column renaming was necessary.

### Race Categories (`subject_race`)
Original values were lowercased and mapped to a consistent set:

| Raw Value | Mapped To |
|---|---|
| `white` | white |
| `black` | black |
| `hispanic` | hispanic |
| `asian/pacific islander`, `asian`, `pacific islander` | asian/pacific islander |
| `other` | other |
| `unknown`, `other/unknown`, `na`, `nan` | unknown |

### Sex Categories (`subject_sex`)
| Raw Value | Mapped To |
|---|---|
| `male`, `m` | male |
| `female`, `f` | female |
| `na`, `nan`, missing | unknown |

### Reason for Stop (`reason_for_stop`)
- Stripped whitespace, converted to Title Case for consistency.
- Values of `Nan`, `Na`, or empty strings were set to `Unknown`.

### Outcome (`outcome`)
- Lowercased and stripped. Original values include: `warning`, `citation`, `arrest`, `summons`, `nan`.

### Boolean Columns
The following columns were converted from string `TRUE`/`FALSE` to Python boolean `True`/`False` (with `NA` → `NaN`):

```
arrest_made, citation_issued, warning_issued, contraband_found,
contraband_drugs, contraband_weapons, frisk_performed,
search_conducted, search_person, search_vehicle
```

---

## 3. Missing Data

### Missingness Rates by Column and City (%, before dropping)

{miss_table}

### Columns Dropped (>50% missing overall)

{dropped_text}

### Outcome Missingness
- **{n_outcome_missing:,}** rows were flagged with `outcome_missing = True` (outcome value was `nan`, `na`, or empty).
- These rows were **retained** in the dataset with the flag for downstream analysis decisions.

---

## 4. Feature Engineering

| New Column | Description | Derivation |
|---|---|---|
| `city` | City name | Extracted from source filename |
| `year` | Year of stop | From `date` column |
| `month` | Month of stop (1–12) | From `date` column |
| `day_of_week` | Day name (Monday–Sunday) | From `date` column |
| `hour` | Hour of stop (0–23) | From `time` column |
| `arrested` | Binary arrest indicator (0/1) | See below |
| `outcome_missing` | Flag for missing outcome data | True if outcome was nan/na/empty |

---

## 5. Outcome Variable Construction (`arrested`)

The binary `arrested` variable was constructed using a two-step approach:

1. **Primary**: If `arrest_made == True` → `arrested = 1`
2. **Fallback**: Else if `outcome` string contains `"arrest"` → `arrested = 1`
3. **Otherwise**: `arrested = 0`

This ensures we capture arrests even if one indicator was missing, while avoiding double-counting.

### Arrest Distribution

{arr_table}

---

## 6. Final Dataset Summary

**Output file**: `nc_traffic_stops_cleaned.csv`

**Shape**: {cleaned.shape[0]:,} rows × {cleaned.shape[1]} columns

### Columns Retained

{dtypes_table}

### Remaining Missingness

{final_miss_table}

### Value Counts — Race

{race_table}

### Value Counts — City

{city_table}

### Value Counts — Outcome

{outcome_table}

### Value Counts — Sex

{sex_table}

### Value Counts — Arrested

{arr_table}

---

## 7. Decisions and Assumptions

1. **Date range**: No explicit filter was applied to restrict years. The raw data spans the full range present in each city's file (roughly 2000–2015, varying by city).

2. **Duplicate rows**: Not explicitly deduplicated. The SOPP data uses `raw_row_number` as a unique identifier within each city file; since we added a `city` column, the combination is unique.

3. **Subject age**: Kept as-is (numeric). Some extreme values (e.g., age < 10 or > 100) may exist but were not filtered, leaving that decision for analysis.

4. **Boolean NA handling**: For boolean columns like `search_conducted` and `contraband_found`, `NA` strings were mapped to `NaN` (not `False`). This preserves the distinction between "not searched" and "search status unknown."

5. **Outcome vs. arrest_made**: Both fields encode arrest information. We prioritized `arrest_made` (boolean) over `outcome` (string) since it is more explicit, but used `outcome` as a fallback.

6. **Contraband found**: Retained only `contraband_found` (the aggregate flag). Dropped `contraband_drugs` and `contraband_weapons` if they exceeded the 50% missingness threshold.

7. **Columns dropped from final output**: Administrative columns (`raw_row_number`, `location`, `county_name`, `officer_id_hash`, `department_name`, `type`, `arrest_made`, `citation_issued`, `warning_issued`, `search_person`, `search_vehicle`, `search_basis`, `reason_for_frisk`, `reason_for_search`, `raw_ethnicity`, `raw_race`, `raw_action_description`, `frisk_performed`) were excluded from the final dataset to keep only analysis-relevant columns. The `arrested` binary variable supersedes `arrest_made`, and `search_conducted` supersedes the granular search columns.

8. **Time parsing**: Times stored as `HH:MM:SS` strings were parsed; entries that could not be parsed (including `NA`) resulted in `NaT`/`NaN` for the `hour` feature.

---

*This summary was auto-generated as part of the data cleaning pipeline for the SDS 357 project.*
"""

md_path = OUTPUT_DIR / "cleaning_summary.md"
with open(md_path, "w") as f:
    f.write(md)

print(f"Saved cleaning summary to {md_path}")
print("\nDone!")
