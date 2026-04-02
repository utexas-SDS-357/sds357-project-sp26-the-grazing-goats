"""
Shared constants and human-readable labels for plots and tables.

Import in notebooks, e.g.:
    from constants import SEED, PALETTE_RACE, pretty_predictor_label
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Reproducibility & ordering (used in data prep and plots)
# ---------------------------------------------------------------------------

SEED = 42

PALETTE_RACE = {
    "black": "#4C72B0",
    "white": "#DD8452",
    "hispanic": "#55A868",
    "asian/pacific islander": "#C44E52",
}

RACE_ORDER = ["black", "white", "hispanic", "asian/pacific islander"]

CITY_ORDER = [
    "Charlotte",
    "Raleigh",
    "Greensboro",
    "Fayetteville",
    "Winston-Salem",
    "Durham",
]

# ---------------------------------------------------------------------------
# DataFrame column name -> axis / table header (no raw snake_case on figures)
# ---------------------------------------------------------------------------

COLUMN_LABELS: dict[str, str] = {
    "violent_crime_rate": "Violent Crime Rate (per 1,000)",
    "property_crime_rate": "Property Crime Rate (per 1,000)",
    "subject_age": "Age (years)",
    "subject_race": "Race",
    "subject_sex": "Sex",
    "city": "City",
    "arrest_rate": "Arrest Rate (%)",
    "search_rate": "Search Rate (%)",
    "outcome": "Outcome",
    "search_conducted": "Search Conducted",
    "arrested": "Arrested",
    "year": "Year",
    "month": "Month",
    "hour": "Hour",
    "day_of_week": "Day of Week",
    "reason_for_stop": "Reason for Stop",
}

# ---------------------------------------------------------------------------
# Categorical values as they appear in data -> plot/tick labels
# ---------------------------------------------------------------------------

OUTCOME_LABELS: dict[str, str] = {
    "warning": "Warning",
    "citation": "Citation",
    "arrest": "Arrest",
}

# ---------------------------------------------------------------------------
# Race-blind classifier feature order (matches X column order in 03_predictive)
# ---------------------------------------------------------------------------

ML_FEATURE_LABELS: list[str] = [
    "Age",
    "Stop Reason",
    "City",
    "Search Conducted",
    "Hour",
    "Year",
    "Month",
    "Day of Week",
    "Violent Crime Rate (FBI)",
]


def label_column(name: str) -> str:
    """Display label for a DataFrame column, or title-cased fallback."""
    key = str(name)
    if key in COLUMN_LABELS:
        return COLUMN_LABELS[key]
    return key.replace("_", " ").title()


def label_outcome(value: str) -> str:
    """Display label for an outcome category."""
    s = str(value)
    return OUTCOME_LABELS.get(s.lower(), s.replace("_", " ").title())


def pretty_predictor_label(name: str) -> str:
    """Human-readable labels for logistic-regression / odds-ratio plot rows."""
    name = str(name)
    if name == "const":
        return "Intercept"
    if name == "subject_age":
        return "Age (years)"
    if name == "violent_crime_rate":
        return "Violent Crime Rate (per 1,000)"
    if name == "property_crime_rate":
        return "Property Crime Rate (per 1,000)"
    if name.startswith("subject_race_"):
        rest = name[len("subject_race_") :]
        rest = "/".join(p.strip().title() for p in rest.split("/"))
        return f"Race: {rest}"
    if name.startswith("subject_sex_"):
        return f"Sex: {name[len('subject_sex_'):].title()}"
    if name.startswith("reason_cat_"):
        return f"Stop reason: {name[len('reason_cat_'):]}"
    if name.startswith("city_"):
        return f"City: {name[len('city_'):]}"
    return name.replace("_", " ").title()
