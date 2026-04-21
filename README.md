# Racial Disparities in North Carolina Traffic Stop Arrests

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/MFzEnxem)

**SDS 357 — Spring 2026 | The Grazing Goats**

Analysis of 4M+ traffic stops across six North Carolina cities (2000–2015) using data from the [Stanford Open Policing Project](https://openpolicing.stanford.edu/) (SOPP), supplemented with FBI Uniform Crime Report statistics.

## Research Question

Do racial disparities exist in traffic stop arrest outcomes across NC cities, and can arrest likelihood be predicted from race-blind situational factors alone?

## Project Structure

```
.
├── 00_FBI_CLEANING_FILE.ipynb            # FBI UCR raw XLS → cleaned CSV pipeline
├── 01_EDA.ipynb                          # Exploratory data analysis & FBI crime context
├── 02_INFERENTIAL_ANALYSIS.ipynb         # Logistic regression (odds ratios)
├── 03_PREDICTIVE_MODEL.ipynb             # Race-blind gradient-boosted classifier
├── 04_FLAGGING_SYSTEM.ipynb              # Flagging system for stop anomalies
├── COMPLETE_PROJECT.ipynb                # Full combined analysis
├── data/
│   ├── nc_traffic_stops_cleaned.parquet  # Cleaned traffic stop data
│   ├── nc_fbi_crime_data_clean.csv       # FBI UCR crime rates by city-year
│   └── NC_FBI_UCR_RAW_DATA/                      # Raw FBI UCR spreadsheets (2000–2015)
├── eda/                                  # Pre-generated EDA visualizations (PNGs)
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the Analysis

All notebook outputs are pre-executed and viewable directly on GitHub.

### 1. Data Cleaning

**`00_FBI_CLEANING_FILE.ipynb`**:
- Reads raw FBI UCR `.xls` spreadsheets from `data/NC_FBI_UCR_RAW_DATA/`
- Standardizes column names across years, filters to the six target cities
- Outputs `nc_fbi_crime_data_clean.csv`

### 2. Exploratory Data Analysis

**`01_EDA.ipynb`**:
- Arrest rates by race (with 95% confidence intervals)
- Arrest rate heatmap (race × city)
- Search rate disparities by race
- FBI crime rate context (violent & property crime vs. arrest rates)

### 3. Inferential Analysis

**`02_INFERENTIAL_ANALYSIS.ipynb`**:
- Logistic regression: P(Arrest | demographics, stop conditions, FBI crime rate)
- Odds ratio visualization with confidence intervals
- Quantifies independent contribution of race after controlling for confounders

### 4. Predictive Modeling

**`03_PREDICTIVE_MODEL.ipynb`**:
- Race-blind `HistGradientBoostingClassifier` (no race or sex features)
- Handles 97:3 class imbalance with `class_weight="balanced"`
- ROC/PR curves, optimal F1 threshold selection, confusion matrix
- Permutation importance to identify key predictors

### 5. Flagging System

**`04_FLAGGING_SYSTEM.ipynb`**:
- Flagging system for identifying anomalous stop patterns

### 6. Complete Project

**`COMPLETE_PROJECT.ipynb`**:
- Full combined end-to-end analysis

## Data Sources

| Source | Description |
|--------|-------------|
| [Stanford Open Policing Project](https://openpolicing.stanford.edu/) | Traffic stop records for Charlotte, Durham, Fayetteville, Greensboro, Raleigh, Winston-Salem |
| [FBI Uniform Crime Reports](https://ucr.fbi.gov/) | City-level violent and property crime rates (2000–2015) |

## Requirements

- Python 3.9+
- See `requirements.txt` for package dependencies (pandas, scikit-learn, statsmodels, seaborn, etc.)
