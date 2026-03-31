[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/MFzEnxem)

# Racial Disparities in Traffic Stops (North Carolina)

## Project Overview
This project analyzes racial disparities in traffic stop outcomes across six major North Carolina cities (Charlotte, Raleigh, Greensboro, Durham, Winston-Salem, Fayetteville) using data from the Stanford Open Policing Project (SOPP) and FBI Uniform Crime Reporting (UCR).

Our goal is to:
- Examine whether arrest outcomes differ by race after controlling for situational factors  
- Build a race-blind predictive model to identify “unexpected” arrests  
- Evaluate fairness using post-search arrest rates  

## Data Description
- **Primary Data:** Stanford Open Policing Project (SOPP)  
  - ~4.3 million traffic stops (2007–2014) across 6 cities  
  - Includes driver demographics, stop reason, and outcomes  

- **Supplemental Data:** FBI’s Uniform Crime Reporting (FBI UCR) 
  - County-level population and law enforcement data for North Carolina cities from 2000-2015
  - Includes city population, total law enforcement employees, total officers, and total civilians 

---

## Data cleaning pipeline

0. Download SOPP CSV files and put them in the `raw_data/` directory in the root:

```text
.
├── clean_data.py
├── requirements.txt
├── raw_data/
│   ├── nc_charlotte_2020_04_01.csv
│   ├── nc_durham_2020_04_01.csv
│   ├── nc_fayetteville_2020_04_01.csv
│   ├── nc_greensboro_2020_04_01.csv
│   ├── nc_raleigh_2020_04_01.csv
│   └── nc_winston-salem_2020_04_01.csv
```

1. Set up the environment
```bash
python3 -m venv .venv # only run this the first time
source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

2. Run the cleaning script

```bash
python clean_data.py
```

## EDA
```bash
python eda_analysis.py
```

## Model
```bash
python modeling_analysis.py
```
---

## Workflow 

**Raw Data → Cleaning → EDA → Modeling → Results**

- `clean_data.py`: merges and cleans city-level datasets, creates features (e.g., time, demographics, arrest indicator)  
- `eda_analysis.py`: generates summary statistics and visualizations of trends across race, city, and stop reason  
- `modeling_analysis.py`: builds a race-blind gradient boosted model to predict arrest outcomes  

### Example use
Given a traffic stop’s context (reason, time, location), the model predicts whether an arrest is expected.  
Stops where actual outcomes differ from predictions are flagged for further fairness analysis.


## Dependencies
All required Python packages are listed in:
```bash
requirements.txt
```
