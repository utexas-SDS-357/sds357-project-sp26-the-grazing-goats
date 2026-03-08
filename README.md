[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/MFzEnxem)

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