[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/MFzEnxem)

## Running the data cleaning pipeline

All commands below assume you are in the project root directory.

### 1. Create and activate the virtual environment (first time only)

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS / Linux
```

If you open a new terminal later, just re‑activate the environment:

```bash
source .venv/bin/activate
```

### 2. Install required Python packages

```bash
pip install -r requirements.txt
```

### 3. Make sure the raw data files are in place

Place the following SOPP CSV files in the `raw_data/` directory in the project root:

- `nc_charlotte_2020_04_01.csv`
- `nc_durham_2020_04_01.csv`
- `nc_fayetteville_2020_04_01.csv`
- `nc_greensboro_2020_04_01.csv`
- `nc_raleigh_2020_04_01.csv`
- `nc_winston-salem_2020_04_01.csv`

The directory structure should look like:

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

### 4. Run the cleaning script

With the virtual environment activated, run:

```bash
python clean_data.py
```

This will:

- Load and combine the six city CSV files from `raw_data/`
- Clean and standardize the data
- Write the following outputs into the same directory as the raw files:
  - `nc_traffic_stops_cleaned.csv`
  - `nc_traffic_stops_cleaned.parquet`
  - `cleaning_summary.md`

You can then use these cleaned files for your analysis and visualizations.
