# Global Philanthropy Dashboard (Web App)

This repo contains a lightweight web dashboard built from the OECD global philanthropy funding dataset, following the 4-view architecture described in `Global_Philanthropy_Dashboard_Design_Doc.docx`.

## What’s included

- `app.py`: Streamlit web app with:
  - Global filters (year, donor country, donor, recipient region/country, sector, flow type)
  - View 1: global choropleth + top recipients
  - View 2: sector deep-dive + trend line + optional thematic marker filtering
  - View 3: donor leaderboard + simple comparison
  - View 4: searchable project explorer
- `scripts/prepare_data.py`: optional preprocessing that writes fast-loading parquet files and aggregates by `row_id` to avoid double counting.

## Setup

Create a venv (recommended) and install dependencies:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Optional: generate parquet outputs (faster app startup):

```bash
./.venv/bin/python scripts/prepare_data.py
```

## Run the app

```bash
./.venv/bin/streamlit run app.py
```

## Data notes (from the design doc)

- Use `usd_disbursements_defl` as the primary metric for year-over-year comparability.
- When computing totals across the full dataset, aggregate by `row_id` to avoid double-counting projects split across multiple sectors.