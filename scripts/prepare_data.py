from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_transactions(csv_path: Path) -> pd.DataFrame:
    # The source CSV has mixed-type columns; be explicit so we don't end up with
    # float columns for IDs/names just because the sample slice is sparse.
    dtype = {
        "year": "string",
        "organization_name": "string",
        "region": "string",
        "region_macro": "string",
        "country": "string",
        "grant_recipient_project_title": "string",
        "project_description": "string",
        "expected_duration": "string",
        "type_of_flow": "string",
        "Donor_country": "string",
        "financial_instrument": "string",
        "modality_of_giving": "string",
        "gender_dimension": "string",
        "additional_info": "string",
        "sdg_focus": "string",
        "row_id": "string",
        "subsector_description": "string",
        "sector_description": "string",
        "channel_code": "string",
        "channel_name": "string",
        "channel_reported_name": "string",
    }

    df = pd.read_csv(csv_path, dtype=dtype, low_memory=False)

    for c in [
        "usd_disbursements_defl",
        "usd_commitment_defl",
        "gender_marker",
        "climate_change_mitigation",
        "climate_change_adaptation",
        "environment",
        "biodiversity",
        "desertification",
        "nutrition",
        "subsector",
        "Sector",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["usd_disbursements_defl"] = df["usd_disbursements_defl"].fillna(0.0)
    if "usd_commitment_defl" in df.columns:
        df["usd_commitment_defl"] = df["usd_commitment_defl"].fillna(0.0)

    return df


def _combine_unique(series: pd.Series) -> str:
    vals = [v for v in series.dropna().astype(str).unique().tolist() if v.strip()]
    return "; ".join(sorted(vals))


def _aggregate_projects(transactions: pd.DataFrame) -> pd.DataFrame:
    # The design doc notes some projects appear multiple times due to sector splits.
    # We aggregate by row_id so totals don't double-count.
    def first_non_null(s: pd.Series):
        s = s.dropna()
        if len(s) == 0:
            return pd.NA
        return s.iloc[0]

    agg = {
        "usd_disbursements_defl": "sum",
    }
    if "usd_commitment_defl" in transactions.columns:
        agg["usd_commitment_defl"] = "sum"

    passthrough_first = [
        "year",
        "organization_name",
        "Donor_country",
        "region",
        "region_macro",
        "country",
        "type_of_flow",
        "grant_recipient_project_title",
        "project_description",
        "expected_duration",
        "channel_name",
        "channel_reported_name",
    ]
    for c in passthrough_first:
        if c in transactions.columns:
            agg[c] = first_non_null

    # Keep sector/subsector context without double-counting.
    for c in ["sector_description", "subsector_description", "sdg_focus"]:
        if c in transactions.columns:
            agg[c] = _combine_unique

    # Marker fields: max is a reasonable "any focus" roll-up (0-2).
    for c in [
        "gender_marker",
        "climate_change_mitigation",
        "climate_change_adaptation",
        "environment",
        "biodiversity",
        "desertification",
        "nutrition",
    ]:
        if c in transactions.columns:
            agg[c] = "max"

    projects = (
        transactions.groupby("row_id", dropna=False, as_index=False)
        .agg(agg)
        .sort_values("usd_disbursements_defl", ascending=False)
    )
    return projects


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="OECD Dataset.xlsx - complete_p4d3_df.csv",
        help="Path to the OECD CSV file",
    )
    parser.add_argument(
        "--out-dir",
        default="data",
        help="Directory to write parquet outputs",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transactions = _read_transactions(csv_path)
    projects = _aggregate_projects(transactions)

    transactions_out = out_dir / "transactions.parquet"
    projects_out = out_dir / "projects.parquet"

    transactions.to_parquet(transactions_out, index=False)
    projects.to_parquet(projects_out, index=False)

    print(f"Wrote {len(transactions):,} transactions -> {transactions_out}")
    print(f"Wrote {len(projects):,} projects      -> {projects_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

