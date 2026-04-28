from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import plotly.express as px
import streamlit as st


DATA_DIR = Path("data")
DEFAULT_CSV = Path("OECD Dataset.xlsx - complete_p4d3_df.csv")

GLOBAL_THEME_COLOR_SCALE = "Teal"


@dataclass(frozen=True)
class DataBundle:
    transactions: pd.DataFrame  # sector-split rows (may double count across sectors)
    projects: pd.DataFrame  # aggregated by row_id (no double counting)


def _money(x: float) -> str:
    try:
        return f"${x:,.2f}M"
    except Exception:
        return str(x)


def _read_transactions(csv_path: Path) -> pd.DataFrame:
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

    for c in ["sector_description", "subsector_description", "sdg_focus"]:
        if c in transactions.columns:
            agg[c] = _combine_unique

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


@st.cache_data(show_spinner="Loading data…")
def load_data() -> DataBundle:
    tx_pq = DATA_DIR / "transactions.parquet"
    pr_pq = DATA_DIR / "projects.parquet"

    if tx_pq.exists() and pr_pq.exists():
        transactions = pd.read_parquet(tx_pq)
        projects = pd.read_parquet(pr_pq)
        return DataBundle(transactions=transactions, projects=projects)

    transactions = _read_transactions(DEFAULT_CSV)
    projects = _aggregate_projects(transactions)
    return DataBundle(transactions=transactions, projects=projects)


def _sorted_unique(values: Iterable) -> list:
    out = sorted({v for v in values if pd.notna(v) and str(v).strip()})
    return [str(v) for v in out]


def pretty_label(s: str) -> str:
    # Convert snake_case / weird casing into "Title Case" for display.
    s = str(s or "").strip()
    if not s:
        return s
    s = s.replace("_", " ").replace("  ", " ").strip()
    return " ".join(w[:1].upper() + w[1:] for w in s.split(" "))


def _is_numeric_series(s: pd.Series) -> bool:
    try:
        return pd.api.types.is_numeric_dtype(s)
    except Exception:
        return False


def _apply_column_filter(df: pd.DataFrame, col: str, mode: str, value):
    if col not in df.columns:
        return df

    s = df[col]
    if mode == "contains":
        q = str(value or "").strip()
        if not q:
            return df
        return df[s.astype("string").str.contains(q, case=False, na=False)]

    if mode == "equals_any":
        vals = value or []
        if not vals:
            return df
        return df[s.astype("string").isin([str(v) for v in vals])]

    if mode == "range":
        lo, hi = value
        sn = pd.to_numeric(s, errors="coerce")
        return df[sn.notna() & (sn >= float(lo)) & (sn <= float(hi))]

    return df


def _explode_multi_value_column(
    df: pd.DataFrame, col: str, sep: str = "; "
) -> pd.DataFrame:
    """
    Split a multi-value string column into multiple rows (explode).
    Used to avoid 'Health; Environment' becoming its own category and instead
    count the same project in both sectors when desired.
    """
    if col not in df.columns:
        return df
    out = df.copy()
    s = out[col].astype("string")
    out[col] = s.fillna("").map(
        lambda x: [p.strip() for p in str(x).split(sep) if p.strip()]
    )
    out = out.explode(col, ignore_index=True)
    out = out[out[col].astype("string").str.len() > 0]
    return out


@dataclass(frozen=True)
class Filters:
    years: Optional[set[str]]
    donor_countries: Optional[set[str]]
    donor_orgs: Optional[set[str]]
    region_macros: Optional[set[str]]
    countries: Optional[set[str]]
    sectors: Optional[set[str]]
    flow_types: Optional[set[str]]
    thematic_field: Optional[str]
    thematic_min: Optional[int]
    sdg_query: Optional[str]


def _reset_global_filters(defaults: dict):
    for k, v in defaults.items():
        st.session_state[k] = v


def build_filters(data: DataBundle) -> Filters:
    tx = data.transactions

    years = _sorted_unique(tx.get("year", pd.Series([], dtype="string")))
    donor_countries = _sorted_unique(tx.get("Donor_country", pd.Series([], dtype="string")))
    donor_orgs = _sorted_unique(tx.get("organization_name", pd.Series([], dtype="string")))
    region_macros = _sorted_unique(tx.get("region_macro", pd.Series([], dtype="string")))
    countries = _sorted_unique(tx.get("country", pd.Series([], dtype="string")))
    sectors = _sorted_unique(tx.get("sector_description", pd.Series([], dtype="string")))
    flow_types = _sorted_unique(tx.get("type_of_flow", pd.Series([], dtype="string")))

    defaults = {
        "gf_years": years,
        "gf_donor_countries": donor_countries,
        "gf_donor_orgs": donor_orgs,
        "gf_region_macros": region_macros,
        "gf_countries": countries,
        "gf_sectors": sectors,
        "gf_flow_types": flow_types,
        "gf_marker_field": "None",
        "gf_marker_min": 1,
        "gf_sdg_query": "",
    }

    with st.sidebar:
        st.markdown("## How to use")
        with st.expander("Quick guide", expanded=True):
            st.write(
                "Use the **Global filters** to slice the dataset, then explore via the tabs:\n"
                "- **Global overview**: where funding goes (country totals)\n"
                "- **Sector deep-dive**: where funding is directed by sector + trends\n"
                "- **Donors**: top donors + compare selected donors by sector\n"
                "- **Explorer**: search and filter projects row-by-row"
            )
            st.write(
                f"**Tip**: totals in the map/leaderboards are computed at the **project ({pretty_label('row_id')})** level to avoid double counting."
            )

        if st.button("Reset all global filters", use_container_width=True):
            _reset_global_filters(defaults)
            st.rerun()

        st.markdown("### Global filters")

        sel_years = st.multiselect("Year", years, default=years, key="gf_years")
        sel_donor_countries = st.multiselect(
            "Donor country", donor_countries, default=donor_countries, key="gf_donor_countries"
        )
        sel_donor_orgs = st.multiselect(
            "Donor (foundation)", donor_orgs, default=donor_orgs, key="gf_donor_orgs"
        )
        sel_region_macros = st.multiselect(
            "Region (macro)", region_macros, default=region_macros, key="gf_region_macros"
        )
        sel_countries = st.multiselect(
            "Recipient country", countries, default=countries, key="gf_countries"
        )
        sel_sectors = st.multiselect("Sector", sectors, default=sectors, key="gf_sectors")
        sel_flow_types = st.multiselect(
            "Type of flow", flow_types, default=flow_types, key="gf_flow_types"
        )

        st.divider()
        st.markdown("### Thematic overlays")
        thematic_options = [
            "None",
            "gender_marker",
            "climate_change_mitigation",
            "climate_change_adaptation",
            "environment",
            "biodiversity",
            "desertification",
            "nutrition",
        ]
        thematic_field = st.selectbox(
            "Marker field",
            thematic_options,
            index=0,
            format_func=lambda x: "None" if x == "None" else pretty_label(x),
            key="gf_marker_field",
        )
        thematic_min = None
        if thematic_field != "None":
            thematic_min = st.radio("Minimum score", [1, 2], horizontal=True, key="gf_marker_min")

        st.divider()
        st.markdown("### SDG filter")
        sdg_query = st.text_input(
            f"Filter {pretty_label('sdg_focus')} (substring match)", value="", key="gf_sdg_query"
        ).strip()

    return Filters(
        years=set(sel_years) if len(sel_years) != len(years) else None,
        donor_countries=set(sel_donor_countries) if len(sel_donor_countries) != len(donor_countries) else None,
        donor_orgs=set(sel_donor_orgs) if len(sel_donor_orgs) != len(donor_orgs) else None,
        region_macros=set(sel_region_macros) if len(sel_region_macros) != len(region_macros) else None,
        countries=set(sel_countries) if len(sel_countries) != len(countries) else None,
        sectors=set(sel_sectors) if len(sel_sectors) != len(sectors) else None,
        flow_types=set(sel_flow_types) if len(sel_flow_types) != len(flow_types) else None,
        thematic_field=None if thematic_field == "None" else thematic_field,
        thematic_min=thematic_min,
        sdg_query=sdg_query or None,
    )


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    out = df

    def in_set(col: str, s: Optional[set[str]]):
        nonlocal out
        if s is None or col not in out.columns:
            return
        out = out[out[col].astype("string").isin(list(s))]

    in_set("year", f.years)
    in_set("Donor_country", f.donor_countries)
    in_set("organization_name", f.donor_orgs)
    in_set("region_macro", f.region_macros)
    in_set("country", f.countries)
    in_set("sector_description", f.sectors)
    in_set("type_of_flow", f.flow_types)

    if f.sdg_query and "sdg_focus" in out.columns:
        out = out[out["sdg_focus"].astype("string").str.contains(f.sdg_query, case=False, na=False)]

    if f.thematic_field and f.thematic_min is not None and f.thematic_field in out.columns:
        s = pd.to_numeric(out[f.thematic_field], errors="coerce")
        out = out[s.notna() & (s >= float(f.thematic_min))]

    return out


def view_global_overview(projects: pd.DataFrame):
    st.markdown("## Global overview map")
    st.caption(
        f"Country totals are computed from **projects aggregated by {pretty_label('row_id')}** to avoid double counting."
    )

    by_country = projects.groupby("country", dropna=False, as_index=False)["usd_disbursements_defl"].sum()

    # Enrich tooltips with "top donor" and "top sector" within each country.
    # Note: sectors in `projects` are stored as multi-value strings; explode so a project
    # can contribute to each sector in the tooltip summary.
    if "organization_name" in projects.columns:
        top_donor = (
            projects.groupby(["country", "organization_name"], as_index=False)["usd_disbursements_defl"]
            .sum()
            .sort_values(["country", "usd_disbursements_defl"], ascending=[True, False])
            .drop_duplicates("country")[["country", "organization_name"]]
            .rename(columns={"organization_name": "top_donor"})
        )
        by_country = by_country.merge(top_donor, on="country", how="left")

    if "sector_description" in projects.columns:
        exploded = _explode_multi_value_column(projects, "sector_description", sep="; ")
        top_sector = (
            exploded.groupby(["country", "sector_description"], as_index=False)["usd_disbursements_defl"]
            .sum()
            .sort_values(["country", "usd_disbursements_defl"], ascending=[True, False])
            .drop_duplicates("country")[["country", "sector_description"]]
            .rename(columns={"sector_description": "top_sector"})
        )
        by_country = by_country.merge(top_sector, on="country", how="left")

    by_country = by_country.sort_values("usd_disbursements_defl", ascending=False)

    fig = px.choropleth(
        by_country,
        locations="country",
        locationmode="country names",
        color="usd_disbursements_defl",
        hover_name="country",
        hover_data={
            "usd_disbursements_defl": ":,.3f",
            "top_sector": True if "top_sector" in by_country.columns else False,
            "top_donor": True if "top_donor" in by_country.columns else False,
        },
        color_continuous_scale=GLOBAL_THEME_COLOR_SCALE,
        labels={"usd_disbursements_defl": "USD disbursements (deflated)"},
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    top = by_country.head(10).copy()
    top["usd_disbursements_defl"] = top["usd_disbursements_defl"].astype(float)
    fig2 = px.bar(
        top[::-1],
        x="usd_disbursements_defl",
        y="country",
        orientation="h",
        labels={"usd_disbursements_defl": "USD disbursements (deflated)", "country": ""},
        title="Top 10 recipient countries",
    )
    st.plotly_chart(fig2, use_container_width=True)


def view_sector_deep_dive(transactions: pd.DataFrame):
    st.markdown("## Sector & thematic deep-dive")
    st.caption(
        "Sector breakdowns use sector-split transaction rows. Multi-sector projects can contribute to multiple sectors."
    )

    selected_for_trend = None
    if "sector_description" in transactions.columns:
        sector_opts = _sorted_unique(transactions["sector_description"].dropna().astype("string"))
        selected_for_trend = st.selectbox(
            "Trend sector (optional)",
            ["All sectors"] + sector_opts,
            index=0,
        )

    by_sector = (
        transactions.groupby("sector_description", dropna=False, as_index=False)["usd_disbursements_defl"]
        .sum()
        .sort_values("usd_disbursements_defl", ascending=False)
    )
    by_sector = by_sector[by_sector["sector_description"].notna()]

    treemap = px.treemap(
        by_sector,
        path=["sector_description"],
        values="usd_disbursements_defl",
        color="usd_disbursements_defl",
        color_continuous_scale="Teal",
    )
    treemap.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(treemap, use_container_width=True)

    if "year" in transactions.columns:
        tdf = transactions
        if selected_for_trend and selected_for_trend != "All sectors" and "sector_description" in tdf.columns:
            tdf = tdf[tdf["sector_description"].astype("string") == selected_for_trend]

        trend = tdf.groupby(["year"], as_index=False)["usd_disbursements_defl"].sum().sort_values("year")
        line = px.line(
            trend,
            x="year",
            y="usd_disbursements_defl",
            markers=True,
            labels={"usd_disbursements_defl": "USD disbursements (deflated)"},
            title="Funding trend over time",
        )
        st.plotly_chart(line, use_container_width=True)


def view_donor_leaderboard(projects: pd.DataFrame):
    st.markdown("## Donor leaderboard")

    donors = (
        projects.groupby("organization_name", dropna=False, as_index=False)["usd_disbursements_defl"]
        .sum()
        .sort_values("usd_disbursements_defl", ascending=False)
    )
    donors = donors[donors["organization_name"].notna()]

    left, right = st.columns([2, 1])
    with left:
        st.markdown("### Top donors")
        st.dataframe(
            donors.head(30),
            use_container_width=True,
            hide_index=True,
            column_config={
                "usd_disbursements_defl": st.column_config.NumberColumn(
                    "USD disbursements (deflated)", format="$%0.3f"
                )
            },
        )
    with right:
        st.markdown("### Compare donors")
        options = donors["organization_name"].tolist()
        sel = st.multiselect("Select 1–3 donors", options, default=options[:2])
        sel = sel[:3]

    if sel:
        subset = projects[projects["organization_name"].isin(sel)]
        # IMPORTANT: `projects` aggregates sector descriptions into a multi-value string
        # (e.g., "Health; Environment"). For the donor-by-sector view we want to count
        # a multi-sector donation in *each* sector, not as its own combined category.
        subset_exploded = _explode_multi_value_column(subset, "sector_description", sep="; ")

        by = (
            subset_exploded.groupby(["organization_name", "sector_description"], as_index=False)[
                "usd_disbursements_defl"
            ]
            .sum()
            .sort_values("usd_disbursements_defl", ascending=False)
        )
        by = by[by["sector_description"].notna()]

        st.markdown("### Selected donors by sector (clean view)")
        top_k = st.slider("Show top K sectors", min_value=5, max_value=25, value=12, step=1)

        # Pick top sectors by total across selected donors, then show grouped bars.
        top_sectors = (
            by.groupby("sector_description", as_index=False)["usd_disbursements_defl"]
            .sum()
            .sort_values("usd_disbursements_defl", ascending=False)
            .head(top_k)["sector_description"]
            .tolist()
        )
        by_top = by[by["sector_description"].isin(top_sectors)].copy()
        by_top["sector_description"] = pd.Categorical(
            by_top["sector_description"], categories=top_sectors[::-1], ordered=True
        )

        fig = px.bar(
            by_top,
            x="usd_disbursements_defl",
            y="sector_description",
            color="organization_name",
            barmode="group",
            orientation="h",
            labels={"sector_description": "", "usd_disbursements_defl": "USD disbursements (deflated)"},
        )
        fig.update_layout(legend_title_text="Donor")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show pivot table (sectors x donors)"):
            pivot = (
                by_top.pivot_table(
                    index="sector_description",
                    columns="organization_name",
                    values="usd_disbursements_defl",
                    aggfunc="sum",
                    fill_value=0.0,
                )
                .sort_index()
            )
            st.dataframe(
                pivot,
                use_container_width=True,
                column_config={c: st.column_config.NumberColumn(pretty_label(c), format="$%0.3f") for c in pivot.columns},
            )


def view_recipient_explorer(projects: pd.DataFrame):
    st.markdown("## Recipient & project explorer")

    q = st.text_input("Search project title / description", value="").strip()
    df = projects

    if q:
        title = df.get("grant_recipient_project_title", pd.Series([], dtype="string")).astype("string")
        desc = df.get("project_description", pd.Series([], dtype="string")).astype("string")
        mask = title.str.contains(q, case=False, na=False) | desc.str.contains(q, case=False, na=False)
        df = df[mask]

    with st.expander("Advanced filters (any column)"):
        st.caption(
            "Add column-specific filters here for more precise exploration (text contains, exact match, or numeric range)."
        )
        all_cols = [c for c in df.columns.tolist() if c not in {"project_description"}]
        chosen_cols = st.multiselect(
            "Columns to filter",
            all_cols,
            default=[],
            format_func=pretty_label,
        )

        for col in chosen_cols:
            s = df[col]
            is_num = _is_numeric_series(s) or col in {"usd_disbursements_defl", "usd_commitment_defl"}

            c1, c2 = st.columns([1, 3])
            with c1:
                mode = st.selectbox(
                    f"{pretty_label(col)} filter type",
                    ["contains", "equals_any"] + (["range"] if is_num else []),
                    key=f"adv_mode_{col}",
                )

            with c2:
                if mode == "contains":
                    v = st.text_input(
                        f"{pretty_label(col)} contains", value="", key=f"adv_contains_{col}"
                    )
                    df = _apply_column_filter(df, col, mode, v)
                elif mode == "equals_any":
                    # Limit option set to keep the UI responsive on high-cardinality columns.
                    options = _sorted_unique(s.dropna().astype("string").unique().tolist())[:300]
                    v = st.multiselect(
                        f"{pretty_label(col)} equals any", options, default=[], key=f"adv_equals_{col}"
                    )
                    df = _apply_column_filter(df, col, mode, v)
                elif mode == "range":
                    sn = pd.to_numeric(s, errors="coerce").dropna()
                    if len(sn) == 0:
                        st.write("No numeric values available for range filtering.")
                    else:
                        lo0 = float(sn.min())
                        hi0 = float(sn.max())
                        lo, hi = st.slider(
                            f"{pretty_label(col)} range",
                            min_value=lo0,
                            max_value=hi0,
                            value=(lo0, hi0),
                            key=f"adv_range_{col}",
                        )
                        df = _apply_column_filter(df, col, mode, (lo, hi))

    st.caption(f"Showing **{len(df):,}** projects after Explorer-only filters.")

    cols = [
        "grant_recipient_project_title",
        "usd_disbursements_defl",
        "sector_description",
        "organization_name",
        "country",
        "region_macro",
        "year",
        "type_of_flow",
        "channel_name",
        "expected_duration",
    ]
    cols = [c for c in cols if c in df.columns]

    st.dataframe(
        df[cols].sort_values("usd_disbursements_defl", ascending=False).head(500),
        use_container_width=True,
        hide_index=True,
        column_config={
            "grant_recipient_project_title": st.column_config.TextColumn(pretty_label("grant_recipient_project_title")),
            "sector_description": st.column_config.TextColumn(pretty_label("sector_description")),
            "organization_name": st.column_config.TextColumn(pretty_label("organization_name")),
            "country": st.column_config.TextColumn(pretty_label("country")),
            "region_macro": st.column_config.TextColumn(pretty_label("region_macro")),
            "year": st.column_config.TextColumn(pretty_label("year")),
            "type_of_flow": st.column_config.TextColumn(pretty_label("type_of_flow")),
            "channel_name": st.column_config.TextColumn(pretty_label("channel_name")),
            "expected_duration": st.column_config.TextColumn(pretty_label("expected_duration")),
            "usd_disbursements_defl": st.column_config.NumberColumn(
                pretty_label("usd_disbursements_defl"), format="$%0.3f"
            ),
        },
    )

    with st.expander("Show a random project detail"):
        if len(df) == 0:
            st.write("No rows match the current filters.")
        else:
            r = df.sample(1).iloc[0]
            st.markdown(f"**Project**: {r.get('grant_recipient_project_title', '')}")
            st.markdown(f"**Donor**: {r.get('organization_name', '')}")
            st.markdown(f"**Recipient country**: {r.get('country', '')} ({r.get('region_macro', '')})")
            st.markdown(f"**Year**: {r.get('year', '')} • **Flow**: {r.get('type_of_flow', '')}")
            st.markdown(f"**Disbursed**: {_money(float(r.get('usd_disbursements_defl', 0.0)))}")
            st.markdown(f"**Sector(s)**: {r.get('sector_description', '')}")
            st.markdown("**Description**")
            st.write(r.get("project_description", ""))


def main():
    st.set_page_config(
        page_title="Global Philanthropy Dashboard",
        layout="wide",
    )
    st.title("Global Philanthropy Dashboard")
    with st.expander("What am I looking at? (data + counting rules)", expanded=True):
        st.write(
            "**Two levels of data are used in this app:**\n"
            f"- **Projects ({pretty_label('row_id')})**: used for totals/leaderboards/maps (avoids double counting).\n"
            "- **Transactions (sector-split rows)**: used for sector breakdowns (a project can appear in multiple sectors).\n\n"
            "**In the donor-by-sector comparison**, multi-sector projects are intentionally counted in **each** sector."
        )

    data = load_data()
    f = build_filters(data)

    tx = apply_filters(data.transactions, f)
    pr = apply_filters(data.projects, f)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Projects ({pretty_label('row_id')})", f"{len(pr):,}")
    k2.metric("Transactions (rows)", f"{len(tx):,}")
    k3.metric("Total disbursed", _money(float(pr["usd_disbursements_defl"].sum())))
    k4.metric("Recipient countries", f"{pr['country'].nunique(dropna=True):,}" if "country" in pr.columns else "—")

    with st.expander("Try a few common questions", expanded=False):
        st.write(
            "These are examples of how to use the filters/tabs (no special logic — just guidance):\n"
            "- **Top donors out of a country**: set **Donor country**, go to **Donors**.\n"
            "- **Top recipient countries for a theme**: use **Sector** + optional **Marker field**, go to **Global overview**.\n"
            "- **Trends over time**: pick a sector in **Sector deep-dive** and watch the line chart.\n"
            "- **Find specific projects**: use **Explorer** search + Advanced filters."
        )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Global overview", "Sector deep-dive", "Donors", "Explorer"]
    )
    with tab1:
        view_global_overview(pr)
    with tab2:
        view_sector_deep_dive(tx)
    with tab3:
        view_donor_leaderboard(pr)
    with tab4:
        view_recipient_explorer(pr)


if __name__ == "__main__":
    main()

