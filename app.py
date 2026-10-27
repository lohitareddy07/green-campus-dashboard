"""
Green Campus Sustainability Dashboard
Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import (building_scores, carbon_footprint, compute_kpis, daily_summary,
                           detect_anomalies, forecast_energy, group_incidents, load_data,
                           recommendations)

st.set_page_config(page_title="Green Campus Dashboard", page_icon="🌿", layout="wide")

ENERGY, SOLAR, WATER, WASTE, LEAF = "#E0A33B", "#F2D16B", "#3C7FB0", "#8C7B6B", "#4C9A5B"
PLOT = dict(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10), height=360)


@st.cache_data
def get_data():
    return load_data()


@st.cache_data
def get_anomalies(usage):
    return group_incidents(detect_anomalies(usage))


@st.cache_data
def get_forecast(usage):
    return forecast_energy(usage)


buildings, usage_all, waste_all = get_data()

# ---------------------------------------------------------------- Sidebar
st.sidebar.title("🌿 Green Campus")
st.sidebar.caption("Energy, water and waste across campus buildings")
min_d, max_d = usage_all["date"].min().date(), usage_all["date"].max().date()
date_range = st.sidebar.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
selected = st.sidebar.multiselect("Buildings", buildings["building"].tolist(),
                                  default=buildings["building"].tolist())
if len(date_range) != 2 or not selected:
    st.info("Pick a start date, an end date and at least one building to see the dashboard.")
    st.stop()

start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
usage = usage_all[usage_all["date"].between(start, end) & usage_all["building"].isin(selected)]
waste = waste_all[waste_all["date"].between(start, end) & waste_all["building"].isin(selected)]
kpis = compute_kpis(usage, waste)

st.title("Green Campus Sustainability Dashboard")
st.caption(f"{start:%d %b %Y} to {end:%d %b %Y} · {len(selected)} buildings")

tab_overview, tab_energy, tab_water, tab_waste, tab_alerts, tab_forecast, tab_board = st.tabs(
    ["Overview", "Energy", "Water", "Waste", "Alerts", "Forecast", "Leaderboard"])

# ---------------------------------------------------------------- Overview
with tab_overview:
    c = st.columns(3) + st.columns(3)
    c[0].metric("Electricity used (MWh)", f"{kpis['energy_kwh'] / 1000:,.0f}")
    c[1].metric("Solar share", f"{kpis['solar_share']:.1%}")
    c[2].metric("Carbon footprint (t CO₂e)", f"{kpis['co2_tonnes']:,.0f}")
    c[3].metric("Water used (kL)", f"{kpis['water_kl']:,.0f}")
    c[4].metric("Waste generated (t)", f"{kpis['waste_kg'] / 1000:,.1f}")
    c[5].metric("Recycling rate", f"{kpis['recycling_rate']:.0%}")

    left, right = st.columns([2, 1])
    daily = daily_summary(usage)
    fig = go.Figure()
    fig.add_bar(x=daily["date"], y=daily["energy_kwh"] - daily["solar_kwh"], name="From grid", marker_color=ENERGY)
    fig.add_bar(x=daily["date"], y=daily["solar_kwh"], name="From solar", marker_color=SOLAR)
    fig.update_layout(barmode="stack", title="Daily electricity: grid vs solar (kWh)", **PLOT)
    left.plotly_chart(fig, width="stretch")

    co2 = pd.DataFrame(carbon_footprint(usage, waste).items(), columns=["source", "kg"])
    fig = px.pie(co2, names="source", values="kg", hole=0.55, title="Where the carbon comes from",
                 color="source", color_discrete_map={"Electricity (grid)": ENERGY, "Water": WATER,
                                                     "Landfill waste": WASTE})
    fig.update_layout(**PLOT)
    right.plotly_chart(fig, width="stretch")

    trees = kpis["co2_tonnes"] * 1000 / 21  # a mature tree absorbs ~21 kg CO2 per year
    st.info(f"Offsetting this period's footprint would take roughly **{trees:,.0f} mature trees** growing for a year.")

# ---------------------------------------------------------------- Energy
with tab_energy:
    left, right = st.columns(2)
    by_b = usage.groupby("building")[["energy_kwh", "solar_kwh"]].sum().reset_index()
    by_b = by_b.merge(buildings[["building", "area_sqm"]], on="building")
    by_b["kWh per m²"] = by_b["energy_kwh"] / by_b["area_sqm"]
    fig = px.bar(by_b.sort_values("kWh per m²"), x="kWh per m²", y="building", orientation="h",
                 title="Energy intensity by building", color_discrete_sequence=[ENERGY])
    fig.update_layout(**PLOT)
    left.plotly_chart(fig, width="stretch")

    heat = usage.assign(weekday=usage["timestamp"].dt.day_name().str[:3]) \
        .groupby(["weekday", "hour"])["energy_kwh"].mean().reset_index()
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot = heat.pivot(index="weekday", columns="hour", values="energy_kwh").reindex(order)
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="YlOrBr",
                    labels=dict(color="avg kWh"), title="When campus uses power (hour × weekday)")
    fig.update_layout(**PLOT)
    right.plotly_chart(fig, width="stretch")

    temp_daily = daily_summary(usage)
    fig = px.scatter(temp_daily, x="temperature_c", y="energy_kwh",
                     title="Hotter days, more cooling load", color_discrete_sequence=[ENERGY],
                     labels={"temperature_c": "Average temperature (°C)", "energy_kwh": "Daily kWh"})
    slope, intercept = np.polyfit(temp_daily["temperature_c"], temp_daily["energy_kwh"], 1)
    xs = np.linspace(temp_daily["temperature_c"].min(), temp_daily["temperature_c"].max(), 20)
    fig.add_scatter(x=xs, y=slope * xs + intercept, mode="lines", name=f"+{slope:,.0f} kWh per °C",
                    line_color="#555")
    fig.update_layout(**PLOT)
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- Water
with tab_water:
    left, right = st.columns(2)
    wd = usage.groupby(["date", "building"])["water_litres"].sum().reset_index()
    fig = px.area(wd, x="date", y="water_litres", color="building", title="Daily water use by building (litres)")
    fig.update_layout(**PLOT)
    left.plotly_chart(fig, width="stretch")

    days = usage["date"].nunique()
    pc = usage.groupby("building")["water_litres"].sum().reset_index().merge(buildings, on="building")
    pc["Litres per person per day"] = pc["water_litres"] / pc["capacity"] / days
    fig = px.bar(pc.sort_values("Litres per person per day"), x="Litres per person per day", y="building",
                 orientation="h", title="Water per person", color_discrete_sequence=[WATER])
    fig.add_vline(x=135, line_dash="dash", annotation_text="135 L urban benchmark")
    fig.update_layout(**PLOT)
    right.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- Waste
with tab_waste:
    left, right = st.columns(2)
    comp = waste[["wet_kg", "dry_kg", "ewaste_kg"]].sum().rename({"wet_kg": "Wet / organic",
                                                                  "dry_kg": "Dry", "ewaste_kg": "E-waste"})
    fig = px.pie(names=comp.index, values=comp.values, hole=0.5, title="Waste composition",
                 color_discrete_sequence=[LEAF, WASTE, "#5E5A80"])
    fig.update_layout(**PLOT)
    left.plotly_chart(fig, width="stretch")

    weekly = waste.set_index("date").resample("W")[["recycled_kg", "total_kg"]].sum()
    weekly["Recycling rate"] = weekly["recycled_kg"] / weekly["total_kg"]
    fig = px.line(weekly.reset_index(), x="date", y="Recycling rate", markers=True,
                  title="Weekly recycling rate", color_discrete_sequence=[LEAF])
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(**PLOT)
    right.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- Alerts
with tab_alerts:
    incidents = get_anomalies(usage)
    if incidents.empty:
        st.success("No unusual usage found for this selection.")
    else:
        show_low = st.toggle("Include low-severity pattern alerts", value=False)
        view = incidents if show_low else incidents[incidents["severity"] != "Low"]
        h = (view["severity"] == "High").sum()
        st.subheader(f"{len(view)} incidents ({h} high severity)")
        st.dataframe(view, width="stretch", hide_index=True,
                     column_config={"energy_kwh": st.column_config.NumberColumn("Energy (kWh)", format="%.0f"),
                                    "water_litres": st.column_config.NumberColumn("Water (L)", format="%.0f")})

        col1, col2 = st.columns([2, 1])
        b = col1.selectbox("Inspect a building", sorted(view["building"].unique()))
        metric = col2.radio("Show", ["Energy", "Water"], horizontal=True)
        y, colour = ("energy_kwh", ENERGY) if metric == "Energy" else ("water_litres", WATER)
        series = usage[usage["building"] == b]
        flagged = view[view["building"] == b]
        fig = px.line(series, x="timestamp", y=y, title=f"{b}: hourly {metric.lower()} with incidents shaded",
                      color_discrete_sequence=[colour])
        for _, r in flagged.iterrows():
            fig.add_vrect(x0=r["start"], x1=r["end"] + pd.Timedelta(hours=1), fillcolor="red", opacity=0.2, line_width=0)
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- Forecast
with tab_forecast:
    if usage["date"].nunique() < 45:
        st.warning("Select at least 45 days to train the forecasting model.")
    else:
        test, future, metrics, importance = get_forecast(usage)
        c = st.columns(3)
        c[0].metric("Next 7 days (forecast)", f"{future['forecast_kwh'].sum() / 1000:,.1f} MWh")
        c[1].metric("Model error (MAE)", f"{metrics['MAE (kWh)']:,.0f} kWh/day")
        c[2].metric("Model error (MAPE)", f"{metrics['MAPE (%)']:.1f}%")

        hist = daily_summary(usage)
        fig = go.Figure()
        fig.add_scatter(x=hist["date"], y=hist["energy_kwh"], name="Actual", line_color=ENERGY)
        fig.add_scatter(x=test["date"], y=test["predicted"], name="Predicted (test period)", line_dash="dot",
                        line_color="#555")
        fig.add_scatter(x=future["date"], y=future["forecast_kwh"], name="Forecast", line_color=LEAF,
                        line_width=3)
        fig.update_layout(title="Daily campus electricity: actual, back-test and 7-day forecast (kWh)", **PLOT)
        st.plotly_chart(fig, width="stretch")

        imp = importance.rename("importance").rename_axis("feature").reset_index()
        fig = px.bar(imp, x="importance", y="feature", orientation="h", title="What drives the forecast",
                     color_discrete_sequence=[LEAF])
        fig.update_layout(**PLOT)
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- Leaderboard
with tab_board:
    scores = building_scores(buildings[buildings["building"].isin(selected)], usage, waste)
    fig = px.bar(scores, x="score", y="building", orientation="h", color="grade", text="score",
                 title="Sustainability score (0–100, higher is greener)",
                 color_discrete_map={"A": LEAF, "B": "#9BC27C", "C": ENERGY, "D": "#C8553D"},
                 category_orders={"building": scores["building"].tolist(), "grade": ["A", "B", "C", "D"]})
    fig.update_layout(**PLOT)
    st.plotly_chart(fig, width="stretch")
    table = scores.assign(recycling_rate=scores["recycling_rate"] * 100, solar_share=scores["solar_share"] * 100)
    st.dataframe(table, width="stretch", hide_index=True,
                 column_config={"kwh_per_sqm": st.column_config.NumberColumn("kWh per m²", format="%.1f"),
                                "water_per_capita_day": st.column_config.NumberColumn("L per person/day", format="%.1f"),
                                "recycling_rate": st.column_config.NumberColumn("Recycling", format="%.0f%%"),
                                "solar_share": st.column_config.NumberColumn("Solar share", format="%.0f%%")})

    st.subheader("Recommended actions")
    for tip in recommendations(scores, get_anomalies(usage), kpis):
        st.markdown(f"- {tip}")

    st.download_button("Download scores as CSV", scores.to_csv(index=False), "building_scores.csv", "text/csv")
