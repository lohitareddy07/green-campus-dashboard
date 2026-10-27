"""
analytics.py
------------
All data-science logic for the Green Campus dashboard, kept separate from the
UI so it can be unit-tested and reused in notebooks.

Modules:
  1. Data loading
  2. KPIs and carbon footprint
  3. Anomaly detection   (Isolation Forest + night-time baseline rule)
  4. Energy forecasting  (Random Forest on lag/calendar/weather features)
  5. Sustainability score and building leaderboard
  6. Rule-based recommendations
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Emission factors (approximate; edit to match your campus / latest CEA data)
GRID_EF_KG_PER_KWH = 0.71      # Indian grid CO2 baseline, kg CO2 per kWh (approx.)
WATER_EF_KG_PER_KL = 0.30      # pumping + treatment, kg CO2e per kilolitre (approx.)
LANDFILL_EF_KG_PER_KG = 0.58   # mixed waste sent to landfill, kg CO2e per kg (approx.)


# ---------------------------------------------------------------- 1. Loading
def load_data(data_dir: Path = DATA_DIR):
    buildings = pd.read_csv(data_dir / "buildings.csv")
    usage = pd.read_csv(data_dir / "hourly_usage.csv", parse_dates=["timestamp"])
    waste = pd.read_csv(data_dir / "daily_waste.csv", parse_dates=["date"])
    usage["date"] = usage["timestamp"].dt.normalize()
    usage["hour"] = usage["timestamp"].dt.hour
    usage["net_grid_kwh"] = (usage["energy_kwh"] - usage["solar_kwh"]).clip(lower=0)
    waste["total_kg"] = waste[["wet_kg", "dry_kg", "ewaste_kg"]].sum(axis=1)
    waste["recycled_kg"] = waste["total_kg"] * waste["recycled_fraction"]
    waste["landfill_kg"] = waste["total_kg"] - waste["recycled_kg"]
    return buildings, usage, waste


# ------------------------------------------------------ 2. KPIs and carbon
def compute_kpis(usage: pd.DataFrame, waste: pd.DataFrame) -> dict:
    energy = usage["energy_kwh"].sum()
    solar = usage["solar_kwh"].sum()
    grid = usage["net_grid_kwh"].sum()
    water_kl = usage["water_litres"].sum() / 1000
    waste_total = waste["total_kg"].sum()
    recycled = waste["recycled_kg"].sum()
    co2 = carbon_footprint(usage, waste)
    return {
        "energy_kwh": energy,
        "solar_kwh": solar,
        "solar_share": solar / energy if energy else 0,
        "grid_kwh": grid,
        "water_kl": water_kl,
        "waste_kg": waste_total,
        "recycling_rate": recycled / waste_total if waste_total else 0,
        "co2_tonnes": sum(co2.values()) / 1000,
        "co2_breakdown_kg": co2,
    }


def carbon_footprint(usage: pd.DataFrame, waste: pd.DataFrame) -> dict:
    return {
        "Electricity (grid)": usage["net_grid_kwh"].sum() * GRID_EF_KG_PER_KWH,
        "Water": usage["water_litres"].sum() / 1000 * WATER_EF_KG_PER_KL,
        "Landfill waste": waste["landfill_kg"].sum() * LANDFILL_EF_KG_PER_KG,
    }


def daily_summary(usage: pd.DataFrame) -> pd.DataFrame:
    return (usage.groupby("date")
            .agg(energy_kwh=("energy_kwh", "sum"), solar_kwh=("solar_kwh", "sum"),
                 water_litres=("water_litres", "sum"), temperature_c=("temperature_c", "mean"))
            .reset_index())


# ------------------------------------------------- 3. Anomaly detection
def detect_anomalies(usage: pd.DataFrame, contamination: float = 0.002) -> pd.DataFrame:
    """
    Two complementary detectors, run per building:
      * Isolation Forest on (hour, weekday, occupancy, temperature, energy, water)
        catches readings that don't fit the building's normal pattern.
      * Night-time rule: energy between 00:00-05:00 more than 2x that
        building's median night load (typical sign of equipment left on).
    Returns flagged hourly rows with a human-readable reason.
    """
    flagged = []
    for building, g in usage.groupby("building"):
        g = g.copy()
        X = pd.DataFrame({
            "hour": g["hour"],
            "weekday": g["timestamp"].dt.dayofweek,
            "occupancy": g["occupancy"],
            "temp": g["temperature_c"],
            "energy": g["energy_kwh"],
            "water": g["water_litres"],
        })
        model = IsolationForest(n_estimators=150, contamination=contamination, random_state=0)
        g["iso_flag"] = model.fit_predict(X) == -1
        g["iso_score"] = -model.score_samples(X)

        night = g["hour"].between(0, 5)
        night_median = g.loc[night, "energy_kwh"].median()
        g["night_flag"] = night & (g["energy_kwh"] > 2 * night_median)

        # Compare water with the same hour on the same kind of day (weekday/weekend)
        day_type = g["timestamp"].dt.dayofweek >= 5
        water_median = g.groupby([g["hour"], day_type])["water_litres"].transform("median")
        excess = g["water_litres"] - water_median
        g["water_flag"] = excess > np.maximum(1000, 0.5 * water_median)

        mask = g["iso_flag"] | g["night_flag"] | g["water_flag"]
        if not mask.any():
            continue
        out = g.loc[mask].copy()
        reasons = []
        for _, r in out.iterrows():
            parts = []
            if r["night_flag"]:
                parts.append(f"Night load {r['energy_kwh'] / night_median:.1f}x normal")
            if r["water_flag"]:
                parts.append("Unusual water flow (possible leak)")
            if r["iso_flag"] and not parts:
                parts.append("Unusual usage pattern")
            reasons.append("; ".join(parts))
        out["reason"] = reasons
        flagged.append(out)

    if not flagged:
        return pd.DataFrame()
    cols = ["timestamp", "building", "energy_kwh", "water_litres", "occupancy", "iso_score", "reason"]
    return pd.concat(flagged)[cols].sort_values("timestamp").reset_index(drop=True)


def group_incidents(anomalies: pd.DataFrame) -> pd.DataFrame:
    """Collapse consecutive flagged hours into incidents for a readable table."""
    if anomalies.empty:
        return anomalies
    a = anomalies.sort_values(["building", "timestamp"]).copy()
    gap = a.groupby("building")["timestamp"].diff() > pd.Timedelta(hours=2)
    a["incident"] = gap.groupby(a["building"]).cumsum()
    inc = (a.groupby(["building", "incident"])
           .agg(start=("timestamp", "min"), end=("timestamp", "max"), hours=("timestamp", "count"),
                energy_kwh=("energy_kwh", "sum"), water_litres=("water_litres", "sum"),
                reason=("reason", lambda s: s.mode().iat[0]))
           .reset_index().drop(columns="incident"))
    rule_based = ~inc["reason"].str.startswith("Unusual usage")
    inc["severity"] = np.select([rule_based & (inc["hours"] >= 3), rule_based], ["High", "Medium"], "Low")
    return inc.sort_values("start", ascending=False).reset_index(drop=True)


# --------------------------------------------------- 4. Forecasting
def _make_features(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["dow"] = d["date"].dt.dayofweek
    d["is_weekend"] = (d["dow"] >= 5).astype(int)
    d["lag1"] = d["energy_kwh"].shift(1)
    d["lag7"] = d["energy_kwh"].shift(7)
    d["roll7"] = d["energy_kwh"].shift(1).rolling(7).mean()
    return d


FEATURES = ["dow", "is_weekend", "lag1", "lag7", "roll7", "temperature_c"]


def forecast_energy(usage: pd.DataFrame, horizon: int = 7, test_days: int = 21):
    """
    Train a Random Forest on daily campus energy, evaluate on the last
    `test_days`, then forecast `horizon` days ahead recursively.
    Returns (history_with_predictions, future_forecast, metrics).
    """
    daily = daily_summary(usage)[["date", "energy_kwh", "temperature_c"]]
    feats = _make_features(daily).dropna().reset_index(drop=True)
    train, test = feats.iloc[:-test_days], feats.iloc[-test_days:]

    model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=0)
    model.fit(train[FEATURES], train["energy_kwh"])
    test = test.copy()
    test["predicted"] = model.predict(test[FEATURES])
    metrics = {
        "MAE (kWh)": mean_absolute_error(test["energy_kwh"], test["predicted"]),
        "MAPE (%)": mean_absolute_percentage_error(test["energy_kwh"], test["predicted"]) * 100,
    }

    # Retrain on everything, then forecast forward one day at a time
    model.fit(feats[FEATURES], feats["energy_kwh"])
    hist = daily.copy()
    recent_temp = daily["temperature_c"].tail(7).mean()  # simple weather assumption
    future = []
    for _ in range(horizon):
        next_date = hist["date"].max() + pd.Timedelta(days=1)
        hist = pd.concat([hist, pd.DataFrame({"date": [next_date], "energy_kwh": [np.nan],
                                              "temperature_c": [recent_temp]})], ignore_index=True)
        row = _make_features(hist).iloc[[-1]]
        pred = float(model.predict(row[FEATURES])[0])
        hist.loc[hist.index[-1], "energy_kwh"] = pred
        future.append({"date": next_date, "forecast_kwh": pred})

    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return test[["date", "energy_kwh", "predicted"]], pd.DataFrame(future), metrics, importance


# ------------------------------------------- 5. Sustainability score
def building_scores(buildings: pd.DataFrame, usage: pd.DataFrame, waste: pd.DataFrame) -> pd.DataFrame:
    """
    Score each building 0-100 (higher is greener) from four normalised metrics:
      energy intensity (kWh/m²)  - lower is better   weight 0.35
      water per capita (L/person) - lower is better  weight 0.25
      recycling rate               - higher is better weight 0.25
      solar share                  - higher is better weight 0.15
    Buildings are compared against others of the same campus, so the score is
    relative, which suits a leaderboard / inter-hostel competition.
    """
    u = usage.groupby("building").agg(energy=("energy_kwh", "sum"), solar=("solar_kwh", "sum"),
                                      water=("water_litres", "sum")).reset_index()
    w = waste.groupby("building").agg(total=("total_kg", "sum"), recycled=("recycled_kg", "sum")).reset_index()
    df = buildings.merge(u, on="building").merge(w, on="building")
    days = usage["date"].nunique()
    df["kwh_per_sqm"] = df["energy"] / df["area_sqm"]
    df["water_per_capita_day"] = df["water"] / df["capacity"] / days
    df["recycling_rate"] = df["recycled"] / df["total"]
    df["solar_share"] = df["solar"] / df["energy"]

    def norm(s, higher_better):
        rng = s.max() - s.min()
        x = (s - s.min()) / rng if rng else s * 0 + 1
        return x if higher_better else 1 - x

    df["score"] = 100 * (0.35 * norm(df["kwh_per_sqm"], False)
                         + 0.25 * norm(df["water_per_capita_day"], False)
                         + 0.25 * norm(df["recycling_rate"], True)
                         + 0.15 * norm(df["solar_share"], True))
    df["score"] = df["score"].round(1)
    df["grade"] = pd.cut(df["score"], [-1, 40, 60, 80, 101], labels=["D", "C", "B", "A"])
    cols = ["building", "type", "score", "grade", "kwh_per_sqm", "water_per_capita_day",
            "recycling_rate", "solar_share"]
    return df[cols].sort_values("score", ascending=False).reset_index(drop=True)


# -------------------------------------------- 6. Recommendations
def recommendations(scores: pd.DataFrame, incidents: pd.DataFrame, kpis: dict) -> list[str]:
    tips = []
    if not incidents.empty:
        night = incidents[incidents["reason"].str.contains("Night")]
        for b, n in night["building"].value_counts().items():
            tips.append(f"{b}: {n} night-time energy spikes. Schedule auto-shutoff for ACs, lights and PCs after 10 pm.")
        leaks = incidents[incidents["reason"].str.contains("water")]
        for b in leaks["building"].unique():
            tips.append(f"{b}: sustained abnormal water flow detected. Inspect tanks, valves and washroom fittings.")
    if kpis["solar_share"] < 0.2:
        tips.append(f"Solar covers only {kpis['solar_share']:.0%} of demand. Rooftop space on high-scoring buildings is the cheapest next step.")
    worst = scores.iloc[-1]
    tips.append(f"{worst['building']} has the lowest sustainability score ({worst['score']}). Start an energy audit there.")
    low_recycle = scores[scores["recycling_rate"] < 0.45]["building"].tolist()
    if low_recycle:
        tips.append("Segregation bins and awareness drives for: " + ", ".join(low_recycle) + ".")
    return tips
