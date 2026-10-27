"""
generate_data.py
----------------
Creates realistic synthetic smart-meter data for a university campus so the
dashboard can run without real IoT hardware.

Outputs (written next to this script):
  buildings.csv     - static info about each building
  hourly_usage.csv  - hourly electricity, solar, water, occupancy, temperature
  daily_waste.csv   - daily waste by category per building

It also injects known anomalies (a lab AC left running at night, a hostel
water leak) so the anomaly-detection module has something real to find.

Run:  python data/generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
OUT = Path(__file__).parent
DAYS = 180
START = pd.Timestamp("2026-04-01")

# name, type, area_sqm, capacity, base_kw, water_lph (litres/hour base), solar_kwp
BUILDINGS = [
    ("Academic Block A", "academic", 6500, 900, 55, 900, 40),
    ("Academic Block B", "academic", 5200, 700, 45, 750, 30),
    ("Central Library", "academic", 3800, 400, 35, 400, 25),
    ("CSE Labs", "lab", 3000, 350, 70, 300, 20),
    ("Boys Hostel", "hostel", 8000, 1200, 60, 4200, 50),
    ("Girls Hostel", "hostel", 6500, 900, 48, 3400, 40),
    ("Cafeteria", "dining", 1800, 600, 40, 2600, 0),
    ("Admin Block", "admin", 2500, 150, 25, 250, 15),
]

# Occupancy profile (0-1) by hour for each building type, weekday vs weekend
def occupancy_profile(btype: str, hour: int, weekend: bool) -> float:
    if btype in ("academic", "lab", "admin"):
        if weekend:
            return 0.08 if 9 <= hour <= 17 else 0.02
        if 9 <= hour <= 17:
            return 0.85 if hour not in (13,) else 0.55
        if 8 <= hour <= 19:
            return 0.35
        return 0.03
    if btype == "hostel":
        if 0 <= hour <= 6:
            return 0.95
        if 9 <= hour <= 16:
            return 0.45 if weekend else 0.20
        return 0.80
    if btype == "dining":
        meals = {7: 0.7, 8: 0.9, 12: 0.8, 13: 1.0, 14: 0.6, 19: 0.8, 20: 1.0, 21: 0.6}
        return meals.get(hour, 0.1 if 6 <= hour <= 22 else 0.0)
    return 0.3


def main() -> None:
    pd.DataFrame(
        BUILDINGS,
        columns=["building", "type", "area_sqm", "capacity", "base_kw", "base_water_lph", "solar_kwp"],
    ).to_csv(OUT / "buildings.csv", index=False)

    hours = pd.date_range(START, periods=DAYS * 24, freq="h")
    day_idx = np.arange(len(hours)) // 24
    # Hyderabad-like temperature: hot April-May, cooler monsoon from mid June
    seasonal = 33 - 6 * np.clip((day_idx - 70) / 60, 0, 1) + 2 * np.sin(day_idx / 180 * np.pi)
    diurnal = 5 * np.sin((hours.hour.values - 9) / 24 * 2 * np.pi)
    temp = seasonal + diurnal + RNG.normal(0, 1.0, len(hours))

    # Daily cloud cover affects solar (more cloud in monsoon)
    cloud_daily = np.clip(RNG.normal(0.25, 0.15, DAYS) + 0.3 * (np.arange(DAYS) > 75), 0, 0.9)
    cloud = cloud_daily[day_idx]
    sun = np.clip(np.sin((hours.hour.values - 6) / 12 * np.pi), 0, None)  # 6am-6pm

    rows = []
    for name, btype, area, cap, base_kw, base_w, kwp in BUILDINGS:
        weekend = hours.dayofweek.values >= 5
        occ = np.array([occupancy_profile(btype, h, w) for h, w in zip(hours.hour.values, weekend)])
        occ = np.clip(occ + RNG.normal(0, 0.04, len(hours)), 0, 1)

        cooling = np.clip((temp - 26) / 10, 0, None)          # AC load above 26°C
        idle = 0.18 if btype != "hostel" else 0.30            # standby / always-on share
        energy = base_kw * (idle + occ * (0.7 + 0.6 * cooling))
        energy *= RNG.normal(1, 0.05, len(hours))

        water = base_w * (0.1 + occ) * RNG.normal(1, 0.08, len(hours))
        solar = kwp * 0.8 * sun * (1 - cloud) * RNG.normal(1, 0.05, len(hours))

        df = pd.DataFrame({
            "timestamp": hours,
            "building": name,
            "occupancy": (occ * cap).round().astype(int),
            "temperature_c": temp.round(1),
            "energy_kwh": energy.clip(0).round(2),
            "solar_kwh": solar.clip(0).round(2),
            "water_litres": water.clip(0).round(1),
        })
        rows.append(df)

    usage = pd.concat(rows, ignore_index=True)

    # --- Inject anomalies -------------------------------------------------
    # 1) CSE Labs AC left on overnight on several nights
    for d in RNG.choice(range(20, DAYS - 5), size=6, replace=False):
        night = (usage.building == "CSE Labs") & (usage.timestamp.dt.normalize() == START + pd.Timedelta(days=int(d))) \
                & (usage.timestamp.dt.hour.isin(range(0, 6)))
        usage.loc[night, "energy_kwh"] *= 4.0
    # 2) Boys Hostel water leak for ~2 days
    leak_start = START + pd.Timedelta(days=118)
    leak = (usage.building == "Boys Hostel") & usage.timestamp.between(leak_start, leak_start + pd.Timedelta(hours=46))
    usage.loc[leak, "water_litres"] += 2500
    # 3) Admin block weekend lights left on
    for d in [41, 90, 132]:
        day = START + pd.Timedelta(days=d)
        mask = (usage.building == "Admin Block") & (usage.timestamp.dt.normalize() == day)
        usage.loc[mask, "energy_kwh"] *= 2.5

    usage.to_csv(OUT / "hourly_usage.csv", index=False)

    # --- Daily waste --------------------------------------------------------
    dates = pd.date_range(START, periods=DAYS, freq="D")
    waste_rows = []
    for name, btype, area, cap, *_ in BUILDINGS:
        wet_rate = {"dining": 0.35, "hostel": 0.08}.get(btype, 0.01)
        dry_rate = {"dining": 0.06, "hostel": 0.05}.get(btype, 0.03)
        ewaste_rate = 0.004 if btype in ("lab", "admin") else 0.0005
        # Recycling improves over the semester (an awareness drive at day 90)
        base = {"dining": 0.22, "hostel": 0.38, "academic": 0.50, "lab": 0.42, "admin": 0.55}[btype]
        base += RNG.normal(0, 0.04)  # each building differs a little
        recycle = np.clip(base + 0.15 * (np.arange(DAYS) > 90) + RNG.normal(0, 0.05, DAYS), 0.05, 0.95)
        weekend = dates.dayofweek >= 5
        factor = np.where(weekend, 0.6 if btype != "hostel" else 1.1, 1.0)
        waste_rows.append(pd.DataFrame({
            "date": dates,
            "building": name,
            "wet_kg": (cap * wet_rate * factor * RNG.normal(1, 0.1, DAYS)).clip(0).round(1),
            "dry_kg": (cap * dry_rate * factor * RNG.normal(1, 0.1, DAYS)).clip(0).round(1),
            "ewaste_kg": (cap * ewaste_rate * RNG.exponential(1, DAYS)).round(2),
            "recycled_fraction": recycle.round(2),
        }))
    pd.concat(waste_rows, ignore_index=True).to_csv(OUT / "daily_waste.csv", index=False)

    print(f"Wrote {len(usage):,} hourly rows and {DAYS * len(BUILDINGS):,} waste rows to {OUT}")


if __name__ == "__main__":
    main()
