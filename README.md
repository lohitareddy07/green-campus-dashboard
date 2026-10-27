# Green Campus Sustainability Dashboard

**Domain:** Resource management  **Tech:** Web analytics, data science
**Stack:** Python, Streamlit, Pandas, scikit-learn, Plotly

## 1. Problem statement

Campuses consume large amounts of electricity, water and generate tonnes of waste, but
administrators rarely see this data in one place. Waste such as ACs left running overnight
or a leaking hostel tank goes unnoticed for days. This project builds a web dashboard that
collects building-level resource data, turns it into clear KPIs and a carbon footprint,
detects wasteful events automatically, forecasts future demand, and ranks buildings by
sustainability to drive behaviour change.

## 2. Objectives

1. Monitor electricity, solar generation, water and waste for every campus building.
2. Convert usage into a carbon footprint (t CO₂e) so impact is easy to understand.
3. Detect anomalies (night-time energy spikes, water leaks) using machine learning and rules.
4. Forecast the next 7 days of campus electricity demand.
5. Score and rank buildings (0–100) to support inter-hostel / inter-department competitions.
6. Generate actionable recommendations for the estates and sustainability teams.

## 3. System architecture

```
 Smart meters / IoT sensors  ──►  CSV / database  ──►  analytics.py  ──►  app.py (Streamlit)
 (simulated by generate_data.py)                       KPIs, ML models      Interactive dashboard
```

| Layer | File | Responsibility |
|---|---|---|
| Data | `data/generate_data.py` | Simulates 6 months of hourly readings for 8 buildings, with injected anomalies |
| Analytics | `src/analytics.py` | KPIs, carbon accounting, anomaly detection, forecasting, scoring, recommendations |
| Presentation | `app.py` | Filters, 7 tabs of charts and tables, CSV export |

## 4. Modules

**Overview.** Six KPIs (electricity, solar share, carbon, water, waste, recycling rate), grid vs
solar daily chart, carbon breakdown donut, and a tree-equivalent offset figure.

**Energy.** Energy intensity (kWh/m²) per building, an hour × weekday heatmap showing when power
is used, and a temperature vs consumption regression showing cooling load.

**Water.** Daily use by building and litres per person per day against the 135 L urban benchmark.

**Waste.** Composition (wet / dry / e-waste) and weekly recycling-rate trend. The simulated data
includes an awareness drive mid-semester, which the trend makes visible.

**Alerts (anomaly detection).** Two methods combined, per building:
- *Isolation Forest* (unsupervised ML) on hour, weekday, occupancy, temperature, energy and water.
- *Rules*: night load (00:00–05:00) above 2× the building's median night load; water flow far above
  the median for the same hour and day type.
Consecutive flagged hours are grouped into incidents with a severity level. On the sample data the
system recovers every injected event (6 lab AC nights, 3 admin "lights on" days, a 2-day hostel leak).

**Forecast.** Random Forest regressor on daily campus energy using day-of-week, weekend flag, lag-1,
lag-7, 7-day rolling mean and temperature. Back-tested on the last 21 days (MAPE ≈ 2% on sample
data), then forecasts 7 days ahead recursively. Feature importances are shown.

**Leaderboard.** Weighted score from energy intensity (35%), water per capita (25%), recycling rate
(25%) and solar share (15%), min-max normalised across buildings, with A–D grades and rule-based
recommendations. Downloadable as CSV.

## 5. How to run

```bash
pip install -r requirements.txt
python data/generate_data.py      # creates the CSVs in data/
streamlit run app.py              # opens http://localhost:8501
```

Smoke test (no browser needed): `python test_app.py`

## 6. Using real data

Replace the three CSVs with exports from your campus meters, keeping these columns:

- `buildings.csv`: building, type, area_sqm, capacity, base_kw, base_water_lph, solar_kwp
- `hourly_usage.csv`: timestamp, building, occupancy, temperature_c, energy_kwh, solar_kwh, water_litres
- `daily_waste.csv`: date, building, wet_kg, dry_kg, ewaste_kg, recycled_fraction

Emission factors are constants at the top of `src/analytics.py`. They are approximate; update the
grid factor to the latest Central Electricity Authority (CEA) CO₂ baseline figure.

## 7. Future scope

- Live data from IoT meters (ESP32 + MQTT) into a time-series database such as InfluxDB.
- Role-based login (admin, warden, student) and email/Telegram alerts for high-severity incidents.
- LSTM / Prophet models for hourly forecasting; solar generation forecasting from weather APIs.
- A student-facing gamified page with hostel challenges and badges.
- Deploy on Streamlit Community Cloud or a campus server with Docker.

## 8. Viva questions you should be ready for

- Why Isolation Forest instead of a supervised classifier? (No labelled anomaly data exists.)
- Why add rules on top of ML? (Explainable, catches known patterns reliably, fewer false alarms.)
- Why lag-7 dominates the forecast? (Strong weekly cycle: weekdays vs weekends.)
- How is the sustainability score relative, and what are its limits? (Min-max across buildings;
  buildings of different types are compared, so a per-type benchmark would be fairer.)
