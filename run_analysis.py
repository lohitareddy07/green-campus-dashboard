"""Run the analytics pipeline from the command line and print a summary."""
from src.analytics import (building_scores, compute_kpis, detect_anomalies, forecast_energy,
                           group_incidents, load_data)

b, u, w = load_data()
k = compute_kpis(u, w)
print(f"Energy: {k['energy_kwh']/1000:,.0f} MWh | Solar share: {k['solar_share']:.1%} | CO2: {k['co2_tonnes']:,.0f} t")
inc = group_incidents(detect_anomalies(u))
print(f"High-severity incidents found: {(inc.severity == 'High').sum()}")
t, f, m, imp = forecast_energy(u)
print(f"Forecast MAE: {m['MAE (kWh)']:.0f} kWh/day | MAPE: {m['MAPE (%)']:.2f}%")
print()
print(building_scores(b, u, w)[["building", "score", "grade"]].to_string(index=False))
