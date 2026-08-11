"""Encoding/resolution sensitivity of the real-stream (tau, r) placements.

Deterministic re-analysis: for each (stream, sampling resolution, time-of-day
bucket width) configuration, re-encode the SAME raw NAB series with the
published binariser and recompute tau_emp, r_emp, T_eff and the proxy value
(n = 6, C = 2, delta = 0.05, W = 20). No training, no seeds, no new data.

Question (from the adversarial re-review): do NYC Taxi and Machine Temperature
stay on opposite sides of the proxy-favourable boundary when the sampling-rate/
encoder-field ratio changes, or is the placement an encoder artefact?
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, r"D:\research\quantum_paper\code")
import real_streams as rs
from data_generators import compute_refreshing_time, compute_repetition_number

KAPPA = 4.0 * np.log(1.0 / 0.05)   # C^2 log(1/delta) ~ 11.98
N_BITS = 6
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "encoder_sensitivity.json")

def proxy(T_eff):
    return max(0.5, 1.0 - KAPPA * N_BITS / max(T_eff, 1.0))

def load_raw(name):
    path = os.path.join(rs.RAW_DIR, name)
    df = rs._load_csv(path)
    return df['value'].astype(float).values, pd.to_datetime(df['timestamp'])

def resample(values, ts, minutes):
    s = pd.Series(values, index=pd.DatetimeIndex(ts))
    agg = s.resample(f"{minutes}min").mean().dropna()
    return agg.values, pd.Series(agg.index)

def analyse(values, ts, bucket_hours, roll_window):
    X, y = rs.binarise(values, ts, hours_bucket_period_hours=bucket_hours,
                       roll_window=roll_window)
    T = len(X)
    tau = float(compute_refreshing_time(X))
    r = float(compute_repetition_number(X, window=20))
    T_eff = T / (tau * r)
    return dict(T=T, tau=round(tau, 2), r=round(r, 2),
                T_eff=round(T_eff, 1), proxy=round(proxy(T_eff), 3))

rows = []

nyc_v, nyc_t = load_raw('nyc_taxi.csv')
mt_v, mt_t = load_raw('nab_machine_temperature.csv')

# --- NYC Taxi (native 30-min) ---
rows.append(("NYC Taxi", "30 min (native)", "3 h (published)", analyse(nyc_v, nyc_t, 3, 336)))
rows.append(("NYC Taxi", "30 min (native)", "6 h", analyse(nyc_v, nyc_t, 6, 336)))
rows.append(("NYC Taxi", "30 min (native)", "1.5 h", analyse(nyc_v, nyc_t, 1.5, 336)))
v60, t60 = resample(nyc_v, nyc_t, 60)
rows.append(("NYC Taxi", "60 min (downsampled)", "3 h", analyse(v60, t60, 3, 168)))

# --- Machine Temperature (native 5-min) ---
rows.append(("Machine Temp", "5 min (native)", "3 h (published)", analyse(mt_v, mt_t, 3, 2016)))
rows.append(("Machine Temp", "5 min (native)", "6 h", analyse(mt_v, mt_t, 6, 2016)))
rows.append(("Machine Temp", "5 min (native)", "1.5 h", analyse(mt_v, mt_t, 1.5, 2016)))
v30, t30 = resample(mt_v, mt_t, 30)
rows.append(("Machine Temp", "30 min (downsampled)", "3 h", analyse(v30, t30, 3, 336)))
rows.append(("Machine Temp", "30 min (downsampled)", "6 h", analyse(v30, t30, 6, 336)))
v60m, t60m = resample(mt_v, mt_t, 60)
rows.append(("Machine Temp", "60 min (downsampled)", "3 h", analyse(v60m, t60m, 3, 168)))

print(f"{'stream':13s} {'resolution':22s} {'ToD bucket':16s} {'T':>6s} {'tau':>6s} {'r':>6s} {'T_eff':>8s} {'proxy':>6s}")
for s, res, b, d in rows:
    print(f"{s:13s} {res:22s} {b:16s} {d['T']:>6d} {d['tau']:>6.2f} {d['r']:>6.2f} {d['T_eff']:>8.1f} {d['proxy']:>6.3f}")

json.dump([dict(stream=s, resolution=res, bucket=b, **d) for s, res, b, d in rows],
          open(OUT, "w"), indent=2)
print(f"\nSaved -> {OUT}")
