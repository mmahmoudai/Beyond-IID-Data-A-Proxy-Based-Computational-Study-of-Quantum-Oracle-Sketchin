# Reproducing the Paper

This directory regenerates the figures and measured tables in
"Proxy-Based Diagnostics of Quantum Oracle Sketching Robustness for
Non-IID Sensor and Telemetry Streams" (MDPI Sensors variant; the same
pipeline also backs the sibling venue variants under `paper/`).

## 1. Environment

```
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r code/requirements.txt
```

Tested with Python 3.10.19, NumPy 1.24, SciPy 1.10, scikit-learn 1.3,
Matplotlib 3.7, Seaborn 0.13, Pandas 2.0.

## 2. Datasets

### Synthetic streams
No download. Generated on demand from `code/data_generators.py` with
explicit seeds (see section 5).

### Real streams
Two series from the Numenta Anomaly Benchmark (MIT licence):

```
mkdir -p data/raw
curl -L -o data/raw/nyc_taxi.csv \
  https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv
curl -L -o data/raw/nab_machine_temperature.csv \
  https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv
```

NYC Taxi: 10,320 half-hour bins covering 2014-07-01 to 2015-01-31.
Machine Temperature: 22,695 five-minute readings preceding a documented
system failure.

## 3. Run all experiments

```
# Figures 1-7 + the Markov-sweep data behind the effect-size table
python code/run_experiments.py

# Figure 8 + real-stream tables (Experiment 8)
python code/real_streams.py

# Figure 9 + tau-estimator/window/length/target ablation table (Experiment 9)
python code/ablations.py

# Effect sizes, exact signed-rank CIs, Holm adjustment, and the C-delta
# crossover grid (deterministic re-analysis of the Markov sweep)
python code/exp5_effectsizes.py
```

Total runtime: roughly 15-25 minutes on a standard laptop.

## 4. Output locations

- `figures/fig*.{pdf,png}` — all paper figures
- `figures/exp5_wilcoxon.txt` — Markov-sweep summary used by the effect-size script
- `results/real_stream_summary.json` — real-stream numerical summary (both datasets, incl. balanced-accuracy/F1 intervals and threshold sensitivity)
- `results/ablations.json` — full ablation table dump
- `results/ablations_table.txt` — LaTeX-ready table rows

## 5. Random seeds

Every stochastic experiment uses 10 seeds and reports mean plus a
1.96×SE half-width (normal approximation): `run_experiments.py`
(Experiments 1–7) uses seed indices 42–51; `ablations.py` and
`real_streams.py` (Experiments 8–9) use indices 0–9. Paired-difference
inference in the effect-size script uses exact signed-rank (Walsh
average) intervals. Non-stochastic landscapes use no seed.

The encoder/resolution sensitivity check for the real streams
(deterministic re-encoding of the same raw series) is
`paper/mdpi_sensors/revision/encoder_sensitivity.py`.
