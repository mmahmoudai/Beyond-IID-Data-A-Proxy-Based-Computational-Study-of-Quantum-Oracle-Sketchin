# Reproducing the Paper

This directory regenerates every figure and table in
"Beyond IID Data: A Proxy-Based Computational Study of Quantum Oracle
Sketching Robustness under Structured Non-IID Streaming."

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
explicit seeds 0-9.

### Real stream
NYC Taxi pickup counts from the Numenta Anomaly Benchmark (MIT licence):

```
mkdir -p data/raw
curl -L -o data/raw/nyc_taxi.csv \
  https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv
```

10,320 half-hour bins covering 2014-07-01 to 2015-01-31.

## 3. Run all experiments

```
# Figures 1-7 + Table 3 (Markov sweep)
python code/run_experiments.py

# Figure 8 (real-stream validation, Section 6.8)
python code/real_streams.py

# Figure 9 + Table 4 (proxy-sensitivity ablations, Section 6.9)
python code/ablations.py
```

Total runtime: roughly 15-20 minutes on a standard laptop.

## 4. Output locations

- `figures/fig*.{pdf,png}` — all paper figures
- `results/real_stream_summary.json` — NYC Taxi numerical summary
- `results/ablations.json` — full ablation table dump
- `results/ablations_table.txt` — LaTeX-ready table rows

## 5. Random seeds

Every stochastic experiment uses seeds 0-9 (10 seeds) and reports mean
plus 95% CI half-width. Non-stochastic landscapes use no seed.

## 6. Determinism

Within a single Python process, results are deterministic given the
seed. Across NumPy/SciPy minor-version changes the third decimal place
of accuracy values may shift; the qualitative ordering of regimes and
the proxy-baseline crossover point are stable.

## 7. Where the proxy formulas live

- `code/quantum_bounds.py` — closed-form heuristic proxies for quantum
  oracle sketching performance
- `code/data_generators.py` — five non-IID stream generators and the
  empirical (tau, r) estimators
- `code/classical_baselines.py` — Online SGD, Averaged SGD, Count-Min
  hash classifier, Frequent Directions
- `code/run_experiments.py` — main figures
- `code/real_streams.py` — real-stream case study (Section 6.8)
- `code/ablations.py` — sensitivity ablations (Section 6.9)
