"""
Real-Stream Validation Case Study (Section 6.8) — TWO datasets.

Loads two real streams from the Numenta Anomaly Benchmark (NAB), converts each
into a streaming binary classification task in {0,1}^n, estimates empirical
refreshing time tau and repetition number r using the same estimators as the
synthetic experiments, runs the three classical baselines, computes the
heuristic quantum accuracy proxy, and places both real-stream points on the
(tau, r) landscape.

DATASETS:
  1. NYC Taxi (NAB realKnownCause/nyc_taxi.csv)
       - T=10,320 half-hour bins, 2014-07 to 2015-01
       - Strong daily/weekly seasonality + documented burst events
       - Target regime overlap: seasonal + burst + low-order Markov

  2. Machine Temperature (NAB realKnownCause/machine_temperature_system_failure.csv)
       - T=22,695 5-min bins, industrial machine sensor
       - Slow drift + spike events leading up to a system failure
       - Target regime overlap: long-memory drift + burst

Both datasets are MIT-licensed and publicly downloadable from the NAB
GitHub repository.

DESIGN CHOICES (transparent and pre-committed):
  - n_bits = 6 binarised features per dataset (matched to ablation §6.9)
  - W_window = 20 forward-window for r (same as synthetic main experiments)
  - Target = 1 if next bin is at or above the 80th percentile of the
    full series. The 80th percentile is chosen because (i) it produces a
    non-trivial ~20%/80% class imbalance that exposes both correctly-
    classified majority and rare-event behaviour, and (ii) it matches
    the convention used in NAB anomaly-detection baselines for "high-
    activity" labelling. We acknowledge this is a design decision; an
    alternative threshold sensitivity check is reported in the JSON
    output for transparency.

OUTPUT:
  - figures/fig8_real_stream_validation.{pdf,png}
  - results/real_stream_summary.json
  - results/real_stream_summary.txt
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generators import compute_refreshing_time, compute_repetition_number
from classical_baselines import (
    OnlineSGDClassifier,
    AveragedSGDClassifier,
    HashTableApproximator,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, 'data', 'raw')
FIGURES_DIR = os.path.join(ROOT, 'figures')
RESULTS_DIR = os.path.join(ROOT, 'results')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
})

N_BITS = 6
W_WINDOW = 20
HIGH_QUANTILE = 0.80
SEEDS = list(range(10))


# ---------------------------------------------------------------------------
def _load_csv(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def binarise(values, ts, hours_bucket_period_hours=3, roll_window=336):
    """Convert continuous time series to (X, y_next) binary streaming form.

    n_bits = 6 features:
      bit 0-2: time-of-day bucket (3-bit, 8 buckets)
      bit 3:   weekend flag
      bit 4:   trend bit (current value above rolling median)
      bit 5:   lag-1 trend bit
    Target y_t = 1 iff next-bin value >= 80th percentile.
    """
    T = len(values)
    roll = pd.Series(values).rolling(window=roll_window, min_periods=min(48, roll_window)).median().values
    roll = np.where(np.isnan(roll), np.nanmedian(values), roll)

    X = np.zeros((T, N_BITS), dtype=np.int8)
    hours = ts.dt.hour.values
    weekends = (ts.dt.dayofweek.values >= 5).astype(np.int8)
    hour_bucket = (hours // hours_bucket_period_hours).astype(int)
    hour_bucket = hour_bucket % 8  # wrap to 3 bits
    X[:, 0] = (hour_bucket >> 0) & 1
    X[:, 1] = (hour_bucket >> 1) & 1
    X[:, 2] = (hour_bucket >> 2) & 1
    X[:, 3] = weekends
    X[:, 4] = (values > roll).astype(np.int8)
    X[:, 5] = np.r_[0, X[:-1, 4]]

    threshold = np.quantile(values, HIGH_QUANTILE)
    y = (values >= threshold).astype(np.int8)
    y_next = np.r_[y[1:], y[-1]]
    return X, y_next


def load_nyc_taxi():
    path = os.path.join(RAW_DIR, 'nyc_taxi.csv')
    df = _load_csv(path)
    values = df['value'].astype(float).values
    X, y = binarise(values, df['timestamp'],
                     hours_bucket_period_hours=3, roll_window=336)
    return X, y, values, df['timestamp']


def load_machine_temp():
    path = os.path.join(RAW_DIR, 'nab_machine_temperature.csv')
    df = _load_csv(path)
    values = df['value'].astype(float).values
    # 5-min bins; 1 day = 288 bins; 7-day rolling = 2016
    X, y = binarise(values, df['timestamp'],
                     hours_bucket_period_hours=3, roll_window=2016)
    return X, y, values, df['timestamp']


# ---------------------------------------------------------------------------
def _balanced_accuracy_and_f1(tp, fp, tn, fn):
    """Compute balanced accuracy and F1 of the positive class from confusion counts."""
    tpr = tp / max(tp + fn, 1)  # recall on positives
    tnr = tn / max(tn + fp, 1)  # recall on negatives
    bal_acc = 0.5 * (tpr + tnr)
    precision = tp / max(tp + fp, 1)
    recall = tpr
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return float(bal_acc), float(f1)


def run_baselines(X, y, seed=0, hash_dim=64, lr=0.01):
    T = len(X)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(T)
    Xp = X[perm]
    yp = y[perm]

    sgd = OnlineSGDClassifier(hash_dim=hash_dim, lr=lr, seed=seed)
    asgd = AveragedSGDClassifier(hash_dim=hash_dim, lr=lr, seed=seed)
    cm = HashTableApproximator(M=hash_dim, seed=seed)

    methods = ['OnlineSGD', 'AveragedSGD', 'CountMin']
    n_correct = {k: 0 for k in methods}
    confusion = {k: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0} for k in methods}
    rolling = {k: np.zeros(T) for k in methods}

    def _update_confusion(name, pred, yt):
        c = confusion[name]
        if pred == 1 and yt == 1:
            c['tp'] += 1
        elif pred == 1 and yt == 0:
            c['fp'] += 1
        elif pred == 0 and yt == 0:
            c['tn'] += 1
        else:
            c['fn'] += 1

    for t in range(T):
        x = Xp[t]
        yt = int(yp[t])
        for name, model in [('OnlineSGD', sgd), ('AveragedSGD', asgd)]:
            pred = 1 if model.predict(x) >= 0.5 else 0
            if pred == yt:
                n_correct[name] += 1
            _update_confusion(name, pred, yt)
            model.update(x, float(yt))
            rolling[name][t] = n_correct[name] / (t + 1)
        pred = 1 if cm.query(x) >= 0.5 else 0
        if pred == yt:
            n_correct['CountMin'] += 1
        _update_confusion('CountMin', pred, yt)
        cm.update(x, float(yt))
        rolling['CountMin'][t] = n_correct['CountMin'] / (t + 1)

    final_acc = {k: rolling[k][-1] for k in rolling}
    final_balf1 = {}
    for k in methods:
        c = confusion[k]
        bal_acc, f1 = _balanced_accuracy_and_f1(c['tp'], c['fp'], c['tn'], c['fn'])
        final_balf1[k] = {'bal_acc': bal_acc, 'f1': f1}
    return rolling, final_acc, final_balf1


def compute_proxy(T_eff, n=N_BITS, C=2.0, delta=0.05):
    eps2 = min(1.0, (C ** 2) * n * np.log(1.0 / delta) / max(T_eff, 1.0))
    return max(0.5, 1.0 - eps2)


def threshold_sensitivity(values, quantiles=(0.70, 0.80, 0.90)):
    """Compute base-rate at alternative percentile thresholds for transparency."""
    out = {}
    for q in quantiles:
        thr = np.quantile(values, q)
        rate = float((values >= thr).mean())
        out[f'q={q:.2f}'] = {'threshold': float(thr), 'positive_rate': rate}
    return out


# ---------------------------------------------------------------------------
def process_dataset(name, X, y, values, ts):
    T, n = X.shape
    print(f'\n=== {name} ===')
    print(f'  T={T:,}  n={n}  base-rate y={y.mean():.3f}')

    tau_emp = compute_refreshing_time(X)
    r_emp = compute_repetition_number(X, window=W_WINDOW)
    T_eff = T / (tau_emp * r_emp)
    proxy_acc = compute_proxy(T_eff, n=n)
    print(f'  tau={tau_emp:.2f}  r={r_emp:.3f}  T_eff={T_eff:.1f}  proxy={proxy_acc:.3f}')

    methods = ['OnlineSGD', 'AveragedSGD', 'CountMin']
    finals_acc = {k: [] for k in methods}
    finals_bal = {k: [] for k in methods}
    finals_f1 = {k: [] for k in methods}
    rolling_runs = {k: [] for k in methods}
    for s in SEEDS:
        rolling, final_acc, final_balf1 = run_baselines(X, y, seed=s)
        for k in methods:
            finals_acc[k].append(final_acc[k])
            finals_bal[k].append(final_balf1[k]['bal_acc'])
            finals_f1[k].append(final_balf1[k]['f1'])
            rolling_runs[k].append(rolling[k])

    def _summary_stats(arr):
        a = np.array(arr)
        ci = 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
        return {'mean': float(a.mean()), 'ci95_half': float(ci),
                'min': float(a.min()), 'max': float(a.max())}

    summary = {}
    for k in methods:
        summary[k] = {
            'accuracy': _summary_stats(finals_acc[k]),
            'balanced_accuracy': _summary_stats(finals_bal[k]),
            'f1': _summary_stats(finals_f1[k]),
            # legacy keys for figure code
            'mean': float(np.mean(finals_acc[k])),
            'ci95_half': float(1.96 * np.array(finals_acc[k]).std(ddof=1) /
                               np.sqrt(len(finals_acc[k]))),
        }
        a = summary[k]
        print(f'  {k:12s}: acc={a["accuracy"]["mean"]:.3f} +- {a["accuracy"]["ci95_half"]:.3f}'
              f'   bal-acc={a["balanced_accuracy"]["mean"]:.3f}'
              f'   F1={a["f1"]["mean"]:.3f}')

    pos_rate = float(y.mean())
    majority_acc = max(pos_rate, 1 - pos_rate)
    # Majority predicts the always-majority class (always 0 here, since 80% are 0):
    majority_pred = 0 if pos_rate < 0.5 else 1
    if majority_pred == 0:
        # all predicted negative: TP=0, FP=0, TN=all_neg, FN=all_pos
        majority_bal = 0.5
        majority_f1 = 0.0
    else:
        majority_bal = 0.5
        majority_f1 = 2 * pos_rate / (pos_rate + 1)

    return {
        'name': name, 'T': int(T), 'n': int(n),
        'tau_emp': float(tau_emp), 'r_emp': float(r_emp),
        'T_eff': float(T_eff), 'proxy_accuracy': float(proxy_acc),
        'classical': summary,
        'base_rate': pos_rate,
        'majority_baseline': float(majority_acc),
        'majority_balanced_accuracy': float(majority_bal),
        'majority_f1': float(majority_f1),
        'threshold_sensitivity': threshold_sensitivity(values),
    }, rolling_runs, values, ts


# ---------------------------------------------------------------------------
def main():
    print('Loading and processing 2 NAB datasets...')

    X1, y1, v1, ts1 = load_nyc_taxi()
    out1, roll1, _, _ = process_dataset('NYC Taxi', X1, y1, v1, ts1)

    X2, y2, v2, ts2 = load_machine_temp()
    out2, roll2, _, _ = process_dataset('Machine Temperature', X2, y2, v2, ts2)

    out = {
        'datasets': [out1, out2],
        'binarisation': {
            'n_bits': N_BITS,
            'features': ['hour-bucket-bit-0', 'hour-bucket-bit-1',
                         'hour-bucket-bit-2', 'weekend',
                         'trend-bit', 'lag1-trend-bit'],
            'target': f'next-bin value >= {int(HIGH_QUANTILE*100)}th percentile',
            'forward_window_W': W_WINDOW,
        },
    }

    json_path = os.path.join(RESULTS_DIR, 'real_stream_summary.json')
    with open(json_path, 'w') as fp:
        json.dump(out, fp, indent=2)
    print(f'\nWrote {json_path}')

    # LaTeX-ready table rows
    txt_path = os.path.join(RESULTS_DIR, 'real_stream_summary.txt')
    with open(txt_path, 'w') as fp:
        # Compact landscape table: Dataset | T | tau | r | T_eff | proxy | acc | majority
        fp.write('% Compact landscape table (acc + majority)\n')
        for d in out['datasets']:
            fp.write(
                f"{d['name']}  &  {d['T']:,}  &  {d['tau_emp']:.1f}  &  "
                f"{d['r_emp']:.2f}  &  {d['T_eff']:.0f}  &  "
                f"{d['proxy_accuracy']:.3f}  &  "
                f"{d['classical']['OnlineSGD']['accuracy']['mean']:.3f}  &  "
                f"{d['classical']['AveragedSGD']['accuracy']['mean']:.3f}  &  "
                f"{d['classical']['CountMin']['accuracy']['mean']:.3f}  &  "
                f"{d['majority_baseline']:.3f}\\\\\n"
            )
        fp.write('\n% Imbalance-aware metrics: per-baseline acc / bal-acc / F1 + majority\n')
        for d in out['datasets']:
            cl = d['classical']
            fp.write(
                f"{d['name']} & majority & "
                f"{d['majority_baseline']:.3f} & "
                f"{d['majority_balanced_accuracy']:.3f} & "
                f"{d['majority_f1']:.3f}\\\\\n"
            )
            for m in ['OnlineSGD', 'AveragedSGD', 'CountMin']:
                fp.write(
                    f"{d['name']} & {m} & "
                    f"{cl[m]['accuracy']['mean']:.3f} & "
                    f"{cl[m]['balanced_accuracy']['mean']:.3f} & "
                    f"{cl[m]['f1']['mean']:.3f}\\\\\n"
                )
    print(f'Wrote {txt_path}')

    # ----------------------------------------------------------------------
    # Figure: 2 rows x 2 cols
    #   (a) NYC Taxi raw   |   (b) Machine Temp raw
    #   (c) Combined cumulative accuracy   |   (d) (tau, r) landscape with both
    # ----------------------------------------------------------------------
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    sub = slice(0, 2 * 24 * 14)
    ax.plot(ts1.iloc[sub], v1[sub], lw=0.6, color='steelblue')
    ax.set_xlabel('timestamp (first 14 days)')
    ax.set_ylabel('pickup count / 30 min')
    ax.set_title('(a) NYC Taxi (NAB)')
    ax.tick_params(axis='x', rotation=30, labelsize=8)

    ax = fig.add_subplot(gs[0, 1])
    sub = slice(0, 12 * 24 * 14)
    ax.plot(ts2.iloc[sub], v2[sub], lw=0.5, color='darkorange')
    ax.set_xlabel('timestamp (first 14 days)')
    ax.set_ylabel('temperature reading')
    ax.set_title('(b) Machine Temperature (NAB)')
    ax.tick_params(axis='x', rotation=30, labelsize=8)

    # (c) cumulative accuracy for both datasets
    ax = fig.add_subplot(gs[1, 0])
    colors = {'OnlineSGD': 'C0', 'AveragedSGD': 'C2', 'CountMin': 'C3'}
    proxy1 = out1['proxy_accuracy']
    proxy2 = out2['proxy_accuracy']

    # Plot end-of-stream accuracies as a grouped bar chart
    methods = ['OnlineSGD', 'AveragedSGD', 'CountMin']
    width = 0.32
    xs = np.arange(len(methods))
    nyc_vals = [out1['classical'][m]['mean'] for m in methods]
    nyc_cis = [out1['classical'][m]['ci95_half'] for m in methods]
    mt_vals = [out2['classical'][m]['mean'] for m in methods]
    mt_cis = [out2['classical'][m]['ci95_half'] for m in methods]
    ax.bar(xs - width/2, nyc_vals, width, yerr=nyc_cis, capsize=4,
           color='steelblue', label='NYC Taxi')
    ax.bar(xs + width/2, mt_vals, width, yerr=mt_cis, capsize=4,
           color='darkorange', label='Machine Temp')
    ax.axhline(proxy1, color='steelblue', ls='--', lw=1.0,
               label=f'NYC proxy={proxy1:.2f}', alpha=0.7)
    ax.axhline(proxy2, color='darkorange', ls='--', lw=1.0,
               label=f'MT proxy={proxy2:.2f}', alpha=0.7)
    ax.axhline(out1['majority_baseline'], color='steelblue', ls=':', lw=1.0, alpha=0.5)
    ax.axhline(out2['majority_baseline'], color='darkorange', ls=':', lw=1.0, alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(methods)
    ax.set_ylabel('final cumulative accuracy')
    ax.set_ylim(0.4, 1.0)
    ax.set_title('(c) Empirical baselines vs proxy (both streams)')
    ax.legend(fontsize=7, loc='lower right', ncol=2)

    # (d) (tau, r) landscape with both real points
    ax = fig.add_subplot(gs[1, 1])
    tau_grid = np.linspace(1, 60, 80)
    r_grid = np.linspace(1, 4, 80)
    Tg, Rg = np.meshgrid(tau_grid, r_grid)
    T_ref = max(out1['T'], out2['T'])
    Teff_grid = T_ref / (Tg * Rg)
    eps2 = np.minimum(1.0, (2.0 ** 2) * N_BITS * np.log(1.0 / 0.05) / Teff_grid)
    acc_grid = np.maximum(0.5, 1.0 - eps2)
    cm = ax.contourf(Tg, Rg, acc_grid, levels=12, cmap='viridis')
    plt.colorbar(cm, ax=ax, label='proxy accuracy')

    synth = {
        'IID':         (1.0, 1.00),
        'Markov':      (8.0, 1.00),
        'Seasonal':    (12.0, 1.05),
        'Burst':       (3.0, 1.50),
        'Long-memory': (50.0, 1.10),
    }
    for name, (tt, rr) in synth.items():
        ax.scatter(tt, rr, marker='x', s=70, c='white', linewidths=2)
        ax.annotate(name, (tt, rr), textcoords='offset points',
                    xytext=(5, 5), fontsize=7, color='white')

    ax.scatter(out1['tau_emp'], out1['r_emp'], marker='*', s=240, c='steelblue',
               edgecolors='white', linewidths=1.4, zorder=5, label='NYC Taxi')
    ax.annotate('NYC Taxi', (out1['tau_emp'], out1['r_emp']),
                textcoords='offset points', xytext=(8, -10),
                fontsize=9, color='steelblue', fontweight='bold')
    ax.scatter(out2['tau_emp'], out2['r_emp'], marker='*', s=240, c='darkorange',
               edgecolors='white', linewidths=1.4, zorder=5, label='Machine Temp')
    ax.annotate('Machine Temp', (out2['tau_emp'], out2['r_emp']),
                textcoords='offset points', xytext=(8, 8),
                fontsize=9, color='darkorange', fontweight='bold')
    ax.set_xlabel(r'refreshing time $\tau$')
    ax.set_ylabel(r'repetition number $r$')
    ax.set_title('(d) Real-stream placement on landscape')
    ax.legend(loc='upper right', fontsize=8)

    out_pdf = os.path.join(FIGURES_DIR, 'fig8_real_stream_validation.pdf')
    out_png = os.path.join(FIGURES_DIR, 'fig8_real_stream_validation.png')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()
    print(f'Wrote {out_pdf}')
    print(f'Wrote {out_png}')


if __name__ == '__main__':
    main()
