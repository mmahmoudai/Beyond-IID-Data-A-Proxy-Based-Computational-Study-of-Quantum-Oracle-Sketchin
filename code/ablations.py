"""
Proxy-Sensitivity and Estimator Ablations (Section 5.6).

Addresses the major-revision critique that several proxy formulas appear
ad hoc. Specifically tests:

  A. ALTERNATIVE tau ESTIMATORS on synthetic streams
     - 1/e-crossing      (paper default)
     - 1/2-crossing      (more aggressive)
     - integrated ACF    (sum of |rho_k| up to first non-positive lag)
     - AR(1) effective   (-1 / log|rho_AR1|)

  B. WINDOW SIZE for r
     - W in {10, 20, 50}

  C. STREAM LENGTH
     - T in {1024, 2048, 4096, 8192}

  D. TARGET FUNCTION (linear vs nonlinear)
     - Linear: f(x) = <w, x> mod 2  (paper default)
     - Parity: f(x) = sum(x) mod 2
     - Threshold: f(x) = 1 if popcount(x) >= n/2

For each ablation we report:
  - empirical tau under each estimator
  - resulting T_eff
  - resulting proxy accuracy
  - empirical OnlineSGD accuracy

The qualitative claim we want to check is that the (tau, r) landscape
ordering between regimes is robust to estimator choice.

OUTPUT:
  - figures/fig9_ablations.pdf
  - results/ablations.json
  - results/ablations_table.txt (LaTeX-ready)
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generators import (
    stream_iid, stream_markov, stream_seasonal, stream_burst,
    stream_long_memory, generate_linear_target, eval_linear,
    compute_refreshing_time, compute_repetition_number,
)
from classical_baselines import OnlineSGDClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

N = 6
T_DEFAULT = 4096
SEEDS = list(range(10))
C, DELTA = 2.0, 0.05


# ---------------------------------------------------------------------------
# Alternative tau estimators
# ---------------------------------------------------------------------------
def _autocorr_curve(X, max_lag=200):
    T, n = X.shape
    Xc = X.astype(np.float64) - X.mean(axis=0, keepdims=True)
    var = np.mean(np.sum(Xc ** 2, axis=1))
    if var < 1e-12:
        return np.zeros(max_lag)
    ac = np.zeros(max_lag)
    for k in range(max_lag):
        if k >= T:
            break
        ac[k] = np.mean(np.sum(Xc[:T - k] * Xc[k:], axis=1)) / var
    return ac


def tau_1over_e(X):
    """Default estimator (lag where ACF < 1/e)."""
    return compute_refreshing_time(X)


def tau_1over_2(X, max_lag=500):
    """More aggressive: lag where ACF < 1/2."""
    ac = _autocorr_curve(X, max_lag=min(max_lag, len(X) // 4))
    cross = np.where(ac < 0.5)[0]
    return float(cross[0]) + 1.0 if len(cross) else float(len(ac))


def tau_integrated(X, max_lag=200):
    """Integrated absolute ACF up to first non-positive lag."""
    ac = _autocorr_curve(X, max_lag=min(max_lag, len(X) // 4))
    s = 0.0
    for k in range(1, len(ac)):
        if ac[k] <= 0:
            break
        s += ac[k]
    return 1.0 + s


def tau_ar1(X, max_lag=10):
    """Effective sample size assuming AR(1) at lag-1: -1/log|rho_1|."""
    ac = _autocorr_curve(X, max_lag=max_lag)
    if len(ac) < 2:
        return 1.0
    rho = abs(ac[1])
    if rho < 1e-3 or rho >= 1 - 1e-6:
        return 1.0
    return float(max(1.0, -1.0 / np.log(rho)))


TAU_ESTIMATORS = {
    '1/e (default)': tau_1over_e,
    '1/2-crossing': tau_1over_2,
    'integrated ACF': tau_integrated,
    'AR(1) effective': tau_ar1,
}


# ---------------------------------------------------------------------------
# Target functions
# ---------------------------------------------------------------------------
def label_linear(X, w):
    return (X @ w) % 2


def label_parity(X, w=None):
    return X.sum(axis=1) % 2


def label_threshold(X, w=None):
    n = X.shape[1]
    return (X.sum(axis=1) >= (n + 1) // 2).astype(np.int8)


TARGET_FUNCS = {
    'linear': label_linear,
    'parity': label_parity,
    'threshold': label_threshold,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
REGIMES = {
    'IID':         (stream_iid,        {}),
    'Markov 0.7':  (stream_markov,     {'rho': 0.7}),
    'Seasonal':    (stream_seasonal,   {'period': 100, 'drift_strength': 0.3}),
    'Burst':       (stream_burst,      {'burst_len': 10, 'burst_prob': 0.3}),
    'Long-memory': (stream_long_memory, {'H': 0.8}),
}


def proxy_acc(T_eff, n=N):
    eps2 = min(1.0, (C ** 2) * n * np.log(1.0 / DELTA) / max(T_eff, 1.0))
    return max(0.5, 1.0 - eps2)


def run_sgd(X, y, hash_dim=64, lr=0.01, seed=0):
    sgd = OnlineSGDClassifier(hash_dim=hash_dim, lr=lr, seed=seed)
    correct = 0
    for t in range(len(X)):
        if sgd.predict(X[t]) == y[t]:
            correct += 1
        sgd.update(X[t], float(y[t]))
    return correct / len(X)


# ---------------------------------------------------------------------------
# Ablation A: alternative tau estimators
# ---------------------------------------------------------------------------
def ablation_tau_estimators():
    print('\n=== Ablation A: alternative tau estimators ===')
    rows = []
    for regime, (fn, params) in REGIMES.items():
        X = fn(N, T_DEFAULT, seed=0, **params)
        r = compute_repetition_number(X, window=20)
        for est_name, est_fn in TAU_ESTIMATORS.items():
            tau = est_fn(X)
            T_eff = T_DEFAULT / (tau * r)
            pa = proxy_acc(T_eff)
            rows.append({
                'regime': regime, 'estimator': est_name,
                'tau': float(tau), 'r': float(r),
                'T_eff': float(T_eff), 'proxy_acc': float(pa),
            })
            print(f'  {regime:14s} | {est_name:18s} | '
                  f'tau={tau:7.2f}  r={r:.2f}  Teff={T_eff:7.1f}  proxy={pa:.3f}')
    return rows


# ---------------------------------------------------------------------------
# Ablation B: forward-window W
# ---------------------------------------------------------------------------
def ablation_window():
    print('\n=== Ablation B: forward-window W in {10, 20, 50} ===')
    rows = []
    for regime, (fn, params) in REGIMES.items():
        X = fn(N, T_DEFAULT, seed=0, **params)
        tau = compute_refreshing_time(X)
        for W in (10, 20, 50):
            r = compute_repetition_number(X, window=W)
            T_eff = T_DEFAULT / (tau * r)
            pa = proxy_acc(T_eff)
            rows.append({'regime': regime, 'W': W,
                         'tau': float(tau), 'r': float(r),
                         'T_eff': float(T_eff), 'proxy_acc': float(pa)})
            print(f'  {regime:14s} | W={W:3d} | r={r:.3f}  Teff={T_eff:7.1f}  proxy={pa:.3f}')
    return rows


# ---------------------------------------------------------------------------
# Ablation C: stream length T
# ---------------------------------------------------------------------------
def ablation_length():
    print('\n=== Ablation C: stream length T ===')
    rows = []
    for regime, (fn, params) in REGIMES.items():
        for T in (1024, 2048, 4096, 8192):
            X = fn(N, T, seed=0, **params)
            tau = compute_refreshing_time(X)
            r = compute_repetition_number(X, window=20)
            T_eff = T / (tau * r)
            pa = proxy_acc(T_eff)
            rows.append({'regime': regime, 'T': T,
                         'tau': float(tau), 'r': float(r),
                         'T_eff': float(T_eff), 'proxy_acc': float(pa)})
            print(f'  {regime:14s} | T={T:5d} | tau={tau:6.2f} r={r:.2f}  Teff={T_eff:7.1f}  proxy={pa:.3f}')
    return rows


# ---------------------------------------------------------------------------
# Ablation D: target function (linear vs parity vs threshold)
# ---------------------------------------------------------------------------
def ablation_target():
    print('\n=== Ablation D: target function ===')
    rows = []
    w = generate_linear_target(N, seed=42)
    for regime, (fn, params) in REGIMES.items():
        for tname, tfn in TARGET_FUNCS.items():
            accs = []
            for s in SEEDS:
                X = fn(N, T_DEFAULT, seed=s, **params)
                if tname == 'linear':
                    y = tfn(X, w)
                else:
                    y = tfn(X)
                accs.append(run_sgd(X, y, seed=s))
            arr = np.array(accs)
            ci = 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))
            X0 = fn(N, T_DEFAULT, seed=0, **params)
            tau = compute_refreshing_time(X0)
            r = compute_repetition_number(X0, window=20)
            T_eff = T_DEFAULT / (tau * r)
            pa = proxy_acc(T_eff)
            rows.append({'regime': regime, 'target': tname,
                         'sgd_mean': float(arr.mean()),
                         'sgd_ci95': float(ci),
                         'proxy_acc': float(pa)})
            print(f'  {regime:14s} | {tname:9s} | '
                  f'SGD={arr.mean():.3f}+-{ci:.3f}  proxy={pa:.3f}')
    return rows


# ---------------------------------------------------------------------------
def main():
    out = {
        'tau_estimators': ablation_tau_estimators(),
        'window_W': ablation_window(),
        'stream_length_T': ablation_length(),
        'target_function': ablation_target(),
    }
    json_path = os.path.join(RESULTS_DIR, 'ablations.json')
    with open(json_path, 'w') as fp:
        json.dump(out, fp, indent=2)
    print(f'\nWrote {json_path}')

    # ------------------------------------------------------------------
    # Figure: 4 panels (one per ablation)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) tau estimators (bar grouped by regime)
    ax = axes[0, 0]
    regimes = list(REGIMES.keys())
    estimators = list(TAU_ESTIMATORS.keys())
    width = 0.18
    xs = np.arange(len(regimes))
    table_a = {est: [] for est in estimators}
    for row in out['tau_estimators']:
        table_a[row['estimator']].append(row['tau'])
    for i, est in enumerate(estimators):
        ax.bar(xs + i * width, table_a[est], width, label=est)
    ax.set_xticks(xs + 1.5 * width)
    ax.set_xticklabels(regimes, rotation=15, fontsize=8)
    ax.set_ylabel(r'estimated $\tau$')
    ax.set_title('(a) Alternative $\\tau$ estimators on synthetic streams')
    ax.set_yscale('log')
    ax.legend(fontsize=7, loc='upper left')

    # (b) proxy-accuracy under each tau estimator
    ax = axes[0, 1]
    for i, est in enumerate(estimators):
        ys = [row['proxy_acc'] for row in out['tau_estimators']
              if row['estimator'] == est]
        ax.plot(regimes, ys, marker='o', label=est, lw=1.5)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel('resulting proxy accuracy')
    ax.set_title('(b) Proxy accuracy under estimator choice')
    ax.tick_params(axis='x', rotation=15, labelsize=8)
    ax.legend(fontsize=7, loc='lower left')

    # (c) window-W effect on r and proxy
    ax = axes[1, 0]
    for regime in regimes:
        Ws = [row['W'] for row in out['window_W'] if row['regime'] == regime]
        rs = [row['r'] for row in out['window_W'] if row['regime'] == regime]
        ax.plot(Ws, rs, marker='s', label=regime, lw=1.4)
    ax.set_xlabel('forward window $W$')
    ax.set_ylabel('estimated $r$')
    ax.set_title('(c) Repetition $r$ vs window size')
    ax.legend(fontsize=7, loc='upper left')

    # (d) target function effect on SGD vs proxy
    ax = axes[1, 1]
    targets = ['linear', 'parity', 'threshold']
    width = 0.22
    xs = np.arange(len(regimes))
    for i, tgt in enumerate(targets):
        ys = [row['sgd_mean'] for row in out['target_function']
              if row['target'] == tgt]
        cis = [row['sgd_ci95'] for row in out['target_function']
               if row['target'] == tgt]
        ax.bar(xs + i * width, ys, width, yerr=cis, capsize=3, label=tgt)
    ax.set_xticks(xs + width)
    ax.set_xticklabels(regimes, rotation=15, fontsize=8)
    ax.set_ylabel('SGD accuracy')
    ax.set_ylim(0.4, 1.0)
    ax.set_title('(d) SGD accuracy across target functions')
    ax.legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    out_pdf = os.path.join(FIGURES_DIR, 'fig9_ablations.pdf')
    out_png = os.path.join(FIGURES_DIR, 'fig9_ablations.png')
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.close()
    print(f'Wrote {out_pdf}')

    # LaTeX-ready summary table for tau-estimator ablation
    txt_path = os.path.join(RESULTS_DIR, 'ablations_table.txt')
    with open(txt_path, 'w') as fp:
        fp.write('% Ablation A: alternative tau estimators\n')
        fp.write('% Format: regime  &  est  &  tau  &  T_eff  &  proxy\n')
        for row in out['tau_estimators']:
            fp.write(
                f"{row['regime']}  &  {row['estimator']}  &  "
                f"{row['tau']:.2f}  &  {row['T_eff']:.0f}  &  "
                f"{row['proxy_acc']:.3f}\\\\\n"
            )
    print(f'Wrote {txt_path}')


if __name__ == '__main__':
    main()
