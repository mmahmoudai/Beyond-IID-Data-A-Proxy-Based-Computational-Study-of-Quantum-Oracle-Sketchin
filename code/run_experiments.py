"""
Main Experiment Runner for the proxy-based computational study of
quantum oracle sketching robustness under structured non-IID streaming.

Multi-seed version: every stochastic figure reports mean plus 95%
confidence band across SEEDS. Deterministic analytic landscapes
(Experiment 3) are not seeded.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generators import (
    stream_iid, stream_markov, stream_seasonal, stream_burst,
    stream_long_memory, generate_linear_target, eval_linear,
    compute_correlation_params, generate_all_streams
)
from classical_baselines import (
    HashTableApproximator, FrequentDirections, OnlineSGDClassifier,
    AveragedSGDClassifier,
)
from quantum_bounds import (
    quantum_oracle_error, quantum_function_mse,
    quantum_classification_accuracy, quantum_pca_subspace_error,
    quantum_memory, classical_memory_lower_bound,
    compute_regime_quantum_bounds, effective_sample_analysis,
    markov_effective_tau, seasonal_effective_tau,
    burst_effective_r, long_memory_effective_tau
)

# ---------------------------------------------------------------------------
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style='whitegrid', font_scale=1.2)
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

COLORS = {
    'IID': '#2196F3',
    'Markov': '#FF5722',
    'Seasonal': '#4CAF50',
    'Burst': '#9C27B0',
    'Long-memory': '#FF9800',
}
QUANTUM_COLOR = '#E91E63'
N_BITS = 10
SEEDS = list(range(42, 52))          # 10 seeds
SEED = SEEDS[0]                       # for legacy single-seed paths (Exp 3)

REGIME_PARAMS = {
    'IID': {},
    'Markov': {'rho': 0.7},
    'Seasonal': {'period': 100, 'drift_strength': 0.3},
    'Burst': {'burst_len': 10, 'burst_prob': 0.3},
    'Long-memory': {'H': 0.8},
}

print("=" * 70)
print("EXPERIMENT SUITE: Proxy-based computational study")
print(f"Multi-seed: {len(SEEDS)} seeds per stochastic experiment")
print("=" * 70)
print(f"Input dimension n = {N_BITS}, domain size N = {2**N_BITS}")
print()


def _gen(regime, n, T, seed):
    if regime == 'IID':
        return stream_iid(n, T, seed=seed)
    elif regime == 'Markov':
        return stream_markov(n, T, rho=0.7, seed=seed)
    elif regime == 'Seasonal':
        return stream_seasonal(n, T, period=100, drift_strength=0.3, seed=seed)
    elif regime == 'Burst':
        return stream_burst(n, T, burst_len=10, burst_prob=0.3, seed=seed)
    elif regime == 'Long-memory':
        return stream_long_memory(n, T, H=0.8, seed=seed)


def _make_labels(X, w):
    return np.array([int(np.dot(x.astype(float) - 0.5, w) > 0) for x in X])


def _ci95(mat):
    """Given array of shape (n_seeds, n_points), return (mean, half-width-95-CI)."""
    mat = np.asarray(mat)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(mean)
    half = 1.96 * std / np.sqrt(mat.shape[0])
    return mean, half


# ---------------------------------------------------------------------------
# Count-Min-style classifier: uses HashTableApproximator over input IDs,
# stores labels, thresholds the averaged label at 0.5.
# ---------------------------------------------------------------------------
class CountMinClassifier:
    def __init__(self, hash_dim=64, seed=42):
        self.ht = HashTableApproximator(M=hash_dim, seed=seed)
        self.hash_dim = hash_dim
        self.memory = hash_dim
        self.n_updates = 0

    def update(self, x, y):
        self.ht.update(x, float(y))
        self.n_updates += 1

    def predict(self, x):
        return 1 if self.ht.query(x) >= 0.5 else 0

    def process_stream(self, X, y):
        for i in range(len(X)):
            self.update(X[i], y[i])

    def evaluate(self, X_test, y_test):
        correct = 0
        for i in range(len(X_test)):
            if self.predict(X_test[i]) == y_test[i]:
                correct += 1
        return correct / len(X_test)


# ===========================================================================
# Experiment 1: Classification accuracy vs samples, multi-seed with CIs.
# Panel (b) shows MSE proxy on log scale (no accuracy-clip artifact).
# ===========================================================================
def experiment_1():
    print("[Experiment 1] Accuracy vs. T, multi-seed (10 seeds)")
    print("-" * 50)

    n = N_BITS
    T_max = 20000
    T_values = np.unique(np.logspace(2, np.log10(T_max), 15).astype(int))
    M_classical = 64

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: three classical baselines, multi-seed
    ax = axes[0]
    for rname in COLORS:
        sgd_mat = np.zeros((len(SEEDS), len(T_values)))
        avg_mat = np.zeros((len(SEEDS), len(T_values)))
        cm_mat = np.zeros((len(SEEDS), len(T_values)))
        for si, seed in enumerate(SEEDS):
            rng = np.random.RandomState(seed)
            w = rng.randn(n); w /= np.linalg.norm(w)
            X_full = _gen(rname, n, T_max, seed)
            y_full = _make_labels(X_full, w)
            X_test = stream_iid(n, 2000, seed=seed + 100)
            y_test = _make_labels(X_test, w)
            for ti, T in enumerate(T_values):
                sgd = OnlineSGDClassifier(hash_dim=M_classical, lr=0.05, seed=seed)
                sgd.process_stream(X_full[:T], y_full[:T])
                sgd_mat[si, ti] = sgd.evaluate(X_test, y_test)
                avg = AveragedSGDClassifier(hash_dim=M_classical, lr=0.05, seed=seed)
                avg.process_stream(X_full[:T], y_full[:T])
                avg_mat[si, ti] = avg.evaluate(X_test, y_test)
                cm = CountMinClassifier(hash_dim=M_classical, seed=seed)
                cm.process_stream(X_full[:T], y_full[:T])
                cm_mat[si, ti] = cm.evaluate(X_test, y_test)
        sgd_m, sgd_h = _ci95(sgd_mat)
        avg_m, avg_h = _ci95(avg_mat)
        cm_m, cm_h = _ci95(cm_mat)
        ax.plot(T_values, sgd_m, '-', color=COLORS[rname], lw=1.8, label=f'{rname}: SGD')
        ax.fill_between(T_values, sgd_m - sgd_h, sgd_m + sgd_h, color=COLORS[rname], alpha=0.12)
        ax.plot(T_values, avg_m, '--', color=COLORS[rname], lw=1.5, alpha=0.8)
        ax.fill_between(T_values, avg_m - avg_h, avg_m + avg_h, color=COLORS[rname], alpha=0.08)
        ax.plot(T_values, cm_m, ':', color=COLORS[rname], lw=1.2, alpha=0.7)
        print(f"  {rname}: SGD={sgd_m[-1]:.3f}+/-{sgd_h[-1]:.3f}  "
              f"Avg={avg_m[-1]:.3f}+/-{avg_h[-1]:.3f}  "
              f"CM={cm_m[-1]:.3f}+/-{cm_h[-1]:.3f}")

    # Legend: compact — show line-style meaning once
    import matplotlib.lines as mlines
    lg_sgd = mlines.Line2D([], [], color='gray', linestyle='-', lw=1.8, label='SGD (solid)')
    lg_avg = mlines.Line2D([], [], color='gray', linestyle='--', lw=1.5, label='Averaged SGD (dashed)')
    lg_cm = mlines.Line2D([], [], color='gray', linestyle=':', lw=1.2, label='Count-Min (dotted)')
    regime_handles = [mlines.Line2D([], [], color=c, lw=3, label=n) for n, c in COLORS.items()]
    first_legend = ax.legend(handles=[lg_sgd, lg_avg, lg_cm], loc='lower right',
                             fontsize=8, framealpha=0.9)
    ax.add_artist(first_legend)
    ax.legend(handles=regime_handles, loc='upper left', fontsize=8, framealpha=0.9, ncol=2)

    ax.set_xscale('log')
    ax.set_xlabel('Number of streaming samples $T$')
    ax.set_ylabel('Classification accuracy (mean, 95% CI band)')
    ax.set_title(f'(a) Classical baselines at $M={M_classical}$')
    ax.set_ylim(0.45, 1.0)

    # Panel B: quantum MSE proxy on log scale (no clip artifact)
    ax = axes[1]
    for rname in COLORS:
        mses = []
        for T in T_values:
            qb = compute_regime_quantum_bounds(rname, T, n, REGIME_PARAMS[rname])
            mses.append(qb['mse'])
        ax.plot(T_values, mses, '--s', color=COLORS[rname], label=rname,
                markersize=3, linewidth=1.5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Number of streaming samples $T$')
    ax.set_ylabel('Oracle MSE proxy $\\epsilon^2$ (log scale)')
    ax.set_title(f'(b) Quantum proxy (oracle error squared, $n = {n}$ qubits)')
    ax.legend(framealpha=0.9, fontsize=9)

    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, 'fig1_accuracy_vs_samples.pdf')
    plt.savefig(p); plt.savefig(p.replace('.pdf', '.png')); plt.close()
    print(f"  -> {p}\n")


# ===========================================================================
# Experiment 2: Classical accuracy vs memory budget, multi-seed with CIs.
# ===========================================================================
def experiment_2():
    print("[Experiment 2] Accuracy vs. memory, multi-seed")
    print("-" * 50)

    n = N_BITS; T = 10000
    budgets = [8, 16, 32, 64, 128, 256]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for rname in COLORS:
        sgd_mat = np.zeros((len(SEEDS), len(budgets)))
        avg_mat = np.zeros((len(SEEDS), len(budgets)))
        for si, seed in enumerate(SEEDS):
            rng = np.random.RandomState(seed)
            w = rng.randn(n); w /= np.linalg.norm(w)
            X_tr = _gen(rname, n, T, seed)
            y_tr = _make_labels(X_tr, w)
            X_te = stream_iid(n, 2000, seed=seed + 100)
            y_te = _make_labels(X_te, w)
            for mi, M in enumerate(budgets):
                sgd = OnlineSGDClassifier(hash_dim=M, lr=0.05, seed=seed)
                sgd.process_stream(X_tr, y_tr)
                sgd_mat[si, mi] = sgd.evaluate(X_te, y_te)
                avg = AveragedSGDClassifier(hash_dim=M, lr=0.05, seed=seed)
                avg.process_stream(X_tr, y_tr)
                avg_mat[si, mi] = avg.evaluate(X_te, y_te)
        sgd_m, sgd_h = _ci95(sgd_mat)
        avg_m, avg_h = _ci95(avg_mat)
        ax.plot(budgets, sgd_m, '-o', color=COLORS[rname], label=f'{rname}',
                linewidth=2, markersize=5)
        ax.fill_between(budgets, sgd_m - sgd_h, sgd_m + sgd_h,
                        color=COLORS[rname], alpha=0.15)
        ax.plot(budgets, avg_m, '--', color=COLORS[rname], linewidth=1.5, alpha=0.7)
        print(f"  {rname}: SGD@M256={sgd_m[-1]:.3f}+/-{sgd_h[-1]:.3f}  "
              f"Avg@M256={avg_m[-1]:.3f}+/-{avg_h[-1]:.3f}")

    qb_iid = compute_regime_quantum_bounds('IID', T, n, {})
    ax.axhline(qb_iid['accuracy'], color=QUANTUM_COLOR, ls='--', lw=2,
               label=f'Quantum proxy (IID, {n} qubits)', alpha=0.8)

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Classical memory budget $M$')
    ax.set_ylabel('Classification accuracy (mean, 95% CI band)')
    ax.set_title(f'Classical accuracy vs. memory ($T={T:,}$; solid=SGD, dashed=Avg-SGD)')
    ax.legend(framealpha=0.9, fontsize=9, ncol=2)
    ax.set_ylim(0.45, 1.0)

    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, 'fig2_accuracy_vs_memory.pdf')
    plt.savefig(p); plt.savefig(p.replace('.pdf', '.png')); plt.close()
    print(f"  -> {p}\n")


# ===========================================================================
# Experiment 3: Analytic (tau, r) landscape. Deterministic — no seeds.
# ===========================================================================
def experiment_3():
    print("[Experiment 3] Analytic (tau, r) landscape")
    print("-" * 50)

    n = N_BITS; T = 10000
    tau_r = np.logspace(0, 2.5, 50)
    r_r = np.logspace(0, 2, 50)
    _, eps_grid = effective_sample_analysis(T, tau_r, r_r, n=n)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    T_eff = T / (tau_r[:, None] * r_r[None, :])
    ax = axes[0]
    im = ax.pcolormesh(r_r, tau_r, np.log10(T_eff), cmap='viridis', shading='auto')
    plt.colorbar(im, ax=ax, label='$\\log_{10}(T_{\\mathrm{eff}})$')
    _mark_regimes(ax, T)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Repetition number $r$'); ax.set_ylabel('Refreshing time $\\tau$')
    ax.set_title('(a) Effective sample count')

    ax = axes[1]
    im = ax.pcolormesh(r_r, tau_r, eps_grid, cmap='RdYlGn_r', shading='auto',
                       vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Oracle error proxy $\\epsilon$')
    _mark_regimes(ax, T, dark=True)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Repetition number $r$'); ax.set_ylabel('Refreshing time $\\tau$')
    ax.set_title('(b) Quantum oracle error proxy')

    ax = axes[2]
    acc_grid = np.clip(1 - eps_grid**2, 0.5, 1.0)
    im = ax.pcolormesh(r_r, tau_r, acc_grid, cmap='RdYlGn', shading='auto',
                       vmin=0.5, vmax=1.0)
    plt.colorbar(im, ax=ax, label='Accuracy proxy')
    ax.contour(r_r, tau_r, acc_grid, levels=[0.7, 0.8, 0.9],
               colors=['black'], linewidths=[1.5, 1.5, 1.5],
               linestyles=['--', '-', '--'])
    _mark_regimes(ax, T)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Repetition number $r$'); ax.set_ylabel('Refreshing time $\\tau$')
    ax.set_title('(c) Classification accuracy proxy')

    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, 'fig3_tau_r_landscape.pdf')
    plt.savefig(p); plt.savefig(p.replace('.pdf', '.png')); plt.close()
    print(f"  -> {p}\n")


def _mark_regimes(ax, T, dark=False):
    n = N_BITS
    pts = {
        'IID': (1.0, 1.0),
        'Markov': (1.0, markov_effective_tau(0.7, n)),
        'Seasonal': (1.0, seasonal_effective_tau(100, 0.3)),
        'Burst': (burst_effective_r(10, 0.3), 1.0),
        'Long-memory': (1.0, long_memory_effective_tau(0.8, T)),
    }
    ec = 'black' if dark else 'white'
    for nm, (rv, tv) in pts.items():
        ax.plot(rv, tv, 'o', color=COLORS[nm], markersize=10,
                markeredgecolor=ec, markeredgewidth=1.5, zorder=5)
        ax.annotate(nm, (rv, tv), fontsize=7, fontweight='bold',
                    color=ec, xytext=(5, 5), textcoords='offset points')


# ===========================================================================
# Experiment 4: Correlation characterization, multi-seed with error bars.
# ===========================================================================
def experiment_4():
    print("[Experiment 4] Correlation characterization, multi-seed")
    print("-" * 50)

    n = N_BITS; T = 10000
    regime_names = ['IID', 'Markov', 'Seasonal', 'Burst', 'Long-memory']

    tau_emp_mat = np.zeros((len(SEEDS), len(regime_names)))
    r_emp_mat = np.zeros((len(SEEDS), len(regime_names)))
    for si, seed in enumerate(SEEDS):
        streams = generate_all_streams(n, T, seed=seed)
        for ri, nm in enumerate(regime_names):
            tau_emp_mat[si, ri] = streams[nm]['tau']
            r_emp_mat[si, ri] = streams[nm]['r']

    tau_emp_m, tau_emp_h = _ci95(tau_emp_mat)
    r_emp_m, r_emp_h = _ci95(r_emp_mat)

    tau_th = np.zeros(len(regime_names))
    r_th = np.zeros(len(regime_names))
    for ri, nm in enumerate(regime_names):
        if nm == 'IID':       tt, rt = 1.0, 1.0
        elif nm == 'Markov':  tt, rt = markov_effective_tau(0.7, n), 1.0
        elif nm == 'Seasonal':tt, rt = seasonal_effective_tau(100, 0.3), 1.0
        elif nm == 'Burst':   tt, rt = 1.0, burst_effective_r(10, 0.3)
        else:                 tt, rt = long_memory_effective_tau(0.8, T), 1.0
        tau_th[ri] = tt; r_th[ri] = rt

    print(f"  {'Regime':<15} {'tau_emp':<14} {'tau_th':<10} {'r_emp':<14} {'r_th':<10}")
    for ri, nm in enumerate(regime_names):
        print(f"  {nm:<15} {tau_emp_m[ri]:.2f}+/-{tau_emp_h[ri]:<8.2f} "
              f"{tau_th[ri]:<10.2f} {r_emp_m[ri]:.2f}+/-{r_emp_h[ri]:<8.2f} "
              f"{r_th[ri]:<10.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(regime_names)); w_bar = 0.35

    ax = axes[0]
    ax.bar(x - w_bar/2, tau_emp_m, w_bar, yerr=tau_emp_h, capsize=4,
           color=[COLORS[n] for n in regime_names], alpha=.75,
           label='Empirical (mean $\\pm$ 95% CI)',
           error_kw={'ecolor': 'black'})
    ax.bar(x + w_bar/2, tau_th, w_bar,
           color=[COLORS[n] for n in regime_names], alpha=.35,
           edgecolor='black', lw=1.5, label='Analytic proxy')
    ax.set_yscale('log'); ax.set_ylabel('Refreshing time $\\tau$')
    ax.set_title('(a) Refreshing time: empirical vs.\\ analytic proxy')
    ax.set_xticks(x); ax.set_xticklabels(regime_names, rotation=30, ha='right')
    ax.legend()

    ax = axes[1]
    ax.bar(x - w_bar/2, r_emp_m, w_bar, yerr=r_emp_h, capsize=4,
           color=[COLORS[n] for n in regime_names], alpha=.75,
           label='Empirical (mean $\\pm$ 95% CI)',
           error_kw={'ecolor': 'black'})
    ax.bar(x + w_bar/2, r_th, w_bar,
           color=[COLORS[n] for n in regime_names], alpha=.35,
           edgecolor='black', lw=1.5, label='Analytic proxy')
    ax.set_ylabel('Repetition number $r$  (expected exact duplicates/sample)')
    ax.set_title('(b) Repetition number: empirical vs.\\ analytic proxy')
    ax.set_xticks(x); ax.set_xticklabels(regime_names, rotation=30, ha='right')
    ax.legend()

    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, 'fig4_correlation_characterization.pdf')
    plt.savefig(p); plt.savefig(p.replace('.pdf', '.png')); plt.close()
    print(f"  -> {p}\n")


# ===========================================================================
# Experiment 5: Markov rho sweep, multi-seed with CIs.
# ===========================================================================
def experiment_5():
    print("[Experiment 5] Markov sweep, multi-seed")
    print("-" * 50)

    n = N_BITS; T = 10000
    rhos = np.linspace(0, 0.95, 15)

    classical_mat = np.zeros((len(SEEDS), len(rhos)))
    q_th_mat = np.zeros((len(SEEDS), len(rhos)))
    q_emp_mat = np.zeros((len(SEEDS), len(rhos)))
    tau_e_mat = np.zeros((len(SEEDS), len(rhos)))
    tau_t_mat = np.zeros((len(SEEDS), len(rhos)))

    for si, seed in enumerate(SEEDS):
        rng = np.random.RandomState(seed)
        w = rng.randn(n); w /= np.linalg.norm(w)
        X_te = stream_iid(n, 2000, seed=seed + 100)
        y_te = _make_labels(X_te, w)
        for ri, rho in enumerate(rhos):
            X_tr = stream_markov(n, T, rho=rho, seed=seed)
            y_tr = _make_labels(X_tr, w)
            clf = OnlineSGDClassifier(hash_dim=64, lr=0.05, seed=seed)
            clf.process_stream(X_tr, y_tr)
            classical_mat[si, ri] = clf.evaluate(X_te, y_te)
            tau_th_val = markov_effective_tau(rho, n)
            q_th_mat[si, ri] = quantum_classification_accuracy(T, n, tau=tau_th_val)
            tau_emp, _ = compute_correlation_params(X_tr)
            tau_e_mat[si, ri] = tau_emp
            q_emp_mat[si, ri] = quantum_classification_accuracy(T, n, tau=tau_emp)
            tau_t_mat[si, ri] = tau_th_val

    c_m, c_h = _ci95(classical_mat)
    qth_m, qth_h = _ci95(q_th_mat)
    qemp_m, qemp_h = _ci95(q_emp_mat)
    te_m, te_h = _ci95(tau_e_mat)
    tt_m, tt_h = _ci95(tau_t_mat)

    print("  rho   classical     q_th         q_emp        tau_emp")
    for ri, rho in enumerate(rhos):
        print(f"  {rho:.2f}  {c_m[ri]:.3f}+/-{c_h[ri]:.3f}  "
              f"{qth_m[ri]:.3f}+/-{qth_h[ri]:.3f}  "
              f"{qemp_m[ri]:.3f}+/-{qemp_h[ri]:.3f}  "
              f"{te_m[ri]:.2f}+/-{te_h[ri]:.2f}")

    # Paired Wilcoxon signed-rank tests across seeds: classical vs empirical-tau
    # quantum proxy, at rho = 0.00, 0.30, 0.50, 0.70, 0.90 (closest available).
    test_targets = [0.00, 0.30, 0.50, 0.70, 0.90]
    print("  Paired Wilcoxon: classical SGD vs. quantum proxy (empirical tau)")
    sig_out = []
    for target in test_targets:
        ri = int(np.argmin(np.abs(rhos - target)))
        a = classical_mat[:, ri]
        b = q_emp_mat[:, ri]
        if np.allclose(a - b, 0.0):
            stat, p = np.nan, 1.0
        else:
            try:
                stat, p = wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
            except ValueError:
                stat, p = np.nan, np.nan
        sig_out.append((rhos[ri], float(c_m[ri]), float(qemp_m[ri]), float(p)))
        print(f"  rho~{rhos[ri]:.2f}: classical={c_m[ri]:.3f}, "
              f"q_emp={qemp_m[ri]:.3f}, p={p:.4g}")

    # Persist for table use.
    try:
        sig_path = os.path.join(FIGURES_DIR, 'exp5_wilcoxon.txt')
        with open(sig_path, 'w') as fh:
            fh.write('rho\tclassical\tq_emp\tp_value\n')
            for rho, c, q, p in sig_out:
                fh.write(f'{rho:.3f}\t{c:.4f}\t{q:.4f}\t{p:.6g}\n')
        print(f"  Wilcoxon results saved -> {sig_path}")
    except Exception as exc:
        print(f"  (warning) Could not save Wilcoxon results: {exc}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.plot(rhos, c_m, '-o', color='#2196F3',
            label='Classical online SGD ($M=64$)', linewidth=2, markersize=5)
    ax.fill_between(rhos, c_m - c_h, c_m + c_h, color='#2196F3', alpha=0.15)
    ax.plot(rhos, qth_m, '--s', color=QUANTUM_COLOR,
            label='Quantum proxy (analytic $\\tau$)', linewidth=2, markersize=5)
    ax.fill_between(rhos, qth_m - qth_h, qth_m + qth_h, color=QUANTUM_COLOR, alpha=0.10)
    ax.plot(rhos, qemp_m, ':^', color='#9C27B0',
            label='Quantum proxy (empirical $\\tau$)', linewidth=2, markersize=5)
    ax.fill_between(rhos, qemp_m - qemp_h, qemp_m + qemp_h, color='#9C27B0', alpha=0.10)
    ax.fill_between(rhos, qemp_m, c_m,
                    where=(qemp_m > c_m),
                    alpha=.12, color='#9C27B0', label='Proxy-model advantage (empirical $\\tau$)')
    ax.set_xlabel('Markov correlation strength $\\rho$')
    ax.set_ylabel('Classification accuracy (mean, 95% CI band)')
    ax.set_title('(a) Accuracy vs.\\ correlation strength')
    ax.legend(framealpha=0.9, fontsize=9)
    ax.set_ylim(0.45, 1.0)

    ax = axes[1]
    ax.plot(rhos, te_m, '-o', color='#2196F3',
            label='Empirical $\\tau$', linewidth=2, markersize=5)
    ax.fill_between(rhos, te_m - te_h, te_m + te_h, color='#2196F3', alpha=0.15)
    ax.plot(rhos, tt_m, '--s', color=QUANTUM_COLOR,
            label='Analytic proxy $\\tau = n/(1-\\rho)$', linewidth=2, markersize=5)
    ax.fill_between(rhos, te_m, tt_m, alpha=.10, color=QUANTUM_COLOR,
                    label='Proxy--practice gap')
    ax.set_xlabel('Markov correlation strength $\\rho$')
    ax.set_ylabel('Refreshing time $\\tau$')
    ax.set_title('(b) Refreshing time: empirical vs.\\ analytic proxy')
    ax.legend(framealpha=0.9); ax.set_yscale('log')

    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, 'fig5_markov_sweep.pdf')
    plt.savefig(p); plt.savefig(p.replace('.pdf', '.png')); plt.close()
    print(f"  -> {p}\n")


# ===========================================================================
# Experiment 6: Dimension scaling, multi-seed (panel b).
# Panel (a) is analytic, deterministic.
# ===========================================================================
def experiment_6():
    print("[Experiment 6] Dimension scaling, multi-seed")
    print("-" * 50)

    n_vals = [6, 8, 10, 12, 14]
    T = 10000

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    c_mem = [classical_memory_lower_bound(n) for n in n_vals]
    q_mem = [quantum_memory(n) for n in n_vals]
    ax.semilogy(n_vals, c_mem, '-o', color='#2196F3',
                label='Worst-case classical $\\Omega(\\sqrt{N})$', linewidth=2, markersize=6)
    ax.semilogy(n_vals, q_mem, '--s', color=QUANTUM_COLOR,
                label='Quantum $O(n)$', linewidth=2, markersize=6)
    ax.fill_between(n_vals, q_mem, c_mem, alpha=.12, color=QUANTUM_COLOR)
    ax.set_xlabel('Input dimension $n$ (domain size $N=2^n$)')
    ax.set_ylabel('Memory requirement')
    ax.set_title('(a) Worst-case memory lower bound')
    ax.legend(framealpha=0.9)
    ax2 = ax.twiny(); ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(n_vals)
    ax2.set_xticklabels([f'$2^{{{nv}}}$' for nv in n_vals])
    ax2.set_xlabel('Domain size $N$')

    ax = axes[1]
    for rname, col in [('IID', '#2196F3'), ('Markov', '#FF5722'),
                        ('Long-memory', '#FF9800')]:
        accs_mat = np.zeros((len(SEEDS), len(n_vals)))
        for si, seed in enumerate(SEEDS):
            for ni, n in enumerate(n_vals):
                rng = np.random.RandomState(seed)
                ww = rng.randn(n); ww /= np.linalg.norm(ww)
                Xte = stream_iid(n, 1000, seed=seed + 100)
                yte = _make_labels(Xte, ww)
                Xtr = _gen(rname, n, T, seed)
                ytr = _make_labels(Xtr, ww)
                clf = OnlineSGDClassifier(hash_dim=64, lr=0.05, seed=seed)
                clf.process_stream(Xtr, ytr)
                accs_mat[si, ni] = clf.evaluate(Xte, yte)
        m, h = _ci95(accs_mat)
        ax.plot(n_vals, m, '-o', color=col, label=f'Classical ({rname})',
                linewidth=2, markersize=6)
        ax.fill_between(n_vals, m - h, m + h, color=col, alpha=0.15)

    for rname, col, ls in [('IID', QUANTUM_COLOR, '--'),
                            ('Markov', '#FF5722', ':')]:
        q_accs = []
        for n in n_vals:
            qb = compute_regime_quantum_bounds(rname, T, n, REGIME_PARAMS[rname])
            q_accs.append(qb['accuracy'])
        ax.plot(n_vals, q_accs, ls, color=col, linewidth=2,
                label=f'Quantum proxy ({rname})', alpha=0.7)

    ax.set_xlabel('Input dimension $n$')
    ax.set_ylabel('Classification accuracy (mean, 95% CI)')
    ax.set_title('(b) Accuracy vs.\\ dimension')
    ax.legend(framealpha=0.9, fontsize=9)
    ax.set_ylim(0.45, 1.0)

    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, 'fig6_dimension_scaling.pdf')
    plt.savefig(p); plt.savefig(p.replace('.pdf', '.png')); plt.close()
    print(f"  -> {p}\n")


# ===========================================================================
# Experiment 7: Rolling accuracy. Quantum proxy is now T-dependent,
# aligned with Fig 1(b) formula.
# ===========================================================================
def experiment_7():
    print("[Experiment 7] Rolling accuracy, multi-seed")
    print("-" * 50)

    n = N_BITS; T = 15000; window = 500

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, rname in enumerate(['IID', 'Markov', 'Seasonal', 'Burst', 'Long-memory']):
        ax = axes[idx]
        time_points = list(range(window, T, 100))

        for M in [32, 64, 128]:
            roll_mat = np.zeros((len(SEEDS), len(time_points)))
            for si, seed in enumerate(SEEDS):
                rng = np.random.RandomState(seed)
                w = rng.randn(n); w /= np.linalg.norm(w)
                X_tr = _gen(rname, n, T, seed)
                y_tr = _make_labels(X_tr, w)
                clf = OnlineSGDClassifier(hash_dim=M, lr=0.05, seed=seed)
                tp_idx = 0
                for t in range(T):
                    clf.update(X_tr[t], y_tr[t])
                    if tp_idx < len(time_points) and t == time_points[tp_idx]:
                        correct = sum(1 for j in range(t - window, t)
                                      if clf.predict(X_tr[j]) == y_tr[j])
                        roll_mat[si, tp_idx] = correct / window
                        tp_idx += 1
            m, h = _ci95(roll_mat)
            ax.plot(time_points, m, '-', linewidth=1.1, alpha=0.85, label=f'SGD $M={M}$')
            ax.fill_between(time_points, m - h, m + h, alpha=0.12)

        # Averaged SGD
        roll_mat = np.zeros((len(SEEDS), len(time_points)))
        for si, seed in enumerate(SEEDS):
            rng = np.random.RandomState(seed)
            w = rng.randn(n); w /= np.linalg.norm(w)
            X_tr = _gen(rname, n, T, seed)
            y_tr = _make_labels(X_tr, w)
            clf = AveragedSGDClassifier(hash_dim=64, lr=0.05, seed=seed)
            tp_idx = 0
            for t in range(T):
                clf.update(X_tr[t], y_tr[t])
                if tp_idx < len(time_points) and t == time_points[tp_idx]:
                    correct = sum(1 for j in range(t - window, t)
                                  if clf.predict(X_tr[j]) == y_tr[j])
                    roll_mat[si, tp_idx] = correct / window
                    tp_idx += 1
        m_avg, h_avg = _ci95(roll_mat)
        ax.plot(time_points, m_avg, '--', color='#607D8B', linewidth=1.6,
                alpha=0.9, label='Avg-SGD $M=64$')
        ax.fill_between(time_points, m_avg - h_avg, m_avg + h_avg,
                        color='#607D8B', alpha=0.10)

        # Quantum proxy: T-dependent trajectory using same formula as Fig 1b
        q_traj = []
        for t in time_points:
            qb = compute_regime_quantum_bounds(rname, t, n, REGIME_PARAMS[rname])
            q_traj.append(qb['accuracy'])
        ax.plot(time_points, q_traj, '--', color=QUANTUM_COLOR, lw=2,
                alpha=0.8, label='Quantum accuracy proxy')

        ax.set_title(rname, fontsize=13, color=COLORS[rname], fontweight='bold')
        ax.set_ylim(0.45, 1.0)
        ax.set_xlabel('Time step $t$')
        if idx % 3 == 0:
            ax.set_ylabel('Rolling accuracy (mean, 95% CI)')
        if idx == 0:
            ax.legend(fontsize=7, framealpha=0.9, ncol=2, loc='lower right')

    axes[5].set_visible(False)

    plt.suptitle('Rolling classification accuracy over time (mean across seeds, 95% CI)',
                 fontsize=15, y=1.01)
    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, 'fig7_rolling_accuracy.pdf')
    plt.savefig(p); plt.savefig(p.replace('.pdf', '.png')); plt.close()
    print(f"  -> {p}\n")


# ===========================================================================
if __name__ == '__main__':
    print()
    experiment_1()
    experiment_2()
    experiment_3()
    experiment_4()
    experiment_5()
    experiment_6()
    experiment_7()
    print("=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Figures saved to: {os.path.abspath(FIGURES_DIR)}")
    print("=" * 70)
