"""Re-analysis of Experiment 5 (Markov sweep) for the MDPI revision.

Reproduces the exp5 per-seed matrices EXACTLY (same seeds, same generators,
same training loop as code/run_experiments.py experiment_5), verifies the
reproduction against figures/exp5_wilcoxon.txt, then computes:

  (1) effect sizes for the five paired Wilcoxon tests:
      - matched-pairs rank-biserial correlation r_rb
      - mean paired difference with t-based 95% CI
      - Hodges-Lehmann estimate with exact signed-rank 95% CI (Walsh averages)
      - Holm-adjusted p-values across the 5 tests
  (2) C/delta sensitivity of the proxy-baseline crossover rho*:
      re-evaluate the closed-form proxy on the SAME per-seed empirical tau
      values under a grid C x delta; find where mean proxy crosses mean
      classical accuracy (linear interpolation on the 15-point rho grid).

This is deterministic re-analysis of existing measured inputs; no new
stochastic experiment is run (the classical accuracies are bit-identical
reproductions of the published ones).
"""
import os, sys, json
import numpy as np
from scipy.stats import wilcoxon

CODE = r"D:\research\quantum_paper\code"
sys.path.insert(0, CODE)

from data_generators import stream_iid, stream_markov, compute_correlation_params
from classical_baselines import OnlineSGDClassifier
from quantum_bounds import quantum_classification_accuracy

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp5_reanalysis.json")

N_BITS = 10
SEEDS = list(range(42, 52))
T = 10000
rhos = np.linspace(0, 0.95, 15)

def _make_labels(X, w):
    return np.array([int(np.dot(x.astype(float) - 0.5, w) > 0) for x in X])

print("Reproducing exp5 matrices (10 seeds x 15 rho)...")
classical_mat = np.zeros((len(SEEDS), len(rhos)))
q_emp_mat = np.zeros((len(SEEDS), len(rhos)))
tau_e_mat = np.zeros((len(SEEDS), len(rhos)))

for si, seed in enumerate(SEEDS):
    rng = np.random.RandomState(seed)
    w = rng.randn(N_BITS); w /= np.linalg.norm(w)
    X_te = stream_iid(N_BITS, 2000, seed=seed + 100)
    y_te = _make_labels(X_te, w)
    for ri, rho in enumerate(rhos):
        X_tr = stream_markov(N_BITS, T, rho=rho, seed=seed)
        y_tr = _make_labels(X_tr, w)
        clf = OnlineSGDClassifier(hash_dim=64, lr=0.05, seed=seed)
        clf.process_stream(X_tr, y_tr)
        classical_mat[si, ri] = clf.evaluate(X_te, y_te)
        tau_emp, _ = compute_correlation_params(X_tr)
        tau_e_mat[si, ri] = tau_emp
        q_emp_mat[si, ri] = quantum_classification_accuracy(T, N_BITS, tau=tau_emp)
    print(f"  seed {seed} done")

# ---------------------------------------------------------------- fidelity
print("\nFidelity check vs figures/exp5_wilcoxon.txt:")
ref = []
with open(r"D:\research\quantum_paper\figures\exp5_wilcoxon.txt") as fh:
    next(fh)
    for line in fh:
        parts = line.split()
        if len(parts) == 4:
            ref.append(tuple(float(x) for x in parts))

test_targets = [0.00, 0.30, 0.50, 0.70, 0.90]
fidelity_ok = True
rows = []
for k, target in enumerate(test_targets):
    ri = int(np.argmin(np.abs(rhos - target)))
    a = classical_mat[:, ri]; b = q_emp_mat[:, ri]
    stat, p = wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
    r_rho, r_c, r_q, r_p = ref[k]
    # tolerance 1e-4 matches the reference file's 4-decimal precision; the
    # p-value match to 6 significant digits is the strict fidelity criterion
    ok = (abs(a.mean() - r_c) < 1e-4) and (abs(b.mean() - r_q) < 1e-4) and (abs(p - r_p) < 1e-6)
    fidelity_ok &= ok
    print(f"  rho={rhos[ri]:.3f}  classical {a.mean():.4f} (ref {r_c:.4f})  "
          f"proxy {b.mean():.4f} (ref {r_q:.4f})  p {p:.6g} (ref {r_p:.6g})  {'OK' if ok else 'MISMATCH'}")
    rows.append((ri, a, b, p))
print(f"FIDELITY: {'EXACT MATCH' if fidelity_ok else 'MISMATCH - do not use'}")

# ------------------------------------------------- exact signed-rank helpers
def signed_rank_cdf_table(n):
    """P(W+ <= k) under H0 for sample size n (no ties), via DP."""
    # counts[w] = number of subsets of {1..n} with sum w
    maxw = n * (n + 1) // 2
    counts = np.zeros(maxw + 1, dtype=np.float64)
    counts[0] = 1.0
    for i in range(1, n + 1):
        new = counts.copy()
        new[i:] += counts[:-i] if i > 0 else 0
        counts = new
    probs = counts / (2.0 ** n)
    return np.cumsum(probs)  # cdf[k] = P(W+ <= k)

def hodges_lehmann_ci(d, alpha=0.05):
    """HL estimate + exact signed-rank CI from Walsh averages (no-ties theory)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    walsh = sorted((d[i] + d[j]) / 2.0 for i in range(n) for j in range(i, n))
    m = len(walsh)  # n(n+1)/2
    hl = float(np.median(walsh))
    cdf = signed_rank_cdf_table(n)
    # largest k with P(W <= k) <= alpha/2
    k = int(np.searchsorted(cdf, alpha / 2.0, side='right') - 1)
    k = max(k, 0)
    lo = walsh[k]              # (k+1)-th smallest
    hi = walsh[m - 1 - k]      # (k+1)-th largest
    achieved = 1.0 - 2.0 * cdf[k]
    return hl, float(lo), float(hi), float(achieved), int(k)

def rank_biserial(a, b):
    """Matched-pairs rank-biserial correlation for signed-rank test."""
    d = np.asarray(a) - np.asarray(b)
    d = d[d != 0]
    n = len(d)
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0  # simple ranks; ties unlikely here
    # use scipy-style average ranks for ties:
    from scipy.stats import rankdata
    ranks = rankdata(np.abs(d))
    wplus = ranks[d > 0].sum()
    wminus = ranks[d < 0].sum()
    tot = n * (n + 1) / 2.0
    return float((wplus - wminus) / tot), float(wplus), float(wminus), int(n)

# ------------------------------------------------------------- effect sizes
print("\nEffect sizes at the five table rho values:")
stats_rows = []
pvals = []
for ri, a, b, p in rows:
    d = a - b
    r_rb, wplus, wminus, nz = rank_biserial(a, b)
    hl, lo, hi, achieved, k = hodges_lehmann_ci(d)
    mean_d = float(d.mean())
    from scipy.stats import t as t_dist
    t_half = float(t_dist.ppf(0.975, len(d) - 1)) * d.std(ddof=1) / np.sqrt(len(d))
    stats_rows.append({
        "rho": round(float(rhos[ri]), 3),
        "classical_mean": round(float(a.mean()), 4),
        "classical_ci95_half": round(float(1.96 * a.std(ddof=1) / np.sqrt(len(a))), 4),
        "proxy_mean": round(float(b.mean()), 4),
        "proxy_ci95_half": round(float(1.96 * b.std(ddof=1) / np.sqrt(len(b))), 4),
        "mean_diff_classical_minus_proxy": round(mean_d, 4),
        "mean_diff_ci95": [round(mean_d - t_half, 4), round(mean_d + t_half, 4)],
        "hodges_lehmann": round(hl, 4),
        "hl_exact_ci95": [round(lo, 4), round(hi, 4)],
        "hl_ci_achieved_coverage": round(achieved, 4),
        "rank_biserial_r": round(r_rb, 3),
        "W_plus": wplus, "W_minus": wminus, "n_nonzero": nz,
        "wilcoxon_p": float(f"{p:.6g}"),
    })
    pvals.append(p)
    print(f"  rho={rhos[ri]:.3f}: D=classical-proxy mean {mean_d:+.4f}  "
          f"HL {hl:+.4f} CI[{lo:+.4f},{hi:+.4f}]  r_rb {r_rb:+.3f}  p {p:.4g}")

# Holm adjustment across the five tests
order = np.argsort(pvals)
holm = [None] * len(pvals)
prev = 0.0
for rank, idx in enumerate(order):
    adj = min(1.0, (len(pvals) - rank) * pvals[idx])
    adj = max(adj, prev)
    prev = adj
    holm[idx] = adj
for row, h in zip(stats_rows, holm):
    row["holm_adjusted_p"] = round(float(h), 4)
print("Holm-adjusted p:", [f"{h:.3f}" for h in holm])

# --------------------------------------------------- C / delta sensitivity
print("\nC/delta sensitivity of the proxy-baseline crossover rho*:")
def proxy_acc(tau, C, delta):
    teff = max(T / tau, 1.0)
    eps2 = (C ** 2) * N_BITS * np.log(1.0 / delta) / teff
    return max(0.5, 1.0 - eps2)

c_mean = classical_mat.mean(axis=0)
grid_C = [1.0, 1.5, 2.0, 2.5]
grid_d = [0.01, 0.05, 0.10]
cross_table = {}
for C in grid_C:
    for delta in grid_d:
        q = np.array([[proxy_acc(tau_e_mat[si, ri], C, delta)
                       for ri in range(len(rhos))] for si in range(len(SEEDS))])
        q_mean = q.mean(axis=0)
        diff = q_mean - c_mean          # positive = proxy above classical
        rho_star = None
        for i in range(len(rhos) - 1):
            if diff[i] > 0 and diff[i + 1] <= 0:
                # linear interpolation
                frac = diff[i] / (diff[i] - diff[i + 1])
                rho_star = float(rhos[i] + frac * (rhos[i + 1] - rhos[i]))
                break
        if rho_star is None:
            rho_star = float('nan') if diff[0] <= 0 else 1.0  # never crosses in range
        cross_table[f"C={C},delta={delta}"] = round(rho_star, 3) if rho_star == rho_star else "below classical at rho=0"
        print(f"  C={C:>3}, delta={delta:>4}: rho* = {cross_table[f'C={C},delta={delta}']}")

# tau_emp means along the sweep (for the divergence discussion)
tau_means = {round(float(rhos[i]), 3): round(float(tau_e_mat[:, i].mean()), 2)
             for i in range(len(rhos))}

out = {
    "fidelity_exact_match": bool(fidelity_ok),
    "seeds": SEEDS, "T": T, "n_bits": N_BITS,
    "test_rows": stats_rows,
    "holm_note": "Holm step-down across the 5 sweep tests",
    "crossover_rho_star_by_C_delta": cross_table,
    "tau_emp_mean_by_rho": tau_means,
}
with open(OUT, "w") as fh:
    json.dump(out, fh, indent=2)
print(f"\nSaved -> {OUT}")
