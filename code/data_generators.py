"""
Non-IID Data Stream Generators for Quantum Oracle Sketching Robustness Study.

Generates streaming data under five correlation regimes:
  1. IID (baseline)
  2. Markov switching
  3. Seasonal / periodic drift
  4. Burst repetition
  5. Long-memory (fractional Gaussian noise)

Each generator produces (x, f(x)) pairs where x in {0,1}^n and f: {0,1}^n -> {0,1}.
Also computes empirical refreshing time tau and repetition number r.
"""

import numpy as np
from scipy.stats import norm


def generate_target_function(n, seed=42):
    """Generate a random Boolean function as a truth table."""
    rng = np.random.RandomState(seed)
    N = 2 ** n
    truth_table = rng.randint(0, 2, size=N).astype(np.int8)
    return truth_table


def generate_linear_target(n, seed=42):
    """Generate a linear Boolean function f(x) = <w, x> mod 2."""
    rng = np.random.RandomState(seed)
    w = rng.randint(0, 2, size=n)
    return w


def eval_linear(w, x):
    """Evaluate linear Boolean function."""
    return (x @ w) % 2


# ---------------------------------------------------------------------------
# Regime 1: IID sampling
# ---------------------------------------------------------------------------
def stream_iid(n, T, seed=0):
    """IID uniform samples from {0,1}^n."""
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 2, size=(T, n)).astype(np.int8)
    return X


# ---------------------------------------------------------------------------
# Regime 2: Markov switching
# ---------------------------------------------------------------------------
def stream_markov(n, T, rho=0.5, seed=0):
    """
    Markov chain on {0,1}^n.  Each bit flips independently with prob (1-rho)/2
    at each step.  rho=0 gives IID, rho->1 gives near-deterministic chain.
    """
    rng = np.random.RandomState(seed)
    X = np.zeros((T, n), dtype=np.int8)
    X[0] = rng.randint(0, 2, size=n)
    for t in range(1, T):
        flip = rng.random(n) > (1 + rho) / 2
        X[t] = X[t - 1].copy()
        X[t, flip] = 1 - X[t - 1, flip]
    return X


# ---------------------------------------------------------------------------
# Regime 3: Seasonal / periodic drift
# ---------------------------------------------------------------------------
def stream_seasonal(n, T, period=100, drift_strength=0.3, seed=0):
    """
    Sampling distribution shifts periodically.  Bit biases oscillate:
    p_i(t) = 0.5 + drift_strength * sin(2*pi*t/period + phase_i)
    """
    rng = np.random.RandomState(seed)
    phases = rng.uniform(0, 2 * np.pi, size=n)
    X = np.zeros((T, n), dtype=np.int8)
    for t in range(T):
        bias = 0.5 + drift_strength * np.sin(2 * np.pi * t / period + phases)
        bias = np.clip(bias, 0.01, 0.99)
        X[t] = (rng.random(n) < bias).astype(np.int8)
    return X


# ---------------------------------------------------------------------------
# Regime 4: Burst repetition
# ---------------------------------------------------------------------------
def stream_burst(n, T, burst_len=10, burst_prob=0.3, seed=0):
    """
    With probability burst_prob, repeat the previous sample for up to
    burst_len steps.  Otherwise draw a fresh IID sample.
    """
    rng = np.random.RandomState(seed)
    X = np.zeros((T, n), dtype=np.int8)
    X[0] = rng.randint(0, 2, size=n)
    t = 1
    while t < T:
        if rng.random() < burst_prob:
            length = min(rng.randint(2, burst_len + 1), T - t)
            X[t:t + length] = X[t - 1]
            t += length
        else:
            X[t] = rng.randint(0, 2, size=n)
            t += 1
    return X


# ---------------------------------------------------------------------------
# Regime 5: Long-memory (fractional Gaussian noise driven)
# ---------------------------------------------------------------------------
def _fgn(T, H=0.8, seed=0):
    """Generate fractional Gaussian noise with Hurst parameter H."""
    rng = np.random.RandomState(seed)
    # Hosking method for exact fGn
    gamma = np.zeros(T)
    gamma[0] = 1.0
    for k in range(1, T):
        gamma[k] = 0.5 * (abs(k + 1) ** (2 * H) - 2 * abs(k) ** (2 * H) +
                           abs(k - 1) ** (2 * H))

    # Use circulant embedding for efficiency
    n_circ = 1
    while n_circ < 2 * T:
        n_circ *= 2
    row = np.zeros(n_circ)
    row[:T] = gamma
    row[n_circ - T + 1:] = gamma[1:][::-1]

    eigenvalues = np.real(np.fft.fft(row))
    eigenvalues = np.maximum(eigenvalues, 0)

    z = rng.standard_normal(n_circ) + 1j * rng.standard_normal(n_circ)
    w = np.fft.ifft(np.sqrt(eigenvalues) * z)
    return np.real(w[:T])


def stream_long_memory(n, T, H=0.8, seed=0):
    """
    Long-memory stream: each bit's sampling probability is driven by
    fractional Gaussian noise with Hurst exponent H.
    H=0.5 is memoryless, H>0.5 is long-range dependent.
    """
    rng = np.random.RandomState(seed + 1000)
    X = np.zeros((T, n), dtype=np.int8)
    for i in range(n):
        noise = _fgn(T, H=H, seed=seed + i)
        probs = norm.cdf(noise)  # map to [0,1]
        X[:, i] = (rng.random(T) < probs).astype(np.int8)
    return X


# ---------------------------------------------------------------------------
# Correlation diagnostics: empirical refreshing time and repetition number
# ---------------------------------------------------------------------------
def compute_refreshing_time(X):
    """
    Estimate refreshing time tau: average number of steps until the
    autocorrelation of the stream drops below 1/e.
    """
    T, n = X.shape
    X_centered = X.astype(np.float64) - X.mean(axis=0, keepdims=True)

    max_lag = min(T // 4, 500)
    autocorr = np.zeros(max_lag)
    var = np.mean(np.sum(X_centered ** 2, axis=1))
    if var < 1e-12:
        return 1.0

    for lag in range(max_lag):
        autocorr[lag] = np.mean(
            np.sum(X_centered[:T - lag] * X_centered[lag:], axis=1)
        ) / var

    # Find first crossing below 1/e
    threshold = 1.0 / np.e
    crossings = np.where(autocorr < threshold)[0]
    if len(crossings) > 0:
        return float(crossings[0]) + 1.0
    return float(max_lag)


def compute_repetition_number(X, window=20):
    """
    Estimate repetition number r: expected number of EXACT duplicates
    per sample within a forward window of `window` steps.

    This matches the analytic proxy's units ("expected duplicate copies
    per fresh sample"), replacing the earlier near-duplicate pair count
    which produced dimensionally incompatible values.
    """
    T, n = X.shape
    if T < window + 1:
        return 1.0

    dup_counts = np.zeros(T - window, dtype=np.float64)
    for i in range(T - window):
        x_i = X[i]
        # Count exact matches (Hamming distance 0) in positions i+1 .. i+window
        for j in range(i + 1, i + window + 1):
            if np.array_equal(x_i, X[j]):
                dup_counts[i] += 1

    # Average duplicate count; "repetition number" = 1 + expected duplicates
    return 1.0 + float(np.mean(dup_counts))


def compute_correlation_params(X):
    """Compute both tau and r for a stream."""
    tau = compute_refreshing_time(X)
    r = compute_repetition_number(X)
    return tau, r


# ---------------------------------------------------------------------------
# All-in-one generator
# ---------------------------------------------------------------------------
REGIMES = {
    'IID': {'func': stream_iid, 'params': {}},
    'Markov': {'func': stream_markov, 'params': {'rho': 0.7}},
    'Seasonal': {'func': stream_seasonal, 'params': {'period': 100, 'drift_strength': 0.3}},
    'Burst': {'func': stream_burst, 'params': {'burst_len': 10, 'burst_prob': 0.3}},
    'Long-memory': {'func': stream_long_memory, 'params': {'H': 0.8}},
}


def generate_all_streams(n, T, seed=0):
    """Generate streams for all correlation regimes."""
    streams = {}
    for name, spec in REGIMES.items():
        X = spec['func'](n, T, seed=seed, **spec['params'])
        tau, r = compute_correlation_params(X)
        streams[name] = {'X': X, 'tau': tau, 'r': r}
    return streams
