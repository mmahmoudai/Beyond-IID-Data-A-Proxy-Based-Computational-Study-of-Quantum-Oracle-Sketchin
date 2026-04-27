"""
Analytical Quantum Oracle Sketching Performance Bounds.

Computes theoretical quantum performance guarantees from the framework of
Zhao et al. (2024) "Exponential quantum advantage in processing massive
classical data". These are analytical evaluations of proven bounds, not
circuit simulations.

Key quantities:
  - n: number of qubits / bits in input
  - N = 2^n: domain size
  - T: number of streaming samples
  - tau: refreshing time of the data stream
  - r: repetition number
  - T_eff = T / (tau * r): effective independent sample count
"""

import numpy as np


# ===========================================================================
# Core quantum oracle sketching bounds
# ===========================================================================

def quantum_oracle_error(T, n, tau=1.0, r=1.0, delta=0.05):
    """
    Oracle approximation error bound for quantum oracle sketching.

    Under the quantum oracle sketching framework, given T samples from a
    stream with refreshing time tau and repetition number r, the oracle
    approximation error satisfies:

        epsilon <= C * sqrt(n / T_eff) * sqrt(log(1/delta))

    where T_eff = T / (tau * r) and C is a universal constant.

    For IID data (tau=1, r=1), this reduces to epsilon ~ sqrt(n/T).

    Returns the error bound epsilon.
    """
    T_eff = T / (tau * r)
    T_eff = max(T_eff, 1.0)
    C = 2.0  # universal constant (conservative)
    epsilon = C * np.sqrt(n * np.log(1.0 / delta) / T_eff)
    return min(epsilon, 1.0)


def quantum_memory(n):
    """
    Quantum memory requirement: O(n) qubits = O(log N) qubits.
    This is polylogarithmic in the domain size N = 2^n.
    """
    return n  # in qubits


def classical_memory_lower_bound(n, task='function'):
    """
    Classical memory lower bound for equivalent task.

    For function approximation: Omega(sqrt(N)) = Omega(2^{n/2})
    For classification: Omega(N^{1/3}) in some regimes
    """
    N = 2 ** n
    if task == 'function':
        return int(np.sqrt(N))
    elif task == 'classification':
        return int(N ** (1.0 / 3))
    else:
        return int(np.sqrt(N))


# ===========================================================================
# Task-specific quantum bounds
# ===========================================================================

def quantum_classification_accuracy(T, n, tau=1.0, r=1.0, delta=0.05):
    """
    Quantum classification accuracy bound.

    The classification error is bounded by the square of the oracle error:
        error_rate <= epsilon^2

    So accuracy >= 1 - epsilon^2.
    """
    eps = quantum_oracle_error(T, n, tau, r, delta)
    accuracy = 1.0 - eps ** 2
    return max(accuracy, 0.5)  # at worst, random guessing


def quantum_pca_subspace_error(T, n, k=3, tau=1.0, r=1.0, delta=0.05):
    """
    Quantum subspace approximation error for streaming PCA.

    The subspace error (average principal angle) is bounded by:
        theta <= epsilon * sqrt(k / n)

    where epsilon is the oracle approximation error.
    """
    eps = quantum_oracle_error(T, n, tau, r, delta)
    theta = eps * np.sqrt(k / n)
    return min(theta, np.pi / 2)


def quantum_function_mse(T, n, tau=1.0, r=1.0, delta=0.05):
    """
    Quantum function approximation MSE bound.

    MSE <= epsilon^2 where epsilon is oracle error.
    """
    eps = quantum_oracle_error(T, n, tau, r, delta)
    return eps ** 2


# ===========================================================================
# Scaling analysis: quantum vs classical as function of parameters
# ===========================================================================

def compute_advantage_ratio(T_values, n, tau=1.0, r=1.0,
                            classical_errors=None, task='function'):
    """
    Compute the ratio of classical error to quantum error bound
    across different sample counts T.

    Returns arrays of quantum bounds and advantage ratios.
    """
    q_errors = np.array([
        quantum_function_mse(T, n, tau, r) if task == 'function'
        else quantum_classification_accuracy(T, n, tau, r)
        for T in T_values
    ])

    if classical_errors is not None:
        if task == 'function':
            # For MSE: ratio = classical / quantum (higher = more advantage)
            ratio = classical_errors / np.maximum(q_errors, 1e-10)
        else:
            # For accuracy: ratio = quantum / classical (higher = more advantage)
            ratio = q_errors / np.maximum(classical_errors, 1e-10)
    else:
        ratio = None

    return q_errors, ratio


def theoretical_advantage_landscape(n_values, tau_values, T=10000):
    """
    Compute advantage landscape: quantum memory advantage as a function
    of data dimension n and refreshing time tau.

    Returns a 2D array where entry [i,j] is the log ratio of
    classical memory requirement to quantum memory requirement.
    """
    landscape = np.zeros((len(n_values), len(tau_values)))

    for i, n in enumerate(n_values):
        q_mem = quantum_memory(n)
        c_mem = classical_memory_lower_bound(n)
        for j, tau in enumerate(tau_values):
            # Quantum: memory stays O(n), but needs tau*r more samples
            # Classical: memory Omega(sqrt(N)), also needs more samples
            # Memory advantage ratio (log scale)
            landscape[i, j] = np.log2(c_mem / max(q_mem, 1))

    return landscape


def effective_sample_analysis(T, tau_range, r_range, n=10):
    """
    Analyze how effective sample count and oracle error depend on
    refreshing time and repetition number.

    Returns 2D arrays for T_eff and epsilon.
    """
    T_eff_grid = np.zeros((len(tau_range), len(r_range)))
    eps_grid = np.zeros((len(tau_range), len(r_range)))

    for i, tau in enumerate(tau_range):
        for j, r in enumerate(r_range):
            T_eff_grid[i, j] = T / (tau * r)
            eps_grid[i, j] = quantum_oracle_error(T, n, tau, r)

    return T_eff_grid, eps_grid


# ===========================================================================
# Non-IID specific bounds: refined for each correlation family
# ===========================================================================

def markov_effective_tau(rho, n):
    """
    For a Markov chain on {0,1}^n with per-bit flip probability (1-rho)/2,
    the mixing time (and thus refreshing time) is:
        tau ~ n / (1 - rho)  for rho < 1
    """
    if rho >= 1.0:
        return float('inf')
    return n / (1.0 - rho)


def seasonal_effective_tau(period, drift_strength):
    """
    For seasonal drift with period P and strength alpha:
        tau ~ P * alpha (effective refreshing time scales with period)
    When alpha is small, the stream is nearly IID.
    """
    return period * drift_strength


def burst_effective_r(burst_len, burst_prob):
    """
    For burst repetition with burst length L and probability p:
        r ~ 1 + p * L  (expected repetitions per effective sample)
    """
    return 1.0 + burst_prob * burst_len


def long_memory_effective_tau(H, T):
    """
    For fractional Gaussian noise with Hurst parameter H:
        tau ~ T^{2H-1} for H > 0.5
    This grows with T for long-memory processes.
    """
    if H <= 0.5:
        return 1.0
    return T ** (2 * H - 1)


def compute_regime_quantum_bounds(regime, T, n, params):
    """
    Compute quantum oracle error for a specific correlation regime
    using refined bounds.

    Args:
        regime: one of 'IID', 'Markov', 'Seasonal', 'Burst', 'Long-memory'
        T: number of samples
        n: input dimension
        params: dict of regime-specific parameters

    Returns:
        dict with tau, r, T_eff, epsilon, memory
    """
    if regime == 'IID':
        tau, r = 1.0, 1.0
    elif regime == 'Markov':
        tau = markov_effective_tau(params.get('rho', 0.7), n)
        r = 1.0
    elif regime == 'Seasonal':
        tau = seasonal_effective_tau(params.get('period', 100),
                                     params.get('drift_strength', 0.3))
        r = 1.0
    elif regime == 'Burst':
        tau = 1.0
        r = burst_effective_r(params.get('burst_len', 10),
                              params.get('burst_prob', 0.3))
    elif regime == 'Long-memory':
        tau = long_memory_effective_tau(params.get('H', 0.8), T)
        r = 1.0
    else:
        tau, r = 1.0, 1.0

    T_eff = T / (tau * r)
    eps = quantum_oracle_error(T, n, tau, r)
    mem = quantum_memory(n)

    return {
        'tau': tau,
        'r': r,
        'T_eff': T_eff,
        'epsilon': eps,
        'mse': eps ** 2,
        'memory': mem,
        'accuracy': quantum_classification_accuracy(T, n, tau, r),
    }
