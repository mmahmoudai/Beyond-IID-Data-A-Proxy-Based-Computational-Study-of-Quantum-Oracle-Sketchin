"""
Classical Memory-Bounded Streaming Algorithms.

Implements streaming algorithms for three downstream tasks:
  1. Function approximation via hash-table with bounded memory
  2. Streaming PCA via Frequent Directions
  3. Streaming classification via online SGD with feature hashing

All operate under strict memory budgets to show scaling limitations.
"""

import numpy as np


# ===========================================================================
# Task 1: Streaming Function Approximation (Hash-Table Approximator)
# ===========================================================================
class HashTableApproximator:
    """
    Hash-table based streaming function approximator with bounded memory.
    Maps inputs to M buckets; maintains running average of f-values per bucket.
    Memory: O(M).  Error floor determined by hash collisions (M << N).
    """

    def __init__(self, M=64, seed=42):
        self.M = M
        self.memory = M
        rng = np.random.RandomState(seed)
        self.a = rng.randint(1, 2**31 - 1)
        self.b = rng.randint(0, 2**31 - 1)
        self.value_sum = np.zeros(M, dtype=np.float64)
        self.count = np.zeros(M, dtype=np.float64)
        self.n_updates = 0

    def _hash(self, x_int):
        return ((self.a * x_int + self.b) % (2**31 - 1)) % self.M

    def _x_to_int(self, x):
        val = 0
        for i in range(len(x)):
            if x[i]:
                val |= (1 << i)
        return val

    def update(self, x, value):
        x_int = self._x_to_int(x)
        h = self._hash(x_int)
        self.value_sum[h] += value
        self.count[h] += 1
        self.n_updates += 1

    def query(self, x):
        x_int = self._x_to_int(x)
        h = self._hash(x_int)
        if self.count[h] == 0:
            return 0.5
        return self.value_sum[h] / self.count[h]

    def process_stream(self, X, labels):
        for i in range(len(X)):
            self.update(X[i], labels[i])

    def evaluate(self, X_test, y_test):
        errors = np.zeros(len(X_test))
        for i in range(len(X_test)):
            pred = self.query(X_test[i])
            errors[i] = (pred - y_test[i]) ** 2
        return np.mean(errors)

    def evaluate_accuracy(self, X_test, y_test):
        correct = 0
        for i in range(len(X_test)):
            pred = 1 if self.query(X_test[i]) >= 0.5 else 0
            if pred == y_test[i]:
                correct += 1
        return correct / len(X_test)


# ===========================================================================
# Task 2: Streaming PCA (Frequent Directions)
# ===========================================================================
class FrequentDirections:
    """
    Frequent Directions algorithm for streaming low-rank approximation.
    Memory: O(ell * d) where ell = sketch size, d = dimension.
    """

    def __init__(self, d, ell=16):
        self.d = d
        self.ell = ell
        self.memory = ell * d
        self.sketch = np.zeros((2 * ell, d), dtype=np.float64)
        self.next_row = 0
        self.n_updates = 0

    def update(self, row):
        self.sketch[self.next_row] = row
        self.next_row += 1
        self.n_updates += 1
        if self.next_row >= 2 * self.ell:
            self._compress()

    def _compress(self):
        try:
            U, s, Vt = np.linalg.svd(self.sketch, full_matrices=False)
        except np.linalg.LinAlgError:
            return
        if len(s) > self.ell:
            delta = s[self.ell - 1] ** 2
            s_new = np.sqrt(np.maximum(s[:self.ell] ** 2 - delta, 0))
            self.sketch[:self.ell] = np.diag(s_new) @ Vt[:self.ell]
            self.sketch[self.ell:] = 0
            self.next_row = self.ell
        else:
            self.next_row = len(s)

    def get_covariance_approx(self):
        B = self.sketch[:self.next_row].copy()
        return B.T @ B

    def process_stream(self, X):
        for i in range(len(X)):
            self.update(X[i].astype(np.float64))

    def covariance_error(self, X):
        """Frobenius-norm error of covariance approximation, normalized."""
        X_f = X.astype(np.float64)
        X_c = X_f - X_f.mean(axis=0, keepdims=True)
        Cov_true = (X_c.T @ X_c) / len(X)
        Cov_sketch = self.get_covariance_approx() / max(self.n_updates, 1)
        err = np.linalg.norm(Cov_true - Cov_sketch, 'fro')
        norm = np.linalg.norm(Cov_true, 'fro')
        return err / max(norm, 1e-10)


# ===========================================================================
# Task 3: Streaming Classification (Online SGD with feature hashing)
# ===========================================================================
class OnlineSGDClassifier:
    """
    Online SGD binary classifier with bounded memory via feature hashing.
    Memory: O(hash_dim) regardless of input dimension.
    """

    def __init__(self, hash_dim=64, lr=0.01, seed=42):
        self.hash_dim = hash_dim
        self.memory = hash_dim
        self.w = np.zeros(hash_dim, dtype=np.float64)
        self.lr = lr
        rng = np.random.RandomState(seed)
        self.hash_a = int(rng.randint(1, 2**16))
        self.hash_b = int(rng.randint(0, 2**16))
        self.n_updates = 0

    def _hash_features(self, x):
        h = np.zeros(self.hash_dim, dtype=np.float64)
        for i, val in enumerate(x):
            if val:
                idx = ((self.hash_a * (i + 1) + self.hash_b) % 65521) % self.hash_dim
                sign = 2 * (((self.hash_a * (i + 1000) + self.hash_b) % 65521) % 2) - 1
                h[idx] += sign * val
        return h

    def predict_proba(self, x):
        h = self._hash_features(x)
        logit = np.dot(self.w, h)
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))

    def predict(self, x):
        return int(self.predict_proba(x) >= 0.5)

    def update(self, x, y):
        h = self._hash_features(x)
        p = self.predict_proba(x)
        grad = (p - y) * h
        self.w -= self.lr * grad
        self.n_updates += 1

    def process_stream(self, X, labels):
        for i in range(len(X)):
            self.update(X[i], labels[i])

    def evaluate(self, X_test, y_test):
        correct = 0
        for i in range(len(X_test)):
            if self.predict(X_test[i]) == y_test[i]:
                correct += 1
        return correct / len(X_test)


# ===========================================================================
# Task 3b: Averaged SGD (Polyak-Ruppert averaging) with feature hashing
# ===========================================================================
class AveragedSGDClassifier(OnlineSGDClassifier):
    """
    Online SGD with Polyak-Ruppert averaging and bounded memory via feature hashing.
    Maintains both a running weight vector w and a running average w_avg.
    Uses w_avg for prediction, w for gradient computation.
    Memory: O(hash_dim).
    """

    def __init__(self, hash_dim=64, lr=0.01, seed=42):
        super().__init__(hash_dim=hash_dim, lr=lr, seed=seed)
        self.w_avg = np.zeros(hash_dim, dtype=np.float64)

    def update(self, x, y):
        h = self._hash_features(x)
        # Compute gradient using w (not w_avg) for proper Polyak-Ruppert averaging
        logit = np.dot(self.w, h)
        p = 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))
        grad = (p - y) * h
        self.w -= self.lr * grad
        self.n_updates += 1
        t = self.n_updates
        self.w_avg = ((t - 1) * self.w_avg + self.w) / t

    def predict_proba(self, x):
        h = self._hash_features(x)
        logit = np.dot(self.w_avg, h)
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))

    def predict(self, x):
        return int(self.predict_proba(x) >= 0.5)
