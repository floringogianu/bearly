"""Self-contained, seeded reimplementations of rliable's computations.

Used as test oracles. Copied from rliable==1.2.0 semantics (upstream master):

- stratified bootstrap resamples runs independently per task, with a fresh
  draw of *task* indices as well when ``task_bootstrap=True``
- percentile CIs at alpha and 1-alpha of the bootstrap distribution
- paired intervals use ``StratifiedIndependentBootstrap``-style resampling:
  both algorithms are resampled with the *same* draw of task indices (paired
  across tasks) but independent run indices

The point estimates of ``aggregate_*`` / ``probability_of_improvement`` are
assumed correct and cross-checked directly against the ``rliable`` package in
``test_stat_parity.py``; only the bootstrap machinery needs a seeded local
copy (rliable's classes offer no seedable standalone API).
"""

import numpy as np
from scipy.stats import mannwhitneyu, trim_mean


def rliable_iqm(matrix: np.ndarray) -> float:
    return float(trim_mean(matrix, proportiontocut=0.25, axis=None))


def rliable_mean(matrix: np.ndarray) -> float:
    return float(np.mean(np.mean(matrix, axis=0)))


def rliable_median(matrix: np.ndarray) -> float:
    return float(np.median(np.mean(matrix, axis=0)))


def rliable_optimality_gap(matrix: np.ndarray, gamma: float = 1.0) -> float:
    return float(gamma - np.mean(np.minimum(matrix, gamma)))


def rliable_poi(x_matrix: np.ndarray, y_matrix: np.ndarray) -> float:
    """P(X > Y) averaged over columns (tasks), rliable metrics.py semantics."""
    probs = []
    for t in range(x_matrix.shape[1]):
        xcol, ycol = x_matrix[:, t], y_matrix[:, t]
        if np.array_equal(xcol, ycol):
            probs.append(0.5)
            continue
        u_stat, _ = mannwhitneyu(xcol, ycol, alternative="greater")
        probs.append(u_stat / (len(xcol) * len(ycol)))
    return float(np.mean(probs))


def rliable_perf_profile(matrix: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """use_score_distribution=True: fraction of ALL runs x task entries > tau."""
    return np.array([np.mean(matrix > tau) for tau in taus])


def rliable_avg_perf_profile(matrix: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """use_score_distribution=False: fraction of task means > tau."""
    task_means = np.mean(matrix, axis=0)
    return np.array([np.mean(task_means > tau) for tau in taus])


def rliable_boot_ci(
    matrix: np.ndarray,
    agg_fn,
    reps: int,
    confidence: float,
    seed: int,
    task_bootstrap: bool = False,
) -> tuple[float, np.ndarray]:
    """Seeded copy of rliable's StratifiedBootstrap + percentile conf_int."""
    rng = np.random.RandomState(seed)
    num_runs, num_tasks = matrix.shape
    estimates = np.empty(reps)
    for b in range(reps):
        run_idx = rng.randint(0, num_runs, size=(num_runs, num_tasks))
        if task_bootstrap:
            task_idx = rng.randint(0, num_tasks, size=num_tasks)
            estimates[b] = agg_fn(matrix[run_idx, task_idx])
        else:
            estimates[b] = agg_fn(matrix[run_idx, np.arange(num_tasks)])
    point = agg_fn(matrix)
    alpha = (1.0 - confidence) / 2.0
    ci = np.quantile(estimates, [alpha, 1.0 - alpha], axis=0)
    return point, ci


def rliable_paired_boot_ci(
    x_matrix: np.ndarray,
    y_matrix: np.ndarray,
    paired_fn,
    reps: int,
    confidence: float,
    seed: int,
) -> tuple[float, np.ndarray]:
    """Seeded copy of StratifiedIndependentBootstrap paired-interval semantics.

    Task indices are drawn ONCE per replicate and applied to both X and Y
    (paired oversampling of tasks); run indices are drawn independently.
    """
    rng = np.random.RandomState(seed)
    nx, nt = x_matrix.shape
    ny, _ = y_matrix.shape
    estimates = np.empty(reps)
    for b in range(reps):
        task_idx = rng.randint(0, nt, size=nt)
        x_idx = rng.randint(0, nx, size=(nx, nt))
        y_idx = rng.randint(0, ny, size=(ny, nt))
        cols = np.arange(nt)
        estimates[b] = paired_fn(
            x_matrix[x_idx, task_idx][:, cols], y_matrix[y_idx, task_idx][:, cols]
        )
    point = paired_fn(x_matrix, y_matrix)
    alpha = (1.0 - confidence) / 2.0
    ci = np.quantile(estimates, [alpha, 1.0 - alpha], axis=0)
    return point, ci
