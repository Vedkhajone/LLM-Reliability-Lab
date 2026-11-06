"""Statistical utilities. Every function documents its estimand.

Dependence note: repeated runs of the same case and the six conditions of one base question
are NOT independent. Rates therefore use a *cluster bootstrap* (resample cases, keep all
of a case's rows together). Paired model comparisons are made at the item level after
collapsing repeats by majority vote.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------- proportions ----

def wilson_ci(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (assumes independent trials)."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(0.5 + level / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cluster_bootstrap_ci(
    values: Sequence[float],
    clusters: Sequence,
    stat=np.mean,
    level: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for `stat(values)`, resampling whole clusters (e.g. cases)."""
    v = np.asarray(values, dtype=float)
    c = np.asarray(clusters)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    uniq = pd.unique(c)
    groups = [v[c == u] for u in uniq]
    if len(groups) == 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(groups), size=(n_resamples, len(groups)))
    est = np.empty(n_resamples)
    for b in range(n_resamples):
        est[b] = stat(np.concatenate([groups[i] for i in idx[b]]))
    a = (1 - level) / 2
    return (float(np.quantile(est, a)), float(np.quantile(est, 1 - a)))


def rate_with_ci(
    values: Sequence[bool | float],
    clusters: Sequence | None = None,
    level: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Mean of a 0/1 (or numeric) outcome with a CI that respects clustering.

    One row per cluster -> Wilson interval (binary only); otherwise cluster bootstrap.
    """
    vals = list(values)
    cls = list(range(len(vals))) if clusters is None else list(clusters)
    if len(cls) != len(vals):
        raise ValueError("values and clusters must have equal length")
    pairs = [
        (float(x), c) for x, c in zip(vals, cls)
        if x is not None and not (isinstance(x, float) and math.isnan(x))
    ]
    n = len(pairs)
    if n == 0:
        return {"value": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    v = np.asarray([p[0] for p in pairs])
    cl = np.asarray([p[1] for p in pairs], dtype=object)
    mean = float(v.mean())
    binary = set(np.unique(v)).issubset({0.0, 1.0})
    one_per_cluster = len(set(cl.tolist())) == n
    if binary and one_per_cluster:
        lo, hi = wilson_ci(int(v.sum()), n, level)
    else:
        lo, hi = cluster_bootstrap_ci(v, cl, level=level, n_resamples=n_resamples, seed=seed)
    return {"value": mean, "ci_low": lo, "ci_high": hi, "n": n}


# ------------------------------------------------------------ paired comparisons ----

def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> dict[str, float]:
    """Exact McNemar test for paired binary outcomes (two-sided binomial on discordant pairs).

    n10 = a correct & b wrong; n01 = a wrong & b correct. H0: P(n10) = P(n01) = 0.5.
    Also returns the paired odds ratio n10/n01 (inf if n01=0) and the accuracy difference.
    """
    a_ = np.asarray(a, dtype=bool)
    b_ = np.asarray(b, dtype=bool)
    if len(a_) != len(b_):
        raise ValueError("paired samples must have equal length")
    n10 = int((a_ & ~b_).sum())
    n01 = int((~a_ & b_).sum())
    disc = n10 + n01
    p = 1.0 if disc == 0 else float(stats.binomtest(n10, disc, 0.5).pvalue)
    return {
        "n": len(a_), "n10": n10, "n01": n01, "p_value": p,
        "acc_a": float(a_.mean()) if len(a_) else float("nan"),
        "acc_b": float(b_.mean()) if len(b_) else float("nan"),
        "diff": float(a_.mean() - b_.mean()) if len(a_) else float("nan"),
        "odds_ratio": (n10 / n01) if n01 else (float("inf") if n10 else float("nan")),
    }


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for a difference between two proportions: 2*asin(sqrt p1) - 2*asin(sqrt p2)."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], clusters: Sequence | None = None,
    level: float = 0.95, n_resamples: int = 2000, seed: int = 0,
) -> dict[str, float]:
    """Cluster-bootstrap CI for mean(a) - mean(b) over paired rows."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    cl = np.arange(len(d)) if clusters is None else np.asarray(list(clusters))
    lo, hi = cluster_bootstrap_ci(d, cl, level=level, n_resamples=n_resamples, seed=seed)
    return {"diff": float(d.mean()) if len(d) else float("nan"), "ci_low": lo, "ci_high": hi, "n": len(d)}


def holm_correction(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls family-wise error rate)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def majority_vote(df: pd.DataFrame, item_cols: list[str], col: str = "passed") -> pd.DataFrame:
    """Collapse repeats to one binary outcome per item (strict majority; ties count as failure)."""
    g = df.groupby(item_cols, dropna=False)[col].agg(lambda s: float(np.mean(s.astype(float))) > 0.5)
    return g.reset_index()


def compare_models_paired(
    scored: pd.DataFrame, model_a: str, model_b: str, col: str = "passed",
) -> dict[str, float] | None:
    """McNemar comparison of two models on the items both answered (repeats -> majority vote)."""
    cols = ["experiment", "item_id"]
    sub = scored[scored["model_key"].isin([model_a, model_b])]
    mv = majority_vote(sub.assign(_m=sub["model_key"]), cols + ["model_key"], col)
    wide = mv.pivot_table(index=cols, columns="model_key", values=col, aggfunc="first").dropna()
    if model_a not in wide or model_b not in wide or len(wide) == 0:
        return None
    res = mcnemar_exact(wide[model_a].astype(bool), wide[model_b].astype(bool))
    res["cohens_h"] = cohens_h(min(max(res["acc_a"], 0), 1), min(max(res["acc_b"], 0), 1))
    return res


# ------------------------------------------------------- agreement & calibration ----

def cohens_kappa(labels_a: Sequence, labels_b: Sequence) -> float:
    """Cohen's kappa for two annotators (e.g. human vs. automatic label, or human vs. human)."""
    from sklearn.metrics import cohen_kappa_score

    if len(labels_a) != len(labels_b) or len(labels_a) == 0:
        return float("nan")
    return float(cohen_kappa_score(labels_a, labels_b))


def brier_score(confidences: Sequence[float], outcomes: Sequence[bool]) -> float:
    """Mean squared error between stated confidence and 0/1 correctness."""
    c = np.asarray(confidences, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    return float(np.mean((c - o) ** 2)) if len(c) else float("nan")


def expected_calibration_error(
    confidences: Sequence[float], outcomes: Sequence[bool], n_bins: int = 5
) -> float:
    """ECE with equal-width bins: sum_b (n_b/N) * |acc_b - conf_b|. Unreliable for small N."""
    c = np.asarray(confidences, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    if len(c) == 0:
        return float("nan")
    bins = np.minimum((c * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(o[mask].mean() - c[mask].mean())
    return float(ece)
