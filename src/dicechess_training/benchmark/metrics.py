"""Row-weighted binary scores with paired, whole-group bootstrap uncertainty."""

import numpy as np

EPSILON = 1e-15


def _arrays(target, probability):
    y, p = np.asarray(target, dtype=float), np.asarray(probability, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or not len(y):
        raise ValueError("scores require nonempty, aligned vectors")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("scores require finite values")
    if not np.isin(y, [0, 1]).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("binary targets and probabilities in [0,1] required")
    return y, p


def losses(target, probability):
    y, p = _arrays(target, probability)
    clipped = np.clip(p, EPSILON, 1 - EPSILON)
    return -y * np.log(clipped) - (1 - y) * np.log1p(-clipped), (p - y) ** 2


def scores(target, probability, bins=10):
    y, p = _arrays(target, probability)
    ll, brier = losses(y, p)
    index = np.minimum((p * bins).astype(int), bins - 1)
    table = []
    ece = 0.0
    for i in range(bins):
        selected = index == i
        n = int(selected.sum())
        pred = float(p[selected].mean()) if n else None
        observed = float(y[selected].mean()) if n else None
        if n:
            ece += n / len(y) * abs(pred - observed)
        table.append(
            {
                "lower": i / bins,
                "upper": (i + 1) / bins,
                "count": n,
                "prediction": pred,
                "observed": observed,
            }
        )
    return {
        "count": len(y),
        "prevalence": float(y.mean()),
        "log_loss": float(ll.mean()),
        "brier": float(brier.mean()),
        "ece": ece,
        "calibration": table,
    }


def confidence(target, probability, groups, reference=None, *, repeats=1000, seed=13):
    """Percentile 95% CIs. Resample groups, preserving all rows and their multiplicity.

    Fixed-bin sufficient statistics keep memory O(groups * bins), independent of repeats.
    Candidate/reference use the same draws. A single group has no estimable interval.
    """
    if not isinstance(repeats, int) or not 1 <= repeats <= 10000:
        raise ValueError("bootstrap repeats must be an integer in [1,10000]")
    y, p = _arrays(target, probability)
    if len(groups) != len(y):
        raise ValueError("groups must align with targets")
    names, inverse = np.unique(groups, return_inverse=True)
    if len(names) < 2:
        return {
            "method": "group-bootstrap-percentile-95",
            "groups": len(names),
            "intervals": None,
            "reason": "fewer-than-two-groups",
        }
    ll, brier = losses(y, p)
    values = np.column_stack([np.ones(len(y)), y, ll, brier])
    bins = np.minimum((p * 10).astype(int), 9)
    onehot = np.eye(10)[bins]
    values = np.column_stack([values, onehot, onehot * p[:, None], onehot * y[:, None]])
    keys = ["prevalence", "log_loss", "brier", "ece"]
    if reference is not None:
        rll, rbrier = losses(y, reference)
        values = np.column_stack([values, ll - rll, brier - rbrier])
        keys += ["log_loss_delta", "brier_delta"]
    totals = np.zeros((len(names), values.shape[1]))
    np.add.at(totals, inverse, values)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repeats):
        s = totals[rng.integers(len(names), size=len(names))].sum(axis=0)
        row = [s[1] / s[0], s[2] / s[0], s[3] / s[0], np.abs(s[14:24] - s[24:34]).sum() / s[0]]
        if reference is not None:
            row += [s[34] / s[0], s[35] / s[0]]
        samples.append(row)
    bounds = np.quantile(samples, [0.025, 0.975], axis=0)
    return {
        "method": "group-bootstrap-percentile-95",
        "groups": len(names),
        "intervals": {key: bounds[:, i].tolist() for i, key in enumerate(keys)},
    }
