"""Sample-size arithmetic for an A/B test of a skills recommendation.

The figures in docs/experiment_design.md come from `run()`. The baseline rate, the effect
worth detecting, the cluster size and the intra-cluster correlation are ASSUMPTIONS chosen to
show the arithmetic. No conversion data are public, so none of them is an estimate.

    python -m jsi.power
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from scipy.stats import chi2, norm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"

ALPHA, POWER = 0.05, 0.80
# Worked example (assumptions, see docs/experiment_design.md)
BASELINE = 0.04        # share of users in control who enrol in a relevant course within 90 days
LIFT_PP = 0.01         # smallest absolute lift worth acting on: 4% -> 5%
CLUSTER_SIZE = 20      # users per employer, if randomising by employer
ICC = 0.02             # intra-cluster correlation of the outcome
WEEKLY_USERS = 5000    # eligible users reaching the recommendation page per week


def n_per_arm(p1: float, p2: float, alpha: float = ALPHA, power: float = POWER) -> int:
    """Users per arm for a two-sided two-proportion z-test (equal arms).

    n = (z_{1-alpha/2} * sqrt(2 * pbar * (1 - pbar)) + z_{power} * sqrt(p1 (1 - p1) + p2 (1 - p2)))^2 / (p2 - p1)^2
    """
    if p1 == p2:
        raise ValueError("p1 and p2 must differ")
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return math.ceil(num ** 2 / (p2 - p1) ** 2)


def power_at(n: int, p1: float, p2: float, alpha: float = ALPHA) -> float:
    """Power of the same test with n users per arm."""
    za = norm.ppf(1 - alpha / 2)
    pbar = (p1 + p2) / 2
    se0 = math.sqrt(2 * pbar * (1 - pbar) / n)
    se1 = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
    d = abs(p2 - p1)
    return float(norm.cdf((d - za * se0) / se1) + norm.cdf((-d - za * se0) / se1))


def design_effect(cluster_size: float, icc: float) -> float:
    """Variance inflation when whole clusters are randomised: 1 + (m - 1) * ICC."""
    return 1.0 + (cluster_size - 1.0) * icc


def mde(n: int, p1: float, alpha: float = ALPHA, power: float = POWER) -> float:
    """Smallest absolute lift detectable with n users per arm (bisection on power_at)."""
    lo, hi = 0.0, 1.0 - p1
    for _ in range(60):
        mid = (lo + hi) / 2
        if mid == 0 or power_at(n, p1, p1 + mid, alpha) < power:
            lo = mid
        else:
            hi = mid
    return hi


def srm_p_value(n_a: int, n_b: int, expected_share_a: float = 0.5) -> float:
    """Sample-ratio-mismatch check: chi-square p-value that the split matches the design."""
    n = n_a + n_b
    ea, eb = n * expected_share_a, n * (1 - expected_share_a)
    stat = (n_a - ea) ** 2 / ea + (n_b - eb) ** 2 / eb
    return float(chi2.sf(stat, 1))


def run(write: bool = True) -> dict:
    p1, p2 = BASELINE, BASELINE + LIFT_PP
    n = n_per_arm(p1, p2)
    deff = design_effect(CLUSTER_SIZE, ICC)
    n_cluster = math.ceil(n * deff)
    grid = []
    for base in (0.02, 0.04, 0.08):
        for rel in (0.10, 0.25, 0.50):
            grid.append({"baseline": base, "relative_lift": rel, "n_per_arm": n_per_arm(base, base * (1 + rel))})
    res = {
        "assumptions": {"alpha_two_sided": ALPHA, "power": POWER, "baseline_rate": p1, "target_rate": p2,
                        "cluster_size": CLUSTER_SIZE, "icc": ICC, "weekly_eligible_users": WEEKLY_USERS},
        "n_per_arm": n, "n_total": 2 * n,
        "achieved_power": round(power_at(n, p1, p2), 4),
        "weeks_at_assumed_traffic": math.ceil(2 * n / WEEKLY_USERS),
        "design_effect": round(deff, 2),
        "n_per_arm_cluster_randomised": n_cluster,
        "clusters_per_arm": math.ceil(n_cluster / CLUSTER_SIZE),
        "mde_pp_if_half_the_sample": round(100 * mde(n // 2, p1), 2),
        "sensitivity": grid,
    }
    if write:
        OUT.mkdir(exist_ok=True)
        (OUT / "experiment_power.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8", newline="\n")
    return res


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
