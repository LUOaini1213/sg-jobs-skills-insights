"""Backtest simple forecasts of the quarterly PME job vacancy rate by industry.

Rolling origin: for every origin quarter in the test window, fit on data up to the origin and
forecast h = 1..4 quarters ahead. Error = mean absolute error in percentage points, averaged
over industries, origins and horizons. Methods are deliberately simple; the question is
whether anything beats "the rate stays where it is" (naive).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import load

H = 4


def series() -> dict[str, pd.Series]:
    q = load.vacancy_rate_quarterly()
    q = q[q["occupation"] == load.PME]
    cw = load.sector_crosswalk().dropna(subset=["mom_industry"])
    out = {}
    for ind in sorted(set(cw["mom_industry"])):
        s = q[q["industry"] == ind].set_index("quarter")["job_vacancy_rate"].sort_index()
        s = trailing_contiguous(s.astype(float))
        if len(s) >= 24:
            out[ind] = s
    return out


def trailing_contiguous(s: pd.Series) -> pd.Series:
    """Longest run of non-missing values that ends at the last observed quarter.

    Quarters are positional, so a gap (a '-' cell) must not be silently skipped.
    """
    s = s.loc[: s.last_valid_index()] if s.last_valid_index() is not None else s.iloc[:0]
    isna = s.isna().to_numpy()
    if not len(s):
        return s
    start = len(s) - int(np.argmax(isna[::-1])) if isna.any() else 0
    return s.iloc[start:]


def f_naive(y: np.ndarray, h: int) -> np.ndarray:
    return np.repeat(y[-1], h)


def f_seasonal_naive(y: np.ndarray, h: int) -> np.ndarray:
    return np.array([y[-4 + (i % 4)] for i in range(h)])


def f_mean4(y: np.ndarray, h: int) -> np.ndarray:
    return np.repeat(y[-4:].mean(), h)


def f_holt_damped(y: np.ndarray, h: int) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = ExponentialSmoothing(y, trend="add", damped_trend=True).fit()
    return np.asarray(m.forecast(h))


METHODS = {"naive": f_naive, "seasonal_naive": f_seasonal_naive, "mean_last4": f_mean4, "holt_damped": f_holt_damped}


def backtest(data: dict[str, pd.Series], n_origins: int = 12, h: int = H) -> pd.DataFrame:
    rows = []
    for ind, s in data.items():
        y = s.to_numpy()
        for o in range(len(y) - h - n_origins + 1, len(y) - h + 1):
            train, test = y[:o], y[o:o + h]
            for name, f in METHODS.items():
                err = np.abs(f(train, h) - test)
                for k, e in enumerate(err, 1):
                    rows.append({"industry": ind, "origin": s.index[o - 1], "method": name, "h": k, "abs_err": e})
    return pd.DataFrame(rows)


def run() -> dict:
    data = series()
    bt = backtest(data)
    mae = bt.groupby("method")["abs_err"].mean().sort_values()
    by_h = bt.pivot_table(index="h", columns="method", values="abs_err", aggfunc="mean").round(3)
    first = min(s.index[0] for s in data.values())
    last = max(s.index[-1] for s in data.values())
    return {
        "n_industries": len(data), "data_from": first, "data_to": last,
        "n_origins": 12, "horizon_quarters": H,
        "mae_pp": {k: round(float(v), 3) for k, v in mae.items()},
        "mae_pp_by_horizon": {m: {int(h): float(v) for h, v in by_h[m].items()} for m in by_h.columns},
    }
