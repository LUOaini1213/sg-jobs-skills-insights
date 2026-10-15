"""Is a role's emerging-skill intensity associated with the pay of its occupation?

Unit: Skills Framework job role with an OFFICIAL SingStat SSOC code (matcher output is not
used here, to keep mapping error out of the estimate). Outcome: log median gross monthly wage
of that SSOC occupation (MOM Occupational Wages, June 2024, residents, full-time).
Several roles share one occupation wage, so standard errors are clustered by SSOC code.
This is an association across occupations, not a causal skill premium.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from . import load, ssoc


def role_table() -> pd.DataFrame:
    rs = load.role_skills()
    per = rs.groupby(["sector", "job_role"]).agg(
        n_skills=("unique_skill", "nunique"),
        n_emerging=("emerging", "sum"),
        max_level=("level", "max"),
    ).reset_index()
    per["emerging_share"] = per["n_emerging"] / per["n_skills"]
    off = ssoc.official_crosswalk()
    off = off[~off["official_codes"].str.contains(";")]  # unambiguous codes only
    w = ssoc.wage_codes().dropna(subset=["code"])[["code", "median_gross_wage"]]
    t = per.merge(off, on=["sector", "job_role"]).rename(columns={"official_codes": "code"})
    t = t.merge(w, on="code", how="inner")
    t["log_wage"] = np.log(t["median_gross_wage"])
    t["major"] = t["code"].str[0]
    t["submajor"] = t["code"].str[:2]
    return t


def fit(t: pd.DataFrame, formula: str):
    groups = pd.factorize(t["code"])[0]
    return smf.ols(formula, data=t).fit(cov_type="cluster", cov_kwds={"groups": groups})


def run() -> dict:
    t = role_table()
    specs = {
        "raw": "log_wage ~ emerging_share",
        "plus_controls": "log_wage ~ emerging_share + np.log(n_skills) + max_level + C(sector)",
        "within_submajor": "log_wage ~ emerging_share + np.log(n_skills) + max_level + C(sector) + C(submajor)",
    }
    out = {"n_roles": int(len(t)), "n_occupations": int(t["code"].nunique()), "models": {}}
    for k, f in specs.items():
        m = fit(t, f)
        b = m.params["emerging_share"]
        lo, hi = m.conf_int().loc["emerging_share"]
        # effect of +10 percentage points of emerging share, in % wage
        out["models"][k] = {
            "formula": f,
            "coef": round(float(b), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "pct_wage_per_10pp": round(float(100 * (np.exp(0.1 * b) - 1)), 2),
            "pct_ci95_per_10pp": [round(float(100 * (np.exp(0.1 * lo) - 1)), 2),
                                  round(float(100 * (np.exp(0.1 * hi) - 1)), 2)],
            "p_value": float(m.pvalues["emerging_share"]),
            "r2": round(float(m.rsquared), 3),
        }
    return out
