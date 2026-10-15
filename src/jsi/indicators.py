"""Jobs-skills indicators: sector skill profiles, labour demand, vacancy-weighted skill demand."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import load

BASE_YEAR = 2019  # last pre-COVID year
LATEST_YEAR = 2025


def sector_skill_profile(rs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per Skills Framework sector: roles, skills per role, share of roles needing an emerging skill."""
    rs = load.role_skills() if rs is None else rs
    per_role = rs.groupby(["sector", "job_role"]).agg(
        n_skills=("unique_skill", "nunique"),
        n_emerging=("emerging", "sum"),
        n_casl=("casl", "sum"),
    )
    per_role["has_emerging"] = per_role["n_emerging"] > 0
    out = per_role.groupby("sector").agg(
        roles=("n_skills", "size"),
        median_skills_per_role=("n_skills", "median"),
        share_roles_with_emerging=("has_emerging", "mean"),
        emerging_share_of_requirements=("n_emerging", "sum"),
        casl_share_of_requirements=("n_casl", "sum"),
        requirements=("n_skills", "sum"),
    )
    out["emerging_share_of_requirements"] /= out["requirements"]
    out["casl_share_of_requirements"] /= out["requirements"]
    return out.drop(columns="requirements").reset_index()


def industry_vacancies(v: pd.DataFrame | None = None) -> pd.DataFrame:
    """Annual vacancies per MOM industry: total (PME + non-PME) and PME."""
    v = load.vacancies() if v is None else v
    v = v[v["occupation"].isin([load.PME, load.NON_PME])]
    p = v.pivot_table(index=["industry", "year"], columns="occupation", values="job_vacancy", aggfunc="sum")
    p = p.rename(columns={load.PME: "pme", load.NON_PME: "non_pme"}).fillna(0)
    p["total"] = p["pme"] + p["non_pme"]
    return p.reset_index()


def growth(a: float, b: float) -> float:
    """Percentage change from a to b; NaN when the base is zero."""
    return float("nan") if a == 0 else 100.0 * (b - a) / a


def sector_demand(base: int = BASE_YEAR, latest: int = LATEST_YEAR) -> pd.DataFrame:
    cw = load.sector_crosswalk()
    iv = industry_vacancies()
    rate = load.vacancy_rate()
    rate = rate[(rate["year"] == latest) & (rate["occupation"] == load.PME)][["industry", "job_vacancy_rate"]]
    wide = iv.pivot_table(index="industry", columns="year", values=["total", "pme"])
    rows = []
    for _, r in cw.iterrows():
        ind = r["mom_industry"]
        if pd.isna(ind):
            rows.append({"sector": r["sector"], "mom_industry": None, "fit": r["fit"]})
            continue
        t0, t1 = wide.loc[ind, ("total", base)], wide.loc[ind, ("total", latest)]
        p0, p1 = wide.loc[ind, ("pme", base)], wide.loc[ind, ("pme", latest)]
        vr = rate.loc[rate["industry"] == ind, "job_vacancy_rate"]
        rows.append({
            "sector": r["sector"], "mom_industry": ind, "fit": r["fit"],
            f"vacancies_{latest}": t1, f"pme_vacancies_{latest}": p1,
            f"pme_share_{latest}": (p1 / t1) if t1 else np.nan,
            f"vacancy_growth_{base}_{latest}_pct": growth(t0, t1),
            f"pme_vacancy_growth_{base}_{latest}_pct": growth(p0, p1),
            f"pme_vacancy_rate_{latest}": float(vr.iloc[0]) if len(vr) else np.nan,
        })
    return pd.DataFrame(rows)


def skill_demand_index(base: int = BASE_YEAR, latest: int = LATEST_YEAR,
                       rs: pd.DataFrame | None = None, iv: pd.DataFrame | None = None,
                       cw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Vacancy-weighted demand per unique skill.

    For each MOM industry i, share_{i,s} = share of the Skills Framework roles mapped to i
    that require skill s (each role weighted equally). Then
        index_s(y) = sum_i vacancies_i(y) * share_{i,s}
    The index moves only through the industry mix of vacancies; it assumes the role mix
    inside an industry is fixed. Read it as "vacancies in industries where this skill is
    common", not as a count of postings that mention the skill.
    """
    rs = load.role_skills() if rs is None else rs
    iv = industry_vacancies() if iv is None else iv
    cw = load.sector_crosswalk() if cw is None else cw
    cw = cw.dropna(subset=["mom_industry"])
    r = rs.merge(cw[["sector", "mom_industry"]], on="sector", how="inner")
    n_roles = r.groupby("mom_industry").apply(lambda g: g[["sector", "job_role"]].drop_duplicates().shape[0])
    share = (
        r.drop_duplicates(["mom_industry", "sector", "job_role", "unique_skill"])
        .groupby(["mom_industry", "unique_skill"]).size()
        .div(n_roles, level="mom_industry")
        .rename("share").reset_index()
    )
    out = None
    for y in (base, latest):
        vy = iv[iv["year"] == y].set_index("industry")["total"]
        s = share.assign(v=share["mom_industry"].map(vy).fillna(0))
        s = (s["share"] * s["v"]).groupby(s["unique_skill"]).sum().rename(f"index_{y}")
        out = s.to_frame() if out is None else out.join(s, how="outer")
    out = out.fillna(0)
    out[f"growth_{base}_{latest}_pct"] = np.where(
        out[f"index_{base}"] > 0, 100 * (out[f"index_{latest}"] / out[f"index_{base}"] - 1), np.nan)
    flags = rs.groupby("unique_skill").agg(emerging=("emerging", "max"), casl=("casl", "max"),
                                           sectors=("sector", "nunique"),
                                           roles=("job_role", "size"))
    return out.join(flags).reset_index().sort_values(f"index_{latest}", ascending=False)
