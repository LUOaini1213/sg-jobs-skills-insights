"""Load the raw public files into tidy pandas tables (cached under data/interim/)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
REF = ROOT / "data" / "reference"

SFW = RAW / "skills_framework_dataset.xlsx"
UNIQUE = RAW / "unique_skills_list.xlsx"
TSCMAP = RAW / "tsc_to_unique_skills_mapping.xlsx"
VAC_N = RAW / "d_889d11a2b0a53b235abb64e3f4e0a47b.csv"
VAC_RATE = RAW / "d_ddcb2546c798f94ba6b4a6508ab42632.csv"
VAC_RATE_Q = RAW / "d_74803e94c5299493f76d71c8994c50c1.csv"
WAGES = RAW / "d_9917e751f7498502f70052a940a3f312.csv"

PME = "professionals, managers and executives"
NON_PME = "non-professionals, managers and executives"


def _cached(name: str, build):
    INTERIM.mkdir(parents=True, exist_ok=True)
    p = INTERIM / f"{name}.pkl"
    if p.exists():
        return pd.read_pickle(p)
    df = build()
    df.to_pickle(p)
    return df


def to_number(s: pd.Series) -> pd.Series:
    """MOM tables use '-' for nil/negligible; treat as 0. Other non-numerics -> NaN."""
    s = s.astype(str).str.strip().str.replace(",", "", regex=False)
    s = s.where(s != "-", "0")
    return pd.to_numeric(s, errors="coerce")


@lru_cache(maxsize=None)
def roles() -> pd.DataFrame:
    """One row per (sector, job role). Tracks are concatenated."""

    def build():
        df = pd.read_excel(SFW, sheet_name="Job Role_Description")
        df.columns = ["sector", "track", "job_role", "description", "performance_expectation"]
        for c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace("nan", "")
        g = df.groupby(["sector", "job_role"], sort=True)
        out = g.agg(
            track=("track", lambda s: "; ".join(sorted(set(s)))),
            description=("description", "first"),
        ).reset_index()
        return out

    return _cached("roles", build)


@lru_cache(maxsize=None)
def tsc_to_unique() -> pd.DataFrame:
    """Sector TSC code -> unique skill with SSG's Emerging / CASL flags (deduplicated)."""

    def build():
        m = pd.read_excel(TSCMAP, sheet_name="data")
        m = m.rename(
            columns={
                "skills_framework_skill_code": "tsc_code",
                "Unique skill_updated_skill_title": "unique_skill",
                "Unique skill_updated_skill_SFS_status": "emerging",
                "Unique skill_updated_CASL_status": "casl",
            }
        )[["tsc_code", "unique_skill", "emerging", "casl"]]
        m = m[m["unique_skill"].notna() & (m["emerging"].astype(str) != "No mapping")]
        for c in ["emerging", "casl"]:
            m[c] = m[c].astype(str).str.upper().eq("TRUE")
        return m.drop_duplicates(["tsc_code", "unique_skill"]).reset_index(drop=True)

    return _cached("tsc_to_unique", build)


@lru_cache(maxsize=None)
def unique_skills() -> pd.DataFrame:
    def build():
        u = pd.read_excel(UNIQUE, sheet_name="Unique Skills List")
        u = u.rename(columns={"skill_title": "unique_skill", "Emerging Skills": "emerging", "CASL Skills": "casl"})
        u["emerging"] = u["emerging"].astype(str).str.upper().eq("TRUE")
        u["casl"] = u["casl"].astype(str).str.upper().eq("TRUE")
        return u[["skill_id", "unique_skill", "skill_description", "skill_type", "emerging", "casl"]]

    return _cached("unique_skills", build)


@lru_cache(maxsize=None)
def role_skills() -> pd.DataFrame:
    """(sector, job_role, unique_skill) with max proficiency level and flags.

    A sector TSC maps to one or more unique skills; a role keeps a unique skill once.
    """

    def build():
        r = pd.read_excel(SFW, sheet_name="Job Role_TSC_CCS")
        r.columns = ["sector", "track", "job_role", "tsc_title", "type", "level", "tsc_code"]
        for c in ["sector", "job_role", "tsc_title", "type", "tsc_code"]:
            r[c] = r[c].astype(str).str.strip()
        r["level"] = pd.to_numeric(r["level"], errors="coerce")
        m = tsc_to_unique()
        j = r.merge(m, on="tsc_code", how="left")
        # CCS (critical core skills) and the few unmapped TSCs keep their own title
        j["mapped"] = j["unique_skill"].notna()
        j["unique_skill"] = j["unique_skill"].fillna(j["tsc_title"])
        j["emerging"] = j["emerging"].fillna(False).astype(bool)
        j["casl"] = j["casl"].fillna(False).astype(bool)
        out = (
            j.groupby(["sector", "job_role", "unique_skill"], sort=True)
            .agg(type=("type", "first"), level=("level", "max"), emerging=("emerging", "max"), casl=("casl", "max"), mapped=("mapped", "max"))
            .reset_index()
        )
        return out

    return _cached("role_skills", build)


def vacancies() -> pd.DataFrame:
    df = pd.read_csv(VAC_N)
    df["job_vacancy"] = to_number(df["job_vacancy"])
    return df


def vacancy_rate() -> pd.DataFrame:
    df = pd.read_csv(VAC_RATE)
    df["job_vacancy_rate"] = to_number(df["job_vacancy_rate"])
    return df


def vacancy_rate_quarterly() -> pd.DataFrame:
    df = pd.read_csv(VAC_RATE_Q)
    df["job_vacancy_rate"] = to_number(df["job_vacancy_rate"])
    return df


def wages() -> pd.DataFrame:
    df = pd.read_csv(WAGES)
    df = df.rename(columns={"occ_desc": "occupation", "mthly_gross_wage_50_pctile": "median_gross_wage"})
    df["median_gross_wage"] = to_number(df["median_gross_wage"])
    return df[["occupation", "median_gross_wage"]].dropna().reset_index(drop=True)


def sector_crosswalk() -> pd.DataFrame:
    return pd.read_csv(REF / "sfw_sector_to_mom_industry.csv")
