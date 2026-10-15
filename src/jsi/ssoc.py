"""SSOC 2020 occupations, and SingStat's official Skills Framework job role -> SSOC mapping.

SingStat's *SSOC 2020 Alphabetical Index* lists Skills Framework job roles next to the
SSOC 5-digit code that SingStat assigned them (column "Singapore Skills Framework's Job
Roles*" = "x"). We use those rows ONLY as an evaluation reference. They are kept out of
the text the matcher sees, so the evaluation does not leak.
"""
from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from .load import RAW, _cached

DEFS = RAW / "ssoc2020_detailed_definitions.xlsx"
ALPHA = RAW / "ssoc2020_alpha_index.xlsx"

_ROLE_RE = re.compile(r"^(?P<title>.*?)\s*(?:\((?P<track>[^()]*)\))?\s*\(SFw-(?P<sector>[^()]*)\)\s*$")


def norm(s: str) -> str:
    """Lower-case, unify slashes/dashes/spaces; used for exact title joins."""
    s = str(s).lower().replace("–", "-").replace("’", "'")
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"[^a-z0-9/&()\-,.' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _clean(x) -> str:
    x = "" if pd.isna(x) else str(x)
    return "" if x.strip() in {"<Blank>", "nan"} else re.sub(r"[•\x9f\xa0]+", " ", x)


@lru_cache(maxsize=None)
def occupations() -> pd.DataFrame:
    """997 SSOC 2020 5-digit occupations with definition text and generic example jobs."""

    def build():
        df = pd.read_excel(DEFS, header=4, dtype=str)
        df.columns = ["code", "title", "groups", "definition", "tasks", "notes", "examples", "elsewhere"]
        df = df[df["code"].str.fullmatch(r"\d{5}", na=False)].copy()
        for c in ["title", "definition", "tasks", "examples"]:
            df[c] = df[c].map(_clean)
        alpha = alpha_index()
        generic = alpha[~alpha["is_sfw"]].groupby("code")["alpha"].apply(lambda s: "; ".join(s))
        df["alpha_titles"] = df["code"].map(generic).fillna("")
        return df[["code", "title", "definition", "tasks", "examples", "alpha_titles"]].reset_index(drop=True)

    return _cached("ssoc_occupations", build)


@lru_cache(maxsize=None)
def alpha_index() -> pd.DataFrame:
    def build():
        df = pd.read_excel(ALPHA, header=7, dtype=str)
        df.columns = ["code", "alpha", "sfw_flag", "sfw_id"]
        df = df[df["code"].str.fullmatch(r"\d{5}", na=False)].copy()
        df["is_sfw"] = df["sfw_flag"].fillna("").str.strip().str.lower().eq("x")
        return df[["code", "alpha", "is_sfw"]].reset_index(drop=True)

    return _cached("ssoc_alpha", build)


def parse_sfw_entry(text: str):
    """'Title (Track) (SFw-Sector)' -> (title, track, sector); track may be None."""
    m = _ROLE_RE.match(str(text).strip())
    if not m:
        return None
    return m.group("title").strip(), (m.group("track") or None), m.group("sector").strip()


@lru_cache(maxsize=None)
def official_crosswalk() -> pd.DataFrame:
    """Official SingStat SFw job role -> SSOC 2020 code, joined to the current SFw roles.

    Joined on (sector, normalised job-role title). A role with several official codes
    (same title in several tracks) keeps all of them as acceptable answers.
    """
    from .load import roles

    def build():
        a = alpha_index()
        a = a[a["is_sfw"]].copy()
        parsed = a["alpha"].map(parse_sfw_entry)
        a = a[parsed.notna()].copy()
        a["title"] = [p[0] for p in parsed.dropna()]
        a["sector_key"] = [norm(p[2]) for p in parsed.dropna()]
        a["title_key"] = a["title"].map(norm)
        r = roles().copy()
        r["sector_key"] = r["sector"].map(norm)
        r["title_key"] = r["job_role"].map(norm)
        j = r.merge(a[["sector_key", "title_key", "code"]], on=["sector_key", "title_key"], how="inner")
        out = j.groupby(["sector", "job_role"])["code"].apply(lambda s: ";".join(sorted(set(s)))).reset_index()
        return out.rename(columns={"code": "official_codes"})

    return _cached("official_crosswalk", build)


def wage_codes() -> pd.DataFrame:
    """MOM occupational-wage titles -> SSOC 2020 code (exact title match, then 4-digit unit group)."""
    from .load import wages

    w = wages().copy()
    occ = occupations()
    by_title = dict(zip(occ["title"].map(norm), occ["code"]))
    w["code"] = w["occupation"].map(norm).map(by_title)
    return w
