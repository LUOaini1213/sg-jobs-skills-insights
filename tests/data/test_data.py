"""Checks on the real public data. Skipped when data/raw/ has not been fetched
(run `python -m jsi.fetch` first); the CI reproduce job runs them."""
import json
from pathlib import Path

import pytest

from jsi import evaluate_mapping, load, ssoc

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not load.SFW.exists(), reason="raw data not fetched")


def test_every_label_resolves_to_an_ssoc_code():
    g = evaluate_mapping.claude_label_codes()
    assert len(g) == 100
    assert (g["codes"].map(len) == 0).sum() == (g["acceptable_occupations"] == "NONE").sum() == 4
    occ = set(ssoc.occupations()["code"])
    for cs in g["codes"]:
        for c in cs:
            assert c in occ or any(o.startswith(c) for o in occ)


def test_official_crosswalk_codes_are_real_occupations():
    cw = ssoc.official_crosswalk()
    occ = set(ssoc.occupations()["code"])
    assert len(cw) >= 1200
    assert all(c in occ for cs in cw["official_codes"] for c in cs.split(";"))


def test_matcher_text_never_contains_the_official_sfw_entries():
    # leakage guard: the alphabetical-index rows flagged as SFw roles must not reach occ_text
    a = ssoc.alpha_index()
    sfw = set(a.loc[a["is_sfw"], "alpha"])
    text = " ".join(ssoc.occupations()["alpha_titles"])
    assert not any(s in text for s in list(sfw)[:300])
    assert "(SFw-" not in text


def test_role_skills_unique_and_flags_from_ssg():
    rs = load.role_skills()
    assert not rs.duplicated(["sector", "job_role", "unique_skill"]).any()
    assert 0.1 < rs["emerging"].mean() < 0.3


def test_crosswalk_covers_every_sector_once():
    cw = load.sector_crosswalk()
    assert set(cw["sector"]) == set(load.roles()["sector"]) and cw["sector"].is_unique


def test_committed_counts_match_data():
    m = json.loads((ROOT / "outputs" / "metrics.json").read_text(encoding="utf-8"))["data"]
    rs = load.role_skills()
    assert m["sfw_roles"] == len(load.roles())
    assert m["role_skill_requirements"] == len(rs)
    assert m["unique_skills_required"] == rs["unique_skill"].nunique()
