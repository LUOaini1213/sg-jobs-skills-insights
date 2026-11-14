"""Checks on the real public data. Skipped when data/raw/ has not been fetched
(run `python -m jsi.fetch` first); the CI reproduce job runs them."""
import json
from pathlib import Path

import pytest

from jsi import evaluate_mapping, load, ssoc

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not load.SFW.exists(), reason="raw data not fetched")
needs_ssoc = pytest.mark.skipif(not ssoc.DEFS.exists() or not ssoc.ALPHA.exists(),
                                reason="SingStat SSOC files not available (not reachable from CI runners)")


@needs_ssoc
def test_every_label_resolves_to_an_ssoc_code():
    g = evaluate_mapping.claude_label_codes()
    assert len(g) == 100
    assert (g["codes"].map(len) == 0).sum() == (g["acceptable_occupations"] == "NONE").sum() == 4
    occ = set(ssoc.occupations()["code"])
    for cs in g["codes"]:
        for c in cs:
            assert c in occ or any(o.startswith(c) for o in occ)


@needs_ssoc
def test_official_crosswalk_codes_are_real_occupations():
    cw = ssoc.official_crosswalk()
    occ = set(ssoc.occupations()["code"])
    assert len(cw) >= 1200
    assert all(c in occ for cs in cw["official_codes"] for c in cs.split(";"))


@needs_ssoc
def test_matcher_text_never_contains_the_official_sfw_entries():
    # leakage guard: the alphabetical-index rows flagged as SFw roles must not reach occ_text
    a = ssoc.alpha_index()
    sfw = set(a.loc[a["is_sfw"], "alpha"])
    text = " ".join(ssoc.occupations()["alpha_titles"])
    assert sfw and not any(s in text for s in sfw)
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
    assert m["requirements_mapped_share"] == round(float(rs["mapped"].mean()), 4)


def test_graph_matches_the_requirements_table():
    from jsi import graph

    rs = load.role_skills()
    g = graph.build(rs)
    m = json.loads((ROOT / "outputs" / "graph_metrics.json").read_text(encoding="utf-8"))["graph"]
    assert g.n_edges == len(rs) == m["edges"]
    assert g.B.shape == (m["roles"], m["skills"]) == (rs.groupby(["sector", "job_role"]).ngroups, rs["unique_skill"].nunique())
    assert m["roles_in_framework_without_skills"] == len(load.roles()) - m["roles"]


def test_committed_transitions_obey_their_own_rules():
    import pandas as pd

    from jsi import graph

    t = pd.read_csv(ROOT / "outputs" / "role_transitions.csv", keep_default_na=False)
    assert (t["share_of_target_held"].astype(float) >= graph.MIN_HELD).all()
    listed = t["skills_to_add"].map(lambda s: len(s.split("; ")) if s else 0)
    assert (listed == t["additional_skills"].astype(int)).all()
    assert not ((t["sector"] == t["to_sector"]) & (t["job_role"] == t["to_job_role"])).any()
    held = {(a, b): set(g["unique_skill"]) for (a, b), g in load.role_skills().groupby(["sector", "job_role"])}
    for r in t.sample(200, random_state=1).itertuples():
        src, dst = held[(r.sector, r.job_role)], held[(r.to_sector, r.to_job_role)]
        assert dst - src == (set(r.skills_to_add.split("; ")) if r.skills_to_add else set())
        assert len(src & dst) == int(r.shared_skills)
