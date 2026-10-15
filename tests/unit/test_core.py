"""Unit tests on small hand-computed inputs (no downloads). Values are chosen to be
asymmetric and non-degenerate so that a wrong implementation gives a different number."""
import math

import numpy as np
import pandas as pd
import pytest

from jsi import forecast, indicators, load, match, ssoc, wages


# ---------- load ----------
def test_to_number_counts_and_rates():
    s = pd.Series(["1,200", "-", "x", " 3.5"])
    out = load.to_number(s)
    assert out.iloc[0] == 1200 and out.iloc[1] == 0 and math.isnan(out.iloc[2]) and out.iloc[3] == 3.5
    rate = load.to_number(s, dash=np.nan)
    assert math.isnan(rate.iloc[1]) and rate.iloc[0] == 1200


# ---------- ssoc ----------
def test_parse_sfw_entry_with_and_without_track():
    assert ssoc.parse_sfw_entry("Tour Leader (Travel Management and Operations) (SFw-Tourism)") == (
        "Tour Leader", "Travel Management and Operations", "Tourism")
    assert ssoc.parse_sfw_entry("Managing Director (SFw-Landscape)") == ("Managing Director", None, "Landscape")
    assert ssoc.parse_sfw_entry("Legislator (government)") is None


def test_norm():
    assert ssoc.norm("Chief Engineer / Senior  Engineer ") == "chief engineer/senior engineer"
    assert ssoc.norm("Head \u2013 Content") == "head - content"


# ---------- match ----------
def test_topk_order_and_stable_ties():
    sc = np.array([[0.1, 0.9, 0.5], [0.7, 0.2, 0.7]])
    assert match.topk(sc, ["a", "b", "c"], 2) == [["b", "c"], ["a", "c"]]


def test_hit_at_k_levels_and_prefix_gold():
    preds = [["11111", "22222"], ["11112"], ["22211"], ["22231"]]
    gold = [{"22222"}, {"11111"}, {"2221"}, {"2221"}]
    assert list(match.hit_at_k(preds, gold, 1, 5)) == [0, 0, 1, 0]
    assert list(match.hit_at_k(preds, gold, 2, 5)) == [1, 0, 1, 0]
    assert list(match.hit_at_k(preds, gold, 1, 4)) == [0, 1, 1, 0]
    assert list(match.hit_at_k(preds, gold, 1, 2)) == [0, 1, 1, 1]


def test_hybrid_weights():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 2.0])
    assert np.allclose(match.hybrid(a, b, 0.25), [0.25, 1.5])
    assert np.allclose(match.hybrid(a, b), [0.5, 1.0])


@pytest.mark.parametrize("analyzer", ["word", "char"])
def test_tfidf_prefers_the_right_occupation(analyzer):
    q = ["Senior Civil Engineer", "Pastry Chef"]
    d = ["chef cooks and prepares food pastry", "civil engineer designs bridges and roads", "bus driver"]
    best = match.tfidf_scores(q, d, analyzer).argmax(axis=1)
    assert list(best) == [1, 0]


def test_bootstrap_ci():
    assert match.bootstrap_ci(np.ones(20)) == (1.0, 1.0)
    lo, hi = match.bootstrap_ci(np.array([0, 1] * 50))
    assert lo < 0.5 < hi and hi - lo < 0.25


# ---------- indicators ----------
def _toy():
    rs = pd.DataFrame([
        ("A", "r1", "x", False, False), ("A", "r1", "y", True, False), ("A", "r2", "x", False, False),
        ("B", "r3", "y", True, False), ("B", "r3", "z", False, True),
    ], columns=["sector", "job_role", "unique_skill", "emerging", "casl"])
    iv = pd.DataFrame([("I1", 2019, 100.0), ("I1", 2025, 300.0), ("I2", 2019, 50.0), ("I2", 2025, 40.0)],
                      columns=["industry", "year", "total"])
    cw = pd.DataFrame({"sector": ["A", "B", "C"], "mom_industry": ["I1", "I2", None]})
    return rs, iv, cw


def test_sector_skill_profile_hand_computed():
    rs, _, _ = _toy()
    p = indicators.sector_skill_profile(rs).set_index("sector")
    assert p.loc["A", "roles"] == 2 and p.loc["A", "median_skills_per_role"] == 1.5
    assert p.loc["A", "share_roles_with_emerging"] == 0.5
    assert p.loc["A", "emerging_share_of_requirements"] == pytest.approx(1 / 3)
    assert p.loc["B", "emerging_share_of_requirements"] == 0.5
    assert p.loc["B", "casl_share_of_requirements"] == 0.5 and p.loc["A", "casl_share_of_requirements"] == 0


def test_skill_demand_index_hand_computed():
    rs, iv, cw = _toy()
    d = indicators.skill_demand_index(2019, 2025, rs=rs, iv=iv, cw=cw).set_index("unique_skill")
    assert d.loc["x", "index_2025"] == 300 and d.loc["x", "index_2019"] == 100
    assert d.loc["y", "index_2025"] == 190 and d.loc["y", "index_2019"] == 100
    assert d.loc["z", "index_2025"] == 40 and d.loc["z", "growth_2019_2025_pct"] == pytest.approx(-20)
    assert d.loc["y", "growth_2019_2025_pct"] == pytest.approx(90)
    assert list(d.index) == ["x", "y", "z"]  # sorted by latest index
    assert d.loc["y", "sectors"] == 2 and bool(d.loc["y", "emerging"]) and bool(d.loc["z", "casl"])


def test_growth():
    assert indicators.growth(4, 5) == 25 and math.isnan(indicators.growth(0, 5))


def test_industry_vacancies_excludes_overlapping_groups():
    v = pd.DataFrame([
        (2025, "I1", load.PME, 30.0), (2025, "I1", load.NON_PME, 70.0),
        (2025, "I1", "professionals, managers, executives and technicians", 55.0),
    ], columns=["year", "industry", "occupation", "job_vacancy"])
    out = indicators.industry_vacancies(v).iloc[0]
    assert out["total"] == 100 and out["pme"] == 30 and out["non_pme"] == 70


# ---------- forecast ----------
def test_trailing_contiguous():
    s = pd.Series([1, np.nan, 2, 3, 4.0])
    assert forecast.trailing_contiguous(s).tolist() == [2, 3, 4]
    assert forecast.trailing_contiguous(pd.Series([1, 2, np.nan])).tolist() == [1, 2]


def test_simple_forecasters():
    y = np.arange(1.0, 9.0)
    assert forecast.f_naive(y, 3).tolist() == [8, 8, 8]
    assert forecast.f_seasonal_naive(y, 5).tolist() == [5, 6, 7, 8, 5]
    assert forecast.f_mean4(y, 2).tolist() == [6.5, 6.5]


def test_backtest_naive_error_grows_with_horizon_on_a_trend():
    q = pd.Index([f"{2000 + i // 4}-Q{i % 4 + 1}" for i in range(30)])
    data = {"i": pd.Series(np.arange(30) * 0.5, index=q)}
    bt = forecast.backtest(data, n_origins=3, h=2)
    nv = bt[bt["method"] == "naive"].groupby("h")["abs_err"].mean()
    assert nv.loc[1] == pytest.approx(0.5) and nv.loc[2] == pytest.approx(1.0)
    assert bt["origin"].nunique() == 3


# ---------- wages ----------
def test_cluster_ols_recovers_known_slope():
    rng = np.random.default_rng(1)
    n = 60
    t = pd.DataFrame({"emerging_share": rng.uniform(0, 0.5, n), "code": [f"{i % 12:05d}" for i in range(n)]})
    t["log_wage"] = 8.0 + 0.7 * t["emerging_share"]
    m = wages.fit(t, "log_wage ~ emerging_share")
    assert m.params["emerging_share"] == pytest.approx(0.7, abs=1e-9)
