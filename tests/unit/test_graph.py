"""Role-skill graph, link-prediction protocol and sample-size arithmetic on toy inputs.

The toy graph has five roles and six skills. Every expected value below was worked out by
hand from this table (skill columns are sorted: u v w x y z).

    sector A   r1 = {x, y, z}       levels 2, 3, 2      track T1
               r2 = {w, x, y, z}    levels 4, 3, 4, 3   tracks T1 and T2
               r3 = {x}             level 1             track T2
    sector B   r4 = {v, w}          levels 2, 3         track T3
               r5 = {u, v, w}       levels 1, 3, 3      track T3
"""
import math

import numpy as np
import pandas as pd
import pytest

from jsi import graph, graph_eval, power

L5, L25, L53 = math.log(5), math.log(2.5), math.log(5 / 3)


def _rs() -> pd.DataFrame:
    rows = [
        ("A", "r1", "x", 2), ("A", "r1", "y", 3), ("A", "r1", "z", 2),
        ("A", "r2", "w", 4), ("A", "r2", "x", 3), ("A", "r2", "y", 4), ("A", "r2", "z", 3),
        ("A", "r3", "x", 1),
        ("B", "r4", "v", 2), ("B", "r4", "w", 3),
        ("B", "r5", "u", 1), ("B", "r5", "v", 3), ("B", "r5", "w", 3),
    ]
    return pd.DataFrame(rows, columns=["sector", "job_role", "unique_skill", "level"]).sample(frac=1, random_state=3)


def _g() -> graph.Bipartite:
    return graph.build(_rs())


TRACKS = {("A", "r1"): frozenset({"T1"}), ("A", "r2"): frozenset({"T1", "T2"}), ("A", "r3"): frozenset({"T2"}),
          ("B", "r4"): frozenset({"T3"}), ("B", "r5"): frozenset({"T3"})}
SECTORS = np.array(["A", "A", "A", "B", "B"])


# ---------- graph ----------
def test_build_orders_nodes_and_keeps_levels():
    g = _g()
    assert list(g.roles["job_role"]) == ["r1", "r2", "r3", "r4", "r5"] and g.skills == list("uvwxyz")
    assert g.B.tolist() == [[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 0, 0], [0, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]]
    assert g.n_edges == 13 and g.level[1, 2] == 4 and np.isnan(g.level[0, 0])


def test_build_keeps_the_highest_level_of_a_repeated_requirement():
    rs = pd.DataFrame([("A", "r", "x", 2), ("A", "r", "x", 5), ("A", "r", "x", 3)],
                      columns=["sector", "job_role", "unique_skill", "level"])
    g = graph.build(rs)
    assert g.B.tolist() == [[1]] and g.level[0, 0] == 5


def test_shared_and_gap_are_hand_computed_and_gap_is_directional():
    B = _g().B
    assert graph.shared(B).tolist() == [[3, 3, 1, 0, 0], [3, 4, 1, 1, 1], [1, 1, 1, 0, 0], [0, 1, 0, 2, 2], [0, 1, 0, 2, 3]]
    gap = graph.skill_gap(B)
    assert gap[0, 1] == 1 and gap[1, 0] == 0      # r1 lacks w; r2 lacks nothing r1 needs
    assert gap[2, 0] == 2 and gap[0, 2] == 0
    assert gap[3, 1] == 3 and gap[1, 3] == 1 and gap[3, 4] == 1 and gap[4, 3] == 0


def test_idf_and_weighted_jaccard():
    B = _g().B
    assert graph.idf(B) == pytest.approx([L5, L25, L53, L53, L25, L25])
    j = graph.weighted_jaccard(B)
    assert j[0, 1] == pytest.approx(3 / 4) and j[0, 2] == pytest.approx(1 / 3) and j[1, 3] == pytest.approx(1 / 5)
    assert j[3, 4] == pytest.approx(2 / 3) and j[0, 3] == 0 and np.allclose(j, j.T) and np.allclose(np.diag(j), 1)
    w = graph.weighted_jaccard(B, graph.idf(B))
    assert w[0, 1] == pytest.approx((L53 + 2 * L25) / (2 * L53 + 2 * L25))       # 0.8210
    assert w[1, 3] == pytest.approx(L53 / (2 * L53 + 3 * L25))                    # 0.1355: w is a common skill
    assert w[1, 3] < j[1, 3] and w[0, 1] > j[0, 1]


def test_reachable_needs_half_of_the_target():
    ok = graph.reachable(_g().B)
    expect = np.zeros((5, 5), dtype=bool)
    for a, b in [(0, 1), (0, 2), (1, 0), (1, 2), (1, 3), (3, 4), (4, 3)]:
        expect[a, b] = True
    # r2 -> r5 fails (1 of 3 held); r3 -> r1 fails (1 of 3); r4 -> r2 fails (1 of 4); nobody reaches themselves
    assert ok.tolist() == expect.tolist()
    assert graph.reachable(_g().B, 1.0)[0].tolist() == [False, False, True, False, False]


def test_transition_order_fewest_additional_then_most_shared():
    top = graph.transition_order(_g().B, 2)
    assert top.tolist() == [[2, 1], [0, 2], [-1, -1], [4, -1], [3, -1]]


def test_rank_targets_tie_breaks():
    major = np.array([[1, 0, 0, 0]])
    minor = np.array([[9, 2, 5, 5]])
    elig = np.array([[True, True, True, True]])
    assert graph.rank_targets(major, minor, elig, 4).tolist() == [[2, 3, 1, 0]]
    elig[0, 2] = False
    assert graph.rank_targets(major, minor, elig, 3).tolist() == [[3, 1, 0]]


def test_seniority_and_direction():
    assert graph.seniority(_g().level) == pytest.approx([7 / 3, 3.5, 1, 2.5, 7 / 3])
    assert math.isnan(graph.seniority(np.array([[np.nan, np.nan]]))[0])
    assert [graph.direction(x) for x in (0.26, 0.25, 0.2, -0.25, -0.3, float("nan"))] == [
        "up", "lateral", "lateral", "lateral", "down", "unknown"]


def test_transitions_table_lists_the_gap():
    t = graph.transitions(_g(), 2, TRACKS).set_index(["job_role", "to_job_role"])
    assert len(t) == 6 and "r3" not in t.index.get_level_values(0)
    up = t.loc[("r1", "r2")]
    assert up["rank"] == 2 and up["additional_skills"] == 1 and up["skills_to_add"] == "w"
    assert up["shared_skills"] == 3 and up["share_of_target_held"] == 0.75 and up["shared_at_higher_level"] == 3
    assert up["level_shift"] == 1.17 and up["direction"] == "up" and up["same_track"] and up["same_sector"]
    down = t.loc[("r1", "r3")]
    assert down["rank"] == 1 and down["additional_skills"] == 0 and down["skills_to_add"] == ""
    assert down["direction"] == "down" and not down["same_track"] and down["shared_at_higher_level"] == 0
    # r5 needs v at level 3 (r4 has 2) and w at level 3 (r4 already has 3): one level-up, not two
    assert t.loc[("r4", "r5"), "skills_to_add"] == "u" and t.loc[("r4", "r5"), "shared_at_higher_level"] == 1
    assert t.loc[("r5", "r4"), "shared_at_higher_level"] == 0 and t.loc[("r5", "r4"), "level_shift"] == 0.17
    assert t.loc[("r5", "r4"), "direction"] == "lateral" and t.loc[("r4", "r5"), "direction"] == "lateral"
    assert t.loc[("r2", "r1"), "skills_to_add"] == "" and t.loc[("r2", "r3"), "rank"] == 2


def test_skill_adjacency_ranks_by_jaccard_and_skips_rare_skills():
    a = graph.skill_adjacency(_g(), k=1, min_roles=2).set_index("skill")
    assert "u" not in a.index and "u" not in set(a["adjacent_skill"])        # u is required by one role only
    assert a.loc["y", "adjacent_skill"] == "z" and a.loc["y", "jaccard"] == 1.0 and a.loc["y", "roles_requiring_both"] == 2
    assert a.loc["x", "adjacent_skill"] == "y" and a.loc["x", "jaccard"] == pytest.approx(2 / 3, abs=1e-4)
    assert a.loc["w", "adjacent_skill"] == "v" and a.loc["v", "adjacent_skill"] == "w" and a.loc["w", "roles"] == 3
    full = graph.skill_adjacency(_g(), k=5, min_roles=1)
    wx = full[(full["skill"] == "w") & (full["adjacent_skill"] == "x")].iloc[0]
    assert wx["jaccard"] == 0.2 and wx["roles_requiring_both"] == 1
    assert set(full[full["skill"] == "u"]["adjacent_skill"]) == {"v", "w"}


def test_louvain_finds_the_two_blocks_and_table_describes_them():
    g = _g()
    role_lab, skill_lab = graph.louvain(g.B, seed=1)
    assert role_lab.tolist() == [0, 0, 0, 1, 1] and skill_lab.tolist() == [1, 1, 1, 0, 0, 0]
    t = graph.community_table(g, role_lab, skill_lab).set_index("community")
    assert t.loc[0, "roles"] == 3 and t.loc[0, "skills"] == 3 and t.loc[0, "top_skills"] == "x; y; z"
    assert t.loc[1, "top_skills"] == "v; w; u" and t.loc[1, "sector_counts"] == "B (2)" and t.loc[0, "sectors"] == 1
    # a skill whose weight is zero is left out of the graph and gets no community
    w = np.array([1.0, 1, 0, 1, 1, 1])
    assert graph.louvain(g.B, w, seed=1)[1][2] == -1


def test_purity():
    assert graph.purity([0, 0, 1, 1, 1], list("aaabb")) == pytest.approx(0.8)
    assert graph.purity([0, 1, 2], list("abc")) == 1.0


# ---------- link prediction ----------
def test_split_edges_is_a_partition_with_the_stated_sizes():
    B = _g().B
    train, val, test = graph_eval.split_edges(B, (0.2, 0.2), 7)
    assert (train + val + test == B).all()
    assert (int(train.sum()), int(val.sum()), int(test.sum())) == (7, 3, 3)
    again = graph_eval.split_edges(B, (0.2, 0.2), 7)
    assert all((a == b).all() for a, b in zip((train, val, test), again))
    assert any(not (graph_eval.split_edges(B, (0.2, 0.2), s)[2] == test).all() for s in (8, 9, 10))


def test_popularity_scorers():
    B = _g().B
    assert graph_eval.score_popularity(B)[2].tolist() == [1, 2, 3, 3, 2, 2]
    s = graph_eval.score_sector_popularity(B, SECTORS)
    # r3 is in sector A: y and z are required by 2 sector-A roles, w by 1, u and v by none
    assert s[2] == pytest.approx([1 / 6, 2 / 6, 1 + 3 / 6, 3 + 3 / 6, 2 + 2 / 6, 2 + 2 / 6])
    assert s[3] == pytest.approx([1 + 1 / 6, 2 + 2 / 6, 2 + 3 / 6, 3 / 6, 2 / 6, 2 / 6])


def test_keep_top_neighbours():
    S = np.array([[1.0, 0.5, 0.2, 0.4], [0.5, 1.0, 0.5, 0.1], [0.2, 0.5, 1.0, 0.3], [0.4, 0.1, 0.3, 1.0]])
    out = graph_eval.keep_top_neighbours(S, 1)
    assert out.tolist() == [[0, 0.5, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0], [0.4, 0, 0, 0]]  # ties at the cut are kept
    assert graph_eval.keep_top_neighbours(S, 2)[0].tolist() == [0, 0.5, 0, 0.4]
    assert S[0, 0] == 1.0  # input not modified


def test_jaccard_knn_scores_copy_from_the_nearest_roles():
    B = _g().B
    one = graph_eval.score_jaccard_knn(B, None, 1)
    assert one[2] == pytest.approx([0, 0, 0, 1 / 3, 1 / 3, 1 / 3])          # r3's nearest role is r1 (1/3)
    two = graph_eval.score_jaccard_knn(B, None, 2)
    assert two[2] == pytest.approx([0, 0, 1 / 4, 7 / 12, 7 / 12, 7 / 12])   # plus r2 (1/4), which brings w
    idf = graph_eval.score_idf_projection_knn(B, None, 1)
    assert idf[3] == pytest.approx(np.array([1, 1, 1, 0, 0, 0]) * (L25 + L53) / (L5 + L25 + L53))


def test_random_walk_scores_hand_computed():
    B = _g().B
    s0 = graph_eval.score_random_walk_rp3(B, None, 0.0)
    # from r3: x -> {r1, r2, r3} each 1/3; r1 spreads 1/3 to x, y, z; r2 spreads 1/4 to w, x, y, z
    assert s0[2] == pytest.approx([0, 0, 1 / 12, 19 / 36, 7 / 36, 7 / 36])
    assert s0.sum(axis=1) == pytest.approx(np.ones(5))
    s1 = graph_eval.score_random_walk_rp3(B, None, 1.0)
    assert s1[2] == pytest.approx([0, 0, 1 / 36, 19 / 108, 7 / 72, 7 / 72])


def test_svd_full_rank_reconstructs_and_low_rank_does_not():
    B = _g().B
    assert np.allclose(graph_eval.score_svd(B, None, 5), B)
    assert not np.allclose(graph_eval.score_svd(B, None, 1), B, atol=0.05)
    assert np.allclose(graph_eval.score_svd(B, None, 99), B)


def test_ranks_exclude_training_edges_and_break_ties_by_column():
    scores = np.array([[0.5, 0.9, 0.9, 0.1], [0.3, 0.2, 0.1, 0.4]])
    train = np.array([[1, 0, 0, 0], [0, 0, 0, 1]])
    test = np.array([[0, 0, 1, 1], [1, 0, 0, 0]])
    rk = graph_eval.ranks(scores, train, test)
    assert rk.values.tolist() == [[0, 2, 2], [0, 3, 3], [1, 0, 1]]
    assert graph_eval.hits_at(rk, 2) == pytest.approx(2 / 3) and graph_eval.hits_at(rk, 1) == pytest.approx(1 / 3)
    assert graph_eval.mrr(rk) == pytest.approx((1 / 2 + 1 / 3 + 1) / 3)


def test_cluster_bootstrap():
    role = np.array([0, 0, 1, 1, 2, 2])
    ones = np.ones(6, dtype=bool)
    assert graph_eval.cluster_bootstrap(ones, role) == (1.0, 1.0)
    assert graph_eval.cluster_bootstrap(ones, role, ones) == (0.0, 0.0)
    hit = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
    lo, hi = graph_eval.cluster_bootstrap(hit, role)
    assert 0 <= lo < 0.5 < hi <= 1
    # whole roles are resampled: all three draws are the all-miss role with probability 1/27 = 3.7%,
    # above 2.5%, so the lower bound is exactly 0. Resampling the six edges would give (1/3)^6 = 0.1%.
    lo, hi = graph_eval.cluster_bootstrap(np.array([1, 1, 0, 0, 1, 1], dtype=bool), role)
    assert lo == 0.0 and hi == 1.0


def _block_graph() -> graph.Bipartite:
    """Two sectors of 12 roles; each role takes 8 of its sector's 30 skills and 1 of 4 shared ones."""
    rng = np.random.default_rng(0)
    rows = []
    for s, sec in enumerate("AB"):
        for r in range(12):
            own = rng.choice(30, size=8, replace=False)
            rows += [(sec, f"{sec}{r:02d}", f"{sec}-skill{k}", 3) for k in own]
            rows.append((sec, f"{sec}{r:02d}", f"common{rng.integers(4)}", 2))
    return graph.build(pd.DataFrame(rows, columns=["sector", "job_role", "unique_skill", "level"]))


def test_link_prediction_protocol_on_a_block_graph():
    g = _block_graph()
    grid = {"popularity": [None], "sector_popularity": [None], "jaccard_knn": [2, 5],
            "idf_projection_knn": [2, 5], "random_walk_rp3": [0.0, 0.3], "svd": [2, 4]}
    res = graph_eval.link_prediction(g, n_splits=2, seed=5, grid=grid)
    assert res["held_out_edges"] == round(0.1 * g.n_edges) and set(res["methods"]) == set(grid)
    m = res["methods"]
    # skills are sector-specific, so knowing the sector must beat global popularity
    assert m["sector_popularity"]["hits_at_10"] > m["popularity"]["hits_at_10"]
    assert res["best_baseline"] in graph_eval.BASELINES and res["best_model"] in graph_eval.MODELS
    lo, hi = res["model_minus_baseline_ci95"]
    assert lo <= res["model_minus_baseline_hits_at_10"] <= hi
    assert res["verdict"] == ("model better" if lo > 0 else "baseline better" if hi < 0 else "no clear difference")
    assert all(mm["param"] in grid[k] for k, mm in m.items())


def test_tuning_never_sees_the_test_edges():
    g = _block_graph()
    train, val, test = graph_eval.split_edges(g.B, (0.1, 0.1), 5)
    out = graph_eval.tune_and_test(g.B, g.roles["sector"].to_numpy(), 5, {"jaccard_knn": [2, 5]})
    rk = out["jaccard_knn"]["ranks"]
    assert len(rk) == int(test.sum()) and all(test[r, s] == 1 for r, s in zip(rk["role"], rk["skill"]))
    # the validation score is computed from the training graph alone
    expect = max(graph_eval.hits_at(graph_eval.ranks(graph_eval.score_jaccard_knn(train, None, k), train, val)) for k in (2, 5))
    assert out["jaccard_knn"]["val_hits"] == pytest.approx(expect)
    # and it is not the score on the test edges (the two differ on this graph, so a leak would show)
    on_test = max(graph_eval.hits_at(graph_eval.ranks(graph_eval.score_jaccard_knn(train, None, k), train, test)) for k in (2, 5))
    assert on_test != pytest.approx(expect)


# ---------- career tracks ----------
def test_same_track_matrix_and_expected_hits():
    peer = graph_eval.same_track_matrix(_g(), TRACKS)
    expect = np.zeros((5, 5), dtype=bool)
    for a, b in [(0, 1), (1, 2), (3, 4)]:
        expect[a, b] = expect[b, a] = True
    assert peer.tolist() == expect.tolist()
    f = graph_eval.expected_any_hit
    assert f(np.array([1.0]), np.array([4.0]), 1)[0] == pytest.approx(0.25)
    assert f(np.array([1.0]), np.array([4.0]), 2)[0] == pytest.approx(0.5)
    assert f(np.array([0.0]), np.array([4.0]), 3)[0] == 0 and f(np.array([2.0]), np.array([2.0]), 1)[0] == 1


def test_track_agreement_hand_computed():
    res = graph_eval.track_agreement(_g(), TRACKS, k=2)
    m = res["methods"]
    assert res["roles_with_a_track_peer"] == 5
    # fewest additional skills: r1 -> r3 (other track), r2 -> r1, r3 -> nobody, r4 -> r5, r5 -> r4
    assert m["fewest_additional_skills"] == {"roles_with_a_candidate": 4, "top1_same_track": 0.6,
                                             "top1_same_sector": 0.8, "any_same_track_in_top2": 0.8}
    # Jaccard: r1 -> r2, r2 -> r1, r3 -> r1 (other track), r4 -> r5, r5 -> r4
    assert m["jaccard"]["top1_same_track"] == 0.8 and m["jaccard"]["roles_with_a_candidate"] == 5
    assert m["random_role_same_sector"]["top1_same_track"] == 0.8      # (1/2 + 1 + 1/2 + 1 + 1) / 5
    assert m["random_role"]["top1_same_track"] == 0.3                  # (1 + 2 + 1 + 1 + 1) / 4 / 5


def test_transition_summary_counts():
    g = _g()
    s = graph_eval.transition_summary(graph.transitions(g, 2, TRACKS), 5)
    assert s["roles_with_a_reachable_role"] == 4 and s["roles_without"] == 1
    assert s["top1_zero_additional_skills"] == 3 and s["top1_same_sector_share"] == 1.0   # r1, r2, r5; r4 needs u
    assert s["top1_direction"] == {"down": 2, "lateral": 2} and s["listed_transitions"] == 6


# ---------- sample size ----------
def test_sample_size_hand_computed():
    # (1.959964 * sqrt(2 * .55 * .45) + 0.841621 * sqrt(.25 + .24))^2 / .1^2 = 387.34
    assert power.n_per_arm(0.5, 0.6) == 388 and power.n_per_arm(0.6, 0.5) == 388
    assert power.n_per_arm(0.5, 0.6, power=0.9) > 388 and power.n_per_arm(0.5, 0.6, alpha=0.01) > 388
    with pytest.raises(ValueError):
        power.n_per_arm(0.3, 0.3)


def test_power_and_mde_are_consistent_with_the_sample_size():
    n = power.n_per_arm(0.04, 0.05)
    assert 0.80 <= power.power_at(n, 0.04, 0.05) < 0.801 and power.power_at(n - 100, 0.04, 0.05) < 0.80
    assert power.mde(n, 0.04) == pytest.approx(0.01, abs=2e-5) and power.mde(n, 0.04) <= 0.01
    assert power.mde(n // 2, 0.04) > 0.0141


def test_design_effect_and_srm():
    assert power.design_effect(20, 0.02) == pytest.approx(1.38) and power.design_effect(1, 0.5) == 1.0
    assert power.srm_p_value(500, 500) == 1.0
    assert power.srm_p_value(5000, 5200) == pytest.approx(0.0477, abs=5e-4)   # chi2 = 2 * 100^2 / 5100 = 3.92
    assert power.srm_p_value(900, 100, expected_share_a=0.9) == pytest.approx(1.0)
    assert power.srm_p_value(900, 100) < 1e-6


def test_simulated_power_matches_the_formula():
    n = power.n_per_arm(0.5, 0.6)
    rng = np.random.default_rng(11)
    sims = 4000

    def reject_rate(p_a, p_b):
        a, b = rng.binomial(n, p_a, sims) / n, rng.binomial(n, p_b, sims) / n
        pbar = (a + b) / 2
        z = (b - a) / np.sqrt(2 * pbar * (1 - pbar) / n)
        return float((np.abs(z) > 1.959964).mean())

    assert 0.77 < reject_rate(0.5, 0.6) < 0.83
    assert 0.035 < reject_rate(0.5, 0.5) < 0.065


def test_run_reports_the_worked_example():
    r = power.run(write=False)
    assert r["n_per_arm"] == power.n_per_arm(0.04, 0.05) and r["n_total"] == 2 * r["n_per_arm"]
    assert r["n_per_arm_cluster_randomised"] == math.ceil(r["n_per_arm"] * 1.38)
    assert r["clusters_per_arm"] == math.ceil(r["n_per_arm_cluster_randomised"] / 20)
    assert r["weeks_at_assumed_traffic"] == math.ceil(r["n_total"] / 5000) and len(r["sensitivity"]) == 9
