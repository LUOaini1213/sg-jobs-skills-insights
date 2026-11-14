"""Evaluate the role-skill graph: can it predict held-out requirements, and does it agree
with the structure SSG published?

1. Link prediction. Hide 10% of role-skill edges, rank every skill the role does not have in
   the training graph, and see where the hidden ones land. Three simple baselines
   (popularity, popularity inside the role's sector, Jaccard nearest roles) are compared with
   three models (IDF-weighted projection, a three-step random walk, truncated SVD).
   Hyper-parameters are chosen on a separate validation split, never on the test edges.
2. Career tracks. The Skills Framework dataset gives each role a sector and a track. A
   track is the published career ladder the role sits on. The check asks how often the
   closest role by skills is on the same track. The dataset has no pathway edges (who moves
   to which role), so this tests grouping, not direction.
3. Communities against sectors.

    python -m jsi.graph_eval
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import graph, load

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"

BASELINES = ("popularity", "sector_popularity", "jaccard_knn")
MODELS = ("idf_projection_knn", "random_walk_rp3", "svd")
GRID = {
    "popularity": [None],
    "sector_popularity": [None],
    "jaccard_knn": [2, 3, 5, 10, 25, 50],         # nearest roles kept
    "idf_projection_knn": [2, 3, 5, 10, 25, 50],
    "random_walk_rp3": [0.0, 0.1, 0.2, 0.3, 0.5],  # beta: penalty on popular skills
    "svd": [32, 64, 128, 256, 512],                # rank
}
N_SPLITS = 5
TOP = 10


# ---------- splitting ----------
def split_edges(B: np.ndarray, fractions: tuple, seed: int) -> list:
    """Randomly partition the edges of B into len(fractions)+1 0/1 matrices.

    The first matrix gets what is left after the given fractions (so it is the training graph).
    """
    r, c = np.nonzero(B)
    perm = np.random.default_rng(seed).permutation(len(r))
    cuts = np.cumsum([int(round(f * len(r))) for f in fractions])
    parts = [perm[cuts[-1]:]] + [perm[a:b] for a, b in zip([0, *cuts[:-1]], cuts)]
    out = []
    for p in parts:
        M = np.zeros_like(B)
        M[r[p], c[p]] = 1
        out.append(M)
    return out


# ---------- scorers: (train matrix, sector codes) -> (n_roles, n_skills) scores ----------
def score_popularity(train: np.ndarray, sectors=None, param=None) -> np.ndarray:
    return np.tile(train.sum(axis=0).astype(float), (train.shape[0], 1))


def score_sector_popularity(train: np.ndarray, sectors, param=None) -> np.ndarray:
    """Share of the role's sector that requires the skill; global popularity only breaks ties."""
    sectors = np.asarray(sectors)
    n = train.shape[0]
    glob = train.sum(axis=0) / (n + 1.0)   # always < 1, so it never outweighs one sector count
    out = np.zeros(train.shape)
    for s in np.unique(sectors):
        m = sectors == s
        out[m] = train[m].sum(axis=0) + glob
    return out


def keep_top_neighbours(S: np.ndarray, k: int) -> np.ndarray:
    """Zero the diagonal and all but each row's k largest similarities (ties at the cut are kept)."""
    S = S.copy()
    np.fill_diagonal(S, 0.0)
    if k < S.shape[1]:
        thr = np.partition(S, -k, axis=1)[:, -k][:, None]
        S[S < thr] = 0.0
    return S


def score_jaccard_knn(train: np.ndarray, sectors=None, param: int = 25) -> np.ndarray:
    return keep_top_neighbours(graph.weighted_jaccard(train), param) @ train


def score_idf_projection_knn(train: np.ndarray, sectors=None, param: int = 25) -> np.ndarray:
    return keep_top_neighbours(graph.weighted_jaccard(train, graph.idf(train)), param) @ train


def score_random_walk_rp3(train: np.ndarray, sectors=None, param: float = 0.4) -> np.ndarray:
    """Three-step random walk role -> skill -> role -> skill, divided by (skill degree)^beta."""
    T = train.astype(float)
    dr = np.maximum(T.sum(axis=1), 1.0)
    ds = np.maximum(T.sum(axis=0), 1.0)
    P_rs = T / dr[:, None]
    P_sr = T.T / ds[:, None]
    return (P_rs @ P_sr) @ P_rs / ds[None, :] ** param


_SVD_CACHE: dict = {}


def score_svd(train: np.ndarray, sectors=None, param: int = 32) -> np.ndarray:
    key = (train.shape, train.tobytes())
    if key not in _SVD_CACHE:
        _SVD_CACHE.clear()
        _SVD_CACHE[key] = np.linalg.svd(train.astype(float), full_matrices=False)
    U, s, Vt = _SVD_CACHE[key]
    k = min(param, len(s))
    return (U[:, :k] * s[:k]) @ Vt[:k]


SCORERS = {
    "popularity": score_popularity, "sector_popularity": score_sector_popularity,
    "jaccard_knn": score_jaccard_knn, "idf_projection_knn": score_idf_projection_knn,
    "random_walk_rp3": score_random_walk_rp3, "svd": score_svd,
}


# ---------- ranking metrics ----------
def ranks(scores: np.ndarray, train: np.ndarray, test: np.ndarray) -> pd.DataFrame:
    """Rank of every held-out edge among the skills the role does not have in `train`.

    Rank 1 is best. Ties are broken by column order (lower index first), so the result is
    deterministic. Returns one row per held-out edge: role, skill, rank.
    """
    scores = np.round(scores, graph.SCORE_DECIMALS)
    rows = []
    for i in np.nonzero(test.sum(axis=1))[0]:
        cand = train[i] == 0
        s = scores[i]
        for j in np.nonzero(test[i])[0]:
            better = int((s[cand] > s[j]).sum())
            tied_before = int((s[:j][cand[:j]] == s[j]).sum())
            rows.append((int(i), int(j), 1 + better + tied_before))
    return pd.DataFrame(rows, columns=["role", "skill", "rank"])


def hits_at(rk: pd.DataFrame, k: int = TOP) -> float:
    return float((rk["rank"] <= k).mean())


def mrr(rk: pd.DataFrame) -> float:
    return float((1.0 / rk["rank"]).mean())


def cluster_bootstrap(hit_a: np.ndarray, role: np.ndarray, hit_b: np.ndarray | None = None,
                      n_boot: int = 1000, seed: int = graph.SEED) -> tuple:
    """95% interval for a hit rate (or for hit_a - hit_b), resampling roles, not edges.

    Edges of one role are not independent, so the role is the resampling unit.
    """
    d = hit_a.astype(float) - (0.0 if hit_b is None else hit_b.astype(float))
    codes, inv = np.unique(role, return_inverse=True)
    num = np.bincount(inv, weights=d)
    den = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(codes), size=(n_boot, len(codes)))
    stat = num[pick].sum(axis=1) / den[pick].sum(axis=1)
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return float(lo), float(hi)


# ---------- protocol ----------
def tune_and_test(B: np.ndarray, sectors, seed: int, grid: dict | None = None) -> dict:
    """One 80/10/10 split: choose each method's parameter on validation, refit on train+validation, test."""
    grid = GRID if grid is None else grid
    train, val, test = split_edges(B, (0.1, 0.1), seed)
    fit = train + val
    out = {}
    for name, params in grid.items():
        f = SCORERS[name]
        val_scores = {p: hits_at(ranks(f(train, sectors, p), train, val)) for p in params}
        best = max(params, key=lambda p: (val_scores[p], -params.index(p)))
        out[name] = {"param": best, "val_hits": val_scores[best], "ranks": ranks(f(fit, sectors, best), fit, test)}
    return out


def link_prediction(g: graph.Bipartite, n_splits: int = N_SPLITS, seed: int = graph.SEED, grid: dict | None = None,
                    peer: np.ndarray | None = None) -> dict:
    """Run the protocol on `n_splits` random splits; the first split carries the intervals.

    With `peer` (same-track matrix) the result also reports the "off-track" edges: held-out
    skills that no other role on the same track requires in the fitted graph. Copying from a
    track peer cannot find those.
    """
    sectors = g.roles["sector"].to_numpy()
    runs = [tune_and_test(g.B, sectors, seed + i, grid) for i in range(n_splits)]
    first = runs[0]
    methods = list(first)
    res = {"n_splits": n_splits, "top": TOP, "held_out_edges": int(len(first[methods[0]]["ranks"])),
           "roles_with_held_out_edges": int(first[methods[0]]["ranks"]["role"].nunique()), "methods": {}}
    for m in methods:
        rk = first[m]["ranks"]
        hit = (rk["rank"] <= TOP).to_numpy()
        lo, hi = cluster_bootstrap(hit, rk["role"].to_numpy())
        per_split = [hits_at(r[m]["ranks"]) for r in runs]
        res["methods"][m] = {
            "kind": "baseline" if m in BASELINES else "model",
            "param": first[m]["param"],
            "params_all_splits": [r[m]["param"] for r in runs],
            "val_hits_at_10": round(first[m]["val_hits"], 4),
            "hits_at_10": round(hits_at(rk), 4), "hits_at_10_ci95": [round(lo, 4), round(hi, 4)],
            "mrr": round(mrr(rk), 4),
            "hits_at_10_mean_all_splits": round(float(np.mean(per_split)), 4),
            "hits_at_10_min_all_splits": round(float(np.min(per_split)), 4),
            "hits_at_10_max_all_splits": round(float(np.max(per_split)), 4),
        }
    # the two contenders are picked on validation hits, then compared on the test edges
    pick = lambda group: max((m for m in methods if m in group), key=lambda m: first[m]["val_hits"])  # noqa: E731
    bb, bm = pick(BASELINES), pick(MODELS)
    ra, rb = first[bm]["ranks"], first[bb]["ranks"]
    assert (ra[["role", "skill"]].to_numpy() == rb[["role", "skill"]].to_numpy()).all()
    ha, hb = (ra["rank"] <= TOP).to_numpy(), (rb["rank"] <= TOP).to_numpy()
    lo, hi = cluster_bootstrap(ha, ra["role"].to_numpy(), hb)
    diffs = [hits_at(r[bm]["ranks"]) - hits_at(r[bb]["ranks"]) for r in runs]
    res["best_baseline"], res["best_model"] = bb, bm
    res["model_minus_baseline_hits_at_10"] = round(float(ha.mean() - hb.mean()), 4)
    res["model_minus_baseline_ci95"] = [round(lo, 4), round(hi, 4)]
    res["model_beats_baseline_in_splits"] = int(sum(d > 0 for d in diffs))
    res["verdict"] = ("model better" if lo > 0 else "baseline better" if hi < 0 else "no clear difference")
    if peer is not None:
        fit = g.B - split_edges(g.B, (0.1, 0.1), seed)[2]
        on_track = (peer.astype(float) @ fit) > 0
        off = ~on_track[ra["role"].to_numpy(), ra["skill"].to_numpy()]
        res["off_track_edges"] = {
            "n": int(off.sum()), "share_of_held_out": round(float(off.mean()), 4),
            "hits_at_10": {m: round(float((first[m]["ranks"]["rank"] <= TOP).to_numpy()[off].mean()), 4) for m in methods},
        }
    return res


# ---------- career tracks ----------
def track_sets(roles: pd.DataFrame) -> dict:
    return {(s, r): frozenset(t.split("; ")) for s, r, t in zip(roles["sector"], roles["job_role"], roles["track"])}


def same_track_matrix(g: graph.Bipartite, tracks: dict) -> np.ndarray:
    """peer[a, b]: a and b are different roles in the same sector that share a track."""
    keys = list(zip(g.roles["sector"], g.roles["job_role"]))
    n = len(keys)
    peer = np.zeros((n, n), dtype=bool)
    by_sector: dict = {}
    for i, (s, _) in enumerate(keys):
        by_sector.setdefault(s, []).append(i)
    for idx in by_sector.values():
        for a in idx:
            for b in idx:
                if a != b and tracks[keys[a]] & tracks[keys[b]]:
                    peer[a, b] = True
    return peer


def expected_any_hit(n_good: np.ndarray, n_pool: np.ndarray, k: int) -> np.ndarray:
    """P(at least one of k draws without replacement from a pool of n_pool is one of n_good)."""
    miss = np.ones(len(n_good), dtype=float)
    for i in range(k):
        miss *= np.clip((n_pool - n_good - i) / np.maximum(n_pool - i, 1), 0.0, 1.0)
    return 1.0 - miss


def track_agreement(g: graph.Bipartite, tracks: dict, k: int = 3) -> dict:
    """How often the nearest roles are on the same published track, against random draws."""
    peer = same_track_matrix(g, tracks)
    n = len(peer)
    has = peer.any(axis=1)
    sec = g.roles["sector"].to_numpy()
    sec_size = pd.Series(sec).map(pd.Series(sec).value_counts()).to_numpy()
    everyone = np.ones((n, n), dtype=bool)
    np.fill_diagonal(everyone, False)
    zeros = np.zeros((n, n))
    rankings = {
        "fewest_additional_skills": graph.transition_order(g.B, k),
        "idf_weighted_jaccard": graph.rank_targets(-graph.weighted_jaccard(g.B, graph.idf(g.B)), zeros, everyone, k),
        "jaccard": graph.rank_targets(-graph.weighted_jaccard(g.B), zeros, everyone, k),
    }
    out = {"roles_with_a_track_peer": int(has.sum()), "roles": int(n), "k": k, "methods": {}}
    for name, top in rankings.items():
        valid = top >= 0
        hit = np.zeros(top.shape, dtype=bool)
        hit[valid] = peer[np.repeat(np.arange(n), k).reshape(n, k)[valid], top[valid]]
        same_sec = np.zeros(n, dtype=bool)
        same_sec[valid[:, 0]] = sec[top[valid[:, 0], 0]] == sec[valid[:, 0]]
        out["methods"][name] = {
            "roles_with_a_candidate": int(valid[has, 0].sum()),
            "top1_same_track": round(float(hit[has, 0].mean()), 4),
            "top1_same_sector": round(float(same_sec[has].mean()), 4),
            f"any_same_track_in_top{k}": round(float(hit[has].any(axis=1).mean()), 4),
        }
    n_peer = peer.sum(axis=1).astype(float)
    for name, pool in (("random_role_same_sector", sec_size - 1.0), ("random_role", np.full(n, n - 1.0))):
        out["methods"][name] = {
            "top1_same_track": round(float(expected_any_hit(n_peer[has], pool[has], 1).mean()), 4),
            f"any_same_track_in_top{k}": round(float(expected_any_hit(n_peer[has], pool[has], k).mean()), 4),
        }
    return out


# ---------- communities against sectors ----------
def community_agreement(g: graph.Bipartite, n_seeds: int = 10, seed: int = graph.SEED) -> tuple:
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    sec = g.roles["sector"].to_numpy()
    w = graph.idf(g.B)

    def stats(role_lab):
        return {"communities": int(role_lab.max() + 1),
                "ami_vs_sector": round(float(adjusted_mutual_info_score(sec, role_lab)), 4),
                "ari_vs_sector": round(float(adjusted_rand_score(sec, role_lab)), 4),
                "purity": round(graph.purity(role_lab, sec), 4)}

    role_lab, skill_lab = graph.louvain(g.B, w, seed)
    table = graph.community_table(g, role_lab, skill_lab)
    main = stats(role_lab)
    others = [stats(graph.louvain(g.B, w, seed + i)[0]) for i in range(1, n_seeds)]
    alls = [main, *others]
    # a sector is "kept whole" if one community holds at least 90% of its roles
    ct = pd.crosstab(pd.Series(sec, name="sector"), pd.Series(role_lab, name="c"))
    share = ct.max(axis=1) / ct.sum(axis=1)
    single = table[table["sectors"] == 1]
    res = {
        **main, "sectors": int(len(ct)), "seed": seed, "weights": "idf",
        "sectors_kept_whole_90pct": int((share >= 0.9).sum()),
        "single_sector_communities": int(len(single)),
        "communities_mixing_3plus_sectors": int((table["sectors"] >= 3).sum()),
        "largest_mixed_community": table[table["sectors"] >= 3].sort_values("roles", ascending=False)
        .iloc[0][["community", "roles", "sector_counts"]].to_dict() if (table["sectors"] >= 3).any() else None,
        "across_seeds": {
            "n_seeds": n_seeds,
            "communities_min": min(s["communities"] for s in alls), "communities_max": max(s["communities"] for s in alls),
            "ami_min": min(s["ami_vs_sector"] for s in alls), "ami_max": max(s["ami_vs_sector"] for s in alls),
        },
        "unweighted": stats(graph.louvain(g.B, None, seed)[0]),
    }
    return res, table


# ---------- run ----------
def transition_summary(tr: pd.DataFrame, n_roles: int) -> dict:
    first = tr[tr["rank"] == 1]
    return {
        "min_share_of_target_held": graph.MIN_HELD,
        "roles_with_a_reachable_role": int(len(first)),
        "roles_without": int(n_roles - len(first)),
        "top1_zero_additional_skills": int((first["additional_skills"] == 0).sum()),
        "top1_median_additional_skills": float(first["additional_skills"].median()),
        "top1_same_sector_share": round(float(first["same_sector"].mean()), 4),
        "top1_cross_sector": int((~first["same_sector"]).sum()),
        "top1_direction": {k: int(v) for k, v in first["direction"].value_counts().sort_index().items()},
        "listed_transitions": int(len(tr)),
        "listed_cross_sector": int((~tr["same_sector"]).sum()),
    }


def run(write: bool = True) -> dict:
    rs = load.role_skills()
    roles = load.roles()
    g = graph.build(rs)
    tracks = track_sets(roles)
    deg_r, deg_s = g.B.sum(axis=1), g.B.sum(axis=0)

    tr = graph.transitions(g, 3, tracks)
    adj = graph.skill_adjacency(g, 3)
    comm, comm_table = community_agreement(g)
    lp = link_prediction(g, peer=same_track_matrix(g, tracks))
    lp_table = pd.DataFrame([{"method": m, **{k: v for k, v in d.items() if not isinstance(v, list)},
                              "hits_at_10_ci95_low": d["hits_at_10_ci95"][0], "hits_at_10_ci95_high": d["hits_at_10_ci95"][1]}
                             for m, d in lp["methods"].items()])
    metrics = {
        "graph": {
            "roles": int(g.B.shape[0]), "skills": int(g.B.shape[1]), "edges": g.n_edges,
            "density": round(float(g.n_edges / g.B.size), 4),
            "median_skills_per_role": float(np.median(deg_r)), "max_skills_per_role": int(deg_r.max()),
            "median_roles_per_skill": float(np.median(deg_s)), "max_roles_per_skill": int(deg_s.max()),
            "skills_in_one_role_only": int((deg_s == 1).sum()),
            "roles_in_framework_without_skills": int(len(roles) - g.B.shape[0]),
        },
        "transitions": transition_summary(tr, g.B.shape[0]),
        "skill_adjacency": {"skills_listed": int(adj["skill"].nunique()), "min_roles": 5, "pairs": int(len(adj))},
        "communities": comm,
        "link_prediction": lp,
        "track_agreement": track_agreement(g, tracks),
    }
    if write:
        OUT.mkdir(exist_ok=True)
        tr.to_csv(OUT / "role_transitions.csv", index=False, lineterminator="\n")
        adj.to_csv(OUT / "skill_adjacency.csv", index=False, lineterminator="\n")
        comm_table.to_csv(OUT / "skill_communities.csv", index=False, lineterminator="\n")
        lp_table.to_csv(OUT / "link_prediction.csv", index=False, lineterminator="\n")
        (OUT / "graph_metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n",
                                                encoding="utf-8", newline="\n")
    return metrics


if __name__ == "__main__":
    m = run()
    print(json.dumps({k: m[k] for k in ("graph", "transitions")}, indent=2))
    print(pd.DataFrame(m["link_prediction"]["methods"]).T.to_string())
