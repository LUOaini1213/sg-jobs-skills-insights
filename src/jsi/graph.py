"""Role-skill bipartite graph from the Skills Framework.

Nodes are job roles (one per sector and title) and SSG unique skills; an edge means the role
requires the skill. From this graph the module builds
  * a role-role projection (IDF-weighted Jaccard similarity),
  * skill-gap transitions (which roles need the fewest additional skills, and which ones),
  * skill adjacency (skills that are required together),
  * communities (Louvain on the bipartite graph) to compare with the official sectors.

Everything here describes what the Skills Framework says roles require. It says nothing
about how people actually move between jobs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 20261114
SCORE_DECIMALS = 9  # scores are rounded before ranking so that ties do not depend on the BLAS build


@dataclass(frozen=True)
class Bipartite:
    roles: pd.DataFrame   # sector, job_role (row order = matrix row order)
    skills: list          # column order
    B: np.ndarray         # (n_roles, n_skills) 0/1, int64
    level: np.ndarray     # (n_roles, n_skills) highest proficiency level required, NaN if none given

    @property
    def n_edges(self) -> int:
        return int(self.B.sum())


def build(rs: pd.DataFrame) -> Bipartite:
    """Bipartite graph from a (sector, job_role, unique_skill[, level]) table."""
    roles = rs[["sector", "job_role"]].drop_duplicates().sort_values(["sector", "job_role"]).reset_index(drop=True)
    skills = sorted(rs["unique_skill"].unique())
    ri = {k: i for i, k in enumerate(zip(roles["sector"], roles["job_role"]))}
    si = {s: j for j, s in enumerate(skills)}
    B = np.zeros((len(roles), len(skills)), dtype=np.int64)
    L = np.full(B.shape, np.nan)
    lev = rs["level"] if "level" in rs.columns else pd.Series(np.nan, index=rs.index)
    for sec, role, skill, lv in zip(rs["sector"], rs["job_role"], rs["unique_skill"], lev):
        i, j = ri[(sec, role)], si[skill]
        B[i, j] = 1
        if not pd.isna(lv):
            L[i, j] = lv if np.isnan(L[i, j]) else max(L[i, j], lv)
    return Bipartite(roles, skills, B, L)


def shared(B: np.ndarray) -> np.ndarray:
    """shared[a, b] = number of columns set in both row a and row b (exact; float BLAS product of 0/1)."""
    F = B.astype(np.float64)
    return np.rint(F @ F.T).astype(np.int64)


def idf(B: np.ndarray) -> np.ndarray:
    """log(n_roles / roles requiring the skill): a skill every role needs carries no weight."""
    deg = B.sum(axis=0)
    return np.where(deg > 0, np.log(B.shape[0] / np.maximum(deg, 1)), 0.0)


def weighted_jaccard(B: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Row-by-row Jaccard similarity; with weights, sum of w over shared / sum of w over union."""
    w = np.ones(B.shape[1]) if w is None else np.asarray(w, dtype=float)
    Bw = B * w
    inter = Bw @ B.T
    tot = Bw.sum(axis=1)
    union = tot[:, None] + tot[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(union > 0, inter / union, 0.0)
    return np.round(s, SCORE_DECIMALS)


def skill_gap(B: np.ndarray) -> np.ndarray:
    """gap[a, b] = number of skills role b requires that role a does not (not symmetric)."""
    return B.sum(axis=1)[None, :] - shared(B)


def seniority(level: np.ndarray) -> np.ndarray:
    """Mean required proficiency level of a role (NaN when no level is given)."""
    n = (~np.isnan(level)).sum(axis=1)
    return np.where(n > 0, np.nansum(level, axis=1) / np.maximum(n, 1), np.nan)


def rank_targets(key_major: np.ndarray, key_minor: np.ndarray, eligible: np.ndarray, k: int) -> np.ndarray:
    """Per row: the k eligible columns with the smallest key_major, ties by largest key_minor, then index.

    Returns an (n, k) array of column indices, -1 where fewer than k columns are eligible.
    """
    n, m = key_major.shape
    out = np.full((n, k), -1, dtype=np.int64)
    cols = np.arange(m)
    for a in range(n):
        idx = cols[eligible[a]]
        order = np.lexsort((idx, -key_minor[a, idx], key_major[a, idx]))
        top = idx[order][:k]
        out[a, : len(top)] = top
    return out


MIN_HELD = 0.5  # a target counts as reachable only if the source already holds this share of its skills


def reachable(B: np.ndarray, min_held: float = MIN_HELD) -> np.ndarray:
    """reachable[a, b]: role a holds at least `min_held` of the skills role b requires (a != b).

    Without this rule a role with three requirements would be "close" to every role in the
    framework, because nobody needs more than three additional skills to reach it.
    """
    inter = shared(B)
    deg = B.sum(axis=1)
    ok = inter >= min_held * deg[None, :]
    ok &= inter > 0
    np.fill_diagonal(ok, False)
    return ok


def transition_order(B: np.ndarray, k: int, min_held: float = MIN_HELD) -> np.ndarray:
    """Top-k reachable targets per role: fewest additional skills, ties by more shared skills, then order."""
    return rank_targets(skill_gap(B), shared(B), reachable(B, min_held), k)


def direction(shift: float, band: float = 0.25) -> str:
    """Label a change in mean required proficiency level: up, down or lateral (within +/- band)."""
    if np.isnan(shift):
        return "unknown"
    return "up" if shift > band else "down" if shift < -band else "lateral"


def transitions(g: Bipartite, k: int = 5, tracks: dict | None = None, min_held: float = MIN_HELD) -> pd.DataFrame:
    """Long table: for each role, the k roles that need the fewest additional skills, with the gap listed."""
    B = g.B
    inter = shared(B)
    deg = B.sum(axis=1)
    top = transition_order(B, k, min_held)
    sen = seniority(g.level)
    skills = np.array(g.skills, dtype=object)
    sec, role = g.roles["sector"].to_numpy(), g.roles["job_role"].to_numpy()
    rows = []
    for a in range(len(sec)):
        for rank, b in enumerate(top[a], 1):
            if b < 0:
                continue
            both = (B[a] == 1) & (B[b] == 1)
            with np.errstate(invalid="ignore"):
                up = int(((g.level[b] > g.level[a]) & both).sum())
            row = {
                "sector": sec[a], "job_role": role[a], "rank": rank,
                "to_sector": sec[b], "to_job_role": role[b],
                "shared_skills": int(inter[a, b]), "additional_skills": int(deg[b] - inter[a, b]),
                "share_of_target_held": round(float(inter[a, b] / deg[b]), 4),
                "shared_at_higher_level": up,
                "level_shift": None if np.isnan(sen[a]) or np.isnan(sen[b]) else round(float(sen[b] - sen[a]), 2),
                "direction": direction(sen[b] - sen[a]),
                "same_sector": bool(sec[a] == sec[b]),
            }
            if tracks is not None:
                row["same_track"] = bool(sec[a] == sec[b] and tracks[(sec[a], role[a])] & tracks[(sec[b], role[b])])
            row["skills_to_add"] = "; ".join(skills[(B[b] == 1) & (B[a] == 0)])
            rows.append(row)
    return pd.DataFrame(rows)


def skill_adjacency(g: Bipartite, k: int = 5, min_roles: int = 5) -> pd.DataFrame:
    """For each skill required by at least `min_roles` roles: the k skills most often required with it.

    Ranked by Jaccard similarity of the two skills' role sets (ties by co-occurrence, then order).
    """
    B = g.B
    co = shared(B.T)
    deg = np.diag(co).copy()
    union = deg[:, None] + deg[None, :] - co
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.round(np.where(union > 0, co / union, 0.0), SCORE_DECIMALS)
    eligible = (co > 0) & (deg[None, :] >= min_roles)
    np.fill_diagonal(eligible, False)
    eligible[deg < min_roles] = False
    top = rank_targets(-jac, co, eligible, k)
    rows = []
    for a in range(len(g.skills)):
        for rank, b in enumerate(top[a], 1):
            if b >= 0:
                rows.append({"skill": g.skills[a], "roles": int(deg[a]), "rank": rank, "adjacent_skill": g.skills[b],
                             "adjacent_roles": int(deg[b]), "roles_requiring_both": int(co[a, b]),
                             "jaccard": round(float(jac[a, b]), 4)})
    return pd.DataFrame(rows)


# ---------- communities ----------
def louvain(B: np.ndarray, w: np.ndarray | None = None, seed: int = SEED, resolution: float = 1.0):
    """Louvain communities on the bipartite graph (roles and skills are both nodes).

    Returns (role_labels, skill_labels). Communities are numbered by size, largest first
    (ties by smallest member index), so labels do not depend on set iteration order.
    Skills with no weighted edge get label -1.
    """
    import networkx as nx

    nr, ns = B.shape
    w = np.ones(ns) if w is None else np.round(np.asarray(w, dtype=float), 6)
    G = nx.Graph()
    G.add_nodes_from(range(nr))
    for i, j in zip(*np.nonzero(B)):
        if w[j] > 0:
            G.add_edge(int(i), int(nr + j), weight=float(w[j]))
    comms = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    comms = sorted((sorted(c) for c in comms), key=lambda c: (-len(c), c[0]))
    lab = np.full(nr + ns, -1, dtype=np.int64)
    for cid, members in enumerate(comms):
        lab[members] = cid
    return lab[:nr], lab[nr:]


def purity(labels, reference) -> float:
    """Share of items whose community's most common reference class is their own."""
    t = pd.crosstab(pd.Series(list(labels), name="c"), pd.Series(list(reference), name="r"))
    return float(t.max(axis=1).sum() / t.to_numpy().sum())


def community_table(g: Bipartite, role_lab: np.ndarray, skill_lab: np.ndarray, n_skills: int = 6) -> pd.DataFrame:
    """One row per community: size, the sectors its roles come from, and its most-required skills."""
    sec = g.roles["sector"].to_numpy()
    names_all = np.array(g.skills, dtype=object)
    rows = []
    for c in sorted(set(role_lab.tolist()) | set(skill_lab[skill_lab >= 0].tolist())):
        r_in, s_in = role_lab == c, skill_lab == c
        counts = pd.Series(sec[r_in]).value_counts().sort_index().sort_values(ascending=False, kind="stable")
        within = g.B[np.ix_(r_in, s_in)].sum(axis=0)
        names = names_all[s_in]
        order = np.lexsort((np.arange(len(names)), -within))[:n_skills]
        rows.append({
            "community": int(c), "roles": int(r_in.sum()), "skills": int(s_in.sum()),
            "sectors": int(len(counts)),
            "largest_sector": counts.index[0] if len(counts) else None,
            "largest_sector_share": round(float(counts.iloc[0] / r_in.sum()), 4) if len(counts) else None,
            "sector_counts": "; ".join(f"{s} ({n})" for s, n in counts.items()),
            "top_skills": "; ".join(names[order]),
        })
    return pd.DataFrame(rows)
