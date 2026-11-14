"""Mutation check: prove that the unit tests can fail.

Each mutant is a small, plausible bug in src/jsi. For every mutant we copy src/ and tests/
to a temporary directory, apply the edit, run the unit tests, and expect them to FAIL.
One CONTROL mutant is behaviour-preserving and must SURVIVE; if the harness reports it
as killed, the harness itself is broken (e.g. every run fails for an unrelated reason).

Exit code 0 only if: unmutated code passes, every real mutant is killed, control survives.
    python tools/mutation_check.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MUTANTS = [
    ("load: '-' treated as 1 instead of the requested value", "load.py",
     'return out.where(s != "-", dash)', 'return out.where(s != "-", 1.0)'),
    ("match: ranks worst-first", "match.py", "np.argsort(-scores", "np.argsort(scores"),
    ("match: top-k only ever looks at the first prediction", "match.py", "for c in p[:k]", "for c in p[:1]"),
    ("match: 4-digit gold codes no longer match by prefix", "match.py",
     "c[: min(level, len(gc))] == gc[:level]", "c[:level] == gc[:level]"),
    ("match: hybrid uses w twice", "match.py", "(1.0 - w) * b", "w * b"),
    ("indicators: every role counted as having an emerging skill", "indicators.py",
     'per_role["n_emerging"] > 0', 'per_role["n_emerging"] >= 0'),
    ("indicators: skill share divided by all roles, not roles in the industry", "indicators.py",
     '.div(n_roles, level="mom_industry")', ".div(n_roles.sum())"),
    ("indicators: growth divides by the end value", "indicators.py",
     "100.0 * (b - a) / a", "100.0 * (b - a) / b"),
    # The first version of this mutant (adding the overlapping PMET group to the filter) SURVIVED:
    # it is equivalent, because totals only ever sum the PME and non-PME columns.
    # See outputs/mutation_report_first_run.json. Replaced by a mutant that changes behaviour:
    ("indicators: total vacancies drop the PME group", "indicators.py",
     'p["total"] = p["pme"] + p["non_pme"]', 'p["total"] = p["non_pme"]'),
    ("forecast: seasonal naive uses a 3-quarter cycle", "forecast.py", "y[-4 + (i % 4)]", "y[-4 + (i % 3)]"),
    ("forecast: gaps no longer trimmed", "forecast.py",
     "start = len(s) - int(np.argmax(isna[::-1])) if isna.any() else 0", "start = 0"),
    ("forecast: training window leaks the first test quarter", "forecast.py",
     "train, test = y[:o], y[o:o + h]", "train, test = y[:o + 1], y[o:o + h]"),
    ("ssoc: track becomes mandatory in SFw entries", "ssoc.py",
     r"(?:\((?P<track>[^()]*)\))?\s*\(SFw", r"(?:\((?P<track>[^()]*)\))\s*\(SFw"),
    ("ssoc: slashes not normalised", "ssoc.py", 's = re.sub(r"\s*/\s*", "/", s)', "s = s"),
    ("wages: regression without intercept", "wages.py",
     "smf.ols(formula, data=t)", 'smf.ols(formula + " - 1", data=t)'),
    ("load: interim cache ignores the raw-file hash", "load.py",
     'p = INTERIM / f"{name}-{key}.pkl"', 'p = INTERIM / f"{name}.pkl"'),
    ("evaluate_mapping: majority baseline picks the rarest code", "evaluate_mapping.py",
     "most_common(1)[0][0]", "most_common()[-1][0]"),
    ("evaluate_mapping: 4-digit labels must equal the 5-digit code", "evaluate_mapping.py",
     "o[: len(c)] == c", "o == c"),
    ("evaluate_mapping: production keeps the last official code", "evaluate_mapping.py",
     '.str.split(";").str[0]', '.str.split(";").str[-1]'),
    # role-skill graph
    ("graph: skill gap counts the source's skills, not the target's", "graph.py",
     "return B.sum(axis=1)[None, :] - shared(B)", "return B.sum(axis=1)[:, None] - shared(B)"),
    ("graph: reachability measured against the source's skill count", "graph.py",
     "ok = inter >= min_held * deg[None, :]", "ok = inter >= min_held * deg[:, None]"),
    ("graph: ties go to the role with fewer shared skills", "graph.py",
     "np.lexsort((idx, -key_minor[a, idx], key_major[a, idx]))", "np.lexsort((idx, key_minor[a, idx], key_major[a, idx]))"),
    ("graph: weighted Jaccard divides by the sum, not the union", "graph.py",
     "union = tot[:, None] + tot[None, :] - inter", "union = tot[:, None] + tot[None, :]"),
    ("graph: equal levels counted as a level-up", "graph.py",
     "(g.level[b] > g.level[a]) & both", "(g.level[b] >= g.level[a]) & both"),
    ("graph: any positive level shift is called 'up'", "graph.py",
     'return "up" if shift > band', 'return "up" if shift > 0'),
    ("graph: rare skills are listed in the adjacency table", "graph.py",
     "eligible[deg < min_roles] = False", "pass"),
    ("graph: a repeated requirement keeps the last level, not the highest", "graph.py",
     "L[i, j] = lv if np.isnan(L[i, j]) else max(L[i, j], lv)", "L[i, j] = lv"),
    ("graph_eval: training edges compete in the ranking", "graph_eval.py",
     "cand = train[i] == 0", "cand = train[i] >= 0"),
    ("graph_eval: ties always resolved in favour of the held-out skill", "graph_eval.py",
     "tied_before = int((s[:j][cand[:j]] == s[j]).sum())", "tied_before = 0"),
    ("graph_eval: random walk rewards popular skills instead of penalising them", "graph_eval.py",
     "@ P_rs / ds[None, :] ** param", "@ P_rs * ds[None, :] ** param"),
    ("graph_eval: a role is its own nearest neighbour", "graph_eval.py",
     "np.fill_diagonal(S, 0.0)", "pass"),
    ("graph_eval: training graph keeps the held-out edges (leak)", "graph_eval.py",
     "parts = [perm[cuts[-1]:]] +", "parts = [perm] +"),
    ("graph_eval: parameter tuned on the test edges", "graph_eval.py",
     "val_scores = {p: hits_at(ranks(f(train, sectors, p), train, val)) for p in params}",
     "val_scores = {p: hits_at(ranks(f(train, sectors, p), train, test)) for p in params}"),
    ("graph_eval: sector popularity ignores the sector", "graph_eval.py",
     "out[m] = train[m].sum(axis=0) + glob", "out[m] = train.sum(axis=0) + glob"),
    ("graph_eval: bootstrap resamples edges, not roles", "graph_eval.py",
     "codes, inv = np.unique(role, return_inverse=True)", "codes, inv = np.unique(np.arange(len(role)), return_inverse=True)"),
    ("graph_eval: chance of a same-track draw computed with replacement", "graph_eval.py",
     "(n_pool - n_good - i) / np.maximum(n_pool - i, 1)", "(n_pool - n_good) / np.maximum(n_pool, 1)"),
    ("power: one-sided critical value", "power.py",
     "za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)", "za, zb = norm.ppf(1 - alpha), norm.ppf(power)"),
    ("power: sample size divides by the lift, not the lift squared", "power.py",
     "math.ceil(num ** 2 / (p2 - p1) ** 2)", "math.ceil(num ** 2 / abs(p2 - p1))"),
    ("power: design effect uses m instead of m - 1", "power.py",
     "1.0 + (cluster_size - 1.0) * icc", "1.0 + cluster_size * icc"),
]
CONTROL = ("CONTROL (behaviour-preserving): growth adds 0.0", "indicators.py",
           "100.0 * (b - a) / a", "100.0 * (b - a) / a + 0.0")


def run_tests(tmp: Path) -> int:
    env = {**__import__("os").environ, "PYTHONPATH": str(tmp / "src"), "PYTHONUTF8": "1"}
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                        "--rootdir", str(tmp), str(tmp / "tests" / "unit")],
                       cwd=tmp, env=env, capture_output=True, text=True)
    return r.returncode


def trial(fname: str | None, old: str | None, new: str | None) -> int:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        shutil.copytree(ROOT / "src", tmp / "src", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "tests" / "unit", tmp / "tests" / "unit", ignore=shutil.ignore_patterns("__pycache__"))
        for extra in ("tools", "outputs", "brief", "docs"):
            shutil.copytree(ROOT / extra, tmp / extra, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(ROOT / "README.md", tmp / "README.md")
        if fname:
            p = tmp / "src" / "jsi" / fname
            src = p.read_text(encoding="utf-8")
            if src.count(old) != 1:
                raise SystemExit(f"mutant anchor not found exactly once in {fname}: {old!r}")
            p.write_text(src.replace(old, new), encoding="utf-8")
        return run_tests(tmp)


def main() -> int:
    report = {"baseline_exit": trial(None, None, None), "mutants": []}
    ok = report["baseline_exit"] == 0
    print(f"baseline (unmutated): exit {report['baseline_exit']}")
    for name, f, old, new in MUTANTS:
        code = trial(f, old, new)
        killed = code != 0
        ok &= killed
        report["mutants"].append({"mutant": name, "killed": killed})
        print(("KILLED  " if killed else "SURVIVED") + "  " + name)
    code = trial(*CONTROL[1:])
    report["control_survived"] = code == 0
    ok &= code == 0
    print(("survived" if code == 0 else "KILLED (harness broken)") + "  " + CONTROL[0])
    k = sum(m["killed"] for m in report["mutants"])
    report["killed"], report["total"] = k, len(MUTANTS)
    print(f"{k}/{len(MUTANTS)} mutants killed; control survived: {report['control_survived']}")
    (ROOT / "outputs" / "mutation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
