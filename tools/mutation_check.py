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
        for extra in ("tools", "outputs", "brief"):
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
    (ROOT / "outputs" / "mutation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
