"""Check that the result numbers quoted in README.md and the brief match outputs/.

Each claim is rebuilt from outputs/*.json and outputs/*.csv, and the test counts from
`pytest --collect-only`. If someone edits a number in the prose, or a re-run changes a
result, the check fails and names the claim. Whitespace (line wrapping) is ignored.
    python tools/check_readme.py
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def ws(text: str) -> str:
    return " ".join(text.split())


def claims(m: dict, mut: dict) -> list[tuple[str, str]]:
    o = m["mapping"]["reference_official"]
    c = m["mapping"]["reference_claude"]
    h, hc = o["hybrid"], c["hybrid"]
    w = m["wages"]["models"]
    fc = m["forecast"]
    f = fc["mae_pp"]
    d = m["data"]
    agree = m["mapping"]["claude_vs_official"]
    prod = m["mapping"]["production_mapping"]
    return [
        ("n_roles", f"{d['sfw_roles']:,} job roles in {d['sfw_sectors']} sectors"),
        ("n_req", f"{d['role_skill_requirements']:,} role-skill requirements"),
        ("n_emerging", f"{d['ssg_emerging_skills_required']} skills flagged as emerging by SSG"),
        ("emerging_share", f"{pct(d['emerging_share_of_requirements'])} of all requirements"),
        ("off_n", f"{o['n']:,} roles that SingStat has coded"),
        ("off_top1", f"top-1 accuracy {pct(h['top1_5digit'])} (95% CI {pct(h['top1_5digit_ci95'][0])}–{pct(h['top1_5digit_ci95'][1])})"),
        ("off_top5", f"top-5 accuracy {pct(h['top5_5digit'])}"),
        ("off_major", f"major group right {pct(h['top1_1digit'])}"),
        ("off_tfidf", f"TF-IDF alone {pct(o['tfidf_word']['top1_5digit'])}"),
        ("off_embed", f"embeddings alone {pct(o['embed']['top1_5digit'])}"),
        ("off_majority", f"majority-class baseline {pct(o['majority_baseline_top1'])}"),
        ("cl_top1", f"{hc['top1_5digit'] * c['n']:.0f} of {c['n']} ({pct(hc['top1_5digit'])}) top-1"),
        ("agree", f"agreed with SingStat on {agree['agree']} of {agree['n_overlap']}"),
        ("ict_emerging", f"Infocomm Technology: {pct(m['sector']['highest_emerging_share']['emerging_share_of_requirements'])}"),
        ("ict_rate", f"PME vacancy rate of {m['sector']['highest_pme_vacancy_rate_2025']['pme_vacancy_rate_2025']:.1f}% in 2025"),
        ("wage_raw", f"+{w['raw']['pct_wage_per_10pp']:.1f}% (95% CI {w['raw']['pct_ci95_per_10pp'][0]:.1f} to {w['raw']['pct_ci95_per_10pp'][1]:.1f})"),
        ("wage_ctrl", f"+{w['plus_controls']['pct_wage_per_10pp']:.1f}% ({w['plus_controls']['pct_ci95_per_10pp'][0]:.1f} to {w['plus_controls']['pct_ci95_per_10pp'][1]:.1f})"),
        ("wage_within", f"+{w['within_submajor']['pct_wage_per_10pp']:.1f}% ({w['within_submajor']['pct_ci95_per_10pp'][0]:.1f} to {w['within_submajor']['pct_ci95_per_10pp'][1]:.1f})"),
        ("wage_n", f"{m['wages']['n_roles']:,} roles in {m['wages']['n_occupations']} occupations"),
        ("fc_scope", f"1–{fc['horizon_quarters']} quarters ahead, {fc['n_industries']} industries"),
        ("fc_naive", f"naive forecast MAE {f['naive']:.2f} pp"),
        ("fc_holt", f"damped Holt {f['holt_damped']:.2f} pp"),
        ("fc_mean4", f"mean of the last four quarters {f['mean_last4']:.2f} pp"),
        ("fc_snaive", f"seasonal naive {f['seasonal_naive']:.2f} pp"),
        ("mut", f"{mut['killed']}/{mut['total']} mutants killed"),
        ("mut_control", "a behaviour-preserving control mutant survives" if mut["control_survived"]
         else "control mutant was KILLED"),
        ("mut_n", f"applies {mut['total']} plausible bugs"),
        ("n_occ", f"{m['mapping']['n_ssoc_occupations']} five-digit occupations"),
        ("mapped", f"{pct(d['requirements_mapped_share'])} of requirements map"),
        ("direct", f"{m['sector']['direct_fit_sectors']} are direct"),
        ("prod_split", f"SingStat's code where one exists ({prod['official']:,}) and the "
                       f"{prod['method_for_unmapped']} prediction otherwise ({prod['predicted']:,})"),
    ]


# claims that must also appear in the brief
BRIEF = {"off_n", "off_top1", "off_top5", "agree", "ict_emerging", "ict_rate", "wage_raw", "wage_within", "fc_naive"}


def first_screen_claims(m: dict) -> list[tuple[str, str]]:
    """Numbers that must appear in the README above '## Headline results'."""
    o = m["mapping"]["reference_official"]
    w = m["wages"]["models"]
    d = m["data"]
    return [
        ("first_roles", f"{d['sfw_roles']:,} job roles in {d['sfw_sectors']} sectors"),
        ("first_top1", f"first for {pct(o['hybrid']['top1_5digit'])} of the {o['n']:,} roles SingStat has coded"),
        ("first_top5", f"top-5 {pct(o['hybrid']['top5_5digit'])}"),
        ("first_wage_raw", f"+{w['raw']['pct_wage_per_10pp']:.1f}% per 10 pp"),
        ("first_wage_within", f"+{w['within_submajor']['pct_wage_per_10pp']:.1f}% within SSOC sub-major groups"),
        ("first_fc", f"naive forecast (MAE {m['forecast']['mae_pp']['naive']:.2f} pp)"),
    ]


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def brief_csv_claims(m: dict, out: Path) -> list[tuple[str, str]]:
    """The brief's key points 1-4, rebuilt from the committed indicator CSVs."""
    dem = {r["sector"]: r for r in _rows(out / "sector_demand.csv")}
    prof = {r["sector"]: r for r in _rows(out / "sector_skill_profile.csv")}
    em = _rows(out / "emerging_skills.csv")

    def n(x: str) -> int:
        return int(round(float(x)))

    ict, ec, be = dem["Infocomm Technology"], dem["Early Childhood"], dem["Built Environment"]
    v25, pme25 = n(be["vacancies_2025"]), n(be["pme_vacancies_2025"])
    v19 = int(round(v25 / (1 + float(be["vacancy_growth_2019_2025_pct"]) / 100) / 100) * 100)
    s = [n(r["sectors"]) for r in em[:4]]
    pair = " and ".join(sorted(r["unique_skill"] for r in em[2:4]))
    tail = f"{pair} in {s[2]} each" if s[2] == s[3] else f"{em[2]['unique_skill']} in {s[2]}"
    return [
        ("brief_ict_pme", f"with {n(ict['pme_vacancies_2025']):,} PME vacancies"),
        ("brief_pa", f"{ec['mom_industry'].capitalize()} had {n(ec['pme_vacancies_2025']):,} PME vacancies in "
                     f"2025 (rate {float(ec['pme_vacancy_rate_2025']):.1f}%)"),
        ("brief_ec_emerging", f"only {pct(float(prof['Early Childhood']['emerging_share_of_requirements']))} "
                              "of its requirements are emerging skills"),
        ("brief_construction", f"vacancies rose from {v19:,} (2019) to {v25:,} (2025), and "
                               f"{v25 - pme25:,} of the 2025 vacancies were non-PME"),
        ("brief_cross", f"{em[0]['unique_skill']} appears in {s[0]} of the {m['data']['sfw_sectors']} sectors, "
                        f"{em[1]['unique_skill']} in {s[1]}, and {tail}"),
    ]


def test_count_claim(n_unit: int, n_data: int) -> tuple[str, str]:
    return ("tests", f"{n_unit + n_data} tests ({n_unit} unit tests on hand-computed inputs, {n_data} on the real data)")


test_count_claim.__test__ = False  # not a pytest test


def collect_test_counts() -> tuple[int, int]:
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                        "tests/unit", "tests/data"], cwd=ROOT, capture_output=True, text=True)
    ids = [ln for ln in r.stdout.splitlines() if "::" in ln]
    if r.returncode != 0 or not ids:
        raise SystemExit("pytest --collect-only failed:\n" + r.stdout + r.stderr)
    norm = [i.replace("\\", "/") for i in ids]
    return sum(i.startswith("tests/unit/") for i in norm), sum(i.startswith("tests/data/") for i in norm)


def check(text: str, m: dict, mut: dict, where: str, only: set | None = None, items=None) -> list[str]:
    flat = ws(text)
    items = claims(m, mut) if items is None else items
    return [f"{where}: missing or changed [{k}] '{s}'" for k, s in items
            if (only is None or k in only) and ws(s) not in flat]


def main() -> int:
    m = json.loads((ROOT / "outputs" / "metrics.json").read_text(encoding="utf-8"))
    mut = json.loads((ROOT / "outputs" / "mutation_report.json").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    brief = (ROOT / "brief" / "jobs_skills_insights_brief.md").read_text(encoding="utf-8")
    head = readme.split("## Headline results", 1)[0]
    tests = [test_count_claim(*collect_test_counts())]
    groups = [
        (readme, "README.md", None, claims(m, mut)),
        (head, "README.md first screen", None, first_screen_claims(m)),
        (readme, "README.md", None, tests),
        (brief, "brief", BRIEF, claims(m, mut)),
        (brief, "brief", None, brief_csv_claims(m, ROOT / "outputs")),
    ]
    errs, total = [], 0
    for text, where, only, items in groups:
        errs += check(text, m, mut, where, only, items)
        total += len(items) if only is None else len(only)
    for e in errs:
        print(e)
    print(f"{total} README and brief claims checked, {len(errs)} problems")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
