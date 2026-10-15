"""Check that every headline number quoted in README.md and the brief matches outputs/.

Each claim is rebuilt from outputs/*.json. If someone edits a number in the prose, or a
re-run changes a result, the check fails and names the claim.
    python tools/check_readme.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def claims(m: dict, mut: dict) -> list[tuple[str, str]]:
    o = m["mapping"]["reference_official"]
    c = m["mapping"]["reference_claude"]
    h, hc = o["hybrid"], c["hybrid"]
    w = m["wages"]["models"]
    f = m["forecast"]["mae_pp"]
    d = m["data"]
    agree = m["mapping"]["claude_vs_official"]
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
        ("fc_naive", f"naive forecast MAE {f['naive']:.2f} pp"),
        ("fc_holt", f"damped Holt {f['holt_damped']:.2f} pp"),
        ("fc_snaive", f"seasonal naive {f['seasonal_naive']:.2f} pp"),
        ("mut", f"{mut['killed']}/{mut['total']} mutants killed"),
    ]


# claims that must also appear verbatim in the brief
BRIEF = {"off_n", "off_top1", "off_top5", "agree", "ict_emerging", "ict_rate", "wage_raw", "wage_within", "fc_naive"}


def check(text: str, m: dict, mut: dict, where: str, only: set | None = None) -> list[str]:
    return [f"{where}: missing or changed [{k}] '{s}'" for k, s in claims(m, mut)
            if (only is None or k in only) and s not in text]


def main() -> int:
    m = json.loads((ROOT / "outputs" / "metrics.json").read_text(encoding="utf-8"))
    mut = json.loads((ROOT / "outputs" / "mutation_report.json").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    errs = check(readme, m, mut, "README.md")
    brief = (ROOT / "brief" / "jobs_skills_insights_brief.md").read_text(encoding="utf-8")
    errs += check(brief, m, mut, "brief", only=BRIEF)
    for e in errs:
        print(e)
    print(f"{len(claims(m, mut))} README claims checked, {len(errs)} problems")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
