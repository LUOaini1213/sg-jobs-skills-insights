"""The README number check must be able to fail."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import check_readme  # noqa: E402


def _inputs():
    m = json.loads((ROOT / "outputs" / "metrics.json").read_text(encoding="utf-8"))
    mut = json.loads((ROOT / "outputs" / "mutation_report.json").read_text(encoding="utf-8"))
    return m, mut


def test_current_readme_passes():
    m, mut = _inputs()
    assert check_readme.check((ROOT / "README.md").read_text(encoding="utf-8"), m, mut, "README") == []


def test_an_edited_number_is_caught():
    m, mut = _inputs()
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    claim = dict(check_readme.claims(m, mut))["off_top1"]
    bad = text.replace(claim, claim.replace("45.0%", "48.0%"))
    assert bad != text
    errs = check_readme.check(bad, m, mut, "README")
    assert len(errs) == 1 and "off_top1" in errs[0]


def test_a_changed_result_is_caught():
    m, mut = _inputs()
    m["mapping"]["reference_official"]["hybrid"]["top5_5digit"] += 0.01
    errs = check_readme.check((ROOT / "README.md").read_text(encoding="utf-8"), m, mut, "README")
    assert any("off_top5" in e for e in errs)


def test_brief_figures_from_csv_are_checked():
    m, mut = _inputs()
    text = (ROOT / "brief" / "jobs_skills_insights_brief.md").read_text(encoding="utf-8")
    items = check_readme.brief_csv_claims(m, ROOT / "outputs")
    assert check_readme.check(text, m, mut, "brief", items=items) == []
    bad = text.replace("10,300 (2025)", "10,800 (2025)")
    assert bad != text
    errs = check_readme.check(bad, m, mut, "brief", items=items)
    assert len(errs) == 1 and "brief_construction" in errs[0]


def test_first_screen_numbers_are_checked():
    m, mut = _inputs()
    head = (ROOT / "README.md").read_text(encoding="utf-8").split("## Headline results", 1)[0]
    items = check_readme.first_screen_claims(m)
    assert check_readme.check(head, m, mut, "head", items=items) == []
    m["forecast"]["mae_pp"]["naive"] = 0.81
    errs = check_readme.check(head, m, mut, "head", items=check_readme.first_screen_claims(m))
    assert len(errs) == 1 and "first_fc" in errs[0]


def test_stated_test_counts_are_checked():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    m, mut = _inputs()
    claim = check_readme.test_count_claim(20, 6)
    assert check_readme.check(text, m, mut, "README", items=[claim]) != []  # stale count is caught
