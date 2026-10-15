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
