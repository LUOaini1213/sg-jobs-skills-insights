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


def _graph_inputs():
    gm = json.loads((ROOT / "outputs" / "graph_metrics.json").read_text(encoding="utf-8"))
    pw = json.loads((ROOT / "outputs" / "experiment_power.json").read_text(encoding="utf-8"))
    return gm, pw


def test_graph_numbers_are_checked():
    gm, pw = _graph_inputs()
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert check_readme.check(text, {}, {}, "README", items=check_readme.graph_claims(gm, pw)) == []
    head = text.split("## Headline results", 1)[0]
    assert check_readme.check(head, {}, {}, "head", items=check_readme.first_screen_graph_claims(gm)) == []
    gm["link_prediction"]["methods"]["jaccard_knn"]["hits_at_10"] -= 0.05     # a re-run that changes the baseline
    errs = check_readme.check(text, {}, {}, "README", items=check_readme.graph_claims(gm, pw))
    assert len(errs) == 1 and "g_lp" in errs[0]
    gm, pw = _graph_inputs()
    gm["link_prediction"]["verdict"] = "model better"                         # a verdict edited to flatter the model
    errs = check_readme.check(text, {}, {}, "README", items=check_readme.graph_claims(gm, pw))
    assert len(errs) == 1 and "g_lp_diff" in errs[0]


def test_experiment_design_numbers_are_checked():
    _, pw = _graph_inputs()
    text = (ROOT / "docs" / "experiment_design.md").read_text(encoding="utf-8")
    assert check_readme.check(text, {}, {}, "docs", items=check_readme.design_claims(pw)) == []
    pw["n_per_arm"] += 1
    errs = check_readme.check(text, {}, {}, "docs", items=check_readme.design_claims(pw))
    assert len(errs) == 1 and "d_n" in errs[0]
    _, pw = _graph_inputs()
    pw["sensitivity"][0]["n_per_arm"] = 80000
    errs = check_readme.check(text, {}, {}, "docs", items=check_readme.design_claims(pw))
    assert len(errs) == 1 and "d_grid_0" in errs[0]


def test_community_groups_are_checked(tmp_path):
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert check_readme.check(text, {}, {}, "README", items=check_readme.community_claims(ROOT / "outputs")) == []
    csv_text = (ROOT / "outputs" / "skill_communities.csv").read_text(encoding="utf-8")
    bad = csv_text.replace("Retail (24)", "Retail (19)")     # Retail would drop out of its mixed community
    assert bad != csv_text
    (tmp_path / "skill_communities.csv").write_text(bad, encoding="utf-8")
    errs = check_readme.check(text, {}, {}, "README", items=check_readme.community_claims(tmp_path))
    assert len(errs) == 1 and "g_mixed_sectors" in errs[0]
