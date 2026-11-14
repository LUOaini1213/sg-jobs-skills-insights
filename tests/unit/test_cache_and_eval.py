"""Unit tests for the interim cache and the evaluation glue behind the headline metrics.

Small hand-built inputs; no downloads."""
import pandas as pd

from jsi import evaluate_mapping, load


# ---------- interim cache ----------
def test_cache_is_rebuilt_when_a_raw_file_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(load, "INTERIM", tmp_path / "interim")
    raw = tmp_path / "table.csv"
    raw.write_text("a\n1\n", encoding="utf-8")
    calls = []

    def build():
        calls.append(1)
        return pd.read_csv(raw)

    first = load._cached("t", build, (raw,))
    again = load._cached("t", build, (raw,))
    assert len(first) == len(again) == 1 and len(calls) == 1  # unchanged file: served from cache

    raw.write_text("a\n1\n2\n3\n", encoding="utf-8")  # a new release of the raw file
    rebuilt = load._cached("t", build, (raw,))
    assert len(calls) == 2 and list(rebuilt["a"]) == [1, 2, 3]
    assert len(list((tmp_path / "interim").glob("t*.pkl"))) == 1  # the stale pickle is removed


def test_cache_key_depends_on_every_source(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("x\n", encoding="utf-8")
    b.write_text("y\n", encoding="utf-8")
    k = load.source_key((a, b))
    b.write_text("y\nz\n", encoding="utf-8")
    assert load.source_key((a, b)) != k
    assert load.source_key((a,)) != load.source_key((a, b))


# ---------- evaluation glue ----------
def test_majority_baseline_counts_each_acceptable_code():
    # "11111" occurs in 3 of 5 gold sets (one of them with a second acceptable code)
    gold = [{"11111"}, {"11111", "22222"}, {"22222"}, {"11111"}, {"33333"}]
    assert evaluate_mapping.majority_baseline_top1(gold) == 3 / 5


def _refs():
    official = pd.DataFrame({
        "sector": ["S1", "S1", "S2", "S2"],
        "job_role": ["A", "B", "C", "D"],
        "official_codes": ["12345", "21110;21120", "31111", "41100"],
    })
    labels = pd.DataFrame({
        "sector": ["S1", "S1", "S2", "S2", "S3"],
        "job_role": ["A", "B", "C", "D", "E"],
        "codes": [{"12345"}, {"2112"}, {"31112"}, {"99999", "41100"}, {"51111"}],
    })
    return labels, official


def test_label_agreement_exact_prefix_and_overlap():
    labels, official = _refs()
    # A exact; B 4-digit group 2112 is a prefix of 21120; C differs; D one of two codes matches;
    # E has no official code and is not in the overlap
    assert evaluate_mapping.label_agreement(labels, official) == {"n_overlap": 4, "agree": 3}


def test_assign_codes_prefers_the_first_official_code():
    _, official = _refs()
    roles = pd.DataFrame({"sector": ["S1", "S1", "S9"], "job_role": ["A", "B", "Z"]})
    m = evaluate_mapping.assign_codes(roles, official, ["99991", "99992", "99993"], "hybrid")
    assert list(m["code"]) == ["12345", "21110", "99993"]
    assert list(m["code_source"]) == ["official", "official", "predicted:hybrid"]
