"""Evaluate the role -> SSOC matchers against two references and write outputs/.

Reference A (primary): SingStat's official SFw job role -> SSOC 2020 codes (alphabetical index).
Reference B (secondary): 100 roles labelled by Claude (AI assistant) before any matcher ran.

    python -m jsi.evaluate_mapping            # TF-IDF methods only (CI)
    python -m jsi.evaluate_mapping --embed    # also bge-small embeddings + hybrid (local)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import load, match, ssoc

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
GOLD_B = ROOT / "data" / "labels" / "gold_role_to_occupation.csv"


def claude_label_codes() -> pd.DataFrame:
    """Turn the acceptable wage-table titles into SSOC codes (5-digit, or 4-digit unit group)."""
    g = pd.read_csv(GOLD_B)
    occ = ssoc.occupations()
    t5 = dict(zip(occ["title"].map(ssoc.norm), occ["code"]))
    # the few wage titles that are SSOC 4-digit group titles
    defs = pd.read_excel(ssoc.DEFS, header=4, dtype=str)
    defs.columns = ["code", "title"] + list(defs.columns[2:])
    t4 = dict(zip(defs.loc[defs["code"].str.fullmatch(r"\d{4}", na=False), "title"].map(ssoc.norm),
                  defs.loc[defs["code"].str.fullmatch(r"\d{4}", na=False), "code"]))
    manual4 = {  # wage-table titles that are worded differently from SSOC titles
        ssoc.norm("registered nurse and other nursing professional (e.g. clinical nurse, nurse educator, excluding enrolled nurse)"): "22200",
    }
    codes = []
    for v in g["acceptable_occupations"]:
        if v == "NONE":
            codes.append(set())
            continue
        cs = set()
        for t in v.split(";"):
            k = ssoc.norm(t)
            c = t5.get(k) or t4.get(k) or manual4.get(k)
            if c is None:
                raise KeyError(f"label title not found in SSOC 2020: {t}")
            cs.add(c)
        codes.append(cs)
    g["codes"] = codes
    return g


def run(use_embed: bool) -> dict:
    roles = load.roles()
    occ = ssoc.occupations()
    codes = list(occ["code"])
    q_full = match.role_text(roles, True)
    q_title = match.role_text(roles, False)
    d = match.occ_text(occ)

    S = {
        "tfidf_word_title_only": match.tfidf_scores(q_title, d, "word"),
        "tfidf_word": match.tfidf_scores(q_full, d, "word"),
        "tfidf_char": match.tfidf_scores(q_full, d, "char"),
    }
    if use_embed:
        S["embed"] = match.embed_scores(q_full, d)
        S["hybrid"] = match.hybrid(S["tfidf_word"], S["embed"])

    key = roles["sector"] + "||" + roles["job_role"]
    idx = {k: i for i, k in enumerate(key)}

    off = ssoc.official_crosswalk()
    off_i = [idx[s + "||" + r] for s, r in zip(off["sector"], off["job_role"])]
    off_gold = [set(c.split(";")) for c in off["official_codes"]]

    cl = claude_label_codes()
    cl = cl[cl["codes"].map(len) > 0].reset_index(drop=True)
    cl_i = [idx[s + "||" + r] for s, r in zip(cl["sector"], cl["job_role"])]
    cl_gold = list(cl["codes"])

    # majority-class baseline: always predict the most common official code
    from collections import Counter

    maj = Counter(c for g in off_gold for c in g).most_common(1)[0][0]

    results = {"n_roles": len(roles), "n_ssoc_occupations": len(occ),
               "reference_official": {"n": len(off_i)}, "reference_claude": {"n": len(cl_i)}}
    results["reference_official"]["majority_baseline_top1"] = float(
        np.mean([maj in g for g in off_gold]))
    for name, sc in S.items():
        preds = match.topk(sc, codes, 5)
        for ref, ii, gold in [("reference_official", off_i, off_gold), ("reference_claude", cl_i, cl_gold)]:
            p = [preds[i] for i in ii]
            r = {}
            for lvl in (5, 4, 2, 1):
                h = match.hit_at_k(p, gold, 1, lvl)
                r[f"top1_{lvl}digit"] = round(float(h.mean()), 4)
                if lvl == 5:
                    lo, hi = match.bootstrap_ci(h)
                    r["top1_5digit_ci95"] = [round(lo, 4), round(hi, 4)]
            h5 = match.hit_at_k(p, gold, 5, 5)
            r["top5_5digit"] = round(float(h5.mean()), 4)
            results[ref][name] = r

    # agreement between the Claude labels and the official codes, where both exist
    both = cl.merge(off, on=["sector", "job_role"], how="inner")
    agree = [int(any(o[: len(c)] == c for o in oc.split(";") for c in cs))
             for cs, oc in zip(both["codes"], both["official_codes"])]
    results["claude_vs_official"] = {"n_overlap": len(both), "agree": int(sum(agree))}

    # production mapping: official code where SingStat gives one, else best matcher
    best = "hybrid" if use_embed else "tfidf_word"
    top = match.topk(S[best], codes, 1)
    sc_top = S[best].max(axis=1)
    m = roles[["sector", "job_role"]].copy()
    m["predicted_code"] = [t[0] for t in top]
    m["predicted_score"] = np.round(sc_top, 4)
    m = m.merge(off, on=["sector", "job_role"], how="left")
    m["code"] = m["official_codes"].str.split(";").str[0].fillna(m["predicted_code"])
    m["code_source"] = np.where(m["official_codes"].notna(), "official", f"predicted:{best}")
    m = m.merge(occ[["code", "title"]].rename(columns={"title": "ssoc_title"}), on="code", how="left")
    results["production_mapping"] = {"method_for_unmapped": best,
                                     "official": int((m["code_source"] == "official").sum()),
                                     "predicted": int((m["code_source"] != "official").sum())}
    OUT.mkdir(exist_ok=True)
    m.to_csv(OUT / ("role_to_ssoc.csv" if use_embed else "role_to_ssoc_tfidf.csv"), index=False, lineterminator="\n")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true")
    a = ap.parse_args(argv)
    res = run(a.embed)
    name = "mapping_eval.json" if a.embed else "mapping_eval_tfidf.json"
    (OUT / name).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
