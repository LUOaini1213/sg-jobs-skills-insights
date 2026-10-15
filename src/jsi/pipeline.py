"""Run every analysis step and write aggregates to outputs/.

    python -m jsi.pipeline            # everything except embeddings (what CI reproduces)
    python -m jsi.pipeline --embed    # also the bge-small embedding matcher (local, ~4 min)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import evaluate_mapping, forecast, indicators, load, wages

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"


def _round(df: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    return df.round(n)


def run(embed: bool, indicators_only: bool = False) -> dict:
    OUT.mkdir(exist_ok=True)
    rs = load.role_skills()
    roles = load.roles()

    prof = indicators.sector_skill_profile(rs)
    _round(prof).to_csv(OUT / "sector_skill_profile.csv", index=False, lineterminator="\n")
    dem = indicators.sector_demand()
    _round(dem).to_csv(OUT / "sector_demand.csv", index=False, lineterminator="\n")
    idx = indicators.skill_demand_index(rs=rs)
    _round(idx, 1).to_csv(OUT / "skill_demand_index.csv", index=False, lineterminator="\n")

    em = idx[idx["emerging"]].sort_values(["sectors", "roles"], ascending=False)
    _round(em, 1).to_csv(OUT / "emerging_skills.csv", index=False, lineterminator="\n")
    if indicators_only:  # no SSOC files needed (CI runners cannot reach SingStat)
        _round(pd.DataFrame([forecast.run()["mae_pp"]]), 3).to_csv(OUT / "forecast_mae_ci.csv", index=False, lineterminator="\n")
        return {"data": {"indicators_only": True}}

    # mapping evaluation (writes outputs/role_to_ssoc.csv)
    mapping = evaluate_mapping.run(embed)
    (OUT / ("mapping_eval.json" if embed else "mapping_eval_tfidf.json")).write_text(
        json.dumps(mapping, indent=2) + "\n", encoding="utf-8")

    w = wages.run()
    fc = forecast.run()

    per_role = rs.groupby(["sector", "job_role"]).agg(n=("unique_skill", "nunique"), e=("emerging", "sum"))
    joined = prof.merge(dem, on="sector")
    direct = joined[joined["fit"] == "direct"]
    metrics = {
        "data": {
            "sfw_sectors": int(roles["sector"].nunique()),
            "sfw_roles": int(len(roles)),
            "sfw_roles_with_skills": int(len(per_role)),
            "role_skill_requirements": int(len(rs)),
            "unique_skills_required": int(rs["unique_skill"].nunique()),
            "ssg_emerging_skills_required": int(rs.loc[rs["emerging"], "unique_skill"].nunique()),
            "emerging_share_of_requirements": round(float(per_role["e"].sum() / per_role["n"].sum()), 4),
        },
        "sector": {
            "highest_emerging_share": prof.sort_values("emerging_share_of_requirements").iloc[-1][
                ["sector", "emerging_share_of_requirements"]].to_dict(),
            "lowest_emerging_share_nonzero": prof[prof["emerging_share_of_requirements"] > 0]
            .sort_values("emerging_share_of_requirements").iloc[0][["sector", "emerging_share_of_requirements"]].to_dict(),
            "sectors_with_zero_emerging": sorted(prof.loc[prof["emerging_share_of_requirements"] == 0, "sector"]),
            "highest_pme_vacancy_rate_2025": dem.dropna(subset=["pme_vacancy_rate_2025"])
            .sort_values("pme_vacancy_rate_2025").iloc[-1][["sector", "mom_industry", "pme_vacancy_rate_2025"]].to_dict(),
            "direct_fit_sectors": int(len(direct)),
        },
        "skill_index_top5_2025": idx.head(5)[["unique_skill", "index_2025", "growth_2019_2025_pct"]].round(1).to_dict("records"),
        "emerging_broadest": em.head(5)[["unique_skill", "sectors", "roles"]].to_dict("records"),
        "mapping": mapping if embed else {k: mapping[k] for k in ("reference_official", "reference_claude", "claude_vs_official")},
        "wages": w,
        "forecast": fc,
    }
    for k, v in metrics["sector"].items():
        if isinstance(v, dict):
            metrics["sector"][k] = {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()}
    name = "metrics.json" if embed else "metrics_ci.json"
    (OUT / name).write_text(json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8")
    return metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--indicators-only", action="store_true", help="skip SSOC-dependent steps")
    a = ap.parse_args(argv)
    m = run(a.embed, a.indicators_only)
    print(json.dumps(m["data"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
