"""Re-run whatever the reachable sources allow and require byte-identical outputs.

Government sites differ in whether they accept connections from cloud CI runners, so this
script checks which raw files exist and regenerates only the outputs that depend on them:
  Skills Framework files            -> outputs/sector_skill_profile.csv
  + MOM (data.gov.sg) tables        -> sector_demand, skill_demand_index, emerging_skills, forecast
  + SingStat SSOC files             -> full TF-IDF pipeline
Prints what was and was not reproduced; exits 1 on any difference.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jsi import indicators, load, pipeline, ssoc  # noqa: E402

SFW = [load.SFW, load.UNIQUE, load.TSCMAP]
MOM = [load.VAC_N, load.VAC_RATE, load.VAC_RATE_Q]
SSOC = [ssoc.DEFS, ssoc.ALPHA, load.WAGES]


def have(paths) -> bool:
    return all(p.exists() for p in paths)


def main() -> int:
    if not have(SFW):
        print("Skills Framework files unavailable; nothing to reproduce")
        return 1
    if have(SFW + MOM + SSOC):
        pipeline.run(embed=False)
        files = ["outputs/"]
        print("reproduced: full TF-IDF pipeline")
    elif have(SFW + MOM):
        pipeline.run(embed=False, indicators_only=True)
        files = [f"outputs/{f}" for f in ("sector_skill_profile.csv", "sector_demand.csv", "skill_demand_index.csv",
                                          "emerging_skills.csv", "forecast_mae_ci.csv")]
        print("reproduced: indicators + forecast; NOT reproduced here: SSOC mapping, wages")
    else:
        indicators.sector_skill_profile().round(4).to_csv(ROOT / "outputs" / "sector_skill_profile.csv", index=False, lineterminator="\n")
        files = ["outputs/sector_skill_profile.csv"]
        print("reproduced: sector skill profiles; NOT reproduced here: MOM-based indicators, forecast, SSOC mapping, wages")
    r = subprocess.run(["git", "diff", "--stat", "--exit-code", "--", *files], cwd=ROOT)
    print("identical to committed outputs" if r.returncode == 0 else "DIFFERS from committed outputs")
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
