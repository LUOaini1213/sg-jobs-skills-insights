"""Download the public source files into data/raw/ and record their SHA-256.

Sources (all public, no login):
  * SkillsFuture Skills Framework dataset, Unique Skills List and the
    TSC-to-Unique-Skills mapping file, published on the Jobs-Skills Portal
    (jobsandskills.skillsfuture.gov.sg -> "Download the Latest Skills Framework Dataset").
  * SingStat SSOC 2020 detailed definitions and alphabetical index (singstat.gov.sg).
  * Ministry of Manpower labour-market tables on data.gov.sg
    (Singapore Open Data Licence v1.0).

Raw files are NOT committed (see .gitignore); only scripts and aggregates are.
Run:  python -m jsi.fetch           (re-download; compare with data/sources.json)
      python -m jsi.fetch --check   (only verify hashes of files already present)
      python -m jsi.fetch --update  (accept a new release and record its hashes)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "sources.json"

SKILLSFUTURE_FILES = {
    "skills_framework_dataset.xlsx": "https://file.go.gov.sg/jobsandskills-skillsfuture-skills-framework-dataset.xlsx",
    "unique_skills_list.xlsx": "https://file.go.gov.sg/jobsandskills-skillsfuture-unique-skills-list.xlsx",
    "tsc_to_unique_skills_mapping.xlsx": "https://file.go.gov.sg/jobsandskills-skillsfuture-tsc-to-unique-skills-mapping.xlsx",
}

SINGSTAT_FILES = {
    "ssoc2020_detailed_definitions.xlsx": "https://www.singstat.gov.sg/files/bbd1b6a5-9246-4f73-8d57-34fbd0a3c084.xlsx",
    "ssoc2020_alpha_index.xlsx": "https://www.singstat.gov.sg/files/03ca0c44-8d8d-4c58-8b51-3025d06df82d.xlsx",
}
UA = {"User-Agent": "Mozilla/5.0 (research script; sg-jobs-skills-insights)"}

# data.gov.sg dataset id -> human title (MOM, Manpower Research & Statistics Department)
DATAGOVSG = {
    "d_889d11a2b0a53b235abb64e3f4e0a47b": "Number of Job Vacancy by Industry and Occupation - Annual",
    "d_ddcb2546c798f94ba6b4a6508ab42632": "Job Vacancy Rate by Industry and Occupation - Annual",
    "d_74803e94c5299493f76d71c8994c50c1": "Job Vacancy Rate by Industry and Occupation - Quarterly",
    "d_9917e751f7498502f70052a940a3f312": "Resident Occupational Wages, June 2024",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _get(url: str, **kw):
    import requests

    for attempt in range(6):
        try:
            r = requests.get(url, timeout=120, **kw)
            if r.status_code == 200:
                return r
        except Exception as exc:  # network hiccup
            print("retry", url, exc, file=sys.stderr)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}")


def fetch_datagovsg(dataset_id: str, dest: Path) -> None:
    base = "https://api-open.data.gov.sg/v1/public/api/datasets"
    _get(f"{base}/{dataset_id}/initiate-download")
    for _ in range(30):
        data = (_get(f"{base}/{dataset_id}/poll-download").json() or {}).get("data") or {}
        if data.get("status") == "DOWNLOAD_SUCCESS":
            dest.write_bytes(_get(data["url"]).content)
            return
        time.sleep(3)
    raise RuntimeError(f"data.gov.sg export for {dataset_id} did not finish")


def main(argv=None) -> int:
    """Exit 0: files present and identical to the hashes in data/sources.json.
    Exit 3: a publisher has released a different file (re-run with --update to accept it).
    Exit 1: a file is missing."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify hashes of files already present; no download")
    ap.add_argument("--update", action="store_true", help="write the new hashes to data/sources.json")
    args = ap.parse_args(argv)
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    files = manifest.setdefault("files", {})
    old = {k: v.get("sha256") for k, v in files.items()}
    if not args.check:
        for name, url in SKILLSFUTURE_FILES.items():
            (RAW / name).write_bytes(_get(url).content)
            files.setdefault(name, {}).update(url=url, publisher="SkillsFuture Singapore / SWDA (Jobs-Skills Portal)")
        for name, url in SINGSTAT_FILES.items():
            (RAW / name).write_bytes(_get(url, headers=UA).content)
            files.setdefault(name, {}).update(url=url, publisher="Singapore Department of Statistics (SSOC 2020)")
        for did, title in DATAGOVSG.items():
            fetch_datagovsg(did, RAW / f"{did}.csv")
            files.setdefault(f"{did}.csv", {}).update(
                url=f"https://data.gov.sg/datasets/{did}/view", title=title,
                publisher="Ministry of Manpower via data.gov.sg (Singapore Open Data Licence v1.0)")
            time.sleep(2)
    missing = changed = 0
    for name, meta in files.items():
        p = RAW / name
        if not p.exists():
            print("missing", name)
            missing += 1
            continue
        digest = sha256(p)
        if old.get(name) and old[name] != digest:
            print("SOURCE CHANGED", name, old[name][:12], "->", digest[:12])
            changed += 1
        meta["sha256"] = digest
        meta["bytes"] = p.stat().st_size
    if args.update:
        manifest["accessed"] = time.strftime("%Y-%m-%d")
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(files)} files, {missing} missing, {changed} changed since data/sources.json")
    return 1 if missing else (3 if changed and not args.update else 0)


if __name__ == "__main__":
    raise SystemExit(main())
