"""Download the public source files into data/raw/ and record their SHA-256.

Sources (all public, no login):
  * SkillsFuture Skills Framework dataset, Unique Skills List and the
    TSC-to-Unique-Skills mapping file, published on the Jobs-Skills Portal
    (jobsandskills.skillsfuture.gov.sg -> "Download the Latest Skills Framework Dataset").
  * Ministry of Manpower labour-market tables on data.gov.sg
    (Singapore Open Data Licence v1.0).

Raw files are NOT committed (see .gitignore); only scripts and aggregates are.
Run:  python -m jsi.fetch          (re-downloads everything)
      python -m jsi.fetch --check  (only verifies hashes of files already present)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify hashes only")
    args = ap.parse_args(argv)
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    files = manifest.setdefault("files", {})
    if not args.check:
        for name, url in SKILLSFUTURE_FILES.items():
            (RAW / name).write_bytes(_get(url).content)
            files[name] = {"url": url, "publisher": "SkillsFuture Singapore / SWDA (Jobs-Skills Portal)"}
        for did, title in DATAGOVSG.items():
            fetch_datagovsg(did, RAW / f"{did}.csv")
            files[f"{did}.csv"] = {
                "url": f"https://data.gov.sg/datasets/{did}/view",
                "title": title,
                "publisher": "Ministry of Manpower via data.gov.sg (Singapore Open Data Licence v1.0)",
            }
            time.sleep(2)
        manifest["accessed"] = time.strftime("%Y-%m-%d")
    bad = 0
    for name, meta in files.items():
        p = RAW / name
        if not p.exists():
            print("missing", name)
            bad += 1
            continue
        digest = sha256(p)
        if args.check and meta.get("sha256") and meta["sha256"] != digest:
            print("HASH CHANGED", name, meta["sha256"][:12], "->", digest[:12])
            bad += 1
        meta["sha256"] = digest
        meta["bytes"] = p.stat().st_size
    if not args.check:
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(files)} files, {bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
