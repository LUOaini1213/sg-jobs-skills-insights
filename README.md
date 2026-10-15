# sg-jobs-skills-insights

A small, reproducible Jobs-Skills Insights pipeline for Singapore, built only from public
government data. It links the SkillsFuture **Skills Framework** (job roles and the skills they
need) to the **SSOC 2020** occupation classification and to **MOM labour-market tables**
(job vacancies and wages). From these it produces sector skill profiles, in-demand and
emerging skill indicators, and a short [stakeholder brief](brief/jobs_skills_insights_brief.md).

It is a personal portfolio project. It is not affiliated with SSG/SWDA, MOM or SingStat, and
nothing here uses the National Jobs-Skills Intelligence engine or any non-public data.

## Headline results

| What | Result |
|---|---|
| Skills Framework coverage | 1,977 job roles in 39 sectors; 43,454 role-skill requirements; 195 skills flagged as emerging by SSG, making up 19.2% of all requirements |
| Role → SSOC matcher (main test) | on the 1,248 roles that SingStat has coded: top-1 accuracy 45.0% (95% CI 42.1%–47.7%), top-5 accuracy 69.3%, major group right 70.5% of the time |
| Matcher components | TF-IDF alone 39.9%, embeddings alone 40.4%, majority-class baseline 4.6% (top-1, 5-digit) |
| Matcher (second, smaller test) | on 96 roles labelled by Claude: 53 of 96 (55.2%) top-1 |
| How hard is the coding? | where both exist, the Claude labels agreed with SingStat on 40 of 58 roles |
| Most emerging-skill-intensive sector | Infocomm Technology: 36.1% of its requirements are SSG emerging skills, and its MOM industry had a PME vacancy rate of 5.7% in 2025, the highest of the mapped industries |
| Emerging skills and pay | +10 pp emerging-skill share goes with +6.3% (95% CI 2.3 to 10.4) occupation median wage before controls, +3.4% (0.5 to 6.3) with sector and seniority controls, and +1.7% (-0.6 to 4.0) within SSOC sub-major groups, which is not significant. Sample: 1,072 roles in 260 occupations |
| Forecasting vacancy rates | 1–4 quarters ahead, 21 industries: naive forecast MAE 0.72 pp, beating damped Holt 0.83 pp, mean of the last four quarters 0.88 pp and seasonal naive 0.95 pp |
| Test suite | 26 tests (20 unit tests on hand-computed inputs, 6 on the real data). Mutation check: 15/15 mutants killed, and a behaviour-preserving control mutant survives |

Every number in this README and the brief is rebuilt from `outputs/*.json` by
`tools/check_readme.py`, which runs in CI.

## Data (all public, cited, raw files not committed)

| Source | Publisher | Used for |
|---|---|---|
| Skills Framework Dataset (Q3 2026), Unique Skills List (Sep 2026), TSC-to-Unique-Skills Mapping File (Jun 2026). [Jobs-Skills Portal](https://jobsandskills.skillsfuture.gov.sg/frameworks/skills-frameworks) → "Download the Latest Skills Framework Dataset" | SkillsFuture Singapore / SWDA | job roles, descriptions, TSC/CCS requirements, SSG **Emerging** and **CASL** flags |
| SSOC 2020 Detailed Definitions and Alphabetical Index ([SingStat SSOC page](https://www.singstat.gov.sg/standard-classifications/national-classifications/singapore-standard-occupational-classification-ssoc)) | Department of Statistics | 997 five-digit occupations; SingStat's own Skills Framework role → SSOC codes (used **only** as the test reference) |
| Number of Job Vacancy by Industry and Occupation, Annual (`d_889d11a2…`); Job Vacancy Rate by Industry and Occupation, Annual (`d_ddcb2546…`) and Quarterly (`d_74803e94…`); Resident Occupational Wages, June 2024 (`d_9917e751…`), all on [data.gov.sg](https://data.gov.sg) | Ministry of Manpower (MRSD) | vacancies, vacancy rates, median wages |

`python -m jsi.fetch` downloads everything into `data/raw/`, which is git-ignored. SHA-256
hashes and URLs are in [`data/sources.json`](data/sources.json) (accessed 2026-10-14).
The repo commits only scripts, aggregates, the sector crosswalk and the labelled sample.
data.gov.sg data are used under the Singapore Open Data Licence v1.0. MOM asks users to cite
the source as "MRSD, MOM".

**Why there are no job postings.** MyCareersFuture is the obvious postings source, and its
robots.txt allows crawling. But its Terms of Use say that no part of the site "may be
reproduced, stored in a retrieval system, or transmitted" without written permission from
SWDA. So this project does not collect postings. It uses MOM's occupation and industry
statistics as the demand signal instead. The brief lists what postings would add.

## Method

1. **Skill profiles** (`indicators.py`). Each sector TSC code is mapped to SSG's *unique
   skill* through the official mapping file, which also carries the Emerging and CASL flags.
   99.9% of requirements map. For each sector, the code computes the share of requirements
   that are emerging skills and the share that are CASL skills.
2. **Role → occupation mapping** (`match.py`, `evaluate_mapping.py`). Role text is the title
   plus the Skills Framework description. Occupation text is the SSOC title, definition and
   generic alphabetical-index job titles. The matchers are word TF-IDF, character TF-IDF,
   `BAAI/bge-small-en-v1.5` embeddings, and a hybrid that weights TF-IDF and embeddings
   0.5/0.5. The weights were fixed in advance and never tuned on the labels. Nothing is
   trained on labels.
   * **Test A (independent):** SingStat's alphabetical index lists Skills Framework job roles
     with the SSOC code SingStat gave them. 1,248 current roles join on sector and title.
     Those rows are removed from the occupation text, and a data test checks that no
     `(SFw-` entry leaks in.
   * **Test B (secondary):** 100 roles drawn at random (seed 20261014). See
     [data/labels/README.md](data/labels/README.md). The labels were assigned by Claude, an
     AI assistant, reading each role's description against the 562 MOM wage-table
     occupations, before any matcher existed (commit `05f7828`). **No human coder reviewed
     them.** Four roles had no adequate occupation (`NONE`).
   * For production, each role uses SingStat's code where one exists (1,248) and the hybrid
     prediction otherwise (729). The result is in `outputs/role_to_ssoc.csv`, which records
     the source of each code.
3. **Demand** (`indicators.py`). Each Skills Framework sector is mapped by hand to one MOM
   industry ([crosswalk](data/reference/sfw_sector_to_mom_industry.csv), graded `direct`,
   `approximate`, `weak` or `cross-cutting`; 16 are direct). The code reports 2025
   vacancies, PME share, PME vacancy rate, and change since 2019, the last pre-COVID year.
   A **vacancy-weighted skill index** is Σ over industries of vacancies × the share of that
   industry's roles that require the skill. It moves only with the industry mix of
   vacancies, so it does not count postings that mention a skill.
4. **Pay** (`wages.py`). This is an OLS regression of the log of the occupation's median gross
   wage on the role's emerging-skill share. It uses only roles with an unambiguous official
   SSOC code, and standard errors are clustered by SSOC code. It measures association,
   not a causal premium.
5. **Forecasting** (`forecast.py`). This is a rolling-origin backtest over 12 origins with
   horizons of 1–4 quarters. It covers the quarterly PME vacancy rate for the 21 mapped
   industries that have at least 24 consecutive quarters of data.

## Limits

* The vacancy tables are rounded to the nearest 100, and "-" means nil or negligible. Growth
  rates on small bases, such as 100 → 300, are therefore not reliable. The brief uses only
  large industries.
* The sector → industry crosswalk is a judgement call. Several sectors share an industry,
  and cross-cutting sectors such as HR, WSH and Carbon Services have no demand figures.
* SSG's Emerging flag marks skills that are eligible for SkillsFuture Series funding. It is
  not a measure of market demand. Arts and Legal Services have no flagged skills at all.
* The matcher reaches 45% top-1 at the 5-digit level. That is good enough to suggest
  candidate codes to a human coder, but not to code roles unattended.
* Test B labels come from an AI assistant, and they agree with SingStat on only 40 of 58
  roles. Treat Test B as indicative only.

## Reproduce

```bash
pip install -e ".[test]"            # add ".[embed]" for the embedding matcher
python -m jsi.fetch                 # download public sources, record SHA-256
python -m jsi.pipeline --embed      # all outputs (about 4 min on CPU); without --embed: TF-IDF only
pytest                              # unit tests; data tests run when data/raw exists
python tools/mutation_check.py      # can the tests fail? (see below)
python tools/check_readme.py        # do README/brief numbers match outputs/?
```

**Mutation check.** `tools/mutation_check.py` applies 15 plausible bugs to `src/jsi`, one at
a time, and requires the unit tests to fail on each. Examples: ranking worst-first, dividing
growth by the end value, and a training window that leaks the first test quarter. A
behaviour-preserving control mutant must *survive*, which shows that the harness is not
failing for unrelated reasons. The check has failed for real. On its first run one mutant
survived ([report](outputs/mutation_report_first_run.json)). That mutant was equivalent:
adding the overlapping PMET group to a filter changes nothing, because totals only sum the
PME and non-PME columns. It was replaced by a mutant that does change behaviour. The data
tests also caught a wrong hand-entered SSOC code for nurses in the label converter (`2221`
should have been `22200`). The fix is in.

CI (`.github/workflows/ci.yml`) runs the unit tests, the mutation check and the README check
on every push. A second job, `reproduce`, downloads the public sources and runs the data
tests. It then re-runs the pipeline and requires the committed outputs to come out
byte-identical. SingStat refuses connections from GitHub-hosted runners (seen on
2026-10-15), so in CI this job re-runs only the steps that do not need SSOC: indicators and
the forecast backtest. The SSOC-dependent steps (mapping evaluation and wages) and their
data tests were run locally against the hashed files in `data/sources.json`.

## Layout

```
src/jsi/        fetch, load, ssoc, match, evaluate_mapping, indicators, wages, forecast, pipeline
data/labels/    100-role labelled sample + how it was made
data/reference/ Skills Framework sector -> MOM industry crosswalk
outputs/        aggregates (CSV/JSON) - no raw records, no personal data
brief/          2-3 page Jobs-Skills Insights brief
tools/          mutation check, README number check
```

Licence: MIT for the code. The data remain under their publishers' terms.
