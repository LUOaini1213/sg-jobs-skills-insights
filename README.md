# sg-jobs-skills-insights

**Built:** a Python pipeline on public data only. It links the SkillsFuture Skills Framework
(1,977 job roles in 39 sectors) to SSOC 2020 occupations and MOM vacancy and wage tables, and
writes sector skill indicators and a [stakeholder brief](brief/jobs_skills_insights_brief.md).
**Found:** an untrained text matcher ranks SingStat's own code first for 45.0% of the 1,248
roles SingStat has coded (top-5 69.3%). Occupation pay goes with emerging-skill share
(+6.3% per 10 pp before controls), but only +1.7% within SSOC sub-major groups, which is not
significant (an association, not a premium). The
naive forecast (MAE 0.72 pp) beat three alternatives for vacancy rates 1–4 quarters ahead.
A role–skill graph lists, for each role, the roles that need the fewest additional skills; on
held-out requirements a plain Jaccard baseline does about as well as the best graph model
(90.0% against 90.4% of held-out requirements in the top 10), and both beat a random walk
and SVD.
**Limits:** no job postings; demand is industry-level through a hand-made crosswalk; the second
test set was labelled by an AI assistant with no human review; CI cannot reach SingStat, so the
matcher and pay numbers are recomputed locally only ([details](#reproduce)); the graph
describes what the framework says roles require, not how people actually move between jobs.

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
| Role–skill graph | 1,948 roles and 2,056 skills joined by 43,454 edges. 1,925 of 1,948 roles have at least one reachable role (the role already holds at least half of the target's skills); for 1,134 of them the closest role needs no additional skill, and 58 closest roles are in another sector. 900 of the closest roles are more junior and 463 more senior |
| Skill communities vs sectors | Louvain on the graph finds 19 communities against 39 sectors (adjusted mutual information 0.86; 19–22 communities over 10 seeds). 29 of 39 sectors stay at least 90% inside one community, and 10 communities mix three or more sectors |
| Predicting held-out requirements | 4,345 held-out edges, share ranked in the role's top 10: Jaccard nearest roles 90.0% (baseline), IDF-weighted projection 90.4%, three-step random walk 80.9%, truncated SVD 69.7%, sector popularity 55.1%, global popularity 15.2%. Best model minus best baseline: +0.4 pp (95% CI -0.2 to 1.0), no clear difference |
| Graph vs published career tracks | the closest role is on the same track for 84.4% of roles by fewest additional skills, 90.5% by plain Jaccard and 91.6% by IDF-weighted Jaccard, against 22.6% for a random role in the same sector |
| Experiment design | [how to A/B test a skills recommendation](docs/experiment_design.md): 6,745 users per arm to detect a rise in enrolment from 4% to 5% (assumed rates, not measured ones) |
| Test suite | 69 tests (61 unit tests on hand-computed inputs, 8 on the real data). Mutation check: 39/39 mutants killed, and a behaviour-preserving control mutant survives |

`tools/check_readme.py`, which runs in CI, checks the result numbers in the summary above,
this table, the method notes, the brief's key points and the experiment-design note against
the committed `outputs/*.json` and `outputs/*.csv`, and the test counts against
`pytest --collect-only`. That shows the prose matches the committed outputs. Which outputs CI
also recomputes from the public sources is listed under [Reproduce](#reproduce).

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

6. **Role–skill graph** (`graph.py`). Roles and SSG unique skills are the two sides of a
   bipartite graph; an edge means the role requires the skill. Three things are read from it.
   * *Transitions* ([`outputs/role_transitions.csv`](outputs/role_transitions.csv)). For role
     A, role B is *reachable* if A already holds at least half of B's skills. Reachable roles
     are ranked by the number of additional skills B needs, and those skills are listed. The
     table also gives the number of shared skills that B needs at a higher proficiency level,
     and whether B is more senior, more junior or lateral (difference in mean required level,
     ±0.25). To see progression moves only, filter the CSV on `direction == "up"`.
   * *Role similarity and skill adjacency.* Role–role similarity is Jaccard overlap weighted
     by inverse document frequency, so a skill that a thousand roles need counts for less.
     [`outputs/skill_adjacency.csv`](outputs/skill_adjacency.csv) lists, for each skill that
     at least five roles need, the three skills most often required with it.
   * *Communities* ([`outputs/skill_communities.csv`](outputs/skill_communities.csv)).
     Louvain on the IDF-weighted bipartite graph puts roles and skills into the same
     clusters. In 4 communities no sector has a majority; their sectors with at least 20
     roles each are Tourism + Hotel and Accommodation Services + Retail + Trade Associations
     and Chambers; Energy and Chemicals + Built Environment + Engineering Services + Marine
     and Offshore; Sea Transport + Wholesale Trade; Food Services + BioPharmaceuticals
     Manufacturing + Food Manufacturing. Louvain puts roles from these sectors together
     because they require many of the same skills.
7. **Does the graph predict anything?** (`graph_eval.py`).
   * *Held-out requirements.* 10% of edges are hidden, and each method ranks every skill the
     role does not have in the remaining graph. Each method's parameter, where it has one,
     is chosen on a separate 10% validation split and never on the test edges. The protocol
     is repeated on 5 random splits; the first split carries the intervals, which resample
     roles, not edges. The best model (IDF-weighted projection) scored higher than the best
     baseline (Jaccard nearest roles) in 5 of 5 splits, but the gap is small and its 95%
     interval on the first split includes zero. The random walk and SVD are clearly worse
     than the Jaccard baseline. The simple method is the one to use.
   * *Published career tracks.* The dataset gives every role a sector and a track, which is
     the career ladder SSG drew it on. It has no pathway edges, so the check is only whether
     the closest role is on the same track. Ranking by fewest additional skills agrees with
     the tracks less often than plain Jaccard does (roles with no reachable role count as
     misses). A role with few requirements is cheap to reach, so this ranking leans towards
     smaller, more junior roles.
8. **Experiment design** ([docs/experiment_design.md](docs/experiment_design.md),
   `power.py`). The note covers the unit of randomisation, the primary and guardrail metrics,
   a sample-size calculation and an analysis plan for an A/B test of a skills
   recommendation. Its rates are assumptions.

## Limits

* The vacancy tables are rounded to the nearest 100, and "-" means nil or negligible. Growth
  rates on small bases, such as 100 → 300, are therefore not reliable. The brief uses only
  large industries.
* The sector → industry crosswalk is a judgement call. Several sectors share an industry,
  and cross-cutting sectors such as HR, WSH and Carbon Services have no demand figures.
* SSG's Emerging flag marks skills that are eligible for SkillsFuture Series funding. It is
  not a measure of market demand. Arts and Legal Services have no flagged skills at all.
* The matcher's 5-digit top-1 accuracy (above) is good enough to suggest candidate codes
  to a human coder, but not to code roles unattended.
* Test B labels come from an AI assistant, and they agree with SingStat on only 40 of 58
  roles. Treat Test B as indicative only.
* The graph holds what the Skills Framework says roles require. It contains no observed job
  moves, so a "transition" here is a small skill gap, not a route people are known to take.
  Agreement with the published tracks is partly built in, because roles on one track sit in
  the same sector framework and share much of their skill lists (the point on held-out edges
  below puts a number on this).
* The "half of the target's skills" rule for a reachable role is a choice. Without it, roles
  with three or four requirements appear closest to everything.
* Hiding random edges is an easy test, because roles on one track share most of their
  skills. For 197 held-out edges (4.5%) no other role on the same track requires the skill;
  there the Jaccard baseline puts 44.7% of them in its top 10 and the IDF-weighted projection
  43.6%.
* Louvain depends on the random seed (the range over 10 seeds is in the table), and a
  community is a statistical grouping, not a proposed sector.
* The sample sizes in the experiment note rest on assumed enrolment rates.

## Reproduce

Python 3.11 (what CI uses). `requirements-ci.txt` pins the library versions that produced the
committed outputs; the byte-identical comparison below relies on those pins.

```bash
pip install -r requirements-ci.txt  # pinned versions (Python 3.11)
pip install -e ".[test]"            # add ".[embed]" for the embedding matcher
python -m jsi.fetch                 # download public sources, compare SHA-256 with data/sources.json
python -m jsi.pipeline --embed      # all outputs (about 6 min on CPU); without --embed: TF-IDF only
python -m jsi.graph_eval            # only the role-skill graph outputs (about 90 s)
pytest                              # unit tests; data tests run when data/raw exists
python tools/mutation_check.py      # can the tests fail? (see below)
python tools/check_readme.py        # do README/brief numbers match outputs/?
python tools/reproduce_ci.py        # regenerate what the fetched sources allow; diff against git
```

Parsed tables are cached in `data/interim/` under a key made from the SHA-256 of the raw files
they are built from. After `python -m jsi.fetch --update` accepts a new release, the next run
rebuilds them instead of reading the old cache.

**Mutation check.** `tools/mutation_check.py` applies 39 plausible bugs to `src/jsi`, one at
a time, and requires the unit tests to fail on each. Examples: ranking worst-first, dividing
growth by the end value, a training window that leaks the first test quarter, and an interim
cache that ignores the raw-file hash. A behaviour-preserving control mutant must *survive*,
which shows that the harness is not failing for unrelated reasons. The check has failed for
real. On its first run one mutant survived ([report](outputs/mutation_report_first_run.json)).
That mutant was equivalent: adding the overlapping PMET group to a filter changes nothing,
because totals only sum the PME and non-PME columns. It was replaced by a mutant that does
change behaviour. The data tests also caught a wrong hand-entered SSOC code for nurses in the
label converter (`2221` should have been `22200`). The fix is in. When the 20 graph and
sample-size mutants were first added, two survived
([report](outputs/mutation_report_graph_first_run.json)): the toy graph had no skill held at
the same level by both roles, and the test meant to show that tuning never sees the test
edges used a toy graph on which the best validation score and the best test score were both
100%, so tuning on either gave the same answer. Both tests were strengthened.

**What CI recomputes.** CI (`.github/workflows/ci.yml`) runs the unit tests, the mutation
check and the README check on every push. A second job, `reproduce`, downloads whatever
public sources the runner can reach, runs the data tests, regenerates the outputs those
sources allow (`tools/reproduce_ci.py`) and requires them to be byte-identical to the
committed files. The job log says which outputs were reproduced. SingStat answers
GitHub-hosted runners with HTTP 403 (seen on 2026-10-15), although the same request works
from Singapore, and data.gov.sg sometimes rate-limits them (HTTP 429). So:

| Recomputed in CI from the public sources | Recomputed only on a local machine |
|---|---|
| Skills Framework coverage and sector skill profiles; MOM vacancy indicators and the vacancy-weighted skill index; the forecast backtest; the role–skill graph outputs (transitions, adjacency, communities, link prediction, track agreement) and the sample-size figures | Needs the SingStat files: TF-IDF matcher scores on Tests A and B, Claude-vs-SingStat agreement, the production mapping, the pay regression. Needs `--embed` (sentence-transformers): the embedding and hybrid matcher, including the headline top-1 and top-5 |

The TF-IDF part was last re-run locally on 2026-11-08 from files whose SHA-256 matched
`data/sources.json`, and `tools/reproduce_ci.py` reported its outputs identical to the
committed ones. The embedding and hybrid outputs (`mapping_eval.json`, `role_to_ssoc.csv` and
the mapping block of `metrics.json`) come from the original local `--embed` run. If a
publisher releases a new file (`jsi.fetch` exits 3), the data tests and the comparison are
skipped with a warning, because outputs are then expected to change.

## Layout

```
src/jsi/        fetch, load, ssoc, match, evaluate_mapping, indicators, wages, forecast, pipeline,
                graph, graph_eval, power
data/labels/    100-role labelled sample + how it was made
data/reference/ Skills Framework sector -> MOM industry crosswalk
outputs/        aggregates (CSV/JSON) - no raw records, no personal data
brief/          2-3 page Jobs-Skills Insights brief
docs/           experiment design note (A/B test of a skills recommendation)
tools/          mutation check, README number check, CI reproduction
```

Licence: MIT for the code. The data remain under their publishers' terms.
