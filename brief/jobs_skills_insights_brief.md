# Jobs-Skills Insights brief: where emerging skills meet labour demand

*Audience: a sector agency or workforce-development partner. The brief is built only from public
data: the Skills Framework (Q3 2026) and the SSG Unique Skills / Emerging flags, SSOC 2020, and
MOM vacancy and wage tables up to 2025 (quarterly to 2026-Q2). It is an independent portfolio
analysis, not an SSG/SWDA product. Reproduce it with `python -m jsi.pipeline --embed`.*

## Key points

1. **ICT has both the highest emerging-skill intensity and the tightest PME market.**
   In Infocomm Technology: 36.1% of the sector's role-skill requirements are SSG emerging
   skills, the highest of the 39 sectors. The national average is 19.2%. Its MOM industry
   (IT and other information services) had a PME vacancy rate of 5.7% in 2025, the highest
   of the mapped industries, with 3,000 PME vacancies.
2. **The largest PME demand is not where emerging skills concentrate.** Public
   administration and education had 5,900 PME vacancies in 2025 (rate 4.3%). Early
   Childhood maps to this industry, and only 9.4% of its requirements are emerging skills.
   Here the demand is for core professional roles, not new skills.
3. **Construction drives the fastest rise in skill demand, mostly outside PME roles.**
   Construction vacancies rose from 2,200 (2019) to 10,300 (2025), and 9,000 of the 2025
   vacancies were non-PME. As a result, Built Environment skills top the vacancy-weighted
   growth list. They include Integrated Digital Delivery, Building Information Modelling and
   Design for Manufacturing and Assembly. This is a single-industry effect, and the brief
   treats it as one.
4. **A few emerging skills cut across most sectors.** Change Management appears in 29 of the
   39 sectors, Business Opportunities Development in 23, and Data Analytics and Market
   Research in 14 each. These are natural candidates for cross-sector modular programmes and
   for public signposting.
5. **Emerging skills are not, on this evidence, a pay premium.** Roles with a 10 pp higher
   emerging-skill share are in occupations that pay +6.3% (95% CI 2.3 to 10.4) more before
   controls. Within the same SSOC sub-major group the gap is +1.7% (-0.6 to 4.0), which is
   not significant. The raw gap mostly reflects *which* occupation groups use emerging skills.

## What this suggests (for discussion, not recommendation)

| Insight | Possible programme / signposting action |
|---|---|
| ICT: high emerging intensity plus the highest PME vacancy rate | Put ICT emerging skills first in career signposting for mid-career switchers. Check with employers which of the flagged skills are actually screened for at hiring (see validation). |
| Education: volume demand, low emerging share | Signpost entry routes into early-childhood and adult-education roles. Do not frame this demand as an "emerging skills" story. |
| Construction: large non-PME rise | Consider supervisor and technician-level digital delivery modules (IDD, BIM) as well as PME courses. Check first whether the 2019→2025 rise persists past post-pandemic catch-up. |
| Cross-sector emerging skills | Use one shared course catalogue and one signposting page per skill (e.g. Data Analytics) rather than one per sector. |
| No clear within-group pay premium | Avoid "learn X, earn more" messaging. Present emerging skills as relevant to mobility and job security, and test that message with users. |

## How the numbers were produced (and how far to trust them)

* **Linking roles to occupations.** To join the Skills Framework to MOM data, each job role
  needs an SSOC code. A text matcher (TF-IDF plus sentence embeddings, no training) was
  tested on the 1,248 roles that SingStat has coded in its SSOC 2020 alphabetical index.
  It reached top-1 accuracy 45.0% (95% CI 42.1%–47.7%) and top-5 accuracy 69.3%. That is
  useful for suggesting codes to a human coder, not for unattended coding. Coding is
  ambiguous even for a careful reviewer: a 100-role sample labelled by an AI assistant
  (Claude, not reviewed by a human) agreed with SingStat on 40 of 58 overlapping roles.
  The analysis uses SingStat's code wherever one exists.
* **Demand.** MOM publishes vacancies by industry and by broad occupation group, not by
  job role. Industry demand is therefore spread over that industry's Skills Framework roles
  with equal weight. The resulting skill index tracks the industry mix of vacancies. It does
  **not** count postings that mention a skill. Vacancies are rounded to 100, so growth on
  small bases is not reported.
* **Short-term monitoring.** In a backtest of the quarterly PME vacancy rate (21 industries,
  1–4 quarters ahead), simple models did not beat "no change": naive forecast MAE 0.72 pp.
  For dashboards, report the latest value with its usual range, and spend analyst time on
  explanation rather than on forecasting models.
* **What is missing.** There are no job postings. MyCareersFuture's terms do not allow
  storing its content without SWDA's written permission. Postings would add skill mentions
  over time, which is the direct "growth in mentions" signal this brief can only
  approximate.

## Validation plan: what to check with employers and experts

| # | Question | With whom | How |
|---|---|---|---|
| 1 | Of the ICT emerging skills, which ones do hiring managers actually screen for, and which are aspirational? | 8–10 ICT employers (mix of tech firms and in-house IT teams); IMDA | Card-sort of the sector's emerging skills into "screened / nice-to-have / not used" |
| 2 | Is the construction rise structural or a post-pandemic backlog? Which roles in the 9,000 non-PME vacancies need digital delivery skills? | BCA, main contractors, trade associations | Short interviews plus the next two MOM releases |
| 3 | Is the sector → industry crosswalk right, especially the `weak` and `approximate` rows (Energy and Power in "others", Tourism, Aerospace)? | Sector agencies, MOM MRSD | Review the 39-row crosswalk table and give the agencies' own sector definitions |
| 4 | Do SingStat's role codes and the matcher's suggestions agree with how agencies code roles in practice? | SingStat / SWDA coders | Double-code a fresh sample of 100 roles with two human coders and report Cohen's kappa |
| 5 | Why do Arts and Legal Services have no SSG-flagged emerging skills? Is it a tagging gap or a real difference? | SSG skills framework owners; sector experts | Check the flagging process for those frameworks |
| 6 | Would "emerging skill" signposting change behaviour if no pay gain is promised? | Individuals and career coaches | Test two message variants on a signposting page and measure click-through to courses |

## Caveats in one place

* The Emerging flag is SSG's eligibility list for SkillsFuture Series courses. It is not a
  demand measure.
* Correlation is not causation. The pay regression compares occupations, not workers.
* The crosswalk is a judgement call. Cross-cutting sectors (HR, WSH, Carbon Services) have
  no demand figures.
* All figures come from public tables as published. The SHA-256 hashes of every input are
  in `data/sources.json`, so a later data release can be told apart from a code change.
