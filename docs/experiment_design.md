# Experiment design: does a skills recommendation change what people do?

This note sketches how an agency such as SWDA could measure the effect of a skills
recommendation, for example "*Data Analyst* needs two skills you do not have yet" shown on a
careers or course-search page. The recommendation itself could come from
the role-skill graph in this repository (`outputs/role_transitions.csv`).

It is a design exercise in a personal portfolio project. I have no access to any SWDA
product, traffic or conversion data. **Every rate below is an assumption chosen to show the
arithmetic**, and the first step of a real study would be to replace them with measured
values. The numbers are produced by `src/jsi/power.py`, written to
`outputs/experiment_power.json`, and checked against this text by `tools/check_readme.py`.

## 1. Question and hypothesis

*Decision the test informs:* whether to show graph-based skill recommendations to all users.

*Hypothesis:* showing a user the skills that separate their current role from nearby roles
raises the share who enrol in a course that teaches one of those skills.

## 2. Unit of randomisation

**The individual user account, assigned once and kept for the whole test.**

| Option | For | Against |
|---|---|---|
| Page view or session | most traffic, fastest | the same person sees both versions; the outcome (enrolment weeks later) cannot be tied to one view |
| **User account** | outcome and exposure belong to the same unit; assignment is stable across devices when logged in | needs login; anonymous traffic is excluded or handled separately |
| Employer or training provider (cluster) | right choice if the recommendation is delivered through an employer, or if colleagues share what they see | far fewer units, so a larger sample (section 5) |

Interference is the main risk to user-level randomisation. If a treated user forwards the
recommendation to a colleague in control, the measured effect shrinks. Where the product is
used through employers, randomise employers instead and analyse with cluster-robust errors.

Assignment is a hash of the user ID and an experiment salt, 50/50. Stratify, or at least
check balance, by sector and by whether the user has enrolled in any course in the past year.

## 3. Metrics

| Role | Metric | Why |
|---|---|---|
| **Primary** | share of users who enrol in at least one course mapped to a recommended skill within 90 days of first exposure | closest observable step to "acquired the skill"; binary, so the power calculation is simple |
| Secondary | click-through on a recommended skill; course completion within 180 days; number of distinct recommended skills pursued | mechanism and persistence |
| Guardrail | overall course enrolment (all skills); bounce rate of the page; complaints or "not relevant" feedback | a recommendation that only moves enrolment from one course to another, or annoys users, should not ship |
| Long-run (not testable in this window) | job change into a recommended role; wage change | needs linked administrative data and a much longer horizon |

One primary metric is fixed before launch. Secondary metrics are reported with their
intervals but do not decide the launch.

## 4. Sample size

Two-sided two-proportion z-test, equal arms, alpha 0.05, power 0.80. Assumed control rate
4%; smallest lift worth acting on 1 percentage point (4% → 5%, a 25% relative lift).

n per arm = (z₀.₉₇₅ · √(2·p̄·(1−p̄)) + z₀.₈₀ · √(p₁(1−p₁) + p₂(1−p₂)))² / (p₂ − p₁)²

With p₁ = 0.04 and p₂ = 0.05 this gives **6,745 users per arm (13,490 in total)**. At an
assumed 5,000 eligible users per week that is 3 weeks of enrolment into the test, followed by
the 90-day outcome window.

The required sample is very sensitive to the two assumptions:

| Control rate | Relative lift to detect | Users per arm |
|---|---|---|
| 2% | 10% | 80,682 |
| 2% | 25% | 13,809 |
| 2% | 50% | 3,826 |
| 4% | 10% | 39,475 |
| 4% | 25% | 6,745 |
| 4% | 50% | 1,863 |
| 8% | 10% | 18,872 |
| 8% | 25% | 3,213 |
| 8% | 50% | 882 |

If only half the planned sample arrives, the smallest detectable lift grows from 1.00 pp to
1.45 pp.

The formula is tested three ways in `tests/unit/test_graph.py`: against a hand calculation
(50% → 60% needs 388 per arm), against its own inverse (power at the returned n is 0.80, and
the detectable lift at that n is the planned lift), and against a simulation of 4,000
experiments.

## 5. If the unit has to be the employer

Outcomes of colleagues are correlated, so the effective sample is smaller than the head
count. With an assumed 20 users per employer and an intra-cluster correlation of 0.02 the
design effect is 1.38, and the test needs **9,309 users per arm in 466 employers per arm**.
The correlation is an assumption; a pilot or historical enrolment data would give a real one.

## 6. Analysis plan (fixed before launch)

1. **Sample-ratio check first.** A chi-square test that the arms are 50/50
   (`power.srm_p_value`). If p < 0.001 the assignment or logging is broken, and no outcome
   is read.
2. **Primary analysis.** Difference in enrolment share with a 95% confidence interval, by
   intention to treat: users are analysed in the arm they were assigned to, whether or not
   they saw or clicked the recommendation.
3. **Variance reduction.** Adjust for pre-period enrolment (CUPED or a regression with the
   pre-period covariate). This can shorten the test, but the plan above does not rely on it.
4. **No peeking without a rule.** Either read the result once at the planned sample, or use
   a group-sequential boundary decided in advance.
5. **Heterogeneity is exploratory.** Effects by sector or seniority are reported with a
   multiple-comparison correction and treated as hypotheses for the next test.
6. **Novelty.** Report the effect by week since first exposure. An effect that fades within
   the 90 days argues against a launch.

## 7. What this test cannot show

* It measures enrolment, not learning, job moves or wages. Those need linked data over years.
* It tests the whole recommendation feature. Which model produces the recommendations is a
  separate comparison (for instance graph-based against popularity-based lists, as a third
  arm). The offline result in this repository, where a simple Jaccard baseline matches the
  best graph model on held-out requirements, suggests that such an arm is worth including.
* Logged-in users are not all users. The result generalises to people who would log in.
