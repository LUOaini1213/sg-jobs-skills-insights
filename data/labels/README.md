# Labelled sample for the job-role -> occupation mapping

`gold_role_to_occupation.csv` has 100 Skills Framework job roles, drawn with
`roles().sample(100, random_state=20261014)` from the 1,977 distinct (sector, job role) pairs.

For each role, `acceptable_occupations` lists the occupation titles from MOM's
*Resident Occupational Wages, June 2024* table (562 detailed occupations) that a
reviewer would accept as a correct mapping. The first title is the preferred one, and
`;` separates alternatives. `NONE` means no title in the 562-occupation list fits
adequately (4 of 100).

**Who labelled it and how.** Claude (an AI coding assistant) assigned the labels on
2026-10-14 while working for the repository owner. It read each role's title and
Skills Framework description against the full list of 562 occupation titles. **No
human coder reviewed the labels.** They were written and committed before any matcher
was run, so the matcher's output did not influence them. Treat them as a provisional
benchmark. A proper benchmark needs two independent human coders and an agreement
statistic (see the validation section of the brief).
