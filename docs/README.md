# AnesTRACE project website

This directory is a dependency-free static site for GitHub Pages. It can be
opened locally via `index.html`; no build step or API is required.

`index.html` is the English page and `zh.html` is the Chinese page. The navigation
language switch links the two pages and preserves the current section. Both use
the same `leaderboard-data.js`, with localized table labels in `app.js`.
Keep both pages in sync when changing shared content or section order.

The leaderboard in `leaderboard-data.js` transcribes the **published aggregate**
results from Tables 3 and 4 of [arXiv:2609.32740v1](https://arxiv.org/pdf/2609.32740v1)
(submitted September 26, 2026). Scores are a paper snapshot, not a live
submission service. L1, L2, and L3 use different metrics and evaluation units;
do not combine them into a cross-level rank. When updating the site, check
every changed value against the public paper and update the version/date note.

The overview and representative multi-step case-study images come from the paper figures. Institution marks are from
the [Zhejiang University standard emblem page](https://www.zju.edu.cn/514/listm.htm)
and the [Second Affiliated Hospital official website](https://en.z2hospital.com/channels/852.html).
Replace them with institution-approved files if required by local brand rules.
The AnesTRACE project logo is stored as `assets/anestrace-primary-logo.png`.
The arXiv, GitHub, and Hugging Face button icons are from
[Simple Icons](https://simpleicons.org/) (CC0).

No patient-level records or restricted source-derived benchmark files belong
in this directory. The [data availability note](../data/README.md) explains
the current nonrelease policy and team-run model-evaluation route.

The model-submission links open the GitHub Issue form defined in
`../.github/ISSUE_TEMPLATE/model-submission.yml`. Publish that file on the
repository's default branch and enable Issues to activate the form. Submissions
are public and require a GitHub account; use shareable model links only.

To publish, configure repository **Settings > Pages > Deploy from a branch**
with branch `main` and folder `/docs`. The resulting project site is
`https://zjudbxai.github.io/AnesTRACE/`. The GitHub repository URL itself
continues to show the repository; it cannot be replaced by the Pages home.
