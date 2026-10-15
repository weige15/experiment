# Final independent M1 review

Reviewer run: `f8a734d8-cb63-4871-8bed-455044608c34`. Read-only review of the complete formal evidence, code, figures, report and execution records. The reviewer did not edit files or execute workloads.

**Verdict: OK — actual evidence supports the limited M1 proposition; requested completion surfaces present; no blockers found.**

Reviewed findings:
- 22/23 full/content replicated pair reversals and 19 replicated bidirectional actual-correctness crossovers are not merely a single global-layer ranking.
- Controlled content-shape association remains positive.
- Label oracle 161/192 versus accuracy-best fixed 148/192, paired gain 6.77 pp [2.60,9.38], matches the artifacts. This is oracle potential, not implementable selection performance.
- Cross-wording KL transfer includes zero; ARC/BoolQ dataset-specific oracle gains are zero; failures and invariant-correctness cases remain included.
- Formal exits/strict raw audits show full coverage. Final logs show 27 passing tests and exact reproduction of all 12 analysis CSVs. The independent raw-table arithmetic audit reproduces core point estimates. Figures distinguish oracles and the unmatched BF16 reference budget.
- One-command resume/re-analysis is demonstrated; a second fresh GPU campaign is not falsely claimed.

## Residual caution

Wild-score inference remains approximate. Actual nuisance fits have very high leverage, near one in output-adjusted ARC, reinforcing finite-sample limitations. This does not negate the observed reversals, actual-answer crossovers or bounded empirical conclusion.

## Parent's post-review diagnostic (not independently executed by reviewer)

Rather than merely changing report wording, executed `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.check_leverage`, exit **0**. This is a post-review exploratory influence check, not a new primary test or a p-value calibration claim.

A one-pass design-only rule flags sources with leverage >=0.95 in either wording. The two flagged sources under output adjustment are `arc:test:963` and `boolq:validation:1941`; the latter alone is flagged for input-only adjustment. In a diagnostic-only copy, omit them and refit the same nuisance model:
- Full/output-adjusted correlation: **0.653906 → 0.652023**, diagnostic n=94.
- Content/output-adjusted correlation: **0.705757 → 0.705178**, diagnostic n=94.
- Across all 16 metric/scope/control combinations, the largest absolute correlation change is about **0.0031**.

The primary 96-source data, all confidence intervals/p-values, all raw answers and all conclusions remain unchanged. The check shows the point association is not driven by those high-leverage sources; it does **not** prove high-leverage finite-sample p-value validity. Full values, source IDs, hashes and caveat: `results/m1/steps/leverage_influence.json` and `leverage-influence.log`.
