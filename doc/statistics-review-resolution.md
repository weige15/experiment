# Statistical review follow-up

Retained reviewer run `06cade59-41e5-4b5c-aa8e-481d2ee28a34` reviewed the fixes before inspecting any formal results. Read-only; no project edits or GPU work.

Outcome: **No additional issues found; OK with notes.**
- Original residual permutation is explicitly exploratory, not an exact conditional-null test.
- New wild-score code follows the predeclared nuisance projection, HC2 adjustment, source-level Rademacher multipliers and add-one one-sided p-value.
- Row shape standardization removes positive request-wide scaling and additive shifts of a fixed global group profile; dataset-specific cubic input-length controls are present.
- Undelimited mixed-content/whitespace token handling was fixed and unit-tested.
- Direct paired accuracy differences are recomputed within source-cluster bootstrap replicates.
- Eight synthetic null scenarios gave empirical rejection rates 0.044–0.060; both strong positive controls rejected 500/500. This supports, but does not prove, calibration.

Residual cautions retained: these simulations do not establish universal validity or calibration at the smaller Holm thresholds; some binomial upper confidence bounds exceed 0.08. Passing the predeclared empirical gate is not proof that the true false-positive rate is bounded by 0.08. Shape association alone does not establish different optimal groups or useful answer-quality gains. The reviewer did not independently rerun the reported tests. Parent-side executed test logs and real full-set measurements remain required evidence.
