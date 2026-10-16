# Final review resolution — 2026-10-16 UTC

Independent read-only reviews are preserved in [scientific review](layer_sensitivity_final_scientific_review.md) and [execution/evidence review](layer_sensitivity_final_evidence_review.md). They initially blocked sign-off on reporting/protocol completion, not on the completed GPU measurements. The following resolutions were implemented and verified by the parent; the reviewers did not rerun the post-fix code.

| Finding | Resolution and actual verification |
|---|---|
| R² percentile ranges could be mistaken for heldout-performance confidence bounds | Preserved all original estimates/ranges; explicitly labeled nominal in-sample refit ranges with unassessed bias/coverage in report, metadata and `statistics.csv:interval_interpretation`. Explained optimism and why positive partial-R² bounds do not prove independent effects. |
| Protocol promised cubic log-output, but first diagnostic used linear log-output | Added dataset-specific powers 1–3 for both log prompt and output lengths. Preserved the earlier diagnostic unchanged. Added tested `cubic_prompt_output_design` and residual/coefficient CSVs. Actual new design rank **16/21**, maximum leverage **1**; interpolated request **arc:test:963**. |
| Nonlinear fit fragility not prominent | Report states old design rank **14/15**, new rank **16/21**, both max leverage **1**, versus primary linear **6/6**, max **0.20514**. Rank-deficient coefficients are nonidentifiable; lower residual RMS is not generalization evidence. No rows were removed. |
| Imported content/nonlinear helper missing from derived source hashes | Added `quantization_sensitivity/analysis.py` and validation dependencies, plus analysis package versions and lock hash. Measurement-source hashes stayed unchanged. |
| Stability headline could imply all datasets | Headline now limits content-half correlation and repeated-direction reversals to **32 eligible GSM8K requests**. ARC/BoolQ have content profiles but no eligible content halves. First-token/content diagnostics are not independent replications. |
| Accuracy oracle terminology/uncertainty | Retained negative/inconclusive KL-oracle comparison, **5 gains / 9 losses**, and both bootstrap/conservative intervals. Printed finite label-oracle accuracy **78/96 = 81.25%**, explicitly limited to five measured configurations and not deployable. Figure distinguishes the all-32-BF16 reference from eight-layer-budget strategies. |
| Stale completion documentation | Historical audit retained as `layer_sensitivity_completion_audit_before_final.md`; final requirement-by-requirement audit and onboarding/runbook updated from inspected actual exits. |

## Verification

- Code checkpoint: `660248b` (review corrections; no change to frozen inference sources).
- `.venv/bin/python -m pytest -q tests`: **63 passed in 75.89 s**, exit **0**; `results/layer_sensitivity/steps/20261015T235314Z_final_full_tests_942734.*`.
- Compileall, Bash syntax and `git diff --check`: exit **0** (`20261015T235432Z_*` ledgers).
- CPU-only standalone report rebuild: exit **0**, `20261015T235515Z_report_rebuild_947027.*`.
- All **648** existing statistical rows remain numerically **exactly unchanged**. Earlier nonlinear residuals, primary residuals, answer table, reversals and request-pair tables are byte-identical: `results/layer_sensitivity/steps/review_fix_numeric_comparison.json`.
- Final-code full workflow resumed/revalidated completed real measurements, reran tests and rebuilt report: exit **0**, `20261015T235931Z_full_verified_resume_954889.*`. No GPU measurements were repeated.
- Separate second CPU report rebuild: exit **0**, `20261016T000504Z_report_rebuild_reproducibility_962219.*`. **20 CSVs, report text and four PNG figures reproduce exactly**; metadata differs only in creation time. Evidence: `results/layer_sensitivity/steps/report_reconstruction_check.json`. PDF creation timestamps are not required to be byte-identical.

## Final artifact checks

`results/layer_sensitivity/steps/final_artifact_check.json` independently verifies all 480 joint answer scores, the exact eight-layer fixed/random/oracle choices, shared BF16 continuations, full restoration and 54 documentation links. Initial ad-hoc audit code assumed a uniform JSON schema and a `generated_text` field; it was corrected to use the actual per-strategy records and `answer` schema. These were audit-script errors, not inference failures or changes to measured results.

The staged whitespace check reports trailing spaces in verbatim `.command` files and the NVIDIA-SMI log. Those raw execution bytes are preserved rather than cleaned up; the source/documentation whitespace check excludes `results/layer_sensitivity/steps/**`. Model tensors/reference caches and common credential patterns were excluded/checked before staging the shareable evidence. No remote push was performed.

## Scientific stopping decision

No W3 or sample escalation was triggered: maximum KL **0.0223338573** exceeds the **1e-6** floor; **462** material reversed pairs exceed two; eligible content-stability lower CI **0.81341 > 0**; heterogeneity lower CI **0.39682 > 0.01**. The independent scientific review agreed that no further GPU cohort is essential for these bounded conclusions. The negative/inconclusive accuracy result is retained, not optimized away. Additional independent requests are a future precision/generalization question, not permission to claim a deployable gain now.
