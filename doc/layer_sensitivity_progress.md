# Individual-layer study — completed execution journal

**Verified complete for the requested bounded study, 2026-10-16 UTC.** See [final requirement audit](layer_sensitivity_completion_audit.md), [report](../results/layer_sensitivity/report.md) and [review resolution](layer_sensitivity_review_resolution.md). No model job remains to be completed; do not rerun the already completed scans to fill an old progress note.

## Executed stages

| Stage | Actual evidence/result |
|---|---|
| Environment/data/official weights | Pinned real checkpoint and dataset files verified; 153 frozen sources including 9 train demonstrations |
| BF16 and parser review | Strict final-marker numeric parser frozen before individual-layer inference; initial parser baseline preserved under `gates/baseline_initial_parser/`; fresh BF16 gate rerun |
| Single-layer gate | Exactly seven matrices changed; other parameters isolated; exact restoration and deterministic replay, exit 0 |
| One request × 32 layers | Real first smoke GSM8K source scanned through all 32; exact replay/restoration, exit 0 |
| Three-dataset smoke | 24 BF16 requests and **768 W4 rows**, validation and separate smoke command exit 0 at 19:35 UTC |
| Train-only selection | 24 requests × 32 = **768 rows**, validated exit 0 at 20:17 UTC |
| Frozen allocation | Fixed BF16 layers **[0,1,8,9,10,11,13,31]**, three random sets. Frozen **20:17:12.884769 UTC**, before formal attempt **20:17:30.229534 UTC** |
| Formal scan | **96 × 32 = 3,072 rows**, completed **23:01:20 UTC**, 9,842.32 s attempt time; raw validator exit 0 |
| Joint allocations | **96 × 5 = 480 real answers**, eight BF16/24 W4 each; completed **23:35:12 UTC**, 2,016.26 s; raw validator exit 0 |
| First full report | Original full workflow exit 0 at **23:36 UTC**; first real statistical report retained in later reconstruction archive |
| Independent review/corrections | No inference defect found. Corrected R² interpretation, completed promised cubic-output diagnostic, added analysis dependency hashes, explicit eligibility/leverage warnings; no measured value changed |
| Final tests | **63 passed in 75.89 s**, exit 0; compileall, Bash syntax, diff check exit 0 |
| Final-code full resume | Exit 0; revalidated all completed evidence and reran tests/report without repeating GPU measurements |
| Standalone CPU report rebuilds | Two exit-0 commands; **20 CSVs, report text and four PNGs reproduce exactly** |

Dates before midnight are **2026-10-15 UTC**; final reconstruction is **2026-10-16 UTC**. GPU: one RTX 3090 24 GiB, physical device 7; host: dual Xeon Gold 6226R, 251 GiB RAM. This is Q/DQ quality simulation, not W4-kernel benchmarking.

## Exact command ledgers

Under `results/layer_sensitivity/steps/`:

- Real separate smoke: `20261015T184927Z_smoke_command_369619.*` (exit 0).
- Real full workflow: `20261015T193554Z_full_command_369613.*` (exit 0).
- Full child stages: `full_20261015T193554Z_466717/`; formal inference `20261015T201713Z_evaluation_551033.*`, joint inference `20261015T230131Z_precision_allocation_877775.*` (both exit 0).
- Final tests: `20261015T235314Z_final_full_tests_942734.*` (63 passed, exit 0).
- First standalone CPU report: `20261015T235515Z_report_rebuild_947027.*` (exit 0).
- Final-code full validated resume: `20261015T235931Z_full_verified_resume_954889.*` (exit 0).
- Standalone reconstruction comparison: `20261016T000504Z_report_rebuild_reproducibility_962219.*` (exit 0).
- Independent raw/cache check: `independent_formal_raw_check.json`: 96 complete vectors, 125,824 token KL values, 1,008,605,184 reference bytes checked.
- Independent valid-answer ground-truth check: `independent_scoring_check.json`: 3,168 baseline/single-layer answers checked, including Decimal numeric equality.
- `review_fix_numeric_comparison.json`: all 648 existing statistical rows unchanged exactly by review fixes.
- `report_reconstruction_check.json`: exact derived-table/text/PNG reconstruction, matching source/input hashes, resolved report links.

## Results and scope

Mean single-layer KL **0.0010087122 nats/token**. Different-request rank Spearman **0.57370 [0.54206, 0.60318]**; Top-8 overlap **0.52223 [0.50291, 0.54063]**. Eligible within-request content-half Spearman **0.83829 [0.81341, 0.86199]**, **32 GSM8K requests only**. These indicate shared global preferences plus measurable request variation—not complete independence of request profiles.

BF16 accuracy **75/96**; fixed-budget fixed configuration **68/96**; request-KL oracle **64/96**; random-set mean **68.403%**. Oracle-minus-fixed **−4.1667 pp [−11.458, +3.125]**, 5 gains/9 losses. This does **not** establish an accuracy gain. Finite label oracle **78/96** is hindsight over five measured candidates, not a deployable selector.

No escalation was triggered under the frozen W4 structure rules. No favorable subset, new primary metric, W3 substitution, router, clustering or batching was introduced. Interpretation limits and future independent-cohort question remain in the report.

## Provenance / preserved history

- Protocol commits: `10310af`, parser/control clarification `46fa39c`, interval safeguard `e1361c0`, stratified-bound clarification `ea0c788`.
- Coherent individual-layer implementation after real smoke: `b440e75`; retained the pre-existing descriptive-name refactor without discarding it.
- Secondary content diagnostic completion before inspecting formal layer outcomes: `f3f8ef5`.
- Post-run review correction checkpoint: `660248b`. Primary inference sources and raw data stayed unchanged.
- Real measurement source snapshot: `results/layer_sensitivity/measurement_source_snapshot/`. Code/input hashes, not HEAD alone, identify earlier measurements made before code commits.
- Each report command archives previous analysis, report and figures in its ledger. Historical grouped `results/m1/` remains separate and never counts as these individual-layer observations.
- Earlier incomplete audit is preserved in `layer_sensitivity_completion_audit_before_final.md`; Git history retains intermediate progress notes.

## Reuse

For inspection, read the report and final audit. For CPU reconstruction, run `bash scripts/run_layer_sensitivity.sh report`. For genuinely fresh inference, use a **new** `LAYER_OUT` with an allocated GPU. Never edit frozen raw results, restamp source hashes, or interpret old pending PIDs as active work.
