# M1 requirement-to-evidence audit

**Decision: M1 evidence and deliverables ACHIEVED within the stated finite protocol.** Independent final review found no blockers; the final one-command execution and evidence audit passed. This decision uses actual real-weight measurements and bounded scientific conclusions, not a test/manifest/plot green light alone. Goal-manager state is a separate tool limitation noted below.

Scope: M1 only, official Llama 3.1 8B, eight single-group W4 interventions, three benchmarks. No M2–M4, router, clustering or batching.

## Requirement audit

| Requirement | Concrete inspected evidence | Status |
|---|---|---|
| Real official Llama 3.1 8B | `results/m1/model_provenance.json`: four safetensors shards + six tokenizer/config files match pinned official digests; reverified by one-click `official_weights.exit=0` | Verified |
| BF16 reference, same tokenizer/template/activation/KV | `m1_model.py`, `chat_template.jinja`, raw token IDs and per-run dtype checks; base model choice disclosed | Verified |
| W4 weight-only first, not simulated output | `m1_quant.py`: real packed RTN codes, BF16 Q/DQ inference; no packed-kernel performance claim | Verified |
| Eight contiguous four-layer groups, one at a time | `steps/eight_groups/`, all raw `quantization/` and restore checks; 28 matrices/872,415,232 weights per group, fixed block-128 method | Verified |
| Only intended parameters changed / exact restoration | Complete before/during/after hashes in `steps/one_group/result.json`; all unselected weights unchanged, exact parameter/logit restoration | Verified |
| Three datasets, fixed seed and isolation | `requests.jsonl`, `data_manifest.json`: seed 20261015, pinned revisions, 9 demos + 24 calibration + 24 smoke + 96 evaluation, question-level duplicate checks | Verified |
| Smoke before formal, ≥32 evaluation/dataset | Smoke A/B complete and validated before formal; formal has exactly 32 GSM8K + 32 ARC-Challenge + 32 BoolQ sources | Verified |
| Calibration-fixed choices before evaluation | `calibration_selection.json` g6 KL/g1 accuracy at 15:33:22 UTC; formal starts after 15:34 UTC; audit compares timestamps and input hashes | Verified |
| Per-token KL, causal alignment, same history | `m1_metrics.py`, analytic/shift tests, full-vocabulary real traces; 62,240 formal token-KL measurements | Verified |
| Per-request eight-group rankings | Full `per_request_measurements.csv` and `request_summary.csv`; rectangular 96×2×8 coverage, no missing records | Verified |
| Actual answers and automatic correctness | `analysis/answers.csv`: 1,728 actual decoded responses/IDs/gold/scores; raw record validator rechecks decoding/scoring | Verified |
| High-sensitivity groups and ranking reversals | Report contains all top/bottom counts; all 28 pairs in `ranking_reversals.csv`, including non-reversing/near-tie cases | Verified |
| Wording robustness | Two predefined instruction wordings for every source; mean full rank Spearman 0.8326 [0.8095,0.8549]; 22/28 full and 23/28 content pair reversals replicated | Verified with argmin-transfer limitation |
| Segment robustness | Full/content halves and first-token diagnostics; content GSM8K half-rank correlation 0.7920 [0.7545,0.8311]; ARC/BoolQ single-token content explicitly non-estimable | Verified, not overgeneralized |
| Control dataset/length and request-wide magnitude | `controlled_shape_wild.csv`: positive affine magnitude removed, dataset-specific cubic log input and optional log output controls; all scopes reported | Verified under specified nuisance models |
| Statistically defensible inference | 2,000 paired source bootstrap draws, fixed group reselected each draw; approximate HC2 wild score, Holm family 16; empirical synthetic null calibration and independent review; original permutation exploratory only | Verified with finite-sample/misspecification caveats |
| Equal-budget request-specific vs fixed quality | Metric-matched calibration/evaluation fixed baselines; BF16-reference KL oracle, label oracle and cross-wording diagnostics; actual chosen answers in `oracle_comparisons.csv` | Verified; all assisted selection labeled oracle |
| Full evaluation set, not favorable examples | 96 independent sources, 192 worded prompts, 1,536 W4 rows; complete raw records and all plot rows; no outcome-based exclusion | Verified |
| Bootstrap confidence intervals | `statistics.csv`, `controlled_shape_wild.csv`, `segment_stability.csv`; dataset-stratified source unit, two wordings/groups kept paired | Verified; conditional on fixed demos/calibration |
| Failures and no-benefit cases | 437 incorrect/1,728 answers, 26 parse failures/truncations retained; `answer_failures.csv`, `no_benefit_or_loss_cases.csv`; 149/192 no correctness variation, 104/192 no material KL gain vs calibration-fixed | Verified |
| Reproducible installation | 60 pinned dependencies, `scripts/install.sh`; initial clean install retry and repeated locked sync exit 0 | Verified on recorded environment |
| One-command M1 | `scripts/run_m1.sh`; `M1_GPU=1 bash scripts/run_m1.sh` actually executed, all logged steps/overall exit 0 | Verified resume/re-analysis; no redundant fresh second campaign claim |
| Necessary unit and actual smoke tests | Final 27 unit tests pass; real smoke 384 W4 rows plus 48 BF16 responses, strict audit exit 0 | Verified |
| Raw data, heatmap, accuracy comparisons, report | `runs/*/{baselines,measurements}.csv`, all raw JSON, 12 analysis CSVs, four PNG/PDF figure pairs, `results/m1/report.md` | Verified; full heatmap/accuracy images visually inspected |
| Hardware, revision, quantizer, seed, arguments | `attempt_*.json`, `run_config.json`, model/data provenance, analysis hashes, per-step command/log/exit ledger | Verified |
| Actual execution and independent checks | `scripts/audit_m1.py` independently recomputes raw-table point estimates; all 12 analysis CSVs exactly reproduced by one-command rerun | Verified; not substituted for scientific reasoning |
| Incremental development / original preservation | Baseline → one group → KL → eight groups → smoke → calibration → formal; original failed/interrupted/pilot analyses and logs retained, archived pre-reanalysis outputs | Verified |
| No premature scientific success / M1-only scope | Two statistical review stages addressed concrete inferential/token-mask issues before formal outcome inspection; final interpretation below limits claims | Verified; independent final review OK (`doc/final-review.md`) |

## Scientific assessment, not just software success

The finite-protocol M1 proposition is supported by a conjunction of independent evidence surfaces:

1. **Relative profile variation, not only request magnitude:** after positive-affine shape normalization and specified dataset/input/output-length controls, content-profile correlation across wording is **0.7058 [95% CI 0.6217,0.7792]**. Full profile is **0.6539 [0.5585,0.7304]**. Approximate Holm-adjusted wild p is **0.007996**, not an exact conditional-null guarantee.
2. **Changed ordering:** full/content pair reversals are replicated in both wordings, and within every dataset. The 22/23 pair counts are descriptive, not 22/23 independent observations.
3. **Real quality differences, beyond tiny KL:** **19/28** group pairs have bidirectional correctness crossovers replicated across wording. Label oracle gets **161/192** versus accuracy-best fixed **148/192**, paired gain **6.77 pp [2.60,9.38]**. Twelve of 13 additional worded correct cases compare two untruncated outputs; all truncations still remain in primary analysis.
4. **Meaningful limitations preserved:** KL oracle improves full KL **16.45% [12.69,19.98]**, but cross-wording choice transfer only **0.12% [-5.04,4.95]**. ARC/BoolQ label-oracle gain over their own best fixed is zero. Most requests have no correctness variation; optimal group agreement is only about 54%. These results do **not** justify a deployable or transferable precision policy.

Thus the supported statement is **request-dependent group sensitivity and limited oracle potential under this fixed W4 protocol**, not universal semantic routing or practical precision-selection gains. The existing W4 evidence is substantive enough for this bounded statement; escalating to W3/finer groups merely to obtain a more favorable transfer result is not warranted. No such follow-up was performed.

## Remaining limitations / failures explicitly retained

- Initial dependency download exceeded the tool timeout; no fabricated exit code. Retry and locked install passed.
- Smoke-B was deliberately interrupted (our process only, exit 143) after shared-GPU contention; resumed successfully. Other users' jobs were not terminated.
- The first BF16 gate answered 8 instead of gold 50. Measurement success is not task correctness.
- Same-path repeat floor was measured as zero at the checked gates; cached/full-route KL was nonzero (0.000370752). No claim that this proves every execution path/hardware is noiseless.
- One fixed seed, 32 independent sources/dataset, fixed demos and small calibration set; two instruction wordings, not arbitrary question paraphrases.
- Length controls are specified regressions, not proof against every nonlinear length confound. Wild-score calibration is approximate and was tested only on the declared synthetic scenarios/nominal threshold. Some actual nuisance fits have leverage near 1. A post-review influence diagnostic (exit 0) omitting the two >=0.95-leverage sources only in a diagnostic copy changes content/output-adjusted correlation 0.705757→0.705178; all primary 96 sources/CIs remain unchanged. This demonstrates point-estimate stability, not exact p-value calibration.
- One-command **resume/re-analysis** was executed, with exact reproduction of 12 CSVs. The fresh-output command was not redundantly run through a second complete GPU campaign, and an empty-machine/untested-hardware rebuild is not claimed.

## Execution ledger

`results/m1/steps/one_click_20261015T165747Z_135143/`: install, unit tests, official model re-verification, data regeneration consistency, all three raw validators, frozen-choice check, method-validation simulation, formal analysis, figures, report and completion-evidence audit all **exit 0**. Outer `results/m1/steps/one-click.exit=0`. Raw GPU inference exits: `steps/{smoke,calibration,evaluation}_{a,b}.exit=0`.

The final updated runner was actually re-executed: `results/m1/steps/one_click_20261015T173520Z_209164/`, including the leverage diagnostic. Every logged step and overall exit is **0**; 27 unit tests pass and all 12 analysis CSVs again match exactly. `steps/one-click-final.exit=0`. Final independent scientific/delivery verdict: **OK, no blockers**, retained in `doc/final-review.md`.

## Goal-manager state

The tools exposed in this continuation include `create_goal` but not `update_goal`. Therefore no existing goal was replaced or programmatically marked complete. The evidence above supports closing the summarized active goal `1792074122621-7926f107485ac8` when that status-control tool is available; this is an administrative capability limitation, not an outstanding experiment or deliverable.
