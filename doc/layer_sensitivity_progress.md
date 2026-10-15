# Individual-layer study execution journal

## Current status — 2026-10-15 20:24 UTC

**Goal remains active / incomplete.** A real 32-layer campaign is running on GPU **7, NVIDIA RTX 3090, 24 GiB**. Do not confuse this with the completed historical four-layer-group experiment.

At this inspection:
- Fresh BF16 reference baseline and exact single-layer isolation/replay/restoration: **passed**.
- One request × **all 32 individual layers**: **passed**.
- Full CPU suite: **61 passed, exit 0**, including added dataset-stratified conservative-accuracy interval checks.
- Smoke: **24/24 BF16 baselines**, **768/768 W4 rows across all 32 layers**, raw validator and separate smoke command both **exit 0**. BF16 correct: GSM8K 3/8, ARC 6/8, BoolQ 8/8; mean W4 KL **0.0011206422293607516 nats/token**, range **0.00004100322436434933–0.010308878053163**. Pilot statistics only, not formal conclusions.
- Train-only selection scan completed **24 × 32 = 768** W4 rows, strict validation **exit 0**. Frozen BF16 layers: **[0, 1, 8, 9, 10, 11, 13, 31]**; three random sets also frozen. Selection timestamp **20:17:12.884769 UTC**, before formal attempt **20:17:30.229534 UTC**.
- Formal evaluation is now running (Python PID **551037**); full **96 × 32 = 3,072** coverage not yet certified. Joint allocations: not started; real-data final report: not yet produced.

## Active process / exact continuation

Launcher PID **369613**:

```bash
export LAYER_GPU=7
bash scripts/record_command.sh results/layer_sensitivity/steps smoke_command \
  bash scripts/run_layer_sensitivity.sh smoke
# Only if the smoke command returns 0:
bash scripts/record_command.sh results/layer_sensitivity/steps full_command \
  bash scripts/run_layer_sensitivity.sh full
```

Smoke Python PID **373592** exited successfully at 19:35 UTC; train-only scan PID **471797** exited successfully at 20:17 UTC. The launcher is now recording the full command; its shell PID is **466717**, formal evaluation Python PID **551037**. Launch log: `results/layer_sensitivity/steps/smoke_then_full_launch.log`. Outer command log: `results/layer_sensitivity/steps/20261015T184927Z_smoke_command_369619.log`.

Completed smoke ledger:
`results/layer_sensitivity/steps/smoke_20261015T184927Z_369634/`

Completed smoke inference log:
`results/layer_sensitivity/steps/smoke_20261015T184927Z_369634/20261015T185116Z_smoke_373587.log`

Active full ledger:
`results/layer_sensitivity/steps/full_20261015T193554Z_466717/`

Outer full command log:
`results/layer_sensitivity/steps/20261015T193554Z_full_command_369613.log`

The full script, once started, validates/reuses source-identical gates/smoke, scans 24 train-only selection requests × 32 layers, freezes fixed/random choices, scans 96 formal requests × 32 layers, then directly evaluates 5 joint configurations × 96 requests and builds the report. It stops on the first error. Each stage gets `.command`, `.environment`, `.log`, `.exit`; each shell workflow gets `overall.exit`. A missing exit file means unfinished, not success.

**Next concrete action:** inspect the active formal evaluation process and ledger, then continue from the first failed or unfinished gate. Do not launch a second writer or repeat completed GPU work. While it is running, review/test analysis and validation code, which is separate from frozen measurement sources. After completion, execute the separate `report` command again and compare derived numerical CSVs, inspect figures/report, apply the full requirement audit, and evaluate the predeclared escalation conditions.

## New study versus prior work

The repository contained a complete 8-group / 4-layers-per-group study plus an uncommitted descriptive naming refactor. That does **not** satisfy the new active goal. Historical `results/m1/`, `WORK_LOG.md` and old completion documents were preserved. The current package reuses tested data/model/KL/weight-level RTN functions, but **never calls `quantized_group` in its individual-layer path**.

Current code uses `layer_sensitivity/`; raw output is exclusively `results/layer_sensitivity/`. Old onboarding/runbook text was preserved as `doc/archived_group_study_{onboarding,runbook}.md`; README/current docs now describe individual layers and explicitly say incomplete.

## Protocol and review decisions

- Protocol committed before individual-layer inference: `10310af`.
- Read-only protocol/reusable-primitives review: `doc/layer_sensitivity_protocol_review.md`; parent copied the actual reviewer output, not a simulated review.
- Corrected a real parser weakness before new W4 inference: GSM8K now uses the final marker and rejects partial numeric prefixes (`1/2`, `1e3`, invalid commas, units). Regression tests added. Initial BF16 integer answer was unaffected; its pre-fix artifact is retained under `gates/baseline_initial_parser/`, and a fresh parser-reviewed baseline was executed.
- Review resolutions committed before W4 scan: `46fa39c`. Explicit shape/control specification, short-content limitations, random-set conditional uncertainty and outcome-triggered exploratory escalation rules were frozen.
- Additional uncertainty safeguard, committed before formal results while smoke was running: `e1361c0`. Retain bootstrap intervals, add exact marginal and conservative paired difference intervals so zero observed discordance cannot imply zero population risk. Follow-up clarification respects fixed dataset strata by Bonferroni allocation and weighted bounds rather than assuming a common binomial probability across datasets.
- Scope: one fixed wording per source; all 96 formal sources included. Same frozen sample IDs were used historically; report discloses this is not an unseen-to-investigator dataset replication.
- At 20:24 UTC, before inspecting formal layer outcomes, completed the planned secondary content/first-token diagnostic implementation: paired-bootstrap request correlations/Top-k overlap for both, with explicit empty-content eligibility. Primary full-trace KL, frozen selections and escalation rules are unchanged. All secondary results will be retained, not selected by favorability. Seven statistics tests passed (`20261015T202354Z_secondary_content_tests_564995.*`).

## Actual completed commands / evidence

All following step exits are **0**. Paths are under `results/layer_sensitivity/steps/`:

| Command | Ledger prefix / result |
|---|---|
| `.venv/bin/python -m scripts.prepare_data --out results/layer_sensitivity` | `20261015T182953Z_prepare_data_327424.*`; 153 frozen records: 9 demos, 24 selection, 24 smoke, 96 formal |
| `.venv/bin/python -m scripts.provenance --out results/layer_sensitivity/model_provenance.json` | `20261015T182954Z_provenance_327464.*`; four official weight shards + tokenizer/config verified |
| `CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m scripts.baseline --out results/layer_sensitivity/gates/baseline` | `20261015T183548Z_baseline_parser_reviewed_339482.*` |
| `CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m layer_sensitivity.gates --stage single_layer --out results/layer_sensitivity/gates/single_layer` | `20261015T183614Z_single_layer_340686.*`; exact 7 changed matrices, all other parameters unchanged, restored exact |
| `CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m layer_sensitivity.gates --stage single_request --out results/layer_sensitivity/gates/single_request` | `20261015T183932Z_single_request_339481.*`; 32/32 individual layers, all replay/restoration checks passed |
| `.venv/bin/python -m pytest -q tests` | `20261015T184023Z_all_tests_before_smoke_351267.*`; 48 passed at that source state |
| `.venv/bin/python -m pytest -q tests/test_layer_statistics.py tests/test_layer_sensitivity.py` | `20261015T185046Z_statistics_and_allocation_tests_372440.*`; 12 passed |
| `.venv/bin/python -m pytest -q tests` | `20261015T185706Z_updated_full_test_suite_387547.*`; **60 passed in 79.27 s** |
| `.venv/bin/python -m pytest -q tests/test_layer_statistics.py` | `20261015T190255Z_accuracy_uncertainty_tests_399267.*`; **6 passed**, includes added exact/conservative CI boundary check |
| `python -m compileall`, Bash syntax, `git diff --check` | `20261015T190309Z_{source_syntax_399862,shell_syntax_399873,whitespace_399243}.*` |

The smoke workflow also reran the 48-test suite, official provenance, frozen data and strict gate validator; all those exits are 0 in its ledger. Its actual 24 × 32 GPU inference, strict raw validator and overall command all finished with exit 0. Full command reused/revalidated these completed measurements rather than repeating them.

Subsequent actual evidence: `20261015T191801Z_final_preformal_tests_429836.*` — **61 passed in 78.28 s, exit 0**; `20261015T191921Z_hardware_inventory_429813.*` — exit 0, dual Intel Xeon Gold 6226R (64 logical CPUs), 251 GiB host RAM, RTX 3090 GPU 7 (24 GiB).

## Bounded measured gate results (not population conclusions)

Request `gsm8k:test:371`: 789 prompt tokens; BF16 continuation 116 tokens; parsed answer **8**, gold **50**, incorrect. Its 32 individual-layer W4 means range from **0.00042655718547465824** to **0.004683268990700904 nats/token**; 3 of the 32 separately generated W4 answers are correct. All 32 Q/DQ audits show seven changed projections, all exact W4 replays and BF16 restorations pass. Layer 0 mean KL is **0.001288549...**. This is only the required one-request gate, not evidence about all requests.

## Frozen source / safe continuation

Measurement hashes are recorded in every role's `run_config.json`; they still matched current files at 19:06 UTC. Exact measurement files and dependencies are additionally copied to `results/layer_sensitivity/measurement_source_snapshot/` (this is a measurement snapshot, not a standalone full repository).

**Do not edit these while the campaign is running:**
- `quantization_sensitivity/{__init__,data,model,metrics,quantization,runner}.py`
- `layer_sensitivity/{__init__,quantization,runner}.py`
- `layer_sensitivity/gates.py` (gate source is also bound)

Selection/allocation code must likewise remain unchanged once their source-bound stages start. Validators/statistics/report/tests/docs are separate and may be reviewed, with prior derived results preserved. Never repair a mismatch by changing recorded hashes. Git working tree already contained naming-refactor changes before this turn; they were preserved in coherent code checkpoint **b440e75**, alongside the new individual-layer pipeline, after smoke passed. Earlier measurement metadata uses its actual earlier HEAD plus exact source hashes; do not equate HEAD alone with measured source state. The protocol's stratified-interval clarification is commit **ea0c788**.

## Remaining required work

1. Complete and inspect all 96 × 32 formal measurements and 480 directly measured allocations. Smoke and train selection/freeze are now complete; do not rerun their GPU measurements.
2. Fix any real execution failure with preserved logs and minimal changes; do not substitute grouped data or synthetic fixtures.
3. Run strict validators, actual 2,000-replicate analysis and figures/report; check every numerical conclusion against raw tables and all failure cases.
4. Run standalone report reconstruction and compare numerical CSVs; check installation/locked versions and final current-source tests.
5. If W4 structure is unresolved, follow recorded diagnostics → deterministic sample extension/long-request check → optional W3, retaining primary W4 and separating exploratory inference.
6. Finish `doc/layer_sensitivity_completion_audit.md`; only then consider marking the active goal complete. Current evidence is insufficient.
