# Individual-layer study — final completion audit

**Status: VERIFIED COMPLETE for the bounded requested study.** Audit date: **2026-10-16 UTC**. This is not a claim of accuracy benefit, universal stability, an optimal allocation, or a deployed router. The [historical in-progress audit](layer_sensitivity_completion_audit_before_final.md) is preserved.

## Objective checked

Actually execute official Llama 3.1 8B BF16 versus **32 individual-layer W4 interventions** for every evaluation request; retain complete vectors and actual answers; directly measure equal-budget joint allocations; analyze all requests with uncertainty and stated limitations; supply reproducible code, tests, smoke/full/report commands, raw tables, figures and a report. Historical four-layer-group measurements do not count.

## Requirement-by-requirement evidence

All artifact paths below are relative to `results/layer_sensitivity/` unless explicitly prefixed with `doc/`, `scripts/`, `tests/` or a package name.

| Requirement | Inspected actual evidence | Verdict |
|---|---|---|
| Exact official model/tokenizer revision | `model_provenance.json`, four official shard checks; attempt metadata pins `meta-llama/Llama-3.1-8B`, revision `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` | PASS |
| Same tokenizer/template/deterministic decoding | `chat_template.jinja`, frozen full prompts/token IDs in baseline JSON/CSV, greedy settings and replay evidence; base model has no official chat template | PASS |
| BF16 reference, activation and KV precision | Real baseline and gate precision probes, every scan's precision/replay/restoration evidence | PASS |
| Exactly one of 32 layers, not four-layer groups | Runner passes `[layer]` for 0–31; real gate fingerprints show exactly seven changed matrices; strict per-layer audits and all-32 coverage | PASS |
| Explicit uniform W4 method | Seven attention/MLP projections, group 128 input-axis, FP32 absmax/7, zero point 0, round-to-even [-7,7], packed-code round trip, BF16 Q/DQ. No integer-kernel performance claim | PASS |
| Three datasets, smoke before formal | **24 × 32 = 768** heldout smoke rows validated before formal; **32 formal per dataset**, **96 × 32 = 3,072** actual rows | PASS |
| Dataset versions/splits/seed/IDs/prompts/parsers | `data_manifest.json`, `requests.jsonl`, baseline prompts, seed 20261015, pinned parquet revisions/digests, strict frozen parsers | PASS |
| Calibration/selection isolation | **24 train-only × 32 = 768** selection rows; 9 separate train demonstrations; all **153** source questions globally unique. Fixed/random sets frozen **20:17:12.884769 UTC**, before formal attempt **20:17:30.229534 UTC** | PASS |
| Complete 32-value vector for every request | `runs/evaluation/sensitivity_vectors.csv`; raw Cartesian-product validation of all **96 × 32** entries, not summaries alone | PASS |
| Same BF16 continuation, causal full-vocabulary mean KL | Every row retains all token KL values; independently checked all **125,824** values' means and **1,008,605,184 bytes** of BF16 reference tensors (96 shapes, finite values, 128,256 vocabulary, hashes). `steps/independent_formal_raw_check.json` | PASS |
| Actual answers and ground-truth scores, separate from KL | All baseline/single-layer/joint raw texts and token IDs retained; full executed raw validators; additional independent Decimal/label comparisons for **3,168** baseline/single-layer answers in `steps/independent_scoring_check.json` | PASS |
| All-request heatmaps and per-layer distributions | Four PNG/PDF figures rendered and visually inspected; all 96 requests/32 layers included, no removed failures or outliers | PASS |
| Top-4/8, ranking correlations/reversals and CIs | `analysis/statistics.csv`, 4,560 distinct request pairs, all 496 layer pairs per scope; 2,000 paired dataset-stratified request bootstraps, excluding duplicate-source self-pairs | PASS |
| Prompt/output length, dataset and difficulty-proxy analysis | Primary six-column controls, scalar log(mean KL) regression, length quartiles, all coefficients/residuals. Protocol cubic-both-lengths diagnostic added; earlier diagnostic preserved. R² ranges explicitly nominal/in-sample, nonlinear rank/leverage warnings surfaced | PASS, descriptive only |
| Within-request token-segment stability | Full/content halves for all eligible traces, explicit eligibility; content-half evidence limited to **32 GSM8K** requests, not ARC/BoolQ | PASS within stated scope |
| Fixed-budget actual joint precision allocation | **480 actual generated joint answers**, five strategies × 96 requests, each eight BF16/24 W4 layers; cached installed values, complements, precision and full restoration audited | PASS |
| Fixed selection independent of final data; random controls | Fixed `[0,1,8,9,10,11,13,31]` from train-only mean KL; three globally frozen seeds 20261016–20261018; current-request oracle separately labeled | PASS |
| Oracle upper bound distinguished from deployable routing | KL-information heuristic is not an accuracy bound; label oracle over five measured candidates is explicitly finite and label-using. No clustering/router/batching implementation | PASS |
| Joint interactions measured rather than additive accuracy prediction | Direct joint KL and generated answers in `precision_allocation_check/measurements.{csv,parquet}`; constituent-KL sum only a diagnostic | PASS |
| Ordered executable development gates | Actual environment/BF16 → single-layer → one-request 32-layer → smoke → train freeze → formal → allocations → analysis ledgers, each exit 0 before expansion | PASS |
| Preserve weak/negative outcomes and escalation order | All 96 requests retained; no W4 replacement or favorable subset. Frozen structure thresholds all pass; accuracy benefit remains unproven and is reported as such | PASS |
| Tests and separate smoke/full/report commands really run | **63 tests pass**; real smoke and full GPU workflow exit 0; final-code validated resume exit 0; two standalone CPU report commands exit 0 | PASS |
| Report reconstruction and human-readable handoff | **20 CSVs, report text and four PNGs exactly reproduce**, all report links resolve; README/onboarding/runbook updated; `report.md` cites real artifacts | PASS |
| Preserve existing work and use clear names | New `layer_sensitivity/` and `precision_allocation_check/`; old grouped data untouched and excluded; shared low-level helpers reused | PASS |

## Actual commands and exits

The wrappers retain exact commands, UTC times, CUDA visibility, stdout/stderr and exits. Prefixes below are under `steps/` and denote `.command`, `.environment`, `.log`, `.exit` files.

| Command | Exit | Evidence prefix |
|---|---:|---|
| `LAYER_GPU=7 bash scripts/run_layer_sensitivity.sh smoke` | 0 | `20261015T184927Z_smoke_command_369619` |
| `LAYER_GPU=7 bash scripts/run_layer_sensitivity.sh full` — real inference | 0 | `20261015T193554Z_full_command_369613` |
| `.venv/bin/python -m pytest -q tests` — 63 passed, 75.89 s | 0 | `20261015T235314Z_final_full_tests_942734` |
| `CUDA_VISIBLE_DEVICES='' bash scripts/run_layer_sensitivity.sh report` | 0 | `20261015T235515Z_report_rebuild_947027` |
| `LAYER_GPU=7 bash scripts/run_layer_sensitivity.sh full` — final-code validated resume, no repeated GPU scan | 0 | `20261015T235931Z_full_verified_resume_954889` |
| `CUDA_VISIBLE_DEVICES='' bash scripts/run_layer_sensitivity.sh report` — exact reconstruction | 0 | `20261016T000504Z_report_rebuild_reproducibility_962219` |

The real formal log `steps/full_20261015T193554Z_466717/20261015T201713Z_evaluation_551033.log` contains all 32 terminal `96/96` records. The allocation log `20261015T230131Z_precision_allocation_877775.log` in the same directory contains all five terminal `96/96` records and `PASS 480`. Original full and all child exits are 0.

## Hardware and numerical conclusions

One **NVIDIA GeForce RTX 3090, 24 GiB**, physical GPU **7**; dual Intel Xeon Gold 6226R, 64 logical CPUs, **251 GiB host RAM**. PyTorch **2.8.0+cu128**, CUDA runtime **12.8**, Transformers **4.56.1**. Hardware inventory: `steps/20261015T191921Z_hardware_inventory_429813.*`.

- Mean single-layer KL **0.0010087122 nats/token**, observed range **0.0000254353–0.0223338573**.
- Different-request Spearman **0.57370 [0.54206, 0.60318]**; Top-4 overlap **0.43783 [0.40692, 0.46948]**, Top-8 **0.52223 [0.50291, 0.54063]**.
- Within-request content-half Spearman **0.83829 [0.81341, 0.86199]**, **GSM8K only**. **462/496** material layer-pair reversals, **135/496** repeated across eligible content halves.
- BF16 **75/96 = 78.125%** (GSM8K 17/32, ARC 28/32, BoolQ 30/32). Fixed allocation **68/96 = 70.833%**, request-KL oracle **64/96 = 66.667%**, three-set random mean **68.403%**.
- Oracle minus fixed **−4.1667 pp [−11.458, +3.125]**: **no demonstrated benefit**, not equivalence or proof of impossibility. Label oracle **78/96 = 81.25%** is a finite-candidate hindsight bound only.

## Scientific stop and residual limitations

`analysis/escalation.json` was checked against the actual numbers, not trusted by itself: max KL 0.02233 > 1e−6 floor; 462 reversed pairs ≥2; eligible content-stability lower CI 0.81341 >0; heterogeneity lower CI 0.39682 >0.01. Independent scientific review agreed a further GPU cohort is not essential for these bounded questions. No extension/W3 was run merely to obtain favorable accuracy.

Limits remain explicit: one base checkpoint/template, small historically used benchmark subsets, deterministic capped outputs, possible pretraining contamination, post-model difficulty proxy, conditional random-set/selection inference, pointwise intervals, nominal in-sample R² ranges, rank-deficient nonlinear controls and short-task content ineligibility. There is no deployment or integer-kernel performance claim. Future independent cohorts could narrow accuracy uncertainty; this is not unfinished required inference.

Final additional artifact audit independently checked all **480** joint answers against ground truth, all fixed/random/oracle eight-layer choices, reference-continuation identity and full restoration; **54 documentation links** resolve (`steps/final_artifact_check.json`). Two initial ad-hoc audit-script schema assumptions were corrected; no experimental data or code changed. Staged artifact hygiene checked **6,256 files / 86,229,626 bytes**, excluding model tensors and finding no common credential patterns. A full staged whitespace check flags only verbatim command/log trailing spaces; these execution records are deliberately not rewritten. Source/documentation whitespace checking excludes that raw ledger directory.

Review findings and verified corrections: [resolution](layer_sensitivity_review_resolution.md). The study completion decision rests on inspected real execution, raw tensors/records, numerical reconstruction and bounded claims together—not a manifest, image or test count alone.
