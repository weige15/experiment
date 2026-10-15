# M1 requirement-to-evidence audit

**Current decision: NOT ACHIEVED.** This checklist is not a substitute for inspecting the artifacts. It must be updated against the completed run before `update_goal(complete)`.

Concrete target: reproducible, real-weight request×layer-group measurements on 96 formal source requests (32/dataset), robust and materially interpretable request-level sensitivity evidence, complete uncertainty/failure reporting, and an executable reproducibility package. No router/M2–M4.

| User requirement | Concrete evidence / inspection | Current status |
|---|---|---|
| 1. Real official Llama 3.1 8B weights | `results/m1/model_provenance.json`: four shards + tokenizer/config files matched pinned official Hub digests; `provenance.exit=0` | Verified |
| 1. BF16 reference, same tokenizer/template/activation/KV | Baseline and one-group metadata precision hooks; `m1_model.py`; `chat_template.jinja`; pinned base model (not Instruct) | Verified at incremental gates; full-set checks pending |
| 1. W4 weight-only first | `m1_quant.py`; integer nibble round-trip tests; real g0 intervention. Q/DQ BF16 compute is disclosed, no integer-kernel speed claim | Verified |
| 2. 8×4 contiguous groups, one W4 at a time, fixed method | `steps/eight_groups/measurements.csv`, eight groups complete; exact restoration per group; all-group full-set audit pending | Partial |
| 3. GSM8K/ARC-Challenge/BoolQ | Frozen `requests.jsonl`, pinned data revisions/parquet digests in `data_manifest.json` | Verified |
| 3. Fixed seed; calibration/evaluation isolation | Seed 20261015; train-only 3 demos + 8 calibration/dataset; test/validation 8 smoke + 32 evaluation; question hashes, disjoint IDs | Verified sampling; calibration measurements pending |
| 3. Smoke 8/dataset before >=32/dataset evaluation | Smoke A/B running, 24 unique source requests; formal inference not started | Pending |
| 4. Token-level BF16→W4 KL | Analytic KL and causal-shift tests; actual 116-token first-request traces; independent repeat/restoration checks; full set pending | Partial |
| 4. Per-request layer ranking | Eight-group first-request CSV ranks; complete formal rankings pending | Partial |
| 4. Actual answers and automatic correctness | Baseline and eight-group raw answers/IDs/gold/scores saved; parser unit tests; no fake numeric fixtures in results | Partial |
| 5. High-sensitivity groups and ranking reversals | Full-set summaries, all 28 pairs, within-dataset and replicated-wording reversals | Missing |
| 5. Wording / segment stability | All formal requests under two predefined wordings; full and content-only token segments, first token; short-segment denominators | Missing |
| 5. Dataset / length controls | Normalized row profiles + residual controls + length-stratified permutation; pure multiplicative-profile null unit test | Missing |
| 5. Equal-budget request-specific vs fixed quality | Calibration-fixed group; eval-best-fixed, BF16-KL, label oracles; actual selected answers, paired CIs, cross-wording transfer | Missing |
| 6. Entire evaluation set, not chosen examples | Complete rectangular raw records and summary CSV, all exclusions/failures accounted for, 96 effective source requests | Missing |
| 6. Bootstrap CIs | 2,000 dataset-stratified source-cluster replicates preserving wordings/groups, paired comparisons, conditional calibration caveat | Missing |
| 6. Failures / no-effect cases / oracle caveats | Full failure/near-tie outputs; report numerical-path discrepancy, truncation, accuracy losses; oracle upper bound is not a router | Missing final analysis |
| 7. Reproducible installation | `requirements.lock`, `scripts/install.sh`, executed `install-locked.exit=0` | Verified |
| 7. One-command M1 | Full orchestrator, including gates and analysis | Missing |
| 7. Executed unit/smoke tests | 16 unit tests passed before smoke (`pre-smoke-tests.*`); real-model stages passed; full smoke validation pending | Partial |
| 7. Raw CSV/Parquet, heatmap, answers/accuracy | First-request raw CSV exists; formal raw CSV, heatmap, accuracy comparisons pending | Partial |
| 7. `results/m1/report.md` | Must reflect actual full-set results, not presumed success | Missing |
| 7. Hardware, revision, quantizer, seed, parameters | Per-stage/attempt metadata, pinned sources, run configs, command logs | Verified so far; final index pending |
| 8. Actual executed experiments/tests; final commands, exit codes, N, stats, paths | Step logs/exit files and intermediate artifacts exist; final verified output pending | Partial |
| Incremental development / preserve earlier results | Baseline → one-group → KL → eight-group actually run in order. Smoke running. Original logs retained | Verified so far |
| Investigate inadequate W4 evidence rather than declare success | Controlled statistics and robustness checks required before deciding; no evidence-based conclusion yet | Pending |
| M1-only scope | Current code contains measurement, scoring, validation only; no request router, clustering or batching | Verified so far |

## Known limitations/failures retained
- Initial dependency download exceeded the 600-second foreground tool timeout; background install succeeded, followed by a successful locked install.
- First BF16 GSM8K answer was wrong (8 vs 50). This is a real task failure, not a failed inference gate.
- Cached decode vs full teacher forcing has measured nonzero numeric-path KL. Primary comparisons use the same full-sequence path; independent same-path replay and restoration were exact.
- One smoke-A BF16 GSM8K answer reached the 256-token limit while looping. Keep it as an unsuccessful answer, not an excluded request.
- GPU 7 was occupied by another user's workload during smoke B. Only our process was terminated (exit 143), its completed checkpoints retained, and it resumed on GPU 1. Retain both attempt logs and hardware metadata.

Passing software tests does **not** yet establish request-dependent sensitivity. The scientific rows above are still missing.
