# Individual-layer study: prompt-to-artifact completion audit

**Status: INCOMPLETE.** The active objective is not achieved. The earlier grouped-study audit and report are not acceptance evidence for a 32-individual-layer study. This checklist is intentionally stricter than a green test suite or generated manifest.

## Concrete success criteria

Deliver a reproducible, actually executed official Llama 3.1 8B BF16-versus-W4 **single-layer** study with complete 32-value vectors and generated answers for at least 32 requests each from GSM8K, ARC-Challenge and BoolQ, after 8-per-dataset smoke. Demonstrate exact intervention and KL validity, independently selected fixed-budget joint allocations with actual answers, all-request statistics/figures/CIs and a linked report. Tests and smoke/full/report commands must really execute successfully. Interpret nulls honestly; run predeclared targeted follow-ups if W4 cannot resolve the question. No router/clustering/production batching. Complete only after every applicable requirement is supported by inspected evidence.

## Evidence checklist (2026-10-15 UTC initial audit)

| User requirement | Concrete artifact/check | Inspected state / remaining evidence |
|---|---|---|
| RQ1: stable measurable request-specific layer differences | `analysis/statistics.csv`, complete heatmap, request pairs, reversals, segment table | **Pending formal data and interpretation**; statistical functions tested synthetically only |
| RQ2: which layers to retain; actual correctness consequence | top-8 selections, direct `precision_allocation_check/measurements.csv`, paired accuracy CIs | **Pending real allocations**; code tests do not prove utility |
| RQ3: request type/length/difficulty association | dataset/prompt/output/BF16-correctness controls, all layer coefficients, residuals, length quartiles | **Pending formal analysis**; difficulty is an explicitly limited BF16 correctness proxy |
| Official model, precise revision, tokenizer, template, decoding | `model_provenance.json`, `gates/baseline/metadata.json`, shared `model.py` | **Verified load and official file hashes**: base revision `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`; fixed explicit Q/A template, greedy deterministic decoding |
| BF16 reference | `gates/baseline/{baseline,metadata}.json` | **Passed real baseline**, `baseline_parser_reviewed` exit 0; actual prompt/token IDs and answer retained |
| Exactly one W4 layer; other 31 BF16; all 32 individually | `gates/single_layer/parameter_fingerprints.json`, `gates/single_request/layer_*.json`, final raw coverage | **Seven-matrix isolation/restoration passed; one request × 32 passed**. Formal 96 × 32 still missing; no group result substituted |
| Module list, W4 method/group size/scale/zero point consistent | `layer_sensitivity/quantization.py`, per-layer audit JSON, protocol | **Inspected and gated**: seven projections, symmetric RTN 128 input features, absmax/7 FP32 scale, zero point 0, [-7,7] round-to-even packed/QDQ; role-wide equality still awaits full runs |
| Activation/KV consistent; simulation disclaimer | real precision hooks in gates and per-role sentinels; README/protocol | **Gate probes BF16/BF16 and KV BF16 passed**; report pending. No integer-kernel performance claim |
| Three named datasets; 8 smoke before ≥32 formal each | frozen `requests.jsonl`, `data_manifest.json`, validators | **8-per-dataset × all-32-layer smoke completed and validated, exit 0**, with 768 raw W4 rows. Frozen 32 formal per dataset; no formal success claim |
| Calibration/selection isolation, dataset revisions/split/seed/IDs | data manifest, normalized-question hashes, train roles | **Frozen data inspected**; 3 demos + 8 selection from train each, heldout smoke/formal, seed 20261015; validators will recheck all identities |
| Record prompts and answer parsers | baseline JSON/CSV; `data.py` tests; parser review resolution | **Actual baseline prompt retained and parser tests pass**. Full prompt/answer retention awaits complete scan |
| Every request has 32 sensitivity values | `runs/evaluation/sensitivity_vectors.csv`, validator `check_coverage` | **Pending formal**. Tests explicitly reject 8-group, missing or duplicate layer rows |
| Same BF16 continuation teacher-forced mean KL | shared `metrics.py`, token lists/hashes, unit tests and real replay gates | **Causal-shift/direction tests and actual zero-discrepancy repeat/restore gates passed**. Every formal row will be checked for matching BF16 IDs/hash and mean token KL |
| Retain actual generated answers and ground-truth correctness | per-layer JSON, raw CSV/Parquet, `analysis/answers.csv` | **One-request 32 actual answers retained**; full data pending. KL and answer metrics separate |
| All-request heatmap; per-layer distributions | `figures/request_layer_heatmap.{png,pdf}`, `layer_distributions.{png,pdf}` | **Pending real-data render and visual inspection**. Synthetic figure test passed, not research evidence |
| Top-4/Top-8 overlap, request ranking correlation, reversal, CIs | `analysis/request_pairs.csv`, `ranking_reversals.csv`, `statistics.csv` | **Implemented/tested, not yet measured**. Request-level paired/stratified bootstrap; no independent-pair pseudo-replication |
| Control prompt/output lengths and dataset; stability across token segments | controls/segment/length tables, eligibility counts | **Implemented/tested, pending data**. Content-short ARC/BoolQ traces must not be presented as stable by imputation |
| Fixed high-precision budget: request/fixed/random | 8 BF16 / 24 W4; selection JSON; 5 actual configurations per request | **CPU state tests passed only**; 480 real joint answers still required |
| Fixed choice isolated from final evaluation | `selection.py`, pre-evaluation guard, timestamps/input hashes | **24 train requests × all 32 layers validated, exit 0**. Fixed layers [0,1,8,9,10,11,13,31] and random sets frozen **20:17:12.884769 UTC**, before formal attempt **20:17:30.229534 UTC**; final validator will recheck hashes/timing |
| Oracle upper bound separately labeled; no deployed router | protocol/README/report wording; finite-candidate label oracle | **Protocol correct**: current-request KL is oracle-information heuristic, not true accuracy bound; label oracle only over measured candidates. Numerical report pending |
| Multi-layer interactions directly measured | joint generation/KL and additive individual KL diagnostic | **Pending real joint measurements**; toy bank correctness is not a substitute |
| If W4 insufficient: quantify → diagnostics → more samples/longer → possible W3, preserve W4 | frozen escalation rules; `analysis/escalation.json`; follow-up journal | **Condition not yet evaluated**. Numerical intervention/parser/KL gates passed; follow-up obligations remain conditional on full W4 evidence |
| Ordered incremental development | actual step ledgers + root script dependency order | **Environment → BF16 → single-layer → single-request 32 → full smoke → train selection/freeze passed**; formal scan running. Remaining stages not yet certified |
| Clear descriptive naming | `layer_sensitivity/`, `precision_allocation_check`, `results/layer_sensitivity/` | **Present**; old names exist only in preserved historical code/evidence and explicit archive references |
| README: questions/method/install/hardware/reproduce | `README.md`, `doc/onboarding.md`, `doc/runbook.md` | **Updated to new study and honest in-progress state**; end-to-end command success still pending |
| Tests and executable scripts | `tests/test_layer_*.py`, existing tests, `run_layer_sensitivity.sh`, `record_command.sh` | **61-test current suite passed (exit 0, 78.28 s)**, including synthetic statistics/figure and joint-state checks. Rerun after any further code change |
| Per-request CSV/Parquet/raw results and figures | run exports + analysis outputs | **Pending complete roles**; initial gate CSV is not formal completion |
| Named `results/layer_sensitivity/report.md`: methods/N/numbers/CI/failures/limits/next testable question | `layer_sensitivity.analysis` renderer + resulting file | **Not yet produced from real complete data**. Must inspect actual report/source links/figures/numbers |
| Separate smoke/full/report commands really succeed | `LAYER_GPU=7 ... smoke`, `... full`, `... report` and `.exit` ledgers | **Separate smoke command exit 0** (`steps/20261015T184927Z_smoke_command_369619.*`); full command running after smoke, report not yet executed. Command definitions/help/syntax are not successful experiments |
| At least one real complete 32-layer formal evaluation | 96×32 JSON rows, exact coverage, exits | **NOT MET**; goal remains active |
| Final response lists actual commands/exit/hardware/N/main numbers/paths | final response bound to latest audit and ledgers | **Pending final completion**. Partial progress responses must say incomplete |
| No fabrication/cherry-picking, no clustering/router/batching | all frozen sample IDs retained, code scope inspection, failure tables | **Protocol/code scope aligned so far**. Final retention and claim audit still required |

## Verification coverage and gaps

The new validator verifies raw layer coverage, tokenizer/rendered prompts, decoding/scoring, KL arithmetic and matching reference identities, source hashes, seven-matrix audits, precision/replay/restoration sentinels and pre-formal selection timing. Allocation validation separately recomputes chosen sets, budget/complement, raw answers and observed isolation hashes.

It does **not** by itself establish a stable population effect, make accuracy improvements causal, quantify unseen random layer sets, prove independent task difficulty, inspect plotted readability, certify every README reproduction command, or decide whether additional experiments are scientifically needed. Those require manual analysis and this prompt-to-artifact audit.

Current decisive gaps: no complete formal or joint-allocation evidence, no real formal statistics/report, no successful full/report command, no final claim/figure review. **Do not call the goal complete.**
