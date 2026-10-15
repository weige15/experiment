# Runbook

## Project Summary

Measure request-dependent layer quantization sensitivity: real Llama 3.1 8B BF16 versus one fixed W4 layer group, complete benchmark measurements, robust statistics and a bounded report. All current commands below run from the repository root and default to `results/quantization_sensitivity/`.

## Setup

Requirements: Linux, Python 3.12, uv (recorded version 0.12.15), NVIDIA driver supporting the pinned CUDA 12.8 Torch build, BF16 GPU with about 24 GiB VRAM. Obtain authorized access to `meta-llama/Llama-3.1-8B` and cache/login through Hugging Face. Credentials are external to the repo.

```bash
bash scripts/install.sh
```

This syncs all 60 versions from `requirements.lock` into `.venv`. Model/datasets use immutable revisions; their cached paths follow Hugging Face settings (`HF_HOME` if configured). Initial model/dataset access requires network unless already cached. Installation, cached model verification and individual inference stages were executed successfully; a separate completely empty-machine rebuild was not performed.

## Run

```bash
EXPERIMENT_GPU=1 bash scripts/run_experiment.sh
# Separate complete inference reproduction, without overwriting a previous run:
EXPERIMENT_OUT=results/quantization_sensitivity_reproduction EXPERIMENT_GPU=1 bash scripts/run_experiment.sh
```

Choose a GPU allocated to you. The script uses one GPU sequentially, takes a repository lock, verifies official weights, executes missing baseline → single-group → KL → eight-group gates, prepares fixed samples, runs smoke → calibration → evaluation, freezes fixed choices before evaluation, and produces analysis/figures/report/audit.

Outputs: `EXPERIMENT_OUT/{report.md,analysis/,runs/,completion_checks.json}`; command/exit/log ledger under `EXPERIMENT_OUT/steps/one_click_*`. Existing completed inference with identical source/input hashes is reused and revalidated. Previous derived analysis/report are archived in that ledger. Success: overall exit 0 and `Quantization sensitivity artifacts verified`; inspect the scientific conclusion separately. Fresh inference runtime depends on GPU contention; no fresh one-command total-runtime benchmark is claimed.

Do not point the renamed runner at the archived `results/m1/` campaign. Renaming imports changes source hashes, so it requires a fresh output directory. See [Operational Notes](#operational-notes) for historical reproduction.

## Test

```bash
.venv/bin/python -m pytest -q tests
# Audit actual completed smoke, not a synthetic substitute:
.venv/bin/python -m scripts.validate_runs --role smoke \
  --runs results/quantization_sensitivity/runs/smoke_a results/quantization_sensitivity/runs/smoke_b \
  --out results/quantization_sensitivity/smoke_validation.json
```

Current CPU suite: 41 passed, including entrypoint and immutable-source regression checks. The original campaign's suite had 27 tests. Smoke: 8 sources/dataset, both wordings, 384 W4 rows. Unit tests use small synthetic tensors only for mathematical/software correctness; raw experiment records always come from the real model.

## Evaluate

```bash
.venv/bin/python -m scripts.validate_runs --role evaluation \
  --runs results/quantization_sensitivity/runs/evaluation_a results/quantization_sensitivity/runs/evaluation_b \
  --out results/quantization_sensitivity/evaluation_validation.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.calibrate_statistics
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.analyze_results --bootstrap 2000
.venv/bin/python -m scripts.check_leverage  # diagnostic copy only; preserves full primary sample
.venv/bin/python -m scripts.plot_results
.venv/bin/python -m scripts.report_results
.venv/bin/python -m scripts.audit_results
```

These commands require a completed current-source campaign. `calibrate_statistics` is explicitly synthetic method-validation, not model evaluation. It must match the current analysis source hash. Preserve earlier analysis before manual methodology changes (or use `scripts.analyze_results --out NEW_DIRECTORY`); the one-command runner archives it automatically. For a custom campaign directory, pass `--requests` to validation, `--root` to analysis/report/audit, and `--analysis`/`--out` to plotting.

Metrics: KL nats/token, rankings/reversals, actual correctness, paired bootstrap intervals, controlled profile association and approximate wild-score tests. Expected formal coverage: 96 independent sources, 192 prompts, 1,536 W4 rows, 1,728 actual answers. Report sources/wordings separately; 192 is not the number of independent bootstrap clusters.

## Common Failures

| Symptom | Likely Cause | Inspect | Fix |
|---|---|---|---|
| Model HTTP 401/403 | Gated model access/login unavailable | Hugging Face account authorization | Obtain authorized access; do not replace the model |
| `ModuleNotFoundError` | Wrong Python/incomplete install | `.venv/bin/python --version`; install log | Run `scripts/install.sh`, use `.venv/bin/python` |
| CUDA OOM / slowdown | Shared GPU occupied | `nvidia-smi`; own attempt metadata | Resume on allocated GPU; never kill others' jobs |
| `inputs/code changed` / `measurement sources changed` | Changed measurement source/sample, including import renames | `run_config.json` hashes | Use new output directory; validate archived evidence with its original source snapshot |
| Missing reference cache during partial resume | Git ignores `.pt` files | `reference_logits/` | Reproduce into new directory; do not fabricate logits |
| Calibration/analysis hash assertion | Source or frozen inputs changed | `calibration_selection.json`, analysis metadata | Diagnose, preserve old analysis, rerun appropriate validated stage |
| Lock already held | Another one-command run | Process/ledger for your run | Wait or stop only your own run; do not launch concurrent writers |
| Approximate permutation mistaken for exact p | Invalid exchangeability interpretation | Statistical review/plan | Use qualified wild-score results and uncertainty, retain caveats |

## Recovery Steps

1. Read the failed stage's `.log` and `.exit` in the one-click ledger. Do not delete successful raw checkpoints.
2. For an interrupted measurement with unchanged code/sample, rerun the same shard command, choosing an allocated GPU:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m quantization_sensitivity.runner \
  --role evaluation --wording a --out results/quantization_sensitivity/runs/evaluation_a
```

3. Revalidate both wordings, then rerun analysis/report/audit. `EXPERIMENT_GPU=1 bash scripts/run_experiment.sh` performs these steps and resumes incomplete shards.
4. If measurement code or frozen inputs changed, set a new `EXPERIMENT_OUT`. Never disable immutable-input guards or rewrite old source manifests. Interrupted attempts remain evidence, not successful measurements.

## Useful Commands

```bash
nvidia-smi
.venv/bin/python -m scripts.provenance --out results/quantization_sensitivity/steps/provenance_recheck.json
.venv/bin/python -m scripts.select_fixed  # checks existing frozen decision, does not reselect from evaluation
.venv/bin/python -m quantization_sensitivity.runner --help
bash -n scripts/run_experiment.sh
git diff --check
```

No destructive cleanup, publishing, or push command is required. There is no project lint/type-check configuration; unit/integration/evidence checks are the declared checks.

## File Locations

| Path | Purpose | Notes |
|---|---|---|
| `results/quantization_sensitivity/requests.jsonl` | All 153 source records | Demos/calibration/smoke/evaluation disjoint |
| `results/quantization_sensitivity/model_provenance.json` | Official shard/tokenizer digests | Four real model shards |
| `results/quantization_sensitivity/runs/{role}_{wording}/` | Actual prompts, answers, scores, KL and audits | Resume only with identical measurement inputs/code |
| `results/quantization_sensitivity/runs/*/reference_logits/` | BF16 local tensors | Retained locally, Git-ignored |
| `results/quantization_sensitivity/analysis/*.csv` | Complete derived tables | Source-clustered statistics and full records |
| `results/quantization_sensitivity/report.md` | Bounded scientific conclusion | Includes failures, null cases and oracle limitations |
| `results/quantization_sensitivity/last_one_click_run.txt` | Latest ledger path | Per-step command/log/exit files |
| `results/m1/` | Original campaign archive | Kept byte-for-byte unchanged; see source snapshot below |
| `doc/statistical-validation-plan.md` | Review-driven analysis amendment | Written before formal outcomes inspected |

## Operational Notes

- `EXPERIMENT_GPU` controls `CUDA_VISIBLE_DEVICES`; no automated GPU eviction or process termination.
- `EXPERIMENT_OUT` selects the experiment output directory (default `results/quantization_sensitivity`). Model/dataset caches are external; do not commit weights or credentials.
- The package is `quantization_sensitivity`; use `python -m quantization_sensitivity.runner` for one shard. The former `m1_*.py` modules and `scripts/*_m1` commands have no compatibility wrappers.
- Historical “M1” evidence and dated protocol/review/audit documents retain their original names. Commit `54fefa363211aa515a7eb70b2be30e01dd8318ff` was checked against **all six archived measurement source manifests**, the final analysis/method-calibration source hashes, and the report-renderer hash. Use that snapshot in a separate checkout with its original instructions for historical validation/resume; do not substitute new code or alter the recorded hashes. A checkout does not include Git-ignored reference tensors needed for partial inference resume.
- Seed 20261015; base model, fixed three-shot explicit template, symmetric block-128 W4 RTN/QDQ. No packed-runtime performance claim.
- Independent source-level uncertainty, finite sample size and fixed calibration/demos matter. Label/BF16-assisted policies are oracles, not routers.
- Cached/full scoring paths differ numerically. Primary KL is always matched full-sequence teacher forcing; actual answers are separate greedy decodes.
- This measurement study does not include request routing, clustering, batching or automatic follow-on research.

## Last Verified

- Naming refactor: `.venv/bin/python -m pytest -q tests` (41 passed), renamed Python CLI `--help` checks without GPU/network, `bash -n scripts/run_experiment.sh scripts/install.sh`, and Python compilation. No new GPU inference is claimed.
- Offline regression check: on temporary copies of the original evidence, `scripts.calibrate_statistics` reproduced all 10 scenario results exactly; `scripts.analyze_results --bootstrap 2000` reproduced all 12 analysis CSVs exactly. `scripts.plot_results` and `scripts.report_results` completed on those copies. Archived raw records were not re-stamped as measurements from the new source.
- Original campaign date: 2026-10-15 UTC (2026-10-16 local +08:00).
- Historical verified commands: locked install, original 27-test suite, all real incremental GPU gates, complete smoke/calibration/evaluation, strict raw validators, synthetic method calibration, 2,000-replicate formal analysis, plots/report/audit, and one-command resume/re-analysis — all final exits 0.
- Final ledger: `results/m1/steps/one_click_20261015T173520Z_209164/` (including leverage influence diagnostic); exact 12-CSV re-analysis comparison passed. The earlier one-click ledger is also preserved.
- Known unverified commands: the renamed **GPU inference and full one-command** workflow have not been run as a new campaign. Its original constituent stages were executed before the refactor. No claim of an empty-machine rebuild or identical answers on untested hardware.
