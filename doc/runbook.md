# Runbook

## Project Summary

M1 only: real Llama 3.1 8B BF16 versus one fixed W4 layer group, complete benchmark measurements, robust statistics and a bounded report. All commands below run from the repository root.

## Setup

Requirements: Linux, Python 3.12, uv (recorded version 0.12.15), NVIDIA driver supporting the pinned CUDA 12.8 Torch build, BF16 GPU with about 24 GiB VRAM. Obtain authorized access to `meta-llama/Llama-3.1-8B` and cache/login through Hugging Face. Credentials are external to the repo.

```bash
bash scripts/install.sh
```

This syncs all 60 versions from `requirements.lock` into `.venv`. Model/datasets use immutable revisions; their cached paths follow Hugging Face settings (`HF_HOME` if configured). Initial model/dataset access requires network unless already cached. Installation, cached model verification and individual inference stages were executed successfully; a separate completely empty-machine rebuild was not performed.

## Run

```bash
M1_GPU=1 bash scripts/run_m1.sh
# Separate complete inference reproduction, without overwriting the original:
M1_OUT=results/m1_reproduction M1_GPU=1 bash scripts/run_m1.sh
```

Choose a GPU allocated to you. The script uses one GPU sequentially, takes a repository lock, verifies official weights, executes missing baseline → single-group → KL → eight-group gates, prepares fixed samples, runs smoke → calibration → evaluation, freezes fixed choices before evaluation, and produces analysis/figures/report/audit.

Outputs: `M1_OUT/{report.md,analysis/,runs/,completion_checks.json}`; command/exit/log ledger under `M1_OUT/steps/one_click_*`. Existing completed inference is reused and revalidated. Previous derived analysis/report are archived in that ledger. Success: overall exit 0 and `M1 artifacts verified`; inspect the scientific conclusion separately. Fresh inference runtime depends on GPU contention; no fresh one-command total-runtime benchmark is claimed.

## Test

```bash
.venv/bin/python -m pytest -q tests
# Audit actual completed smoke, not a synthetic substitute:
.venv/bin/python -m scripts.validate_runs --role smoke \
  --runs results/m1/runs/smoke_a results/m1/runs/smoke_b \
  --out results/m1/smoke_validation.json
```

Recorded final suite: 27 passed. Smoke: 8 sources/dataset, both wordings, 384 W4 rows. Unit tests use small synthetic tensors only for mathematical/software correctness; raw experiment records always come from the real model.

## Evaluate

```bash
.venv/bin/python -m scripts.validate_runs --role evaluation \
  --runs results/m1/runs/evaluation_a results/m1/runs/evaluation_b \
  --out results/m1/evaluation_validation.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.calibrate_statistics
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.analyze_m1 --bootstrap 2000
.venv/bin/python -m scripts.check_leverage  # diagnostic copy only; preserves full primary sample
.venv/bin/python -m scripts.plot_m1
.venv/bin/python -m scripts.report_m1
.venv/bin/python -m scripts.audit_m1
```

`calibrate_statistics` is explicitly synthetic method-validation, not model evaluation. It must match the current analysis source hash. Preserve earlier analysis before manual methodology changes (or use `analyze_m1 --out NEW_DIRECTORY`); the one-command runner archives it automatically.

Metrics: KL nats/token, rankings/reversals, actual correctness, paired bootstrap intervals, controlled profile association and approximate wild-score tests. Expected formal coverage: 96 independent sources, 192 prompts, 1,536 W4 rows, 1,728 actual answers. Report sources/wordings separately; 192 is not the number of independent bootstrap clusters.

## Common Failures

| Symptom | Likely Cause | Inspect | Fix |
|---|---|---|---|
| Model HTTP 401/403 | Gated model access/login unavailable | Hugging Face account authorization | Obtain authorized access; do not replace the model |
| `ModuleNotFoundError` | Wrong Python/incomplete install | `.venv/bin/python --version`; install log | Run `scripts/install.sh`, use `.venv/bin/python` |
| CUDA OOM / slowdown | Shared GPU occupied | `nvidia-smi`; own attempt metadata | Resume on allocated GPU; never kill others' jobs |
| `inputs/code changed` | Changed measurement source/sample | `run_config.json` hashes | Use new output directory; preserve old evidence |
| Missing reference cache during partial resume | Git ignores `.pt` files | `reference_logits/` | Reproduce into new directory; do not fabricate logits |
| Calibration/analysis hash assertion | Source or frozen inputs changed | `calibration_selection.json`, analysis metadata | Diagnose, preserve old analysis, rerun appropriate validated stage |
| Lock already held | Another one-command run | Process/ledger for your run | Wait or stop only your own run; do not launch concurrent writers |
| Approximate permutation mistaken for exact p | Invalid exchangeability interpretation | Statistical review/plan | Use qualified wild-score results and uncertainty, retain caveats |

## Recovery Steps

1. Read the failed stage's `.log` and `.exit` in the one-click ledger. Do not delete successful raw checkpoints.
2. For an interrupted measurement with unchanged code/sample, rerun the same shard command, choosing an allocated GPU:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m m1_run \
  --role evaluation --wording a --out results/m1/runs/evaluation_a
```

3. Revalidate both wordings, then rerun analysis/report/audit. `M1_GPU=1 bash scripts/run_m1.sh` performs these steps and resumes incomplete shards.
4. If measurement code or frozen inputs changed, set a new `M1_OUT`. Never disable immutable-input guards. Interrupted attempts remain evidence, not successful measurements.

## Useful Commands

```bash
nvidia-smi
.venv/bin/python -m scripts.provenance --out results/m1/steps/provenance_recheck.json
.venv/bin/python -m scripts.select_fixed  # checks existing frozen decision, does not reselect from evaluation
bash -n scripts/run_m1.sh
git diff --check
```

No destructive cleanup, publishing, or push command is required. There is no project lint/type-check configuration; unit/integration/evidence checks are the declared checks.

## File Locations

| Path | Purpose | Notes |
|---|---|---|
| `results/m1/requests.jsonl` | All 153 source records | Demos/calibration/smoke/evaluation disjoint |
| `results/m1/model_provenance.json` | Official shard/tokenizer digests | Four real model shards |
| `results/m1/runs/{role}_{wording}/` | Actual prompts, answers, scores, KL and audits | Resume only with identical measurement inputs/code |
| `results/m1/runs/*/reference_logits/` | BF16 local tensors | Retained locally, Git-ignored |
| `results/m1/analysis/*.csv` | Complete derived tables | All 12 matched exactly on repeated analysis |
| `results/m1/report.md` | Bounded scientific conclusion | Includes failures, null cases and oracle limitations |
| `results/m1/last_one_click_run.txt` | Latest ledger path | Per-step command/log/exit files |
| `doc/statistical-validation-plan.md` | Review-driven analysis amendment | Written before formal outcomes inspected |

## Operational Notes

- `M1_GPU` controls `CUDA_VISIBLE_DEVICES`; no automated GPU eviction or process termination.
- `M1_OUT` selects the experiment output directory. Model/dataset caches are external; do not commit weights or credentials.
- Seed 20261015; base model, fixed three-shot explicit template, symmetric block-128 W4 RTN/QDQ. No packed-runtime performance claim.
- Independent source-level uncertainty, finite sample size and fixed calibration/demos matter. Label/BF16-assisted policies are oracles, not routers.
- Cached/full scoring paths differ numerically. Primary KL is always matched full-sequence teacher forcing; actual answers are separate greedy decodes.
- M1 does not include M2–M4 or automatic follow-on research.

## Last Verified

- Date: 2026-10-15 UTC (2026-10-16 local +08:00).
- Verified commands: locked install, 27-test suite, all real incremental GPU gates, complete smoke/calibration/evaluation, strict raw validators, synthetic method calibration, 2,000-replicate formal analysis, plots/report/audit, and one-command resume/re-analysis — all final exits 0.
- Final ledger: `results/m1/steps/one_click_20261015T173520Z_209164/` (including leverage influence diagnostic); exact 12-CSV re-analysis comparison passed. The earlier one-click ledger is also preserved.
- Known unverified commands: the separate **fresh-output one-command** campaign has not been redundantly run a second time; its individual inference commands were actually executed during the original staged campaign. No claim of an empty-machine rebuild or identical answers on untested hardware.
