# Runbook

## Project Summary

Run the real **32 individual-layer** Llama 3.1 8B W4 sensitivity study, then a fixed-budget joint precision-allocation check. Commands run from the repository root; default output `results/layer_sensitivity/`. The preserved earlier group study is not interchangeable; see [archived runbook](archived_group_study_runbook.md).

## Setup

Linux/Bash/flock, Python 3.12, uv, authorized official model login/cache, BF16-capable NVIDIA GPU (~24 GiB), driver supporting pinned CUDA 12.8 PyTorch. Recommend ≥48 GiB available host RAM and storage for ~16 GB model weights plus several GB output. Never place credentials in repo.

```bash
bash scripts/install.sh
```

This syncs `requirements.lock` into `.venv`. Hugging Face cache locations follow normal `HF_HOME` settings. Model revision and dataset revisions are pinned in shared model/data modules; model shards are checked against official digests.

## Run

```bash
# 24 heldout requests × all 32 layers, after incremental real-model gates:
LAYER_GPU=7 bash scripts/run_layer_sensitivity.sh smoke

# Reuse/revalidate identical smoke, then 24 train-only selection requests,
# freeze choices, 96 formal requests × 32 layers, 480 joint answers, report:
LAYER_GPU=7 bash scripts/run_layer_sensitivity.sh full

# Fresh inference reproduction in a separate root:
LAYER_OUT=results/layer_sensitivity_reproduction LAYER_GPU=7 \
  bash scripts/run_layer_sensitivity.sh full
```

`LAYER_GPU` must refer to a device allocated to you. No automatic eviction or batching. Runtime is several hours on a shared RTX 3090; no measured full-runtime claim until the new campaign finishes. Each root is locked. Existing matching completed stages are revalidated, not rerun; partial stages resume only if input/source hashes match. Train-only choices must predate formal evaluation.

Success surfaces: smoke validator for all 24 × 32 rows; formal validator for all 96 × 32 rows; allocation validator for all 96 × 5 joint configurations; per-command `.exit` and final ledger `overall.exit` equal 0. These are necessary but not sufficient for completion: [requirement audit](layer_sensitivity_completion_audit.md) includes interpretation and manual verification.

## Test

```bash
.venv/bin/python -m pytest -q tests
bash -n scripts/run_layer_sensitivity.sh scripts/record_command.sh scripts/install.sh
git diff --check
```

CPU tests exercise integer codes, causal KL, strict parsing, all individual-layer targets, exception restoration, cached joint allocation state, source guards, full-32 coverage, paired bootstrap and figures. Synthetic values are confined to tests, not model results. Real incremental gates are launched by smoke/full; inspect `gates/` and their logged exits.

## Evaluate

```bash
.venv/bin/python -m layer_sensitivity.validate --gates
.venv/bin/python -m layer_sensitivity.validate --role smoke
.venv/bin/python -m layer_sensitivity.validate --role calibration
.venv/bin/python -m layer_sensitivity.validate --role evaluation
.venv/bin/python -m layer_sensitivity.validate --allocation

# Separate CPU-only report rebuild, after real inference is complete:
bash scripts/run_layer_sensitivity.sh report
```

Use `--root PATH` for validators or `LAYER_OUT=PATH` for the shell command. Report reconstruction reruns formal/allocation validation, archives previous derived analysis/report, then runs `python -m layer_sensitivity.analysis --bootstrap 2000`. It does not load a GPU model or regenerate raw answers. Primary values are full-vocabulary mean KL in nats/token, 32-layer ranks, Top-4/8 overlap, pair correlations/reversals, actual correctness, paired request-bootstrap CIs and descriptive controls. See the protocol for token-segment eligibility, oracle distinctions and escalation criteria.

## Common Failures

| Symptom | Likely cause | Inspect | Fix |
|---|---|---|---|
| Model 401/403 | Missing gated authorization | Authorized HF account/cache | Obtain permission, never swap model |
| Missing Python module | Wrong environment | `.venv/bin/python --version`, lock | Use `.venv`, install before launching inference |
| CUDA OOM/slow run | GPU contention | `nvidia-smi`, attempt metadata | Resume on an allocated device; preserve failure |
| Input/source assertion | Measurement code/data changed | `run_config.json` | New output root; never restamp old hashes |
| Lock held | Another writer | `ps`, command ledger | Do not launch concurrent writes into one root |
| Missing reference tensors | Partial run/cache not preserved | `reference_logits/` | Fresh reproduction root, never invent logits |
| Coverage assertion | Missing or duplicate layers/requests | JSON rows, run summary | Complete actual missing inference, revalidate |
| Frozen selection assertion | Selection differs or postdates evaluation | selection JSON, attempts | Investigate contamination; no manual success flag |
| Escalation required | W4 signal/segment evidence unresolved | `analysis/escalation.json` | Recorded diagnostics → deterministic sample extension → optional W3 |

## Recovery Steps

1. Inspect the failed step's `.command`, `.environment`, `.log`, `.exit`. Preserve all attempts.
2. Ensure no process still holds the output lock and measurement code/sample hashes have not changed.
3. Resume the same shell command on an allocated device:

```bash
LAYER_GPU=7 bash scripts/run_layer_sensitivity.sh full
```

For diagnosis of an individual shard (do not run concurrently with the shell workflow):

```bash
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m layer_sensitivity.runner \
  --role evaluation --requests results/layer_sensitivity/requests.jsonl \
  --out results/layer_sensitivity/runs/evaluation
```

4. Revalidate raw results and rebuild the report. If inputs/code changed, use a new root and retain old evidence. Analysis changes alone do not require GPU reruns, but must be recorded and previous derived outputs preserved.

## Useful Commands

```bash
nvidia-smi
.venv/bin/python -m layer_sensitivity.runner --help
.venv/bin/python -m layer_sensitivity.precision_allocation_check --help
.venv/bin/python -m layer_sensitivity.analysis --help
find results/layer_sensitivity/steps -name '*.exit' -print
```

No publishing, destructive cleanup, lint framework or production server is required.

## File Locations

| Path | Purpose | Notes |
|---|---|---|
| `results/layer_sensitivity/{requests.jsonl,data_manifest.json}` | Frozen 153 sources and dataset revisions | Demo/selection/smoke/eval isolated |
| `results/layer_sensitivity/model_provenance.json` | Official model/tokenizer hashes | Four real shards |
| `results/layer_sensitivity/gates/` | BF16, 7-matrix isolation, 1-request 32-layer proof | Actual model only |
| `results/layer_sensitivity/runs/{smoke,calibration,evaluation}/` | Raw per-request JSON, CSV, Parquet, vectors, audits | Each request × 32 layers |
| `results/layer_sensitivity/runs/*/reference_logits/` | Fixed BF16 logits | Local/Git-ignored |
| `results/layer_sensitivity/precision_allocation_selection.json` | Train-only fixed top-8 + three random sets | Frozen before evaluation |
| `results/layer_sensitivity/precision_allocation_check/` | Direct joint configuration results | 8 BF16 / 24 W4 layers |
| `results/layer_sensitivity/{analysis,figures}/` | All-request statistics and plots | No cherry-picked sample exclusion |
| `results/layer_sensitivity/report.md` | Methods, numerical results/CIs, failures, limits | Exists after actual complete inference |
| `results/layer_sensitivity/steps/` | Commands/environment/logs/exits | Append-only ledgers |
| `doc/layer_sensitivity_{protocol,progress,completion_audit}.md` | Design and evidence-based status | Manual completion audit required |

## Operational Notes

- Base checkpoint has no upstream chat template; the recorded explicit three-shot Q/A template is fixed across configurations.
- Main scan passes exactly one layer to `quantized_layers`. Do not use the preserved grouped runner for this goal.
- Weight Q/DQ uses BF16 kernels. No claimed W4 throughput/VRAM savings.
- Precision allocation CPU bank installs cached Q/DQ weights and audits exact state. Combined configurations are directly measured; not predicted by summing individual effects.
- Request-KL oracle uses evaluation information. Label oracle is separately an upper bound over five measured candidates. Neither is a router.
- New GSM8K parser rejects partial numeric prefixes and uses the last answer marker; historical raw grouped scores are not silently rewritten.
- Old grouped source snapshot `54fefa363211aa515a7eb70b2be30e01dd8318ff` is required to validate its archived manifests; do not alter those manifests.

## Last Verified

- Date: 2026-10-15 UTC (2026-10-16 local).
- Verified: pinned real weights and data; fresh BF16 baseline; single-layer exact 7-matrix isolation/restoration/replay; single-request all-32-layer scan; strict gate validator; **61-test full suite**; Bash syntax; **24 × 32 smoke and raw validation**, including separate smoke command exit 0.
- Verified next stage: train-only 24 × 32 selection scan and validation; fixed/random choices frozen before formal attempt.
- Running: full command, currently formal 96-request × 32-layer scan.
- Not yet verified: formal 96 × 32 scan, real joint allocations, real-data report reconstruction, and final objective completion. Update this section only from actual new exits.
