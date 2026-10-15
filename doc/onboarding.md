# Onboarding

## What This Project Does

M1 measures whether requests differ in their sensitivity to quantizing different four-layer groups of real Llama 3.1 8B weights. The output is `results/m1/report.md` with complete raw measurements, actual answers, uncertainty and negative results. It does not implement a deployable router.

## Quickstart

From the repository root, with Python 3.12, uv, an authorized official-model login/cache and an allocated BF16 CUDA GPU:

```bash
bash scripts/install.sh
.venv/bin/python -m pytest -q tests
M1_GPU=1 bash scripts/run_m1.sh
```

Successful execution ends with `M1 artifacts verified`, exit 0, and a ledger identified in `results/m1/last_one_click_run.txt`. Read the report's limitations; the exit code alone is not proof of a research hypothesis.

## Important Files

| File | Role |
|---|---|
| `doc/protocol.md` | Fixed design and transparent pre-result amendments |
| `m1_data.py` | Pinned benchmark revisions, sampling, formatting, automatic scoring |
| `m1_model.py` | Official BF16 model, greedy decoding, dtype/finite-logit checks |
| `m1_quant.py` | Fixed RTN W4 packing/QDQ, exactly one group, restoration |
| `m1_metrics.py` | Causally aligned full-vocabulary KL |
| `m1_run.py` | Immutable-input, resumable real measurement shards |
| `m1_analysis.py` | Paired source bootstrap, rankings, controls, oracle summaries |
| `scripts/run_m1.sh` | Ordered, logged M1 workflow; no fake/simulated model path |
| `scripts/audit_m1.py` | Provenance, cardinalities and independent raw-table arithmetic |
| `requirements.lock` | Tested 60-package Python environment |
| `results/m1/` | Frozen requests, raw evidence, analysis, plots, report |

## Architecture Map

Pinned official Hub weights/parquet files → fixed disjoint demos/calibration/smoke/evaluation → BF16 continuations → eight independent single-group W4 interventions → raw JSON/CSV → strict validation → frozen calibration choices → clustered analysis → figures/report → evidence audit.

KL compares matched BF16 histories; W4 task answers are separately generated. Q/DQ runs BF16 kernels: no measured integer-kernel speedup. Synthetic null calibration is a separate **statistical-method test**, not a source of model results.

## Development Workflow

1. Inspect the protocol, report and completion audit before changing anything.
2. Never alter completed raw records to obtain a preferred conclusion.
3. Measurement code/sample hashes are immutable within an output directory. Use a new directory if they change; keep prior evidence.
4. Keep analysis changes separate, record rationale before inspecting new outcomes when possible, preserve old analyses, rerun tests and statistical-method calibration.
5. Run raw validation, analysis, plots/report and the final audit. Update this guide/runbook when commands change.
6. No M2–M4, request router, clustering or batching without a new explicit scope.

## Testing

- `.venv/bin/python -m pytest -q tests`: 27 tests in the recorded final run, exit 0.
- Real smoke: 24 sources × two wordings × eight groups, strictly validated before formal inference.
- Formal: 96 sources × two wordings × eight groups, all actual outputs retained.
- One-command resume was actually executed; all stages exit 0. All 12 analysis CSVs matched the archived analysis exactly on re-execution.
- Manual checks: full-set sensitivity heatmap and accuracy figure were inspected; scientific statements were compared with raw/summary tables. Read [runbook](runbook.md) for exact commands and logs.

## Troubleshooting

- Missing gated model permission: accept the license/authenticate; never substitute another model.
- Missing dependencies: use `.venv/bin/python`; rerun `bash scripts/install.sh`.
- GPU contention/OOM: inspect `nvidia-smi`, stop only your own job, resume the identical command on an allocated device. Preserve interrupted logs.
- `inputs/code changed`: do not bypass the guard; use a new output directory.
- Failed artifact audit: inspect the named ledger log and restore consistency by rerunning the producing command, never by changing a success flag.
- Missing `.pt` references from a checkout: they are local-only caches. A new output-directory run regenerates them; completed published CSV/JSON can be analyzed without them.

## Documentation Freshness Checklist

Use this checklist for the **next change**; current evidence is recorded in `completion-audit.md`.

- [ ] README quickstart still works.
- [ ] Run commands match the current code.
- [ ] Test commands match the current code.
- [ ] Important files list is still accurate.
- [ ] Architecture map matches the current implementation.
- [ ] Troubleshooting section includes recent known failures.
