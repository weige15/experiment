# Onboarding

## What This Project Does

This experiment measures whether requests differ in their sensitivity to quantizing different four-layer groups of real Llama 3.1 8B weights. New runs produce `results/quantization_sensitivity/report.md` with complete raw measurements, actual answers, uncertainty and negative results. It does not implement a deployable router.

The old milestone name “M1” appears only in historical evidence, migration notes and regression tests. `results/m1/` and the protocol/review/audit documents are preserved, not migrated or relabeled. See the [runbook](runbook.md#operational-notes) for the verified archived source snapshot.

## Quickstart

From the repository root, with Python 3.12, uv, an authorized official-model login/cache and an allocated BF16 CUDA GPU:

```bash
bash scripts/install.sh
.venv/bin/python -m pytest -q tests
EXPERIMENT_GPU=1 bash scripts/run_experiment.sh
```

Successful execution ends with `Quantization sensitivity artifacts verified`, exit 0, and a ledger identified in `results/quantization_sensitivity/last_one_click_run.txt`. Read the report's limitations; the exit code alone is not proof of a research hypothesis.

## Important Files

| File | Role |
|---|---|
| `doc/protocol.md` | Fixed design and transparent pre-result amendments |
| `quantization_sensitivity/data.py` | Pinned benchmark revisions, sampling, formatting, automatic scoring |
| `quantization_sensitivity/model.py` | Official BF16 model, greedy decoding, dtype/finite-logit checks |
| `quantization_sensitivity/quantization.py` | Fixed RTN W4 packing/QDQ, exactly one group, restoration |
| `quantization_sensitivity/metrics.py` | Causally aligned full-vocabulary KL |
| `quantization_sensitivity/runner.py` | Immutable-input, resumable real measurement shards |
| `quantization_sensitivity/analysis.py` | Paired source bootstrap, rankings, controls, oracle summaries |
| `scripts/run_experiment.sh` | Ordered, logged experiment workflow; no fake/simulated model path |
| `scripts/{analyze,plot,report,audit}_results.py` | Analysis, figures, report and independent evidence audit |
| `requirements.lock` | Tested 60-package Python environment |
| `results/quantization_sensitivity/` | Default output for new runs |
| `results/m1/` | Immutable original campaign, including its original names and source hashes |

## Architecture Map

Pinned official Hub weights/parquet files → fixed disjoint demos/calibration/smoke/evaluation → BF16 continuations → eight independent single-group W4 interventions → raw JSON/CSV → strict validation → frozen calibration choices → clustered analysis → figures/report → evidence audit.

KL compares matched BF16 histories; W4 task answers are separately generated. Q/DQ runs BF16 kernels: no measured integer-kernel speedup. Synthetic null calibration is a separate **statistical-method test**, not a source of model results.

## Development Workflow

1. Inspect the protocol, report and completion audit before changing anything.
2. Never alter completed raw records to obtain a preferred conclusion.
3. Measurement code/sample hashes are immutable within an output directory. Even import/path renames change these hashes. Use a new directory if they change; keep prior evidence. Never rewrite old manifests to make a resume pass.
4. Keep analysis changes separate, record rationale before inspecting new outcomes when possible, preserve old analyses, rerun tests and statistical-method calibration.
5. Run raw validation, analysis, plots/report and the final audit. Update this guide/runbook when commands change.
6. No request router, clustering, batching or follow-on research without a new explicit scope.

## Testing

- `.venv/bin/python -m pytest -q tests`: CPU-only correctness, renamed CLI, source-metadata and resume-guard tests. The original campaign's recorded suite had 27 tests.
- Real smoke: 24 sources × two wordings × eight groups, strictly validated before formal inference.
- Formal: 96 sources × two wordings × eight groups, all actual outputs retained.
- The original campaign's one-command resume was executed; all stages exited 0. All 12 analysis CSVs matched exactly on re-execution. These are historical results, not a claim that the refactored GPU workflow was rerun.
- Manual checks: full-set sensitivity heatmap and accuracy figure were inspected; scientific statements were compared with raw/summary tables. Read [runbook](runbook.md) for exact commands and logs.

## Troubleshooting

- Missing gated model permission: accept the license/authenticate; never substitute another model.
- Missing dependencies: use `.venv/bin/python`; rerun `bash scripts/install.sh`.
- GPU contention/OOM: inspect `nvidia-smi`, stop only your own job, resume the identical command on an allocated device. Preserve interrupted logs.
- `inputs/code changed` / `measurement sources changed`: do not bypass the guard; use a new output directory. Historical validation/resume requires the original source snapshot, not compatibility wrappers.
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
