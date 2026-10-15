# Onboarding

## What This Project Does

Measure request-dependent **individual-layer** W4 sensitivity of the real Llama 3.1 8B base model. Produce complete 32-layer vectors, actual answers, uncertainty, all-request plots and a bounded report in `results/layer_sensitivity/`. The earlier four-layer-group campaign is historical only and cannot certify the current goal.

## Quickstart

From the repository root, with Python 3.12, uv, authorized model access and an allocated BF16-capable ~24 GiB GPU:

```bash
bash scripts/install.sh
.venv/bin/python -m pytest -q tests
LAYER_GPU=7 bash scripts/run_layer_sensitivity.sh smoke
LAYER_GPU=7 bash scripts/run_layer_sensitivity.sh full
bash scripts/run_layer_sensitivity.sh report
```

Use `LAYER_OUT=results/layer_sensitivity_reproduction` for a fresh campaign. Host RAM ≥48 GiB recommended for the allocation weight bank. Each command records actual exits under `steps/`; success flags alone do not prove the scientific objective. Current execution status: [progress](layer_sensitivity_progress.md).

## Important Files

| Path | Role |
|---|---|
| `doc/layer_sensitivity_protocol.md` | Locked design, parser/control/escalation amendments |
| `layer_sensitivity/quantization.py` | Exactly seven matrices per selected individual layer |
| `layer_sensitivity/gates.py` | Real isolation/replay/restoration and one-request 32-layer gates |
| `layer_sensitivity/runner.py` | Immutable-input resumable 32-layer scans |
| `layer_sensitivity/selection.py` | Train-only fixed top-8 and three frozen random sets |
| `layer_sensitivity/precision_allocation_check.py` | Direct joint 24-layer W4 inference, 8 BF16 layers |
| `layer_sensitivity/validate.py` | Raw coverage, inputs, scores, quantization and gate checks |
| `layer_sensitivity/statistics.py` | Request-level paired/stratified bootstrap and controls |
| `layer_sensitivity/analysis.py` | Tables, figures and report from validated raw data |
| `quantization_sensitivity/{data,model,metrics,quantization}.py` | Reused shared low-level tools; not the grouped runner |
| `scripts/run_layer_sensitivity.sh` | Separate smoke/full/report workflows |
| `scripts/record_command.sh` | Append-only command/exit ledger |
| `requirements.lock` | Pinned Python environment |
| `tests/test_layer_*.py` | Individual-layer, joint-state and statistics regression tests |

## Architecture Map

Pinned official model + pinned dataset parquet → disjoint train demonstrations/selection and heldout smoke/evaluation → fresh BF16 continuation → 32 single-layer Q/DQ interventions → JSON/CSV/Parquet and complete vectors → strict validator → train-frozen top-8 choice → formal scan → directly measured joint allocations → request-level statistics/figures/report → manual requirement-to-artifact completion audit.

All inference is single-request BF16 compute. Integer code Q/DQ measures quality only. Same-request KL selection is explicitly oracle information, not a router.

## Development Workflow

1. Read the protocol, progress and completion audit before choosing an unfinished gate.
2. Inspect active processes/ledger before launching another writer. The shell runner locks each output root.
3. Do not edit measurement sources while a scan is in progress. Source hashes must match every attempt; changed code requires a new root and revalidation.
4. Analysis/validator/tests/docs may be improved separately, with rationale and previous analysis preserved. Never edit completed raw measurements or restamp manifests.
5. Keep all samples and failures. Treat unit/synthetic statistics tests as code evidence, never model measurements.
6. Before claiming completion, check every user requirement against actual files/command exits, including full 32-layer coverage and direct joint accuracy. Old grouped audit does not apply.

## Testing

```bash
.venv/bin/python -m pytest -q tests
.venv/bin/python -m layer_sensitivity.validate --gates
.venv/bin/python -m layer_sensitivity.validate --role smoke
.venv/bin/python -m layer_sensitivity.validate --role evaluation
.venv/bin/python -m layer_sensitivity.validate --allocation
```

Validators need the corresponding completed raw stage. Add `--root` for custom output. Synthetic test models exercise restoration and budget semantics only. Real gates verify official 8B weights, exact seven-matrix isolation, full BF16 restoration, precision and exact numerical replay. Formal coverage must be 96 requests × 32 layers, not 8 groups or partial layers. Inspect the rendered figures and numerical claims in addition to software checks.

## Troubleshooting

- Model permission denied: authenticate with authorized Hugging Face access; never substitute a different model.
- CUDA OOM/contention: inspect `nvidia-smi`; resume unchanged inputs on an allocated GPU, never kill others' work.
- Source/input mismatch: preserve evidence and use a new output root; do not rewrite recorded hashes.
- Missing `.pt` in a checkout: fresh inference regenerates local references; report rebuild needs raw JSON/CSV, not tensors.
- Failed validator: inspect its log, fix the root cause and preserve failed attempt evidence. Do not turn a status field green manually.
- Missing report: check whether smoke/formal/allocation really finished; implementation is not completion.

Historical onboarding/runbook text is preserved as `archived_group_study_{onboarding,runbook}.md`. It describes a different grouped experiment.

## Documentation Freshness Checklist

- [ ] README quickstart still works.
- [ ] Run commands match the current code.
- [ ] Test commands match the current code.
- [ ] Important files list is still accurate.
- [ ] Architecture map matches the current implementation.
- [ ] Troubleshooting section includes recent known failures.
