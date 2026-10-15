# M1: request-dependent layer quantization sensitivity

Real **meta-llama/Llama-3.1-8B (base)**; BF16 reference versus exactly one W4 four-layer group at a time. GSM8K, ARC-Challenge, BoolQ. **M1 only: no router, clustering, batching, or packed-integer performance claims.**

## Results

[Full report](results/m1/report.md): 96 independent evaluation sources, two wordings, 1,536 W4 measurements and 1,728 actual answers. The evidence supports bounded request×group sensitivity, **not** deployable configuration selection. Label-oracle accuracy gain over accuracy-best fixed: **6.77 pp [95% CI 2.60–9.38]**; cross-wording KL-choice transfer is not clearly beneficial. Null cases and failures are retained.

## Quickstart

Linux with `bash`/`flock`, Python 3.12, `uv`, CUDA-compatible NVIDIA driver, and a BF16-capable GPU with roughly 24 GiB VRAM. Accept the model's Meta/Hugging Face license and use your authorized login/cache. Do not put credentials in this repository.

```bash
bash scripts/install.sh
.venv/bin/python -m pytest -q tests
M1_GPU=1 bash scripts/run_m1.sh   # choose a GPU allocated to you
```

The one-command runner installs the 60 pinned dependencies, verifies official weights, executes missing incremental gates, runs/resumes smoke → calibration → evaluation, validates raw records, and renders statistics/figures/report. It serializes work on one GPU and preserves prior derived results. Existing completed inference is **reused and audited**, not silently rerun.

For a separate full inference reproduction:

```bash
M1_OUT=results/m1_reproduction M1_GPU=1 bash scripts/run_m1.sh
```

All constituent GPU stages were actually executed. The one-command **resume/re-analysis** path was also executed successfully; a second full fresh-directory GPU campaign was not redundantly repeated.

## Common commands

```bash
# Recompute statistics from verified real measurements (CPU)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.analyze_m1
.venv/bin/python -m scripts.plot_m1
.venv/bin/python -m scripts.report_m1
.venv/bin/python -m scripts.audit_m1
```

Use a new analysis output directory or preserve the previous one when changing methodology. `run_m1.sh` archives existing derived results automatically. The full smoke/validation and recovery commands are in the [runbook](doc/runbook.md).

## Project structure / documentation

- `m1_{data,model,quant,metrics,run,analysis}.py`: fixed data/protocol, real inference, W4 intervention and analysis.
- `scripts/`: incremental gates, one-command execution, validation, plots/report.
- `tests/`: quantization/isolation, scoring, causal KL, audit corruption checks and statistics tests.
- `results/m1/runs/`: actual prompts, token IDs, answers, KL vectors and precision/restoration audits.
- [Protocol](doc/protocol.md) · [Onboarding](doc/onboarding.md) · [Runbook](doc/runbook.md)
- [Execution journal](WORK_LOG.md) · [Requirement audit](doc/completion-audit.md)
- [Statistical review](doc/statistics-review.md) · [Resolution](doc/statistics-review-resolution.md) · [Final review](doc/final-review.md)

## Troubleshooting

Use `.venv/bin/python`, not an unrelated environment. On OOM, inspect `nvidia-smi`, stop **only your own** process if necessary, and resume on an allocated device. Changed measurement code/samples require a new output directory. Raw reference logits are retained locally but Git-ignored; CSV/JSON contain the full public measurement evidence. See the runbook for detailed recovery and interpretation limits.
