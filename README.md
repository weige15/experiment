# Llama 3.1 8B individual-layer quantization sensitivity

## Research questions
1. Do different requests have stable, measurable differences in **individual-layer** W4 sensitivity?
2. Do these differences change which layers to retain in BF16, and actual answer accuracy?
3. Are they associated with dataset, prompt/output length or difficulty proxies?

**Current study: `layer_sensitivity/`, output `results/layer_sensitivity/`.** This is a new 32-layer study. The historical `results/m1/` campaign quantized four-layer groups and **does not satisfy this study's completion criteria**. Its raw evidence is preserved, not renamed as individual-layer results.

## Method
- Official **meta-llama/Llama-3.1-8B base**, revision `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`; fixed tokenizer, explicit Q/A template, train-only demonstrations and deterministic greedy decoding.
- BF16 reference; quantize **one layer at a time**, all layers **0–31**. Seven attention/MLP matrices per layer, symmetric RTN W4, input-axis group size 128, FP32 `absmax/7` scale, zero point 0, round-to-even codes [-7,7]. Activations/KV remain BF16.
- Packed-code round-trip then BF16 dequantization: **quantization sensitivity simulation**, no W4 kernel speed/memory claims.
- GSM8K, ARC-Challenge, BoolQ: 8 disjoint smoke requests + 32 formal requests each; 8 train-only selection requests each. Fixed BF16 continuation, full-vocabulary per-token mean `KL(BF16 || W4)`. Free-generation answers are scored separately and all failures retained.
- Direct allocation check: retain 8 BF16 layers, jointly quantize 24. Compare current-request KL **oracle-information** choice, train-selected fixed choice and three seeded random choices. This is not a deployable router or a guaranteed accuracy upper bound.

See the [locked protocol](doc/layer_sensitivity_protocol.md), [review](doc/layer_sensitivity_protocol_review.md), and [requirement/evidence audit](doc/layer_sensitivity_completion_audit.md). No clustering, request router or production batching.

## Quickstart / separate commands
Linux, Bash/flock, Python 3.12, `uv`, authorized access/cache for the gated model, CUDA driver supporting PyTorch CUDA 12.8, one BF16-capable GPU (~24 GiB VRAM). Recommend ≥48 GiB available host RAM for allocation CPU copies; allow model cache (~16 GB) and several GB of results. Credentials stay outside the repository.

```bash
bash scripts/install.sh
.venv/bin/python -m pytest -q tests

# Choose a GPU allocated to you. Smoke: 24 requests × all 32 layers.
LAYER_GPU=7 bash scripts/run_layer_sensitivity.sh smoke

# Full experiment: validates/reuses identical gates and smoke, then
# train-only selection → 96 × 32 formal scan → 480 joint allocations → report.
LAYER_GPU=7 bash scripts/run_layer_sensitivity.sh full

# CPU-only raw validation and report/figure reconstruction after completion.
bash scripts/run_layer_sensitivity.sh report

# Fresh independent reproduction, preserving previous evidence:
LAYER_OUT=results/layer_sensitivity_reproduction LAYER_GPU=7 \
  bash scripts/run_layer_sensitivity.sh full
```

**Executed and verified (2026-10-16 UTC):** 24 × 32 smoke, 24 × 32 train-only selection, **96 × 32 formal measurements**, and **480 directly measured joint configurations**. **63 tests pass**; separate smoke/full/CPU-report commands exited **0**. Report reconstruction reproduced **20 CSVs, report text and four PNGs exactly**. See the [execution journal](doc/layer_sensitivity_progress.md), [requirement audit](doc/layer_sensitivity_completion_audit.md) and [review resolutions](doc/layer_sensitivity_review_resolution.md).

## Main findings
- Different-request layer-rank Spearman: **0.5737 [0.5421, 0.6032]**; Top-8 overlap: **0.5222 [0.5029, 0.5406]**. Profiles share global preferences but also vary measurably by request.
- Within-request content-half Spearman: **0.8383 [0.8134, 0.8620]**, limited to **32 eligible GSM8K requests**, not short ARC/BoolQ answers.
- Eight-layer-budget accuracy: fixed **70.83%**, request-KL oracle **66.67%**, random-set mean **68.40%**. Oracle minus fixed: **−4.17 pp [−11.46, +3.13]**—**no demonstrated accuracy benefit**. Neither oracle is a deployed router.

Read the [actual report](results/layer_sensitivity/report.md) for all data, uncertainty, negative cases and limitations. R² refit ranges are descriptive, not validated heldout-performance confidence bounds.

## Outputs and code
- `layer_sensitivity/{quantization,runner,gates}.py`: seven-matrix single-layer intervention and real-model scan.
- `layer_sensitivity/{selection,precision_allocation_check}.py`: isolated fixed selection and directly measured joint models.
- `layer_sensitivity/{validate,statistics,analysis}.py`: coverage/raw auditing, request-level bootstrap, all-request figures and report.
- `quantization_sensitivity/{data,model,metrics,quantization}.py`: reused pinned datasets/tokenizer/model, parsers, causal KL and weight-level RTN primitives. Historical grouped runner is not used by the new experiment.
- `results/layer_sensitivity/runs/evaluation/{measurements.csv,measurements.parquet,sensitivity_vectors.csv}`: every request × 32 values, plus raw JSON/prompts/answers.
- `results/layer_sensitivity/precision_allocation_check/`: every directly evaluated joint configuration.
- `results/layer_sensitivity/{analysis/,figures/,report.md}`: verified real-data statistics/CIs, all-request figures, failures and limitations.
- `results/layer_sensitivity/steps/`: append-only command, UTC start/end, GPU visibility, log and exit evidence.

## Documentation / troubleshooting
[Onboarding](doc/onboarding.md) · [Runbook](doc/runbook.md) · [Protocol](doc/layer_sensitivity_protocol.md) · [Progress](doc/layer_sensitivity_progress.md) · [Audit](doc/layer_sensitivity_completion_audit.md).

Use `.venv/bin/python`. On OOM inspect `nvidia-smi`, never terminate another user's process. Changed measurement source/sample hashes require a **new output directory**; do not bypass resume guards or overwrite raw results. BF16 `.pt` references are local-only/Git-ignored and regenerated by a fresh inference run. CPU report reconstruction uses published JSON/CSV, not those tensors.

Historical grouped-campaign operation is documented separately in [archived group runbook](doc/archived_group_study_runbook.md); use its original source snapshot for old manifest validation.
