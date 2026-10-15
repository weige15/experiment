# M1: request-dependent layer quantization sensitivity

Real **meta-llama/Llama-3.1-8B (base)**, BF16 reference, one W4-quantized four-layer group at a time. GSM8K, ARC-Challenge, BoolQ. This is a sensitivity experiment, **not** a router or a packed-integer performance benchmark.

## Install

Linux, Python 3.12, `uv`, NVIDIA CUDA-compatible driver, and a BF16-capable GPU with approximately 24 GiB VRAM. Model access requires accepting Meta's license on Hugging Face and an authorized local login/cache. Credentials are never stored in this repository.

```bash
bash scripts/install.sh
.venv/bin/python -m pytest -q tests
```

`requirements.lock` pins the tested 60-package environment (Torch 2.8.0+cu128, Transformers 4.56.1). Model and dataset revisions are pinned in source and in [the protocol](doc/protocol.md).

## Incremental execution

Choose an available GPU; do not disturb other users' jobs. Run from the repository root:

```bash
export CUDA_VISIBLE_DEVICES=5  # replace with your available GPU
.venv/bin/python -m scripts.provenance
.venv/bin/python -m scripts.baseline
.venv/bin/python -m scripts.one_group
.venv/bin/python -m scripts.sensitivity --groups 0 --repeat-check
.venv/bin/python -m scripts.sensitivity --groups 0 1 2 3 4 5 6 7 --out results/m1/steps/eight_groups
.venv/bin/python -m scripts.prepare_data
.venv/bin/python -m m1_run --role smoke --wording a --out results/m1/runs/smoke_a
.venv/bin/python -m m1_run --role smoke --wording b --out results/m1/runs/smoke_b
.venv/bin/python -m scripts.validate_runs --role smoke --runs results/m1/runs/smoke_a results/m1/runs/smoke_b --out results/m1/smoke_validation.json
```

The runner checkpoints each request/configuration atomically. Repeat the **same command** to resume; it refuses to mix changed measurement code or samples into an existing run. Raw BF16 logits (`reference_logits/*.pt`) are retained locally but Git-ignored. JSON and CSV contain actual answers, token IDs, token-level KL and scores. No synthetic values are used as experiment results.

The base tokenizer has no official chat template. The explicit three-shot Q/A template is saved as `results/m1/chat_template.jinja` and held constant across precision configurations. W4 uses packed signed RTN codes with FP32 block scales, decoded to BF16 for computation; activations and KV cache stay BF16. See the protocol for scope and limitations.

## Status

M1 is **not complete**. Baseline → single-group → single-request KL → eight-group → full smoke gates have passed. Calibration is running; formal evaluation, statistical analysis, heatmap, final report and full one-command orchestrator are still pending. Current evidence and failures: [WORK_LOG.md](WORK_LOG.md).
