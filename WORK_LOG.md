# M1 execution journal

## 2026-10-15 — initial state and baseline
- Working directory `/nfs/home/s314511048/experiment` was empty. Initialized a local Git repository so changes do not affect the enclosing home-directory repository. No prior M1 work was available here.
- Found official base model snapshot `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`. GPUs 5 and 7 were idle RTX 3090 24 GiB devices. Do not stop other users' processes.
- Protocol committed before inference: `a6b8ae9`. The base model has no official chat template; fixed plain-text three-shot Q/A is explicitly recorded.
- Clean `uv` installation hit the foreground tool's 600-second timeout while downloading packages. Preserved `results/m1/steps/install.log`; background retry is recorded separately. Initial gates use the already-installed environment `/nfs/home/s314511048/.venvs/qaq-gptq425/bin/python` (Torch 2.8.0+cu128, Transformers 4.56.1), with versions saved in per-stage metadata. No claim of clean-install success yet.
- `.../qaq-gptq425/bin/python -m pytest -q tests/test_data.py`: exit **0**, 3 tests passed (`results/m1/steps/data-tests.*`).
- `CUDA_VISIBLE_DEVICES=5 .../qaq-gptq425/bin/python -m scripts.baseline`: exit **0**, 1 real GSM8K smoke request (`gsm8k:test:371`), 789 prompt tokens. BF16 answered **8**, gold **50**, incorrect; retained unchanged. Activation/KV dtype probe passed. Artifacts: `results/m1/steps/baseline/{baseline,metadata}.json`.
- Independent read-only review returned four validity improvements (`doc/protocol-review.md`): control request-wide magnitude in interaction analysis; check every generated logit for finiteness; enforce question-level deduplication; test causal alignment and numerical repeatability. Accepted and recorded in the pre-evaluation protocol addendum. No W4/evaluation result had been observed at amendment time.
- `CUDA_VISIBLE_DEVICES=5 .../qaq-gptq425/bin/python -m pytest -q tests`: exit **0**, 7 tests passed (`results/m1/steps/quant-tests.*`). Single-group real-model gate launched; do not proceed to sensitivity claims until it passes.

## 2026-10-15 — quantization and KL validity gates
- Clean installation completed (background retry exit **0**). Froze all 60 packages in `requirements.lock`; `bash scripts/install.sh` from that lock also exit **0** (`install-locked.*`).
- `.../qaq-gptq425/bin/python -m scripts.provenance`: exit **0**. All four safetensors shards and six config/tokenizer files matched the official pinned Hub digests. Evidence: `results/m1/model_provenance.json`.
- `CUDA_VISIBLE_DEVICES=5 .../qaq-gptq425/bin/python -m scripts.one_group`: exit **0**. Exactly 28 g0 matrices changed, every other parameter digest identical; all parameters and greedy BF16 output restored exactly. BF16 and g0 W4 answers were both incorrect (8 vs gold 50). Evidence: `results/m1/steps/one_group/`.
- `.venv/bin/python -m pytest -q tests`: exit **0**, 15 tests (`metrics-tests.*`), including analytic KL direction, causal shift, nibble packing, zero blocks, group isolation, exception restoration, duplicate detection, parser and finite-logit guard.
- `CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m scripts.sensitivity --groups 0 --repeat-check`: exit **0**, 1 real request × 1 W4 group; KL **0.00651636 nats/token**, 116 tokens. Independent BF16 and W4 replays and restored BF16 logits exact; repeat discrepancy 0. Cached-vs-full teacher-forcing KL **0.000370752**, argmax agreement **0.991379**: real numeric-path difference, not a precision effect. All primary BF16/W4 comparisons use the same full-sequence path. Evidence: `results/m1/steps/single_sensitivity/`.
- `.venv/bin/python -m scripts.prepare_data`: exit **0**. Frozen 153 source records: 9 demonstrations, 24 calibration, 24 smoke, 96 formal evaluation. Zero normalized-question duplicates encountered. Dataset revisions, parquet digests, sample IDs/text/gold saved in `results/m1/{requests.jsonl,data_manifest.json}` before formal evaluation.
- Eight-group single-request gate launched; wait for full eight-group validation before scaling inference.

## Current completion status
**Not achieved.** No full request×group evaluation, inferential result, heatmap, or report exists yet. Follow the incremental gates in `doc/protocol.md`.
