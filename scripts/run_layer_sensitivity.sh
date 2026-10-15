#!/usr/bin/env bash
# Separate smoke, full real experiment, and CPU-only report reconstruction commands.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-full}
OUT=${LAYER_OUT:-results/layer_sensitivity}
PY=.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
case "$MODE" in smoke|full|report) ;; *) echo 'usage: run_layer_sensitivity.sh {smoke|full|report}' >&2; exit 2;; esac
if [[ "$MODE" != report ]]; then
  : "${LAYER_GPU:?Set LAYER_GPU to a BF16-capable CUDA device allocated to you}"
  export CUDA_VISIBLE_DEVICES="$LAYER_GPU"
fi
mkdir -p "$OUT/steps"
exec 9>"$OUT/.layer-sensitivity-run.lock"
flock -n 9 || { echo "Another layer study holds the output lock: $OUT" >&2; exit 1; }
LEDGER="$OUT/steps/${MODE}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$LEDGER"
trap 'printf "%s\n" "$?" > "$LEDGER/overall.exit"' EXIT
printf 'mode=%q\noutput=%q\nCUDA_VISIBLE_DEVICES=%q\n' "$MODE" "$OUT" "${CUDA_VISIBLE_DEVICES:-}" > "$LEDGER/environment.txt"
run() { bash scripts/record_command.sh "$LEDGER" "$@"; }

if [[ "$MODE" != report ]]; then
  run unit_tests "$PY" -m pytest -q tests
  run official_weights "$PY" -m scripts.provenance --out "$OUT/model_provenance.json"
  run frozen_data "$PY" -m scripts.prepare_data --out "$OUT"
  if [[ ! -f "$OUT/gates/baseline/baseline.json" ]]; then
    run baseline "$PY" -m scripts.baseline --out "$OUT/gates/baseline"
  fi
  for stage in single_layer single_request; do
    if [[ ! -f "$OUT/gates/$stage/result.json" ]]; then
      run "$stage" "$PY" -m layer_sensitivity.gates --stage "$stage" --baseline "$OUT/gates/baseline/baseline.json" --out "$OUT/gates/$stage"
    fi
  done
  run validate_gates "$PY" -m layer_sensitivity.validate --root "$OUT" --gates
  for role in smoke calibration evaluation; do
    if [[ "$MODE" == smoke && "$role" != smoke ]]; then break; fi
    if [[ ! -f "$OUT/runs/$role/summary.json" ]]; then
      run "$role" "$PY" -m layer_sensitivity.runner --role "$role" --requests "$OUT/requests.jsonl" --out "$OUT/runs/$role"
    fi
    run "validate_$role" "$PY" -m layer_sensitivity.validate --root "$OUT" --role "$role"
    if [[ "$role" == calibration ]]; then
      run freeze_allocation "$PY" -m layer_sensitivity.selection --root "$OUT"
    fi
  done
  if [[ "$MODE" == smoke ]]; then echo "Smoke evidence verified: $OUT/smoke_validation.json"; exit 0; fi
  if [[ ! -f "$OUT/precision_allocation_check/summary.json" ]]; then
    run precision_allocation "$PY" -m layer_sensitivity.precision_allocation_check --root "$OUT"
  fi
fi

run validate_evaluation "$PY" -m layer_sensitivity.validate --root "$OUT" --role evaluation
run validate_allocation "$PY" -m layer_sensitivity.validate --root "$OUT" --allocation
# Rebuilding a report does not overwrite raw inference. Preserve older derived output.
if [[ -d "$OUT/analysis" ]]; then cp -a "$OUT/analysis" "$LEDGER/analysis_before"; fi
if [[ -f "$OUT/report.md" ]]; then cp "$OUT/report.md" "$LEDGER/report_before.md"; fi
if [[ -d "$OUT/figures" ]]; then cp -a "$OUT/figures" "$LEDGER/figures_before"; fi
run report "$PY" -m layer_sensitivity.analysis --root "$OUT" --bootstrap 2000
printf '%s\n' "$LEDGER" > "$OUT/last_successful_command.txt"
echo "Artifacts produced: $OUT/report.md. This is not by itself a completion audit."
