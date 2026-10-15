#!/usr/bin/env bash
# One-command M1 only. Existing real inference checkpoints are verified/resumed.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${M1_GPU:?Set M1_GPU to a CUDA device allocated to you (e.g. M1_GPU=1)}"
export CUDA_VISIBLE_DEVICES="$M1_GPU"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
OUT=${M1_OUT:-results/m1}
PY=.venv/bin/python
mkdir -p "$OUT/steps"
exec 9>.m1-run.lock
flock -n 9 || { echo 'Another one-command M1 run holds this repository lock' >&2; exit 1; }
LEDGER="$OUT/steps/one_click_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$LEDGER"
trap 'printf "%s\n" "$?" > "$LEDGER/overall.exit"' EXIT
printf 'CUDA_VISIBLE_DEVICES=%q\nM1_OUT=%q\n' "$CUDA_VISIBLE_DEVICES" "$OUT" > "$LEDGER/environment.txt"

run() {
  local name=$1 rc=0
  shift
  printf '%q ' "$@" > "$LEDGER/$name.command"
  printf '\n' >> "$LEDGER/$name.command"
  echo "RUN $name — $LEDGER/$name.log"
  "$@" > "$LEDGER/$name.log" 2>&1 || rc=$?
  printf '%s\n' "$rc" > "$LEDGER/$name.exit"
  if [[ $rc != 0 ]]; then echo "FAILED $name (exit $rc); original log retained" >&2; fi
  return "$rc"
}

gate() {
  local name=$1 artifact=$2
  shift 2
  if [[ -f "$artifact" && -f "$(dirname "$artifact")/metadata.json" ]]; then
    echo "REUSE completed gate $name: $artifact" | tee "$LEDGER/$name.reused"
  else
    run "$name" "$@"
  fi
}

run install bash scripts/install.sh
run unit_tests "$PY" -m pytest -q tests
run official_weights "$PY" -m scripts.provenance --out "$LEDGER/model_provenance.json"
if [[ -f "$OUT/model_provenance.json" ]]; then
  run provenance_compare "$PY" -c 'import json,sys; a,b=[json.load(open(p)) for p in sys.argv[1:]]; assert (a["model_id"],a["revision"])==(b["model_id"],b["revision"]); assert {x["file"]:x for x in a["files"]}=={x["file"]:x for x in b["files"]}' "$OUT/model_provenance.json" "$LEDGER/model_provenance.json"
else
  cp "$LEDGER/model_provenance.json" "$OUT/model_provenance.json"
fi
gate baseline "$OUT/steps/baseline/baseline.json" "$PY" -m scripts.baseline --out "$OUT/steps/baseline"
gate one_group "$OUT/steps/one_group/result.json" "$PY" -m scripts.one_group --baseline "$OUT/steps/baseline/baseline.json" --out "$OUT/steps/one_group"
gate single_sensitivity "$OUT/steps/single_sensitivity/measurements.csv" "$PY" -m scripts.sensitivity --groups 0 --repeat-check --baseline "$OUT/steps/baseline/baseline.json" --out "$OUT/steps/single_sensitivity"
gate eight_groups "$OUT/steps/eight_groups/measurements.csv" "$PY" -m scripts.sensitivity --groups 0 1 2 3 4 5 6 7 --baseline "$OUT/steps/baseline/baseline.json" --out "$OUT/steps/eight_groups"
run frozen_data "$PY" -m scripts.prepare_data --out "$OUT"

for role in smoke calibration evaluation; do
  for wording in a b; do
    directory="$OUT/runs/${role}_${wording}"
    if [[ -f "$directory/summary.json" ]]; then
      echo "REUSE raw $role $wording; strict validation follows" | tee "$LEDGER/${role}_${wording}.reused"
    else
      run "${role}_${wording}" "$PY" -m m1_run --role "$role" --wording "$wording" --requests "$OUT/requests.jsonl" --out "$directory"
    fi
  done
  run "validate_$role" "$PY" -m scripts.validate_runs --role "$role" --requests "$OUT/requests.jsonl" --runs "$OUT/runs/${role}_a" "$OUT/runs/${role}_b" --out "$OUT/${role}_validation.json"
  if [[ "$role" == calibration ]]; then run freeze_choices "$PY" -m scripts.select_fixed --root "$OUT"; fi
done

# Preserve previous derived results. Raw model measurements are never regenerated as simulations.
if [[ -d "$OUT/analysis" ]]; then cp -a "$OUT/analysis" "$LEDGER/analysis_before"; fi
if [[ -f "$OUT/report.md" ]]; then cp "$OUT/report.md" "$LEDGER/report_before.md"; fi
if [[ -f "$OUT/steps/statistical_null_calibration.json" ]]; then
  cp "$OUT/steps/statistical_null_calibration.json" "$LEDGER/statistical_null_calibration_before.json"
fi
run synthetic_method_validation "$PY" -m scripts.calibrate_statistics --out "$OUT/steps/statistical_null_calibration.json"
run analysis "$PY" -m scripts.analyze_m1 --root "$OUT" --bootstrap 2000
run leverage_influence "$PY" -m scripts.check_leverage --root "$OUT"
run figures "$PY" -m scripts.plot_m1 --analysis "$OUT/analysis" --out "$OUT"
run report "$PY" -m scripts.report_m1 --root "$OUT"
compare=()
if [[ -d "$LEDGER/analysis_before" ]]; then compare=(--compare-analysis "$LEDGER/analysis_before"); fi
run completion_evidence "$PY" -m scripts.audit_m1 --root "$OUT" "${compare[@]}"
printf '%s\n' "$LEDGER" > "$OUT/last_one_click_run.txt"
echo "M1 artifacts verified: $OUT/report.md — inspect bounded conclusion/limitations before claiming research success."
