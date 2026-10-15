#!/usr/bin/env bash
# Append-only command, environment, timestamps, output and exit evidence.
set -uo pipefail
LEDGER=$1
NAME=$2
shift 2
mkdir -p "$LEDGER"
PREFIX="$LEDGER/$(date -u +%Y%m%dT%H%M%SZ)_${NAME}_$$"
printf '%q ' "$@" > "$PREFIX.command"
printf '\n' >> "$PREFIX.command"
printf 'started_utc=%s\nCUDA_VISIBLE_DEVICES=%s\n' "$(date -u +%FT%TZ)" "${CUDA_VISIBLE_DEVICES:-}" > "$PREFIX.environment"
"$@" > "$PREFIX.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$PREFIX.exit"
printf 'finished_utc=%s\n' "$(date -u +%FT%TZ)" >> "$PREFIX.environment"
printf '%s exit=%s\n' "$PREFIX" "$rc"
exit "$rc"
