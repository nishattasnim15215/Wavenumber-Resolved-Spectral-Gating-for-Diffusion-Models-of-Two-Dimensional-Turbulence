#!/usr/bin/env bash
# Single-GPU claim-queue worker. Launch one per free GPU; all share Results/_queue/claims
# so no job runs twice. Add workers as cards free (never exceed the 3-GPU cap). Resumable:
# re-running re-claims only unfinished jobs; landed training cells re-load and re-eval fast.
# Usage: bash code/run_pool.sh <gpu> [jobsfile]   (run from the repo root)
set -u
cd "$(dirname "$0")/.."
GPU="$1"; JOBS="${2:-Results/_queue/jobs.txt}"
DIR="Results/_queue"; mkdir -p "$DIR/claims"
LOG="$DIR/gpu${GPU}.log"
echo "[pool gpu$GPU] start $(date 2>/dev/null)" >> "$DIR/pool.log"
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  case "$cmd" in \#*) continue;; esac
  key=$(printf '%s' "$cmd" | md5sum | cut -d' ' -f1)
  if mkdir "$DIR/claims/$key" 2>/dev/null; then
    echo "[gpu$GPU] CLAIM $cmd" >> "$DIR/pool.log"
    CUDA_VISIBLE_DEVICES=$GPU bash -c "$cmd" >> "$LOG" 2>&1
    echo "$cmd" >> "$DIR/claims/$key/cmd"
  fi
done < "$JOBS"
echo "[pool gpu$GPU] drained $(date 2>/dev/null)" >> "$DIR/pool.log"
