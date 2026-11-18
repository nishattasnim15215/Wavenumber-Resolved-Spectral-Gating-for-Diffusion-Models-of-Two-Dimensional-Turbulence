#!/usr/bin/env bash
# One-shot status snapshot of the training pool. Run repeatedly with:  watch -n 15 bash code/status.sh
cd "$(dirname "$0")/.."
echo "==== $(date) ===="
done=$(ls Results/_sweep_cells/*.pt Results/_lambda_cells/*.pt Results/_sigma_cells/*.pt \
        Results/_scalar_cells/*.pt Results/_cell_cells/*.pt 2>/dev/null | wc -l)
echo "cells done: $done / 42   |   claimed: $(ls Results/_queue/claims 2>/dev/null | wc -l)/43"
echo "  gate=$(ls Results/_sweep_cells/*.pt 2>/dev/null|wc -l)/12  sigma=$(ls Results/_sigma_cells/*.pt 2>/dev/null|wc -l)/6  lambda=$(ls Results/_lambda_cells/*.pt 2>/dev/null|wc -l)/6  scalar=$(ls Results/_scalar_cells/*.pt 2>/dev/null|wc -l)/9  cellular=$(ls Results/_cell_cells/*.pt 2>/dev/null|wc -l)/9"
echo
echo "-- workers alive --"
pgrep -af "run_pool.sh" | grep -v grep | sed 's/ bash/  /'
echo
echo "-- cell training now (variant + epoch) --"
for g in 1 2 3; do
  cur=$(tac Results/_queue/pool.log 2>/dev/null | grep -m1 "\[gpu$g\] CLAIM" | sed 's/.*CLAIM //')
  ep=$(grep "ep [0-9]*/160" Results/_queue/gpu$g.log 2>/dev/null | tail -1 | grep -oE "ep [0-9]+/160")
  echo "  gpu$g: ${ep:-starting}   $cur"
done
echo
echo "-- GPU util --"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
