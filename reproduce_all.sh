#!/usr/bin/env bash
# Enumerate every job needed to regenerate all paper tables (ablations, transfers, UQ,
# conditional). By default this PRINTS the commands; pass `run` to execute them sequentially:
#   bash code/reproduce_all.sh            # dry run (print the full grid)
#   bash code/reproduce_all.sh run        # execute (long; use one GPU, or adapt to --claim_queue)
# The main headline sweep is separate (see README): python code/WRSG.py --claim_queue / --aggregate_only.
set -u
cd "$(dirname "$0")/.."
DO="${1:-print}"
SWEEP_SEEDS=(29 47 89)
j() { if [ "$DO" = run ]; then echo "+ $*"; "$@"; else echo "$*"; fi; }
PY="python code/WRSG.py"

echo "### Gate bin/rank ablation  -> gate_sensitivity.csv (baseline bins=16,rank=2 from the main sweep)"
for br in "8 2" "16 1" "16 4" "32 2"; do set -- $br; for s in "${SWEEP_SEEDS[@]}"; do
  j $PY reviews sweepcell --bins $1 --rank $2 --seed $s; done; done

echo "### Loss-weight sweep  -> lambda_sensitivity.csv (baseline scale=1 from the main sweep)"
for sc in 0.5 2.0; do for s in "${SWEEP_SEEDS[@]}"; do j $PY reviews lambdacell --scale $sc --seed $s; done; done

echo "### Integral-length-weight sweep  -> lambdaL_sensitivity.csv"
for L in 2 4; do for s in "${SWEEP_SEEDS[@]}"; do j $PY reviews lambdaLcell --lamL $L --seed $s; done; done

echo "### Low-k spectral-loss intervention  -> lowk_sensitivity.csv"
for lk in 0.1 0.3; do for s in "${SWEEP_SEEDS[@]}"; do j $PY reviews lowkcell --lamlowk $lk --seed $s; done; done

echo "### sigma_max sweep  -> sigma_sensitivity.csv (baseline 20 from the main sweep)"
for sm in 10 40; do for s in "${SWEEP_SEEDS[@]}"; do j $PY reviews sigmacell --sigma_max $sm --seed $s; done; done

echo "### Loss-configuration probes  -> loss_probe.csv"
for s in "${SWEEP_SEEDS[@]}"; do
  j $PY reviews losscell --tag flux2     --flux_scale 2   --seed $s
  j $PY reviews losscell --tag struct0   --struct_scale 0 --seed $s
  j $PY reviews losscell --tag logspec05 --logspec 0.05   --seed $s
  j $PY reviews losscell --tag logspec10 --logspec 0.10   --seed $s
done

echo "### Sampling-step sweep + ablation aggregation"
j $PY reviews steps
j $PY reviews aggregate
j $PY reviews lossagg

echo "### Passive-scalar transfer  -> scalar_poc_metrics.csv, scalar_retune.csv"
j $PY secondflow gendata
for s in "${SWEEP_SEEDS[@]}"; do
  j $PY secondflow traincell --variant vanilla   --seed $s
  j $PY secondflow traincell --variant wrsd_gate  --seed $s
  j $PY secondflow traincell --variant wrsd        --seed $s --profile full --scale 1.0
  j $PY secondflow traincell --variant wrsd        --seed $s --profile full --scale 0.25
  j $PY secondflow traincell --variant wrsd        --seed $s --profile full --scale 0.5
  j $PY secondflow traincell --variant wrsd        --seed $s --profile spectral_only
  j $PY secondflow traincell --variant wrsd        --seed $s --profile spectral_lowk
done
for s in 101 149; do j $PY secondflow traincell --variant wrsd --seed $s --profile spectral_only; done
j $PY secondflow aggregate
j $PY secondflow retuneagg

echo "### Cellular-forcing transfer  -> cellular_poc_metrics.csv"
j $PY thirdflow gendata
for v in vanilla wrsd_gate wrsd; do for s in "${SWEEP_SEEDS[@]}"; do j $PY thirdflow traincell --variant $v --seed $s; done; done
j $PY thirdflow aggregate

echo "### 64^2 resolution transfer  -> res64_poc_metrics.csv"
j $PY resolution gendata
for v in vanilla wrsd_gate wrsd; do for s in "${SWEEP_SEEDS[@]}"; do j $PY resolution traincell --variant $v --seed $s; done; done
j $PY resolution aggregate

echo "### Reviewer diagnostics + figures"
for c in ood speed steps mechanism rootcause gradflow uqdepth figs; do j $PY reviews $c; done
j $PY downstream

echo "### Standalone scripts (conditional emulator + UQ)"
j python code/conditional.py --gpu 0
j python code/conditional.py --robust --gpu 0
j python code/uq_improve.py --gpu 0

echo "### done (${DO})"
