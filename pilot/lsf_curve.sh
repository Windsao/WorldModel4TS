#!/bin/bash
# budget-curve diagnostic: usage lsf_curve.sh <gpu> <run_dir> <tagprefix> <step> [<step> ...]; three quick H=96 cells per checkpoint
G=$1; R=$2; P=$3; shift 3
cd /home/mzh1800/WorldModel4TS
for S in "$@"; do for ds in ETTh1 ETTh2 ETTm1; do pilot/lsf_run.sh $G $ds 96 $R/step_$S ${P}_s$S --mode auto; done; echo "CURVE_DONE $P step_$S $(date)"; done
