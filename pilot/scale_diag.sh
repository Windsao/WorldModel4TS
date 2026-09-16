#!/bin/bash
# ETTh1-96 per-scale diagnostic for one checkpoint: k = 1, 2, 4 sequentially on one GPU.
# usage: scale_diag.sh <ckpt> <tagprefix> [env]
M=$1; P=$2; ENV=${3:-wm4ts}
cd /home/mzh1800/WorldModel4TS
for s in 1 2 4; do WM4TS_ENV=$ENV pilot/lsf_run_env.sh 0 ETTh1 96 $M ${P}_sc$s --mode multiperiod --scales $s --out /nyx-storage1/hanliu/wm4ts/lsf_probe; done
echo "SCALE_DIAG_DONE $P $(date)"
