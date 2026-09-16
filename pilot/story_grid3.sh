#!/bin/bash
# like story_grid2 but forwards extra evaluator args. usage: story_grid3.sh <gpu> <model> <tag> [extra args...]
G=$1; M=$2; T=$3; shift 3
HS=${HS:-"96 336 720"}
cd /home/mzh1800/WorldModel4TS
for H in $HS; do for ds in ETTh1 ETTh2 ETTm1 ETTm2 weather; do pilot/lsf_run.sh $G $ds $H $M $T --mode auto "$@"; done; echo "STORY_H_DONE $T H$H $(date)"; done
echo "STORY_DONE $T $(date)"
