#!/bin/bash
# init-comparison grid under the paper protocol (no electricity): usage story_grid.sh <gpu> <model_path> <tag>
G=$1; M=$2; T=$3
cd /home/mzh1800/WorldModel4TS
HS=${HS:-"96 336 720"}; for H in $HS; do for ds in ETTh1 ETTh2 ETTm1 ETTm2 weather; do pilot/lsf_run.sh $G $ds $H $M $T --mode auto; done; echo "STORY_H_DONE $T H$H $(date)"; done
echo "STORY_DONE $T $(date)"
