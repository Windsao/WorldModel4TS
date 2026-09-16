#!/bin/bash
# missing paper-table cells for one checkpoint on ONE GPU: H=192 for the 5 grid datasets and electricity (all horizons, halves).
# usage: paper_cells.sh <model_path> <tag> <slot 0-3>
M=$1; T=$2; K=$3; shift 3
cd /home/mzh1800/WorldModel4TS
case $K in
 0) pilot/lsf_run.sh 0 ETTh1 192 $M $T --mode auto "$@"; pilot/lsf_run.sh 0 ETTh2 192 $M $T --mode auto "$@"
    pilot/lsf_run.sh 0 electricity 96 $M $T --mode auto "$@" --origin-range 0:2600; pilot/lsf_run.sh 0 electricity 96 $M $T --mode auto "$@" --origin-range 2600:5165 ;;
 1) pilot/lsf_run.sh 0 ETTm1 192 $M $T --mode auto "$@"; pilot/lsf_run.sh 0 ETTm2 192 $M $T --mode auto "$@"
    pilot/lsf_run.sh 0 electricity 192 $M $T --mode auto "$@" --origin-range 0:2600; pilot/lsf_run.sh 0 electricity 192 $M $T --mode auto "$@" --origin-range 2600:5069 ;;
 2) pilot/lsf_run.sh 0 weather 192 $M $T --mode auto "$@"; pilot/lsf_run.sh 0 electricity 336 $M $T --mode auto "$@" ;;
 3) pilot/lsf_run.sh 0 electricity 720 $M $T --mode auto "$@" --origin-range 0:2300; pilot/lsf_run.sh 0 electricity 720 $M $T --mode auto "$@" --origin-range 2300:4541 ;;
esac
echo "PAPER_CELLS_DONE $T slot$K $(date)"
