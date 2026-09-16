#!/bin/bash
# Cluster-side waiter (runs on the login node, never on the local Mac): when a training run writes DONE,
# cancel the job that would otherwise evaluate it sequentially, then launch the parallel grid.
#
# usage: wait_par.sh <run_dir> <slurm_job_to_cancel> <ckpt_subdir> <result_tag> <conda_env> <grid5|full6>
RD=$1; JID=$2; SUB=$3; T=$4; ENV=$5; WHAT=$6
cd /home/mzh1800/WorldModel4TS
while [ ! -f "$RD/DONE" ]; do
  sleep 300
  if ! squeue -j "$JID" -h > /dev/null 2>&1 || [ -z "$(squeue -j "$JID" -h -o %T 2>/dev/null)" ]; then
    [ -f "$RD/DONE" ] || { echo "WAIT_PAR_ABORT job $JID ended without DONE $(date)"; exit 1; }
  fi
done
sleep 60
scancel "$JID"
bash pilot/par_grid.sh "$RD/$SUB" "$T" "$ENV" "$WHAT"
echo "WAIT_PAR_LAUNCHED $T $(date)"
