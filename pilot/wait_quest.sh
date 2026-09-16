#!/bin/bash
# Cluster-side waiter (NU CS login node, never the local Mac): when a training run writes DONE, cancel the job that
# would evaluate it sequentially, push the checkpoint to Quest, submit one Quest job per LSF cell, then pull the
# result JSONs back into /nyx-storage1/hanliu/wm4ts/lsf/ as they appear (all cells, or give up after 3 days).
#
# usage: wait_quest.sh <run_dir> <slurm_job_to_cancel|none> <ckpt_subdir> <result_tag> <grid5|full6>
RD=$1; JID=$2; SUB=$3; T=$4; WHAT=${5:-grid5}
Q=mzh1800@quest.northwestern.edu; QW=/scratch/mzh1800/wm4ts_eval
S="ssh -o BatchMode=yes -o ConnectTimeout=30"
LOCAL_OUT=/nyx-storage1/hanliu/wm4ts/lsf
RUN=$(basename "$RD")
cd /home/mzh1800/WorldModel4TS

while [ ! -f "$RD/DONE" ]; do sleep 300; done
sleep 60
[ "$JID" != none ] && scancel "$JID"
echo "$(date) DONE seen for $RUN; pushing $SUB to Quest"

for i in 1 2 3; do
  tar -C "$(dirname "$RD")" -cf - "$RUN/$SUB" $( [ -f "$RD/run_config.json" ] && echo "$RUN/run_config.json" ) \
    | $S $Q "mkdir -p $QW/ckpt && tar -C $QW/ckpt -xmf -" && break
  echo "push attempt $i failed"; sleep 60
done
M=$QW/ckpt/$RUN/$SUB

if [ "$WHAT" = full6 ]; then HS="96 192 336 720"; DSS="ETTh1 ETTh2 ETTm1 ETTm2 weather electricity"; else HS="96 336 720"; DSS="ETTh1 ETTh2 ETTm1 ETTm2 weather"; fi
N=0
for ds in $DSS; do for H in $HS; do
  $S $Q "cd $QW/code && sbatch --job-name=wm4ts_${ds}_${H} quest_cell.sh $M $T $ds $H" && N=$((N+1))
done; done
echo "$(date) submitted $N Quest cells for $T"

deadline=$(( $(date +%s) + 3*86400 ))
while [ $(date +%s) -lt $deadline ]; do
  sleep 600
  tar_list=$($S $Q "cd $QW/lsf_results && ls *_${T}_*.json 2>/dev/null")
  [ -n "$tar_list" ] && $S $Q "cd $QW/lsf_results && tar -cf - $(echo $tar_list | tr '\n' ' ')" | tar -C $LOCAL_OUT -xf -
  have=$(ls $LOCAL_OUT | grep -c "_${T}_")
  echo "$(date) pulled: $have / $N cells"
  [ "$have" -ge "$N" ] && { echo "WAIT_QUEST_COMPLETE $T $(date)"; exit 0; }
done
echo "WAIT_QUEST_TIMEOUT $T $(date)"
