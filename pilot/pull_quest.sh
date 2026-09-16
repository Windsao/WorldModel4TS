#!/bin/bash
# Cluster-side puller (NU CS login node): copy LSF result JSONs for one tag from Quest scratch into
# /nyx-storage1/hanliu/wm4ts/lsf/ every 10 min until <n_expected> files are present (or 3 days pass).
# usage: pull_quest.sh <result_tag> <n_expected>
T=$1; N=$2
Q=mzh1800@quest.northwestern.edu; QW=/scratch/mzh1800/wm4ts_eval/lsf_results; OUT=/nyx-storage1/hanliu/wm4ts/lsf
S="ssh -o BatchMode=yes -o ConnectTimeout=30"
deadline=$(( $(date +%s) + 3*86400 ))
while [ $(date +%s) -lt $deadline ]; do
  files=$($S $Q "cd $QW && ls *_${T}_*.json 2>/dev/null" | tr '\n' ' ')
  [ -n "$files" ] && $S $Q "cd $QW && tar -cf - $files" | tar -C $OUT -xf -
  have=$(ls $OUT | grep -c "_${T}_")
  echo "$(date) $T: $have / $N"
  [ "$have" -ge "$N" ] && { echo "PULL_QUEST_COMPLETE $T $(date)"; exit 0; }
  sleep 600
done
echo "PULL_QUEST_TIMEOUT $T $(date)"
