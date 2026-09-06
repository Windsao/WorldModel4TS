#!/bin/bash
# Route B six-dataset zero-shot evaluation for one checkpoint step, all four arms.
# usage: run_route_b_eval.sh <step> [stage F|E] [gpus "0 1 2 3"]
# Datasets are spread over the given GPUs (one process per GPU, round robin).
STEP=${1:-20000}; STAGE=${2:-F}; GPUS=(${3:-0 1 2 3})
source /nyx-storage1/hanliu/miniconda3/etc/profile.d/conda.sh; conda activate wm4ts
export HF_HOME=/nyx-storage1/hanliu/hf HUGGINGFACE_HUB_CACHE=/nyx-storage1/hanliu/hf/hub TRANSFORMERS_CACHE=/nyx-storage1/hanliu/hf/transformers
cd /home/mzh1800/WorldModel4TS
OUT=/nyx-storage1/hanliu/wm4ts/route_b/eval
mkdir -p $OUT logs
DS=(ETTh1 ETTh2 ETTm2 electricity traffic solar)
NG=${#GPUS[@]}
for g in $(seq 0 $((NG-1))); do
  (
    for i in $(seq $g $NG $((${#DS[@]}-1))); do
      ds=${DS[$i]}
      CUDA_VISIBLE_DEVICES=${GPUS[$g]} python pilot/eval_route_b.py --dataset $ds --stage $STAGE --step $STEP \
          --arms vmae_full,vmae_enc,imae_enc,random --out $OUT 2>&1 | grep -v -i warn > logs/route_b_eval_${STAGE}_${STEP}_${ds}.log
    done
  ) &
done
wait
echo "EVAL_DONE step=$STEP stage=$STAGE"
